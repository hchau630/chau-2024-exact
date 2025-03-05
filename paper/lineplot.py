import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import torch
import numpy as np
import pandas as pd

from niarb.special.resolvent import laplace_r
from niarb import nn, neurons, viz, exceptions

CM = 1 / 2.54  # cm to inch
CM_TO_PT = 28.3465  # cm to pt
AXSIZE = (2.1 * CM, 2.1 * CM)
CAXSIZE = (2.5 * CM, 2.1 * CM)
FIGSIZE = (3.6 * CM, 3.4 * CM)
CFIGSIZE = (4.3 * CM, 3.4 * CM)
RECT = (
    0.95 * CM / FIGSIZE[0],
    0.9 * CM / FIGSIZE[1],
    AXSIZE[0] / FIGSIZE[0],
    AXSIZE[1] / FIGSIZE[1],
)
CRECT = (
    0.95 * CM / CFIGSIZE[0],
    0.9 * CM / CFIGSIZE[1],
    CAXSIZE[0] / CFIGSIZE[0],
    CAXSIZE[1] / CFIGSIZE[1],
)
GREY = "#666666"
rcParams["font.size"] = 7.25  # default: 10 pts
rcParams["axes.labelpad"] = 0.0  # default: 4.0 pts
rcParams["axes.titlepad"] = 4.0  # default: 6.0 pts


def eigvals2x2(tr, det, as_complex=False):
    D = tr**2 - 4 * det
    if (D < 0).any() or as_complex:
        D = D + 0.0j
    l0 = 0.5 * (tr - torch.sqrt(D))
    l1 = 0.5 * (tr + torch.sqrt(D))
    return l0, l1


def kernel(d, s, rlim, axsize=None):
    r = torch.linspace(rlim[0], rlim[1], 1000)

    if axsize is None:
        axsize = (2.5, 2)

    fig, ax = plt.subplots(1, 1, figsize=axsize)
    for si in s:
        y = laplace_r(d, si**-2, r)
        ax.plot(r, y / y[0], label=f"$\sigma = {si}$ $\mu$m")
    ax.legend()
    ax.axhline(0, color="grey", ls="--")
    ax.set_xlabel(r"$r$ ($\mu$m)")
    # ax.set_ylabel(r"$\frac{G_3(r; \sigma^{-2})}{G_3(20; \sigma^{-2})}$")
    ax.set_ylabel(r"$G_%d(r; \sigma^{-2})$ (normalized)" % d, loc="top")
    return fig


def space(
    d,
    s,
    W,
    rlim,
    N_space=100,
    dh=1,
    normalize=True,
    cell_type="E",
    axsize=(2.5, 2),
):
    if len(s) != 2:
        raise ValueError("Two sigmas are required.")

    cell_type = {"E": "PYR", "I": "PV"}[cell_type]
    rho = s[1] / s[0]

    model = nn.V1(
        ["cell_type", "space"],
        cell_types=["PYR", "PV"],
        sigma_symmetry="pre",
        sigma_optim=False,
        kappa_optim=False,
        tau=[1.0, 0.5],
    )
    x = neurons.as_grid(
        2,
        (N_space,) * d,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    x["dh"] = torch.zeros(x.shape)
    x["dh"][(0, *((N_space // 2,) * d))] = dh
    x["distance"] = x["space"].norm(dim=-1)

    gW = [Wi.abs() for Wi in W]
    W = [Wi * torch.tensor([[1, -1], [1, -1]]) for Wi in W]

    dfs = {}
    for Wi, gWi in zip(W, gW, strict=True):
        model.load_state_dict(
            {"sigma": torch.as_tensor(s).reshape(-1, 2), "gW": gWi},
            strict=False,
        )
        # print(model.state_dict())
        spectral_summary = model.spectral_summary(kind="J")
        print(spectral_summary)
        # if spectral_summary.abscissa > 1:
        #     raise ValueError("Spectral abscissa must be less than 1.")

        with torch.inference_mode():
            df = model(x, ndim=x.ndim).to_pandas()
        df = df.query(f"cell_type == '{cell_type}'")

        if cell_type == "PYR":
            tr = rho * Wi[0, 0] + Wi[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(Wi) + (rho**2 - 1) * Wi[0, 0]
        else:
            tr = rho * Wi[0, 0] + Wi[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(Wi) + (rho ** (-2) - 1) * Wi[1, 1]
        tr, det = tr.item(), det.item()

        if det > tr**2 / 4:
            category = "∞"
        elif det > 0 and tr < 0:
            category = "1"
        else:
            category = "0"
        print(tr, det, category)
        df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}").copy()

        if normalize:
            rmin = df["distance"].min()
            df["dr"] = df["dr"] / df.query(f"distance == {rmin}")["dr"].mean()

        dfs[f"{tr=:.1f}, {det=:.1f}"] = df

    df = pd.concat(dfs, names=["param"]).reset_index(0)

    mapping = {"distance": "Distance ($\mu$m)", "param": "Parameter"}
    mapping["dr"] = "Norm. response" if normalize else "Response"
    lineplot = viz.mapped(sns.lineplot, mapping)

    fig, ax = plt.subplots(1, 1, figsize=axsize)
    lineplot(df, x="distance", y="dr", hue="param", errorbar=None, ax=ax)
    ax.axhline(0, color="grey", ls="--")
    ax.get_legend().set_title(None)
    sns.move_legend(ax, "upper right")

    return fig


def response(
    d,
    s,
    W,
    rlim,
    N_space=100,
    N_ori=50,
    mode="space",
    dh=1,
    cell_type="E",
    normalize=True,
    axsize=(2.5, 2),
):
    if mode not in {"space", "ori", "space_ori"}:
        raise ValueError("mode must be 'space', 'ori' or 'space_ori'")

    if len(s) != 2:
        raise ValueError("Two sigmas are required.")

    cell_type = {"E": "PYR", "I": "PV"}[cell_type]
    rho = s[1] / s[0]

    model = nn.V1(
        ["cell_type", "space", "ori"],
        cell_types=["PYR", "PV"],
        sigma_symmetry="pre",
        sigma_optim=False,
        kappa_optim=False,
    )
    x = neurons.as_grid(
        2,
        (N_space,) * d,
        N_ori,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    x["dh"] = torch.zeros(x.shape)
    x["dh"][(0, *((N_space // 2,) * d), N_ori // 2)] = dh
    x["dori"] = x["ori"].norm(dim=-1)
    x["distance"] = x["space"].norm(dim=-1)

    if mode == "ori":
        kappa = [0.5 * Wi.sign() * torch.tensor([[1, -1], [1, -1]]) for Wi in W]
        gW = [Wi.abs() * 2 for Wi in W]
    elif mode == "space_ori":
        W = list(zip(W[::2], W[1::2], strict=True))
        kappa = [W1 / W0 for W0, W1 in W]
        assert all((k.abs() <= 0.5).all() for k in kappa)
        gW = [W0.abs() for W0, _ in W]
        W = [W0 for W0, _ in W]
    else:
        kappa = [torch.zeros_like(Wi) for Wi in W]
        gW = [Wi.abs() for Wi in W]
        W = [Wi * torch.tensor([[1, -1], [1, -1]]) for Wi in W]

    dfs = {}
    for Wi, gWi, kappai in zip(W, gW, kappa, strict=True):
        model.load_state_dict(
            {"sigma": torch.as_tensor(s).reshape(-1, 2), "gW": gWi, "kappa": kappai},
            strict=False,
        )
        # print(model.state_dict())
        # spectral_summary = model.spectral_summary()
        # print(spectral_summary)
        # if spectral_summary.abscissa > 1:
        #     raise ValueError("Spectral abscissa must be less than 1.")

        with torch.inference_mode():
            df = model(x, ndim=x.ndim).to_pandas()
        df = df.query(f"cell_type == '{cell_type}'")

        if cell_type == "PYR":
            tr = rho * Wi[0, 0] + Wi[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(Wi) + (rho**2 - 1) * Wi[0, 0]
        else:
            tr = rho * Wi[0, 0] + Wi[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(Wi) + (rho ** (-2) - 1) * Wi[1, 1]

        if mode in {"space", "space_ori"}:
            if det > tr**2 / 4:
                category = "∞"
            elif det > 0 and tr < 0:
                category = "1"
            else:
                category = "0"
            print(tr, det, category)
            df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}").copy()

            if mode == "space_ori":
                Wi = Wi * kappai
                tr = rho * Wi[0, 0] + Wi[1, 1] / rho + rho - 1 / rho
                det = torch.linalg.det(Wi) + (rho**2 - 1) * Wi[0, 0]

                if det > tr**2 / 4:
                    category = "∞"
                elif det > 0 and tr < 0:
                    category = "1"
                else:
                    category = "0"
                print(tr, det, category)
            #     df = df.query("dori < 0.1 or dori > 89.9")
            #     assert df["dori"].nunique() == 2
            #     min_dori, max_dori = df["dori"].min(), df["dori"].max()
            #     df = df.groupby(["dori", "distance"])["dr"].mean()
            #     df = df.loc[min_dori] - df.loc[max_dori]
            #     df = df.reset_index()
            if normalize:
                rmin = df["distance"].min()
                df["dr"] = df["dr"] / df.query(f"distance == {rmin}")["dr"].mean()
        elif mode == "ori":
            det = torch.linalg.det(Wi)
            print(det, Wi[0, 0])
            if det > Wi[0, 0]:
                category = "antituned"
            else:
                category = "cotuned"
            df = df.query("distance > 1")

        dfs[category] = df

    df = pd.concat(dfs, names=["category"]).reset_index(0)

    mapping = {
        "distance": "Distance ($\mu$m)",
        "dr": {
            "space": "Norm. response",
            "ori": "Response",
            # "space_ori": "Normalized\niso - ortho response",
            "space_ori": "Normalized response",
        }[mode],
        "dori": "Δ tuning pref. (°)",
        "category": {
            "space": "N crossings",
            "ori": "Tuning",
            "space_ori": "N crossings",
        }[mode],
    }
    lineplot = viz.mapped(sns.lineplot, mapping)
    relplot = viz.mapped(sns.relplot, mapping)
    if mode == "space":
        fig, ax = plt.subplots(1, 1, figsize=axsize)
        lineplot(df, x="distance", y="dr", hue="category", errorbar=None, ax=ax)
        ax.axhline(0, color="grey", ls="--")
        ax.get_legend().set_title(None)
        sns.move_legend(ax, "upper right")
    elif mode == "ori":
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes(RECT)
        lineplot(
            df,
            x="dori",
            y="dr",
            hue="category",
            errorbar=None,
            palette=["C3", "C0"],
            hue_order=["cotuned", "antituned"],
            ax=ax,
        )
        ax.set_xticks([0, 45, 90])
        ax.get_legend().set_title(None)
    else:
        df["dori"] = pd.cut(
            df["dori"], bins=[0, 10, 80, 90], labels=["iso", "neither", "ortho"]
        )
        df = df.query("dori != 'neither'").copy()
        df["dori"] = df["dori"].cat.remove_unused_categories()
        g = relplot(
            df,
            kind="line",
            x="distance",
            y="dr",
            col="category",
            hue="dori",
            errorbar=None,
            height=axsize[1],
            aspect=axsize[0] / axsize[1],
        )
        fig = g.figure
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 1))
        # ax.axhline(0, color="grey", ls="--")

    return fig


def nonlinear(d, s, W, rlim, N_space=100, dh=1, f="SSN", order=None, axsize=(2.5, 2)):
    if len(s) != 2:
        raise ValueError("Two sigmas are required.")

    if len(W) > 1:
        raise ValueError("Only one W can be plotted.")

    if order is None:
        order = [1, 2]

    W = W[0] * torch.tensor([[1, -1], [1, -1]])
    rho = s[1] / s[0]

    tr = rho * W[0, 0] + W[1, 1] / rho + rho - 1 / rho
    det = torch.linalg.det(W) + (rho**2 - 1) * W[0, 0]

    if det > tr**2 / 4:
        category = "∞"
    elif det > 0 and tr < 0:
        category = "1"
    else:
        category = "0"
    print(tr, det, category)

    x = neurons.as_grid(
        2,
        (N_space,) * d,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    x["dh"] = torch.zeros(x.shape)
    x["dh"][(0, *((N_space // 2,) * d))] = dh
    x["distance"] = x["space"].norm(dim=-1)

    dfs = {}
    for o in order:
        model = nn.V1(
            ["cell_type", "space"],
            cell_types=["PYR", "PV"],
            sigma_symmetry="pre",
            f=f,
            nonlinear_kwargs={"max_num_steps": o, "assert_convergence": False},
        )
        model.load_state_dict(
            {"sigma": torch.as_tensor(s).reshape(-1, 2), "gW": W.abs()},
            strict=False,
        )
        # print(model.state_dict())
        # spectral_summary = model.spectral_summary()
        # print(spectral_summary)
        # if spectral_summary.abscissa > 1:
        #     raise ValueError("Spectral abscissa must be less than 1.")

        with torch.inference_mode():
            df = model(x, ndim=x.ndim).to_pandas()
        df = df.query(
            f"cell_type == 'PYR' and distance >= {rlim[0]} and distance < {rlim[1]}"
        ).copy()

        # rmin = df["distance"].min()
        # df["dr"] = df["dr"] / df.query(f"distance == {rmin}")["dr"].mean()

        dfs[o] = df

    df = pd.concat(dfs, names=["order"]).reset_index(0)

    mapping = {"distance": "Distance ($\mu$m)", "dr": "Response"}
    lineplot = viz.mapped(sns.lineplot, mapping)

    fig, ax = plt.subplots(1, 1, figsize=axsize)
    lineplot(df, x="distance", y="dr", hue="order", errorbar=None, ax=ax)
    ax.axhline(0, color="grey", ls="--")
    ax.get_legend().set_title(None)
    sns.move_legend(ax, "upper right")

    return fig


def gain(d, s, W, rlim, N_space=100, dg=1e-1, axsize=(2.5, 2)):
    if len(W) > 1:
        raise ValueError("Only one W can be plotted.")

    rho = s[1] / s[0]

    model = nn.V1(
        ["cell_type", "space"],
        cell_types=["PYR", "PV"],
        tau=[1.0, 0.5],
        sigma_symmetry="pre" if len(s) == 2 else None,
        sigma_optim=False,
        kappa_optim=False,
    )
    x = neurons.as_grid(
        2,
        (N_space,) * d,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    x["dh"] = torch.zeros(x.shape)
    x["dh"][(0, *((N_space // 2,) * d))] = 1
    x["distance"] = x["space"].norm(dim=-1)

    fig, ax = plt.subplots(1, 1, figsize=axsize)
    dfs = {}
    for g in [1, 1 + dg]:
        gW = W[0].abs() * g * torch.tensor([[1, -1], [1, -1]])
        model.load_state_dict(
            {"sigma": torch.as_tensor(s).reshape(-1, 2), "gW": gW.abs()},
            strict=False,
        )
        if (abscissa := model.spectral_summary(kind="J").abscissa) > 0:
            raise ValueError(f"Network is unstable with {abscissa=}.")

        with torch.inference_mode():
            df = model(x, ndim=x.ndim).to_pandas()
        df = df.query("cell_type == 'PYR'")

        if len(s) == 2:
            tr = rho * gW[0, 0] + gW[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(gW) + (rho**2 - 1) * gW[0, 0]

            print(tr, det)
            if det > tr**2 / 4:
                category = "∞"
            elif det > 0 and tr < 0:
                category = "1"
            elif tr < 2 * rho:
                category = "0"
            else:
                raise ValueError("Network is unstable")
            print(category)

        df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}").copy()

        dfs[g] = df

    df = pd.concat(dfs, names=["gain"]).reset_index(0)

    mapping = {
        "distance": "Distance ($\mu$m)",
        "dr": "Response",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)
    lineplot(df, x="distance", y="dr", hue="gain", errorbar=None, ax=ax)
    ax.axhline(0, color="grey", ls="--")

    return fig


def compare(
    d,
    s,
    W,
    kappa,
    rlim,
    N_space=100,
    N_ori=8,
    N_osi=7,
    mode="space_ori",
    approx_order=None,
    axsize=(4.25, 2),
):
    if mode not in {"space_ori", "ori_osi", "space"}:
        raise ValueError("mode must be 'space', 'space_ori' or 'ori_osi'")

    model = nn.V1(
        (
            ["cell_type", "space"]
            if mode == "space"
            else ["cell_type", "space", "ori", "osi"]
        ),
        cell_types=["PYR", "PV"],
        sigma_optim=False,
        kappa_optim=False,
        tau=[1.0, 0.5],
    )
    x = neurons.as_grid(
        2,
        (N_space,) * d,
        N_ori=(0 if mode == "space" else N_ori),
        N_osi=(0 if mode == "space" else N_osi),
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    x["dh"] = torch.zeros(x.shape)
    if mode == "space":
        x["dh"][(0, *((N_space // 2,) * d))] = 1
    else:
        x["dh"][(0, *((N_space // 2,) * d), N_ori // 2, -1)] = 10000

    figsize = (FIGSIZE[0] * 2, FIGSIZE[1])
    rect = (
        0.95 * CM / figsize[0],
        0.9 * CM / figsize[1],
        AXSIZE[0] / figsize[0],
        AXSIZE[1] / figsize[1],
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    model.load_state_dict(
        {
            "sigma": torch.as_tensor(s).reshape(2, 2),
            "gW": torch.as_tensor(W),
            "kappa": torch.as_tensor(kappa),
        },
        strict=False,
    )
    analytic_model = model
    numerical_model = copy.deepcopy(model)
    numerical_model.mode = "numerical"
    numerical_model.simulation_kwargs = {
        "options": {"max_num_steps": 1000},
        "dx_rtol": 1e-5,
    }
    if approx_order:
        approx_model = copy.deepcopy(model)
        approx_model.mode = "matrix_approx"
        approx_model.approx_order = approx_order

    with torch.inference_mode():
        W = analytic_model(x, output="weight", ndim=x.ndim, to_dataframe=False)
        radius = torch.linalg.eigvals(W).abs().max()
        # _, radius = model.spectral_summary()  # slightly less accurate but faster
        print(f'Jacobian abscissa: {model.spectral_summary(kind="J").abscissa}')
        print(f'W spectral radius: {model.spectral_summary(kind="W").radius}')
        df_analytics = analytic_model(x, ndim=x.ndim).to_pandas()
        df_numerics = numerical_model(x, ndim=x.ndim).to_pandas()
        if approx_order:
            df_approx = approx_model(x, ndim=x.ndim).to_pandas()

    df = {"Theory": df_analytics, "Numerics": df_numerics}
    if approx_order:
        df["Approx"] = df_approx

    df = pd.concat(df, names=["Method"]).reset_index(0)
    if mode != "space":
        df["dori"] = df["ori[0]"].abs()

    if mode == "space":
        df = df.query(
            f"cell_type == 'PYR' and `space[0]` >= {rlim[0]} and `space[0]` < {rlim[1]} "
            "and `space[1]` > -0.1 and `space[1]` < 0.1"
        )

        mapping = {
            "space[0]": "Distance (μm)",
            "dr": "Response (a.u.)",
        }
        viz.mapped(sns.lineplot, mapping)(
            df, x="space[0]", y="dr", style="Method", errorbar=None, ax=ax
        )
    elif mode == "space_ori":
        df = df.query(
            f"cell_type == 'PYR' and `space[0]` >= {rlim[0]} and `space[0]` < {rlim[1]} "
            "and `space[1]` > -0.1 and `space[1]` < 0.1 and (dori < 0.1 "
            " or (dori > 44.9 and dori < 45.1) or (dori > 89.9 and dori < 90.1))"
        )

        mapping = {
            "space[0]": "Distance (μm)",
            "dori": "Δ Tuning pref. (°)",
            "dr": "Response (a.u.)",
        }
        viz.mapped(sns.lineplot, mapping)(
            df, x="space[0]", y="dr", hue="dori", style="Method", errorbar=None, ax=ax
        )
    else:
        df = df.query(
            f"cell_type == 'PYR' and `space[0]` >= {rlim[0]} and `space[0]` < {rlim[1]} "
            "and `space[1]` > -0.1 and `space[1]` < 0.1 and (osi < 0.01 or osi > 0.99 "
            "or (osi > 0.49 and osi < 0.51))"
        )

        mapping = {
            "osi": "Selectivity",
            "dori": "Δ Tuning pref. (°)",
            "dr": "Response (a.u.)",
        }
        viz.mapped(sns.lineplot, mapping)(
            df, x="dori", y="dr", hue="osi", style="Method", errorbar=None, ax=ax
        )
        ax.set_xticks([0, 45, 90])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 1))
    ax.axhline(0, color="grey", ls="--")
    ax.text(
        0.25,
        0.85,
        r"$\mathrm{max}_i |\lambda_i| = %.1f$" % radius.item(),
        transform=ax.transAxes,
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot",
        type=str,
        choices=["kernel", "space", "response", "compare", "gain", "nonlinear"],
    )
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("--sigma", "-s", type=float, nargs="+", default=[50, 100])
    parser.add_argument("--wee", type=float, nargs="+", default=[2.0])
    parser.add_argument("--wei", type=float, nargs="+", default=[4.0])
    parser.add_argument("--wie", type=float, nargs="+", default=[4.0])
    parser.add_argument("--wii", type=float, nargs="+", default=[2.0])
    parser.add_argument("--kee", type=float, nargs="+", default=[0.5])
    parser.add_argument("--kei", type=float, nargs="+", default=[-0.25])
    parser.add_argument("--kie", type=float, nargs="+", default=[-0.25])
    parser.add_argument("--kii", type=float, nargs="+", default=[0.25])
    parser.add_argument("--rlim", "-r", type=float, nargs=2, default=[30, 300])
    parser.add_argument("--N-space", type=int)
    parser.add_argument("--N-ori", type=int)
    parser.add_argument("--N-osi", type=int)
    parser.add_argument("--dg", type=float)
    parser.add_argument(
        "--mode", "-m", type=str, choices=["space", "ori", "space_ori", "ori_osi"]
    )
    parser.add_argument("-f", type=str)
    parser.add_argument("--dh", type=float)
    parser.add_argument("--cell-type", "-c", choices=["E", "I"])
    parser.add_argument("--order", type=int, nargs="+")
    parser.add_argument("--approx-order", type=int)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--axsize", type=float, nargs=2)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    W = [
        torch.tensor([[wee, wei], [wie, wii]])
        for wee, wei, wie, wii in zip(
            args.wee, args.wei, args.wie, args.wii, strict=True
        )
    ]
    kappa = [
        torch.tensor([[kee, kei], [kie, kii]])
        for kee, kei, kie, kii in zip(
            args.kee, args.kei, args.kie, args.kii, strict=True
        )
    ]

    kwargs = {
        k: getattr(args, k)
        for k in [
            "N_space",
            "N_ori",
            "N_osi",
            "mode",
            "dg",
            "f",
            "dh",
            "order",
            "approx_order",
            "cell_type",
            "axsize",
        ]
        if getattr(args, k)
    }
    if args.plot == "kernel":
        fig = kernel(args.d, args.sigma, args.rlim, axsize=args.axsize)
    elif args.plot == "space":
        fig = space(args.d, args.sigma, W, args.rlim, axsize=args.axsize)
    elif args.plot == "response":
        fig = response(
            args.d, args.sigma, W, args.rlim, normalize=args.normalize, **kwargs
        )
    elif args.plot == "gain":
        fig = gain(args.d, args.sigma, W, args.rlim, **kwargs)
    elif args.plot == "nonlinear":
        fig = nonlinear(args.d, args.sigma, W, args.rlim, **kwargs)
    elif args.plot == "compare":
        if len(W) > 1 or len(kappa) > 1:
            raise ValueError("Only one W and kappa can be plotted.")
        fig = compare(args.d, args.sigma, W[0], kappa[0], args.rlim, **kwargs)
    # fig.tight_layout()

    if args.out:
        fig.savefig(
            args.out, metadata={"Subject": " ".join(sys.argv[1:])}, bbox_inches="tight"
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
