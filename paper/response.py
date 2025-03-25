import argparse
import sys
import math

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from niarb import viz, nn, neurons
from niarb.nn.modules.v1 import compute_osi_scale
from mpl_config import set_rcParams, get_sizes


def response(
    W, sigma, kappa, N_space, N_ori, N_osi, dh, mode, tau_i=0.5, rtol=1e-5, maxiter=100
):
    if any(Ni % 2 != 0 for Ni in N_space):
        raise ValueError("N_space must be even.")
    if N_ori % 2 != 0:
        raise ValueError("N_ori must be even.")

    sigma = torch.as_tensor(sigma).reshape(2, 2)
    d = len(N_space)

    x = neurons.as_grid(
        2,
        N_space=N_space,
        N_ori=N_ori,
        N_osi=N_osi,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )

    variables = ["cell_type"]
    idx = (0,)
    if N_space:
        variables.append("space")
        idx = idx + tuple(Ni // 2 for Ni in N_space)
        x["distance"] = x["space"].norm(dim=-1)
    if N_ori:
        variables.append("ori")
        idx = idx + (N_ori // 2,)
        x["dori"] = x["ori"].norm(dim=-1)
    if N_osi:
        variables.append("osi")
        idx = idx + (-1,)

    x["dh"] = torch.zeros(x.shape)
    x["dh"][idx] = dh

    model = nn.V1(
        variables,
        cell_types=["PYR", "PV"],
        sigma_optim=False,
        kappa_optim=False,
        tau=[1.0, tau_i],
        mode=mode,
        simulation_kwargs={"dx_rtol": rtol, "options": {"max_num_steps": maxiter}},
    )

    model.double()
    x = x.double()

    with torch.inference_mode():
        model.load_state_dict(
            {"sigma": sigma, "gW": W.abs(), "kappa": kappa}, strict=False
        )
        if (abscissa := model.spectral_summary(kind="J").abscissa) >= 0:
            raise ValueError(f"The model is unstable: {abscissa=}")
        print(model.spectral_summary(kind="J"))
        return model(x, ndim=x.ndim, to_dataframe="pandas")


def plot_ori_response(
    W,
    kappas,
    sigma=(125, 90, 85, 110),
    N_space=(),
    N_ori=12,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    cell_type="PYR",
    **kwargs,
):
    dfs = {}
    for kappa in kappas:
        k = compute_osi_scale(torch.distributions.Uniform(0, 1), nn.Identity())
        tw = W.abs() * kappa * k
        if tw[0, 1] * tw[1, 0] - tw[0, 0] * tw[1, 1] > tw[0, 0]:
            category = "Opp.-favoring"
        else:
            category = "Same-favoring"
        print(category)

        df = response(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0"
        if N_osi:
            query += " and osi == 1"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        df = response(W, sigma, kappa, (), 50, N_osi, dh, "analytical", **kwargs)
        df["dr"] = df["dr"] / math.prod(N_space) * (50 / N_ori)
        dfs[(category, "Theory")] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Response",
        "dori": "Δ tuning pref. (°)",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)

    figsize, rect = get_sizes(1, 1.7, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    lineplot(
        df,
        x="dori",
        y="dr",
        hue="category",
        style="kind",
        errorbar=None,
        palette=["#FF766A", "#88CBEC"],
        hue_order=["Same-favoring", "Opp.-favoring"],
        style_order=["Theory", "Simulation"],
        ax=ax,
    )
    ax.set_xticks([0, 45, 90])
    viz.remove_legend_subtitles(ax, [2, 2], bbox_to_anchor=(1, 1))

    return fig


def plot_space_response(
    Ws,
    sigma=(40, 40),
    kappa=(0.2, 0.1, 0.1, 0.0),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    rlim=(30, 300),
    normalize=False,
    cell_type="PYR",
    **kwargs,
):
    if len(sigma) != 2:
        raise ValueError("sigma must be a tuple of length 2.")

    rho = sigma[1] / sigma[0]
    sigma = (*sigma, *sigma)
    kappa = torch.tensor(kappa).reshape(2, 2)

    dfs = {}
    for W in Ws:
        if cell_type == "PYR":
            tr = rho * W[0, 0] + W[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(W) + (rho**2 - 1) * W[0, 0]
        else:
            tr = rho * W[0, 0] + W[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(W) + (rho ** (-2) - 1) * W[1, 1]

        if det > tr**2 / 4:
            category = "∞"
        elif det > 0 and tr < 0:
            category = "1"
        else:
            category = "0"
        print(tr, det, category)

        df = response(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        df = response(W, sigma, kappa, N_space, 0, 0, dh, "analytical", **kwargs)
        df["dr"] = df["dr"] / ((N_ori or 1) * (N_osi or 1))
        dfs[(category, "Theory")] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])
    df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}")
    df = df.reset_index(drop=True)
    if normalize:
        norm = np.zeros(len(df))
        for category, sf in df.groupby("category", observed=True):
            norm[sf.index] = sf.query("kind == 'Theory'")["dr"].max()
        df["dr"] = df["dr"] / norm

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Norm. response" if normalize else "Response",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)

    figsize, rect = get_sizes(1, 1.6, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    lineplot(
        df,
        x="distance",
        y="dr",
        hue="category",
        style="kind",
        style_order=["Theory", "Simulation"],
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0, color="grey", ls="--")
    viz.remove_legend_subtitles(ax, [3, 2], bbox_to_anchor=(1, 1))

    return fig


def plot_space_ori_response(
    Ws,
    kappas,
    sigma=(100, 100),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    rlim=(30, 300),
    normalize=False,
    cell_type="PYR",
    **kwargs,
):
    if len(sigma) != 2:
        raise ValueError("sigma must be a tuple of length 2.")

    rho = sigma[1] / sigma[0]
    sigma = (*sigma, *sigma)

    dfs = {}
    for W, kappa in zip(Ws, kappas, strict=True):
        k = compute_osi_scale(torch.distributions.Uniform(0, 1), nn.Identity())
        tW = W * kappa * k
        if cell_type == "PYR":
            tr = rho * tW[0, 0] + tW[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(tW) + (rho**2 - 1) * tW[0, 0]
        else:
            tr = rho * tW[0, 0] + tW[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(tW) + (rho ** (-2) - 1) * tW[1, 1]

        if det > tr**2 / 4:
            category = "∞"
        elif det > 0 and tr < 0:
            category = "1"
        else:
            category = "0"
        print(tr, det, category)

        df = response(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0 and (dori == 0 or dori == 90)"
        if N_osi:
            query += " and osi == 1"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        df = response(
            W, sigma, kappa, N_space, N_ori, N_osi, dh, "analytical", **kwargs
        )
        dfs[(category, "Theory")] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])
    df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}")
    df = df.reset_index(drop=True)
    if normalize:
        norm = np.zeros(len(df))
        for category, sf in df.groupby("category", observed=True):
            norm[sf.index] = sf.query("kind == 'Theory'")["dr"].max()
        df["dr"] = df["dr"] / norm

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Norm. response" if normalize else "Response",
        "dori": "Δ pref. (°)",
        "category": "# transitions",
    }
    relplot = viz.mapped(sns.relplot, mapping)

    figsize, _ = get_sizes(1, 1, 0.8, 1)
    g = relplot(
        df,
        kind="line",
        x="distance",
        y="dr",
        row="category",
        hue="dori",
        style="kind",
        row_order=["0", "1"],
        style_order=["Theory", "Simulation"],
        errorbar=None,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        palette=["#AA2A6C", "#4A489C"],
    )

    return g.figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wee", type=float, nargs="+")
    parser.add_argument("--wei", type=float, nargs="+")
    parser.add_argument("--wie", type=float, nargs="+")
    parser.add_argument("--wii", type=float, nargs="+")
    parser.add_argument("--kee", type=float, nargs="*", default=())
    parser.add_argument("--kei", type=float, nargs="*", default=())
    parser.add_argument("--kie", type=float, nargs="*", default=())
    parser.add_argument("--kii", type=float, nargs="*", default=())
    parser.add_argument("--N-space", type=int, nargs="*", default=())
    parser.add_argument("--N-ori", type=int, default=0)
    parser.add_argument("--N-osi", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tau-i", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--mode", "-m", choices=["space", "ori", "space_ori"])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    Ws = [
        torch.tensor(W).reshape(2, 2) * torch.tensor([1, -1])
        for W in zip(args.wee, args.wei, args.wie, args.wii, strict=True)
    ]
    kappas = [
        torch.tensor(kappa).reshape(2, 2)
        for kappa in zip(args.kee, args.kei, args.kie, args.kii, strict=True)
    ]

    set_rcParams()
    if args.mode == "ori":
        fig = plot_ori_response(
            Ws[0],
            kappas,
            N_space=args.N_space,
            N_ori=args.N_ori,
            N_osi=args.N_osi,
            scale=args.scale,
            tau_i=args.tau_i,
            rtol=args.rtol,
            maxiter=args.maxiter,
        )
    elif args.mode == "space":
        fig = plot_space_response(
            Ws,
            N_space=args.N_space,
            N_ori=args.N_ori,
            N_osi=args.N_osi,
            scale=args.scale,
            normalize=args.normalize,
            tau_i=args.tau_i,
            rtol=args.rtol,
            maxiter=args.maxiter,
        )
    else:
        fig = plot_space_ori_response(
            Ws,
            kappas,
            N_space=args.N_space,
            N_ori=args.N_ori,
            N_osi=args.N_osi,
            scale=args.scale,
            normalize=args.normalize,
            tau_i=args.tau_i,
            rtol=args.rtol,
            maxiter=args.maxiter,
        )

    if args.out:
        fig.savefig(
            args.out, bbox_inches="tight", metadata={"Subject": " ".join(sys.argv)}
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
