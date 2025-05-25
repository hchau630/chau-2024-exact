import argparse
import sys
import math
from numbers import Number
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
import numpy as np
import torch

from niarb import nn, neurons, random
from niarb.nn.modules.frame import ParameterFrame
from niarb.zero_crossing import find_root
from mpl_config import FIGSIZE, CFIGSIZE, RECT, CRECT, GREY, set_rcParams


def sample2x2(tr, det, rho, w00=None, scale=1.0):
    r"""
    tr: trace of (I - W)\tilde{S}^{-1} = \rho^{-1} (1 - w00) + \rho (1 - w11)
    det: determinant of (I - W)\tilde{S}^{-1} = (1 - w00) (1 - w11) - w01 w10
    """
    if w00 is None:
        tr, det = np.broadcast_arrays(tr, det)
        # A sufficient condition for w11 < 0 and w01w10 < 0 is that
        # w00 must be greater than max(rho**2 - rho * tr + 1, 1)
        min_w00 = np.maximum(rho**2 - rho * tr + 1, 1)
        w00 = min_w00 + scale * np.abs(np.random.randn(tr.shape))
        # w00 = min_w00 + 1e-8
    else:
        tr, det, w00 = np.broadcast_arrays(tr, det, w00)

    w11 = 1 - (tr - (1 - w00) / rho) / rho
    if (w11 > 0).any():
        raise ValueError(f"w11 must be less than 0, but {np.max(w11)=}")

    w0110 = (1 - w00) * (1 - w11) - det
    if (w0110 > 0).any():
        raise ValueError(f"w0110 must be less than 0, but {np.max(w0110)=}")

    w10 = scale * np.abs(np.random.randn(tr.shape))
    w01 = w0110 / w10

    W = np.stack([[w00, w01], [w10, w11]])  # (2, 2, *shape)
    W = np.moveaxis(W, (0, 1), (-2, -1))  # (*shape, 2, 2)

    return W


def eigvals2x2(tr, det, as_complex=False):
    D = tr**2 - 4 * det
    if (D < 0).any() or as_complex:
        D = D.astype(complex)
    l0 = 0.5 * (tr - np.sqrt(D))
    l1 = 0.5 * (tr + np.sqrt(D))
    return l0, l1


def trdet2w00w11(tr, det, rho, w0110=-1):
    D = tr**2 - 4 * (det + w0110)
    return 1 + rho * (-tr + np.sqrt(D)) / 2, 1 - (tr + np.sqrt(D)) / (2 * rho)


def steady_state(d, gW, sigma, x):
    """
    Args:
        d: int
        gW: Tensor with shape (*, 2, 2)
        sigma: Tensor with shape (*, 1/2, 2)
        x: ndarray with shape (N,)
    """
    if sigma.ndim < 2:
        raise ValueError("sigma must have at least 2 dimensions")

    if sigma.shape[-2:] == (2, 2):
        symmetry = None
    elif sigma.shape[-2:] == (1, 2):
        symmetry = "pre"
    elif sigma.shape[-2:] == (2, 1):
        symmetry = "post"
    elif sigma.shape[-2:] == (1, 1):
        symmetry = "full"
    else:
        raise ValueError(f"Invalid shape for sigma: {sigma.shape}")

    shape = torch.broadcast_shapes(gW.shape[:-2], sigma.shape[:-2])
    model = nn.V1(
        ["cell_type", "space"],
        cell_types=["PYR", "PV"],
        init_stable=False,
        sigma_symmetry=symmetry,
        batch_shape=shape,
    )
    model.load_state_dict({"gW": gW, "sigma": sigma}, strict=False)
    space = torch.tensor(np.r_[0, x], dtype=torch.float)  # (N + 1,)
    space = torch.stack(
        [space, *([torch.zeros_like(space)] * (d - 1))], dim=-1
    )  # (N + 1, d)
    dh = torch.zeros((2, space.shape[0]))  # (2, N + 1,)
    dh[0, 0] = 1.0
    x = ParameterFrame(
        {
            "cell_type": torch.tensor([[0], [1]]),  # (2, 1)
            "space": space[None, ...],  # (1, N + 1, d)
            "dV": torch.tensor([[1.0]]),  # (1, 1)
            "dh": dh,  # (2, N + 1)
        },
        ndim=2,
    )  # (2, N + 1)
    with torch.inference_mode():
        out = model(
            x, ndim=x.ndim, check_circulant=False, to_dataframe=False
        )  # (*, 2, N + 1)
    return out["dr"][..., 1:]  # (*, 2, N)


def frho0(l0, l1, w11, rho, d=3):
    out = np.stack(
        [
            np.stack([(1 - l0 * rho) * (1 - l0 / rho - w11), 1 - l0 * rho]),
            np.stack([1 - l0 / rho, (1 - l0 / rho) * (1 - l1 / rho - w11)]),
        ]
    )
    if d < 2:
        out = out * l1**0.5
    return out


def frho1(l0, l1, w11, rho, d=3):
    out = np.stack(
        [
            np.stack([(1 - l1 * rho) * (1 - l1 / rho - w11), 1 - l1 * rho]),
            np.stack([1 - l1 / rho, (1 - l1 / rho) * (1 - l0 / rho - w11)]),
        ]
    )
    if d < 2:
        out = out * l0**0.5
    return out


def plot_space_phases(
    tr, det, rho=1, w00=None, numerical=False, coord="I-W", transpose=False, axsize=None
):
    r"""
    Note: here rho is defined as s1 / s0, as opposed to s0 / s1 as in the rest of the code.
    If coord == "I - W", then
        tr: trace of (I - W)\tilde{S}^{-1} = \rho * (1 - w00) + \rho^{-1} * (1 - w11)
        det: determinant of (I - W)\tilde{S}^{-1} = (1 - w00) * (1 - w11) - w01 * w10
    else:
        tr: \mathrm{tr}(W\tilde{S}^{-1}) + \rho - \rho^{-1} = \rho w00 - \rho^{-1} |w11| + \rho - \rho^{-1}
        det: \det(W) - \tr(W(I - \rho^{\pm1}\tilde{S}^{-1}))
    where \tilde{S} = diag([\rho^{-1}, \rho])
    """
    if coord not in {"I-W", "W"}:
        raise ValueError(f"Invalid value for argument `coord`: {coord}")

    if isinstance(rho, Number):
        rho = [rho]

    xx, yy = tr, det

    if numerical:
        shape = np.broadcast_shapes(tr.shape, det.shape)
        model = nn.V1(
            ["cell_type", "space"],
            cell_types=["PYR", "PV"],
            sigma_symmetry="pre",
            sigma_optim=False,
            batch_shape=shape,
        )
        x = neurons.as_grid(
            2, (1500, 1, 1), cell_types=["PYR", "PV"], space_extent=(30, 30, 30)
        )
        x["dh"] = torch.zeros(x.shape)
        x["dh"][0, 0, 0, 0] = 1.0

    if axsize is None:
        axsize = (2.25, 2.25)

    if transpose:
        fig, axes = plt.subplots(
            len(rho), 2, figsize=(2 * axsize[0], len(rho) * axsize[1])
        )
        axes = axes.reshape(-1, 2).T
    else:
        fig, axes = plt.subplots(
            2, len(rho), figsize=(len(rho) * axsize[0], 2 * axsize[1])
        )
        axes = axes.reshape(2, -1)

    if coord == "I-W":
        fig.supxlabel(r"$\rho (1 - w_{EE}) + \rho^{-1} (1 + |w_{II}|)$")
        fig.supylabel(r"$(1 - w_{EE}) (1 + |w_{II}|) + |w_{EI}| w_{IE}$")
    else:
        fig.supxlabel(r"$\rho w_{EE} - \rho^{-1} |w_{II}| + \rho - \rho^{-1}$")
        for j in range(len(rho)):
            if not transpose and j > 0:
                continue
            axes[0, j].set_ylabel(
                r"$|w_{EI}| w_{IE} - w_{EE} |w_{II}|$"
                + "\n"
                + r"$+ (\rho^2 - 1)w_{EE}$",
            )
            axes[1, j].set_ylabel(
                r"$|w_{EI}| w_{IE} - w_{EE} |w_{II}|$"
                + "\n"
                + r"$+ (1 - \rho^{-2})|w_{II}|$",
            )

    if w00 is not None:
        fig.suptitle(r"$w_{EE} = %.2f$" % w00)
        titles = [r"E $\to$ E or I $\to$ E", r"E $\to$ I or I $\to$ I"]
    else:
        fig.suptitle(r"$w_{EE} > 1$")
        titles = [r"E $\to$ E", r"E $\to$ I"]

    for i, j in np.ndindex((2, len(rho))):
        ax = axes[i, j]

        if coord == "W":
            if i == 0:
                tr, det = 2 * rho[j] - xx, rho[j] ** 2 - xx * rho[j] + yy
            else:
                tr, det = 2 / rho[j] - xx, rho[j] ** (-2) - xx / rho[j] + yy

        l0, l1 = eigvals2x2(tr, det)

        D = tr**2 - 4 * det
        is_real = (D >= 0) & (l0.real > 0)

        l0real = l0.real.copy()
        l0real[D < 0] = 1

        if numerical:
            with random.set_seed(0):
                W = sample2x2(tr, det, 1 / rho[j], w00=w00)  # (*shape, 2, 2)
            W = torch.from_numpy(W).to(torch.float)
            sigma = torch.tensor([[1.0, rho[j]]], dtype=torch.float)
            sigma = sigma.broadcast_to(W.shape[:-2] + sigma.shape)
            model.load_state_dict({"gW": W.abs(), "sigma": sigma}, strict=False)
            with torch.inference_mode():
                y = model(x, ndim=x.ndim, to_dataframe=False)  # (*shape, 2, N, 1, 1)
            y = y.squeeze((-2, -1))  # (*shape, 2, N)
            y = y.iloc[..., 1 : y.shape[-1] // 2]  # (*shape, 2, ?)
            dr = y["dr"]  # (*shape, 2, ?)
            N_cross = dr.signbit().diff(dim=-1).count_nonzero(dim=-1)  # (*shape, 2)
        elif w00 is not None:
            w11 = 1 - (tr - (1 - w00) * rho[j]) * rho[j]
            w0110 = (1 - w00) * (1 - w11) - det
            z = frho0(l0, l1, w11, 1 / rho[j]) / frho1(l0, l1, w11, 1 / rho[j])
            z[:, :, ~is_real] = np.nan
            z = z.real
        else:
            z = np.stack([l0 - rho[j], l0 - 1 / rho[j]])
            z[:, ~is_real] = np.nan
            z = z.real
            # w00_, w11_ = trdet2w00w11(tr, det, 1 / rho[j], w0110=-1)

        ax.contourf(xx, yy, D, levels=[-1e8, 0, 1e8], colors=["C0", "C2"])

        if numerical:
            cs = ax.contourf(
                xx,
                yy,
                N_cross[..., i],
                levels=[-0.5, 0.5, 1.5, 100],
                colors=["C0", "C1", "C2"],
            )
            fig.colorbar(cs, ax=ax)
        elif w00 is not None:
            ax.contourf(xx, yy, z[i, 0], levels=[0, 1], colors=["C1"])
        else:
            ax.contourf(xx, yy, z[i], levels=[0, 1e8], colors=["C1"])
            # w00_, w11_ = trdet2w00w11(tr, det, 1 / rho[j], w0110=-1)
            # cs = ax.contour(
            #     xx, yy, w00_, levels=np.linspace(0, np.nanmax(w00_), 10)
            # )
            # # fig.colorbar(cs, ax=ax, label=r"$w_{EE}$")
            # cs = ax.contour(
            #     xx,
            #     yy,
            #     -w11_,
            #     levels=np.linspace(0, np.nanmax(-w11_), 10),
            #     linestyles="dashed",
            # )
            # # fig.colorbar(cs, ax=ax, label=r"$-w_{II}$")
        ax.contourf(xx, yy, l0real, levels=[-1e8, 0], colors="black")
        if w00 is not None:
            ax.contourf(xx, yy, w11, levels=[0, 1e8], colors="red")
            ax.contourf(xx, yy, w0110, levels=[0, 1e8], colors="purple")
        ax.set_title(r"%s, $\rho = %.2f$" % (titles[i], rho[j]))
    return fig


def plot_space_phases_2(x, y, rho=1, cell_type="E", mode=None, axsize=(2.25, 1.85)):
    r"""
    Note: here rho is defined as s1 / s0, as opposed to s0 / s1 as in the rest of the code.
    x: \mathrm{tr}(W\tilde{S}^{-1}) + \rho - \rho^{-1} = \rho w00 - \rho^{-1} |w11| + \rho - \rho^{-1}
    y: \det(W) - \tr(W(I - \rho^{\pm1}\tilde{S}^{-1}))
    where \tilde{S} = diag([\rho^{-1}, \rho])
    """
    if cell_type not in {"E", "I", "EI"}:
        raise ValueError(f"Invalid value for argument `cell_type`: {cell_type}")

    if mode and mode not in {"decay"}:
        raise ValueError(f"Invalid argument `mode`: {mode}")

    if isinstance(rho, Number):
        rho = [rho]

    if mode == "decay" and len(rho) > 1:
        raise ValueError("Only one value of rho is allowed for mode='decay'")

    if cell_type == "EI" and len(rho) > 1:
        raise ValueError("Only one value of rho is allowed for cell_type='EI'")

    fig = plt.figure(figsize=CFIGSIZE if mode == "decay" else FIGSIZE)
    ax = fig.add_axes(CRECT if mode == "decay" else RECT)

    if len(rho) == 1 and rho[0] == 1:
        ax.set_xlabel(r"$w_{EE} - |w_{II}|$")
        # ax.set_ylabel(r"$|w_{EI}| w_{IE} - w_{EE} |w_{II}|$")
        ax.set_ylabel(r"$\mathrm{det}(\mathbf{W})$")
    else:
        ax.set_xlabel(
            r"$\rho w_{EE} - \rho^{-1} |w_{II}| + \rho - \rho^{-1}$",
            labelpad=0,
        )
        if cell_type in {"E", "EI"}:
            ax.set_ylabel(
                # r"$|w_{EI}| w_{IE} - w_{EE} |w_{II}|$"
                # + "\n"
                r"$\mathrm{det}(\mathbf{W}) $" + r"$+ (\rho^2 - 1)w_{EE}$",
                labelpad=0,
            )
        else:
            ax.set_ylabel(
                # r"$|w_{EI}| w_{IE} - w_{EE} |w_{II}|$"
                # +"\n"
                r"$\mathrm{det}(\mathbf{W}) $" + r"$+ (1 - \rho^{-2})|w_{II}|$",
                labelpad=0,
            )

    if cell_type == "EI" or mode == "decay":
        ax.set_title(r"$\rho = %g$" % rho[0])

    z = np.zeros_like(x)
    z[(x < 0) & (y >= 0) & (y < x**2 / 4)] = 1
    z[y >= x**2 / 4] = 2

    if cell_type == "EI":
        x0 = 2 * (rho[0] - 1 / rho[0])
        y0 = (rho[0] - 1 / rho[0]) * (x - (rho[0] - 1 / rho[0]))
        z[(x < x0) & (y > y0) & (y < x**2 / 4)] += 0.5

    ax.contour(
        x, y, z, levels=[0.5, 1.5], colors=GREY, linewidths=rcParams["grid.linewidth"]
    )
    if mode != "decay":
        if cell_type == "EI":
            c = ["C2", "C4", "C1", "C3", "C0"]
            ax.contourf(x, y, z, levels=np.arange(-0.25, 2.26, 0.5), colors=c)
        else:
            ax.contourf(
                x, y, z, levels=[-0.5, 0.5, 1.5, 2.5], colors=["C2", "C1", "C0"]
            )

    for rhoi in rho:
        if cell_type == "E":
            tr = 2 * rhoi - x
            det = rhoi**2 - x * rhoi + y
            z2 = y - x * rhoi + rhoi**2
            z2[x >= 2 * rhoi] = np.nan
        else:
            tr = 2 / rhoi - x
            det = 1 / rhoi**2 - x / rhoi + y
            z2 = y - x / rhoi + 1 / rhoi**2
            z2[x >= 2 / rhoi] = np.nan

        if mode == "decay":
            l0, l1 = eigvals2x2(tr, det, as_complex=True)
            decay_rate = np.minimum((l0**0.5).real, (l1**0.5).real)
            decay_rate[decay_rate <= 0] = np.nan
            tau = 1 / decay_rate

            levels = np.logspace(np.log10(np.nanmin(tau)), 1)

            cnorm = colors.CenteredNorm()
            loghalfrange = max(abs(levels[0]), abs(levels[1]))
            norm = colors.FuncNorm(
                (lambda x: cnorm(np.log10(x)), lambda x: 10 ** cnorm.inverse(x)),
                vmin=10**-loghalfrange,
                vmax=10**loghalfrange,
            )

            cs = ax.contourf(x, y, tau, norm=norm, levels=levels, cmap="bwr")
            # min_tick, max_tick = np.ceil(levels[0]), np.floor(levels[-1])
            # cbar.set_ticks(np.linspace(min_tick, max_tick, 5))
            ticks = 10.0 ** np.arange(
                math.ceil(math.log10(levels[0])),
                math.floor(math.log10(levels[-1])) + 1,
                1,
            )
            cbar = fig.colorbar(cs, format="%g")
            cbar.set_ticks(ticks=ticks)
            cbar.set_label(label=r"$\frac{\sigma_\infty}{\sqrt{\sigma_E \sigma_I}}$")

        z2[(y < x**2 / 4) & np.isnan(z2)] = -1
        ax.contourf(x, y, z2, levels=[-1e8, 0], colors="black", alpha=1 / len(rho))

        if cell_type == "EI":
            z3 = y - (rhoi - 1 / rhoi) * (x - (rhoi - 1 / rhoi))
            ax.contour(
                x, y, z3, levels=[0], colors="black", linestyles="--", linewidths=1
            )

    if cell_type != "EI":
        ax.set_yticks([y.min(), 0, y.max()])

    return fig


def plot_space_zero_loc(
    x,
    y,
    rho=(1,),
    w00=None,
    w11=None,
    mode=None,
    bias=0,
    vector_fields=(),
    shade=None,
    levels=(-2, 1.5),
    N_levels=40,
    linthresh=1,
    spacing="log",
    cell_type=None,
    symmetry="pre",
    numerical=False,
    Nr=1000,
    rmax=10.0,
    d=3,
    dg=1e-5,
    coord="W",
    axsize=(2.5, 2),
):
    r"""
    If coord == "I - W" and symmetry == "pre":
        x: \tr((I - W)\tilde{S}^{-1}) = \rho (1 - w00) + \rho^{-1}(1 - w11)
        y: \det((I - W)\tilde{S}^{-1}) = (1 - w00)(1 - w11) - w01w10
    elif coord == "W" and symmetry == "pre":
        x_\pm: \rho w00 - \rho^{-1}|w11| \pm (\rho - \rho^{-1})
        y_+: |w01|w10 - w00|w11| + (\rho^2 - 1)w00
        y_-: |w01|w10 - w00|w11| + (1 - \rho^2)|w11|
    elif coord == "W" and symmetry == "diag":
        x: \tr(W) = w00 - |w11|
        y: \det(W) = |w01|w10 - w00|w11|
        # x_\pm: \rho(w00 - |w11|) \pm (\rho - \rho^{-1})
        # y_+: |w01|w10 - w00|w11| + (\rho^2 - 1)(w00 - |w11|)
    else:
        x: \tr(I - W) = 2 - (w00 - |w11|)
        y: \det(I - W) = 1 - (w00 - |w11|) + (|w01| * w10 - w00 * |w11|)
        # x: \rho(1 - w00 - w11) + \rho^{-1}
        # y: (1 - w00)(1 - w11) - w01w10
    where \tilde{S} = diag([\rho^{-1}, \rho])
    """
    if coord not in {"I-W", "W"}:
        raise ValueError(f"Invalid value for argument `coord`: {coord}")

    if mode and mode not in {
        "ratio",
        "diff_EI",
        "rel_diff",
        "diff",
        "min_diff",
        "gain",
        "num",
    }:
        raise ValueError(f"Invalid mode: {mode}")

    if spacing not in {
        "log",
        "halflog",
        "neghalflog",
        "symlog",
        "linear",
        "halflinear",
    }:
        raise ValueError(f"Invalid value for argument `spacing`: {spacing}")

    if symmetry not in {"pre", "diag"}:
        raise ValueError(f"Invalid value for argument `symmetry`: {symmetry}")

    if symmetry == "diag" and not numerical:
        raise ValueError("symmetry='diag' requires numerical=True")

    if (w00 is None) == (w11 is None):
        raise ValueError("Exactly one of `w00` and `w11` must be provided")

    if bias != 0 and not numerical:
        raise ValueError("bias != 0 requires numerical=True")

    if cell_type is None:
        cell_type = [0, -1]
    elif cell_type not in {"E", "I"}:
        raise ValueError(f"Invalid value for argument `cell_type`: {cell_type}")
    else:
        cell_type = {"E": [0], "I": [-1]}[cell_type]

    for f in vector_fields:
        if f not in {"w00", "absw11"}:
            raise ValueError(f"Invalid value for argument `vector_fields`: {f}")

    w00_passed = w00 is not None

    if len(rho) == len(cell_type) == 1:
        fig = plt.figure(figsize=CFIGSIZE)
        axes = fig.add_axes(CRECT)
        axes = np.array([[axes]])
    else:
        fig = plt.figure(figsize=(len(rho) * CFIGSIZE[0], len(cell_type) * CFIGSIZE[1]))
        axes = []
        for i, j in np.ndindex(len(cell_type), len(rho)):
            rect = [
                (CRECT[0] + j) / len(rho),
                (CRECT[1] + i) / len(cell_type),
                CRECT[2] / len(rho),
                CRECT[3] / len(cell_type),
            ]
            ax = fig.add_axes(rect)
            axes.append(ax)
        axes = np.array(axes).reshape(len(cell_type), len(rho))
        # fig, axes = plt.subplots(
        #     len(cell_type),
        #     len(rho),
        #     figsize=(len(rho) * CFIGSIZE[0], len(cell_type) * CFIGSIZE[1]),
        # )
        # axes = axes.reshape(len(cell_type), len(rho))

    if coord == "I - W":
        fig.supxlabel(r"$\rho (1 - w_{EE}) + \rho^{-1} (1 + |w_{II}|)$")
        fig.supylabel(r"$(1 - w_{EE}) (1 + |w_{II}|) + |w_{EI}| w_{IE}$")
    elif symmetry == "diag" or (len(rho) == len(cell_type) == 1 and rho[0] == 1):
        axes[0, 0].set_xlabel(r"$w_{EE} - |w_{II}|$")
        # axes[0, 0].set_ylabel(r"$|w_{EI}|w_{IE} - w_{EE}|w_{II}|$")
        axes[0, 0].set_ylabel(r"$\mathrm{det}(\mathbf{W})$")
        if w00_passed:
            axes[0, 0].set_title(r"$w_{EE} = %g, \rho = %g$" % (w00, rho[0]))
        else:
            axes[0, 0].set_title(r"$w_{II} = %g, \rho = %g$" % (w11, rho[0]))
    else:
        xlabel = r"$\rho w_{EE} - \rho^{-1} |w_{II}| + \rho - \rho^{-1}$"
        if axes.shape[1] == 1:
            axes[-1, 0].set_xlabel(xlabel)
        else:
            fig.supxlabel(xlabel)

        # ylabel = r"$|w_{EI}|w_{IE} - w_{EE}|w_{II}|$" + "\n"
        ylabel = r"$\mathrm{det}(\mathbf{W}) $"
        ylabels = [
            ylabel + r"$+ (\rho^2 - 1)w_{EE}$",
            ylabel + r"$+ (1 - \rho^{-2})|w_{II}|$",
        ]

        for i, j in product(cell_type, range(len(rho))):
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(ylabels[i])
            if i == 0:
                if w00_passed:
                    ax.set_title(r"$w_{EE} = %g, \rho = %g$" % (w00, rho[j]))
                else:
                    ax.set_title(r"$w_{II} = %g, \rho = %g$" % (w11, rho[j]))

    if mode == "ratio":
        cbar_labels = [
            r"$\frac{r^\mathrm{I}_0}{r^\mathrm{E}_0}$",
            r"$\frac{r^\mathrm{EI}_0}{r^\mathrm{EE}_0}$",
        ]
    elif mode == "diff_EI":
        cbar_labels = [
            # r"$r^\mathrm{I}_0 - r^\mathrm{E}_0$",
            # r"$r^\mathrm{EI}_0 - r^\mathrm{EE}_0$",
            r"$\frac{r^\mathrm{I}_0 - r^\mathrm{E}_0}{\sqrt{\sigma_E \sigma_I}}$",
            r"$\frac{r^\mathrm{EI}_0 - r^\mathrm{EE}_0}{\sqrt{\sigma_E \sigma_I}}$",
        ]
    elif mode == "rel_diff":
        cbar_labels = [
            r"$\frac{r^{(1)}_{EE} - r^{(0)}_{EE}}{r^{(0)}_{EE}}$",
            r"$\frac{r^{(1)}_{IE} - r^{(0)}_{IE}}{r^{(0)}_{IE}}$",
        ]
    elif mode == "diff":
        # cbar_labels = [r"$r_1 - r_0$", r"$r_1 - r_0$"]
        cbar_labels = [r"$\frac{r_1 - r_0}{\sqrt{\sigma_E \sigma_I}}$"] * 2
    elif mode == "min_diff":
        # cbar_labels = [r"$r^\mathrm{min}_0 - r_0$", r"$r^\mathrm{min}_0 - r_0$"]
        cbar_labels = [r"$\frac{r^\mathrm{min}_0 - r_0}{\sqrt{\sigma_E \sigma_I}}$"] * 2
    elif mode == "gain":
        cbar_labels = [
            r"$\frac{1}{r_0}\frac{dr_0}{dg}$",
            r"$\frac{1}{r_0}\frac{dr_0}{dg}$",
        ]
    elif mode == "num":
        cbar_labels = ["# of crossings", "# of crossings"]
    else:
        # cbar_labels = [r"$r_{EE}$ or $r_{EI}$", r"$r_{IE}$ or $r_{II}$"]
        # cbar_labels = [r"$r_0$", r"$r_0$"]
        cbar_labels = [r"$\frac{r_0}{\sqrt{\sigma_E \sigma_I}}$"] * 2

    cnorm = colors.CenteredNorm()
    if spacing == "log":
        loghalfrange = max(abs(levels[0]), abs(levels[1]))
        norm = colors.FuncNorm(
            (lambda x: cnorm(np.log10(x)), lambda x: 10 ** cnorm.inverse(x)),
            vmin=10**-loghalfrange,
            vmax=10**loghalfrange,
        )
        levels = np.logspace(*levels, N_levels)
    elif spacing == "halflog":
        norm = colors.SymLogNorm(
            linthresh, vmin=-(10 ** levels[1]), vmax=10 ** levels[1]
        )
        levels = np.logspace(*levels, N_levels)
    elif spacing == "neghalflog":
        norm = colors.SymLogNorm(
            linthresh, vmin=-(10 ** levels[1]), vmax=10 ** levels[1]
        )
        levels = -np.logspace(*levels, N_levels)[::-1]
    elif spacing == "symlog":
        norm = colors.SymLogNorm(linthresh)
        levels = np.logspace(*levels, N_levels)
        levels = np.r_[-levels[::-1], 0, levels]
    elif spacing == "linear":
        norm = colors.CenteredNorm(vcenter=1)
        levels = np.linspace(*levels, N_levels)
    elif spacing == "halflinear":
        norm = cnorm
        levels = np.linspace(*levels, N_levels)

    for i, j in product(cell_type, range(len(rho))):
        ax = axes[i, j]

        # compute trace and determinant of (I - W)\tilde{S}^{-1} if symmetry == "pre"
        # else compute trace and determinant of I - W
        if symmetry == "pre":
            if coord == "W":
                if i == 0:
                    tr, det = 2 * rho[j] - x, rho[j] ** 2 - x * rho[j] + y
                else:
                    tr, det = 2 / rho[j] - x, rho[j] ** (-2) - x / rho[j] + y
            else:
                tr, det = x, y

            if w00_passed:
                w11 = 1 - (tr - (1 - w00) * rho[j]) * rho[j]
            else:
                w00 = 1 - (tr - (1 - w11) / rho[j]) / rho[j]
            w0110 = (1 - w00) * (1 - w11) - det

        else:
            if coord == "I - W":
                raise NotImplementedError()

            if w00_passed:
                w11 = x - w00
            else:
                w00 = x - w11
            w0110 = w00 * w11 - y

        if vector_fields:
            if symmetry != "pre" or coord != "W":
                raise NotImplementedError()

            if cell_type == 0:
                fields = {
                    "w00": (rho[j], rho[j] ** 2 - 1 + w11),
                    "absw11": (-1 / rho[j], -w00),
                }
            else:
                fields = {
                    "w00": (rho[j], w11),
                    "absw11": (-1 / rho[j], 1 - 1 / rho[j] ** 2 - w00),
                }
            vector_fields = {f: fields[f] for f in vector_fields}

        if numerical:
            w00_ = torch.tensor(w00, dtype=torch.float)
            w11_ = torch.tensor(w11, dtype=torch.float)
            w0110_ = torch.tensor(w0110, dtype=torch.float)
            w00_, w11_, w0110_ = torch.broadcast_tensors(w00_, w11_, w0110_)
            gW0 = torch.stack([w00_, -w0110_.sign() * (w0110_.abs() ** 0.5)], dim=-1)
            gW1 = torch.stack([w0110_.abs() ** 0.5, -w11_], dim=-1)
            gW = torch.stack([gW0, gW1], dim=-2)
            if symmetry == "diag":
                sigma = torch.tensor(
                    [
                        [rho[j] ** (-0.5), rho[j] ** 0.5],
                        [rho[j] ** 0.5, rho[j] ** (-0.5)],
                    ]
                )
            else:
                sigma = torch.tensor([[rho[j] ** (-0.5), rho[j] ** 0.5]])
            sigma = sigma.broadcast_to(gW.shape[:-2] + sigma.shape)
            r = np.linspace(0, rmax, Nr + 1)[1:]  # (Nr,)
            dr = steady_state(d, gW, sigma, r)  # (*, 2, Nr)

        if symmetry == "pre":
            l0, l1 = eigvals2x2(tr, det)

            D = tr**2 - 4 * det
            is_real = (D >= 0) & (l0.real > 0)
            is_cplx = D < 0

            l0real = l0.real.copy()
            l0real[D < 0] = 1
        else:
            # numerical must be True since symmetry == "diag"
            V = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
            a_, b_ = w00_.numpy(), (w0110_.sign() * (w0110_.abs() ** 0.5)).numpy()
            c_, d_ = (w0110_.abs() ** 0.5).numpy(), w11_.numpy()
            zeros = np.zeros_like(w00_.numpy())
            U = np.array(
                [
                    [a_, b_, zeros, zeros],
                    [zeros, zeros, c_, d_],
                ]
            )
            U = np.moveaxis(U, (-2, -1), (0, 1))
            Sinv = np.diag([rho[j], 1 / rho[j], 1 / rho[j], rho[j]])
            M = (np.eye(4) - V @ U) @ Sinv
            eigvals = np.linalg.eigvals(M)  # (*, 4)
            eigvals[eigvals.imag != 0] = 1
            l0real = eigvals.real.min(axis=-1)  # (*)

        if not numerical:
            z_num, z_dom = frho0(l0, l1, w11, 1 / rho[j], d=d), frho1(
                l0, l1, w11, 1 / rho[j], d=d
            )
            z = z_num / z_dom
            z[:, :, ~is_real] = np.nan
            z = z.real

            c = frho0(l0, l1, w11, 1 / rho[j]) / frho1(l0, l1, w11, 1 / rho[j])
            c[..., (l0.real <= 0) & (l0.imag == 0)] = np.nan
            r = find_root(d, l0, l1, c)

            if mode == "rel_diff":
                if d != 3:
                    raise NotImplementedError()
                rel_diff = np.full_like(r, np.nan)
                rel_diff[:, :, is_cplx] = (np.pi / (np.pi - np.angle(z_num)))[
                    :, :, is_cplx
                ]

            elif mode == "diff":
                if d in {1, 3}:
                    diff = find_root(d, l0, l1, c, n=2) - r
                    # diff = np.full_like(l1, np.nan)
                    # diff[is_cplx] = (np.pi / np.imag(np.sqrt(l1)))[is_cplx]
                    # diff = np.broadcast_to(diff, (2, 1, *diff.shape))
                else:
                    diff = find_root(d, l0, l1, c, n=2) - r

            elif mode == "min_diff":
                min_diff = find_root(d + 2, l0, l1, c) - r

            elif mode == "gain":
                ddet = dg * (
                    2 * det
                    + w11
                    + (rho[j] + 1 / rho[j] - tr - w11 / rho[j]) / rho[j]
                    - 2
                )
                dtr = dg * (tr - (rho[j] + 1 / rho[j]))
                tr_, det_ = tr + dtr, det + ddet
                l0_, l1_ = eigvals2x2(tr_, det_)
                D_ = tr_**2 - 4 * det_
                is_real_ = (D_ >= 0) & (l0_.real > 0)
                w11_ = w11 + dg * w11
                z_num_, z_dom_ = frho0(l0_, l1_, w11_, 1 / rho[j], d=d), frho1(
                    l0_, l1_, w11_, 1 / rho[j], d=d
                )
                z_ = z_num_ / z_dom_
                z_[:, :, ~is_real_] = np.nan
                z_ = z_.real

                c_ = frho0(l0_, l1_, w11_, 1 / rho[j]) / frho1(
                    l0_, l1_, w11_, 1 / rho[j]
                )
                c_[..., (l0.real <= 0) & (l0.imag == 0)] = np.nan
                r_ = find_root(d, l0_, l1_, c_)

                drdg = (r_ - r) / dg
                drdg = drdg / r
        else:
            dr = dr + bias * dr.min(keepdim=True, dim=-1).values.abs()  # (*, 2, Nr)
            mask = (dr.signbit().diff(dim=-1) != 0).long()  # (*, 2, Nr - 1)
            mid = torch.tensor(r[1:] + r[:-1], dtype=torch.float) / 2  # (Nr - 1,)
            idx = mask.argmax(dim=-1)  # (*, 2), index of first zero crossing
            r = torch.where(mask.any(dim=-1), mid[idx], torch.nan)  # (*, 2)
            r = r.movedim(-1, 0)[:, None]  # (2, 1, *)
            if mode == "diff":
                mask.scatter_(-1, idx[..., None], 0)
                idx = mask.argmax(dim=-1)  # (*, 2), index of second zero crossing
                r_ = torch.where(mask.any(dim=-1), mid[idx], torch.nan)  # (*, 2)
                diff = r_.movedim(-1, 0)[:, None] - r  # (2, 1, *)
            elif mode == "min_diff":
                ddr = dr.diff(dim=-1)  # (*, 2, Nr - 1)
                mask = (ddr.signbit().diff(dim=-1) != 0).long()  # (*, 2, Nr - 2)
                idx = mask.argmax(dim=-1)  # (*, 2), index of first critical point
                mid = (mid[1:] + mid[:-1]) / 2
                r_ = torch.where(mask.any(dim=-1), mid[idx], torch.nan)  # (*, 2)
                min_diff = r_.movedim(-1, 0)[:, None] - r  # (2, 1, *)
            elif mode == "gain":
                dr = steady_state(
                    d, gW * (1 + dg), sigma, np.linspace(0, rmax, Nr + 1)[1:]
                )  # (*, 2, Nr)
                mask = (dr.signbit().diff(dim=-1) != 0).long()  # (*, 2, Nr - 1)
                idx = mask.argmax(dim=-1)  # (*, 2), index of first zero crossing
                r_ = torch.where(mask.any(dim=-1), mid[idx], torch.nan)  # (*, 2)
                r_ = r_.movedim(-1, 0)[:, None]  # (2, 1, *)
                drdg = (r_ - r) / (r * dg)
            elif mode == "num":
                N_cross = mask.sum(dim=-1).movedim(-1, 0)[:, None]  # (2, 1, *)
            elif mode is not None:
                raise NotImplementedError()

        if symmetry == "pre":
            ax.contour(
                x, y, D, levels=[0], colors=GREY, linewidths=rcParams["grid.linewidth"]
            )

        if mode == "ratio":
            ratio_ = r[(1 - i), i] / r[0, 0]
            cs = ax.contourf(x, y, ratio_, levels=levels, norm=norm, cmap="bwr")
        elif mode == "diff_EI":
            diff_ = r[(1 - i), i] - r[0, 0]
            cs = ax.contourf(x, y, diff_, levels=levels, norm=norm, cmap="bwr")
            z = y - (rho[j] - 1 / rho[j]) * (x - (rho[j] - 1 / rho[j]))
            ax.contour(
                x, y, z, levels=[0], colors="black", linewidths=1, linestyles="--"
            )
            z = y - (rho[j] ** 2 - 1) * w00
            ax.contour(
                x, y, z, levels=[0], colors="black", linewidths=1, linestyles="--"
            )
        elif mode == "rel_diff":
            cs = ax.contourf(x, y, rel_diff[i, 0], levels=levels, norm=norm, cmap="bwr")
            ax.contourf(x, y, z[i, 0], levels=[0, 1], colors=["red"])
        elif mode == "diff":
            value = diff[i, 0]
            value[l0real < 0] = np.nan
            cs = ax.contourf(x, y, value, levels=levels, norm=norm, cmap="bwr")
            if shade:
                # ax.contourf(x, y, value, levels=shade[1:], colors="C2", alpha=0.25)
                # ax.contour(x, y, value, levels=[shade[0]], colors="C2", linewidths=1)
                ax.contourf(
                    x, y, value, levels=shade[1:], colors="none", hatches=["//"]
                )
                ax.contour(x, y, value, levels=shade[1:], colors="black", linewidths=1)
        elif mode == "min_diff":
            value = min_diff[i, 0]
            cs = ax.contourf(x, y, value, levels=levels, norm=norm, cmap="bwr")
            if shade:
                # ax.contourf(
                #     x, y, value, levels=shade[1:], colors="C2", alpha=0.25
                # )
                # ax.contour(
                #     x, y, value, levels=[shade[0]], colors="C2", linewidths=1
                # )
                ax.contourf(
                    x, y, value, levels=shade[1:], colors="none", hatches=["//"]
                )
                ax.contour(x, y, value, levels=shade[1:], colors="black", linewidths=1)
        elif mode == "gain":
            cs = ax.contourf(x, y, drdg[i, 0], levels=levels, norm=norm, cmap="bwr")
            lost_zero = np.array(~np.isnan(r) & np.isnan(r_)).astype(float)
            print(np.sum(lost_zero))
            ax.contourf(x, y, lost_zero[i, 0], levels=[0.5, 1.5], colors="grey")
        elif mode == "num":
            max_N_cross = N_cross[i, 0, l0real > 0].max()
            levels = np.r_[np.arange(-0.5, max_N_cross, 1), 1e8]
            print((N_cross[i, 0, l0real > 0] == 1).count_nonzero())
            cs = ax.contourf(
                x,
                y,
                N_cross[i, 0],
                levels=levels,
                colors=[f"C{i}" for i in range(len(levels) - 1)][::-1],
            )
        else:
            value = r[i, 0]
            cs = ax.contourf(x, y, value, levels=levels, norm=norm, cmap="bwr")
            if shade:
                # ax.contourf(x, y, value, levels=shade[1:], colors="C2", alpha=0.25)
                # ax.contour(x, y, value, levels=[shade[0]], colors="C2", linewidths=1)
                ax.contourf(
                    x, y, value, levels=shade[1:], colors="none", hatches=["//"]
                )
                ax.contour(x, y, value, levels=shade[1:], colors="black", linewidths=1)

        ax.contourf(x, y, l0real, levels=[-1e8, 0], colors="black")
        # ax.contourf(x, y, w11, levels=[0, 1e8], colors="grey")
        # ax.contourf(x, y, w0110, levels=[0, 1e8], colors="grey")
        # ax.set_xlim(
        #     np.min(tr), np.minimum(np.max(tr), rho[j] + 1 / rho[j] * (1 - w11))
        # )
        if vector_fields:
            shape = (10, 10)
            assert x.shape == y.shape
            skip = (x.shape[0] // shape[0], x.shape[1] // shape[1])
            x_, y_ = x[:: skip[0], :: skip[1]], y[:: skip[0], :: skip[1]]
            for f, (u, v) in vector_fields.items():
                u, v = np.array(u), np.array(v)
                u, v = np.broadcast_to(u, x.shape), np.broadcast_to(v, x.shape)
                u, v = u[:: skip[0], :: skip[1]], v[:: skip[0], :: skip[1]]
                ax.quiver(
                    x_, y_, u, v, color={"w00": "#DDCB76", "absw11": "#89CCED"}[f]
                )

        ticklabels = None
        if mode == "num":
            ticks = (levels[:-1] + levels[1:]) / 2
            ticklabels = [int(t) for t in ticks[:-1]] + [f"$\geq$ {int(ticks[-2])}"]
        elif spacing == "symlog":
            halfticks = np.arange(
                math.ceil(math.log10(linthresh)),
                math.ceil(math.log10(levels[-1])) + 1,
                1,
            )
            ticks = np.r_[-(10**halfticks), 0, 10**halfticks]
        elif spacing in {"log", "halflog"}:
            ticks = 10.0 ** np.arange(
                math.ceil(math.log10(levels[0])),
                math.floor(math.log10(levels[-1])) + 1,
                1,
            )
        elif spacing == "neghalflog":
            ticks = 10.0 ** np.arange(
                math.ceil(math.log10(-levels[-1])),
                math.floor(math.log10(-levels[0])) + 1,
                1,
            )
            ticks = -ticks[::-1]
        else:
            ticks = np.linspace(
                levels[0],
                levels[-1],
                1 + (N_levels - 1) // 2 ** math.ceil(math.log2((N_levels - 1) / 3)),
            )
        cbar = fig.colorbar(cs, ax=ax, format="%g")
        cbar.set_ticks(ticks=ticks, labels=ticklabels)
        cbar.set_label(cbar_labels[i])
    return fig


def plot_space_ori_phases(
    tr,
    det,
    rho=(1,),
    w00=1,
    coord="W",
    transpose=False,
    cell_type=None,
    axsize=(2.25, 2.25),
):
    r"""
    Note: here rho is defined as s1 / s0, as opposed to s0 / s1 as in the rest of the code.
    If coord == "I - W", then
        tr: trace of (I - W)\tilde{S}^{-1} = \rho * (1 - w00) + \rho^{-1} * (1 - w11)
        det: determinant of (I - W)\tilde{S}^{-1} = (1 - w00) * (1 - w11) - w01 * w10
    else:
        tr: \mathrm{tr}(W\tilde{S}^{-1}) + \rho - \rho^{-1} = \rho w00 - \rho^{-1} |w11| + \rho - \rho^{-1}
        det: \det(W) - \tr(W(I - \rho^{\pm1}\tilde{S}^{-1}))
    """
    if coord not in {"I-W", "W"}:
        raise ValueError(f"Invalid value for argument `coord`: {coord}")

    if cell_type is None:
        cell_type = [0, -1]
    elif cell_type not in {"E", "I"}:
        raise ValueError(f"Invalid value for argument `cell_type`: {cell_type}")
    else:
        cell_type = {"E": [0], "I": [-1]}[cell_type]

    xx, yy = tr, det

    if len(rho) == len(cell_type) == 1:
        fig = plt.figure(figsize=FIGSIZE)
        axes = fig.add_axes((RECT[0] * 1.1, RECT[1] * 0.95, RECT[2], RECT[3]))
        axes = np.array([[axes]])
    elif transpose:
        fig, axes = plt.subplots(
            len(rho),
            len(cell_type),
            figsize=(len(cell_type) * axsize[0], len(rho) * axsize[1]),
        )
        axes = axes.reshape(len(rho), len(cell_type)).T
    else:
        fig, axes = plt.subplots(
            len(cell_type),
            len(rho),
            figsize=(len(rho) * axsize[0], len(cell_type) * axsize[1]),
        )
        axes = axes.reshape(len(cell_type), len(rho))

    if coord == "I-W":
        fig.supxlabel(
            r"$\rho (1 - w_{EE}\kappa_{EE}) + \rho^{-1} (1 + |w_{II}|\kappa_{II})$"
        )
        fig.supylabel(
            r"$(1 - w_{EE}\kappa_{EE}) (1 + |w_{II}|\kappa_{II}) + |w_{EI}| w_{IE}\kappa_{EI}\kappa_{IE}$"
        )
    else:
        xlabel = r"$\rho \tilde{w}_{EE} - \rho^{-1} \tilde{w}_{II} + \rho - \rho^{-1}$"
        if axes.shape[1] == 1:
            axes[-1, 0].set_xlabel(xlabel)
        else:
            fig.supxlabel(xlabel)

        ylabel = r"$\tilde{w}_{EI}\tilde{w}_{IE} - \tilde{w}_{EE}\tilde{w}_{II}$" + "\n"
        ylabels = [
            ylabel + r"$+ (\rho^2 - 1)\tilde{w}_{EE}$",
            ylabel + r"$+ (1 - \rho^{-2})\tilde{w}_{II}$",
        ]
        for i, j in product(cell_type, range(len(rho))):
            ax = axes[i, j]
            if transpose or j == 0:
                ax.set_ylabel(ylabels[i], labelpad=-8)
            if transpose or i == 0:
                ax.set_title(r"$\tilde{w}_{EE} = %g, \rho = %g$" % (w00, rho[j]))

    # fig.suptitle(r"$w_{EE}\kappa_{EE} = %.2f$" % w00)
    # titles = [r"E $\to$ E", r"E $\to$ I"]

    for i, j in product(cell_type, range(len(rho))):
        ax = axes[i, j]
        if coord == "W":
            if i == 0:
                tr, det = 2 * rho[j] - xx, rho[j] ** 2 - xx * rho[j] + yy
            else:
                tr, det = 2 / rho[j] - xx, rho[j] ** (-2) - xx / rho[j] + yy

        l0, l1 = eigvals2x2(tr, det)

        D = tr**2 - 4 * det
        is_real = (D >= 0) & (l0.real > 0)

        l0real = l0.real.copy()
        l0real[D < 0] = 1

        w11 = 1 - (tr - (1 - w00) * rho[j]) * rho[j]
        w0110 = (1 - w00) * (1 - w11) - det
        # z = frho0(l0, l1, w11, 1 / rho[j]) / frho1(l0, l1, w11, 1 / rho[j])
        z = frho0(l0, l1, w11, 1 / rho[j])
        z[:, :, ~is_real] = np.nan
        z = z.real

        ax.contourf(xx, yy, D, levels=[-1e8, 0, 1e8], colors=["C0", "C2"])
        ax.contour(
            xx, yy, D, levels=[0], colors=GREY, linewidths=rcParams["grid.linewidth"]
        )
        ax.contour(
            xx,
            yy,
            z[i, 0],
            levels=[0],
            colors=GREY,
            linewidths=rcParams["grid.linewidth"],
        )
        # ax.contourf(xx, yy, z[i, 0], levels=[0, 1], colors=["C1"])
        ax.contourf(xx, yy, z[i, 0], levels=[-1e8, 0], colors=["C1"])
        ax.contourf(xx, yy, l0real, levels=[-1e8, 0], colors="black")
        ax.contour(xx, yy, w0110, levels=[0], colors="black", linewidths=1)
        # ax.contourf(xx, yy, w0110, levels=[0, 1e8], hatches=["xx", None], colors="none")
        ax.contourf(
            xx, yy, w0110, levels=[-1e8, 0], hatches=["xx", None], colors="none"
        )
        # ax.contour(xx, yy, w11, levels=[0], colors="black", linewidths=1)
        # ax.contourf(xx, yy, w11, levels=[-1, 0], hatches=[r"\\", None], colors="none")
        # ax.set_title(r"%s, $\rho = %.2f$" % (titles[i], rho[j]))
        ax.set_yticks([yy.min(), 0, yy.max()])
    return fig


def plot_mean_phases(
    x,
    y,
    w00=None,
    w11=None,
    coord="wee",
    axes="both",
    mode=None,
    levels=(-2, 3),
    linthresh=1,
    dg=1e-5,
    axsize=(2, 1.75),
):
    if coord not in {"wee", "tr"}:
        raise ValueError(f"Invalid value for argument `coord`: {coord}")

    if mode and mode not in {"value", "gain"}:
        raise ValueError(f"Invalid mode: {mode}")

    if w00 is not None and w00 < 0:
        raise ValueError("w00 must be non-negative")

    if w11 is not None and w11 > 0:
        raise ValueError("w11 must be non-positive")

    if coord == "wee":
        if w00 is not None:
            raise ValueError("w00 must be None if coord is 'wee'")

        w00, det = x, y
        if w11 is not None:
            tr = w00 + w11

    else:
        if (w00 is None) == (w11 is None):
            raise ValueError("Exactly one of `w00` and `w11` must be provided")

        tr, det = x, y
        if w11 is None:
            w11 = tr - w00
        else:
            w00 = tr - w11

    if axes not in {"both", "mean", "ori"}:
        raise ValueError(f"Invalid axes: {axes}")

    levels = np.logspace(*levels, 20)
    levels = np.r_[-levels[::-1], 0, levels]
    norm = colors.SymLogNorm(linthresh)
    halfticks = np.arange(
        math.ceil(math.log10(linthresh)),
        math.ceil(math.log10(levels[-1])),
        1 if levels[-1] < 500 else 2,
    )
    ticks = np.r_[-(10**halfticks), 0, 10**halfticks]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)

    if isinstance(w00, Number):
        ax.set_title(r"$w_{EE} = %.1f$" % w00)
    elif isinstance(w11, Number):
        ax.set_title(r"$|w_{II}| = %.1f$" % -w11)

    if mode is None:
        ax.contourf(x, y, w00 - det, levels=[-1e8, 0, 1e8], colors=["C0", "C3"])
        ax.contour(
            x,
            y,
            w00 - det,
            levels=[0],
            colors=GREY,
            linewidths=rcParams["grid.linewidth"],
        )
    elif mode == "value":
        d = 1 - tr + det
        cs = ax.contourf(x, y, (1 - w11) / d - 1, cmap="bwr", levels=levels, norm=norm)
        ax.contourf(x, y, d, levels=[-1e8, 0], colors="black")
        fig.colorbar(cs, ax=ax, ticks=ticks, label=r"$\langle r_E \rangle$")
    else:
        d = 1 - tr + det
        c = (1 - w11) / d - 1
        w11_, tr_, det_ = w11 * (1 + dg), tr * (1 + dg), det * (1 + 2 * dg)
        d_ = 1 - tr_ + det_
        c_ = (1 - w11_) / d_ - 1
        dcdg = (c_ - c) / dg

        cs = ax.contourf(x, y, dcdg, levels=levels, norm=norm, cmap="bwr")
        ax.contourf(x, y, d, levels=[-1e8, 0], colors="black")
        ax.contour(x, y, w00 - det, levels=[0], colors="black")
        if coord == "tr":
            ax.contour(x, y, y - x**2 / 4, levels=[0])
        # y0 = y
        # y0[x > 0] = np.nan
        # ax.contour(x, y, y0, levels=[0], colors="grey", linestyles="--")
        # ax.set_xlim(ax.get_xlim()[0], min(w00 + 1, ax.get_xlim()[1]))
        cbar_labels = [
            r"$\frac{d\langle r_E \rangle}{dg}$",
            r"$\frac{d\langle \tilde{r}_E \rangle}{dg}$",
        ]
        cbar_label = {
            "mean": cbar_labels[0],
            "ori": cbar_labels[1],
            "both": cbar_labels[0] + " or " + cbar_labels[1],
        }[axes]
        fig.colorbar(cs, ax=ax, ticks=ticks, label=cbar_label)

    mean_labels = [
        r"$w_{EE}$" if coord == "wee" else r"$w_{EE} - |w_{II}|$",
        r"$|w_{EI}|w_{IE} - w_{EE}|w_{II}|$",
    ]
    ori_labels = [
        r"$\tilde{w}_{EE}$" if coord == "wee" else r"$\tilde{w}_{EE} - \tilde{w}_{II}$",
        r"$\tilde{w}_{EI}\tilde{w}_{IE} - \tilde{w}_{EE}\tilde{w}_{II}$",
    ]
    if axes == "ori":
        ax.set_xlabel(ori_labels[0])
        ax.set_ylabel(ori_labels[1])
    elif axes == "mean":
        ax.set_xlabel(mean_labels[0])
        ax.set_ylabel(mean_labels[1])
    else:
        ax.set_xlabel(mean_labels[0] + " or " + ori_labels[0])
        ax.set_ylabel(mean_labels[1] + "\nor " + ori_labels[1])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot",
        type=str,
        choices=["space", "space_2", "space_zero", "space_ori", "mean"],
    )
    parser.add_argument("-x", type=float, nargs=2, default=(-5, 2.5))
    parser.add_argument("-y", type=float, nargs=2, default=(-5, 5))
    parser.add_argument("-d", type=int, default=3)
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--rho", type=float, nargs="+")
    parser.add_argument("--w00", type=float)
    parser.add_argument("--w11", type=float)
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=[
            "ratio",
            "diff_EI",
            "rel_diff",
            "diff",
            "min_diff",
            "gain",
            "num",
            "decay",
            "value",
        ],
    )
    parser.add_argument("--bias", type=float)
    parser.add_argument(
        "--vector-fields",
        "--vf",
        dest="vector_fields",
        type=str,
        nargs="*",
        choices=["w00", "absw11"],
        default=[],
    )
    parser.add_argument("--axes", "-a", type=str, choices=["mean", "ori", "both"])
    parser.add_argument("--levels", "-l", type=float, nargs=2)
    parser.add_argument("--N-levels", "-n", type=int)
    parser.add_argument("--shade", type=float, nargs=3)
    parser.add_argument("--linthresh", "-t", type=float)
    parser.add_argument(
        "--spacing",
        "-s",
        type=str,
        choices=["log", "halflog", "neghalflog", "symlog", "linear", "halflinear"],
    )
    parser.add_argument("--coord", type=str, choices=["I-W", "W", "wee", "tr"])
    parser.add_argument(
        "--symmetry", "--sym", dest="symmetry", type=str, choices=["pre", "diag"]
    )
    parser.add_argument("--cell-type", "-c", type=str, choices=["E", "I", "EI"])
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--Nr", type=int)
    parser.add_argument("--rmax", type=float)
    parser.add_argument("--dg", type=float)
    parser.add_argument("--transpose", "-T", action="store_true")
    parser.add_argument("--axsize", type=float, nargs=2)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    x = np.linspace(*args.x, args.N)
    y = np.linspace(*args.y, args.N)
    x, y = np.meshgrid(x, y)

    kwargs = {
        k: getattr(args, k)
        for k in [
            "rho",
            "w00",
            "w11",
            "mode",
            "bias",
            "shade",
            "vector_fields",
            "levels",
            "N_levels",
            "linthresh",
            "spacing",
            "symmetry",
            "Nr",
            "rmax",
            "dg",
            "coord",
            "cell_type",
            "axes",
            "axsize",
        ]
        if getattr(args, k)
    }

    set_rcParams()
    if args.plot == "space":
        fig = plot_space_phases(
            x, y, numerical=args.numerical, transpose=args.transpose, **kwargs
        )
    elif args.plot == "space_2":
        fig = plot_space_phases_2(x, y, **kwargs)
    elif args.plot == "space_zero":
        fig = plot_space_zero_loc(x, y, d=args.d, numerical=args.numerical, **kwargs)
    elif args.plot == "space_ori":
        fig = plot_space_ori_phases(x, y, transpose=args.transpose, **kwargs)
    else:
        fig = plot_mean_phases(x, y, **kwargs)

    # fig.tight_layout()

    if args.out:
        fig.savefig(
            args.out,
            metadata={"Subject": " ".join(sys.argv[1:])},
            # bbox_inches="tight",
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
