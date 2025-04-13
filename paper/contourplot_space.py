import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import torch
import matplotlib.pyplot as plt

from niarb import nn
from niarb.nn.modules.frame import ParameterFrame
from niarb.zero_crossing import find_root, find_n_crossings
from niarb.tensors import periodic
from mpl_config import GREY, GRID_WIDTH, get_sizes, get_cbar_configs, set_rcParams


def sample_W(x, y, w00=None, rho=1.0, scale=5.0, ori=False):
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape

    if w00 is None:
        if ori:
            # condition for w11 < 1 to gaurantee stability
            min_w00 = x / rho - 1
        else:
            # condition for w11 < 0, also we want w00 > 1
            min_w00 = np.maximum((rho * x - rho**2 + 1) / rho**2, 1)
            # condition for w01*w10 < 0
            mask = y < x**2 / 4
            min_w00[mask] = np.maximum(
                min_w00, ((x + np.sqrt(x**2 - 4 * y)) / (2 * rho))
            )[mask]
        w00 = min_w00 + scale * np.abs(np.random.randn(*shape))
    else:
        w00 = np.full_like(x, w00)

    w11 = rho * x - rho**2 * w00 - rho**2 + 1
    w0110 = -((rho * w00) ** 2) + rho * w00 * x - y

    # if not ori:
    # assert (w11 <= 0).all(), (w11[w11 > 0], x[w11 > 0], y[w11 > 0])
    # assert (w0110 <= 0).all(), (w0110[w0110 > 0], x[w0110 > 0], y[w0110 > 0])
    np.testing.assert_allclose(rho * w00 + w11 / rho + rho - 1 / rho, x, atol=1e-7)
    np.testing.assert_allclose(w00 * w11 - w0110 + (rho**2 - 1) * w00, y, atol=1e-7)

    w10 = scale * np.abs(np.random.randn(*shape))
    w01 = w0110 / w10

    W = np.stack([[w00, w01], [w10, w11]])  # (2, 2, *shape)
    W = np.moveaxis(W, (0, 1), (-2, -1))  # (*shape, 2, 2)
    assert W.shape == (*shape, 2, 2)

    return W


def eigvals2x2(W, rho, as_complex=False):
    S = np.diag([1 / rho, rho])
    M = (np.eye(2) - W) @ np.linalg.inv(S)
    tr, det = np.trace(M, axis1=-2, axis2=-1), np.linalg.det(M)
    D = tr**2 - 4 * det
    if (D < 0).any() or as_complex:
        D = D.astype(complex)
    l0 = 0.5 * (tr - np.sqrt(D))
    l1 = 0.5 * (tr + np.sqrt(D))
    return l0, l1


def ratio_coef(d, l0, l1, rho, w11):
    c00 = ((1 - l0 / rho) * (1 - l0 * rho - w11)) / (
        (1 - l1 / rho) * (1 - l1 * rho - w11)
    )
    c10 = (1 - l0 * rho) / (1 - l1 * rho)
    if d < 2:
        c00 = c00 * np.sqrt(l1 / l0)
        c10 = c10 * np.sqrt(l1 / l0)
    return c00, c10


def response(
    d: int,
    W: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    tau_i: float,
    ori: bool,
    stability_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        d: int
        W: array-like with shape (*, 2, 2)
        sigma: array-like with shape (*, 1 | 2, 1 | 2)
        r: array-like with shape (N,)
    """
    W = torch.tensor(W, dtype=torch.float)
    sigma = torch.tensor(sigma, dtype=torch.float)
    r = torch.tensor(r, dtype=torch.float)

    if W.ndim < 2 or W.shape[-2:] != (2, 2):
        raise ValueError("W must have at least 2 dimensions and shape (*, 2, 2)")

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

    shape = torch.broadcast_shapes(W.shape[:-2], sigma.shape[:-2])
    W = W.broadcast_to(shape + W.shape[-2:])
    sigma = sigma.broadcast_to(shape + sigma.shape[-2:])
    kappa = torch.rand((*shape, 2, 2)) - 0.5  # [-0.5, 0.5)
    if ori:
        kappa = kappa.abs()
        kappa[..., :, 0] = kappa[..., :, 0] * W[..., :, 0].sign()
        kappa[..., :, 1] = -kappa[..., :, 1] * W[..., :, 1].sign()
        W = W / kappa

    if (W[..., :, 0] < 0).any():
        raise ValueError("W_EE and W_IE must be non-negative.")

    # if (W[..., :, 1] > 0).any():
    #     raise ValueError("W_EI and W_II must be non-positive.")

    W[..., :, 1] = -W[..., :, 1]  # could also just take abs()

    model = nn.V1(
        ["cell_type", "space"] + (["ori"] if ori else []),
        cell_types=["PYR", "PV"],
        init_stable=False,
        sigma_symmetry=symmetry,
        batch_shape=shape,
        tau=[1.0, tau_i],
    )
    model.load_state_dict({"gW": W, "sigma": sigma, "kappa": kappa}, strict=False)

    is_stable = model.spectral_summary(kind="J").abscissa < 0
    if stability_only:
        return is_stable.numpy(), None

    space = torch.tensor(np.r_[0, r], dtype=torch.float)  # (N + 1,)
    space = torch.stack(
        [space, *([torch.zeros_like(space)] * (d - 1))], dim=-1
    )  # (N + 1, d)
    dh = torch.zeros((2, space.shape[0]))  # (2, N + 1,)
    dh[0, 0] = 1.0

    # Note that dV does not matter since we are only interested in the response shape
    x = ParameterFrame(
        {
            "cell_type": torch.tensor([[0], [1]]),  # (2, 1)
            "space": space[None, ...],  # (1, N + 1, d)
            "dV": torch.tensor([[1.0]]),  # (1, 1)
            "dh": dh,  # (2, N + 1)
        },
        ndim=2,
    )  # (2, N + 1)
    if ori:
        x = x.unsqueeze(-1)  # (2, N + 1, 1)
        x["ori"] = periodic.tensor(
            [[[[0.0], [90.0]]]], extents=[(-90.0, 90.0)]
        )  # (1, 1, 2, 1)
        x["dh"] = torch.stack([dh, torch.zeros_like(dh)], dim=-1)  # (2, N + 1, 2)

    # I found out that my code for computing K0 on single-precision tensors can
    # sometimes return NaN values on valid inputs. But the double-precision routines
    # seem to work fine. So we convert everything to double-precision.
    model.double()
    x = x.double()

    with torch.inference_mode():
        out = model(
            x, ndim=x.ndim, check_circulant=False, to_dataframe=False
        )  # (*, 2, N + 1) or (*, 2, N + 1, 2)
    dr = out["dr"]  # (*, 2, N + 1) or (*, 2, N + 1, 2)
    if ori:
        dr = dr[..., 1] - dr[..., 0]  # (*, 2, N + 1)

    return is_stable.numpy(), dr[..., 1:].numpy()  # (*,), (*, 2, N)


def fit_decay(d, r, y):
    r"""Fit the curve cr^{-(d-1)/2}e^(-r/σ) to the data.

    Args:
        d: int
        r: array-like with shape (*, N)
        y: array-like with shape (*, N)

    Returns:
        σ: np.ndarray with shape (*)

    """
    r, y = torch.as_tensor(r), torch.as_tensor(y)
    r, y = torch.broadcast_tensors(r, y)
    A = torch.stack([r, torch.ones_like(r)], dim=-1)  # (*, N, 2)
    B = torch.log(y * r ** ((d - 1) / 2)).unsqueeze(-1)  # (*, N, 1)

    # If we let X = [-1 / σ, c], then AX = B.
    X = torch.linalg.lstsq(A, B).solution  # (*, 2, 1)
    return -1 / X[..., 0, 0].numpy()  # (*,)


def plot_E_phase_diagram(x, y, xr1, xr2, ax):
    z = np.zeros_like(xr1, dtype=np.long)  # no crossings, green, "C2"
    z[~np.isnan(xr1)] = 1  # has at least 1 crossing, orange, "C1"
    z[~np.isnan(xr2)] = 2  # has at least 2 crossings, blue, "C0"

    ax.contourf(x, y, z, levels=[-0.5, 0.5, 1.5, 2.5], colors=["C2", "C1", "C0"])


def plot_EI_phase_diagram(x, y, xrE1, xrE2, xrI1, xrI2, ax):
    zE = np.zeros_like(xrE1, dtype=np.long)
    zE[~np.isnan(xrE1)] = 1  # has at least 1 E crossing
    zE[~np.isnan(xrE2)] = 2  # has at least 2 E crossings
    zI = np.zeros_like(xrI1, dtype=np.long)
    zI[~np.isnan(xrI1)] = 1  # has at least 1 E crossing
    zI[~np.isnan(xrI2)] = 2  # has at least 2 E crossings
    z = np.full_like(zE, -1, dtype=np.long)
    z[(zE == 0) & (zI == 0)] = 0  # green, "C2"
    z[(zE == 1) & (zI == 0)] = 1  # orange, "C1"
    z[(zE == 1) & (zI == 1)] = 2  # red, "C3"
    z[(zE == 2) & (zI == 2)] = 3  # blue, "C0"

    ax.contourf(
        x, y, z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=["C2", "C1", "C3", "C0"]
    )


def plot_generic(x, y, z, levels, norm, ticks, clabel, shade, fig, ax):
    cs = ax.contourf(x, y, z, cmap="bwr", levels=levels, norm=norm)
    cbar = fig.colorbar(cs, ax=ax, format="%g")
    if ticks is not None:
        cbar.set_ticks(ticks=ticks)
    cbar.set_label(clabel)

    if shade:
        ax.contourf(x, y, z, levels=shade, colors="none", hatches=["//"])
        ax.contour(x, y, z, levels=shade, colors="black", linewidths=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices={"E", "EI", "r0", "r1", "rmin", "rEI", "decay", "dr0dg"}
    )
    parser.add_argument("--numerical", "--num", dest="numerical", action="store_true")
    parser.add_argument("-x", type=float, nargs=2, default=(-5, 2.5))
    parser.add_argument("-y", type=float, nargs=2, default=(-5, 5))
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--ori", action="store_true")
    parser.add_argument("--w00", type=float)
    parser.add_argument("--rho", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dg", type=float, default=1e-5)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--tau-i", type=float, default=0.01)
    parser.add_argument("--rmax", type=float, default=3000.0)
    parser.add_argument("--decay-rmin", type=float, default=1500.0)
    parser.add_argument("--rN", type=int, default=3000)
    parser.add_argument("--shade", type=float, nargs=2)
    parser.add_argument(
        "--spacing",
        "-s",
        choices=["log", "halflog", "neghalflog", "symlog", "linear", "halflinear"],
        default="log",
    )
    parser.add_argument("--levels", "-l", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--N-levels", "-n", type=int, default=9)
    parser.add_argument("--linthresh", "-t", type=float, default=1.0)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rho0 = args.rho[0]
    x = np.linspace(*args.x, args.N)
    y = np.linspace(*args.y, args.N)
    x, y = np.meshgrid(x, y)
    s0, s1 = args.s0, rho0 * args.s0

    # get the 2x2 W matrix. If w00 is None then W is randomly sampled.
    W = sample_W(x, y, w00=args.w00, rho=rho0, ori=args.ori)

    # compute various zero crossings
    if args.numerical:
        # first compute perturbation response
        sigma = np.array([[s0, s1]])
        r = np.linspace(0, args.rmax, args.rN + 1)[1:]
        is_stable, _dr = response(args.d, W, sigma, r, args.tau_i, args.ori)
        drE, drI = _dr[..., 0, :], _dr[..., 1, :]

        xrE1, xrE2 = find_n_crossings(r, drE, n=2)
        xrEm = find_n_crossings(r, np.diff(drE, axis=-1))[0]
        xrI1, xrI2 = find_n_crossings(r, drI, n=2)
        s = np.sqrt(np.prod(sigma))
        xrE1, xrE2, xrEm, xrI1, xrI2 = xrE1 / s, xrE2 / s, xrEm / s, xrI1 / s, xrI2 / s
    else:
        l0, l1 = eigvals2x2(W, rho0)
        cE, cI = ratio_coef(args.d, l0, l1, rho0, W[..., 1, 1])
        is_stable = (l0.real > 0) | (l0.imag != 0)
        xrE1 = find_root(args.d, l0, l1, cE)
        xrE2 = find_root(args.d, l0, l1, cE, n=2)
        xrEm = find_root(args.d + 2, l0, l1, cE)
        xrI1 = find_root(args.d, l0, l1, cI)
        xrI2 = find_root(args.d, l0, l1, cI, n=2)

    # set some plotting defaults
    set_rcParams()
    lmargin, rmargin, cbar = 1.0, 1.0, False
    if args.ori and args.rho != [1.0]:
        lmargin = 1.15
    if args.mode not in {"E", "EI"}:
        cbar = True
    figsize, rect = get_sizes(lmargin, rmargin, 1, 1, cbar=cbar)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    # plot the phase diagram
    if args.mode == "E":
        plot_E_phase_diagram(x, y, xrE1, xrE2, ax)
    elif args.mode == "EI":
        plot_EI_phase_diagram(x, y, xrE1, xrE2, xrI1, xrI2, ax)
    else:
        levels, norm, ticks = get_cbar_configs(
            args.spacing, args.levels, args.N_levels, args.linthresh
        )

        if args.mode == "r0":
            z = xrE1
            clabel = r"$\frac{r_0}{\sqrt{\sigma_E \sigma_I}}$"
        elif args.mode == "r1":
            z = xrE2 - xrE1
            clabel = r"$\frac{r_1 - r_0}{\sqrt{\sigma_E \sigma_I}}$"
        elif args.mode == "rmin":
            z = xrEm - xrE1
            clabel = r"$\frac{r^\mathrm{min}_0 - r_0}{\sqrt{\sigma_E \sigma_I}}$"
        elif args.mode == "rEI":
            z = xrI1 - xrE1
            clabel = (
                r"$\frac{r^\mathrm{I}_0 - r^\mathrm{E}_0}{\sqrt{\sigma_E \sigma_I}}$"
            )
        elif args.mode == "decay":
            if args.numerical:
                mask = r > args.decay_rmin
                z = fit_decay(args.d, r[mask], np.abs(drE[..., mask]))
                z = z / np.sqrt(np.prod(sigma))
            else:
                z = 1 / (l0**0.5).real
            clabel = r"$\frac{\sigma_\infty}{\sqrt{\sigma_E \sigma_I}}$"
            levels, norm, ticks = get_cbar_configs(
                "log", [np.log10(z[z > 0].min()), 1], 50, 1
            )
        elif args.mode == "dr0dg":
            gW = W * (1 + args.dg)
            if args.numerical:
                gdr = response(args.d, gW, sigma, r, args.tau_i, args.ori)[1]
                gxrE1 = find_n_crossings(r, gdr[..., 0, :])[0] / s
            else:
                gl0, gl1 = eigvals2x2(gW, rho0)
                gcE, _ = ratio_coef(args.d, gl0, gl1, rho0, gW[..., 1, 1])
                gxrE1 = find_root(args.d, gl0, gl1, gcE)
            z = (gxrE1 - xrE1) / (xrE1 * args.dg)
            clabel = r"$\frac{1}{r_0}\frac{dr_0}{dg}$"

        plot_generic(x, y, z, levels, norm, ticks, clabel, args.shade, fig, ax)

    # plot anti-like-to-like E->I->E region
    if args.ori:
        # note that sign(kappa_EI * kappa_IE) = -sign(w0110)
        w0110 = W[..., 0, 1] * W[..., 1, 0]
        ax.contour(x, y, w0110, levels=[0], colors="black", linewidths=1)
        ax.contourf(x, y, w0110, levels=[-1e8, 0], colors=["none"], hatches=["xx"])

    # plot analytic boundaries for EI plots
    if args.mode in {"EI", "rEI"}:
        # boundary for plots where rho < 1
        z = y - (rho0 - 1 / rho0) * (x - (rho0 - 1 / rho0))
        ax.contour(x, y, z, levels=[0], colors="black", linewidths=1, linestyles="--")
        # boundary for plots where rho > 1
        z = y - (rho0**2 - 1) * W[..., 0, 0]
        ax.contour(x, y, z, levels=[0], colors="black", linewidths=1, linestyles="--")

    # plot instability region
    alpha = 1 / len(args.rho)
    ax.contourf(x, y, is_stable, levels=[-0.5, 0.5], colors=["black"], alpha=alpha)
    if len(args.rho) > 1:
        for rhoi in args.rho[1:]:
            s0, s1 = args.s0, rhoi * args.s0
            W = sample_W(x, y, w00=args.w00, rho=rhoi, ori=args.ori)
            if args.numerical:
                sigma = np.array([[s0, s1]])
                r = np.linspace(0, args.rmax, args.rN + 1)[1:]
                is_stable, _ = response(
                    args.d, W, sigma, r, args.tau_i, args.ori, stability_only=True
                )
            else:
                l0, l1 = eigvals2x2(W, rhoi)
                cE, cI = ratio_coef(args.d, l0, l1, rhoi, W[..., 1, 1])
                is_stable = (l0.real > 0) | (l0.imag != 0)
        ax.contourf(x, y, is_stable, levels=[-0.5, 0.5], colors=["black"], alpha=alpha)

    # plot analytic phase boundaries
    ylim = ax.get_ylim()
    _x = np.linspace(*ax.get_xlim())
    ax.plot(_x, _x**2 / 4, color=GREY, linewidth=GRID_WIDTH)
    _x = np.linspace(ax.get_xlim()[0], 0)
    ax.plot(_x, np.zeros_like(_x), color=GREY, linewidth=GRID_WIDTH)
    ax.set_ylim(*ylim)

    # plot region where wII > 0 or wEI > 0
    if not args.ori:
        # make zorder a big number so that it is plotted on top of everything
        ax.contourf(x, y, W[..., 0, 1], levels=[0, 1e8], colors=["purple"], zorder=100)
        ax.contourf(x, y, W[..., 1, 1], levels=[0, 1e8], colors=["grey"], zorder=101)

    # nicer looking y-axis
    ax.set_yticks([ylim[0], sum(ylim) / 2, ylim[1]])
    ax.yaxis.set_major_formatter("{x:g}")

    # add labels
    if args.ori:
        if args.rho == [1.0]:
            ax.set_xlabel(r"$\tilde{w}_{EE} - \tilde{w}_{II}$")
            ax.set_ylabel(
                r"$\tilde{w}_{EI}\tilde{w}_{IE} - \tilde{w}_{EE}\tilde{w}_{II}$"
            )
        else:
            ax.set_xlabel(
                r"$\rho \tilde{w}_{EE} - \rho^{-1} \tilde{w}_{II} + \rho - \rho^{-1}$",
                labelpad=0,
            )
            ax.set_ylabel(
                (
                    r"$\tilde{w}_{EI}\tilde{w}_{IE} - \tilde{w}_{EE}\tilde{w}_{II}$"
                    "\n"
                    r"$+ (\rho^2 - 1)\tilde{w}_{EE}$"
                ),
                labelpad=0,
            )
    else:
        if args.rho == [1.0]:
            ax.set_xlabel(r"$w_{EE} - |w_{II}|$")
            ax.set_ylabel(r"$\mathrm{det}(\mathbf{W})$")
        else:
            ax.set_xlabel(
                r"$\rho w_{EE} - \rho^{-1} |w_{II}| + \rho - \rho^{-1}$",
                labelpad=0,
            )
            ax.set_ylabel(
                r"$\mathrm{det}(\mathbf{W}) + (\rho^2 - 1)w_{EE}$",
                labelpad=0,
            )

    # set title
    if args.w00 and args.ori:
        ax.set_title(r"$\tilde{w}_{EE} = %g, \rho = %g$" % (args.w00, rho0))
    elif args.w00:
        ax.set_title(r"$w_{EE} = %g, \rho = %g$" % (args.w00, rho0))
    else:
        ax.set_title(r"$\rho = %g$" % rho0)

    # save figure
    if args.out:
        fig.savefig(args.out, metadata={"Subject": " ".join(["python"] + sys.argv)})

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
