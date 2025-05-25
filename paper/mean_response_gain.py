import argparse
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from niarb import nn, neurons
from mpl_config import FIGSIZE, RECT, set_rcParams


def simulate(
    gain,
    w00,
    tr,
    det,
    sigma=(125, 90, 85, 110),
    kappa=(0.5, -0.25, -0.25, 0.25),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    tau_i=0.5,
    rtol=1e-5,
    maxiter=100,
):
    if w00 < 0:
        raise ValueError("w00 must be non-negative")

    w11 = tr - w00
    if w11 > 0:
        raise ValueError("w11 must be non-positive")

    w01w10 = w00 * w11 - det
    if w01w10 > 0:
        raise ValueError("w01 * w10 must be non-positive.")

    variables = ["cell_type"]
    if N_space:
        variables.append("space")
    if N_ori:
        variables.append("ori")
    if N_osi:
        variables.append("osi")
    d = len(N_space)

    w01, w10 = -np.sqrt(-w01w10), np.sqrt(-w01w10)
    state_dict = {
        "sigma": torch.tensor(sigma).reshape(2, 2),
        "kappa": torch.tensor(kappa).reshape(2, 2),
        "gW": gain * torch.tensor([[w00, -w01], [w10, -w11]]),
    }

    model = nn.V1(
        variables,
        cell_types=["PYR", "PV"],
        sigma_optim=False,
        kappa_optim=False,
        tau=[1.0, tau_i],
        mode="numerical",
        simulation_kwargs={"dx_rtol": rtol, "options": {"max_num_steps": maxiter}},
    )
    x = neurons.as_grid(
        2,
        N_space=N_space,
        N_ori=N_ori,
        N_osi=N_osi,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )
    idx = (0,) * x.ndim
    if "osi" in variables:
        idx = idx[:-1] + (-1,)
    x["dh"] = torch.zeros(x.shape)
    x["dh"][idx] = dh

    model.double()
    x = x.double()

    with torch.inference_mode():
        model.load_state_dict(state_dict, strict=False)
        print(model.spectral_summary(kind="J"))
        return model(x, ndim=x.ndim, to_dataframe="pandas")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("w00", type=float)
    parser.add_argument("tr", type=float)
    parser.add_argument("det", type=float)
    parser.add_argument("--N-space", type=int, nargs="*", default=())
    parser.add_argument("--N-ori", type=int, default=0)
    parser.add_argument("--N-osi", type=int, default=0)
    parser.add_argument("--N-gain-numerics", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tau-i", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    w00, tr, det = args.w00, args.tr, args.det
    g = np.linspace(0, 2, 100)
    g_numerics = np.linspace(0, 2, args.N_gain_numerics)
    y = (g * w00 - g**2 * det) / (1 - g * tr + g**2 * det)
    y_numerics = []
    for gi in g_numerics:
        outi = simulate(
            gi,
            w00,
            tr,
            det,
            N_space=args.N_space,
            N_ori=args.N_ori,
            N_osi=args.N_osi,
            dh=args.scale,
            tau_i=args.tau_i,
            rtol=args.rtol,
            maxiter=args.maxiter,
        )
        outi["dr"] = outi["dr"] / args.scale
        E_sum = outi.query("cell_type == 'PYR' and dh == 0")["dr"].sum()
        y_numerics.append(E_sum)

    set_rcParams()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    ax.plot(g, y, label="Theory")
    ax.plot(g_numerics, y_numerics, color="C0", linestyle="--", label="Simulation")
    ax.axvline(1, color="black", linestyle="--")
    ax.axhline(0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"])
    ax.set_ylabel(r"$N_E \langle r_E \rangle$")
    ax.set_xlabel("Gain")
    ax.set_xticks([0, 1, 2])
    fig.legend()
    fig.tight_layout()
    if args.out:
        plt.savefig(
            args.out, bbox_inches="tight", metadata={"Subject": " ".join(sys.argv)}
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
