import argparse
from pathlib import Path
from itertools import product

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from niarb import neurons, nn, perturbation, viz, io
from niarb.cell_type import CellType


def response(variables, kappa_x, state_dict, mode, N_ori=360):
    model = nn.V1(
        variables, cell_types=(CellType.PYR, CellType.PV), kappa_x=kappa_x, mode=mode
    )
    model.load_state_dict(state_dict, strict=False)

    x = neurons.as_grid(n=(model.n if "cell_type" in variables else 0), N_ori=N_ori)
    ndim = x.ndim

    x = x.unsqueeze(0)  # (1, [2,] N_ori)
    dh = torch.eye(x.shape[-1])  # (N_ori, N_ori)
    if mode == "analytical":
        # no need to calculate all perturbations
        dh = dh[:1]  # (1, N_ori)
    if "cell_type" in variables:
        dh = dh.unsqueeze(1)  # (1 | N_ori, 1, N_ori)
        dh = torch.cat([dh, torch.zeros_like(dh)], dim=1)  # (1 | N_ori, 2, N_ori)
    x["dh"] = dh  # (1 | N_ori, [2,] N_ori)
    x["rel_ori"] = x.apply(
        perturbation.abs_relative_ori, dim=range(1, x.ndim), keepdim=True
    )

    with torch.no_grad():
        return model(x, ndim=ndim, to_dataframe="pandas")


def plot_comparison(path):
    variables = ["cell_type", "ori"]
    loss = io.iterdir(path, pattern="*.pt", indices=0, stem=True)
    print(f"{loss=}")
    state_dict = torch.load(path / f"{loss}.pt", map_location="cpu", weights_only=True)
    kappa_x = np.arange(6) / 10

    out, expected = {}, {}
    for kappa_xi in kappa_x:
        out[kappa_xi] = response(variables, kappa_xi, state_dict, "analytical")
        expected[kappa_xi] = response(variables, kappa_xi, state_dict, "matrix")

    out, expected = pd.concat(out), pd.concat(expected)
    out = out.reset_index(0, names="kappa").reset_index(drop=True)
    expected = expected.reset_index(0, names="kappa").reset_index(drop=True)
    df = pd.concat({"theory": out, "simulation": expected})
    df = df.reset_index(0, names="method").reset_index(drop=True)

    bins = np.arange(0, 90.1, 2)
    df["rel_ori"] = pd.cut(df["rel_ori"], bins=bins, right=False)
    df = df.query("dh == 0").copy()

    g = viz.figplot(
        df,
        "relplot",
        kind="line",
        x="rel_ori",
        y="dr",
        hue="method",
        col="kappa",
        row="cell_type",
        errorbar="se",
        height=2,
        aspect=0.8,
        grid="yzero",
    )
    return g


def plot_scaling(path, scale_g, cell_type, n):
    """
    If scale_g is True, then g = (0.5 + 2κ_g)(1 + 2κ_g/(0.5 + 2κ_g)cos(θ)) = 0.5 + 2κ_g(1 + cos(θ))
    Otherwise, g = 1 + 2κ_gcos(θ)
    """
    variables = ["cell_type", "ori"]
    losses = io.iterdir(path, pattern="*.pt", indices=range(n), stem=True)
    kappa_x = np.linspace(0, 0.25 if scale_g else 0.5, 40)

    df = {}
    for loss, kappa_xi in tqdm(list(product(losses, kappa_x))):
        state_dict = torch.load(
            path / f"{loss}.pt", map_location="cpu", weights_only=True
        )
        if scale_g:
            state_dict["gW"] = state_dict["gW"] * (0.5 + 2 * kappa_xi)
            kappa_xi = kappa_xi / (0.5 + 2 * kappa_xi)
        df[(loss, kappa_xi)] = response(
            variables, kappa_xi, state_dict, "analytical", N_ori=180
        )

    df = pd.concat(df, names=["loss", "kappa"]).reset_index([0, 1]).query("dh == 0")
    df["rel_ori"] = pd.cut(
        df["rel_ori"], bins=[0.0, 1.1, 89.9, 90.0], labels=["iso", "other", "ortho"]
    )
    df = df.pivot_table(
        index=["loss", "kappa", "cell_type"], columns="rel_ori", values="dr"
    ).reset_index()
    df["dr_kappa"] = df["iso"] - df["ortho"]
    if not cell_type:
        df = df.pivot(
            index=["loss", "kappa"], columns="cell_type", values="dr_kappa"
        ).reset_index()
        df["dr_kappa"] = 0.85 * df["PYR"] + 0.15 * df["PV"]

    print(df)
    g = viz.figplot(
        df,
        "relplot",
        kind="line",
        x="kappa",
        y="dr_kappa",
        hue=("cell_type" if cell_type else None),
        col="loss",
        col_wrap=10,
        errorbar="se",
        height=1.5,
        aspect=1.2,
        ylim=(-0.01, 0.01),
        grid="yzero",
        facet_kws={"sharey": False},
    )
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--plot", "-p", choices={"comparison", "scaling"})
    parser.add_argument("--scale-g", "-g", action="store_true")
    parser.add_argument("--cell-type", "-c", action="store_true")
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()

    if args.plot == "comparison":
        g = plot_comparison(args.path)
    else:
        g = plot_scaling(args.path, args.scale_g, args.cell_type, args.n)
    plt.show()


if __name__ == "__main__":
    main()
