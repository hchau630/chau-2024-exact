import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def area(r, s):
    case0 = np.pi * r**2
    case1 = np.pi * r**2 + 4 * s * np.sqrt(r**2 - s**2) - 4 * r**2 * np.arccos(s / r)
    out = np.where(r < s, case0, case1)
    out[r > s * np.sqrt(2)] = (2 * s) ** 2
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("-s", type=float, default=np.inf)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cells = pd.concat(
        {
            f.stem: pd.DataFrame(
                scipy.io.loadmat(f)["spaceXYZ"], columns=["x", "y", "z"]
            )
            for f in args.dir.glob("postSyn_*.mat")
        }
    ).reset_index(0, names=["filename"])
    spatial_e = pd.concat(
        {
            f.stem: pd.DataFrame(
                scipy.io.loadmat(f)["spaceXYZ"], columns=["x", "y", "z"]
            )
            for f in args.dir.glob("spatialEX_*.mat")
        }
    ).reset_index(0, names=["filename"])
    spatial_i = pd.concat(
        {
            f.stem: pd.DataFrame(
                scipy.io.loadmat(f)["spaceXYZ"], columns=["x", "y", "z"]
            )
            for f in args.dir.glob("spatialIN_*.mat")
        }
    ).reset_index(0, names=["filename"])

    data = pd.concat({"E": spatial_e, "I": spatial_i}).reset_index(
        0, names=["cell_type"]
    )
    data["ID"] = data.filename.str[-2:].astype(np.int8)
    cells["ID"] = cells.filename.str[-2:].astype(np.int8)
    data = data.merge(cells, on="ID", suffixes=("_pre", "_post"))

    data = data.query("z_pre > 100 and z_pre < 310").copy()  # L2/3

    data["z_pre"] = -data["z_pre"]
    data["z_post"] = -data["z_post"]
    data["dx"] = data["x_pre"] - data["x_post"]
    data["dy"] = data["y_pre"] - data["y_post"]
    data["dz"] = data["z_pre"] - data["z_post"]
    data["dr"] = (data["dx"] ** 2 + data["dy"] ** 2) ** 0.5

    # # sns.scatterplot(data, x="x_pre", y="z_pre", hue="cell_type")
    # # sns.kdeplot(data, x="x_pre", y="z_pre", hue="cell_type")
    # sns.relplot(
    #     data,
    #     x="dx",
    #     y="dy",
    #     col="ID",
    #     hue="cell_type",
    #     col_wrap=4,
    #     height=1.75,
    #     aspect=1,
    # )
    # plt.show()
    # sns.relplot(data, x="dx", y="dy", hue="cell_type")
    # # sns.scatterplot(data, x="dr", y="dz", hue="cell_type")
    # plt.show()

    data["dr_binned"] = pd.cut(data["dr"], bins=np.arange(0, 500, 25))

    # counts = data.groupby(
    #     ["dr_binned", "cell_type", "ID"], as_index=False, observed=False
    # ).size()
    # counts["dr"] = pd.IntervalIndex(counts["dr_binned"]).mid
    # counts["dr_left"] = pd.IntervalIndex(counts["dr_binned"]).left
    # counts["dr_right"] = pd.IntervalIndex(counts["dr_binned"]).right
    # counts["total"] = counts.groupby(["cell_type", "ID"])["size"].transform("sum")
    # counts["proportion"] = counts["size"] / counts["total"]
    # counts["density"] = counts["proportion"] / (
    #     area(counts["dr_right"], args.s) - area(counts["dr_left"], args.s)
    # )
    # print(counts)

    # sns.lineplot(
    #     counts, x="dr", y="density", hue="cell_type", units="ID", estimator=None
    # )
    # sns.lineplot(counts, x="dr", y="density", hue="cell_type")
    # plt.show()
    # sns.histplot(data, x="r", hue="cell_type")
    # plt.show()

    counts = data.groupby(
        ["dr_binned", "cell_type"], as_index=False, observed=False
    ).size()
    counts["dr"] = pd.IntervalIndex(counts["dr_binned"]).mid
    counts["dr_left"] = pd.IntervalIndex(counts["dr_binned"]).left
    counts["dr_right"] = pd.IntervalIndex(counts["dr_binned"]).right
    counts["total"] = counts.groupby("cell_type")["size"].transform("sum")
    counts["proportion_low"], counts["proportion_high"] = proportion_confint(
        counts["size"], counts["total"], method="binom_test"
    )
    counts["proportion"] = counts["size"] / counts["total"]
    counts["dA"] = area(counts["dr_right"], args.s) - area(counts["dr_left"], args.s)

    for s in ["", "_low", "_high"]:
        counts[f"density{s}"] = counts[f"proportion{s}"] / counts["dA"]

    max_density = counts["density"][counts["density"] < np.inf].max()
    for s in ["", "_low", "_high"]:
        counts[f"rel_density{s}"] = counts[f"density{s}"] / max_density

    for i, (cell_type, sf) in enumerate(counts.groupby("cell_type")):
        if args.out:
            args.out.mkdir(exist_ok=True)
            np.savetxt(
                args.out / f"E{cell_type}.csv",
                sf[["dr", "rel_density", "rel_density_low", "rel_density_high"]],
                delimiter=",",
            )
        print(sf[["dr", "rel_density", "rel_density_low", "rel_density_high"]])
        plt.plot(sf["dr"], sf["rel_density"], label=cell_type)
        plt.fill_between(
            sf["dr"],
            sf["rel_density_low"],
            sf["rel_density_high"],
            alpha=0.15,
            edgecolor=f"C{i}",
        )
    plt.legend(title="Cell type")
    plt.gca().axhline(
        0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"]
    )
    plt.ylabel("Density (μm$^{-2}$)")
    plt.xlabel("Distance (μm)")
    if args.out:
        plt.savefig(args.out / "density.pdf", bbox_inches="tight")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
