import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
from mpl_config import GRID_WIDTH, GRID_COLOR, get_sizes

rcParams["axes.labelpad"] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--ori", action="store_true")
    parser.add_argument("--reg", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = np.loadtxt(args.path, delimiter=",")

    x = "Δ ori. pref. (deg)" if args.ori else "Distance (μm)"
    y = "IPSP (mV)" if args.path.stem == "EI" else "EPSP (mV)"
    data = pd.DataFrame(data, columns=[x, y])

    figsize, rect = get_sizes(1, 1, 1, 1.05)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    scatter_kws = {"color": "C1", "s": 2 if args.reg else 8}
    if args.reg:
        sns.regplot(
            data, x=x, y=y, ax=ax, scatter_kws=scatter_kws, line_kws={"color": "black"}
        )
    else:
        sns.scatterplot(data, x=x, y=y, ax=ax, **scatter_kws)

    ax.set_xlim(0, 90 if args.ori else 500)
    ax.axhline(0, color=GRID_COLOR, linewidth=GRID_WIDTH)

    if args.out:
        fig.savefig(args.out, metadata={"Subject": " ".join(sys.argv[1:])})

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
