import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_config import GREY, GRID_WIDTH, get_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = np.loadtxt(args.path, delimiter=",")
    ylabel = "IPSP (mV)" if args.path.stem == "EI" else "EPSP (mV)"

    figsize, rect = get_sizes(1, 1, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    ax.scatter(data[:, 0], data[:, 1], s=2, color="C1")
    ax.set_xlabel("Distance (μm)", labelpad=0)
    ax.set_ylabel(ylabel, labelpad=0)
    ax.set_xlim(0, 500)
    ax.axhline(0, color=GREY, linewidth=GRID_WIDTH)
    fig.tight_layout()

    if args.out:
        fig.savefig(
            args.out, metadata={"Subject": " ".join(sys.argv[1:])}, bbox_inches="tight"
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
