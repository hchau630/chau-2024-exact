import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

CM = 1 / 2.54  # cm to inch
AXSIZE = (2.1 * CM, 2.1 * CM)
FIGSIZE = (3.6 * CM, 3.4 * CM)
RECT = (
    0.95 * CM / FIGSIZE[0],
    0.9 * CM / FIGSIZE[1],
    AXSIZE[0] / FIGSIZE[0],
    AXSIZE[1] / FIGSIZE[1],
)
rcParams["font.size"] = 7.25  # default: 10 pts
rcParams["axes.labelpad"] = 0.0  # default: 4.0 pts
rcParams["axes.titlepad"] = 4.0  # default: 6.0 pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    g = np.linspace(0, 2, 100)
    w00 = 2
    tr, det = 1, 3
    y = (g * w00 - g**2 * det) / (1 - g * tr + g**2 * det)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    ax.plot(g, y)
    ax.axvline(1, color="black", linestyle="--")
    ax.axhline(0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"])
    ax.set_ylabel(r"$\langle r_E \rangle$")
    ax.set_xlabel("Gain")
    ax.set_xticks([0, 1, 2])
    fig.tight_layout()
    if args.output:
        plt.savefig(
            args.output, bbox_inches="tight", metadata={"Subject": " ".join(sys.argv)}
        )
    plt.show()


if __name__ == "__main__":
    main()
