import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from mpl_config import get_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "-b", type=float, default=1.0)
    parser.add_argument("--sigma", "-s", type=float, default=np.inf)
    parser.add_argument("--kappa", "-k", type=float, default=0.0)
    parser.add_argument("--ymax", "-y", type=float)
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    x = np.linspace(0, 1000, 1000)
    figsize, rect = get_sizes(1.15, 1.5, 1, 1.1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    for theta in [0, np.pi / 4, np.pi / 2]:
        y = args.baseline + (1 - args.baseline) * np.exp(
            -(x**2) / (2 * args.sigma**2)
        ) * (1 + 2 * args.kappa * np.cos(2 * theta))
        ax.plot(x, y, label=f"{np.rad2deg(theta):.0f}°")
    ax.set_xlabel("Distance (μm)")
    ax.set_ylabel("Gain")
    ax.set_ylim(0, ax.get_ylim()[1] if args.ymax is None else args.ymax)
    ax.legend(title="Pref. ori.", bbox_to_anchor=(1, 1.1), loc="upper left")
    fig.tight_layout()

    if args.out:
        fig.savefig(
            args.out, metadata={"Subject": " ".join(sys.argv[1:])}, bbox_inches="tight"
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
