import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=Path, nargs="+")
    parser.add_argument("sigma_mean", type=float)
    parser.add_argument("sigma_std", type=float)
    parser.add_argument("--estimator", "-e", type=str, default="f[0]")
    parser.add_argument("-q", type=float, default=0.025)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    f = [np.loadtxt(filename, delimiter=",") for filename in args.filenames]

    estimates = eval(args.estimator, {"f": f})
    sigma = np.random.normal(args.sigma_mean, args.sigma_std, size=len(estimates))
    estimates = estimates / sigma

    stats = [
        f"mean: {np.mean(estimates)}",
        f"median: {np.median(estimates)}",
        f"std: {np.std(estimates, ddof=1)}",
        f"quantiles (q={args.q}): {np.nanquantile(estimates, [args.q, 1 - args.q])}",
    ]
    print(stats)

    plt.hist(estimates, bins=20)

    if args.out:
        plt.savefig(
            args.out,
            bbox_inches="tight",
            metadata={"Subject": f"command: '{' '.join(sys.argv)}', {' '.join(stats)}"},
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
