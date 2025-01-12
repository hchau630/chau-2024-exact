import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probability", type=Path)
    parser.add_argument("strength", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    x0, y0 = np.loadtxt(args.probability, delimiter=",").T
    x1, y1 = np.loadtxt(args.strength, delimiter=",").T

    if not np.allclose(np.diff(x0, n=2), 0):
        raise ValueError("Data must be binned")

    binwidth = x0[1] - x0[0]
    bins = np.r_[x0 - binwidth / 2, x0[-1] + binwidth / 2]

    df = pd.DataFrame({"x": x1, "y": y1})
    df["x"] = pd.cut(df["x"], bins=bins)
    y1 = df.groupby("x", observed=False)["y"].mean().to_numpy()
    print(f"Interpolated data points: {x0[np.isnan(y1)]}")
    y1 = np.interp(x0, x0[~np.isnan(y1)], y1[~np.isnan(y1)])

    y = y0 * y1

    if args.output:
        header = f"command: {' '.join(sys.argv)}"
        np.savetxt(args.output, np.stack([x0, y], axis=1), delimiter=",", header=header)

    if args.show:
        # for sanity checking
        plt.plot(x0, y0, label="Probability")
        plt.scatter(*np.loadtxt(args.probability, delimiter=",").T)
        plt.plot(x0, y1, label="Strength")
        plt.scatter(*np.loadtxt(args.strength, delimiter=",").T)
        plt.plot(x0, y, label="Product")
        plt.legend()
        plt.xlabel("Distance (μm)")
        plt.ylabel("Value")
        plt.show()


if __name__ == "__main__":
    main()
