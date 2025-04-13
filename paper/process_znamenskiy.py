import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--suffix", "-s", type=str, default="")
    parser.add_argument("--ori", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    for output_name, input_name in [("EI", "IPSP"), ("IE", "EPSP")]:
        data = np.loadtxt(args.dir / f"{input_name}{args.suffix}.csv", delimiter=",")
        data = data.T
        data = data[np.argsort(data[:, 0])]
        data[:, 1] = 10 ** data[:, 1]

        if args.show:
            # should look the same as Znamenskiy et al. 2024 Figure S4M and S5M
            X, y = sm.add_constant(data[:, :1]), data[:, 1]
            results = sm.OLS(np.log10(y) if args.log else y, X).fit()
            print(results.summary())
            plt.scatter(data[:, 0], data[:, 1])
            if args.log:
                plt.yscale("log")
            plt.xlabel("Δ pref. ori. (deg)" if args.ori else "Distance (μm)")
            plt.ylabel(f"{input_name} (mV)")
            plt.xlim(0, 90 if args.ori else 500)
            plt.gca().axhline(
                0, color=rcParams["grid.color"], lw=rcParams["grid.linewidth"]
            )
            plt.show()

        if args.output:
            header = f"command: {' '.join(sys.argv)}"
            np.savetxt(
                args.output / f"{output_name}{args.suffix}.csv",
                data,
                delimiter=",",
                header=header,
            )


if __name__ == "__main__":
    main()
