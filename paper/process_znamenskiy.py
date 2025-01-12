import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--suffix", "-s", type=str, default="")
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
            plt.scatter(data[:, 0], data[:, 1])
            plt.yscale("log")
            plt.xlabel("Distance (μm)")
            plt.ylabel(f"{input_name} (mV)")
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
