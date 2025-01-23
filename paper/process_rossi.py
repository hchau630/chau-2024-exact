import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = {}
    for name in ["rbins", "rEX", "rIN"]:
        data[name] = np.loadtxt(args.dir / f"fig1l_{name}.csv", delimiter=",")
    is_positive = data["rbins"] >= 0
    data = {k: v[is_positive] for k, v in data.items()}

    for filename, name in [("EE", "rEX"), ("EI", "rIN")]:
        out = np.stack([data["rbins"], data[name].mean(axis=1)], axis=1)
        if args.out:
            header = f"command: {' '.join(sys.argv)}"
            np.savetxt(args.out / f"{filename}.csv", out, delimiter=",", header=header)


if __name__ == "__main__":
    main()
