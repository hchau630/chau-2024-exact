import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--bootstrap", "-b", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = {}
    for name in ["rbins", "rEX", "rIN"]:
        data[name] = np.loadtxt(args.dir / f"fig1l_{name}.csv", delimiter=",")
    is_positive = data["rbins"] >= 0
    data = {k: v[is_positive] for k, v in data.items()}

    for filename, name in [("EE", "rEX"), ("EI", "rIN")]:
        out = [data["rbins"], data[name].mean(axis=1)]
        if args.bootstrap:
            out += stats.bootstrap((data[name].T,), np.mean).confidence_interval
        out = np.stack(out, axis=1)
        if args.out:
            header = f"command: {' '.join(sys.argv)}"
            args.out.mkdir(exist_ok=True)
            np.savetxt(args.out / f"{filename}.csv", out, delimiter=",", header=header)


if __name__ == "__main__":
    main()
