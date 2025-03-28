import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--bootstrap", "-b", action="store_true")
    parser.add_argument("--file-type", "-t", choices=["csv", "pkl"], default="csv")
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
            args.out.mkdir(exist_ok=True)
            if args.file_type == "csv":
                header = f"command: {' '.join(sys.argv)}"
                np.savetxt(
                    args.out / f"{filename}.csv", out, delimiter=",", header=header
                )
            else:
                breaks = (out[:-1, 0] + out[1:, 0]) / 2
                breaks = np.r_[0, breaks, breaks[-1] + breaks[0]]
                out = pd.DataFrame(out, columns=["distance", "mean", "low", "high"])
                out["distance"] = pd.IntervalIndex.from_breaks(breaks)
                out.attrs["command"] = " ".join(sys.argv)
                out.to_pickle(args.out / f"{filename}.pkl")


if __name__ == "__main__":
    main()
