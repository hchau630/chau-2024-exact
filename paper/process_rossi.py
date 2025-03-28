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
    parser.add_argument("--file-type", "-t", choices=["np", "pd"], default="np")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = {}
    for name in ["rbins", "rEX", "rIN"]:
        data[name] = np.loadtxt(args.dir / f"fig1l_{name}.csv", delimiter=",")
    is_positive = data["rbins"] >= 0
    data = {k: v[is_positive] for k, v in data.items()}

    outs = {}
    for filename, name in [("EE", "rEX"), ("EI", "rIN")]:
        out = [data["rbins"], data[name].mean(axis=1)]
        if args.bootstrap:
            out += stats.bootstrap((data[name].T,), np.mean).confidence_interval
        outs[filename] = np.stack(out, axis=1)
    
    if args.out:
        if args.file_type == "csv":
            args.out.mkdir(exist_ok=True)
            header = f"command: {' '.join(sys.argv)}"
            np.savetxt(args.out / f"{filename}.csv", out, delimiter=",", header=header)
        else:
            breaks = (data["rbins"][:-1] + data["rbins"][1:]) / 2
            breaks = np.r_[0, breaks, breaks[-1] + breaks[0]]
            df = []
            for k, v in outs.items():
                v = pd.DataFrame(v, columns=["distance", "density", "low", "high"])
                v["distance"] = pd.IntervalIndex.from_breaks(breaks)
                v["postsynaptic_cell_type"] = k[0]
                v["presynaptic_cell_type"] = k[1]
                df.append(v)
            df = pd.concat(df).reset_index(drop=True)
            df.attrs["command"] = " ".join(sys.argv)
            df.to_pickle(args.out)


if __name__ == "__main__":
    main()
