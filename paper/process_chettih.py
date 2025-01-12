import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--ori", action="store_true")
    parser.add_argument("--max-space-dist", type=float, default=float("inf"))
    parser.add_argument("--max-ori-dist", type=float, default=float("inf"))
    parser.add_argument("--binwidth", type=float)
    parser.add_argument("--assume-pyr", action="store_true")
    parser.add_argument("--transpose", "-t", action="store_true")
    parser.add_argument("--format", choices=["csv", "pkl"], default="pkl")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=str)
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=",")
    if args.transpose:
        data = data.T

    if len(data) == 2:
        x, dr = data
        upper_dr_se = None
    elif len(data) == 4:
        x, dr, upper_dr_se, lower_dr_se = data

        if not np.allclose(upper_dr_se - dr, dr - lower_dr_se):
            raise ValueError("Errorbars are not symmetric.")

        if (upper_dr_se < dr).any():
            raise ValueError("Upper errorbars cannot be smaller than the mean.")
    else:
        raise ValueError("Data must have 2 or 4 rows.")

    mask = x < args.max_space_dist
    x, dr = x[mask], dr[mask]
    if upper_dr_se is not None:
        upper_dr_se = upper_dr_se[mask]

    if args.format == "csv":
        out = [x, dr]
        if upper_dr_se is not None:
            out.append(upper_dr_se - dr)
        out = np.stack(out, axis=1)

        if args.out:
            header = f"command: `{' '.join(sys.argv)}`"  # add metadata
            np.savetxt(args.out, out, delimiter=",", header=header)
        else:
            print(out)

    else:
        if args.binwidth:
            if (x != np.sort(x)).all():
                raise ValueError("x must be sorted.")

            print(x[0], x[-1])
            print(dr[0], dr[-1])
            print(x[np.argmin(dr)])
            indices = [0]
            while x[indices[-1]] <= x[-1] - args.binwidth:
                diffs = x - (x[indices[-1]] + args.binwidth)
                # idx = np.count_nonzero(diffs < 0)  # assumes x is sorted
                idx = np.argmin(np.abs(diffs))  # assumes x is sorted
                indices.append(idx)
            indices = np.array(indices)
            x, dr = x[indices], dr[indices]
            if upper_dr_se is not None:
                upper_dr_se = upper_dr_se[indices]

        edges = (x[1:] + x[:-1]) / 2
        first_edge, last_edge = 2 * x[0] - edges[0], 2 * x[-1] - edges[-1]
        if not args.ori:
            # neurons within 25 microns are excluded from analysis
            assert edges[0] > 25.0  # second edge must be at least 25 microns
            first_edge = max(first_edge, 25.0)  # first edge >= 25 microns
        edges = np.r_[first_edge, edges, last_edge]

        x = pd.IntervalIndex.from_breaks(edges, closed="left").astype("category")
        x_key = "rel_ori" if args.ori else "distance"
        df = pd.DataFrame({x_key: x, "dr": dr})

        if args.ori:
            # neurons within 25 microns are excluded from analysis due to
            # potential off-target activations
            df["distance"] = pd.Interval(25.0, args.max_ori_dist, closed="left")
            df["distance"] = df["distance"].astype("category")

        if upper_dr_se is not None:
            df["dr_se"] = upper_dr_se - dr

        if args.assume_pyr:
            df["cell_type"] = "PYR"

        df.attrs["command"] = " ".join(sys.argv)  # add metadata
        if args.out:
            df.to_pickle(args.out)
        else:
            print(df)
            print(df.attrs)

        if args.show:
            x, y = pd.IntervalIndex(df[x_key]).mid.values, df["dr"].values
            print(x, y)
            print(x[1:] - x[:-1])
            plt.plot(x, y)
            if "dr_se" in df:
                yerr = df["dr_se"].values
                plt.fill_between(x, y - yerr, y + yerr, alpha=0.5)
            plt.show()


if __name__ == "__main__":
    main()
