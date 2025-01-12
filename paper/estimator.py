from pathlib import Path
from functools import partial
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def validate(x, y):
    if x.ndim != 1 or y.ndim == 0 or x.shape[0] != y.shape[-1]:
        raise ValueError(f"Invalid input shapes: {x.shape=}, {y.shape=}.")

    if np.any(x[:-1] > x[1:]):
        raise ValueError("x must be sorted.")


def crossings(x, y, n=0, m=None):
    crossings = np.diff(np.signbit(y), axis=-1) != 0  # (*, len(x) - 1)
    if m:
        mask = np.count_nonzero(crossings, axis=-1) <= m
        crossings = crossings[mask]
        y = y[mask]
    idx = np.argmax(crossings, keepdims=True, axis=-1)  # (*, 1), 1st zero crossing
    for _ in range(n):
        np.put_along_axis(crossings, idx, False, -1)
        idx = np.argmax(crossings, keepdims=True, axis=-1)  # (*, 1), nth zero crossing
    has_crossings = np.any(crossings, axis=-1)  # (*)
    idx_ = np.squeeze(idx, -1)  # (*)
    x0 = np.where(has_crossings, x[:-1][idx_], np.nan)
    x1 = np.where(has_crossings, x[1:][idx_], np.nan)
    y0 = np.squeeze(np.take_along_axis(y[..., :-1], idx, axis=-1), -1)
    y1 = np.squeeze(np.take_along_axis(y[..., 1:], idx, axis=-1), -1)
    y0 = np.where(has_crossings, y0, np.nan)
    y1 = np.where(has_crossings, y1, np.nan)
    return x0 - y0 * (x1 - x0) / (y1 - y0)  # (*), estimate by linear interpolation


def r0(x, y, **kwargs):
    return crossings(x, y, **kwargs)


# def r1(x, y):
#     return crossings(x, y, n=1)


def rmin(x, y):
    idx = np.argmin(y, axis=-1)  # (*)
    return x[idx]  # (*)


# ESTIMATORS = {"r0": r0, "r1": r1, "rmin": rmin}
ESTIMATORS = {"r0": r0, "rmin": rmin}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument(
        # "--estimator", "-e", type=str, choices=["r0", "r1", "rmin"], default="r0"
        "--estimator",
        "-e",
        type=str,
        choices=["r0", "rmin"],
        default="r0",
    )
    parser.add_argument("-N", type=int, default=100000)
    parser.add_argument("-q", type=float, default=0.05)
    parser.add_argument("--max-crossings", "-m", type=int)
    parser.add_argument("--bias", "-b", type=float, default=0)
    parser.add_argument("--rmin", type=float, default=0)
    parser.add_argument("--rmax", type=float, default=np.inf)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Data must be 2D with 3 columns")

    x, y, yerr = data.T

    # apply bias
    y = y + args.bias * np.abs(np.min(y))

    # apply limits on r
    mask = (x > args.rmin) & (x < args.rmax)
    x, y, yerr = x[mask], y[mask], yerr[mask]

    # sample y
    y_sampled = np.random.normal(y, yerr, size=(args.N, len(y)))

    # validate input
    validate(x, y)

    # compute estimates
    kwargs = {"m": args.max_crossings} if args.max_crossings else {}
    estimator = partial(ESTIMATORS[args.estimator], **kwargs)
    raw_estimate = estimator(x, y)
    estimates = estimator(x, y_sampled)  # (?)
    while len(estimates) < args.N:
        estimates = np.concatenate([estimates, estimator(x, y_sampled)])
    estimates = estimates[: args.N]

    stats = (
        f"raw estimate: {raw_estimate}\n"
        f"mean: {np.mean(estimates)}\n"
        f"median: {np.median(estimates)}\n"
        f"std: {np.std(estimates, ddof=1)}\n"
        f"quantiles (q={args.q}): {np.nanquantile(estimates, [args.q, 1 - args.q])}"
    )
    print(stats)

    if args.show:
        plt.hist(estimates, bins=20)
        plt.show()

    if args.out:
        header = f"command: {' '.join(sys.argv)}\n{stats}"
        np.savetxt(args.out, estimates, delimiter=",", header=header)


if __name__ == "__main__":
    main()
