import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from mpl_config import GREY, GRID_WIDTH, get_sizes, set_rcParams


def func(x, a, kappa):
    return a * (1 + 2 * kappa * np.cos(x / 90 * np.pi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--cov", action="store_true")
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    x, y, yerr = np.loadtxt(args.filename, delimiter=",")
    p0 = [np.mean(y), 0]
    popt, pcov = optimize.curve_fit(func, x, y, p0=p0, sigma=yerr)
    psigma = np.diag(pcov) ** 0.5

    info = f"Parameters: ['a', 'kappa'], popt: {popt}, psigma: {psigma}"
    print(info)

    if args.out or args.show:
        set_rcParams()
        figsize, rect = get_sizes(1, 1, 1, 1)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        ax.plot(x, y, ls="--")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25)

        x_smooth = np.linspace(-180, 180, 100)
        ypred = func(x_smooth, *popt)
        ax.plot(x_smooth, ypred, color=("C1" if args.cov else "C0"))

        if args.cov:
            samples = np.random.multivariate_normal(popt, pcov, 10000).T[..., None]
            y_samples = func(x_smooth, *samples)  # (10000, N_x)
            ypred_err = np.std(y_samples, axis=0, ddof=1)
            # ypred_pi = np.percentile(y_samples, [2.5, 97.5], axis=0)
            ax.fill_between(
                x_smooth, ypred - 2 * ypred_err, ypred + 2 * ypred_err, alpha=0.5
            )
            # ax.fill_between(x_smooth, *ypred_pi, alpha=0.5)

        ax.axhline(0, color=GREY, linewidth=GRID_WIDTH)
        ax.set_xlabel("$\Delta$ dir. pref. (deg)")
        ax.set_ylabel("Fraction")
        ax.set_xticks([-180, 0, 180])
        fig.tight_layout()
    if args.out:
        header = f"Command: `{' '.join(sys.argv)}`, {info}"
        fig.savefig(args.out, bbox_inches="tight", metadata={"Subject": header})
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
