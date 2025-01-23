import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def func(x, a, kappa):
    return a * (1 + 2 * kappa * np.cos(x / 90 * np.pi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
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
        samples = np.random.multivariate_normal(popt, pcov, 10000).T[..., None]
        x_smooth = np.linspace(-180, 180, 100)
        y_samples = func(x_smooth, *samples)  # (10000, N_x)
        ypred = func(x_smooth, *popt)
        ypred_err = np.std(y_samples, axis=0, ddof=1)
        # ypred_pi = np.percentile(y_samples, [2.5, 97.5], axis=0)
        plt.plot(x, y)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.5)
        plt.plot(x_smooth, ypred)
        plt.fill_between(
            x_smooth, ypred - 2 * ypred_err, ypred + 2 * ypred_err, alpha=0.5
        )
        # plt.fill_between(x_smooth, *ypred_pi, alpha=0.5)
        plt.xlabel("$\Delta$ pref. ori. (deg)")
        plt.ylabel("Connection probability")
        plt.tight_layout()
    if args.out:
        header = f"Command: `{' '.join(sys.argv)}`, {info}"
        plt.savefig(args.out, bbox_inches="tight", metadata={"Subject": header})
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
