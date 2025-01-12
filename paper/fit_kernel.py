import sys
import argparse
from functools import partial, wraps
from pathlib import Path
import math
import cmath

import numpy as np
import torch
from scipy import optimize, integrate
import pandas as pd
import matplotlib.pyplot as plt

from niarb.special.resolvent import laplace_r
from niarb.viz import figplot


def integrand(d, sigma, r):
    return r ** (d - 1) * laplace_r(d, sigma**-2, torch.as_tensor(r)[None])


def fitted(kernel):
    @wraps(kernel)
    def fitted_kernel(*args):
        *args, a = args
        out = kernel(*args)
        return a * out

    return fitted_kernel


@fitted
def binned_bessel_kernel(d, binwidth, r, sigma):
    out = laplace_r(d, sigma**-2, torch.as_tensor(r), dr=binwidth / 2)
    mask = (r > 0) & (r <= binwidth * 2)
    indices = mask.nonzero()[0]
    for i in indices:
        out[i] = integrate.quad(
            partial(integrand, d, sigma), r[i] - binwidth / 2, r[i] + binwidth / 2
        )[0]
        out[i] = out[i] / (
            (r[i] + binwidth / 2) ** d / d - (r[i] - binwidth / 2) ** d / d
        )
    return out.numpy()


@fitted
def bessel_kernel(d, r, sigma):
    return laplace_r(d, sigma**-2, torch.as_tensor(r)).numpy()


@fitted
def gaussian_kernel(r, sigma):
    return np.exp(-(r**2) / (2 * sigma**2))


@fitted
def real_resp_kernel(d, r, s0, s1, c):
    out0 = laplace_r(d, s0**-2, torch.as_tensor(r))
    out1 = laplace_r(d, s1**-2, torch.as_tensor(r))
    return (out0 + c * out1).numpy()


@fitted
def cplx_resp_kernel(d, r, sigma, ltheta, ctheta):
    l = sigma**-2 * cmath.exp(1j * ltheta)
    out0 = laplace_r(d, l, torch.as_tensor(r))
    out1 = laplace_r(d, l.conjugate(), torch.as_tensor(r))
    c = cmath.exp(1j * ctheta)
    return (c * out0 + c.conjugate() * out1).real.numpy()


def curve_fit(
    func, xdata, ydata, p0, sigma=None, bootstrap=None, full_output=False, **kwargs
):
    n, m = len(ydata), len(p0)
    out = optimize.curve_fit(
        func, xdata, ydata, p0=p0, sigma=sigma, full_output=full_output, **kwargs
    )

    y_fit = func(xdata, *out[0])
    if sigma is None:
        chisq = np.sum((ydata - y_fit) ** 2)
    else:
        chisq = np.sum(((ydata - y_fit) / sigma) ** 2)
    reduced_chisq = chisq / (n - m)

    if bootstrap:
        if sigma is None:
            sigma = reduced_chisq**0.5

        ydata_samples = np.random.normal(ydata, sigma, (bootstrap, len(ydata)))

        outs = [curve_fit(func, xdata, yi, p0, **kwargs) for yi in ydata_samples]

        # keep 95% of best fits to get rid of trials where curve_fit got
        # stuck at a local minimum
        outs = list(sorted(outs, key=lambda o: o[0]))  # sort by loss
        outs = outs[: int(0.95 * bootstrap)]

        popts = np.stack([o[1] for o in outs], axis=-1)
        pcov = np.cov(popts)
        infodict = {
            f"percentile ({p})": np.percentile(popts, p, axis=-1)
            for p in [2.5, 50, 97.5]
        }
        infodict["mean"] = np.mean(popts, axis=-1)

        out = (out[0], pcov, infodict, *out[-2:]) if full_output else (out[0], pcov)

    return (reduced_chisq**0.5, *out)


KERNELS = {
    "gaussian": gaussian_kernel,
    "real_resp": real_resp_kernel,
    "cplx_resp": cplx_resp_kernel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=Path, nargs="+")
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument(
        "--kernel",
        "-k",
        choices=["bessel", "gaussian", "real_resp", "cplx_resp"],
        default="bessel",
    )
    parser.add_argument("--min-dist", "-m", type=float, default=0)
    parser.add_argument("--bootstraps", "-B", type=int)
    parser.add_argument("--binned", "-b", action="store_true")
    parser.add_argument("--bins", "-n", type=int)
    parser.add_argument("--s0", "-s", type=float, default=100)
    parser.add_argument("--s1", type=float, default=100)
    parser.add_argument("-c", type=float, default=1)
    parser.add_argument("--ltheta", type=float, default=0)
    parser.add_argument("--ctheta", type=float, default=0)
    parser.add_argument("--ord", type=float, default=2)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # initialize kernel-specific parameters
    if args.kernel in {"bessel", "gaussian"}:
        p0 = [args.s0]
        bounds = ([0], [np.inf])
        fit_label = r"$W_{\alpha \beta}(\mathbf{x} - \mathbf{y})$"
        y_label = r"E/IPSP $\times$ prob. (mV)"
    elif args.kernel == "real_resp":
        p0 = [args.s0, args.s1, args.c]
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        fit_label = r"$\tilde{L}_{\alpha \beta}(\mathbf{x} - \mathbf{y})$"
        y_label = "Response"
    elif args.kernel == "cplx_resp":
        p0 = [args.s0, args.ltheta, args.ctheta]
        bounds = ([0, -np.pi, -np.pi], [np.inf, np.pi, np.pi])
        fit_label = r"$\tilde{L}_{\alpha \beta}(\mathbf{x} - \mathbf{y})$"
        y_label = "Response"
    else:
        raise ValueError(f"Unknown kernel: {args.kernel}")
    bounds = (bounds[0] + [0], bounds[1] + [np.inf])

    s, df, info = {}, {}, []
    for filename in args.filenames:
        data = np.loadtxt(filename, delimiter=",")

        if data.ndim != 2:
            raise ValueError("Data must be 2D")

        if data.shape[-1] not in {2, 3}:
            raise ValueError("Data must have 2 or 3 columns")

        if data.shape[-1] == 2:
            x, y = data.T
            yerr = None
        else:
            x, y, yerr = data.T

        if args.binned:
            if yerr is not None:
                raise ValueError("Data must not have errorbars")

            if args.bins is None:
                if not np.allclose(np.diff(x, n=2), 0):
                    raise ValueError("Data must be binned")
            else:
                df = pd.DataFrame({"x": x, "y": y})
                df["x"] = pd.cut(df["x"], bins=args.bins)
                df = df.groupby("x", as_index=False)["y"].mean()
                x, y = pd.IntervalIndex(df["x"]).mid, df["y"]
                x, y = x.to_numpy(), y.to_numpy()
            binwidth = x[1] - x[0]

        if args.kernel == "bessel":
            if args.binned:
                kernel = partial(binned_bessel_kernel, args.d, binwidth)
            else:
                kernel = partial(bessel_kernel, args.d)
        else:
            if args.binned:
                raise NotImplementedError(f"{args.kernel} cannot be binned")
            kernel = KERNELS[args.kernel]
            if args.kernel != "gaussian":
                kernel = partial(kernel, args.d)

        # only fit to data where x >= min_dist
        mask = x >= args.min_dist

        # initialize amplitude such that norm(kernel(x[mask], *p0)) == norm(y[mask])
        a = np.linalg.norm(y[mask], ord=args.ord) / np.linalg.norm(
            kernel(x[mask], *p0, 1), ord=args.ord
        )
        print(f"{p0=}, {a=}")

        # fit kernel to data
        sigma = None if yerr is None else yerr[mask]
        loss, popt, pcov, infodict, _, _ = curve_fit(
            kernel,
            x[mask],
            y[mask],
            [*p0, a],
            sigma=sigma,
            bounds=bounds,
            bootstrap=args.bootstraps,
            max_nfev=1e4,
            full_output=True,
        )
        info += [
            f"{filename.stem}: popt={popt.tolist()}, psigma={np.diag(pcov) ** 0.5}, {loss=:.6e}"
        ]
        if args.bootstraps:
            info[-1] = ", ".join(
                [info[-1], ", ".join(f"{k}={v}" for k, v in infodict.items())]
            )
        print(info[-1])

        s[filename.stem] = (popt[0], pcov[0, 0] ** 0.5)
        df[(filename.stem, "data")] = pd.DataFrame({"x": x, "y": y})
        df[(filename.stem, fit_label)] = pd.DataFrame({"x": x, "y": kernel(x, *popt)})

    s_gmean = math.prod(si[0] for si in s.values()) ** (1 / len(s))
    s_gmean_err = sum((si[1] / si[0]) ** 2 for si in s.values()) ** 0.5 * s_gmean
    info += [f"Geometric mean of s: {s_gmean:.1f}±{s_gmean_err:.1f}"]
    print(info[-1])

    df = pd.concat(df, names=["filename", "kind"]).reset_index(level=[0, 1])
    df["filename"] = df["filename"].replace({"EI": r"I $\to$ E", "IE": r"E $\to$ I"})
    # print(df)

    g = figplot(
        df,
        "relplot",
        x="x",
        y="y",
        hue="filename" if len(args.filenames) > 1 else None,
        style="kind",
        style_order=[fit_label, "data"],
        kind="line",
        facet_kws={"legend_out": False},
        legend_title=False,
        height=2,
        aspect=1.25,
        mapping={"x": "Distance (μm)", "y": y_label},
        grid="yzero",
    )

    # get rid of legend subtitles, very ugly
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) == len(labels) == 6:
        plt.gca().legend(
            [handles[i] for i in [1, 2, 4, 5]],
            [labels[i] for i in [1, 2, 4, 5]],
            loc="upper right",
            bbox_to_anchor=(1, 1.1),
        )

    if args.output:
        g.figure.savefig(
            args.output,
            bbox_inches="tight",
            metadata={"Subject": f"command: '{' '.join(sys.argv)}', {' '.join(info)}"},
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
