import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
import torch
import matplotlib.pyplot as plt

from niarb import nn
from niarb.nn.modules.frame import ParameterFrame
from niarb.tensors import periodic
from mpl_config import GREY, GRID_WIDTH, get_sizes, set_rcParams


def sample_W(x, y, scale=5.0):
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape

    w00 = x
    w11 = stats.truncnorm.rvs(
        -np.inf, np.minimum((1 - x + y), 1) / scale, scale=scale, size=shape
    )
    w0110 = w00 * w11 - y  # y = w00*w11 - w0110

    # check stability
    assert (w11 < 1).all()
    assert ((1 - w00 - w11 + y) > 0).all()
    np.testing.assert_allclose(w00 * w11 - w0110, y, atol=1e-7)

    w10 = scale * np.random.randn(*shape)
    w01 = w0110 / w10

    W = np.stack([[w00, w01], [w10, w11]])  # (2, 2, *shape)
    W = np.moveaxis(W, (0, 1), (-2, -1))  # (*shape, 2, 2)
    assert W.shape == (*shape, 2, 2)

    return W


def response(
    W: ArrayLike,
    tau_i: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        W: array-like with shape (*, 2, 2)
    """
    W = torch.tensor(W, dtype=torch.float)

    if W.ndim < 2 or W.shape[-2:] != (2, 2):
        raise ValueError("W must have at least 2 dimensions and shape (*, 2, 2)")

    shape = W.shape[:-2]

    kappa = torch.rand((*shape, 2, 2)) / 2  # [0, 0.5)
    sign = W.sign()
    sign[sign == 0] = 1
    kappa[..., :, 0] = kappa[..., :, 0] * sign[..., :, 0]
    kappa[..., :, 1] = -kappa[..., :, 1] * sign[..., :, 1]
    W = W / kappa

    assert (W[..., :, 0] >= 0).all()
    assert (W[..., :, 1] <= 0).all()

    W[..., :, 1] = -W[..., :, 1]  # could also just take abs()

    model = nn.V1(
        ["cell_type", "ori"],
        cell_types=["PYR", "PV"],
        init_stable=False,
        batch_shape=shape,
        tau=[1.0, tau_i],  # very fast inhibition
    )
    model.load_state_dict({"gW": W, "kappa": kappa}, strict=False)

    is_stable = model.spectral_summary(kind="J").abscissa < 0

    ori = periodic.tensor([[0], [1e-5], [90.0]], extents=[(-90.0, 90.0)])  # (3, 1)
    dh = torch.zeros((3,))  # (3,)
    dh[0] = 1.0

    # Note that dV does not matter since we are only interested in the response shape
    x = ParameterFrame(
        {
            "cell_type": torch.tensor([0]),  # (1,)
            "ori": ori,  # (3, 1)
            "dV": torch.tensor([1.0]),  # (1,)
            "dh": dh,  # (3,)
        },
        ndim=1,
    )  # (3,)

    # I found out that my code for computing K0 on single-precision tensors can
    # sometimes return NaN values on valid inputs. But the double-precision routines
    # seem to work fine. So we convert everything to double-precision.
    model.double()
    x = x.double()

    with torch.inference_mode():
        out = model(x, ndim=x.ndim, check_circulant=False, to_dataframe=False)  # (*, 3)
    drE = out["dr"][..., 1:]  # (*, 2)

    return is_stable.numpy(), drE.numpy()  # (*,), (*, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical", "--num", dest="numerical", action="store_true")
    parser.add_argument("-x", type=float, nargs=2, default=(0, 2))
    parser.add_argument("-y", type=float, nargs=2, default=(-2, 2))
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--tau-i", type=float, default=0.01)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    x = np.linspace(*args.x, args.N)
    y = np.linspace(*args.y, args.N)
    x, y = np.meshgrid(x, y)

    # get the 2x2 W matrix
    W = sample_W(x, y)

    # compute various zero crossings
    if args.numerical:
        is_stable, drE = response(W, args.tau_i)
        z = drE[..., 0] - drE[..., 1]  # iso - ortho
    else:
        is_stable = (np.linalg.det(np.eye(2) - W) > 0) & (W[..., 1, 1] < 1)
        z = np.linalg.det(W) - W[..., 0, 0]

    # set some plotting defaults
    set_rcParams()
    figsize, rect = get_sizes(1.15, 1.0, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    ax.contourf(x, y, z, levels=[-1e8, 0, 1e8], colors=["#FF766A", "#88CBEC"])

    # plot instability region
    ax.contourf(x, y, is_stable, levels=[-0.5, 0.5], colors=["black"])

    # plot analytic phase boundaries
    ylim = ax.get_ylim()
    _x = np.linspace(*ax.get_xlim())
    ax.plot(_x, _x, color=GREY, linewidth=GRID_WIDTH)
    ax.set_ylim(*ylim)

    # nicer looking y-axis
    ax.set_yticks([ylim[0], sum(ylim) / 2, ylim[1]])
    ax.yaxis.set_major_formatter("{x:g}")

    # add labels
    ax.set_xlabel(r"$\tilde{w}_{EE}$")
    ax.set_ylabel(r"$\tilde{w}_{EI}\tilde{w}_{IE} - \tilde{w}_{EE}\tilde{w}_{II}$")

    # save figure
    if args.out:
        fig.savefig(args.out, metadata={"Subject": " ".join(["python"] + sys.argv)})

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
