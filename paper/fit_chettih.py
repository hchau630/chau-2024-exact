import argparse
from functools import partial
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from niarb import nn
from niarb.nn.modules.frame import ParameterFrame

model = nn.V1(
    ["cell_type", "space"],
    cell_types=["PYR", "PV"],
    tau=[1.0, 0.33],
    init_stable=False,
).double()


def func(sigma, x, wee, weiie, wii):
    gW = torch.tensor([[wee, weiie**0.5], [weiie**0.5, wii]], dtype=torch.double)
    model.load_state_dict({"gW": gW, "sigma": sigma.double()}, strict=False)
    space = torch.tensor(np.r_[0, x], dtype=torch.double)
    space = torch.stack([space, torch.zeros_like(space)], dim=-1)
    dh = torch.zeros(space.shape[0], dtype=torch.double)
    dh[0] = 1.0
    x = ParameterFrame(
        {
            "cell_type": torch.tensor([0]),
            "space": space,
            "dV": torch.tensor([1.0], dtype=torch.double),
            "dh": dh,
            "mask": ~dh.bool(),
        }
    )
    with torch.inference_mode():
        y = model(x, ndim=x.ndim, check_circulant=False)["dr"]
        y = y / y.norm()
    # print(model.spectral_summary(kind="J").abscissa)
    return y.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("-N", type=int, default=150)
    parser.add_argument("--sigma", "-s", type=float, nargs=2, default=[125, 100])
    args = parser.parse_args()

    assert args.N % 2 == 0

    data = np.loadtxt(args.data, delimiter=",")  # (N, 2)
    x, y = data.T  # (N,), (N,)
    y = y / np.linalg.norm(y)

    sigma = torch.tensor(
        [[args.sigma[0], args.sigma[1]], [args.sigma[1], args.sigma[0]]]
    )

    popt, pcov, infodict, mesg, ier = curve_fit(
        partial(func, sigma),
        x,
        y,
        p0=[2.0, 4.0, 1.0],
        bounds=([1, 1e-5, 1e-1], [np.inf] * 3),
        full_output=True,
    )
    print(popt)
    print(pcov)
    print(mesg)
    print(ier)

    plt.plot(x, func(sigma, x, *popt))
    plt.plot(x, y)
    plt.gca().axhline(0, color="grey")
    plt.show()


if __name__ == "__main__":
    main()
