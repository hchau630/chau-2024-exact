from collections.abc import Callable, Sequence
from itertools import chain

import torch
from torch import Tensor

from .functions import Function
from .frame import ParameterFrame
from ..functional import diff
from niarb.tensors.periodic import PeriodicTensor
from niarb.special.resolvent import laplace_r


__all__ = [
    "MatrixKernel",
    "GaussianKernel",
    "LaplaceKernel",
    "PiecewiseKernel",
    "TuningKernel",
]


class Kernel(Function):
    kernel: Callable[[*tuple[Tensor, ...]], Tensor]

    def __init__(
        self,
        x_keys: Sequence[str] | str = (),
        y_keys: Sequence[str] | str | None = None,
    ):
        super().__init__()
        if isinstance(x_keys, str):
            x_keys = (x_keys,)
        if isinstance(y_keys, str):
            y_keys = (y_keys,)

        self.x_keys = x_keys
        self.y_keys = x_keys if y_keys is None else y_keys

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        return self.kernel(*chain.from_iterable(zip(x_, y_)))


class RadialKernel(Kernel):
    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        args = tuple(chain.from_iterable(zip(x_, y_)))
        if len(args) < 2:
            raise ValueError("RadialKernel requires at least two arguments")
        return self.kernel(diff(args[0], args[1]).norm(dim=-1), *args[2:])


class MatrixKernel(Kernel):
    def __init__(self, matrix: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix = matrix

    def kernel(self, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        return self.matrix[idx_x, idx_y]


class GaussianKernel(RadialKernel):
    def __init__(self, sigma: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        sigma = self.sigma[idx_x, idx_y]
        return torch.exp(-(r**2) / (2 * sigma**2))


class LaplaceKernel(RadialKernel):
    def __init__(self, d: int, sigma: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = d
        self.sigma = sigma

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        sigma = self.sigma[idx_x, idx_y]
        return laplace_r(self.d, 1 / sigma**2, r)


class PiecewiseKernel(RadialKernel):
    def __init__(
        self,
        f: RadialKernel,
        g: RadialKernel,
        radius: Tensor,
        *args,
        continuous: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.f = f
        self.g = g
        self.radius = radius
        self.continuous = continuous

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        radius = self.radius[idx_x, idx_y]
        ratio = (
            self.f.kernel(radius, idx_x, idx_y) / self.g.kernel(radius, idx_x, idx_y)
            if self.continuous
            else 1
        )
        return torch.where(
            r < radius,
            self.f.kernel(r, idx_x, idx_y),
            ratio * self.g.kernel(r, idx_x, idx_y),
        )


class TuningKernel(Kernel):
    def __init__(self, kappa: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = kappa

    def kernel(
        self, x: PeriodicTensor, y: PeriodicTensor, idx_x: Tensor, idx_y: Tensor
    ) -> Tensor:
        theta = diff(x, y)
        return 1 + 2 * self.kappa[idx_x, idx_y] * torch.cos(
            theta.to_period(2 * torch.pi).norm(dim=-1)
        )
