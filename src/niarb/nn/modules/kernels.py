from collections.abc import Callable, Sequence
from itertools import chain

import torch
from torch import Tensor

from .functions import Function
from .frame import ParameterFrame
from ..functional import diff
from niarb.tensors.periodic import PeriodicTensor


__all__ = ["MatrixKernel", "GaussianKernel", "TuningKernel"]


class Kernel(Function):
    kernel: Callable[[*tuple[Tensor, ...]], Tensor]

    def __init__(
        self, x_keys: Sequence[str] | str, y_keys: Sequence[str] | str | None = None
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


class MatrixKernel(Kernel):
    def __init__(self, matrix: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix = matrix

    def kernel(self, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        return self.matrix[idx_x, idx_y]


class GaussianKernel(Kernel):
    def __init__(self, sigma: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def kernel(self, x: Tensor, y: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        sigma = self.sigma[idx_x, idx_y]
        return torch.exp(-diff(x, y) ** 2 / (2 * sigma**2))


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
