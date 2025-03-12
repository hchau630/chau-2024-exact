from collections.abc import Callable, Sequence
from numbers import Number
from itertools import chain

import torch
from torch import Tensor

from .functions import Function
from .frame import ParameterFrame
from ..functional import diff
from niarb.tensors.periodic import PeriodicTensor
from niarb.special.resolvent import laplace_r
from niarb.optimize import elementwise


__all__ = [
    "Matrix",
    "Gaussian",
    "Laplace",
    "Monotonic",
    "Piecewise",
    "Tuning",
    "Radial",
    "radial",
]


class Kernel(Function):
    kernel: Callable[[*tuple[Tensor, ...]], Tensor]
    n: int

    def __init__(
        self,
        x_keys: Sequence[str] | str = (),
        y_keys: Sequence[str] | str | None = None,
        validate_args: bool = True,
    ):
        super().__init__()
        if isinstance(x_keys, str):
            x_keys = (x_keys,)
        if isinstance(y_keys, str):
            y_keys = (y_keys,)

        self.x_keys = tuple(x_keys)
        self.y_keys = self.x_keys if y_keys is None else tuple(y_keys)
        self.validate_args = validate_args

        if len(self.x_keys) != self.n:
            raise ValueError(f"Expected {self.n} x_keys, but got {len(self.x_keys)}.")
        if len(self.y_keys) != self.n:
            raise ValueError(f"Expected {self.n} y_keys, but got {len(self.y_keys)}.")

    def validate(self, x: ParameterFrame, y: ParameterFrame):
        if any(k not in x for k in self.x_keys):
            raise ValueError(f"x must contain all keys {self.x_keys}.")
        if any(k not in y for k in self.y_keys):
            raise ValueError(f"y must contain all keys {self.y_keys}.")

        try:
            torch.broadcast_shapes(x.shape, y.shape)
        except RuntimeError:
            raise ValueError(
                f"x and y must have broadcastable shapes, but {x.shape=} and {y.shape=}."
            )

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        if self.validate_args:
            self.validate(x, y)

        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        return self.kernel(*chain.from_iterable(zip(x_, y_)))


class Radial(Kernel):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.n < 1:
            raise ValueError(f"Subclasses of Radial must have n >= 1, but {cls.n=}.")

    def validate(self, x: ParameterFrame, y: ParameterFrame):
        super().validate(x, y)

        x_rkey, y_rkey = self.x_keys[0], self.y_keys[0]
        if x.data[x_rkey].ndim != x.ndim + 1:
            raise ValueError(f"x['{x_rkey}'] must have one more dimension than x.")
        if y.data[y_rkey].ndim != y.ndim + 1:
            raise ValueError(f"y['{y_rkey}'] must have one more dimension than y.")
        if x.data[x_rkey].shape[-1] != y.data[y_rkey].shape[-1]:
            raise ValueError(
                f"Last dimension of x['{x_rkey}'] and y['{y_rkey}'] must be the same."
            )

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        if self.validate_args:
            self.validate(x, y)

        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        args = tuple(chain.from_iterable(zip(x_, y_)))
        return self.kernel(diff(args[0], args[1]).norm(dim=-1), *args[2:])


class Matrix(Kernel):
    n = 1

    def __init__(self, matrix: Tensor | Sequence[Sequence[Number]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("matrix", torch.as_tensor(matrix), persistent=False)

    def kernel(self, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        return self.matrix[idx_x, idx_y]


class Gaussian(Radial):
    n = 2

    def __init__(self, sigma: Tensor | Sequence[Sequence[Number]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("sigma", torch.as_tensor(sigma), persistent=False)

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        # Weird indexing due to vmap bug: https://github.com/pytorch/pytorch/issues/124423
        sigma = self.sigma[idx_x[None], idx_y[None]][0]
        return torch.exp(-(r**2) / (2 * sigma**2))


class Laplace(Radial):
    n = 2

    def __init__(
        self,
        d: int,
        sigma: Tensor | Sequence[Sequence[Number]],
        *args,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.d = d
        self.register_buffer("sigma", torch.as_tensor(sigma), persistent=False)
        self.normalize = normalize

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        # Weird indexing due to vmap bug: https://github.com/pytorch/pytorch/issues/124423
        sigma = self.sigma[idx_x[None], idx_y[None]][0]
        out = laplace_r(self.d, 1 / sigma, r, is_sqrt=True, validate_args=False)
        if self.normalize:
            zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
            Z = laplace_r(self.d, 1 / self.sigma, zero, is_sqrt=True, validate_args=False)
            out = out / Z[idx_x[None], idx_y[None]][0]
        return out


class Monotonic(Radial):
    n = 1

    def __init__(self, f: Radial, *args, x0: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        if self.x_keys[0] != f.x_keys[0]:
            raise ValueError(f"First x_key of f must be {self.x_keys[0]}.")
        if self.y_keys[0] != f.y_keys[0]:
            raise ValueError(f"First y_key of f must be {self.y_keys[0]}.")

        self.x_keys = self.x_keys + f.x_keys[1:]
        self.y_keys = self.y_keys + f.y_keys[1:]
        self.f = f
        self.x0 = x0

    def kernel(self, r: Tensor, *args: Tensor) -> Tensor:
        x0 = torch.full_like(r, self.x0)
        rmin = elementwise.minimize_newton(self.f.kernel, x0, args=args)
        mask = r > rmin
        r[mask] = rmin[mask]
        return self.f.kernel(r, *args)


class Piecewise(Radial):
    n = 2

    def __init__(
        self,
        f: Radial,
        g: Radial,
        radius: Tensor | Sequence[Sequence[Number]],
        *args,
        continuous: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.x_keys[0] == f.x_keys[0] == g.x_keys[0]:
            raise ValueError(f"First x_key of f and g must be {self.x_keys[0]}.")
        if not self.y_keys[0] == f.y_keys[0] == g.y_keys[0]:
            raise ValueError(f"First y_key of f and g must be {self.y_keys[0]}.")

        self.x_keys = self.x_keys + f.x_keys[1:] + g.x_keys[1:]
        self.y_keys = self.y_keys + f.y_keys[1:] + g.y_keys[1:]
        self.f = f
        self.g = g
        self.k = 2 * (f.n - 1)
        self.register_buffer("radius", torch.as_tensor(radius), persistent=False)
        self.continuous = continuous

    def kernel(self, r: Tensor, idx_x: Tensor, idx_y: Tensor, *args: Tensor) -> Tensor:
        radius = self.radius[idx_x, idx_y]
        args0, args1 = args[: self.k], args[self.k :]
        ratio = (
            self.f.kernel(radius, *args0) / self.g.kernel(radius, *args1)
            if self.continuous
            else 1
        )
        return torch.where(
            r < radius, self.f.kernel(r, *args0), ratio * self.g.kernel(r, *args1)
        )


class Tuning(Kernel):
    n = 2

    def __init__(self, kappa: Tensor | Sequence[Sequence[Number]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("kappa", torch.as_tensor(kappa), persistent=False)

    def kernel(
        self, x: PeriodicTensor, y: PeriodicTensor, idx_x: Tensor, idx_y: Tensor
    ) -> Tensor:
        theta = diff(x, y)
        return 1 + 2 * self.kappa[idx_x, idx_y] * torch.cos(
            theta.to_period(2 * torch.pi).norm(dim=-1)
        )


def radial(
    *,
    func: Callable[[ParameterFrame, ParameterFrame], Tensor] | None = None,
    kernel: Callable[[*tuple[Tensor, ...]], Tensor] | None = None,
    x_keys: Sequence[str] | str = (),
    y_keys: Sequence[str] | str | None = None,
    name: str = "CustomRadial",
    **kwargs,
) -> Radial:
    if bool(kernel) is bool(func):
        raise ValueError("Exactly one of func and kernel must be provided.")

    if kernel is None:

        def kernel_(self, r: Tensor, *args: Tensor) -> Tensor:
            x_keys, y_keys = self.x_keys, self.y_keys
            zeros, r_ = torch.zeros_like(r)[..., None], r[..., None]
            x = {x_keys[0]: zeros} | dict(zip(x_keys[1:], args[: len(x_keys[1:])]))
            y = {y_keys[0]: r_} | dict(zip(y_keys[1:], args[len(y_keys[1:]) :]))
            x, y = ParameterFrame(x, ndim=r.ndim), ParameterFrame(y, ndim=r.ndim)
            return func(x, y)

    else:

        def kernel_(self, *args: Tensor) -> Tensor:
            return kernel(*args)

    n = 1 if isinstance(x_keys, str) else len(x_keys)

    return type(name, (Radial,), {"kernel": kernel_, "n": n})(
        x_keys=x_keys, y_keys=y_keys, **kwargs
    )
