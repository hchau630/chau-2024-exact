from collections.abc import Hashable, Callable

import torch
from torch import Tensor

__all__ = [
    "Identity",
    "Pow",
    "Compose",
    "Match",
]


class FunctionMixin:
    """
    A Mixin class that adds algebraic operations to torch.nn.Module
    """

    def __add__(self, g):
        return Add(self, g)

    def __radd__(self, g):
        return Add(self, g)

    def __sub__(self, g):
        return Sub(self, g)

    def __rsub__(self, g):
        return Sub(self, g)

    def __mul__(self, g):
        return Mul(self, g)

    def __rmul__(self, g):
        return Mul(self, g)

    def __truediv__(self, g):
        return TrueDiv(self, g)

    def __rtruediv__(self, g):
        return TrueDiv(self, g)

    def __pow__(self, p):
        return Compose(Pow(p), self)


class Function(FunctionMixin, torch.nn.Module):
    pass


class Identity(FunctionMixin, torch.nn.Identity):
    def inv(self):
        return Identity()


class Pow(Function):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return x**self.p

    def inv(self):
        return Pow(1 / self.p)


class BinOp(Function):
    def __init__(self, f, g):
        if not isinstance(f, torch.nn.Module):
            raise ValueError(
                f"f must be an instance of torch.nn.Module, but {type(f)=}."
            )

        super().__init__()
        self.f = f
        self.g = g


class Add(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) + self.g(*args, **kwargs)


class Sub(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) - self.g(*args, **kwargs)


class Mul(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) * self.g(*args, **kwargs)


class TrueDiv(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) / self.g(*args, **kwargs)


class Compose(Function):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = torch.nn.ModuleList(args)
        self.kwargs = torch.nn.ModuleDict(kwargs)

    def forward(self, *args, **kwargs):
        return self.f(
            *[v(*args, **kwargs) for v in self.args],
            **{k: v(*args, **kwargs) for k, v in self.kwargs.items()},
        )

    def inv(self):
        if len(self.args) != 1 and len(self.kwargs) != 0:
            raise NotImplementedError()

        return Compose(self.args[0].inv(), self.f.inv())


class Match(Function):
    def __init__(
        self,
        cases: dict[Hashable, Callable[[Tensor], Tensor]],
        default: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        self.cases = cases
        self.default = default

    def forward(self, key: Tensor, x: Tensor) -> Tensor:
        key, x = torch.broadcast_tensors(key, x)
        out = self.default(x)
        for k, v in self.cases.items():
            mask = key == k
            out[mask] = v(x[mask])
        return out
