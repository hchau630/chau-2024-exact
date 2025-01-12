import abc
from collections.abc import Iterable

import torch

from niarb.cell_type import CellType
from niarb import nn

__all__ = ["Constraint", "StabilityCon"]


class Constraint(abc.ABC):
    def __init__(self, is_equality: bool):
        if not isinstance(is_equality, bool):
            raise ValueError(f"is_equality must be a bool, but {type(is_equality)=}.")

        super().__init__()
        self.is_equality = is_equality

    def __repr__(self):
        properties = [
            f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")
        ]
        return f'{type(self).__name__}({", ".join(properties)})'

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        pass


class StabilityCon(Constraint):
    def __init__(
        self,
        eps: float = 0.1,
        cell_types: Iterable[CellType | str] = None,
        stable: bool = True,
    ):
        super().__init__(is_equality=False)
        self.eps = eps
        self.cell_types = cell_types
        self.stable = stable

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        a = m.spectral_summary(cell_types=self.cell_types, kind="J").abscissa
        out = -a if self.stable else a
        return out - self.eps


class EISigmaDiagCon(Constraint):
    def __init__(self, eps: float = 0.0):
        super().__init__(is_equality=False)
        self.eps = eps

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        if len(m.cell_types) != 2:
            raise ValueError(
                f"model must have exactly 2 cell types, but got {m.cell_types=}."
            )

        S0 = torch.minimum(m.S[..., 0, 0], m.S[..., 1, 1])
        S1 = torch.maximum(m.S[..., 0, 1], m.S[..., 1, 0])
        return S0 - S1 - self.eps
