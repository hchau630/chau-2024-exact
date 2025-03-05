import contextlib
from numbers import Number
from collections.abc import Callable
from typing import Concatenate

import torch
from torch import Tensor

from .core import yukawa, k0, kd, scaled_kd, irkd
from niarb.linalg import is_diagonal
from niarb.utils import take_along_dims


# @profile
def laplace_r(
    d: int,
    l: Number | Tensor,
    r: Tensor,
    dr: float | Tensor = 0.0,
    validate_args: bool = True,
) -> Tensor:
    r"""Radial component of the kernel of the resolvent of the Laplacian.

    Radial component of the kernel of the resolvent of the
    laplace operator in d dimensions, i.e.
    $\bra{x}(l - \nabla^2)^{-1}\ket{y} = laplace(d, l, ||x - y||)$
    but with the modification that the output is 0 when d > 1 and r == 0,
    due to the fact that the resolvent diverges when d > 1 and r == 0.
    Explicitly, the equation is given by
    $1 / (2\pi)^{d/2} (\sqrt{l} / r)^{d/2 - 1} K_{d/2 - 1}(\sqrt{l}r)$
    The resolvent of the Laplacian is only well-defined for
    $\lambda \in \mathbb{C} \ (-\infty, 0]$, but here we will just plug in negative
    $\lambda$ into the above expression directly.

    Args:
        d: Dimension of the space.
        l: Parameter $\lambda$.
        r: Tensor of distances, must be non-negative and real.
        dr (optional): Small positive number to avoid singularity at r = 0.
        validate_args (optional): Whether to validate the arguments.

    Returns:
        Tensor with shape broadcast(*, **)

    """
    if validate_args:
        if not isinstance(d, int):
            raise ValueError(f"d must be an integer, but {d=}.")

        if r.is_complex():
            raise TypeError(f"r must be real, but {r.dtype=}.")

        if r.requires_grad and (r == 0).any():
            raise NotImplementedError(
                "Backpropagtion with respect to r when r contains 0 is currently untested."
            )

        if not (r >= 0).all():
            raise ValueError(f"r must be non-negative, but {r.min()=}.")

    if isinstance(l, torch.Tensor) and not l.is_complex() and (l < 0).any():
        # cast real tensors with negative elements to corresponding complex dtype
        # so that the square root operation does not result in NaNs.
        dtypes = {
            torch.float16: torch.complex32,
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }
        l = l.to(dtypes[l.dtype])

    s = l**0.5

    if d <= 0:
        # kd is faster for even dimensions and scaled_kd is faster for odd dimensions
        # WARNING: current implementation is incomplete - it returns NaN instead of a
        # finite number for r = 0.
        if d % 2 == 0:
            out = (r / s) ** int(1 - d / 2) * kd(d, s * r)
        else:
            out = s ** int((d - 3) / 2) * r ** int((1 - d) / 2) * scaled_kd(d, s * r)
        return (2 * torch.pi) ** (-d / 2) * out

    if d == 1:
        # need to treat the 1D case separately since the limit at r = 0 is finite,
        # but the more general approach below will lead to torch.nan at r = 0.
        # the 1D case equation is 1 / (2 \sqrt{\lambda}) e^{-\sqrt{\lambda}r)
        return 1 / (2 * s) * torch.exp(-s * r)

    if not isinstance(dr, float) or dr != 0.0:
        dr = torch.as_tensor(dr)
        singularity = d * irkd(d, s * dr) / (dr**d * l)
    else:
        singularity = 0.0

    if d == 2:
        return k0(s * r, singularity=singularity) / (2 * torch.pi)

    if d == 3:
        # faster computation for 3D case
        singularity = (2 / torch.pi) ** 0.5 * singularity
        return 1 / (4 * torch.pi) * yukawa(torch.as_tensor(s), r, singularity)

    requires_grad = (isinstance(s, torch.Tensor) and s.requires_grad) or r.requires_grad
    if requires_grad:
        # all the masking is needed in lieu of torch.where due to NaN gradients for
        # r = 0 entries, see https://github.com/pytorch/pytorch/issues/68425
        # for an explanation of the issue of NaN gradient propagation in pytorch
        if isinstance(s, torch.Tensor):
            s, r = torch.broadcast_tensors(s, r)
        out = singularity * torch.ones(
            r.shape, dtype=torch.result_type(s, r), device=r.device
        )
        mask = r != 0
        r = r[mask]  # Bessel function diverges at r = 0 for d > 1.
        if isinstance(s, torch.Tensor):
            s = s[mask]

    # kd is faster for even dimensions and scaled_kd is faster for odd dimensions
    if d % 2 == 0:
        tmp = (s / r) ** int(d / 2 - 1) * kd(d, s * r)
    else:
        tmp = s ** int((d - 3) / 2) * r ** int((1 - d) / 2) * scaled_kd(d, s * r)

    if requires_grad:
        out[mask] = tmp
    else:
        out = torch.where(r != 0, tmp, singularity)

    return (2 * torch.pi) ** (-d / 2) * out


def laplace(l: Number | Tensor, r: Tensor, **kwargs) -> Tensor:
    r"""Resolvent of the Laplacian, $R(\lambda; \nabla^2)$.

    Args:
        l: Parameter $\lambda$.
        r: Tensor with shape (*, d). Must be non-negative and real.

    Returns:
        Tensor with shape broadcast(l.shape, r.shape

    """
    return laplace_r(r.shape[-1], l, r.norm(dim=-1), **kwargs)


def laplace_beltrami(g, l, r, **kwargs):
    L, V = torch.linalg.eig(g)
    sqrt_g = V @ (L**0.5) @ torch.linalg.inv(V)
    r = sqrt_g @ r
    return torch.linalg.det(g) ** 0.5 * laplace_r(
        r.shape[-1], l, r.norm(dim=-1), **kwargs
    )


# @profile
def mixture(
    S: Tensor,
    U: Tensor,
    V: Tensor,
    R: Callable[Concatenate[Tensor, ...], Tensor],
    l: Number | Tensor,
    i: Tensor,
    j: Tensor,
    *args,
) -> Tensor:
    r"""Compute a mixture of resolvents.

    Computes $UP(l)R(L(l); D)P(l)^{-1}V$ where $P(l)L(l)P(l)^{-1}$ is the
    eigendecomposition of $S^{-1} + lVU$.

    Args:
        S: Tensor with shape (*S, m, m)
        U: Tensor with shape (*U, n, m)
        V: Tensor with shape (*V, m, n)
        R: Resolvent of D, which takes l as its first argument.
        l: Number or tensor with shape (*l, 1 | n, 1 | m)
        i: Tensor with dtype torch.long with shape i
        j: Tensor with dtype torch.long with shape j
        args: Remaining arguments to R, with shapes (*a, ?)

    Returns:
        Tensor with shape SUVlija

    """
    m = S.shape[-1]

    Sinv = torch.linalg.inv(S)  # (*S, m, m)
    if (
        (isinstance(l, Number) and l == 0) or (isinstance(l, Tensor) and (l == 0).all())
    ) and is_diagonal(Sinv):
        # sometimes linalg.eig cause backprop issues, so avoid it if possible
        L = Sinv.diagonal(dim1=-2, dim2=-1)  # (*S, m)
        P = torch.eye(m, dtype=U.dtype, device=U.device)  # (m, m)
    else:
        L, P = torch.linalg.eig(Sinv + V @ (l * U))  # (*SUVl, m), (*SUVl, m, m)
    PinvV = torch.linalg.inv(P) @ V.to(P.dtype)  # (*SUVl, m, n)
    UP = U.to(P.dtype) @ P  # (*SUVl, n, m)

    # reduce memory usage by half if real
    # TODO: Also reduce memory usage by half if complex by taking advantage of
    # conjugate symmetry in diagonalization of real matrices
    if L.isreal().all():
        L, PinvV, UP = L.real, PinvV.real, UP.real

    if (flag := is_diagonal(UP)) or is_diagonal(PinvV):
        # faster path for special case where either UP or PinvV is diagonal
        # (occurs when computing the weight matrix)
        A = PinvV if flag else UP  # (*SUVl, n, n)
        A = take_along_dims(A, i[..., None, None], j[..., None, None])  # SUVlij

        B, i = (UP, i) if flag else (PinvV, j)  # (*SUVl, n, n), i or j
        B = B.diagonal(dim1=-2, dim2=-1)  # (*SUVl, n)
        B = take_along_dims(B, i[..., None])  # SUVli or SUVlj
        L = take_along_dims(L, i[..., None])  # SUVli or SUVlj

        out = R(L, *args)  # SUVlia or SUVlja
        out = A * B * out  # SUVlija

    else:
        UP = take_along_dims(UP, i[..., None, None], dims=(-2,))  # (*SUVli, m)
        PinvV = take_along_dims(PinvV, j[..., None, None], dims=(-1,))  # (*SUVlj, m)

        # separate into two lines for easy line-by-line profiling
        out = [R(L[..., i], *args) for i in range(m)]
        out = sum(UP[..., i] * PinvV[..., i] * out[i] for i in range(m))  # SUVlija
        # out = sum(UP[..., i] * PinvV[..., i] * R(L[..., i], *args) for i in range(m))  # SUVlija
    return out

