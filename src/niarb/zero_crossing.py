from functools import partial

import numpy as np
import torch

from niarb.special.resolvent import laplace_r


def func(d, r, l0, l1, z, **kwargs):
    return (laplace_r(d, l1, r, **kwargs) - z * laplace_r(d, l0, r, **kwargs)).real


def bisect(func, a, b, args=(), tol=1e-8):
    """
    Find a root of a function using the bisection method.

    Args:
        func: callable
        a: Tensor
        b: Tensor
        tol: float

    Returns:
        Tensor

    """
    if (b <= a).any():
        raise ValueError("b must be greater than a")

    a, b, *args = torch.broadcast_tensors(a, b, *args)
    out = torch.empty_like(a)
    fa, fb = func(a, *args), func(b, *args)
    valid = fa * fb < 0
    a, b, fa, fb = a[valid], b[valid], fa[valid], fb[valid]
    args = [arg[valid] for arg in args]
    while (b - a > tol).any():
        c = (a + b) / 2
        fc = func(c, *args)
        left = fa * fc < 0
        right = ~left
        b[left], fb[left] = c[left], fc[left]
        a[right], fa[right] = c[right], fc[right]
    out[valid] = (a + b) / 2
    return out


def find_root(d, l0, l1, z, n=1):
    """
    Find the nth zero crossing of the function
    $$G_d(r; \lambda_1) / G_d(r; \lambda_0) = z$$
    If l0 and l1 are real, l0 must be less than l1.

    Args:
        d: int
        l0: ndarray with shape (*)
        l1: ndrray with shape (*)
        z: ndarray with shape (**, *)
        n (optional): int

    Returns:
        ndarray with shape (**, *) of the nth zero crossing of the function

    """
    if l0.shape != l1.shape:
        raise ValueError("l0 and l1 must have the same shape")

    is_cplx = l0.imag != 0

    # if complex, l0 and l1 must be complex conjugates
    np.testing.assert_allclose(l0.imag[is_cplx], -l1.imag[is_cplx])
    np.testing.assert_allclose(l1.imag[~is_cplx], 0)

    # if real, l0 must be less than or equal to l1
    if not np.all(l0.real[~is_cplx] <= l1.real[~is_cplx]):
        raise ValueError("l0 must be less than or equal to l1")

    # z has unit norm if complex
    a = z.imag[..., ~is_cplx]
    np.testing.assert_allclose(a[~np.isnan(a)], 0)
    a = np.abs(z[..., is_cplx])
    np.testing.assert_allclose(a[~np.isnan(a)], 1)

    r = np.full_like(z, np.nan)

    if d == 1:
        z = z * np.sqrt(l1) / np.sqrt(l0)
        z[..., (l0.real <= 0) & (l0.imag == 0)] = np.nan

    if d in {1, 3}:
        if n == 1:
            one_cross = (z.imag == 0) & (z.real > 0) & (z.real < 1)
            r[one_cross] = (np.log(z) / (np.sqrt(l0) - np.sqrt(l1)))[one_cross]
        arg_z_num = (np.angle(z) % (2 * np.pi)) / 2
        r[..., is_cplx] = ((arg_z_num - n * np.pi) / np.sqrt(l0).imag)[..., is_cplx]
        r = r.real

    else:
        if d == 2:
            r1 = find_root(3, l0, l1, z, n=n)
            if n == 1:
                r0 = torch.full(r1.shape, 1e-5).double()
            else:
                r0 = find_root(3, l0, l1, z, n=n - 1)
        else:
            one_cross = (z.imag == 0) & (z.real > 0) & (z.real < 1)
            r0 = find_root(3, l0, l1, z, n=n)
            r1 = find_root(3, l0, l1, z, n=n + 1)
            r1[one_cross] = 2 * r0[one_cross]  # just a heuristic
        r0, r1 = torch.tensor(r0), torch.tensor(r1)
        l0, l1, z = torch.tensor(l0), torch.tensor(l1), torch.tensor(z)
        l0, l1 = l0.broadcast_to(z.shape), l1.broadcast_to(z.shape)
        r = torch.full_like(r0, torch.nan)
        valid = (r0 > 0) & (r1 > 0) & (r1 > r0)
        r[valid] = bisect(
            partial(func, d, validate_args=False),
            r0[valid],
            r1[valid],
            args=(l0[valid], l1[valid], z[valid]),
        )
        r = r.numpy()
    return r
