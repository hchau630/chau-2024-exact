from functools import partial
from typing import overload, Literal

import numpy as np
from numpy.typing import ArrayLike
import torch

from niarb.optimize.elementwise import bisect
from niarb.special.resolvent import laplace_r


def func(d, r, l0, l1, z, **kwargs):
    return (laplace_r(d, l1, r, **kwargs) - z * laplace_r(d, l0, r, **kwargs)).real


def find_root(
    d: int, l0: np.ndarray, l1: np.ndarray, z: np.ndarray, n: int = 1
) -> np.ndarray:
    """
    Find the nth zero crossing of the function
    $$G_d(r; \lambda_1) / G_d(r; \lambda_0) = z$$
    If l0 and l1 are real, l0 must be less than l1.

    Args:
        d: Number of spatial dimensions
        l0: ndarray with shape (*)
        l1: ndrray with shape (*)
        z: ndarray with shape (**, *)
        n (optional): Which zero crossing to find

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


@overload
def find_n_crossings(
    x: ArrayLike,
    y: ArrayLike,
    n: int = ...,
    return_indices: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def find_n_crossings(
    x: ArrayLike,
    y: ArrayLike,
    n: int = ...,
    return_indices: Literal[True] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


def _sign_change(x: ArrayLike) -> np.ndarray:
    """Return a mask of sign changes in x along the last dimension.

    This is a surprisingly annoying problem. Solutions found using stackexchange
    all fail in different edge cases. For example, solutions like
    `np.diff(np.signbit(x), axis=-1) != 0`
    fail on inputs like [-1, 0, -1]. The key idea of this implementation is to
    modify np.sign(x) by replacing values of 0 with the sign of the last nonzero value.

    Args:
        x: ndarray with shape (*, N)

    Returns:
        ndarray with shape (*, N - 1) of sign changes in x.

    """
    x = np.sign(np.asarray(x))

    idx = np.broadcast_to(np.arange(x.shape[-1]), x.shape)
    idx = np.where(x == 0, -1, idx)
    idx = np.maximum(np.maximum.accumulate(idx, axis=-1), 0)

    x = np.take_along_axis(x, idx, axis=-1)
    return (x[..., :-1] != x[..., 1:]) & (x[..., :-1] != 0)


def find_n_crossings(
    x: ArrayLike, y: ArrayLike, n: int = 1, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Args:
        x: array-like with shape (N)
        y: array-like with shape (*, N)
        n (optional): Find the first n zero crossings. If n == -1, find all zero
          crossings.
        return_indices (optional): If True, return the indices of the zero crossings.

    Returns:
        ndarray with shape (n, *) of the first n zero crossings of the function.
        If return_indices is True, also return a ndarray with shape (n, *) of the
        indices of the zero crossings.

    """
    # if n < 1 and n != -1:
    #     raise ValueError("n must be -1 or greater than or equal to 1")

    if n < 1:
        raise ValueError("n must greater than or equal to 1")

    x, y = np.asarray(x), np.asarray(y)

    # get midpoints
    mid = (x[1:] + x[:-1]) / 2  # (N - 1,)

    idx, indices, crossings = np.empty(y.shape[:-1], dtype=np.long), [], []
    while len(crossings) < n:
        if len(crossings) == 0:
            # get mask of where y changes sign
            mask = _sign_change(y).astype(np.long)  # (*, N - 1)

        else:
            # set the mask to zero at the index of nonzero element found
            np.put_along_axis(mask, idx[..., None], 0, axis=-1)  # (*, N - 1)

        # find the next index of nonzero element by using the fact that argmax returns
        # the first occurrence of the maximum value. If there are no nonzero elements,
        # then idx is 0.
        idx = mask.argmax(axis=-1)  # (*,)

        # mask of whether the nth crossing exists
        has_crossing = mask.any(axis=-1)  # (*,)

        # get nth crossing if it exists, else NaN
        crossing = np.where(has_crossing, mid[idx], np.nan)  # (*,)

        indices.append(idx)
        crossings.append(crossing)

        # if len(crossings) == n or (n == -1 and not has_crossing.any()):
        #     break

    if return_indices:
        return np.stack(crossings), np.stack(indices)
    return np.stack(crossings)
