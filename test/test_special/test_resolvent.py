import numpy as np
from scipy.special import kv
from scipy.integrate import quad
import torch
import pytest

from niarb import special


def get_x(is_double):
    out = np.linspace(0, 5, num=10)
    if not is_double:
        out = out.astype(np.float32)
    return out, torch.from_numpy(out)


@pytest.mark.parametrize("d", list(range(1, 6)))
@pytest.mark.parametrize("is_double", [False, True])
@pytest.mark.parametrize(
    "l",
    [
        0.5,
        0.5 + 0.7j,
        np.linspace(0.1, 1, num=10).tolist(),
        np.linspace(0.1, 1, num=10)[:, None].tolist(),  # test broadcasting
        np.linspace(-1, 1, num=10).tolist(),  # test negative l
    ],
)
@pytest.mark.parametrize("dr", [0.0, 0.1])
def test_laplace_r(d, is_double, l, dr):
    l = np.array(l, dtype=np.float32) if isinstance(l, list) else l
    x_np, x_torch = get_x(is_double)
    if isinstance(l, np.ndarray) and (l < 0).any():
        s = (l + 0.0j) ** 0.5
    else:
        s = l**0.5

    prefactor = (2 * np.pi) ** (-d / 2)

    y_np = prefactor * (s / x_np) ** (d / 2 - 1) * kv(d / 2 - 1, s * x_np)
    if d == 1:
        y_np[..., 0] = (1 / (2 * s) * np.exp(-s * x_np))[..., 0]
    elif dr == 0.0:
        y_np[..., 0] = 0.0
    else:
        func = lambda r, s: r ** (d - 1) * (s / r) ** (d / 2 - 1) * kv(d / 2 - 1, s * r)
        if isinstance(s, np.ndarray):
            is_complex = s.dtype == np.complex64
            integral = np.array(
                [quad(func, 0, dr, args=(si,), complex_func=is_complex)[0] for si in s]
            )
            integral = integral.reshape(s.shape)[..., 0]
        else:
            is_complex = isinstance(s, complex)
            integral = quad(func, 0, dr, args=(s,), complex_func=is_complex)[0]
        y_np[..., 0] = (
            prefactor * special.solid_angle(d) * integral / special.ball_volume(d, dr)
        )

    if isinstance(l, np.ndarray):
        l = torch.from_numpy(l)
    y_torch = special.resolvent.laplace_r(d, l, x_torch, dr=dr)

    torch.testing.assert_close(
        torch.from_numpy(y_np), y_torch, equal_nan=True, rtol=5e-4, atol=1e-6
    )


@pytest.mark.parametrize("d", list(range(1, 6)))
@pytest.mark.parametrize(
    "l",
    [
        0.5,
        torch.linspace(0.1, 1, 10).tolist(),
        (torch.linspace(0.1, 1, 10) + 7.0j).tolist(),
        torch.linspace(0.1, 1, 10)[:, None].tolist(),  # test broadcasting
        torch.linspace(-1, 1, 10).tolist(),  # test negative l
    ],
)
@pytest.mark.parametrize(
    "x_requires_grad, l_requires_grad",
    [
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_laplace_r_grad(d, l, x_requires_grad, l_requires_grad):
    l = torch.tensor(l)
    l = l.to(torch.complex128 if torch.is_complex(l) else torch.double)
    _, x = get_x(True)

    if l_requires_grad:
        l.requires_grad = True
    if x_requires_grad:
        x.requires_grad = True
        x = x[1:]  # gradient w.r.t x at x = 0 is undefined
        l = l[1:] if l.ndim > 0 else l

    torch.autograd.gradcheck(special.resolvent.laplace_r, (d, l, x))


@pytest.mark.parametrize(
    "n, d, S, U, i, j, x",
    [
        (4, 3, (), (), (), (), ()),
        (4, 3, (), (), (5, 1), (1, 6), ()),
        (4, 3, (), (), (5, 1), (1, 6), (5, 6)),
        (4, 3, (5, 1), (1, 6), (), (), ()),
        (4, 3, (5, 1, 1, 2), (1, 1, 1, 2), (6, 1, 1), (1, 7, 1), (6, 7, 1)),
    ],
)
@pytest.mark.parametrize("l", [0, -1])
@pytest.mark.parametrize("diag", [True, False])
def test_mixture(n, d, S, U, l, i, j, x, diag):
    shape = torch.broadcast_shapes(S, U, i, j, x)

    S = torch.randn(*S, n, n)
    U = torch.randn(*U, n, n)
    V = torch.eye(n)
    i = torch.randint(n, size=i)
    j = torch.randint(n, size=j)
    x = torch.randn(*x, d)

    if diag:
        S = S.triu().tril()

    S = S @ S.transpose(-1, -2)  # positive definite
    U = -U @ U.transpose(-1, -2)  # negative definite

    R = special.resolvent.laplace
    out = special.resolvent.mixture(S, U, V, R, l, i, j, x).real

    Sinv = torch.linalg.inv(S)
    L, P = torch.linalg.eig(Sinv + l * V @ U)
    PinvV = torch.linalg.inv(P) @ V.to(P.dtype)
    UP = U.to(P.dtype) @ P

    L = L.broadcast_to(*shape, n)
    PinvV, UP = PinvV.broadcast_to(*shape, n, n), UP.broadcast_to(*shape, n, n)
    i, j = i.broadcast_to(shape), j.broadcast_to(shape)
    x = x.broadcast_to(*shape, d)

    expected = torch.empty(shape)
    for idx in np.ndindex(shape):
        expected[idx] = sum(
            UP[idx][i[idx], k] * PinvV[idx][k, j[idx]] * R(L[idx][k], x[idx])
            for k in range(n)
        ).real

    torch.testing.assert_close(out, expected)
