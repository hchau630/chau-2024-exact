import torch
import numpy as np
import pytest

from niarb.special.resolvent import laplace_r
from niarb.zero_crossing import find_root, find_n_crossings


@pytest.mark.parametrize("d", list(range(1, 4)))
@pytest.mark.parametrize("n", [1, 2])
def test_find_root(d, n):
    l0 = np.array([0.5, 2.0, 1.5 - 1.0j, -1.5 - 1.0j])
    l1 = np.array([1.0, 3.0, 1.5 + 1.0j, -1.5 + 1.0j])
    z = np.array(
        [
            [0.5, 0.75, 1.0j, (1.0 + 0.1j) / (1.0 - 0.1j)],
            [0.25, 0.5, -1.0, -1.0j],
        ]
    )
    out = find_root(d, l0, l1, z, n=n)
    if d == 2:
        # Expected output obtained by plotting and using FindRoot in Mathematica
        if n == 1:
            expected = np.array(
                [
                    [1.83753, 0.631435, 5.6888, 1.90202],
                    [4.17514, 1.88276, 3.67533, 0.240215],
                ]
            )
        else:
            expected = np.array(
                [
                    [np.nan, np.nan, 13.757, 4.33016],
                    [np.nan, np.nan, 11.7392, 2.58333],
                ]
            )
        np.testing.assert_allclose(out, expected, equal_nan=True, rtol=1e-5)
    elif d in {1, 3}:
        # analytical result
        zz = z if d == 3 else z * np.sqrt(l1) / np.sqrt(l0)
        expected = np.full((2, 4), np.nan)
        if n == 1:
            expected[:, :2] = (np.log(zz) / (np.sqrt(l0) - np.sqrt(l1)))[:, :2]
        if d == 1:
            expected[:, 2:] = np.array([[5.300098, 1.373574], [3.281525, 2.06231]])
        else:
            expected[:, 2:] = np.array([[6.055718, 2.367139], [4.037146, 0.611175]])
        if n == 2:
            denom = 2 * np.sqrt(l0).imag
            expected[:, 2:] = expected[:, 2:] - 2 * np.pi / denom[2:]
        np.testing.assert_allclose(out, expected, equal_nan=True, rtol=1e-5)
    else:
        # Too lazy to manually compute the expected output for d=4, so just do
        # some basic checks.
        assert out.shape == (2, 4)
        if n == 2:
            assert np.isnan(out[:, :2]).all()  # no second zero crossing
            out, l0, l1, z = out[:, 2:], l0[2:], l1[2:], z[:, 2:]
        l0, l1, out, z = map(torch.from_numpy, (l0, l1, out, z))
        torch.testing.assert_close(laplace_r(d, l1, out) / laplace_r(d, l0, out), z)


def test_find_n_crossings():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )
    expected = np.array(
        [
            [np.nan, np.nan, np.nan, 1.5, 1.5, 0.5, 0.5, 1.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.5, np.nan],
        ]
    )
    out = find_n_crossings(x, y, n=2)
    np.testing.assert_allclose(out, expected, equal_nan=True)
