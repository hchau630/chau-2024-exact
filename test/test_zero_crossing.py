import torch
import numpy as np
import pytest

from niarb.special.resolvent import laplace_r
from niarb.zero_crossing import find_root


@pytest.mark.parametrize("d", list(range(1, 4)))
@pytest.mark.parametrize("n", [1, 2])
def test_find_root(d, n):
    l0 = np.array([0.5, 2.0, 1.5 - 1.0j, -1.5 - 1.0j])
    l1 = np.array([1.0, 3.0, 1.5 + 1.0j, -1.5 + 1.0j])
    z = np.array(
        [[0.5, 0.75, 1.0j, (1.0 + 0.1j) / (1.0 - 0.1j)], [0.25, 0.5, -1.0, -1.0j]]
    )
    out = find_root(d, l0, l1, z, n=n)
    assert out.shape == (2, 4)
    if n == 2:
        assert np.isnan(out[:, :2]).all()  # no second zero crossing
        out, l0, l1, z = out[:, 2:], l0[2:], l1[2:], z[:, 2:]
    l0, l1, out, z = map(torch.from_numpy, (l0, l1, out, z))
    torch.testing.assert_close(laplace_r(d, l1, out) / laplace_r(d, l0, out), z)
