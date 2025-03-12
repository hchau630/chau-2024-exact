import torch
import pytest

from niarb import nn
from niarb.nn.modules import frame
from niarb.tensors import periodic
from niarb.special.resolvent import laplace_r


@pytest.fixture
def x():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
            "ori": periodic.tensor(
                [[-45.0], [0.0], [45.0], [90.0]], extents=[(-90.0, 90.0)]
            ),
            "cell_type": torch.tensor([0, 1, 1, 0]),
        }
    )


@pytest.fixture
def y():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0], [2.0], [1.0], [5.0]]),
            "ori": periodic.tensor(
                [[0.0], [90.0], [45.0], [-45.0]], extents=[(-90.0, 90.0)]
            ),
            "cell_type": torch.tensor([0, 0, 1, 1]),
        }
    )


def test_matrix(x, y):
    W = nn.Matrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "cell_type")
    out = W(x, y)
    expected = torch.tensor([1.0, 3.0, 4.0, 2.0])
    torch.testing.assert_close(out, expected)


def test_gaussian(x, y):
    sigma = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel = nn.Gaussian(sigma, ["space", "cell_type"])
    out = kernel(x, y)
    # (x - y)^2 = [0, 1, 1, 4], 2 * sigma^2 = [1, 3, 4, 2]
    expected = torch.exp(-torch.tensor([0, 1 / 3, 1 / 4, 2]))
    torch.testing.assert_close(out, expected)


def test_laplace(x, y):
    sigma = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel = nn.Laplace(2, sigma, ["space", "cell_type"])
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma = [1, 3, 4, 2]
    expected = torch.special.modified_bessel_k0(torch.tensor([0, 1 / 3, 1 / 4, 1]))
    expected[0] = 0.0
    expected = expected / (2 * torch.pi)
    torch.testing.assert_close(out, expected)


def test_monotonic(x, y):
    sigma1 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    sigma2 = torch.tensor([[1.5, 1.0], [0.5, 2.0]])
    keys = ["space", "cell_type"]
    kernel1 = nn.Laplace(2, sigma1, keys)
    kernel2 = nn.Gaussian(sigma2, keys)
    kernel = nn.Monotonic(nn.radial(func=kernel1 / kernel2, x_keys=keys), "space")
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma1 = [3, 1, 2, 4], sigma2 = [1.5, 0.5, 2, 1]
    expected = (kernel1 / kernel2)(x, y)
    expected[1] = 0.241606  # calculated with Mathematica
    expected[3] = 0.382324  # calculated with Mathematica

    torch.testing.assert_close(out, expected)


def test_monotonic_2(x, y):
    sigma1 = torch.tensor([[3.0, 1.5], [1.0, 2.0]])
    sigma2 = sigma1 / 2
    keys = ["space", "cell_type"]
    kernel1 = nn.Laplace(2, sigma1, keys)
    kernel2 = nn.Laplace(0, sigma2, keys)
    kernel = nn.Monotonic(nn.radial(func=kernel1 / kernel2, x_keys=keys), "space")
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma1 = [3, 1, 2, 1.5], sigma2 = [1.5, 0.5, 1, 0.75]
    expected = (kernel1 / kernel2)(x, y)
    expected[1] = 0.934132  # calculated with Mathematica
    expected[3] = 0.41517  # calculated with Mathematica

    torch.testing.assert_close(out, expected)


def test_radial(x, y):
    sigma = torch.tensor([[3.0, 1.5], [1.0, 2.0]])
    keys = ["space", "cell_type"]
    kernel1 = nn.Laplace(2, sigma, keys)
    kernel2 = nn.Laplace(0, sigma / 2, keys)
    kernel = nn.radial(func=kernel1 / kernel2, x_keys=keys)
    out = kernel(x, y)
    expected = (kernel1 / kernel2)(x, y)
    assert isinstance(kernel, nn.Radial)
    torch.testing.assert_close(out, expected)


def test_radial_2(x, y):
    kernel_ = lambda r: laplace_r(2, 1.0, r) / laplace_r(0, 0.5, r)
    kernel = nn.radial(kernel=kernel_, x_keys="space")
    out = kernel(x, y)
    expected = kernel_((x["space"] - y["space"]).norm(dim=-1))
    assert isinstance(kernel, nn.Radial)
    torch.testing.assert_close(out, expected)


def test_piecewise(x, y):
    sigma1 = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel1 = nn.Gaussian(sigma1, ["space", "cell_type"])
    sigma2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel2 = nn.Laplace(2, sigma2, ["space", "cell_type"])
    radius = torch.tensor([[0.5, 1.5], [1.5, 0.5]])
    kernel = nn.Piecewise(kernel1, kernel2, radius, ["space", "cell_type"])
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], 2 * sigma1^2 = [1, 3, 4, 2], sigma2 = [1, 3, 4, 2]
    # radius = [0.5, 1.5, 0.5, 1.5]
    ratio1 = torch.exp(-torch.tensor(0.5**2 / 4)) / torch.special.modified_bessel_k0(
        torch.tensor(0.5 / 4)
    )
    ratio2 = torch.exp(-torch.tensor(1.5**2 / 2)) / torch.special.modified_bessel_k0(
        torch.tensor(1.5 / 2)
    )
    expected = torch.tensor(
        [
            torch.exp(-torch.tensor(0)),
            torch.exp(-torch.tensor(1 / 3)),
            ratio1 * torch.special.modified_bessel_k0(torch.tensor(1 / 4)),
            ratio2 * torch.special.modified_bessel_k0(torch.tensor(1)),
        ]
    )
    torch.testing.assert_close(out, expected)


def test_tuning(x, y):
    kappa = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2
    kernel = nn.Tuning(kappa, ["ori", "cell_type"])
    out = kernel(x, y)
    # Δ ori = [45, 90, 0, 45], cos(Δ ori) = [0, -1, 1, 0]
    # 2 * kappa = [1, 3, 4, 2]
    expected = torch.tensor([1.0, -2.0, 5.0, 1.0])
    torch.testing.assert_close(out, expected)
