import torch
import pytest

from niarb import nn
from niarb.nn.modules import frame
from niarb.tensors import periodic


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


def test_matrix_kernel(x, y):
    W = nn.MatrixKernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "cell_type")
    out = W(x, y)
    expected = torch.tensor([1.0, 3.0, 4.0, 2.0])
    torch.testing.assert_close(out, expected)


def test_gaussian_kernel(x, y):
    sigma = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel = nn.GaussianKernel(sigma, ["space", "cell_type"])
    out = kernel(x, y)
    # (x - y)^2 = [0, 1, 1, 4], 2 * sigma^2 = [1, 3, 4, 2]
    expected = torch.exp(-torch.tensor([0, 1 / 3, 1 / 4, 2]))
    torch.testing.assert_close(out, expected)


def test_laplace_kernel(x, y):
    sigma = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel = nn.LaplaceKernel(2, sigma, ["space", "cell_type"])
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma = [1, 3, 4, 2]
    expected = torch.special.modified_bessel_k0(torch.tensor([0, 1 / 3, 1 / 4, 1]))
    expected[0] = 0.0
    expected = expected / (2 * torch.pi)
    torch.testing.assert_close(out, expected)


def test_piecewise_kernel(x, y):
    sigma1 = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel1 = nn.GaussianKernel(sigma1)
    sigma2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel2 = nn.LaplaceKernel(2, sigma2)
    radius = torch.tensor([[0.5, 1.5], [1.5, 0.5]])
    kernel = nn.PiecewiseKernel(kernel1, kernel2, radius, ["space", "cell_type"])
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


def test_tuning_kernel(x, y):
    kappa = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2
    kernel = nn.TuningKernel(kappa, ["ori", "cell_type"])
    out = kernel(x, y)
    # Δ ori = [45, 90, 0, 45], cos(Δ ori) = [0, -1, 1, 0]
    # 2 * kappa = [1, 3, 4, 2]
    expected = torch.tensor([1.0, -2.0, 5.0, 1.0])
    torch.testing.assert_close(out, expected)
