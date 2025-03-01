import torch
import pytest

from niarb import nn
from niarb.nn.modules import frame
from niarb.tensors import periodic


@pytest.fixture
def x():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([0.0, 1.0, 2.0, 3.0]),
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
            "space": torch.tensor([0.0, 2.0, 1.0, 5.0]),
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


def test_tuning_kernel(x, y):
    kappa = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2
    kernel = nn.TuningKernel(kappa, ["ori", "cell_type"])
    out = kernel(x, y)
    # Δ ori = [45, 90, 0, 45], cos(Δ ori) = [0, -1, 1, 0]
    # 2 * kappa = [1, 3, 4, 2]
    expected = torch.tensor([1.0, -2.0, 5.0, 1.0])
    torch.testing.assert_close(out, expected)
