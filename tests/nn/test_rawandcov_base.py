"""Tests for the two-stream (covariance + raw data) SPDnet layers."""

import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.nn.rawandcov_base as nn_rawandcov
from utils import is_orthogonal, is_spd
from yetanotherspdnet.nn.base import BiMap


# ----------
# Fixtures
# ----------
@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="module")
def seed():
    return 777


@pytest.fixture(scope="function")
def generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _raw_and_cov(batch_shape, n_in, n_samples, device, dtype, generator):
    """Random raw data X and its covariance Sigma = X X^T / n (eq. 1)."""
    X = torch.randn(
        (*batch_shape, n_in, n_samples), device=device, dtype=dtype, generator=generator
    )
    Sigma = X @ X.transpose(-1, -2) / n_samples
    return X, Sigma


# ----------------
# BiMapTwoStream
# ----------------
class TestBiMapTwoStream:
    """
    Test suite for the BiMapTwoStream module
    """

    @pytest.mark.parametrize("n_in, n_out", [(20, 8), (10, 10)])
    @pytest.mark.parametrize("parametrized", [True, False])
    def test_initialization(self, n_in, n_out, parametrized, device, dtype, generator):
        """W has the right shape, is on the Stiefel manifold, single parameter."""
        layer = nn_rawandcov.BiMapTwoStream(
            n_in,
            n_out,
            parametrized=parametrized,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        assert layer.weight.shape == (n_in, n_out)
        assert layer.weight.dtype == dtype
        assert layer.weight.device.type == device.type
        assert is_orthogonal(layer.weight)
        assert repr(layer).startswith("BiMapTwoStream(")
        # a single trainable tensor, shared by both streams
        assert len(list(layer.parameters())) == 1

    def test_n_out_larger_than_n_in(self, device, dtype):
        """n_out > n_in is not allowed."""
        with pytest.raises(AssertionError):
            nn_rawandcov.BiMapTwoStream(4, 10, device=device, dtype=dtype)

    @pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
    @pytest.mark.parametrize("parametrized", [True, False])
    def test_forward_backward(
        self, batch_shape, parametrized, device, dtype, generator
    ):
        """Forward (shapes, SPD, link) and gradients reach both inputs and W."""
        n_in, n_out, n_samples = 20, 8, 50
        layer = nn_rawandcov.BiMapTwoStream(
            n_in,
            n_out,
            parametrized=parametrized,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X, Sigma = _raw_and_cov(batch_shape, n_in, n_samples, device, dtype, generator)
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        P, Y = layer(Sigma, X)
        assert P.shape == (*batch_shape, n_out, n_out)
        assert Y.shape == (*batch_shape, n_out, n_samples)
        assert is_spd(P)
        assert_close(P, Y @ Y.transpose(-1, -2) / n_samples)

        (P.sum() + Y.pow(2).sum()).backward()
        assert Sigma.grad is not None and Sigma.grad.shape == Sigma.shape
        assert X.grad is not None and X.grad.shape == X.shape
        param = next(layer.parameters())
        assert param.grad is not None

    def test_stays_on_stiefel_after_training(self, device, dtype, generator):
        """With the parametrization, W stays orthogonal after optimizer steps."""
        n_in, n_out, n_samples = 20, 8, 50
        layer = nn_rawandcov.BiMapTwoStream(
            n_in, n_out, device=device, dtype=dtype, generator=generator
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=1e-1)
        X, Sigma = _raw_and_cov((6,), n_in, n_samples, device, dtype, generator)
        for _ in range(5):
            optimizer.zero_grad()
            P, Y = layer(Sigma, X)
            (P.sum() + Y.pow(2).sum()).backward()
            optimizer.step()
        assert is_orthogonal(layer.weight)
        P, Y = layer(Sigma, X)
        assert_close(P, Y @ Y.transpose(-1, -2) / n_samples)

    def test_same_as_bimap(self, device, dtype):
        """With the same seed, the SPD stream equals the classical BiMap."""
        g1 = torch.Generator(device=device).manual_seed(1)
        g2 = torch.Generator(device=device).manual_seed(1)
        bimap = BiMap(10, 4, device=device, dtype=dtype, generator=g1)
        two = nn_rawandcov.BiMapTwoStream(
            10, 4, device=device, dtype=dtype, generator=g2
        )
        X, Sigma = _raw_and_cov((3,), 10, 30, device, dtype, None)
        assert_close(bimap(Sigma), two(Sigma, X)[0])
