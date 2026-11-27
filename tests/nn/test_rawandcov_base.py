"""Tests for the two-stream (covariance + raw data) SPDnet layers."""

import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.nn.rawandcov_base as nn_rawandcov
from utils import is_orthogonal, is_spd
from yetanotherspdnet.nn.base import BiMap, LogEig, ReEig


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
    """Random raw data X and its covariance Sigma = X X^T / n ."""
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


# ----------------
# ReEigTwoStream
# ----------------
class TestReEigTwoStream:
    """
    Test suite for the ReEigTwoStream module
    """

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    def test_initialization(self, mode):
        """No parameter, attributes and representation."""
        layer = nn_rawandcov.ReEigTwoStream(eps=1e-3, mode=mode, dim=10)
        assert layer.eps == 1e-3
        assert layer.mode == mode
        assert len(list(layer.parameters())) == 0
        assert repr(layer).startswith("ReEigTwoStream(")

    def test_invalid_mode(self):
        """Unknown mode is not allowed."""
        with pytest.raises(AssertionError):
            nn_rawandcov.ReEigTwoStream(mode="unknown")

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    @pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
    def test_forward_backward(self, mode, batch_shape, device, dtype, generator):
        """Forward (shapes, SPD, same SPD stream as ReEig) and gradients."""
        n_features, n_samples, eps = 10, 30, 0.5
        layer = nn_rawandcov.ReEigTwoStream(eps=eps, mode=mode)
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        Q, V = layer(Sigma, X)
        assert Q.shape == Sigma.shape
        assert V.shape == X.shape
        assert is_spd(Q)
        assert_close(Q, ReEig(eps=eps)(Sigma))

        (Q.sum() + V.pow(2).sum()).backward()
        assert Sigma.grad is not None and Sigma.grad.shape == Sigma.shape
        assert X.grad is not None and X.grad.shape == X.shape

    def test_bimap_then_reeig(self, device, dtype, generator):
        """The two-stream layers can be chained: BiMap -> ReEig."""
        n_in, n_out, n_samples = 20, 8, 50
        bimap = nn_rawandcov.BiMapTwoStream(
            n_in, n_out, device=device, dtype=dtype, generator=generator
        )
        reeig = nn_rawandcov.ReEigTwoStream(eps=0.5, mode="whitening_Q")
        X, Sigma = _raw_and_cov((4,), n_in, n_samples, device, dtype, generator)

        Q, V = reeig(*bimap(Sigma, X))
        assert Q.shape == (4, n_out, n_out)
        assert V.shape == (4, n_out, n_samples)
        (Q.sum() + V.pow(2).sum()).backward()
        assert next(bimap.parameters()).grad is not None


# -----------------
# LogEigTwoStream and VecTwoStream
# -----------------
LOGEIG_LAYERS = {
    "version_1": nn_rawandcov.LogEigTwoStream,
    "version_2": nn_rawandcov.LogEigWhitenedTwoStream,
}


class TestLogEigTwoStream:
    """
    Test suite for the two-stream LogEig modules (versions 1 and 2)
    """

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    def test_initialization(self, version):
        """No parameter, attributes and representation."""
        layer = LOGEIG_LAYERS[version](dim=10)
        assert layer.dim == 10
        assert len(list(layer.parameters())) == 0
        assert repr(layer).startswith(type(layer).__name__ + "(")

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    @pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
    def test_forward_backward(self, version, batch_shape, device, dtype, generator):
        """Shapes, same SPD stream as the LogEig, and gradients."""
        n_features, n_samples = 10, 30
        layer = LOGEIG_LAYERS[version]()
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        L, W = layer(Sigma, X)
        assert L.shape == Sigma.shape
        assert W.shape == X.shape
        assert_close(L, LogEig()(Sigma))

        (torch.norm(L) + torch.norm(W)).backward()
        assert Sigma.grad is not None and Sigma.grad.shape == Sigma.shape
        assert X.grad is not None and X.grad.shape == X.shape


class TestVecTwoStream:
    """
    Test suite for the two-stream vectorization module
    """

    def test_initialization(self):
        """No parameter and representation."""
        layer = nn_rawandcov.VecTwoStream()
        assert len(list(layer.parameters())) == 0
        assert repr(layer) == "VecTwoStream()"

    @pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
    def test_forward_backward(self, batch_shape, device, dtype, generator):
        """Output shape and gradients."""
        n_features, n_samples = 10, 30
        layer = nn_rawandcov.VecTwoStream()
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        L = LogEig()(Sigma).detach().requires_grad_(True)
        X.requires_grad_(True)

        z = layer(L, X)
        n_coord = n_features * (n_features + 1) // 2
        n_out = n_coord + n_features * n_samples
        assert z.shape == (*batch_shape, n_out)

        torch.norm(z).backward()
        assert L.grad is not None and L.grad.shape == L.shape
        assert X.grad is not None and X.grad.shape == X.shape

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    def test_end_of_network(self, version, device, dtype, generator):
        """BiMap -> ReEig -> LogEig -> Vec, for versions 1 and 2."""
        n_in, n_out, n_samples = 20, 8, 50
        layers = [
            nn_rawandcov.BiMapTwoStream(
                n_in, n_out, device=device, dtype=dtype, generator=generator
            ),
            nn_rawandcov.ReEigTwoStream(eps=0.5, mode="whitening_Q"),
            LOGEIG_LAYERS[version](),
            nn_rawandcov.VecTwoStream(),
        ]
        X, Sigma = _raw_and_cov((4,), n_in, n_samples, device, dtype, generator)

        data = (Sigma, X)
        for layer in layers:
            data = layer(*data)
        z = data

        n_coord = n_out * (n_out + 1) // 2
        assert z.shape == (4, n_coord + n_out * n_samples)
        torch.norm(z).backward()
        assert next(layers[0].parameters()).grad is not None


# -----------------
# MixingTwoStream
# -----------------
class TestMixingTwoStream:
    """
    Test suite for the MixingTwoStream module
    """

    def test_initialization(self):
        """No parameter, attributes and representation."""
        layer = nn_rawandcov.MixingTwoStream(alpha=0.3, dim=10)
        assert layer.alpha == 0.3
        assert len(list(layer.parameters())) == 0
        assert repr(layer).startswith("MixingTwoStream(")

    def test_invalid_alpha(self):
        """alpha outside (0, 1] is not allowed."""
        with pytest.raises(AssertionError):
            nn_rawandcov.MixingTwoStream(alpha=0.0)
        with pytest.raises(AssertionError):
            nn_rawandcov.MixingTwoStream(alpha=1.5)

    @pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
    def test_forward_backward(self, batch_shape, device, dtype, generator):
        """Shapes, P is SPD, Y = X, and gradients."""
        n_features, n_samples = 10, 30
        layer = nn_rawandcov.MixingTwoStream(alpha=0.3)
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        P, Y = layer(Sigma, X)
        assert P.shape == Sigma.shape
        assert Y.shape == X.shape
        assert is_spd(P)
        assert_close(Y, X)

        (torch.norm(P) + torch.norm(Y)).backward()
        assert Sigma.grad is not None and Sigma.grad.shape == Sigma.shape
        assert X.grad is not None and X.grad.shape == X.shape

    def test_restores_link_after_reeig(self, device, dtype, generator):
        """After ReEig, the Mixing brings P closer to the covariance of Y."""
        n_features, n_samples = 10, 30
        reeig = nn_rawandcov.ReEigTwoStream(eps=0.5, mode="whitening_Q")
        mixing = nn_rawandcov.MixingTwoStream(alpha=0.3)
        X, Sigma = _raw_and_cov((4,), n_features, n_samples, device, dtype, generator)

        Q, V = reeig(Sigma, X)
        P, Y = mixing(Q, V)
        cov_V = V @ V.transpose(-1, -2) / n_samples
        # distance to the covariance of the data, before and after the Mixing
        assert torch.norm(P - cov_V) < torch.norm(Q - cov_V)

    def test_full_two_stream_network(self, device, dtype, generator):
        """BiMap -> ReEig -> Mixing -> BiMap -> ReEig -> LogEig."""
        n_in, n_mid, n_out, n_samples = 20, 12, 6, 50
        layers = [
            nn_rawandcov.BiMapTwoStream(
                n_in, n_mid, device=device, dtype=dtype, generator=generator
            ),
            nn_rawandcov.ReEigTwoStream(eps=0.1, mode="whitening_Q"),
            nn_rawandcov.MixingTwoStream(alpha=0.5),
            nn_rawandcov.BiMapTwoStream(
                n_mid, n_out, device=device, dtype=dtype, generator=generator
            ),
            nn_rawandcov.ReEigTwoStream(eps=0.1, mode="whitening_Q"),
        ]
        logeig = nn_rawandcov.LogEigTwoStream()
        vectorization = nn_rawandcov.VecTwoStream()
        X, Sigma = _raw_and_cov((4,), n_in, n_samples, device, dtype, generator)

        data = (Sigma, X)
        for layer in layers:
            data = layer(*data)
        z = vectorization(*logeig(*data))
        assert z.shape == (4, n_out * (n_out + 1) // 2 + n_out * n_samples)

        torch.norm(z).backward()
        for layer in layers:
            for param in layer.parameters():
                assert param.grad is not None
