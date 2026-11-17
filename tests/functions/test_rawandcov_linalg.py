"""Tests for the two-stream (covariance + raw data) linear algebra functions."""

import pytest
import torch
from torch.testing import assert_close

from utils import is_spd
from yetanotherspdnet.functions import rawandcov_linalg
from yetanotherspdnet.functions.spd_linalg import congruence_fullrank, symmetrize
from yetanotherspdnet.random.stiefel import _init_weights_stiefel


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


def _stiefel(n_in, n_out, device, dtype, generator):
    """Random weight W of shape (n_in, n_out) on the Stiefel manifold."""
    W = torch.empty((n_in, n_out), device=device, dtype=dtype)
    _init_weights_stiefel(W, generator=generator)
    return W


# ------------------------------------------
# Two-stream BiMap: congruence + projection
# ------------------------------------------
class TestCongruenceAndProjection:
    """
    Test suite for the two-stream BiMap function (congruence + projection)
    """

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    @pytest.mark.parametrize("n_in, n_out, n_samples", [(20, 8, 50), (10, 10, 30)])
    def test_forward(
        self, batch_shape, n_in, n_out, n_samples, device, dtype, generator
    ):
        """Shapes, P is SPD, and the link P = Y Y^T / n is preserved (eq. 5)."""
        X, Sigma = _raw_and_cov(batch_shape, n_in, n_samples, device, dtype, generator)
        W = _stiefel(n_in, n_out, device, dtype, generator)

        P, Y = rawandcov_linalg.CongruenceAndProjection.apply(Sigma, X, W)
        P_ref, Y_ref = rawandcov_linalg.congruence_and_projection(Sigma, X, W)

        # shapes
        assert P.shape == (*batch_shape, n_out, n_out)
        assert Y.shape == (*batch_shape, n_out, n_samples)
        # P stays SPD
        assert is_spd(P)
        # the class and the function give the same result
        assert_close(P, P_ref)
        assert_close(Y, Y_ref)
        # SPD stream: same as the classical BiMap
        assert_close(P, congruence_fullrank(Sigma, W))
        # link between both streams: P = Y Y^T / n
        assert_close(P, Y @ Y.transpose(-1, -2) / n_samples)

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    @pytest.mark.parametrize("n_in, n_out, n_samples", [(20, 8, 50), (10, 10, 30)])
    def test_backward(
        self, batch_shape, n_in, n_out, n_samples, device, dtype, generator
    ):
        """Hand-derived gradient (Prop. 1) equals torch autograd."""
        X, Sigma = _raw_and_cov(batch_shape, n_in, n_samples, device, dtype, generator)
        W = _stiefel(n_in, n_out, device, dtype, generator)
        # symmetric upstream gradient E for P, arbitrary one F for Y
        E = symmetrize(
            torch.randn(
                (*batch_shape, n_out, n_out),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        F = torch.randn(
            (*batch_shape, n_out, n_samples),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        grads = {}
        for name, fun in [
            ("manual", rawandcov_linalg.CongruenceAndProjection.apply),
            ("auto", rawandcov_linalg.congruence_and_projection),
        ]:
            S_, X_, W_ = (
                t.clone().detach().requires_grad_(True) for t in (Sigma, X, W)
            )
            P, Y = fun(S_, X_, W_)
            # loss whose errors are exactly dl/dP = E and dl/dY = F
            ((P * E).sum() + (Y * F).sum()).backward()
            grads[name] = (S_.grad, X_.grad, W_.grad)

        # compare dl/dSigma, dl/dX and dl/dW
        for g_manual, g_auto in zip(grads["manual"], grads["auto"]):
            assert_close(g_manual, g_auto)

    def test_gradcheck(self, device, dtype, generator):
        """Hand-derived gradient against finite differences."""
        X, Sigma = _raw_and_cov((4,), 6, 10, device, dtype, generator)
        W = _stiefel(6, 3, device, dtype, generator)
        X.requires_grad_(True)
        W.requires_grad_(True)

        # gradcheck perturbs Sigma entry by entry (non-symmetrically):
        # only check X and W
        def fun(X_, W_):
            return rawandcov_linalg.CongruenceAndProjection.apply(Sigma, X_, W_)

        assert torch.autograd.gradcheck(fun, (X, W))
