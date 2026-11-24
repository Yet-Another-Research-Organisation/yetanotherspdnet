"""Tests for the two-stream (covariance + raw data) linear algebra functions."""

import pytest
import torch
from torch.testing import assert_close

from utils import is_spd
from yetanotherspdnet.functions import rawandcov_linalg
from yetanotherspdnet.functions.spd_linalg import (
    congruence_fullrank,
    eigh_relu,
    symmetrize,
)
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


# ------------------------------------------------------------
# Two-stream ReEig: projection and whitening versions
# ------------------------------------------------------------
# eps chosen so that some eigenvalues of Sigma = X X^T / n are rectified
REEIG_EPS = 0.5


def _reeig_cases():
    """(manual Function, autograd function) for the three versions."""
    return {
        "projection": (
            lambda S, X: rawandcov_linalg.EighReLuAndProjection.apply(S, X, REEIG_EPS),
            lambda S, X: rawandcov_linalg.eigh_relu_and_projection(S, X, REEIG_EPS)[:2],
        ),
        "whitening_Q": (
            lambda S, X: rawandcov_linalg.EighReLuAndWhitening.apply(
                S, X, REEIG_EPS, "Q"
            ),
            lambda S, X: rawandcov_linalg.eigh_relu_and_whitening(S, X, REEIG_EPS, "Q")[
                :2
            ],
        ),
        "whitening_Sigma": (
            lambda S, X: rawandcov_linalg.EighReLuAndWhitening.apply(
                S, X, REEIG_EPS, "Sigma"
            ),
            lambda S, X: rawandcov_linalg.eigh_relu_and_whitening(
                S, X, REEIG_EPS, "Sigma"
            )[:2],
        ),
    }


class TestEighReLuTwoStream:
    """
    Test suite for the two-stream ReEig functions (projection and whitening)
    """

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_forward(self, mode, batch_shape, device, dtype, generator):
        """Shapes, Q is SPD and equals the classical ReEig, V as expected."""
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        manual, auto = _reeig_cases()[mode]

        Q, V = manual(Sigma, X)
        Q_ref, V_ref = auto(Sigma, X)

        # shapes
        assert Q.shape == Sigma.shape
        assert V.shape == X.shape
        # the class and the function give the same result
        assert_close(Q, Q_ref)
        assert_close(V, V_ref)
        # SPD stream: same as the classical ReEig, eigenvalues >= eps
        assert is_spd(Q)
        assert_close(Q, eigh_relu(Sigma, REEIG_EPS)[0])
        assert torch.all(torch.linalg.eigvalsh(Q) >= REEIG_EPS - 1e-10)

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    def test_raw_stream_definition(self, mode, device, dtype, generator):
        """V matches its definition, computed independently."""
        X, Sigma = _raw_and_cov((4,), 6, 20, device, dtype, generator)
        manual, _ = _reeig_cases()[mode]
        _, V = manual(Sigma, X)

        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        if mode == "projection":
            # V depends on the sign of the eigenvectors: same convention
            eigvecs = rawandcov_linalg._eigh_sign_convention(eigvecs)
            V_ref = eigvecs @ torch.relu(eigvecs.transpose(-1, -2) @ X)
        else:
            r = (
                torch.clamp(eigvals, min=REEIG_EPS)
                if mode == "whitening_Q"
                else eigvals
            )
            sqrt_R = (eigvecs * r.sqrt().unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
            inv_sqrt_R = (eigvecs * r.rsqrt().unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
            V_ref = sqrt_R @ torch.relu(inv_sqrt_R @ X)
        assert_close(V, V_ref)

    def test_whitening_sigma_without_rectification(self, device, dtype, generator):
        """Whitening by Sigma: if all whitened data are positive, V = X."""
        X, Sigma = _raw_and_cov((3,), 5, 20, device, dtype, generator)
        # make the whitened data positive: X = Sigma^{1/2} |W| with W whitened
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        sqrt_S = (eigvecs * eigvals.sqrt().unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        inv_sqrt_S = (eigvecs * eigvals.rsqrt().unsqueeze(-2)) @ eigvecs.transpose(
            -1, -2
        )
        X_pos = sqrt_S @ (inv_sqrt_S @ X).abs()
        _, V = rawandcov_linalg.EighReLuAndWhitening.apply(Sigma, X_pos, 1e-6, "Sigma")
        assert_close(V, X_pos)

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_backward(self, mode, batch_shape, device, dtype, generator):
        """Hand-derived gradient equals torch autograd through eigh."""
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        E = symmetrize(
            torch.randn(
                (*batch_shape, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        G = torch.randn(
            (*batch_shape, n_features, n_samples),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        grads = {}
        for name, fun in zip(("manual", "auto"), _reeig_cases()[mode]):
            S_, X_ = (t.clone().detach().requires_grad_(True) for t in (Sigma, X))
            Q, V = fun(S_, X_)
            ((Q * E).sum() + (V * G).sum()).backward()
            grads[name] = (S_.grad, X_.grad)

        # the gradient wrt Sigma is only defined up to its antisymmetric part
        assert_close(grads["manual"][0], symmetrize(grads["auto"][0]))
        assert_close(grads["manual"][1], grads["auto"][1])

    @pytest.mark.parametrize("mode", ["projection", "whitening_Q", "whitening_Sigma"])
    def test_gradcheck(self, mode, device, dtype, generator):
        """Hand-derived gradient against finite differences."""
        X, Sigma = _raw_and_cov((2,), 5, 12, device, dtype, generator)
        manual, _ = _reeig_cases()[mode]
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        # Sigma is kept symmetric through the parametrization M -> sym(M)
        def fun(M, X_):
            return manual(symmetrize(M), X_)

        assert torch.autograd.gradcheck(fun, (Sigma, X))
