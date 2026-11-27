"""Tests for the two-stream (covariance + raw data) linear algebra functions."""

import pytest
import torch
from torch.testing import assert_close

from utils import is_spd, is_symmetric
from yetanotherspdnet.functions import rawandcov_linalg
from yetanotherspdnet.functions.spd_linalg import (
    congruence_fullrank,
    eigh_relu,
    logm_SPD,
    sym_matrix_to_coordinates,
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
        """
        Test of backward of CongruenceAndProjection Function class
        """
        X, Sigma = _raw_and_cov(batch_shape, n_in, n_samples, device, dtype, generator)
        W = _stiefel(n_in, n_out, device, dtype, generator)

        Sigma_manual = Sigma.clone().detach()
        Sigma_manual.requires_grad = True
        Sigma_auto = Sigma.clone().detach()
        Sigma_auto.requires_grad = True

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        W_manual = W.clone().detach()
        W_manual.requires_grad = True
        W_auto = W.clone().detach()
        W_auto.requires_grad = True

        P_manual, Y_manual = rawandcov_linalg.CongruenceAndProjection.apply(
            Sigma_manual, X_manual, W_manual
        )
        P_auto, Y_auto = rawandcov_linalg.congruence_and_projection(
            Sigma_auto, X_auto, W_auto
        )

        loss_manual = torch.norm(P_manual) + torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(P_auto) + torch.norm(Y_auto)
        loss_auto.backward()

        assert Sigma_manual.grad is not None
        assert Sigma_auto.grad is not None
        assert torch.isfinite(Sigma_manual.grad).all()
        assert torch.isfinite(Sigma_auto.grad).all()
        assert is_symmetric(Sigma_manual.grad)
        assert is_symmetric(Sigma_auto.grad)
        assert_close(Sigma_manual.grad, Sigma_auto.grad)

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

        assert W_manual.grad is not None
        assert W_auto.grad is not None
        assert torch.isfinite(W_manual.grad).all()
        assert torch.isfinite(W_auto.grad).all()
        assert_close(W_manual.grad, W_auto.grad)

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
        """
        Test of backward of EighReLuAndProjection / EighReLuAndWhitening
        Function classes
        """
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        manual, auto = _reeig_cases()[mode]

        Sigma_manual = Sigma.clone().detach()
        Sigma_manual.requires_grad = True
        Sigma_auto = Sigma.clone().detach()
        Sigma_auto.requires_grad = True

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        Q_manual, V_manual = manual(Sigma_manual, X_manual)
        Q_auto, V_auto = auto(Sigma_auto, X_auto)

        loss_manual = torch.norm(Q_manual) + torch.norm(V_manual)
        loss_manual.backward()
        loss_auto = torch.norm(Q_auto) + torch.norm(V_auto)
        loss_auto.backward()

        assert Sigma_manual.grad is not None
        assert Sigma_auto.grad is not None
        assert torch.isfinite(Sigma_manual.grad).all()
        assert torch.isfinite(Sigma_auto.grad).all()
        assert is_symmetric(Sigma_manual.grad)
        assert is_symmetric(Sigma_auto.grad)
        assert_close(Sigma_manual.grad, Sigma_auto.grad)

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

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


# ------------------------------------------------------------
# Two-stream LogEig and vectorization
# ------------------------------------------------------------
def _logeig_cases():
    """(manual Function, autograd function) for the two LogEig versions."""
    return {
        "version_1": (
            rawandcov_linalg.LogmTwoStream.apply,
            lambda S, X: rawandcov_linalg.logm_two_stream(S, X)[:2],
        ),
        "version_2": (
            rawandcov_linalg.LogmAndWhitening.apply,
            lambda S, X: rawandcov_linalg.logm_and_whitening(S, X)[:2],
        ),
    }


class TestLogmTwoStream:
    """
    Test suite for the two-stream LogEig functions (versions 1 and 2)
    """

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_forward(self, version, batch_shape, device, dtype, generator):
        """Shapes, log of Sigma, and W = X or Sigma^{-1/2} X."""
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        manual, auto = _logeig_cases()[version]

        L, W = manual(Sigma, X)
        L_ref, W_ref = auto(Sigma, X)

        assert L.shape == Sigma.shape
        assert W.shape == X.shape
        assert_close(L, L_ref)
        assert_close(W, W_ref)
        # SPD stream: same as the classical LogEig
        assert_close(L, logm_SPD(Sigma)[0])
        if version == "version_1":
            assert_close(W, X)
        else:
            # whitened data have identity covariance
            assert_close(
                W @ W.transpose(-1, -2) / n_samples,
                torch.eye(n_features, device=device, dtype=dtype).expand_as(Sigma),
            )

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_backward(self, version, batch_shape, device, dtype, generator):
        """
        Test of backward of LogmTwoStream / LogmAndWhitening Function classes
        """
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        manual, auto = _logeig_cases()[version]

        Sigma_manual = Sigma.clone().detach()
        Sigma_manual.requires_grad = True
        Sigma_auto = Sigma.clone().detach()
        Sigma_auto.requires_grad = True

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        L_manual, W_manual = manual(Sigma_manual, X_manual)
        L_auto, W_auto = auto(Sigma_auto, X_auto)

        loss_manual = torch.norm(L_manual) + torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(L_auto) + torch.norm(W_auto)
        loss_auto.backward()

        assert Sigma_manual.grad is not None
        assert Sigma_auto.grad is not None
        assert torch.isfinite(Sigma_manual.grad).all()
        assert torch.isfinite(Sigma_auto.grad).all()
        assert is_symmetric(Sigma_manual.grad)
        assert is_symmetric(Sigma_auto.grad)
        assert_close(Sigma_manual.grad, Sigma_auto.grad)

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

    @pytest.mark.parametrize("version", ["version_1", "version_2"])
    def test_gradcheck(self, version, device, dtype, generator):
        """Hand-derived gradient against finite differences."""
        X, Sigma = _raw_and_cov((2,), 5, 12, device, dtype, generator)
        manual, _ = _logeig_cases()[version]
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        # Sigma is kept symmetric through the parametrization M -> sym(M)
        def fun(M, X_):
            return manual(symmetrize(M), X_)

        assert torch.autograd.gradcheck(fun, (Sigma, X))


class TestCoordinatesAndVec:
    """
    Test suite for the two-stream vectorization function
    """

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_forward(self, batch_shape, device, dtype, generator):
        """Shapes, and both parts of z match their definition."""
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        L = logm_SPD(Sigma)[0]
        manual = rawandcov_linalg.CoordinatesAndVec.apply
        auto = rawandcov_linalg.coordinates_and_vec

        z = manual(L, X)
        z_ref = auto(L, X)

        n_coord = n_features * (n_features + 1) // 2
        n_out = n_coord + n_features * n_samples
        assert z.shape == (*batch_shape, n_out)
        assert_close(z, z_ref)
        assert_close(z[..., :n_coord], sym_matrix_to_coordinates(L))
        assert_close(z[..., n_coord:], X.reshape(*X.shape[:-2], -1))

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_backward(self, batch_shape, device, dtype, generator):
        """
        Test of backward of CoordinatesAndVec Function class
        """
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        L = logm_SPD(Sigma)[0]
        manual = rawandcov_linalg.CoordinatesAndVec.apply
        auto = rawandcov_linalg.coordinates_and_vec

        L_manual = L.clone().detach()
        L_manual.requires_grad = True
        L_auto = L.clone().detach()
        L_auto.requires_grad = True

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        loss_manual = torch.norm(manual(L_manual, X_manual))
        loss_manual.backward()
        loss_auto = torch.norm(auto(L_auto, X_auto))
        loss_auto.backward()

        assert L_manual.grad is not None
        assert torch.isfinite(L_manual.grad).all()
        assert is_symmetric(L_manual.grad)
        assert_close(L_manual.grad, L_auto.grad)

        assert X_manual.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

    def test_gradcheck(self, device, dtype, generator):
        """Hand-derived gradient against finite differences."""
        X, Sigma = _raw_and_cov((2,), 5, 12, device, dtype, generator)
        L = logm_SPD(Sigma)[0].requires_grad_(True)
        X.requires_grad_(True)
        manual = rawandcov_linalg.CoordinatesAndVec.apply

        # L is kept symmetric through the parametrization M -> sym(M)
        def fun(M, X_):
            return manual(symmetrize(M), X_)

        assert torch.autograd.gradcheck(fun, (L, X))


# ------------------------------------------------------------
# Two-stream Mixing (Euclidean path)
# ------------------------------------------------------------
MIXING_ALPHA = 0.3


class TestEuclideanMixing:
    """
    Test suite for the two-stream Mixing function (Euclidean path)
    """

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_forward(self, batch_shape, device, dtype, generator):
        """Shapes, P is SPD and matches its definition, Y = X."""
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        # Sigma different from the covariance of X
        Sigma = Sigma + torch.eye(n_features, device=device, dtype=dtype)

        P, Y = rawandcov_linalg.EuclideanMixing.apply(Sigma, X, MIXING_ALPHA)
        P_ref, Y_ref = rawandcov_linalg.euclidean_mixing(Sigma, X, MIXING_ALPHA)

        assert P.shape == Sigma.shape
        assert Y.shape == X.shape
        assert is_spd(P)
        assert_close(P, P_ref)
        assert_close(Y, Y_ref)
        assert_close(Y, X)
        assert_close(
            P,
            MIXING_ALPHA * Sigma
            + (1 - MIXING_ALPHA) * X @ X.transpose(-1, -2) / n_samples,
        )

    def test_special_cases(self, device, dtype, generator):
        """alpha = 1 leaves Sigma unchanged; no effect if the link holds."""
        n_features, n_samples = 6, 20
        X, Sigma = _raw_and_cov((4,), n_features, n_samples, device, dtype, generator)
        Sigma_other = Sigma + torch.eye(n_features, device=device, dtype=dtype)

        # alpha = 1: P = Sigma
        P, _ = rawandcov_linalg.EuclideanMixing.apply(Sigma_other, X, 1.0)
        assert_close(P, Sigma_other)

        # Sigma is the covariance of X: P = Sigma for any alpha
        P, _ = rawandcov_linalg.EuclideanMixing.apply(Sigma, X, MIXING_ALPHA)
        assert_close(P, Sigma)

    @pytest.mark.parametrize("batch_shape", [(), (7,), (3, 5)])
    def test_backward(self, batch_shape, device, dtype, generator):
        """
        Test of backward of EuclideanMixing Function class
        """
        n_features, n_samples = 10, 30
        X, Sigma = _raw_and_cov(
            batch_shape, n_features, n_samples, device, dtype, generator
        )
        Sigma = Sigma + torch.eye(n_features, device=device, dtype=dtype)

        Sigma_manual = Sigma.clone().detach()
        Sigma_manual.requires_grad = True
        Sigma_auto = Sigma.clone().detach()
        Sigma_auto.requires_grad = True

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        P_manual, Y_manual = rawandcov_linalg.EuclideanMixing.apply(
            Sigma_manual, X_manual, MIXING_ALPHA
        )
        P_auto, Y_auto = rawandcov_linalg.euclidean_mixing(
            Sigma_auto, X_auto, MIXING_ALPHA
        )

        loss_manual = torch.norm(P_manual) + torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(P_auto) + torch.norm(Y_auto)
        loss_auto.backward()

        assert Sigma_manual.grad is not None
        assert Sigma_auto.grad is not None
        assert torch.isfinite(Sigma_manual.grad).all()
        assert torch.isfinite(Sigma_auto.grad).all()
        assert is_symmetric(Sigma_manual.grad)
        assert is_symmetric(Sigma_auto.grad)
        assert_close(Sigma_manual.grad, Sigma_auto.grad)

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

    def test_gradcheck(self, device, dtype, generator):
        """Hand-derived gradient against finite differences."""
        X, Sigma = _raw_and_cov((2,), 5, 12, device, dtype, generator)
        Sigma.requires_grad_(True)
        X.requires_grad_(True)

        # Sigma is kept symmetric through the parametrization M -> sym(M)
        def fun(M, X_):
            return rawandcov_linalg.EuclideanMixing.apply(
                symmetrize(M), X_, MIXING_ALPHA
            )

        assert torch.autograd.gradcheck(fun, (Sigma, X))
