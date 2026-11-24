"""Tests for Bures-Wasserstein geometry functions."""

import itertools

import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_geometries.bures_wasserstein as bw
from utils import is_spd, is_symmetric
from yetanotherspdnet.random.spd import random_SPD


class TestBuresWassersteinDistance:
    """Tests for BW distance."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_zero_self_distance(self, n_features, cond, device, dtype, generator):
        X = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        d2 = bw.bures_wasserstein_distance_squared(X, X)
        assert_close(
            d2, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-10, rtol=0
        )

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_symmetry(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        d12 = bw.bures_wasserstein_distance_squared(pts[0], pts[1])
        d21 = bw.bures_wasserstein_distance_squared(pts[1], pts[0])
        assert_close(d12, d21, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_positive(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        d2 = bw.bures_wasserstein_distance_squared(pts[0], pts[1])
        assert d2 > 0


class TestBuresWassersteinGeodesic:
    """Tests for BW geodesic."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_endpoints(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        p0 = bw.bures_wasserstein_geodesic(pts[0], pts[1], 0.0)
        p1 = bw.bures_wasserstein_geodesic(pts[0], pts[1], 1.0)
        assert_close(p0, pts[0], atol=1e-10, rtol=0)
        assert_close(p1, pts[1], atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_spd(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        mid = bw.bures_wasserstein_geodesic(pts[0], pts[1], 0.5)
        assert is_spd(mid)


class TestBuresWassersteinLogExp:
    """Tests for log/exp at identity and round-trip."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_log_exp_roundtrip_identity(
        self, n_features, cond, device, dtype, generator
    ):
        X = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        log_X = bw.bures_wasserstein_log_identity(X)
        recovered = bw.bures_wasserstein_exp_identity(log_X)
        assert_close(recovered, X, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_log_identity_of_identity(self, n_features, cond, device, dtype, generator):
        eye = torch.eye(n_features, device=device, dtype=dtype)
        log_I = bw.bures_wasserstein_log_identity(eye)
        assert_close(log_I, torch.zeros_like(eye), atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_log_at_base_roundtrip(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        base, X = pts[0], pts[1]
        log_X = bw.bures_wasserstein_log(X, base)
        recovered = bw._bures_wasserstein_exp(log_X, base)
        assert_close(recovered, X, atol=1e-8, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("t", [0.3, 0.7])
    def test_exp_follows_geodesic(self, n_features, cond, t, device, dtype, generator):
        # Exp_B(t Log_B X) is the BW geodesic from B to X evaluated at t
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        base, X = pts[0], pts[1]
        log_X = bw.bures_wasserstein_log(X, base)
        on_geodesic = bw._bures_wasserstein_exp(t * log_X, base)
        expected = bw.bures_wasserstein_geodesic(base, X, t)
        assert_close(on_geodesic, expected, atol=1e-8, rtol=0)


class TestBuresWassersteinMean:
    """Tests for BW barycenter."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_shape(self, n_features, n_matrices, cond, device, dtype, generator):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data, n_iterations=1)
        assert mean.shape == (n_features, n_features)
        assert is_spd(mean)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_autograd_vs_manual(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean_auto = bw.bures_wasserstein_mean(data, n_iterations=3)
        mean_manual = bw.BuresWassersteinMean(data, n_iterations=3)
        assert_close(mean_auto, mean_manual, atol=1e-10, rtol=0)


class TestBuresWassersteinStdScalar:
    """Tests for BW scalar std."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_positive(self, n_features, n_matrices, cond, device, dtype, generator):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data, n_iterations=3)
        std = bw.bures_wasserstein_std_scalar(data, mean)
        assert std > 0

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_autograd_vs_manual(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data, n_iterations=3)
        std_auto = bw.bures_wasserstein_std_scalar(data, mean)
        std_manual = bw.BuresWassersteinStdScalar.apply(data, mean)
        assert_close(std_auto, std_manual, atol=1e-10, rtol=0)


class TestBuresWassersteinCenterScaleBias:
    """Tests for centering, scaling, and biasing operations."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_center_produces_spd(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data, n_iterations=3)
        centered = bw.bures_wasserstein_center(data, mean)
        assert is_spd(centered)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_transported_logs_sum_to_zero(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Centering is exact in the tangent space: the logs at the barycenter sum
        # to zero, and the (linear) transport to the identity preserves that.
        mean = bw.bures_wasserstein_mean(data, n_iterations=50)
        log_at_mean = bw.bures_wasserstein_log(data, mean)
        transported = bw.bures_wasserstein_parallel_transport_to_identity(
            log_at_mean, mean
        )
        zero = torch.zeros(n_features, n_features, device=device, dtype=dtype)
        assert_close(transported.mean(dim=0), zero, atol=1e-8, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_centered_mean_is_identity_for_concentrated_data(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        # The centered BW mean is exactly the identity only while every transported
        # vector V keeps I + V/2 positive definite (injectivity domain of Exp_I);
        # spread-out data leave that domain and Exp_I folds them. Build data
        # concentrated around a base point so the property must hold exactly.
        base = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        far = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        data = bw.bures_wasserstein_geodesic(base.expand_as(far), far, 0.2)
        mean = bw.bures_wasserstein_mean(data, n_iterations=50)
        centered = bw.bures_wasserstein_center(data, mean)
        centered_mean = bw.bures_wasserstein_mean(centered, n_iterations=50)
        eye = torch.eye(n_features, device=device, dtype=dtype)
        assert_close(centered_mean, eye, atol=1e-8, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_scale_produces_spd(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data, n_iterations=3)
        centered = bw.bures_wasserstein_center(data, mean)
        std = bw.bures_wasserstein_std_scalar(data, mean)
        shift = torch.tensor(1.0, device=device, dtype=dtype)
        scaled = bw.bures_wasserstein_scale(centered, std**2, shift)
        assert is_spd(scaled)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("n_matrices", [5])
    def test_bias_produces_spd(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        biased = bw.bures_wasserstein_bias(data, G)
        assert is_spd(biased)


class TestBuresWassersteinParallelTransport:
    """Tests for parallel transport operations."""

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_to_identity_symmetric(self, n_features, cond, device, dtype, generator):
        pts = random_SPD(
            n_features, 2, cond=cond, device=device, dtype=dtype, generator=generator
        )
        base = pts[0]
        tangent = bw.bures_wasserstein_log(pts[1], base)
        transported = bw.bures_wasserstein_parallel_transport_to_identity(tangent, base)
        assert is_symmetric(transported)

    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    def test_from_identity_symmetric(self, n_features, cond, device, dtype, generator):
        X = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        tangent = bw.bures_wasserstein_log_identity(X)
        target = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )
        transported = bw.bures_wasserstein_parallel_transport_from_identity(
            tangent, target
        )
        assert is_symmetric(transported)


class TestBuresWassersteinGradients:
    """Test gradient flow through BW operations."""

    @pytest.mark.parametrize("n_features, cond", [(5, 10)])
    @pytest.mark.parametrize("n_matrices", [3])
    def test_mean_gradient(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        data.requires_grad_(True)
        mean = bw.bures_wasserstein_mean(data, n_iterations=1)
        loss = mean.sum()
        loss.backward()
        assert data.grad is not None
        assert not torch.any(torch.isnan(data.grad))

    @pytest.mark.parametrize("n_features, cond", [(5, 10)])
    @pytest.mark.parametrize("n_matrices", [3])
    def test_std_gradient_autograd(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        data.requires_grad_(True)
        mean = bw.bures_wasserstein_mean(data.detach(), n_iterations=1)
        std = bw.bures_wasserstein_std_scalar(data, mean)
        std.backward()
        assert data.grad is not None
        assert not torch.any(torch.isnan(data.grad))

    @pytest.mark.parametrize("n_features, cond", [(5, 10)])
    @pytest.mark.parametrize("n_matrices", [3])
    def test_std_manual_vs_autograd(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        """Check manual backward matches autograd for BW std."""
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        mean = bw.bures_wasserstein_mean(data.detach(), n_iterations=1)

        data_auto = data.clone().detach().requires_grad_(True)
        std_auto = bw.bures_wasserstein_std_scalar(data_auto, mean)
        std_auto.backward()

        data_manual = data.clone().detach().requires_grad_(True)
        std_manual = bw.BuresWassersteinStdScalar.apply(data_manual, mean)
        std_manual.backward()

        assert_close(data_auto.grad, data_manual.grad, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n_features, cond", [(5, 10)])
    @pytest.mark.parametrize("n_matrices", [3])
    def test_full_pipeline_gradient(
        self, n_features, n_matrices, cond, device, dtype, generator
    ):
        """Test gradient flows through full center->scale->bias pipeline."""
        data = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        data.requires_grad_(True)
        mean = bw.bures_wasserstein_mean(data, n_iterations=1)
        std = bw.bures_wasserstein_std_scalar(data.detach(), mean.detach())
        centered = bw.bures_wasserstein_center(data, mean.detach())
        shift = torch.tensor(1.0, device=device, dtype=dtype)
        scaled = bw.bures_wasserstein_scale(centered, std.detach() ** 2, shift)
        G = torch.eye(n_features, device=device, dtype=dtype)
        biased = bw.bures_wasserstein_bias(scaled, G)
        loss = biased.sum()
        loss.backward()
        assert data.grad is not None
        assert not torch.any(torch.isnan(data.grad))


def _symmetrize(x):
    return 0.5 * (x + x.transpose(-2, -1))


class TestDegenerateSafeGradients:
    """
    The BW bias step differentiates with respect to a learnable SPD point that
    starts at the identity (all eigenvalues equal). Autograd through
    torch.linalg.eigh returns NaN there; the dedicated Functions must not.
    """

    @pytest.mark.parametrize("point", ["identity", "random"])
    def test_lyapunov_solve_gradcheck(self, point, device, dtype, generator):
        n = 5
        base = (
            torch.eye(n, device=device, dtype=dtype)
            if point == "identity"
            else random_SPD(
                n, 1, cond=10, device=device, dtype=dtype, generator=generator
            )
        )
        rhs = _symmetrize(
            torch.randn(3, n, n, device=device, dtype=dtype, generator=generator)
        )
        fun = lambda b, v: bw.LyapunovSolveSPD.apply(_symmetrize(b), _symmetrize(v))
        inputs = (base.clone().requires_grad_(), rhs.clone().requires_grad_())
        assert torch.autograd.gradcheck(fun, inputs)

    @pytest.mark.parametrize("point", ["identity", "random"])
    def test_transport_from_identity_gradcheck(self, point, device, dtype, generator):
        n = 5
        target = (
            torch.eye(n, device=device, dtype=dtype)
            if point == "identity"
            else random_SPD(
                n, 1, cond=10, device=device, dtype=dtype, generator=generator
            )
        )
        tangent = _symmetrize(
            torch.randn(3, n, n, device=device, dtype=dtype, generator=generator)
        )
        fun = lambda s, g: bw.ParallelTransportFromIdentityBW.apply(
            _symmetrize(s), _symmetrize(g)
        )
        inputs = (tangent.clone().requires_grad_(), target.clone().requires_grad_())
        assert torch.autograd.gradcheck(fun, inputs)

    def test_transport_matches_reference(self, device, dtype, generator):
        n = 6
        target = random_SPD(
            n, 1, cond=10, device=device, dtype=dtype, generator=generator
        )
        tangent = _symmetrize(
            torch.randn(4, n, n, device=device, dtype=dtype, generator=generator)
        )
        assert_close(
            bw.ParallelTransportFromIdentityBW.apply(tangent, target),
            bw.bures_wasserstein_parallel_transport_from_identity(tangent, target),
        )

    def test_bias_gradient_finite_at_identity(self, device, dtype, generator):
        n = 5
        data = random_SPD(
            n, 4, cond=10, device=device, dtype=dtype, generator=generator
        )
        bias_point = torch.eye(n, device=device, dtype=dtype, requires_grad=True)
        bw.bures_wasserstein_bias(data, bias_point).sum().backward()
        assert torch.isfinite(bias_point.grad).all()


def _spectral(matrix, fun):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    return (eigvecs * fun(eigvals).unsqueeze(-2)) @ eigvecs.transpose(-2, -1)


def _trace(matrix):
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)


class TestPolarCrossTerm:
    """
    The cross term (B^1/2 X B^1/2)^1/2 is computed from the SVD of X^1/2 B^1/2
    (polar factor W). These tests check it against the eigendecomposition
    formulas at moderate conditioning, against exact values at kappa up to 1e9,
    and the manual backwards (gradcheck, including repeated singular values).
    """

    @pytest.fixture
    def pair(self, device, dtype, generator):
        X = random_SPD(5, 6, cond=50, device=device, dtype=dtype, generator=generator)
        B = random_SPD(5, 1, cond=50, device=device, dtype=dtype, generator=generator)
        return X, B

    def test_matches_eigh_formulas(self, pair):
        X, B = pair
        B_sqrt, B_isqrt = _spectral(B, torch.sqrt), _spectral(B, torch.rsqrt)
        C = _spectral(B_sqrt @ X @ B_sqrt, torch.sqrt)
        dist_sq = _trace(B) + _trace(X) - 2 * _trace(C)
        BX_sqrt = B_sqrt @ C @ B_isqrt
        log = BX_sqrt + BX_sqrt.transpose(-2, -1) - 2 * B
        assert_close(bw.bures_wasserstein_distance_squared(B, X), dist_sq)
        assert_close(bw.bures_wasserstein_distance_squared(X, B), dist_sq)
        for fun in (bw.bures_wasserstein_log, bw.BuresWassersteinLog):
            assert_close(fun(X, B), log)
        std = dist_sq.mean().sqrt()
        for fun in (
            bw.bures_wasserstein_std_scalar,
            bw.BuresWassersteinStdScalar.apply,
        ):
            assert_close(fun(X, B), std)
        t = 0.3
        geodesic = (1 - t) ** 2 * B + t**2 * X + t * (1 - t) * log.add(2 * B)
        for fun in (bw.bures_wasserstein_geodesic, bw.BuresWassersteinGeodesic):
            assert_close(fun(B.expand_as(X), X, t), geodesic)
        # three fixed-point iterations of the inverse-square-root form
        G = X.mean(0)
        for _ in range(3):
            G_sqrt, G_isqrt = _spectral(G, torch.sqrt), _spectral(G, torch.rsqrt)
            T = _spectral(G_sqrt @ X @ G_sqrt, torch.sqrt).mean(0)
            G = G_isqrt @ T @ T @ G_isqrt
        for fun in (bw.bures_wasserstein_mean, bw.BuresWassersteinMean):
            assert_close(fun(X, n_iterations=3), G)

    @pytest.mark.parametrize("kappa", [1e6, 1e9])
    def test_ill_conditioned_exact(self, kappa, device, dtype, generator):
        """X = T B T with T SPD: T is the optimal transport map from B to X, so
        (B^1/2 X B^1/2)^1/2 = B^1/2 T B^1/2 and Log_B(X) = TB + BT - 2B exactly.
        The eigendecomposition of B^1/2 X B^1/2 (condition number ~kappa^2)
        misses these by 5e-10 at kappa = 1e6 and fails at kappa = 1e9."""
        n = 6
        B = random_SPD(
            n, 1, cond=kappa, device=device, dtype=dtype, generator=generator
        )
        T = random_SPD(n, 4, cond=10, device=device, dtype=dtype, generator=generator)
        X = _symmetrize(T @ B @ T)
        dist_sq = _trace(B) + _trace(X) - 2 * _trace(T @ B)
        log = T @ B + B @ T - 2 * B
        eye = torch.eye(n, device=device, dtype=dtype)
        E = 0.7 * eye + 0.3 * T
        geodesic = E @ B @ E
        rtol = 1e-11
        assert_close(
            bw.bures_wasserstein_distance_squared(B, X), dist_sq, rtol=rtol, atol=0
        )
        for fun in (bw.bures_wasserstein_log, bw.BuresWassersteinLog):
            assert_close(fun(X, B), log, rtol=rtol, atol=rtol * log.abs().max())
        for fun in (bw.bures_wasserstein_geodesic, bw.BuresWassersteinGeodesic):
            result = fun(B.expand_as(X), X, 0.3)
            assert_close(result, geodesic, rtol=rtol, atol=rtol * geodesic.abs().max())
        for fun in (
            bw.bures_wasserstein_std_scalar,
            bw.BuresWassersteinStdScalar.apply,
        ):
            assert_close(fun(X, B), dist_sq.mean().sqrt(), rtol=rtol, atol=0)

    def test_distance_nonnegative_when_close(self, device, dtype, generator):
        """The Procrustes form has no cancellation: d^2(X, X + dX) ~ |dX|^2."""
        X = random_SPD(5, 4, cond=1e4, device=device, dtype=dtype, generator=generator)
        assert (bw.bures_wasserstein_distance_squared(X, X) >= 0).all()
        assert bw.bures_wasserstein_distance_squared(X, X).max() < 1e-24

    @pytest.mark.parametrize("case", ["random", "scalar", "equal", "kappa_1e3"])
    def test_manual_gradcheck(self, case, device, dtype, generator):
        n = 4
        eye = torch.eye(n, device=device, dtype=dtype)
        if case == "random":
            X = random_SPD(
                n, 3, cond=20, device=device, dtype=dtype, generator=generator
            )
            B = random_SPD(
                n, 1, cond=20, device=device, dtype=dtype, generator=generator
            )
        elif case == "scalar":  # all singular values of X^1/2 B^1/2 equal
            X = torch.stack([2.0 * eye, 3.0 * eye, 0.5 * eye])
            B = eye.clone()
        elif case == "equal":  # X = B: W = I, std = 0 is avoided by a second point
            B = random_SPD(
                n, 1, cond=20, device=device, dtype=dtype, generator=generator
            )
            X = torch.stack([B, 2.0 * B])
        else:
            X = random_SPD(
                n, 3, cond=1e3, device=device, dtype=dtype, generator=generator
            )
            B = random_SPD(
                n, 1, cond=1e3, device=device, dtype=dtype, generator=generator
            )
        inputs = (X.clone().requires_grad_(), B.clone().requires_grad_())

        def sym(fun):
            return lambda x, b: fun(_symmetrize(x), _symmetrize(b))

        functions = {
            "log": bw.BuresWassersteinLog,
            "std": bw.BuresWassersteinStdScalar.apply,
            "geodesic": lambda x, b: bw.BuresWassersteinGeodesic(
                b.expand_as(x), x, 0.3
            ),
            "mean": lambda x, b: bw.BuresWassersteinMean(x, n_iterations=2) @ b,
        }
        for name, fun in functions.items():
            assert torch.autograd.gradcheck(sym(fun), inputs, rtol=1e-5), name

    def test_manual_matches_autograd_ill_conditioned(self, device, dtype, generator):
        """kappa = 1e6 (beyond the finite differences of gradcheck)."""
        n = 5
        X = random_SPD(n, 4, cond=1e6, device=device, dtype=dtype, generator=generator)
        B = random_SPD(n, 1, cond=1e6, device=device, dtype=dtype, generator=generator)
        pairs = {
            "log": (bw.bures_wasserstein_log, bw.BuresWassersteinLog),
            "std": (
                bw.bures_wasserstein_std_scalar,
                bw.BuresWassersteinStdScalar.apply,
            ),
            "geodesic": (
                lambda x, b: bw.bures_wasserstein_geodesic(b.expand_as(x), x, 0.3),
                lambda x, b: bw.BuresWassersteinGeodesic(b.expand_as(x), x, 0.3),
            ),
        }
        for name, functions in pairs.items():
            grads = []
            for fun in functions:
                x, b = X.clone().requires_grad_(), B.clone().requires_grad_()
                out = fun(x, b)
                cotangent = (
                    torch.ones_like(out)
                    + torch.arange(out.numel(), device=device, dtype=dtype).reshape(
                        out.shape
                    )
                    / out.numel()
                )
                (out * cotangent).sum().backward()
                grads.append((x.grad, b.grad))
            assert_close(grads[0][0], grads[1][0], rtol=1e-6, atol=1e-8), name
            assert_close(grads[0][1], grads[1][1], rtol=1e-6, atol=1e-8), name

    def test_std_manual_matches_autograd(self, pair):
        grads = []
        for fun in (
            bw.bures_wasserstein_std_scalar,
            bw.BuresWassersteinStdScalar.apply,
        ):
            X, B = (p.clone().requires_grad_() for p in pair)
            fun(X, B).backward()
            grads.append((X.grad, B.grad))
        assert_close(grads[0][0], grads[1][0])
        assert_close(grads[0][1], grads[1][1])

    def test_scalar_batch_finite(self, device, dtype):
        """A batch of scalar matrices cI: repeated singular values everywhere."""
        n = 4
        eye = torch.eye(n, device=device, dtype=dtype)
        X = torch.stack([c * eye for c in (0.5, 1.0, 2.0, 4.0)]).requires_grad_()
        mean = bw.BuresWassersteinMean(X)
        log = bw.BuresWassersteinLog(X, mean)
        std = bw.BuresWassersteinStdScalar.apply(X, mean)
        (log.sum() + std).backward()
        assert torch.isfinite(X.grad).all()
        # the BW mean of scalar matrices c_i I is (mean sqrt(c_i))^2 I
        expected = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=dtype).sqrt().mean() ** 2
        assert_close(mean, expected * eye)


def _lyapunov(P, R):
    """Solve P L + L P = R (P SPD)."""
    eigvals, eigvecs = torch.linalg.eigh(P)
    rotated = eigvecs.transpose(-2, -1) @ R @ eigvecs
    middle = rotated / (eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2))
    return eigvecs @ middle @ eigvecs.transpose(-2, -1)


def _bw_metric(base, tangent):
    """g_B(S, S) = tr(L S) / 2 with B L + L B = S."""
    return _trace(_lyapunov(base, tangent) @ tangent) / 2


def _transport_to_identity_ode(tangent, base, n_steps):
    """Levi-Civita transport along the geodesic B -> I by the horizontal lift.

    BW is the quotient of GL(n) (Euclidean metric) by O(n), pi(A) = A A^T. The
    geodesic B -> I lifts to A(t) = (1 - t) B^1/2 + t I (its velocity
    (B^-1/2 - I) A is horizontal), and a parallel field lifts to a horizontal
    field Z(t) = L(t) A(t) (L symmetric) whose derivative is vertical: each
    explicit Euler step projects Z onto the horizontal space at the next A
    (first order, error O(1/n_steps)). Tangent vectors map as V = Z A^T + A Z^T.
    """
    eye = torch.eye(base.shape[-1], dtype=base.dtype, device=base.device)
    base_sqrt = _spectral(base, torch.sqrt)
    A = base_sqrt
    Z = _lyapunov(base, tangent) @ A
    for k in range(1, n_steps + 1):
        t = k / n_steps
        A = (1 - t) * base_sqrt + t * eye
        Z = (
            _lyapunov(
                A @ A.transpose(-2, -1),
                Z @ A.transpose(-2, -1) + A @ Z.transpose(-2, -1),
            )
            @ A
        )
    return Z @ A.transpose(-2, -1) + A @ Z.transpose(-2, -1)


class TestBuresWassersteinTransportGeometry:
    """Parallel transport and barycenter properties of the BW geometry."""

    @pytest.fixture
    def base_and_tangents(self, device, dtype, generator):
        B = random_SPD(4, 1, cond=20, device=device, dtype=dtype, generator=generator)
        S = _symmetrize(
            torch.randn(5, 4, 4, device=device, dtype=dtype, generator=generator)
        )
        return B, S

    @pytest.mark.parametrize("manual", [False, True])
    def test_transport_is_isometry(self, manual, base_and_tangents):
        """g_B(S, S) = g_I(Gamma S, Gamma S), and g_I(S, S) = tr(S^2) / 4."""
        B, S = base_and_tangents
        to_identity = (
            bw.ParallelTransportToIdentityBW.apply
            if manual
            else bw.bures_wasserstein_parallel_transport_to_identity
        )
        transported = to_identity(S, B)
        eye = torch.eye(4, dtype=B.dtype, device=B.device)
        assert_close(
            _bw_metric(eye, transported), _trace(transported @ transported) / 4
        )
        assert_close(_bw_metric(B, S), _bw_metric(eye, transported))
        # polarization: inner products are preserved too
        cross = lambda base, u, v: _trace(_lyapunov(base, u) @ v) / 2
        assert_close(cross(B, S[0], S[1]), cross(eye, transported[0], transported[1]))

    @pytest.mark.parametrize("manual", [False, True])
    def test_transport_round_trip(self, manual, base_and_tangents):
        B, S = base_and_tangents
        if manual:
            to_identity = bw.ParallelTransportToIdentityBW.apply
            from_identity = bw.ParallelTransportFromIdentityBW.apply
        else:
            to_identity = bw.bures_wasserstein_parallel_transport_to_identity
            from_identity = bw.bures_wasserstein_parallel_transport_from_identity
        assert_close(from_identity(to_identity(S, B), B), S)
        assert_close(to_identity(from_identity(S, B), B), S)

    def test_transport_matches_integrated_levi_civita(self, base_and_tangents):
        """The closed-form transport agrees with the integrated Levi-Civita
        transport along the geodesic, and the integration error is O(1/K)."""
        B, S = base_and_tangents
        exact = bw.bures_wasserstein_parallel_transport_to_identity(S, B)
        errors = [
            float(
                (_transport_to_identity_ode(S, B, n_steps) - exact).norm()
                / exact.norm()
            )
            for n_steps in (25, 50, 100, 200)
        ]
        ratios = [a / b for a, b in itertools.pairwise(errors)]
        assert all(1.7 < r < 2.3 for r in ratios), (errors, ratios)
        assert errors[-1] < 2e-2
        # first-order extrapolation removes the leading term
        assert 2 * errors[-1] - errors[-2] < errors[-1] / 10

    @pytest.mark.parametrize("weights", [(1, 1), (1, 3), (3, 2)])
    def test_weighted_barycenter_is_geodesic_point(
        self, weights, device, dtype, generator
    ):
        """The barycenter of X1 (weight k) and X2 (weight m) is the geodesic point
        at t = m / (k + m); weights are given as multiplicities."""
        k, m = weights
        X = random_SPD(4, 2, cond=20, device=device, dtype=dtype, generator=generator)
        data = torch.cat([X[0].expand(k, 4, 4), X[1].expand(m, 4, 4)])
        t = m / (k + m)
        expected = bw.bures_wasserstein_geodesic(X[0], X[1], t)
        for mean in (bw.bures_wasserstein_mean, bw.BuresWassersteinMean):
            assert_close(mean(data, n_iterations=100), expected, rtol=1e-9, atol=1e-11)

    @pytest.mark.parametrize("manual", [False, True])
    def test_scaling_sets_dispersion(self, manual, device, dtype, generator):
        """GBWBN Prop. 3.1: inside the injectivity domain, the scaling step
        multiplies each BW distance to I by s / sigma, so centred data of
        dispersion sigma around I get dispersion s."""
        n = 4
        eye = torch.eye(n, device=device, dtype=dtype)
        tangent = 0.3 * _symmetrize(
            torch.randn(16, n, n, device=device, dtype=dtype, generator=generator)
        )
        data = bw.bures_wasserstein_exp_identity(tangent)
        sigma = bw.bures_wasserstein_std_scalar(data, eye)
        shift = torch.tensor(0.5, device=device, dtype=dtype)
        scale = bw.BuresWassersteinScale if manual else bw.bures_wasserstein_scale
        scaled = scale(data, sigma**2, shift, eps=0.0)
        factor = shift / sigma
        root = _spectral(data, torch.sqrt)
        assert (torch.linalg.eigvalsh(eye + factor * (root - eye)) > 0).all()
        assert_close(
            bw.bures_wasserstein_distance_squared(eye, scaled).sqrt(),
            factor * bw.bures_wasserstein_distance_squared(eye, data).sqrt(),
        )
        assert_close(bw.bures_wasserstein_std_scalar(scaled, eye), shift)
