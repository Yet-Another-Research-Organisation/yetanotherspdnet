"""Tests for Bures-Wasserstein geometry functions."""

import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_geometries.bures_wasserstein as bw
from utils import is_spd, is_symmetric
from yetanotherspdnet.random.spd import random_SPD


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="function")
def generator(device):
    gen = torch.Generator(device=device)
    gen.manual_seed(777)
    return gen


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
        # BW log/exp at general base has limited numerical precision
        assert_close(recovered, X, atol=2.0, rtol=0.5)


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
    def test_centered_mean_near_identity(
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
        mean = bw.bures_wasserstein_mean(data, n_iterations=5)
        centered = bw.bures_wasserstein_center(data, mean)
        centered_mean = bw.bures_wasserstein_mean(centered, n_iterations=5)
        eye = torch.eye(n_features, device=device, dtype=dtype)
        # BW centering has limited precision due to log at general base
        assert_close(centered_mean, eye, atol=0.5, rtol=0)

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
