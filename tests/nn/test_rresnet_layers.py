"""Tests for Riemannian Residual Network layers: SpectralVectorField and ResidualBlock."""

import pytest
import torch
from torch.testing import assert_close

from utils import is_spd, is_symmetric
from yetanotherspdnet.nn.rresnet_layers import ResidualBlock, SpectralVectorField
from yetanotherspdnet.random.spd import random_SPD


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="function")
def generator(device):
    gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    gen.manual_seed(777)
    return gen


# ========================================================================
# SpectralVectorField tests
# ========================================================================
class TestSpectralVectorField:
    """Tests for SpectralVectorField layer."""

    @pytest.mark.parametrize("n_features", [5, 10])
    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_output_shape(self, n_features, spectrum_type, device, dtype, generator):
        """Output has same shape as input."""
        vf = SpectralVectorField(
            n_features=n_features,
            spectrum_type=spectrum_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(n_features, 4, device=device, dtype=dtype, generator=generator)
        tangent = vf(X)
        assert tangent.shape == X.shape
        assert tangent.dtype == dtype

    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_output_symmetric(self, spectrum_type, device, dtype, generator):
        """Output tangent vectors are symmetric."""
        vf = SpectralVectorField(
            n_features=6,
            spectrum_type=spectrum_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(6, 4, device=device, dtype=dtype, generator=generator)
        tangent = vf(X)
        assert is_symmetric(tangent), "Tangent vector must be symmetric"

    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_unbatched_input(self, spectrum_type, device, dtype, generator):
        """Works with single SPD matrix (no batch dim)."""
        vf = SpectralVectorField(
            n_features=5,
            spectrum_type=spectrum_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 1, device=device, dtype=dtype, generator=generator).squeeze(0)
        tangent = vf(X)
        assert tangent.shape == (5, 5)

    def test_repr(self, device, dtype, generator):
        """repr is informative."""
        vf = SpectralVectorField(
            n_features=5,
            spectrum_type="conv1d",
            device=device,
            dtype=dtype,
            generator=generator,
        )
        r = repr(vf)
        assert "SpectralVectorField" in r
        assert "n_features=5" in r
        assert "conv1d" in r

    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_backward(self, spectrum_type, device, dtype, generator):
        """Backward pass produces finite gradients."""
        vf = SpectralVectorField(
            n_features=5,
            spectrum_type=spectrum_type,
            use_autograd=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)
        X.requires_grad_(True)
        tangent = vf(X)
        loss = tangent.sum()
        loss.backward()
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()

    def test_q_orthogonal(self, device, dtype, generator):
        """Q matrix remains orthogonal after parametrization."""
        vf = SpectralVectorField(
            n_features=5,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Q = vf.Q
        eye = torch.eye(5, device=device, dtype=dtype)
        assert_close(Q.T @ Q, eye, atol=1e-6, rtol=1e-6)

    def test_invalid_spectrum_type(self, device, dtype, generator):
        """Invalid spectrum_type raises AssertionError."""
        with pytest.raises(AssertionError, match="spectrum_type"):
            SpectralVectorField(
                n_features=5,
                spectrum_type="invalid",
                device=device,
                dtype=dtype,
                generator=generator,
            )

    @pytest.mark.parametrize("n_layers", [1, 3])
    def test_different_n_layers(self, n_layers, device, dtype, generator):
        """Different number of hidden layers works."""
        vf = SpectralVectorField(
            n_features=5,
            spectrum_n_layers=n_layers,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)
        tangent = vf(X)
        assert tangent.shape == X.shape

    @pytest.mark.parametrize("kernel_size", [3, 7])
    def test_different_kernel_size(self, kernel_size, device, dtype, generator):
        """Different Conv1d kernel sizes work."""
        vf = SpectralVectorField(
            n_features=8,
            spectrum_type="conv1d",
            spectrum_kernel_size=kernel_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(8, 4, device=device, dtype=dtype, generator=generator)
        tangent = vf(X)
        assert tangent.shape == X.shape


# ========================================================================
# ResidualBlock tests
# ========================================================================
class TestResidualBlock:
    """Tests for ResidualBlock layer."""

    @pytest.mark.parametrize("n_features", [5, 8])
    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_output_shape(self, n_features, spectrum_type, device, dtype, generator):
        """Output has same shape as input."""
        block = ResidualBlock(
            n_features=n_features,
            spectrum_type=spectrum_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(
            n_features,
            4,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Y = block(X)
        assert Y.shape == X.shape
        assert Y.dtype == dtype

    @pytest.mark.parametrize("spectrum_type", ["conv1d", "mlp"])
    def test_output_spd(self, spectrum_type, device, dtype, generator):
        """Output is SPD (symmetric positive definite)."""
        block = ResidualBlock(
            n_features=5,
            spectrum_type=spectrum_type,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)
        Y = block(X)
        assert is_symmetric(Y), "Output must be symmetric"
        assert is_spd(Y), "Output must be positive definite"

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward(self, use_autograd, device, dtype, generator):
        """Backward pass with both autograd and manual modes."""
        block = ResidualBlock(
            n_features=5,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)
        X.requires_grad_(True)
        Y = block(X)
        loss = Y.sum()
        loss.backward()
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()

    def test_autograd_vs_manual_forward(self, device, dtype, generator):
        """Autograd and manual modes produce valid SPD output."""
        block_auto = ResidualBlock(
            n_features=5,
            use_autograd=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen2 = (
            torch.Generator(device=device)
            if device.type == "cuda"
            else torch.Generator()
        )
        gen2.manual_seed(777)
        block_manual = ResidualBlock(
            n_features=5,
            use_autograd=False,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        X = random_SPD(5, 4, device=device, dtype=dtype)
        Y_auto = block_auto(X)
        Y_manual = block_manual(X)
        # Both should produce valid SPD outputs (weights differ due to parametrization)
        assert is_spd(Y_auto)
        assert is_spd(Y_manual)

    def test_autograd_vs_manual_backward(self, device, dtype, generator):
        """Both autograd and manual backward produce finite gradients."""
        block_auto = ResidualBlock(
            n_features=5,
            use_autograd=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen2 = (
            torch.Generator(device=device)
            if device.type == "cuda"
            else torch.Generator()
        )
        gen2.manual_seed(777)
        block_manual = ResidualBlock(
            n_features=5,
            use_autograd=False,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        X_auto = random_SPD(5, 4, device=device, dtype=dtype)
        X_manual = X_auto.clone()
        X_auto.requires_grad_(True)
        X_manual.requires_grad_(True)

        loss_auto = block_auto(X_auto).sum()
        loss_manual = block_manual(X_manual).sum()
        loss_auto.backward()
        loss_manual.backward()

        assert X_auto.grad is not None
        assert X_manual.grad is not None
        assert torch.isfinite(X_auto.grad).all()
        assert torch.isfinite(X_manual.grad).all()

    def test_repr(self, device, dtype, generator):
        """repr is informative."""
        block = ResidualBlock(
            n_features=5,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        r = repr(block)
        assert "ResidualBlock" in r
        assert "n_features=5" in r

    @pytest.mark.parametrize("stiefel_mode", ["static", "dynamic"])
    def test_stiefel_modes(self, stiefel_mode, device, dtype, generator):
        """Block works with both static and dynamic Stiefel parametrization."""
        block = ResidualBlock(
            n_features=5,
            stiefel_parametrization_mode=stiefel_mode,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)
        Y = block(X)
        assert Y.shape == X.shape

    def test_train_eval_modes(self, device, dtype, generator):
        """Block produces valid output in both train and eval modes."""
        block = ResidualBlock(
            n_features=5,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X = random_SPD(5, 4, device=device, dtype=dtype, generator=generator)

        block.train()
        Y_train = block(X)
        assert is_spd(Y_train)

        block.eval()
        Y_eval = block(X)
        assert is_spd(Y_eval)


# ========================================================================
# Ill-conditioned inputs (real covariances, e.g. HyperLeaf: eigenvalues
# spanning 1e-4 .. 3e3). The unnormalized affine-invariant step overflowed.
# ========================================================================
def _ill_conditioned_spd(n, n_matrices, device, dtype, generator):
    """SPD matrices with log-uniform eigenvalues in [1e-4, 3e3]."""
    from yetanotherspdnet.random.stiefel import random_stiefel

    eigvecs = random_stiefel(
        n, n, n_matrices=n_matrices, device=device, dtype=dtype, generator=generator
    )
    log_eigvals = torch.empty(n_matrices, n, device=device, dtype=dtype)
    log_eigvals.uniform_(-9.2, 8.0, generator=generator)  # log(1e-4), log(3e3)
    return (eigvecs * log_eigvals.exp().unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


class TestResidualBlockIllConditioned:
    @pytest.mark.parametrize("metric", ["affine_invariant", "log_euclidean"])
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_finite_spd_and_gradients(
        self, metric, use_autograd, device, dtype, generator
    ):
        X = _ill_conditioned_spd(20, 8, device, dtype, generator)
        X.requires_grad_(True)
        block = ResidualBlock(
            20,
            metric=metric,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Y = block(X)
        assert torch.isfinite(Y).all() and is_spd(Y.detach())
        Y.square().sum().backward()
        grads = [X.grad] + [p.grad for p in block.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)

    def test_affine_invariant_step_has_unit_length(self, device, dtype, generator):
        from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
            affine_invariant_log,
        )
        from yetanotherspdnet.nn.rresnet_layers import affine_invariant_norm

        X = _ill_conditioned_spd(10, 6, device, dtype, generator)
        block = ResidualBlock(10, device=device, dtype=dtype, generator=generator)
        with torch.no_grad():
            step = affine_invariant_norm(X, affine_invariant_log(X, block(X)))
        assert_close(step, torch.ones_like(step), atol=1e-6, rtol=0)

    def test_affine_invariant_norm(self, device, dtype, generator):
        from yetanotherspdnet.functions.spd_linalg import inv_sqrtm_SPD
        from yetanotherspdnet.nn.rresnet_layers import affine_invariant_norm

        X = random_SPD(6, 4, cond=100, device=device, dtype=dtype, generator=generator)
        V = torch.randn(4, 6, 6, device=device, dtype=dtype, generator=generator)
        V = V + V.transpose(-1, -2)
        X_isqrt = inv_sqrtm_SPD(X)[0]
        expected = torch.linalg.matrix_norm(X_isqrt @ V @ X_isqrt)
        assert_close(affine_invariant_norm(X, V), expected)

    def test_log_euclidean_step(self, device, dtype, generator):
        from yetanotherspdnet.functions.spd_linalg import expm_symmetric, logm_SPD

        X = _ill_conditioned_spd(8, 4, device, dtype, generator)
        block = ResidualBlock(
            8, metric="log_euclidean", device=device, dtype=dtype, generator=generator
        )
        with torch.no_grad():
            expected = expm_symmetric(logm_SPD(X)[0] + block.vector_field(X))[0]
            assert_close(block(X), expected)

    @pytest.mark.parametrize("metric", ["affine_invariant", "log_euclidean"])
    def test_gradient_paths_agree(self, metric, device, dtype, generator):
        """Same weights: the manual and autograd paths give the same gradients."""
        X = random_SPD(6, 4, cond=50, device=device, dtype=dtype, generator=generator)
        blocks = [
            ResidualBlock(
                6, metric=metric, use_autograd=use_autograd, device=device, dtype=dtype
            )
            for use_autograd in (True, False)
        ]
        blocks[1].load_state_dict(blocks[0].state_dict())
        outputs, grads = [], []
        for block in blocks:
            block.eval()  # BatchNorm1d of the spectrum map: deterministic
            X_leaf = X.clone().requires_grad_(True)
            Y = block(X_leaf)
            Y.square().sum().backward()
            outputs.append(Y.detach())
            grads.append(X_leaf.grad)
        assert_close(outputs[0], outputs[1])
        assert_close(grads[0], grads[1])

    def test_invalid_metric(self, device, dtype):
        with pytest.raises(AssertionError):
            ResidualBlock(4, metric="bures_wasserstein", device=device, dtype=dtype)
