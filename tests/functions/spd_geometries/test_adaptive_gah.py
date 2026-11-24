"""
Test file for Adaptive Geometric-Arithmetic-Harmonic mean functions.

Tests:
1. Verify that AdaptiveGeometricArithmeticHarmonicMean gives the same result as
   GeometricArithmeticHarmonicMean when t=0.5
2. Test BatchNorm instantiation with the new adaptive_geometric_arithmetic_harmonic mean type
"""

import pytest
import torch

from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    HarmonicMean,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    AdaptiveGeometricArithmeticHarmonicMean,
    GeometricArithmeticHarmonicMean,
    adaptive_geometric_arithmetic_harmonic_mean,
    geometric_arithmetic_harmonic_mean,
)
from yetanotherspdnet.nn.batchnorm import BatchNormSPDMean
from yetanotherspdnet.random.spd import random_SPD


class TestAdaptiveGAHMean:
    """Tests for Adaptive Geometric-Arithmetic-Harmonic mean functions"""

    @pytest.fixture
    def random_spd_batch(self):
        """Generate random SPD matrices for testing"""
        torch.manual_seed(42)
        n_matrices = 10
        n_features = 4
        return random_SPD(n_features, n_matrices, cond=10, dtype=torch.float64)

    def test_adaptive_gah_equals_gah_at_t_05(self, random_spd_batch):
        """
        Test that AdaptiveGeometricArithmeticHarmonicMean gives the same result as
        GeometricArithmeticHarmonicMean when t=0.5
        """
        data = random_spd_batch
        t = torch.tensor(0.5, dtype=torch.float64)

        # Compute with adaptive function (autograd version)
        mean_adaptive = adaptive_geometric_arithmetic_harmonic_mean(data, t)

        # Compute with standard function
        mean_standard = geometric_arithmetic_harmonic_mean(data)

        # They should be equal
        assert torch.allclose(mean_adaptive, mean_standard, rtol=1e-10, atol=1e-10), (
            f"Adaptive GAH mean (t=0.5) should equal standard GAH mean.\n"
            f"Max diff: {(mean_adaptive - mean_standard).abs().max()}"
        )

    def test_adaptive_gah_function_equals_gah_at_t_05(self, random_spd_batch):
        """
        Test that AdaptiveGeometricArithmeticHarmonicMean (Function class) gives the same result as
        GeometricArithmeticHarmonicMean when t=0.5
        """
        data = random_spd_batch
        t = torch.tensor(0.5, dtype=torch.float64)

        # Compute with adaptive Function class
        mean_adaptive = AdaptiveGeometricArithmeticHarmonicMean(data, t)

        # Compute with standard Function class
        mean_standard = GeometricArithmeticHarmonicMean(data)

        # They should be equal
        assert torch.allclose(mean_adaptive, mean_standard, rtol=1e-10, atol=1e-10), (
            f"Adaptive GAH mean Function (t=0.5) should equal standard GAH mean.\n"
            f"Max diff: {(mean_adaptive - mean_standard).abs().max()}"
        )

    def test_different_t_values(self, random_spd_batch):
        """
        Test that different t values produce different results.
        """
        data = random_spd_batch

        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []

        for t_val in t_values:
            t = torch.tensor(t_val, dtype=torch.float64)
            result = AdaptiveGeometricArithmeticHarmonicMean(data, t)
            results.append(result)

        # Check that t=0 gives harmonic mean
        mean_harmonic = HarmonicMean.apply(data)
        assert torch.allclose(results[0], mean_harmonic, rtol=1e-10, atol=1e-10), (
            "t=0 should give harmonic mean"
        )

        # Check that t=1 gives arithmetic mean
        mean_arithmetic = ArithmeticMean.apply(data)
        assert torch.allclose(results[4], mean_arithmetic, rtol=1e-10, atol=1e-10), (
            "t=1 should give arithmetic mean"
        )

        # Check that intermediate values are different from each other
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not torch.allclose(
                    results[i], results[j], rtol=1e-6, atol=1e-6
                ), f"t={t_values[i]} and t={t_values[j]} should give different results"


class TestBatchNormAdaptiveGAH:
    """Tests for BatchNorm with adaptive_geometric_arithmetic_harmonic mean type"""

    @pytest.fixture
    def random_spd_batch(self):
        """Generate random SPD matrices for testing"""
        torch.manual_seed(42)
        n_matrices = 10
        n_features = 4
        return random_SPD(n_features, n_matrices, cond=10, dtype=torch.float64)

    def test_batchnorm_instantiation(self):
        """
        Test that BatchNormSPDMean can be instantiated with adaptive_geometric_arithmetic_harmonic.
        """
        n_features = 4

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        # Check that t_gah parameter exists
        assert hasattr(bn, "t_gah"), "BatchNorm should have t_gah parameter"
        assert bn.t_gah.requires_grad, "t_gah should be a learnable parameter"
        assert torch.isclose(bn.t_gah, torch.tensor(0.5, dtype=torch.float64)), (
            "t_gah should be initialized to 0.5"
        )

    def test_batchnorm_forward_pass(self, random_spd_batch):
        """
        Test that BatchNormSPDMean forward pass works with adaptive_geometric_arithmetic_harmonic.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        bn.train()
        output = bn(data)

        # Check output shape
        assert output.shape == data.shape, "Output shape should match input shape"

        # Check that output consists of SPD matrices
        eigvals = torch.linalg.eigvalsh(output)
        assert torch.all(eigvals > 0), "Output should be SPD matrices"

    def test_batchnorm_gradient_through_t(self, random_spd_batch):
        """
        Test that gradients flow through t_gah during training.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        bn.train()
        output = bn(data)

        # Simple loss function
        loss = output.sum()
        loss.backward()

        # With parametrization, the gradient is on the underlying parameter
        # Access it via parametrizations.t_gah.original
        underlying_param = bn.parametrizations.t_gah.original
        assert underlying_param.grad is not None, (
            "underlying t_gah parameter should have gradient"
        )
        assert underlying_param.grad != 0, (
            "underlying t_gah gradient should be non-zero"
        )

    def test_batchnorm_with_autograd(self, random_spd_batch):
        """
        Test BatchNormSPDMean with use_autograd=True.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            use_autograd=True,
            dtype=torch.float64,
        )

        bn.train()
        output = bn(data)

        # Check output shape
        assert output.shape == data.shape, "Output shape should match input shape"

        # Check gradient flow
        loss = output.sum()
        loss.backward()
        # With parametrization, the gradient is on the underlying parameter
        underlying_param = bn.parametrizations.t_gah.original
        assert underlying_param.grad is not None, (
            "underlying t_gah parameter should have gradient with use_autograd=True"
        )

    def test_batchnorm_eval_mode(self, random_spd_batch):
        """
        Test that BatchNormSPDMean works in eval mode.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        # Run a few training iterations to update running_mean
        bn.train()
        for _ in range(3):
            _ = bn(data)

        # Switch to eval mode
        bn.eval()
        output = bn(data)

        # Check output shape
        assert output.shape == data.shape, (
            "Output shape should match input shape in eval mode"
        )

    def test_batchnorm_t_update_during_optimization(self, random_spd_batch):
        """
        Test that t_gah can be updated during optimization.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        optimizer = torch.optim.SGD(bn.parameters(), lr=0.1)

        t_initial = bn.t_gah.clone()

        bn.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = bn(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Check that t_gah has been updated
        assert not torch.isclose(bn.t_gah, t_initial), (
            "t_gah should be updated after optimization steps"
        )

    def test_batchnorm_t_constrained_to_01(self, random_spd_batch):
        """
        Test that t_gah remains in [0, 1] even with extreme gradients.
        """
        data = random_spd_batch
        n_features = data.shape[-1]

        bn = BatchNormSPDMean(
            n_features=n_features,
            mean_type="adaptive_geometric_arithmetic_harmonic",
            momentum=0.01,
            dtype=torch.float64,
        )

        # Use a very high learning rate to try to push t outside [0, 1]
        optimizer = torch.optim.SGD(bn.parameters(), lr=10.0)

        bn.train()
        for _ in range(100):
            optimizer.zero_grad()
            output = bn(data)
            # Use a loss that would push t in one direction
            loss = output.sum()
            loss.backward()
            optimizer.step()

            # Check constraint is maintained after each step
            assert 0.0 <= bn.t_gah.item() <= 1.0, (
                f"t_gah should remain in [0, 1], got {bn.t_gah.item()}"
            )

        # Final check
        assert 0.0 <= bn.t_gah.item() <= 1.0, (
            f"t_gah should remain in [0, 1] after optimization, got {bn.t_gah.item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
