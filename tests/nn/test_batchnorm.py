import pytest

import torch
from torch.testing import assert_close

from functools import partial

from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    AffineInvariantMean,
    affine_invariant_geodesic,
    affine_invariant_mean,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    arithmetic_mean,
    euclidean_geodesic,
    harmonic_curve,
    harmonic_mean,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    geometric_arithmetic_harmonic_mean,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    log_euclidean_geodesic,
    log_euclidean_mean,
)

import yetanotherspdnet.nn.batchnorm as batchnorm
from yetanotherspdnet.random.spd import random_SPD

from utils import is_spd, is_symmetric


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


class TestBatchNormSPDMean:
    """
    Test suite for BatchNormSPDMean module
    """

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that initialization goes as expected
        """
        momentum = 0.1
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=momentum,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        assert layer.n_features == n_features
        assert layer.mean_type == mean_type
        assert layer.mean_options == mean_options
        assert layer.adaptive_mean_type == adaptive_mean_type
        assert layer.momentum == momentum
        assert layer.use_autograd == use_autograd
        assert layer.device == device
        assert layer.dtype == dtype
        # check that we have one and only one parameter
        assert len(list(layer.parameters())) == 1
        assert is_spd(layer.Covbias)
        assert_close(layer.Covbias, torch.eye(n_features, device=device, dtype=dtype))

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 30}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that forward pass returns correct shape, etc.
        and that the output is actually normalized and biased as expected
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        output = layer(X)

        assert is_spd(layer.Covbias)
        assert_close(layer.Covbias, torch.eye(n_features, device=device, dtype=dtype))

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

        # check output normalization and bias
        # TODO: would be better to perform this test with layer.Covbias not equal to identity
        if mean_type == "affine_invariant" and mean_options is not None:
            # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
            # a sufficient number of iterations
            G_output = affine_invariant_mean(output, n_iterations=30)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "log_euclidean":
            # This actually does not work. This is normal.
            # It is due to the way normalization is done.
            G_output = log_euclidean_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "arithmetic":
            G_output = arithmetic_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "harmonic":
            G_output = harmonic_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "geometric_arithmetic_harmonic":
            G_output = geometric_arithmetic_harmonic_mean(output)[0]
            assert_close(G_output, layer.Covbias @ layer.Covbias)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    def test_both_modes_give_same_result(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        adaptive_mean_type,
        device,
        dtype,
        generator,
    ):
        """
        Test that both autograd and manual gradient yield same output
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer_manual = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )

        output_manual = layer_manual(X)
        output_auto = layer_auto(X)

        assert_close(output_manual, output_auto)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that backward pass works and updates gradients
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X.requires_grad = True

        output = layer(X)
        loss = torch.norm(output)
        loss.backward()

        # check input gradient
        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()
        assert is_symmetric(X.grad)
        # check Covbias gradient
        original_bias = layer.parametrizations.Covbias.original
        assert original_bias.grad is not None
        assert original_bias.grad.shape == layer.Covbias.shape
        assert not torch.isnan(original_bias.grad).any()
        assert not torch.isinf(original_bias.grad).any()
        assert is_symmetric(original_bias.grad)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_parameter_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that parameters can be updated via optimization
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # store initial Covbias
        initial_Covbias = layer.Covbias.clone().detach()

        # Forward + backward + update
        output = layer(X)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that Covbias changed
        assert not torch.allclose(layer.Covbias, initial_Covbias)

    @pytest.mark.parametrize("n_matrices", [1, 10])
    @pytest.mark.parametrize("n_features, cond", [(5, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    def test_both_modes_give_same_gradient(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        adaptive_mean_type,
        device,
        dtype,
        generator,
    ):
        """
        Test that both autograd and manual modes yield same gradient
        """
        layer_manual = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )
        X1 = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        # Forward + backward + update
        output_manual = layer_manual(X1)
        loss_manual = output_manual.sum()
        loss_manual.backward()

        output_auto = layer_auto(X2)
        loss_auto = output_auto.sum()
        loss_auto.backward()

        print(X1.grad[0])
        print(X2.grad[0])

        # check input gradients
        assert_close(X1.grad, X2.grad, rtol=1e-6, atol=1e-6)
        # check Covbias gradient
        # due to parametrization, gradient not directly on layer.weight
        original_bias1 = layer_manual.parametrizations.Covbias.original
        original_bias2 = layer_auto.parametrizations.Covbias.original
        assert_close(original_bias1.grad, original_bias2.grad)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(
        self,
        n_features,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test string representations
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"mean_type={mean_type}" in repr_str
        assert f"mean_options={mean_options}" in repr_str
        assert f"adaptive_mean_type={adaptive_mean_type}" in repr_str
        assert f"momentum={0.1}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options, adaptive_mean_type",
        [
            ("affine_invariant", None, "affine_invariant"),
            ("affine_invariant", {"n_iterations": 5}, "affine_invariant"),
            ("log_euclidean", None, "log_euclidean"),
            ("arithmetic", None, "arithmetic"),
            ("harmonic", None, "harmonic"),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_simple",
            ),
            (
                "geometric_arithmetic_harmonic",
                None,
                "geometric_arithmetic_harmonic_exact",
            ),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(
        self,
        n_features,
        mean_type,
        mean_options,
        adaptive_mean_type,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that module respects train/eval mode
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            adaptive_mean_type,
            momentum=0.1,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False
