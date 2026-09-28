import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.nn.batchnorm as batchnorm
from utils import is_spd, is_symmetric
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_mean,
    affine_invariant_std_scalar,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    arithmetic_mean,
    harmonic_mean,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    geometric_arithmetic_harmonic_mean,
)
from yetanotherspdnet.random.spd import random_SPD


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
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=momentum,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        assert layer.n_features == n_features
        assert layer.mean_type == mean_type
        assert layer.mean_options == mean_options
        assert layer.momentum == momentum
        assert layer.norm_strategy == norm_strategy
        assert layer.minibatch_momentum == momentum
        assert layer.use_autograd == use_autograd
        assert layer.device.type == device.type
        assert layer.dtype == dtype
        # check that we have one and only one parameter
        assert len(list(layer.parameters())) == 1
        assert is_spd(layer.Covbias)
        assert_close(layer.Covbias, torch.eye(n_features, device=device, dtype=dtype))

        # TODO: add check that the correct inner functions are selected

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 30}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that forward pass returns correct shape, etc.
        and that the output is actually normalized and biased as expected
        """
        momentum = 0.1
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            momentum=momentum,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        if norm_strategy == "minibatch":
            return  # if minibatch, no checking mean output
        # TODO: would be better to perform this test with layer.Covbias not equal to identity
        if mean_type == "affine_invariant" and mean_options is not None:
            # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
            # a sufficient number of iterations
            G_output = affine_invariant_mean(output, n_iterations=30)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "log_euclidean":
            # This actually does not work. This is normal.
            # It is due to the way normalization is done.
            # G_output = log_euclidean_mean(output)
            # assert_close(G_output, layer.Covbias @ layer.Covbias)
            pass
        elif mean_type == "arithmetic":
            G_output = arithmetic_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "harmonic":
            G_output = harmonic_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
        elif mean_type == "geometric_arithmetic_harmonic":
            G_output = geometric_arithmetic_harmonic_mean(output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    def test_both_modes_give_same_result(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )

        output_manual = layer_manual(X)
        output_auto = layer_auto(X)

        assert_close(output_manual, output_auto)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        if norm_strategy == "minibatch":
            print(layer.get_minibatch_momentum())

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

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_parameter_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    def test_both_modes_give_same_gradient(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

        # check input gradients
        assert_close(X1.grad, X2.grad, rtol=1e-6, atol=1e-6)
        # check Covbias gradient
        # due to parametrization, gradient not directly on layer.weight
        original_bias1 = layer_manual.parametrizations.Covbias.original
        original_bias2 = layer_auto.parametrizations.Covbias.original
        assert_close(original_bias1.grad, original_bias2.grad)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that dynamic parametrization is initialized as expected, i.e.,
        that initial tangent vector is zero
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        Covbias_tangent = layer.parametrizations.Covbias.original
        assert_close(
            Covbias_tangent,
            torch.zeros((n_features, n_features), device=device, dtype=dtype),
        )
        assert_close(layer.Covbias, layer.spd_parametrization.reference_point)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_ref_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that reference point is updated as expected
        """
        layer = batchnorm.BatchNormSPDMean(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            n_steps_ref_update=2,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        layer.train()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        layer.register_optimizer_hook(optimizer)

        initial_ref = layer.spd_parametrization.reference_point.clone().detach()

        # first time going through the layer
        X1 = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X1)
        # check current_ref_step is correct
        assert layer.current_ref_step == 1
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert_close(layer.spd_parametrization.reference_point, initial_ref)

        # second time going through the layer
        X2 = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X2)
        # check current_ref_step is correct
        assert layer.current_ref_step == 2
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check that reference_point changed, tangent vector and current_ref_step re-initialized
        assert not torch.allclose(
            layer.spd_parametrization.reference_point, initial_ref
        )
        assert_close(
            layer.parametrizations.Covbias.original,
            torch.zeros((n_features, n_features), dtype=dtype, device=device),
        )
        assert layer.current_ref_step == 0

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"mean_type={mean_type}" in repr_str
        assert f"mean_options={mean_options}" in repr_str
        assert f"momentum={0.1}" in repr_str
        assert f"norm_strategy={norm_strategy}" in repr_str
        assert f"minibatch_mode={minibatch_mode}" in repr_str
        assert f"minibatch_momentum={0.1}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestBatchNormSPDMeanScalarVariance:
    """
    Test suite for BatchNormSPDMeanScalarVariance module
    """

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that initialization goes as expected
        """
        momentum = 0.1
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=momentum,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        assert layer.n_features == n_features
        assert layer.mean_type == mean_type
        assert layer.mean_options == mean_options
        assert layer.momentum == momentum
        assert layer.norm_strategy == norm_strategy
        assert layer.minibatch_momentum == momentum
        assert layer.use_autograd == use_autograd
        assert layer.device == device
        assert layer.dtype == dtype
        # check that we have one and only one parameter
        assert len(list(layer.parameters())) == 2
        assert is_spd(layer.Covbias)
        assert_close(layer.Covbias, torch.eye(n_features, device=device, dtype=dtype))
        assert layer.stdScalarbias > 0
        assert_close(layer.stdScalarbias, torch.tensor(1.0, device=device, dtype=dtype))

        # TODO: add check that the correct inner functions are selected

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 30}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that forward pass returns correct shape, etc.
        and that the output is actually normalized and biased as expected
        """
        momentum = 0.1
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=momentum,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        if norm_strategy == "minibatch":
            return  # if minibatch, no checking mean output
        # TODO: would be better to perform this test with layer.Covbias not equal to identity
        if mean_type == "affine_invariant" and mean_options is not None:
            # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
            # a sufficient number of iterations
            G_output = affine_invariant_mean(output, n_iterations=30)
            std_output = affine_invariant_std_scalar(output, G_output)
            assert_close(G_output, layer.Covbias @ layer.Covbias)
            assert_close(std_output, torch.tensor(1.0, device=device, dtype=dtype))
        # As is, this test only makes sens for affine-invariant geometry
        # it is approximate for others
        elif mean_type == "log_euclidean":
            # G_output = log_euclidean_mean(output)
            # assert_close(G_output, layer.Covbias @ layer.Covbias)
            pass
        elif mean_type == "arithmetic":
            # G_output = arithmetic_mean(output)
            # assert_close(G_output, layer.Covbias @ layer.Covbias)
            pass
        elif mean_type == "harmonic":
            # G_output = harmonic_mean(output)
            # assert_close(G_output, layer.Covbias @ layer.Covbias)
            pass
        elif mean_type == "geometric_arithmetic_harmonic":
            # G_output = geometric_arithmetic_harmonic_mean(output)
            # assert_close(G_output, layer.Covbias @ layer.Covbias)
            pass

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    def test_both_modes_give_same_result(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        device,
        dtype,
        generator,
    ):
        """
        Test that both autograd and manual modes have same forward output
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer_manual = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )

        output_manual = layer_manual(X)
        output_auto = layer_auto(X)

        assert_close(output_manual, output_auto)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that backward pass works and updates gradients
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        if norm_strategy == "minibatch":
            print(layer.get_minibatch_momentum())

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
        original_Covbias = layer.parametrizations.Covbias.original
        assert original_Covbias.grad is not None
        assert original_Covbias.grad.shape == layer.Covbias.shape
        assert not torch.isnan(original_Covbias.grad).any()
        assert not torch.isinf(original_Covbias.grad).any()
        assert is_symmetric(original_Covbias.grad)
        # check stdScalarbias gradient
        original_stdScalarbias = layer.parametrizations.stdScalarbias.original
        assert original_stdScalarbias.grad is not None
        assert original_stdScalarbias.grad.shape == layer.stdScalarbias.shape
        assert not torch.isnan(original_stdScalarbias.grad)
        assert not torch.isinf(original_stdScalarbias.grad)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_parameter_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that parameters can be updated via optimization
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

        # store initial Covbias and stdScalarbias
        initial_Covbias = layer.Covbias.clone().detach()
        initial_stdScalarbias = layer.stdScalarbias.clone().detach()

        # Forward + backward + update
        output = layer(X)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that Covbias and stdScalarbias changed
        assert not torch.allclose(layer.Covbias, initial_Covbias)
        assert not torch.allclose(layer.stdScalarbias, initial_stdScalarbias)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    def test_both_modes_give_same_gradient(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        device,
        dtype,
        generator,
    ):
        """
        Test that both autograd and manual modes yield same gradient
        """
        layer_manual = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        layer_auto = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

        # check input gradients
        assert_close(X1.grad, X2.grad, rtol=1e-6, atol=1e-6)
        # check Covbias gradient
        # due to parametrization, gradient not directly on layer.weight
        original_Covbias1 = layer_manual.parametrizations.Covbias.original
        original_Covbias2 = layer_auto.parametrizations.Covbias.original
        assert_close(original_Covbias1.grad, original_Covbias2.grad)
        # check stdScalarbias gradient
        original_stdbias1 = layer_manual.parametrizations.stdScalarbias.original
        original_stdbias2 = layer_auto.parametrizations.stdScalarbias.original
        assert_close(original_stdbias1, original_stdbias2)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that dynamic parametrization is initialized as expected, i.e.,
        that initial tangent vector is zero
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        Covbias_tangent = layer.parametrizations.Covbias.original
        assert_close(
            Covbias_tangent,
            torch.zeros((n_features, n_features), device=device, dtype=dtype),
        )
        assert_close(layer.Covbias, layer.spd_parametrization.reference_point)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_ref_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that reference point is updated as expected
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            n_steps_ref_update=2,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        layer.train()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        layer.register_optimizer_hook(optimizer)

        initial_ref = layer.spd_parametrization.reference_point.clone().detach()

        # first time going through the layer
        X1 = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X1)
        # check current_ref_step is correct
        assert layer.current_ref_step == 1
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert_close(layer.spd_parametrization.reference_point, initial_ref)

        # second time going through the layer
        X2 = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X2)
        # check current_ref_step is correct
        assert layer.current_ref_step == 2
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check that reference_point changed, tangent vector and current_ref_step re-initialized
        assert not torch.allclose(
            layer.spd_parametrization.reference_point, initial_ref
        )
        assert_close(
            layer.parametrizations.Covbias.original,
            torch.zeros((n_features, n_features), dtype=dtype, device=device),
        )
        assert layer.current_ref_step == 0

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test string representations
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"mean_type={mean_type}" in repr_str
        assert f"mean_options={mean_options}" in repr_str
        assert f"momentum={0.1}" in repr_str
        assert f"norm_strategy={norm_strategy}" in repr_str
        assert f"minibatch_mode={minibatch_mode}" in repr_str
        assert f"minibatch_momentum={0.1}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
            ("minibatch", "decay"),
            ("minibatch", "growth"),
        ],
    )
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("parametrization_mode", ["static", "dynamic"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
    ):
        """
        Test that module respects train/eval mode
        """
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type,
            mean_options,
            momentum=0.1,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestBatchNormBuresWasserstein:
    """Tests specific to Bures-Wasserstein batch normalization (GBWBN)."""

    @pytest.mark.parametrize("n_features", [10])
    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    @pytest.mark.parametrize("parametrization", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_initialization(
        self, n_features, bw_theta, parametrization, use_autograd, device, dtype
    ):
        """Test that BW batchnorm initializes correctly with extra parameters."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            parametrization=parametrization,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        assert layer.mean_type == "bures_wasserstein"
        assert layer.bw_theta == bw_theta
        assert layer.bw_M.shape == (n_features, n_features)
        assert layer.bw_G.shape == (n_features, n_features)
        eye = torch.eye(n_features, device=device, dtype=dtype)
        assert_close(layer.bw_M, eye)
        assert_close(layer.bw_G, eye)

    @pytest.mark.parametrize("n_matrices", [5])
    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    @pytest.mark.parametrize(
        "norm_strategy, minibatch_mode",
        [
            ("classical", ""),
            ("minibatch", "constant"),
        ],
    )
    @pytest.mark.parametrize("use_autograd", [True])
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        bw_theta,
        norm_strategy,
        minibatch_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """Test that BW batchnorm forward returns SPD matrices of correct shape."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
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

        layer.train()
        output = layer(X)
        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

    @pytest.mark.parametrize("n_matrices", [5])
    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    @pytest.mark.parametrize("use_autograd", [True])
    def test_eval_mode(
        self,
        n_matrices,
        n_features,
        cond,
        bw_theta,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """Test train then eval mode works correctly."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
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

        layer.train()
        for _ in range(3):
            layer(X)

        layer.eval()
        output = layer(X)
        assert output.shape == X.shape
        assert is_spd(output)

    @pytest.mark.parametrize("n_matrices", [5])
    @pytest.mark.parametrize("n_features, cond", [(10, 100)])
    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    @pytest.mark.parametrize("use_autograd", [True])
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        bw_theta,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """Test that gradients flow through BW batchnorm."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
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
        X.requires_grad_(True)

        layer.train()
        output = layer(X)
        loss = output.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.any(torch.isnan(X.grad))

    @pytest.mark.parametrize("n_features", [10])
    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    def test_repr(self, n_features, bw_theta, device, dtype):
        """Test that repr includes bw_theta."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            device=device,
            dtype=dtype,
        )
        r = repr(layer)
        assert "bures_wasserstein" in r
        assert f"bw_theta={bw_theta}" in r

    @pytest.mark.parametrize("n_features", [10])
    def test_n_iterations_option(self, n_features, device, dtype):
        """Test that mean_options n_iterations works for BW."""
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            mean_options={"n_iterations": 3},
            device=device,
            dtype=dtype,
        )
        assert layer.mean_type == "bures_wasserstein"


class TestSingleMatrixBatch:
    """
    A batch holding a single SPD matrix has no batch statistics (the mean is the
    matrix itself, the dispersion is zero). Training-mode forward must fall back
    to the running statistics, leave them untouched, and warn.
    """

    @pytest.mark.parametrize(
        "layer_class, mean_type",
        [
            (batchnorm.BatchNormSPDMean, "affine_invariant"),
            (batchnorm.BatchNormSPDMean, "log_euclidean"),
            (batchnorm.BatchNormSPDMeanScalarVariance, "affine_invariant"),
            (batchnorm.BatchNormSPDMeanScalarVariance, "bures_wasserstein"),
        ],
    )
    @pytest.mark.parametrize("unbatched", [True, False])
    def test_single_matrix_uses_running_statistics(
        self, layer_class, mean_type, unbatched, device, dtype, generator
    ):
        n_features = 5
        layer = layer_class(n_features, mean_type=mean_type, device=device, dtype=dtype)
        data = random_SPD(
            n_features, 1, device=device, dtype=dtype, generator=generator
        )
        if not unbatched:
            data = data.unsqueeze(0)
        state_before = {k: v.clone() for k, v in layer.state_dict().items()}
        step_before = layer.training_step

        layer.train()
        with pytest.warns(UserWarning, match="single matrix"):
            out_train = layer(data)
        layer.eval()
        out_eval = layer(data)

        assert_close(out_train, out_eval)
        assert layer.training_step == step_before
        for key, value in layer.state_dict().items():
            assert_close(value, state_before[key])
        # the output is not collapsed to the identity
        eye = torch.eye(n_features, device=device, dtype=dtype)
        assert not torch.allclose(out_train.reshape(n_features, n_features), eye)
        assert is_spd(out_train)


class TestBuresWassersteinGradients:
    """
    GBWBN parameters bw_M and bw_G start at the identity; their gradients
    must be finite and correct there (they were NaN through autograd eigh).
    """

    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    def test_parameters_train(self, bw_theta, device, dtype, generator):
        n_features = 5
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            device=device,
            dtype=dtype,
        )
        data = random_SPD(
            n_features, 8, cond=10, device=device, dtype=dtype, generator=generator
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            layer(data).square().sum().backward()
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), name
            optimizer.step()
        assert is_spd(layer.bw_M) and is_spd(layer.bw_G)

    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    def test_parameters_gradcheck_at_init(self, bw_theta, device, dtype, generator):
        from torch.func import functional_call

        n_features = 4
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            device=device,
            dtype=dtype,
        )
        data = random_SPD(
            n_features, 6, cond=10, device=device, dtype=dtype, generator=generator
        )
        params = dict(layer.named_parameters())
        names = (
            "parametrizations.bw_M.original",
            "parametrizations.bw_G.original",
        )

        def fun(m_original, g_original):
            overrides = dict(params)
            overrides[names[0]] = 0.5 * (m_original + m_original.T)
            overrides[names[1]] = 0.5 * (g_original + g_original.T)
            return functional_call(layer, overrides, (data,))

        inputs = tuple(params[n].detach().clone().requires_grad_() for n in names)
        assert torch.autograd.gradcheck(fun, inputs, atol=1e-6)


def _spectral(matrix: torch.Tensor, fun) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    return (eigvecs * fun(eigvals).unsqueeze(-2)) @ eigvecs.mT


def _gbwbn_reference(data, M, G, theta, shift, eps=1e-5):
    """Algorithm 1 of the GBWBN paper, written independently of the library
    in the style of the reference code (jjscc/GBWBN): pal1, scale1, pal2, ExpG.
    Batch statistics: one fixed-point iteration from the arithmetic mean and
    the Frechet variance of the theta-deformed metric."""
    eye = torch.eye(data.shape[-1], dtype=data.dtype, device=data.device)
    M_isqrt = _spectral(M, lambda s: s.rsqrt())
    M_sqrt = _spectral(M, torch.sqrt)
    X = M_isqrt @ _spectral(data, lambda s: s**theta) @ M_isqrt
    G_hat = M_isqrt @ _spectral(G, lambda s: s**theta) @ M_isqrt
    # mean: one fixed-point iteration
    G0 = X.mean(0)
    G0_sqrt, G0_isqrt = _spectral(G0, torch.sqrt), _spectral(G0, lambda s: s.rsqrt())
    T = _spectral(G0_sqrt @ X @ G0_sqrt, torch.sqrt).mean(0)
    B = G0_isqrt @ T @ T @ G0_isqrt
    B_sqrt, B_isqrt = _spectral(B, torch.sqrt), _spectral(B, lambda s: s.rsqrt())
    # variance of the deformed metric: d_BW^2(B, X_i) / theta^2
    cross = _spectral(B_sqrt @ X @ B_sqrt, torch.sqrt)
    tr = lambda A: torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)  # noqa: E731
    var = (tr(B) + tr(X) - 2 * tr(cross)).mean() / theta**2
    # centering: Log_B, transport B -> I, Exp_I   (LogG, pal1)
    C = B_sqrt @ cross @ B_isqrt
    V = C + C.mT - 2 * B
    b, U = torch.linalg.eigh(B)
    V = U @ (torch.sqrt(2 / (b[:, None] + b[None, :])) * (U.mT @ V @ U)) @ U.mT
    X = eye + V + V @ V / 4
    # scaling at I   (scale1)
    V = shift / torch.sqrt(var + eps) * (2 * _spectral(X, torch.sqrt) - 2 * eye)
    X = eye + V + V @ V / 4
    # bias: transport I -> G_hat, Exp_{G_hat}   (pal2, ExpG)
    V = 2 * _spectral(X, torch.sqrt) - 2 * eye
    a, P = torch.linalg.eigh(G_hat)
    V = P @ (torch.sqrt((a[:, None] + a[None, :]) / 2) * (P.mT @ V @ P)) @ P.mT
    Z = P @ ((P.mT @ V @ P) / (a[:, None] + a[None, :])) @ P.mT
    X = G_hat + V + Z @ G_hat @ Z
    return _spectral(M_sqrt @ X @ M_sqrt, lambda s: s ** (1 / theta))


class TestGBWBNConformity:
    """GBWBN against the paper's Algorithm 1 (Wang et al., 2025)."""

    @pytest.mark.parametrize("bw_theta", [1.0, 0.75, 0.5, 0.25])
    @pytest.mark.parametrize("use_autograd", [False, True])
    def test_matches_algorithm_1(self, bw_theta, use_autograd, generator):
        n_features, dtype = 5, torch.float64
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_theta=bw_theta,
            use_autograd=use_autograd,
            dtype=dtype,
        )
        M = random_SPD(n_features, 1, cond=5, dtype=dtype, generator=generator)
        G = random_SPD(n_features, 1, cond=5, dtype=dtype, generator=generator)
        with torch.no_grad():
            layer.bw_M = M
            layer.bw_G = G
            layer.stdScalarbias = torch.tensor(1.7, dtype=dtype)
        data = random_SPD(n_features, 8, cond=20, dtype=dtype, generator=generator)
        expected = _gbwbn_reference(data, M, G, bw_theta, shift=1.7)
        assert_close(layer(data), expected, rtol=1e-8, atol=1e-10)

    def test_default_theta_is_paper_best(self):
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            4, mean_type="bures_wasserstein"
        )
        assert layer.bw_theta == 0.5 and layer.bw_batch_stats_grad is True

    @pytest.mark.parametrize("bw_batch_stats_grad", [True, False])
    def test_batch_statistics_gradient_switch(self, bw_batch_stats_grad, generator):
        """With bw_batch_stats_grad=False (reference code), the batch mean and
        variance are constants: the output depends on each input only through
        its own normalization, so the Jacobian of output i w.r.t. input j != i
        vanishes."""
        n_features, dtype = 4, torch.float64
        layer = batchnorm.BatchNormSPDMeanScalarVariance(
            n_features,
            mean_type="bures_wasserstein",
            bw_batch_stats_grad=bw_batch_stats_grad,
            dtype=dtype,
        )
        data = random_SPD(n_features, 6, cond=10, dtype=dtype, generator=generator)
        data.requires_grad_(True)
        layer(data)[0].sum().backward()
        cross = data.grad[1:].abs().max()
        if bw_batch_stats_grad:
            assert cross > 1e-6
        else:
            assert cross == 0

    def test_same_output_for_both_gradient_modes(self, generator):
        dtype = torch.float64
        data = random_SPD(4, 6, cond=10, dtype=dtype, generator=generator)
        outputs = [
            batchnorm.BatchNormSPDMeanScalarVariance(
                4, mean_type="bures_wasserstein", bw_batch_stats_grad=flag, dtype=dtype
            )(data)
            for flag in (True, False)
        ]
        assert_close(outputs[0], outputs[1])

    @pytest.mark.parametrize("bw_theta", [1.0, 0.5])
    def test_models_forward_bw_options(self, bw_theta):
        from yetanotherspdnet.model import GBWBNRResNet, RResNet, SPDnet

        options = {"bw_theta": bw_theta, "bw_batch_stats_grad": False}
        common = {
            "batchnorm": True,
            "batchnorm_type": "mean_var_scalar",
            "batchnorm_mean_type": "bures_wasserstein",
            "batchnorm_bw_options": options,
        }
        models = [
            SPDnet(8, [6, 4], 3, **common),
            GBWBNRResNet(8, 6, 3, **common),
            RResNet(8, [6, 4], [1, 1], 3, **common),
        ]
        for model in models:
            layers = [
                m
                for m in model.modules()
                if isinstance(m, batchnorm.BatchNormSPDMeanScalarVariance)
            ]
            assert layers
            assert all(
                m.bw_theta == bw_theta and m.bw_batch_stats_grad is False
                for m in layers
            )
