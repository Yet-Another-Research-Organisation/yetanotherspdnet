import pytest
import torch
from torch.testing import assert_close
from torch.nn.utils.parametrize import is_parametrized

from yetanotherspdnet.functions.spd_geometries.affine_invariant_gevd import (
    affine_invariant_mean_gevd,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    log_euclidean_mean,
    log_euclidean_std_scalar,
)
import yetanotherspdnet.nn.batchnorm as batchnorm
from utils import is_spd, is_symmetric
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_mean,
    affine_invariant_std_scalar,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    arithmetic_mean,
    harmonic_mean,
    left_kullback_leibler_std_scalar,
    right_kullback_leibler_std_scalar,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    geometric_arithmetic_harmonic_mean,
    symmetrized_kullback_leibler_std_scalar,
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
        )

        assert layer.n_features == n_features
        assert layer.mean_type == mean_type
        assert layer.mean_options == mean_options
        assert layer.momentum == momentum
        assert layer.norm_only == norm_only
        assert layer.norm_and_bias_mode == norm_and_bias_mode
        assert layer.norm_strategy == norm_strategy
        assert layer.minibatch_mode == minibatch_mode
        assert layer.minibatch_momentum == momentum
        assert layer.parametrization == parametrization
        assert layer.parametrization_mode == parametrization_mode
        assert layer.device.type == device.type
        assert layer.dtype == dtype
        if norm_only:
            # check no parameters in that case
            assert len(list(layer.parameters())) == 0
        else:
            # check that we have one and only one parameter
            assert len(list(layer.parameters())) == 1
            assert is_spd(layer.Covbias)
            assert_close(
                layer.Covbias, torch.eye(n_features, device=device, dtype=dtype)
            )

        # TODO: add check that the correct inner functions are selected ?

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 30}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        if norm_only:
            assert not isinstance(layer.Covbias, torch.nn.Parameter)
            assert not is_parametrized(layer, "Covbias")
            assert not layer.Covbias.requires_grad
            # no learnable bias among the parameters, and nothing got a gradient
            assert all("Covbias" not in name for name, _ in layer.named_parameters())
        else:
            assert is_parametrized(layer, "Covbias")
            assert isinstance(
                layer.parametrizations.Covbias.original, torch.nn.Parameter
            )
            assert layer.Covbias.requires_grad
            # no learnable bias among the parameters, and nothing got a gradient
            assert any("Covbias" in name for name, _ in layer.named_parameters())

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

        # check output normalization and bias
        if norm_strategy == "minibatch":
            return  # if minibatch, no checking mean output
        # TODO: would be better to perform this test with layer.Covbias not equal to identity
        if mean_type == "affine_invariant":
            if mean_options is not None:
                # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
                # a sufficient number of iterations
                G_output = affine_invariant_mean(output, n_iterations=30)
            else:
                return
        if mean_type == "affine_invariant_gevd":
            if mean_options is not None:
                # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
                # a sufficient number of iterations
                G_output = affine_invariant_mean_gevd(output, n_iterations=30)
            else:
                return
        elif mean_type == "log_euclidean":
            # This work only when norm_and_bias_mode is "parallel_transport"
            if norm_and_bias_mode == "parallel_transport":
                G_output = log_euclidean_mean(output)
            else:
                return
        elif mean_type == "arithmetic":
            G_output = arithmetic_mean(output)
        elif mean_type == "harmonic":
            G_output = harmonic_mean(output)
        elif mean_type == "geometric_arithmetic_harmonic":
            G_output = geometric_arithmetic_harmonic_mean(output)

        if norm_only:
            assert_close(
                G_output,
                torch.eye(n_features, dtype=dtype, device=device),
                atol=1e-3,  # necessary for gevd mean
                rtol=1e-4,
            )
        else:
            assert_close(G_output, layer.Covbias, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        if norm_only:
            assert_close(
                layer.Covbias, torch.eye(n_features, device=device, dtype=dtype)
            )
            assert not isinstance(layer.Covbias, torch.nn.Parameter)
            assert not layer.Covbias.requires_grad
            assert not is_parametrized(layer, "Covbias")
            # no learnable bias among the parameters, and nothing got a gradient
            assert all("Covbias" not in name for name, _ in layer.named_parameters())
            assert all(p.grad is None for p in layer.parameters())
        else:
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_parameter_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_dynamic_parametrization_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_dynamic_parametrization_ref_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            n_steps_ref_update=2,
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_repr_and_str(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"mean_type={mean_type}" in repr_str
        assert f"mean_options={mean_options}" in repr_str
        assert f"momentum={0.1}" in repr_str
        assert f"norm_only={norm_only}" in repr_str
        assert f"norm_and_bias_mode={norm_and_bias_mode}" in repr_str
        assert f"norm_strategy={norm_strategy}" in repr_str
        assert f"minibatch_mode={minibatch_mode}" in repr_str
        assert f"minibatch_momentum={0.1}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_module_mode(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
        )

        assert layer.n_features == n_features
        assert layer.mean_type == mean_type
        assert layer.mean_options == mean_options
        assert layer.momentum == momentum
        assert layer.norm_only == norm_only
        assert layer.norm_and_bias_mode == norm_and_bias_mode
        assert layer.norm_strategy == norm_strategy
        assert layer.minibatch_mode == minibatch_mode
        assert layer.minibatch_momentum == momentum
        assert layer.parametrization == parametrization
        assert layer.parametrization_mode == parametrization_mode
        assert layer.device.type == device.type
        assert layer.dtype == dtype

        assert is_spd(layer.Covbias)
        assert_close(layer.Covbias, torch.eye(n_features, device=device, dtype=dtype))
        assert layer.stdScalarbias > 0
        assert_close(layer.stdScalarbias, torch.tensor(1.0, device=device, dtype=dtype))

        if norm_only:
            # check no parameters in that case
            assert len(list(layer.parameters())) == 0
        else:
            # check that we have two and only two parameters
            assert len(list(layer.parameters())) == 2

        if norm_only:
            # Covbias
            assert not isinstance(layer.Covbias, torch.nn.Parameter)
            assert not is_parametrized(layer, "Covbias")
            assert not layer.Covbias.requires_grad
            # no learnable bias among the parameters, and nothing got a gradient
            assert all("Covbias" not in name for name, _ in layer.named_parameters())
            # stdScalarbias
            assert not isinstance(layer.stdScalarbias, torch.nn.Parameter)
            assert not is_parametrized(layer, "stdScalarbias")
            assert not layer.stdScalarbias.requires_grad
            # no learnable bias among the parameters, and nothing got a gradient
            assert all(
                "stdScalarbias" not in name for name, _ in layer.named_parameters()
            )
        else:
            # Covbias
            assert is_parametrized(layer, "Covbias")
            assert isinstance(
                layer.parametrizations.Covbias.original, torch.nn.Parameter
            )
            assert layer.Covbias.requires_grad
            # learnable Covbias among the parameters
            assert any("Covbias" in name for name, _ in layer.named_parameters())
            # stdScalarbias
            assert is_parametrized(layer, "stdScalarbias")
            assert isinstance(
                layer.parametrizations.stdScalarbias.original, torch.nn.Parameter
            )
            assert layer.stdScalarbias.requires_grad
            # learnable stdScalarbias among the parameters
            assert any("stdScalarbias" in name for name, _ in layer.named_parameters())

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 30}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_forward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=momentum,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

        # check output normalization and bias
        if norm_strategy == "minibatch":
            return  # if minibatch, no checking mean output
        # TODO: would be better to perform this test with layer.Covbias not equal to identity
        # and layer.stdScalarbias not equal to one
        if mean_type == "affine_invariant":
            if mean_options is not None:
                # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
                # a sufficient number of iterations
                G_output = affine_invariant_mean(output, n_iterations=30)
                std_output = affine_invariant_std_scalar(output, G_output)
            else:
                return
        if mean_type == "affine_invariant_gevd":
            if mean_options is not None:
                # for affine-invariant mean, we can do this check only if good quality estimation, i.e.,
                # a sufficient number of iterations
                G_output = affine_invariant_mean_gevd(output, n_iterations=30)
                std_output = affine_invariant_std_scalar(output, G_output)
            else:
                return
        elif mean_type == "log_euclidean":
            # This work only when norm_and_bias_mode is "parallel_transport"
            if norm_and_bias_mode == "parallel_transport":
                G_output = log_euclidean_mean(output)
                std_output = log_euclidean_std_scalar(output, G_output)
            else:
                return
        elif mean_type == "arithmetic":
            G_output = arithmetic_mean(output)
            std_output = left_kullback_leibler_std_scalar(output, G_output)
            # The std will not be normalized exactly, exiting now
            return
        elif mean_type == "harmonic":
            G_output = harmonic_mean(output)
            std_output = right_kullback_leibler_std_scalar(output, G_output)
            # The std will not be normalized exactly, exiting now
            return
        elif mean_type == "geometric_arithmetic_harmonic":
            G_output = geometric_arithmetic_harmonic_mean(output)
            std_output = symmetrized_kullback_leibler_std_scalar(output, G_output)
            # The std will not be normalized exactly, exiting now
            # Note : normalization appears closer as compared to arithmetci and harmonic
            return

        if norm_only:
            assert_close(
                G_output,
                torch.eye(n_features, dtype=dtype, device=device),
                atol=1e-3,  # necessary for gevd mean
                rtol=1e-4,
            )
            assert_close(
                std_output,
                torch.tensor(1.0, device=device, dtype=dtype),
                atol=1e-5,  # necessary for gevd mean
                rtol=1e-5,
            )
        else:
            assert_close(G_output, layer.Covbias, atol=1e-3, rtol=1e-4)
            assert_close(std_output, layer.stdScalarbias, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("n_matrices", [10])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_backward_pass(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        if norm_only:
            # Covbias
            assert_close(
                layer.Covbias, torch.eye(n_features, device=device, dtype=dtype)
            )
            assert not isinstance(layer.Covbias, torch.nn.Parameter)
            assert not layer.Covbias.requires_grad
            assert not is_parametrized(layer, "Covbias")
            assert all("Covbias" not in name for name, _ in layer.named_parameters())
            # stdScalarbias
            assert_close(
                layer.stdScalarbias, torch.ones((), device=device, dtype=dtype)
            )
            assert not isinstance(layer.stdScalarbias, torch.nn.Parameter)
            assert not layer.stdScalarbias.requires_grad
            assert not is_parametrized(layer, "stdScalarbias")
            assert all(
                "stdScalarbias" not in name for name, _ in layer.named_parameters()
            )
            # nothing get a gradient inside the layer
            assert all(p.grad is None for p in layer.parameters())
        else:
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_parameter_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
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
        loss = (output**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that Covbias and stdScalarbias changed
        assert not torch.allclose(layer.Covbias, initial_Covbias)
        assert not torch.allclose(layer.stdScalarbias, initial_stdScalarbias)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_dynamic_parametrization_initialization(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_dynamic_parametrization_ref_update(
        self,
        n_matrices,
        n_features,
        cond,
        mean_type,
        mean_options,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
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
            norm_only=False,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode="dynamic",
            n_steps_ref_update=2,
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
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_repr_and_str(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"mean_type={mean_type}" in repr_str
        assert f"mean_options={mean_options}" in repr_str
        assert f"momentum={0.1}" in repr_str
        assert f"norm_only={norm_only}" in repr_str
        assert f"norm_and_bias_mode={norm_and_bias_mode}" in repr_str
        assert f"norm_strategy={norm_strategy}" in repr_str
        assert f"minibatch_mode={minibatch_mode}" in repr_str
        assert f"minibatch_momentum={0.1}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize(
        "mean_type, mean_options",
        [
            ("affine_invariant", None),
            ("affine_invariant", {"n_iterations": 5}),
            ("affine_invariant_gevd", None),
            ("affine_invariant_gevd", {"n_iterations": 5}),
            ("log_euclidean", None),
            ("arithmetic", None),
            ("harmonic", None),
            (
                "geometric_arithmetic_harmonic",
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("norm_only", [True, False])
    @pytest.mark.parametrize("norm_and_bias_mode", ["whitening", "parallel_transport"])
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
    def test_module_mode(
        self,
        n_features,
        mean_type,
        mean_options,
        norm_only,
        norm_and_bias_mode,
        norm_strategy,
        minibatch_mode,
        parametrization,
        parametrization_mode,
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
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=0.1,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
        )

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False
