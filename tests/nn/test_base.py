import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.nn.base as nn_spd_base
from utils import is_orthogonal, is_spd, is_symmetric
from yetanotherspdnet.functions.spd_linalg import symmetrize
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


class TestBiMap:
    """
    Test suite for BiMap module
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_initialization(
        self,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that initialization goes as expected

        Note: all initialization options are not explored because some of them are only here to
              ensure better modularity.
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        assert layer.n_in == n_in
        assert layer.n_out == n_out
        if parametrized:
            assert layer.parametrization_mode is parametrization_mode
        assert layer.weight.shape == (n_in, n_out)
        assert layer.weight.dtype == dtype
        assert layer.weight.device.type == device.type
        assert layer.weight.requires_grad is True
        assert is_orthogonal(layer.weight)
        # check that we have one and only one parameter
        assert len(list(layer.parameters())) == 1

    def test_invalid_dimensions(self, device, dtype, generator):
        """
        Test that n_out > n_in raises AssertionError
        """
        with pytest.raises(AssertionError, match="must have n_out <= n_in"):
            nn_spd_base.BiMap(
                n_in=5, n_out=10, dtype=dtype, device=device, generator=generator
            )

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self,
        n_matrices,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        X = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )

        output = layer(X)

        # As is, even if not parametrized, weight initialization is orthogonal but it feels better
        # not to ensure orthogonality in testing when not parametrized
        if parametrized:
            assert is_orthogonal(layer.weight)

        assert output.shape == (*X.shape[:-2], n_out, n_out)
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    def test_both_modes_give_same_result(
        self,
        n_matrices,
        n_in,
        parametrized,
        parametrization_mode,
        n_out,
        device,
        dtype,
        generator,
        seed,
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        X = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )

        # we need to reset the generator seed to ensure we generate same weights in both case
        generator.manual_seed(seed)
        layer_manual = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=False,
        )
        generator.manual_seed(seed)
        layer_autograd = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=False,
        )
        assert_close(layer_manual.weight, layer_autograd.weight)

        output_manual = layer_manual(X)
        output_autograd = layer_autograd(X)

        assert_close(output_manual, output_autograd)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self,
        n_matrices,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that backward pass works and updates gradients
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        X = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
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
        # check weight gradient
        if parametrized:
            # due to parametrization, gradient not directly on layer.weight
            original_weight = layer.parametrizations.weight.original
            assert original_weight.grad is not None
            assert original_weight.grad.shape == layer.weight.shape
            assert not torch.isnan(original_weight.grad).any()
            assert not torch.isinf(original_weight.grad).any()
        else:
            assert layer.weight.grad is not None
            assert layer.weight.grad.shape == layer.weight.shape
            assert not torch.isnan(layer.weight.grad).any()
            assert not torch.isinf(layer.weight.grad).any()

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_parameter_update(
        self,
        n_matrices,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that parameters can be updated via optimization
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        X = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )

        # Store initial weights
        initial_weights = layer.weight.clone().detach()

        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check that weights changed
        assert not torch.allclose(layer.weight, initial_weights)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    def test_both_modes_give_same_gradient(
        self,
        n_matrices,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        device,
        dtype,
        generator,
        seed,
    ):
        """
        Test that parameters are updated the same way in both modes
        """
        # For parametrization.orthogonal, we need to set use_trivialization option to False when n_in > n_out
        # because it appears that the corresponding module then use a torch.randn that we cannot control
        # which makes it impossible to get the same results.
        if (n_out < n_in) and (parametrization_mode == "static"):
            parametrization_options = {"use_trivialization": False}
        else:
            parametrization_options = None

        # we need to reset the generator seed to ensure we generate same weights in both case
        generator.manual_seed(seed)
        layer_manual = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            parametrization_options=parametrization_options,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=False,
        )
        #
        generator.manual_seed(seed)
        layer_auto = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            parametrization_options=parametrization_options,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=True,
        )
        X1 = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        assert_close(layer_manual.weight, layer_auto.weight)

        # Forward + backward + update
        output_manual = layer_manual(X1)
        loss_manual = output_manual.sum()
        loss_manual.backward()

        output_auto = layer_auto(X2)
        loss_auto = output_auto.sum()
        loss_auto.backward()

        # check input gradients
        assert_close(X1.grad, X2.grad)
        # check weight gradients
        if parametrized:
            # due to parametrization, gradient not directly on layer.weight
            original_weight1 = layer_manual.parametrizations.weight.original
            original_weight2 = layer_auto.parametrizations.weight.original
            assert_close(original_weight1.grad, original_weight2.grad)
        else:
            assert_close(layer_manual.weight.grad, layer_auto.weight.grad)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_initialization(
        self, n_in, n_out, use_autograd, device, dtype, generator
    ):
        """
        Test that dynamic parametrization is initialized as expected, i.e.,
        that initial tangent vector is zero
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=True,
            parametrization_mode="dynamic",
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        weight_tangent = layer.parametrizations.weight.original
        assert_close(
            weight_tangent, torch.zeros((n_in, n_out), device=device, dtype=dtype)
        )
        assert_close(layer.weight, layer.stiefel_parametrization.reference_point)

    @pytest.mark.parametrize("n_matrices", [30])
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_dynamic_parametrization_ref_update(
        self, n_matrices, n_in, n_out, use_autograd, device, dtype, generator
    ):
        """
        Test that reference point is updated as expected
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=True,
            parametrization_mode="dynamic",
            n_steps_ref_update=2,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )
        layer.train()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        layer.register_optimizer_hook(optimizer)

        initial_ref = layer.stiefel_parametrization.reference_point.clone().detach()

        # first time going through the layer
        X1 = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X1)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert_close(layer.stiefel_parametrization.reference_point, initial_ref)

        # second time going through the layer
        X2 = random_SPD(
            n_in, n_matrices, device=device, dtype=dtype, generator=generator
        )
        # Forward + backward + update
        optimizer.zero_grad()
        output = layer(X2)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check that reference_point changed
        assert not torch.allclose(
            layer.stiefel_parametrization.reference_point, initial_ref
        )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(
        self,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test string representations
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_in={n_in}" in repr_str
        assert f"n_out={n_out}" in repr_str
        assert f"parametrized={parametrized}" in repr_str
        assert f"parametrization_mode={parametrization_mode}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert f"generator={generator}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("parametrized", [True, False])
    @pytest.mark.parametrize(
        "parametrization_mode",
        ["static", "dynamic"],
    )
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(
        self,
        n_in,
        n_out,
        parametrized,
        parametrization_mode,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test that module respects train/eval mode
        """
        layer = nn_spd_base.BiMap(
            n_in=n_in,
            n_out=n_out,
            parametrized=parametrized,
            parametrization_mode=parametrization_mode,
            device=device,
            dtype=dtype,
            generator=generator,
            use_autograd=use_autograd,
        )

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestReEig:
    """
    Test suite for ReEig module
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
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

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_output_symmetric(
        self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator
    ):
        """
        Test that output matrices are SPD
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        output = layer(X)
        assert is_spd(output)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    def test_both_modes_give_same_result(
        self, n_matrices, n_features, cond, eps, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer_autograd = nn_spd_base.ReEig(eps=eps, use_autograd=True)
        layer_manual = nn_spd_base.ReEig(eps=eps, use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
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

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert is_symmetric(X.grad)
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    def test_both_modes_give_same_gradient(
        self, n_matrices, n_features, cond, eps, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
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

        layer_autograd = nn_spd_base.ReEig(eps=eps, use_autograd=True)
        layer_manual = nn_spd_base.ReEig(eps=eps, use_autograd=False)

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, eps, use_autograd):
        """
        Test string representations
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"eps={eps}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, eps, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestLogEig:
    """
    Test suite for LogEig module
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_matrices, n_features, cond, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
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

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_output_symmetric(
        self, n_matrices, n_features, cond, use_autograd, device, dtype, generator
    ):
        """
        Test that output matrices are symmetric
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        output = layer(X)
        assert is_symmetric(output)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_both_modes_give_same_result(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer_autograd = nn_spd_base.LogEig(use_autograd=True)
        layer_manual = nn_spd_base.LogEig(use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_matrices, n_features, cond, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
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

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert is_symmetric(X.grad)
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_both_modes_give_same_gradient(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
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

        layer_autograd = nn_spd_base.LogEig(use_autograd=True)
        layer_manual = nn_spd_base.LogEig(use_autograd=False)

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, use_autograd):
        """
        Test string representations
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)

        # Should work in both modes
        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestVec:
    """
    Test suite for Vec module
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_matrices, n_features, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        output = layer(X)

        assert output.dim() == X.dim() - 1
        if n_matrices > 1:
            assert output.shape[0] == n_matrices
        assert output.shape[-1] == n_features**2
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    def test_both_modes_give_same_result(
        self, n_matrices, n_features, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        layer_autograd = nn_spd_base.Vec(use_autograd=True)
        layer_manual = nn_spd_base.Vec(use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_matrices, n_features, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X.requires_grad = True

        output = layer(X)
        loss = torch.norm(output)
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert is_symmetric(X.grad)
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    def test_both_modes_give_same_gradient(
        self, n_matrices, n_features, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
        # random batch of symmetric matrices
        X1 = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        layer_autograd = nn_spd_base.Vec(use_autograd=True)
        layer_manual = nn_spd_base.Vec(use_autograd=False)

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, use_autograd):
        """
        Test string representations
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestVech:
    """
    Test suite for Vech module
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    def test_forward_shape(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.Vech()
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        output = layer(X)

        assert output.dim() == X.dim() - 1
        if n_matrices > 1:
            assert output.shape[0] == n_matrices
        assert output.shape[-1] == n_features * (n_features + 1) // 2
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    def test_backward_pass(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.Vech()
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X.requires_grad = True

        output = layer(X)
        loss = torch.norm(output)
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert is_symmetric(X.grad)
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    def test_module_mode(self):
        """
        Test that module respects train/eval mode
        """
        layer = nn_spd_base.Vech()

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False
