import pytest

import torch
from torch.testing import assert_close


import yetanotherspdnet.nn.parametrizations as nn_parametrizations
from yetanotherspdnet.functions.spd_linalg import inv_sqrtm_SPD, sqrtm_SPD, symmetrize

from utils import is_symmetric, is_spd, is_orthogonal
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.random.stiefel import random_stiefel


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


class TestSPDParametrization:
    """
    Test suite for SPDParametrization
    """

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=use_autograd
        )
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        output = layer(X)

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    def test_both_modes_give_same_result(
        self, n_features, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        layer_autograd = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=False
        )

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=use_autograd
        )
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
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

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    def test_both_modes_give_same_gradient(
        self, n_features, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
        # random batch of symmetric matrices
        X1 = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        layer_autograd = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=False
        )

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, mapping, use_autograd):
        """
        Test string representations
        """
        layer = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=use_autograd
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"mapping={mapping}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, mapping, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.SPDParametrization(
            mapping=mapping, use_autograd=use_autograd
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestSPDAdaptiveParametrization:
    """
    Test suite for SPDAdaptiveParametrization module
    """

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        output = layer(X)

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_initial_reference(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test initial_reference initialization
        """
        initial_reference = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        assert_close(layer.reference_point, initial_reference)
        assert_close(layer.reference_point_sqrtm, sqrtm_SPD(initial_reference)[0])
        assert_close(
            layer.reference_point_inv_sqrtm, inv_sqrtm_SPD(initial_reference)[0]
        )

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("invalid_dim", [70, 150])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_invalid_initial_reference(
        self, n_features, invalid_dim, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test invalid initial_reference
        """
        initial_reference = random_SPD(
            invalid_dim,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        with pytest.raises(AssertionError, match="Got incoherent initial_reference,"):
            nn_parametrizations.SPDAdaptiveParametrization(
                n_features,
                initial_reference=initial_reference,
                mapping=mapping,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
            )

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["dumb", ""])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_invalid_mapping(self, n_features, mapping, use_autograd, device, dtype):
        """
        Test invalid mapping
        """
        with pytest.raises(AssertionError, match="mapping must be in"):
            nn_parametrizations.SPDAdaptiveParametrization(
                n_features,
                initial_reference=None,
                mapping=mapping,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
            )

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    def test_both_modes_give_same_result(
        self, n_features, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        initial_reference = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # random symmetric matrix
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        layer_autograd = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )
        layer_manual = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        initial_reference = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
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

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    def test_both_modes_give_same_gradient(
        self, n_features, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
        initial_reference = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # random batch of symmetric matrices
        X1 = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        layer_autograd = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )
        layer_manual = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_reference_update(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that reference update works
        """
        initial_reference = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        # random symmetric matrix
        X = symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )

        layer.train()
        output = layer(X)
        layer.update_reference_point()

        assert not torch.allclose(layer.reference_point, initial_reference)
        assert_close(layer.reference_point, output)
        assert_close(layer.reference_point_sqrtm, sqrtm_SPD(output)[0])
        assert_close(layer.reference_point_inv_sqrtm, inv_sqrtm_SPD(output)[0])

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_right_inverse(
        self, n_features, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that right inverse works, i.e., actually yields a tangent vector
        """
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )
        # random SPD matrix
        X = random_SPD(
            n_features,
            n_matrices=1,
            cond=100,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        right_inv = layer.right_inverse(X)

        assert right_inv.shape == X.shape
        assert right_inv.device == X.device
        assert right_inv.dtype == X.dtype
        assert is_symmetric(right_inv)

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, n_features, mapping, use_autograd, device, dtype):
        """
        Test string representations
        """
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_features={n_features}" in repr_str
        assert f"initial_reference={None}" in repr_str
        assert f"mapping={mapping}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_features", [100])
    @pytest.mark.parametrize("mapping", ["softplus", "exp"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, n_features, mapping, use_autograd, device, dtype):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.SPDAdaptiveParametrization(
            n_features,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestStiefelAdaptiveParametrization:
    """
    Test suite for StiefelAdaptiveParametrization module
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        weight = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )

        output = layer(weight)

        assert output.shape == weight.shape
        assert output.device == weight.device
        assert output.dtype == weight.dtype
        assert is_orthogonal(output)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_initial_reference(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test initial_reference initialization
        """
        initial_reference = random_stiefel(
            n_in,
            n_out,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        assert_close(layer.reference_point, initial_reference)

    @pytest.mark.parametrize("n_in, n_out", [(100, 200), (50, 100)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_invalid_dims(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test invalid n_in and n_out combinations
        """
        with pytest.raises(AssertionError, match="Must have n_in >= n_out"):
            nn_parametrizations.StiefelAdaptiveParametrization(
                n_in,
                n_out,
                initial_reference=None,
                mapping=mapping,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
                generator=generator,
            )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("invalid_n_in, invalid_n_out", [(70, 30)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_invalid_initial_reference(
        self,
        n_in,
        n_out,
        invalid_n_in,
        invalid_n_out,
        mapping,
        use_autograd,
        device,
        dtype,
        generator,
    ):
        """
        Test invalid initial_reference
        """
        initial_reference = random_stiefel(
            invalid_n_in,
            invalid_n_out,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        with pytest.raises(AssertionError, match="Got incoherent initial_reference,"):
            nn_parametrizations.StiefelAdaptiveParametrization(
                n_in,
                n_out,
                initial_reference=initial_reference,
                mapping=mapping,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
                generator=generator,
            )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["dumb", ""])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_invalid_mapping(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test invalid mapping
        """
        with pytest.raises(AssertionError, match="mapping must be in"):
            nn_parametrizations.StiefelAdaptiveParametrization(
                n_in,
                n_out,
                initial_reference=None,
                mapping=mapping,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
                generator=generator,
            )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    def test_both_modes_give_same_result(
        self, n_in, n_out, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same forward results
        """
        initial_reference = random_stiefel(
            n_in,
            n_out,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer_autograd = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer_manual = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=False,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        weight = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )

        output_autograd = layer_autograd(weight)
        output_manual = layer_manual(weight)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that backward pass works (gradient computation)
        """
        if (mapping == "polar") and (use_autograd is True):
            pytest.skip(
                "Skipping polar with autograd because it is highly unstable (probably due to use of svd)."
            )

        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        weight = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )
        weight.requires_grad = True

        output = layer(weight)
        loss = torch.norm(output)
        loss.backward()

        assert weight.grad is not None
        assert weight.grad.shape == weight.shape
        assert not torch.isnan(weight.grad).any()
        assert not torch.isinf(weight.grad).any()

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    # no testing of polar because autograd not working
    @pytest.mark.parametrize("mapping", ["QR"])
    def test_both_modes_give_same_gradient(
        self, n_in, n_out, mapping, device, dtype, generator
    ):
        """
        Test that autograd and manual gradient give same gradients
        """
        initial_reference = random_stiefel(
            n_in,
            n_out,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # random batch of symmetric matrices
        X1 = torch.randn(
            (n_in, n_out),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X2 = X1.clone()
        X1.requires_grad = True
        X2.requires_grad = True

        layer_autograd = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=True,
            device=device,
            dtype=dtype,
        )
        layer_manual = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=False,
            device=device,
            dtype=dtype,
        )

        output1 = layer_autograd(X1)
        output2 = layer_manual(X2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(X1.grad, X2.grad)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_reference_update(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that reference update works
        """
        initial_reference = random_stiefel(
            n_in,
            n_out,
            n_matrices=1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=initial_reference,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # random matrix
        X = torch.randn(
            (n_in, n_out),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        layer.train()
        output = layer(X)
        layer.update_reference_point()

        assert not torch.allclose(layer.reference_point, initial_reference)
        assert_close(layer.reference_point, output)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_right_inverse(
        self, n_in, n_out, mapping, use_autograd, device, dtype, generator
    ):
        """
        Test that right_inverse works, i.e., that it actually yields a tangent vector
        """
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # random orthogonal matrix
        X = random_stiefel(
            n_in, n_out, n_matrices=1, device=device, dtype=dtype, generator=generator
        )

        right_inv = layer.right_inverse(X)

        assert right_inv.shape == X.shape
        assert right_inv.device == X.device
        assert right_inv.dtype == X.dtype
        assert_close(
            layer.reference_point.transpose(-2, -1) @ right_inv
            + right_inv.transpose(-2, -1) @ layer.reference_point,
            torch.zeros((n_out, n_out), device=device, dtype=dtype),
        )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, n_in, n_out, mapping, use_autograd, device, dtype):
        """
        Test string representations
        """
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"n_in={n_in}" in repr_str
        assert f"n_out={n_out}" in repr_str
        assert f"initial_reference={None}" in repr_str
        assert f"mapping={mapping}" in repr_str
        assert f"use_autograd={use_autograd}" in repr_str
        assert f"device={device}" in repr_str
        assert f"dtype={dtype}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("mapping", ["QR", "polar"])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, n_in, n_out, mapping, use_autograd, device, dtype):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.StiefelAdaptiveParametrization(
            n_in,
            n_out,
            initial_reference=None,
            mapping=mapping,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False
