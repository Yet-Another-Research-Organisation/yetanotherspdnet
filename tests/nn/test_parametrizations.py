import pytest

import torch
from torch.testing import assert_close


import yetanotherspdnet.nn.parametrizations as nn_parametrizations
from yetanotherspdnet.functions.spd_linalg import symmetrize

from utils import is_symmetric, is_spd, is_orthogonal


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


class TestSPDSoftPlusParametrization:
    """
    Test suite for SPDSoftPlusParametrization
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
        layer = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=use_autograd
        )
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

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

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

        layer_autograd = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=False
        )

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
        layer = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=use_autograd
        )
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

        layer_autograd = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=False
        )

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
        layer = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=use_autograd
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.SPDSoftPlusParametrization(
            use_autograd=use_autograd
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestSPDLogEuclideanParametrization:
    """
    Test suite for SPDLogEuclideanParametrization module
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
        layer = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=use_autograd
        )
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

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device
        assert is_spd(output)

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

        layer_autograd = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=False
        )

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
        layer = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=use_autograd
        )
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

        layer_autograd = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=False
        )

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
        layer = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=use_autograd
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.SPDLogEuclideanParametrization(
            use_autograd=use_autograd
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestStiefelProjectionPolarParametrization:
    """
    Test suite for StiefelProjectionPolarParametrization module
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(self, n_in, n_out, use_autograd, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=use_autograd
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
    def test_both_modes_give_same_result(self, n_in, n_out, device, dtype, generator):
        """
        Test that autograd and manual gradient give same forward results
        """
        weight = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )

        layer_autograd = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=False
        )

        output_autograd = layer_autograd(weight)
        output_manual = layer_manual(weight)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(self, n_in, n_out, use_autograd, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=use_autograd
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
    def test_both_modes_give_same_gradient(self, n_in, n_out, device, dtype, generator):
        """
        Test that autograd and manual gradient give same gradients
        """
        weight1 = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )
        weight2 = weight1.clone()
        weight1.requires_grad = True
        weight2.requires_grad = True

        layer_autograd = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=False
        )

        output1 = layer_autograd(weight1)
        output2 = layer_manual(weight2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(weight1.grad, weight2.grad)

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, use_autograd):
        """
        Test string representations
        """
        layer = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=use_autograd
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.StiefelProjectionPolarParametrization(
            use_autograd=use_autograd
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False


class TestStiefelProjectionQRParametrization:
    """
    Test suite for StiefelProjectionQRParametrization module
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(self, n_in, n_out, use_autograd, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=use_autograd
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
    def test_both_modes_give_same_result(self, n_in, n_out, device, dtype, generator):
        """
        Test that autograd and manual gradient give same forward results
        """
        weight = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )

        layer_autograd = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=False
        )

        output_autograd = layer_autograd(weight)
        output_manual = layer_manual(weight)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(self, n_in, n_out, use_autograd, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=use_autograd
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
    def test_both_modes_give_same_gradient(self, n_in, n_out, device, dtype, generator):
        """
        Test that autograd and manual gradient give same gradients
        """
        weight1 = torch.randn(
            (n_in, n_out), device=device, dtype=dtype, generator=generator
        )
        weight2 = weight1.clone()
        weight1.requires_grad = True
        weight2.requires_grad = True

        layer_autograd = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=True
        )
        layer_manual = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=False
        )

        output1 = layer_autograd(weight1)
        output2 = layer_manual(weight2)

        loss1 = torch.norm(output1)
        loss2 = torch.norm(output2)

        loss1.backward()
        loss2.backward()

        assert_close(weight1.grad, weight2.grad)

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_repr_and_str(self, use_autograd):
        """
        Test string representations
        """
        layer = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=use_autograd
        )

        repr_str = repr(layer)
        str_str = str(layer)

        assert f"use_autograd={use_autograd}" in repr_str
        assert repr_str == str_str

    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_module_mode(self, use_autograd):
        """
        Test that module respects train/eval mode
        """
        layer = nn_parametrizations.StiefelProjectionQRParametrization(
            use_autograd=use_autograd
        )

        layer.train()
        assert layer.training is True

        layer.eval()
        assert layer.training is False
