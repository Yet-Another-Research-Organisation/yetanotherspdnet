import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close


import yetanotherspdnet.nn.base as nn_spd_base
from yetanotherspdnet.functions.spd_linalg import symmetrize
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.random.stiefel import _init_weights_stiefel

from utils import is_symmetric, is_spd


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64

@pytest.fixture(scope="function")
def generator(device):
    generator = torch.Generator(device=device)
    generator.manual_seed(777)
    return generator



class TestReEig:
    """
    Test suite for ReEig module
    """
    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond = cond, device=device, dtype=dtype)

        output = layer(X)

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_output_symmetric(self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator):
        """
        Test that output matrices are SPD
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
        output = layer(X)
        assert is_spd(output)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    def test_both_modes_give_same_result(self, n_matrices, n_features, cond, eps, device, dtype, generator):
        """
        Test that autograd and manual gradient give same forward results
        """
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)

        layer_autograd = nn_spd_base.ReEig(eps=eps, use_autograd=True)
        layer_manual = nn_spd_base.ReEig(eps=eps, use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(self, n_matrices, n_features, cond, eps, use_autograd, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.ReEig(eps=eps, use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("eps", [1e-4])
    def test_both_modes_give_same_gradient(self, n_matrices, n_features, cond, eps, device, dtype, generator):
        """
        Test that autograd and manual gradient give same gradients
        """
        X1 = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(self, n_matrices, n_features, cond, use_autograd, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)

        output = layer(X)

        assert output.shape == X.shape
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_output_symmetric(self, n_matrices, n_features, cond, use_autograd, device, dtype, generator):
        """
        Test that output matrices are symmetric
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
        output = layer(X)
        assert is_symmetric(output)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    def test_both_modes_give_same_result(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that autograd and manual gradient give same forward results
        """
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)

        layer_autograd = nn_spd_base.LogEig(use_autograd=True)
        layer_manual = nn_spd_base.LogEig(use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(self, n_matrices, n_features, cond, use_autograd, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.LogEig(use_autograd=use_autograd)
        X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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
    @pytest.mark.parametrize("n_features, cond",[(100, 1000)])
    def test_both_modes_give_same_gradient(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test that autograd and manual gradient give same gradients
        """
        X1 = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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
    @pytest.mark.parametrize("n_features",[100])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_forward_shape(self, n_matrices, n_features, use_autograd, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
        )

        output = layer(X)

        assert output.dim() == X.dim() - 1
        if n_matrices > 1 :
            assert output.shape[0] == n_matrices
        assert output.shape[-1] == n_features**2
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features",[100])
    def test_both_modes_give_same_result(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that autograd and manual gradient give same forward results
        """
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
        )

        layer_autograd = nn_spd_base.Vec(use_autograd=True)
        layer_manual = nn_spd_base.Vec(use_autograd=False)

        output_autograd = layer_autograd(X)
        output_manual = layer_manual(X)

        assert_close(output_autograd, output_manual)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features",[100])
    @pytest.mark.parametrize("use_autograd", [True, False])
    def test_backward_pass(self, n_matrices, n_features, use_autograd, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.Vec(use_autograd=use_autograd)
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
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
    @pytest.mark.parametrize("n_features",[100])
    def test_both_modes_give_same_gradient(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that autograd and manual gradient give same gradients
        """
        # random batch of symmetric matrices
        X1 = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
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
    @pytest.mark.parametrize("n_features",[100])
    def test_forward_shape(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that forward pass returns correct shape
        """
        layer = nn_spd_base.Vech()
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
        )

        output = layer(X)

        assert output.dim() == X.dim() - 1
        if n_matrices > 1 :
            assert output.shape[0] == n_matrices
        assert output.shape[-1] == n_features*(n_features + 1) // 2
        assert output.dtype == X.dtype
        assert output.device == X.device

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features",[100])
    def test_backward_pass(self, n_matrices, n_features, device, dtype, generator):
        """
        Test that backward pass works (gradient computation)
        """
        layer = nn_spd_base.Vech()
        # random batch of symmetric matrices
        X = symmetrize(
            torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
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

