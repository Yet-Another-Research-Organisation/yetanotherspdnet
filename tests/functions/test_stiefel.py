import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.stiefel as stiefel
from yetanotherspdnet.random.stiefel import _init_weights_stiefel

from utils import is_orthogonal



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



class TestStiefelProjection:
    """
    Test suite for the projection on Stiefel and
    orthogonal projection on tangent space
    """
    @pytest.mark.parametrize("n_in, n_out", [(100, 50), (70,30)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on Stiefel is working
        """
        Z = torch.randn((n_out, n_in), device=device, dtype=dtype, generator=generator)

        W = stiefel.projection_stiefel(Z)
        WW = stiefel.projection_stiefel(W)

        V = stiefel.ProjectionStiefel.apply(Z)
        VV = stiefel.ProjectionStiefel.apply(V)

        assert W.shape == Z.shape
        assert W.device == Z.device
        assert W.dtype == Z.dtype
        assert is_orthogonal(W)
        assert is_orthogonal(WW)
        assert_close(W, WW)

        assert V.shape == Z.shape
        assert V.device == Z.device
        assert V.dtype == Z.dtype
        assert is_orthogonal(V)
        assert is_orthogonal(VV)
        assert_close(V, VV)

        assert_close(V, W)

    @pytest.mark.parametrize("n_in", [100, 70])
    def test_identity_matrix(self, n_in, device, dtype):
        """
        Test that the projection of the identity matrix is the identity matrix
        """
        I = torch.eye(n_in, device=device, dtype=dtype)
        proj_I = stiefel.projection_stiefel(I)
        assert_close(I, proj_I)

    @pytest.mark.parametrize("n_in, n_out", [(100, 50), (70, 30)])
    def test_projection_tangent(self, n_in, n_out, device, dtype, generator):
        """
        Test that the orthogonal projection on the tangent space is working
        """
        W = stiefel.projection_stiefel(torch.randn((n_out, n_in), device=device, dtype=dtype, generator=generator))
        Z = torch.randn((n_out, n_in), device=device, dtype=dtype, generator=generator)

        tangent_vec = stiefel.projection_tangent_stiefel(Z, W)
        tantan_vec = stiefel.projection_tangent_stiefel(tangent_vec, W)

        assert tangent_vec.shape == Z.shape
        assert tangent_vec.device == Z.device
        assert tangent_vec.dtype == Z.dtype
        assert_close(tangent_vec, tantan_vec)
        assert_close(W @ tangent_vec.transpose(-2, -1) + tangent_vec @ W.transpose(-2, -1), torch.zeros((n_out, n_out), device=device, dtype=dtype))

    @pytest.mark.parametrize("n_in, n_out", [(100, 50), (70, 30)])
    def test_tangent_zero_matrix(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on the tangent space of the zero matrix is the zero matrix
        """
        W = stiefel.projection_stiefel(torch.randn((n_out, n_in), device=device, dtype=dtype, generator=generator))
        Z = torch.zeros((n_out, n_in), device=device, dtype=dtype)

        tangent_vec = stiefel.projection_tangent_stiefel(Z, W)

        assert_close(tangent_vec, torch.zeros((n_out, n_in), device=device, dtype=dtype))

    @pytest.mark.parametrize("n_in, n_out", [(100, 50), (70, 30)])
    def test_backward(self, n_in, n_out, device, dtype, generator):
        """
        Test of backward of StiefelProjection
        """
        Z = torch.randn((n_out, n_in), device=device, dtype=dtype, generator=generator)
        Z_manual = Z.clone().detach()
        Z_manual.requires_grad = True
        Z_auto = Z.clone().detach()
        Z_auto.requires_grad = True

        W_manual = stiefel.ProjectionStiefel.apply(Z_manual)
        W_auto = stiefel.projection_stiefel(Z_auto)

        loss_manual = torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(W_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)



