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



class TestProjectionStiefelPolar:
    """
    Test suite for polar based projection on Stiefel and
    orthogonal projection on tangent space
    """
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100,50)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on Stiefel is working
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        W = stiefel.projection_stiefel_polar(Z)
        WW = stiefel.projection_stiefel_polar(W)

        V = stiefel.ProjectionStiefelPolar.apply(Z)
        VV = stiefel.ProjectionStiefelPolar.apply(V)

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
        proj_I = stiefel.projection_stiefel_polar(I)
        assert_close(I, proj_I)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_projection_tangent(self, n_in, n_out, device, dtype, generator):
        """
        Test that the orthogonal projection on the tangent space is working
        """
        W = stiefel.projection_stiefel_polar(torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator))
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        tangent_vec = stiefel.projection_tangent_stiefel_orthogonal(Z, W)
        tantan_vec = stiefel.projection_tangent_stiefel_orthogonal(tangent_vec, W)

        assert tangent_vec.shape == Z.shape
        assert tangent_vec.device == Z.device
        assert tangent_vec.dtype == Z.dtype
        assert_close(tangent_vec, tantan_vec)
        assert_close(W.transpose(-2, -1) @ tangent_vec + tangent_vec.transpose(-2, -1) @ W, torch.zeros((n_out, n_out), device=device, dtype=dtype))

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_tangent_zero_matrix(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on the tangent space of the zero matrix is the zero matrix
        """
        W = stiefel.projection_stiefel_polar(torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator))
        Z = torch.zeros((n_in, n_out), device=device, dtype=dtype)

        tangent_vec = stiefel.projection_tangent_stiefel_orthogonal(Z, W)

        assert_close(tangent_vec, torch.zeros((n_in, n_out), device=device, dtype=dtype))

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_backward(self, n_in, n_out, device, dtype, generator):
        """
        Test of backward of StiefelProjection
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Z_manual = Z.clone().detach()
        Z_manual.requires_grad = True
        Z_auto = Z.clone().detach()
        Z_auto.requires_grad = True

        W_manual = stiefel.ProjectionStiefelPolar.apply(Z_manual)
        W_auto = stiefel.projection_stiefel_polar(Z_auto)

        loss_manual = torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(W_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)


class TestProjectionStiefelQR:
    """
    Test suite for the QR based projection on Stiefel
    and its differential and differential adjoint
    """
    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100,50)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on Stiefel is working
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        W = stiefel.projection_stiefel_qr(Z)
        WW = stiefel.projection_stiefel_qr(W)

        V = stiefel.ProjectionStiefelQR.apply(Z)
        VV = stiefel.ProjectionStiefelQR.apply(V)

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
        proj_I = stiefel.projection_stiefel_qr(I)
        assert_close(I, proj_I)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_differential_and_adjoint(self, n_in, n_out, device, dtype, generator):
        """
        Test that the differential of QR based projection and its adjoint are working
        """
        W = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(W)

        Y = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        tangent_vec = stiefel.differential_projection_stiefel_qr(Y, Q, R)
        adjoint = stiefel.differential_adjoint_projection_stiefel_qr(Z, Q, R)

        assert tangent_vec.shape == Y.shape
        assert tangent_vec.device == Y.device
        assert tangent_vec.dtype == Y.dtype
        assert_close(Q.transpose(-2, -1) @ tangent_vec + tangent_vec.transpose(-2, -1) @ Q, torch.zeros((n_out, n_out), device=device, dtype=dtype))

        assert adjoint.shape == Z.shape
        assert adjoint.device == Z.device
        assert adjoint.dtype == Z.dtype
        # test that they are actually adjoints
        assert_close(torch.trace(Y.transpose(-2, -1) @ adjoint), torch.trace(Z.transpose(-2,-1) @ tangent_vec))

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_zero_matrix_differential_adjoint(self, n_in, n_out, device, dtype, generator):
        """
        Test that the differential and adjoint of the zero matrix is the zero matrix
        """
        W = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(W)

        Z = torch.zeros((n_in, n_out), device=device, dtype=dtype)

        tangent_vec = stiefel.differential_projection_stiefel_qr(Z, Q, R)
        adjoint = stiefel.differential_adjoint_projection_stiefel_qr(Z, Q, R)

        assert_close(tangent_vec, torch.zeros((n_in, n_out), device=device, dtype=dtype))
        assert_close(adjoint, torch.zeros((n_in, n_out), device=device, dtype=dtype))

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_backward(self, n_in, n_out, device, dtype, generator):
        """
        Test of backward of StiefelProjection
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Z_manual = Z.clone().detach()
        Z_manual.requires_grad = True
        Z_auto = Z.clone().detach()
        Z_auto.requires_grad = True

        W_manual = stiefel.ProjectionStiefelQR.apply(Z_manual)
        W_auto = stiefel.projection_stiefel_qr(Z_auto)

        loss_manual = torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(W_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)

