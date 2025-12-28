import pytest
import torch
from torch.testing import assert_close

import yetanotherspdnet.functions.stiefel as stiefel
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


class TestStiefelProjectionTangentOrthogonal:
    """
    Test suite for orthogonal projection on tangent spaces
    of the Stiefel manifold
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the orthogonal projection on tangent spaces of
        the Stiefel manifold
        """
        W = stiefel.stiefel_projection_polar(
            torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        )
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        tangent_vec_auto = stiefel.stiefel_projection_tangent_orthogonal(Z, W)
        tantan_vec_auto = stiefel.stiefel_projection_tangent_orthogonal(
            tangent_vec_auto, W
        )
        tangent_vec_manual = stiefel.StiefelProjectionTangentOrthogonal.apply(Z, W)
        tantan_vec_manual = stiefel.StiefelProjectionTangentOrthogonal.apply(
            tangent_vec_manual, W
        )

        assert tangent_vec_auto.shape == Z.shape
        assert tangent_vec_auto.device == Z.device
        assert tangent_vec_auto.dtype == Z.dtype
        assert_close(tangent_vec_auto, tantan_vec_auto)
        assert_close(
            W.transpose(-2, -1) @ tangent_vec_auto
            + tangent_vec_auto.transpose(-2, -1) @ W,
            torch.zeros((n_out, n_out), device=device, dtype=dtype),
        )

        assert tangent_vec_manual.shape == Z.shape
        assert tangent_vec_manual.device == Z.device
        assert tangent_vec_manual.dtype == Z.dtype
        assert_close(tangent_vec_manual, tantan_vec_manual)
        assert_close(
            W.transpose(-2, -1) @ tangent_vec_manual
            + tangent_vec_manual.transpose(-2, -1) @ W,
            torch.zeros((n_out, n_out), device=device, dtype=dtype),
        )

        assert_close(tangent_vec_manual, tangent_vec_auto)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_tangent_zero_matrix(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on the tangent space of the zero matrix is the zero matrix
        """
        W = stiefel.stiefel_projection_polar(
            torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        )
        Z = torch.zeros((n_in, n_out), device=device, dtype=dtype)

        tangent_vec_auto = stiefel.stiefel_projection_tangent_orthogonal(Z, W)
        tangent_vec_manual = stiefel.StiefelProjectionTangentOrthogonal.apply(Z, W)

        assert_close(
            tangent_vec_auto, torch.zeros((n_in, n_out), device=device, dtype=dtype)
        )
        assert_close(
            tangent_vec_manual, torch.zeros((n_in, n_out), device=device, dtype=dtype)
        )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_backward(self, n_in, n_out, device, dtype, generator):
        """
        Test of backward of StiefelProjectionTangentOrthogonal
        """
        W = stiefel.stiefel_projection_polar(
            torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        )
        W.requires_grad = False

        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Z_manual = Z.clone().detach()
        Z_manual.requires_grad = True
        Z_auto = Z.clone().detach()
        Z_auto.requires_grad = True

        xi_manual = stiefel.StiefelProjectionTangentOrthogonal.apply(Z_manual, W)
        xi_auto = stiefel.stiefel_projection_tangent_orthogonal(Z_auto, W)

        loss_manual = torch.norm(xi_manual)
        loss_manual.backward()
        loss_auto = torch.norm(xi_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)


class TestStiefelProjectionPolar:
    """
    Test suite for polar based projection on Stiefel and
    orthogonal projection on tangent space
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on Stiefel is working
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        W = stiefel.stiefel_projection_polar(Z)
        WW = stiefel.stiefel_projection_polar(W)

        V = stiefel.StiefelProjectionPolar.apply(Z)
        VV = stiefel.StiefelProjectionPolar.apply(V)

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
        identity_matrix = torch.eye(n_in, device=device, dtype=dtype)
        proj_I = stiefel.stiefel_projection_polar(identity_matrix)
        assert_close(identity_matrix, proj_I)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_projection_tangent(self, n_in, n_out, device, dtype, generator):
        """
        Test that the orthogonal projection on the tangent space is working
        """
        W = stiefel.stiefel_projection_polar(
            torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        )
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        tangent_vec = stiefel.stiefel_projection_tangent_orthogonal(Z, W)
        tantan_vec = stiefel.stiefel_projection_tangent_orthogonal(tangent_vec, W)

        assert tangent_vec.shape == Z.shape
        assert tangent_vec.device == Z.device
        assert tangent_vec.dtype == Z.dtype
        assert_close(tangent_vec, tantan_vec)
        assert_close(
            W.transpose(-2, -1) @ tangent_vec + tangent_vec.transpose(-2, -1) @ W,
            torch.zeros((n_out, n_out), device=device, dtype=dtype),
        )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_tangent_zero_matrix(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on the tangent space of the zero matrix is the zero matrix
        """
        W = stiefel.stiefel_projection_polar(
            torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        )
        Z = torch.zeros((n_in, n_out), device=device, dtype=dtype)

        tangent_vec = stiefel.stiefel_projection_tangent_orthogonal(Z, W)

        assert_close(
            tangent_vec, torch.zeros((n_in, n_out), device=device, dtype=dtype)
        )

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

        W_manual = stiefel.StiefelProjectionPolar.apply(Z_manual)
        W_auto = stiefel.stiefel_projection_polar(Z_auto)

        loss_manual = torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(W_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)


class TestStiefelProjectionQR:
    """
    Test suite for the QR based projection on Stiefel
    and its differential and differential adjoint
    """

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_forward(self, n_in, n_out, device, dtype, generator):
        """
        Test that the projection on Stiefel is working
        """
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        W = stiefel.stiefel_projection_qr(Z)
        WW = stiefel.stiefel_projection_qr(W)

        V = stiefel.StiefelProjectionQR.apply(Z)
        VV = stiefel.StiefelProjectionQR.apply(V)

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
        identity_matrix = torch.eye(n_in, device=device, dtype=dtype)
        proj_I = stiefel.stiefel_projection_qr(identity_matrix)
        assert_close(identity_matrix, proj_I)

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_differential_and_adjoint(self, n_in, n_out, device, dtype, generator):
        """
        Test that the differential of QR based projection and its adjoint are working
        """
        W = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(W)

        Y = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Z = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)

        tangent_vec = stiefel.stiefel_differential_projection_qr(Y, Q, R)
        adjoint = stiefel.stiefel_adjoint_differential_projection_qr(Z, Q, R)

        assert tangent_vec.shape == Y.shape
        assert tangent_vec.device == Y.device
        assert tangent_vec.dtype == Y.dtype
        assert_close(
            Q.transpose(-2, -1) @ tangent_vec + tangent_vec.transpose(-2, -1) @ Q,
            torch.zeros((n_out, n_out), device=device, dtype=dtype),
        )

        assert adjoint.shape == Z.shape
        assert adjoint.device == Z.device
        assert adjoint.dtype == Z.dtype
        # test that they are actually adjoints
        assert_close(
            torch.trace(Y.transpose(-2, -1) @ adjoint),
            torch.trace(Z.transpose(-2, -1) @ tangent_vec),
        )

    @pytest.mark.parametrize("n_in, n_out", [(100, 100), (100, 50)])
    def test_zero_matrix_differential_adjoint(
        self, n_in, n_out, device, dtype, generator
    ):
        """
        Test that the differential and adjoint of the zero matrix is the zero matrix
        """
        W = torch.randn((n_in, n_out), device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(W)

        Z = torch.zeros((n_in, n_out), device=device, dtype=dtype)

        tangent_vec = stiefel.stiefel_differential_projection_qr(Z, Q, R)
        adjoint = stiefel.stiefel_adjoint_differential_projection_qr(Z, Q, R)

        assert_close(
            tangent_vec, torch.zeros((n_in, n_out), device=device, dtype=dtype)
        )
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

        W_manual = stiefel.StiefelProjectionQR.apply(Z_manual)
        W_auto = stiefel.stiefel_projection_qr(Z_auto)

        loss_manual = torch.norm(W_manual)
        loss_manual.backward()
        loss_auto = torch.norm(W_auto)
        loss_auto.backward()

        assert Z_manual.grad is not None
        assert Z_auto.grad is not None
        assert torch.isfinite(Z_manual.grad).all()
        assert torch.isfinite(Z_auto.grad).all()
        assert_close(Z_manual.grad, Z_auto.grad)
