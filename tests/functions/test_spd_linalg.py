import pytest
import torch
from torch.testing import assert_close

from scipy.linalg import expm, logm, solve_sylvester, sqrtm

import yetanotherspdnet.functions.spd_linalg as spd_linalg
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


# ----------
# Symmetrize
# ----------
@pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
def test_symmetrize(n_matrices, n_features, device, dtype, generator):
    """
    Test of symmetrize function
    """
    X = torch.squeeze(
        torch.randn(
            (n_matrices, n_features, n_features),
            device=device,
            dtype=dtype,
            generator=generator,
        )
    )
    X_sym = spd_linalg.symmetrize(X)
    X_skew = X - X_sym
    assert_close(X_sym, X_sym.transpose(-1, -2))
    assert_close(X_skew, -X_skew.transpose(-1, -2))
    assert_close(X, X_sym + X_skew)
    assert X_sym.device == X.device
    assert X_sym.dtype == X.dtype


# -------------
# Vectorization
# -------------
class TestVec:
    """
    Test suite for vectorization of batch of matrices
    """

    @pytest.mark.parametrize(
        "n_matrices, n_rows, n_columns",
        [(1, 100, 100), (1, 100, 30), (50, 100, 100), (50, 100, 30)],
    )
    def test_vec_batch(self, n_matrices, n_rows, n_columns, device, dtype, generator):
        """
        Test of vec_batch function
        """
        X = torch.squeeze(
            torch.randn(
                (n_matrices, n_rows, n_columns),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        X_vec = spd_linalg.vec_batch(X)
        assert X_vec.dim() == X.dim() - 1
        if n_matrices > 1:
            assert X_vec.shape[0] == n_matrices
        assert X_vec.shape[-1] == n_rows * n_columns
        assert_close(spd_linalg.unvec_batch(X_vec, n_rows), X)
        assert X_vec.device == X.device
        assert X_vec.dtype == X.dtype

    @pytest.mark.parametrize(
        "n_matrices, n_rows, n_columns",
        [(1, 100, 100), (1, 100, 30), (50, 100, 100), (50, 100, 30)],
    )
    def test_unvec_batch(self, n_matrices, n_rows, n_columns, device, dtype, generator):
        """
        Test of unvec_batch function
        """
        X_vec = torch.squeeze(
            torch.randn(
                (n_matrices, n_rows * n_columns),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        X_unvec = spd_linalg.unvec_batch(X_vec, n_rows)
        assert X_unvec.dim() == X_vec.dim() + 1
        if n_matrices > 1:
            assert X_unvec.shape[0] == n_matrices
        assert X_unvec.shape[-2] == n_rows
        assert X_unvec.shape[-1] == n_columns
        assert_close(spd_linalg.vec_batch(X_unvec), X_vec)
        assert X_unvec.device == X_vec.device
        assert X_unvec.dtype == X_vec.dtype

    @pytest.mark.parametrize(
        "n_matrices, n_rows, n_columns",
        [(1, 100, 100), (1, 100, 30), (50, 100, 100), (50, 100, 30)],
    )
    def test_forward(self, n_matrices, n_rows, n_columns, device, dtype, generator):
        """
        Test forward of VecBatch Function class
        """
        X = torch.squeeze(
            torch.randn(
                (n_matrices, n_rows, n_columns),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        X_Vec = spd_linalg.VecBatch.apply(X)
        assert X_Vec.dim() == X.dim() - 1
        if n_matrices > 1:
            assert X_Vec.shape[0] == n_matrices
        assert X_Vec.shape[-1] == n_rows * n_columns
        assert_close(spd_linalg.unvec_batch(X_Vec, n_rows), X)
        assert X_Vec.device == X.device
        assert X_Vec.dtype == X.dtype

    @pytest.mark.parametrize(
        "n_matrices, n_rows, n_columns",
        [(1, 100, 100), (1, 100, 30), (50, 100, 100), (50, 100, 30)],
    )
    def test_backward(self, n_matrices, n_rows, n_columns, device, dtype, generator):
        """
        Test forward of VecBatch Function class
        """
        X = torch.squeeze(
            torch.randn(
                (n_matrices, n_rows, n_columns),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        X_manual_vec = spd_linalg.VecBatch.apply(X_manual)
        X_auto_vec = spd_linalg.vec_batch(X_auto)

        loss_manual = torch.norm(X_manual_vec)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_vec)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert_close(X_manual.grad, X_auto.grad)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features", [100])
    def test_backward_symmetric(self, n_matrices, n_features, device, dtype, generator):
        """
        Test of backward of VecBatch Function class for symmetric matrices
        """
        # random batch of symmetric matrices
        X = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        X_manual_vec = spd_linalg.VecBatch.apply(X_manual)
        X_auto_vec = spd_linalg.vec_batch(X_auto)

        loss_manual = torch.norm(X_manual_vec)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_vec)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


class TestVech:
    """
    Test suite for half vectorization of a batch of symmetric matrices
    """

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_vech_batch(self, n_matrices, n_features, device, dtype, generator):
        """
        Test of vech_batch function
        """
        # random batch of symmetric matrices
        X = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_vech = spd_linalg.vech_batch(X)
        assert X_vech.dim() == X.dim() - 1
        if n_matrices > 1:
            assert X_vech.shape[0] == n_matrices
        assert X_vech.shape[-1] == n_features * (n_features + 1) // 2
        assert_close(spd_linalg.unvech_batch(X_vech, n_features), X)
        assert X_vech.device == X.device
        assert X_vech.dtype == X.dtype

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_unvech_batch(self, n_matrices, n_features, device, dtype, generator):
        """
        Test of unvech_batch function
        """
        dim = n_features * (n_features + 1) // 2
        X_vech = torch.squeeze(
            torch.randn(
                (n_matrices, dim), device=device, dtype=dtype, generator=generator
            )
        )
        X_unvech = spd_linalg.unvech_batch(X_vech, n_features)
        assert X_unvech.dim() == X_vech.dim() + 1
        if n_matrices > 1:
            assert X_unvech.shape[0] == n_matrices
        assert X_unvech.shape[-2] == n_features
        assert X_unvech.shape[-1] == n_features
        assert_close(spd_linalg.vech_batch(X_unvech), X_vech)
        assert X_unvech.device == X_vech.device
        assert X_unvech.dtype == X_vech.dtype

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_forward(self, n_matrices, n_features, device, dtype, generator):
        """
        Test forward of VechBatch Function class
        """
        # random batch of symmetric matrices
        X = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_Vech = spd_linalg.VechBatch.apply(X)
        assert X_Vech.dim() == X.dim() - 1
        if n_matrices > 1:
            assert X_Vech.shape[0] == n_matrices
        assert X_Vech.shape[-1] == n_features * (n_features + 1) // 2
        assert_close(spd_linalg.unvech_batch(X_Vech, n_features), X)
        assert X_Vech.device == X.device
        assert X_Vech.dtype == X.dtype

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_backward(self, n_matrices, n_features, device, dtype, generator):
        """
        Test backward of VechBatch Function class
        Can't rely on automatic differentiation here because it's not working
        due to the fact that we drop values in forward but still need gradient with respect to these
        """
        # random batch of symmetric matrices
        X = spd_linalg.symmetrize(
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
        X_Vech = spd_linalg.VechBatch.apply(X)
        # create random upstream gradient
        grad_X_Vech = torch.squeeze(
            torch.randn(
                (n_matrices, n_features * (n_features + 1) // 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        # backward
        X_Vech.backward(grad_X_Vech)
        # we expect vech_batch(X.grad) == grad_X_Vech
        expected_grad = spd_linalg.vech_batch(X.grad)
        assert_close(expected_grad, grad_X_Vech)


# ----------------------------------------------------------
# Remark:
# -------
# No testing of eigh_operation and eigh_operation_grad,
# it is done indirectly when testing sqrtm, inv_sqrtm, etc.
# ---------------------------------------------------------


# -------------------
# Solve Sylvester SPD
# -------------------
class TestSylvesterSPD:
    """
    Test suite for solving Sylvester equations in SPD case
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_solve(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of solve_sylvester_SPD function with symmetric matrices
        """
        A = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        Q = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        eigvals, eigvecs = torch.linalg.eigh(A)
        X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)
        assert X.shape == A.shape
        assert X.device == A.device
        assert X.dtype == A.dtype
        assert is_symmetric(X)
        assert_close(A @ X + X @ A, Q)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_scipy_comparison(self, n_features, cond, device, dtype, generator):
        """
        Comparison of solve_sylvester_SPD and scipy.linalg.solve_sylvester
        """
        if device.type != "cpu" or dtype != torch.float64:
            pytest.skip("Scipy comparison only valid on CPU with float64")

        A = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        Q = spd_linalg.symmetrize(
            torch.randn(
                (n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        eigvals, eigvecs = torch.linalg.eigh(A)
        X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)
        X_scipy = torch.from_numpy(
            solve_sylvester(A.numpy(), A.numpy(), Q.numpy()),
        ).to(dtype=dtype)
        assert_close(X, X_scipy)

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype):
        """
        Test with identity matrix
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        Q = spd_linalg.symmetrize(
            torch.randn((n_features, n_features), device=device, dtype=dtype)
        )

        eigvals, eigvecs = torch.linalg.eigh(I)
        X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)

        # For A=I: IX + XI = 2X = Q, so X = Q/2
        expected = Q / 2
        assert_close(X, expected)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_solve_skew_symmetric(
        self, n_matrices, n_features, cond, device, dtype, generator
    ):
        """
        Test of solve_sylvester_SPD function with skew-symmetric matrices
        """
        A = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Create skew-symmetric matrix
        Q = torch.squeeze(
            torch.randn(
                (n_matrices, n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        Q = Q - spd_linalg.symmetrize(Q)

        eigvals, eigvecs = torch.linalg.eigh(A)
        X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)

        assert X.shape == A.shape
        assert X.device == A.device
        assert X.dtype == A.dtype
        # Check skew-symmetry
        assert_close(X, -X.transpose(-1, -2))
        assert_close(A @ X + X @ A, Q)


# --------------------------------------------------------------------------
# Remark
# ------
# In the following, on all operations that rely on eignevalue decomposition,
# no use of torch.autograd.gradcheck performed because it is unstable in the
# case of an eigenvalue decomposition.
# --------------------------------------------------------------------------


# ---------
# Sqrtm SPD
# ---------
class TestSqrtmSPD:
    """
    Test suite for matrix square root in SPD case
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of sqrtm_SPD function and forward of SqrtmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_sqrtm, _, _ = spd_linalg.sqrtm_SPD(X)
        X_Sqrtm = spd_linalg.SqrtmSPD.apply(X)
        assert X_sqrtm.shape == X.shape
        assert X_sqrtm.device == X.device
        assert X_sqrtm.dtype == X.dtype
        assert is_spd(X_sqrtm)
        assert_close(X_sqrtm @ X_sqrtm, X)
        assert X_Sqrtm.shape == X.shape
        assert X_Sqrtm.device == X.device
        assert X_Sqrtm.dtype == X.dtype
        assert is_spd(X_Sqrtm)
        assert_close(X_Sqrtm @ X_Sqrtm, X)
        assert_close(X_Sqrtm, X_sqrtm)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_scipy_comparison(self, n_features, cond, device, dtype, generator):
        """
        Comparison of SqrtmSPD and scipy.linalg.sqrtm
        """
        if device.type != "cpu" or dtype != torch.float64:
            pytest.skip("Scipy comparison only valid on CPU with float64")

        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        X_Sqrtm = spd_linalg.SqrtmSPD.apply(X)
        X_sqrtm_scipy = torch.from_numpy(
            sqrtm(X.numpy()),
        ).to(dtype=dtype)
        assert_close(X_Sqrtm, X_sqrtm_scipy)

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype):
        """
        Test that the matrix square root of the identity is the identity
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        I_Sqrtm = spd_linalg.SqrtmSPD.apply(I)
        assert_close(I_Sqrtm, I)

    @pytest.mark.parametrize("n_features", [100])
    def test_diagonal_matrix(self, n_features, device, dtype, generator):
        """
        Test matrix square root of diagonal matrix
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        D = torch.diag(diag_vals)
        D_Sqrtm = spd_linalg.SqrtmSPD.apply(D)
        expected = torch.diag(torch.sqrt(diag_vals))
        assert_close(D_Sqrtm, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_eigenvalues(self, n_features, cond, device, dtype, generator):
        """
        Test that eigenvalues of the matrix square root of X are the square root of the eigenvalues of X
        """
        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        eigvals_X = torch.linalg.eigvalsh(X)
        X_Sqrtm = spd_linalg.SqrtmSPD.apply(X)
        eigvals_X_Sqrtm = torch.linalg.eigvalsh(X_Sqrtm)
        # Sort for comparison (probably not needed)
        expected = torch.sqrt(eigvals_X).sort()[0]
        actual = eigvals_X_Sqrtm.sort()[0]
        assert_close(actual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of SqrtmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        X_manual_sqrtm = spd_linalg.SqrtmSPD.apply(X_manual)
        X_auto_sqrtm, _, _ = spd_linalg.sqrtm_SPD(X_auto)

        loss_manual = torch.norm(X_manual_sqrtm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_sqrtm)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# ------------
# InvSqrtm SPD
# ------------
class TestInvSqrtmSPD:
    """
    Test suite for inverse matrix square root in SPD case
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of inv_sqrtm_SPD function and forward of InvSqrtmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_inv_sqrtm, _, _ = spd_linalg.inv_sqrtm_SPD(X)
        X_InvSqrtm = spd_linalg.InvSqrtmSPD.apply(X)
        assert X_inv_sqrtm.shape == X.shape
        assert X_inv_sqrtm.device == X.device
        assert X_inv_sqrtm.dtype == X.dtype
        assert is_spd(X_inv_sqrtm)
        assert_close(
            X_inv_sqrtm @ X_inv_sqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(X)),
        )
        assert X_InvSqrtm.shape == X.shape
        assert X_InvSqrtm.device == X.device
        assert X_InvSqrtm.dtype == X.dtype
        assert is_spd(X_InvSqrtm)
        assert_close(
            X_InvSqrtm @ X_InvSqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(X)),
        )
        assert_close(X_InvSqrtm, X_inv_sqrtm)

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype):
        """
        Test that the matrix inverse square root of the identity is the identity
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        I_InvSqrtm = spd_linalg.InvSqrtmSPD.apply(I)
        assert_close(I_InvSqrtm, I)

    @pytest.mark.parametrize("n_features", [100])
    def test_diagonal_matrix(self, n_features, device, dtype, generator):
        """
        Test of the matrix inverse square root of a diagonal matrix
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        D = torch.diag(diag_vals)
        D_InvSqrtm = spd_linalg.InvSqrtmSPD.apply(D)
        expected = torch.diag(1 / torch.sqrt(diag_vals))
        assert_close(D_InvSqrtm, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_eigenvalues(self, n_features, cond, device, dtype, generator):
        """
        Test that eigenvalues of the matrix inverse square root of X are the inverse square root of the eigenvalues of X
        """
        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        eigvals_X = torch.linalg.eigvalsh(X)
        X_InvSqrtm = spd_linalg.InvSqrtmSPD.apply(X)
        eigvals_X_InvSqrtm = torch.linalg.eigvalsh(X_InvSqrtm)
        # Sort for comparison
        expected = (1 / torch.sqrt(eigvals_X)).sort()[0]
        actual = eigvals_X_InvSqrtm.sort()[0]
        assert_close(actual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of SqrtmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        X_manual_inv_sqrtm = spd_linalg.InvSqrtmSPD.apply(X_manual)
        X_auto_inv_sqrtm, _, _ = spd_linalg.inv_sqrtm_SPD(X_auto)

        loss_manual = torch.norm(X_manual_inv_sqrtm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_inv_sqrtm)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# --------
# Powm SPD
# --------
class TestPowmSPD:
    """
    Test suite for matrix power in SPD case
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of powm_spd function and forward of PowmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)

        X_powm, _, _ = spd_linalg.powm_SPD(X, exponent)
        X_Powm = spd_linalg.PowmSPD.apply(X, exponent)

        X_orig, _, _ = spd_linalg.powm_SPD(X_powm, 1 / exponent)
        X_Orig = spd_linalg.PowmSPD.apply(X_Powm, 1 / exponent)

        assert X_powm.shape == X.shape
        assert X_powm.device == X.device
        assert X_powm.dtype == X.dtype
        assert is_spd(X_powm)
        assert_close(X_orig, X)
        assert X_Powm.shape == X.shape
        assert X_Powm.device == X.device
        assert X_Powm.dtype == X.dtype
        assert is_spd(X_Powm)
        assert_close(X_Orig, X)
        assert_close(X_Powm, X_powm)

        # compare with some specific powers
        assert_close(spd_linalg.powm_SPD(X, 2)[0], X @ X)
        assert_close(spd_linalg.PowmSPD.apply(X, 2), X @ X)
        assert_close(
            spd_linalg.powm_SPD(X, -1)[0],
            torch.cholesky_inverse(torch.linalg.cholesky(X)),
        )
        assert_close(
            spd_linalg.PowmSPD.apply(X, -1),
            torch.cholesky_inverse(torch.linalg.cholesky(X)),
        )
        assert_close(spd_linalg.powm_SPD(X, 0.5)[0], spd_linalg.sqrtm_SPD(X)[0])
        assert_close(spd_linalg.PowmSPD.apply(X, 0.5), spd_linalg.sqrtm_SPD(X)[0])

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype, generator):
        """
        Test that the matrix power of the identity is the identity
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)
        I_Powm = spd_linalg.PowmSPD.apply(I, exponent)
        assert_close(I_Powm, I)

    @pytest.mark.parametrize("n_features", [100])
    def test_diagonal_matrix(self, n_features, device, dtype, generator):
        """
        Test of the matrix power of a diagonal matrix
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        D = torch.diag(diag_vals)
        exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)
        D_Powm = spd_linalg.PowmSPD.apply(D, exponent)
        expected = torch.diag(torch.pow(diag_vals, exponent))
        assert_close(D_Powm, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_eigenvalues(self, n_features, cond, device, dtype, generator):
        """
        Test that eigenvalues of the matrix power of X are the power of the eigenvalues of X
        """
        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        eigvals_X = torch.linalg.eigvalsh(X)
        exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)
        X_Powm = spd_linalg.PowmSPD.apply(X, exponent)
        eigvals_X_Powm = torch.linalg.eigvalsh(X_Powm)
        # Sort for comparison
        expected = torch.pow(eigvals_X, exponent).sort()[0]
        actual = eigvals_X_Powm.sort()[0]
        assert_close(actual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of PowmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        # exponent = torch.tensor(0.05)
        exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)
        exponent_manual = exponent.clone().detach()
        exponent_manual.requires_grad = True
        exponent_auto = exponent.clone().detach()
        exponent_auto.requires_grad = True

        X_manual_powm = spd_linalg.PowmSPD.apply(X_manual, exponent_manual)
        X_auto_powm, _, _ = spd_linalg.powm_SPD(X_auto, exponent_auto)

        loss_manual = torch.norm(X_manual_powm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_powm)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# --------
# Logm SPD
# --------
class TestLogmSPD:
    """
    Test suite for matrix logarithm in SPD case
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of logm_SPD function and forward of LogmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_Logm = spd_linalg.LogmSPD.apply(X)
        X_logm, _, _ = spd_linalg.logm_SPD(X)

        assert X_Logm.shape == X.shape
        assert X_Logm.device == X.device
        assert X_Logm.dtype == X.dtype
        assert is_symmetric(X_Logm)
        assert_close(spd_linalg.expm_symmetric(X_Logm)[0], X)

        assert X_logm.shape == X.shape
        assert X_logm.device == X.device
        assert X_logm.dtype == X.dtype
        assert is_symmetric(X_logm)
        assert_close(spd_linalg.expm_symmetric(X_logm)[0], X)
        assert_close(X_Logm, X_logm)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_scipy_comparison(self, n_features, cond, device, dtype, generator):
        """
        Comparison of LogmSPD with scipy.linalg.logm
        """
        if device.type != "cpu" or dtype != torch.float64:
            pytest.skip("Scipy comparison only valid on CPU with float64")

        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        X_Logm = spd_linalg.LogmSPD.apply(X)
        X_logm_scipy = torch.from_numpy(
            logm(X.numpy()),
        ).to(dtype=dtype)
        assert_close(X_Logm, X_logm_scipy)

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype):
        """
        Test that the matrix logarithm of the identity is the identity
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        I_Logm = spd_linalg.LogmSPD.apply(I)
        assert_close(I_Logm, torch.zeros_like(I))

    @pytest.mark.parametrize("n_features", [100])
    def test_diagonal_matrix(self, n_features, device, dtype, generator):
        """
        Test of the matrix logarithm of a diagonal matrix
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        D = torch.diag(diag_vals)
        D_Logm = spd_linalg.LogmSPD.apply(D)
        expected = torch.diag(torch.log(diag_vals))
        assert_close(D_Logm, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_eigenvalues(self, n_features, cond, device, dtype, generator):
        """
        Test that eigenvalues of the matrix logarithm of X are the logarithm of the eigenvalues of X
        """
        X = random_SPD(
            n_features, cond=cond, device=device, dtype=dtype, generator=generator
        )
        eigvals_X = torch.linalg.eigvalsh(X)
        X_Logm = spd_linalg.LogmSPD.apply(X)
        eigvals_X_Logm = torch.linalg.eigvalsh(X_Logm)
        # Sort for comparison
        expected = (torch.log(eigvals_X)).sort()[0]
        actual = eigvals_X_Logm.sort()[0]
        assert_close(actual, expected)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of LogmSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        X_manual_logm = spd_linalg.LogmSPD.apply(X_manual)
        X_auto_logm, _, _ = spd_linalg.logm_SPD(X_auto)

        loss_manual = torch.norm(X_manual_logm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_logm)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# --------------
# Expm Symmetric
# --------------
class TestExpmSymmetric:
    """
    Test suite for matrix exponential in SPD case
    """

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_forward(self, n_matrices, n_features, device, dtype, generator):
        """
        Test of expm_symmetric function and forward of ExpmSymmetric Function class
        """
        X_symmetric = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_expm, _, _ = spd_linalg.expm_symmetric(X_symmetric)
        X_Expm = spd_linalg.ExpmSymmetric.apply(X_symmetric)

        assert X_expm.shape == X_symmetric.shape
        assert X_expm.device == X_symmetric.device
        assert X_expm.dtype == X_symmetric.dtype
        assert is_spd(X_expm)
        assert_close(spd_linalg.logm_SPD(X_expm)[0], X_symmetric, atol=1e-5, rtol=1e-6)

        assert X_Expm.shape == X_symmetric.shape
        assert X_Expm.device == X_symmetric.device
        assert X_Expm.dtype == X_symmetric.dtype
        assert is_spd(X_Expm)
        assert_close(spd_linalg.logm_SPD(X_Expm)[0], X_symmetric, atol=1e-5, rtol=1e-6)
        assert_close(X_Expm, X_expm)

    @pytest.mark.parametrize("n_features", [100])
    def test_scipy_comparison(self, n_features, device, dtype, generator):
        """
        Comparison of ExpmSymmetric with scipy.linalg.expm
        """
        if device.type != "cpu" or dtype != torch.float64:
            pytest.skip("Scipy comparison only valid on CPU with float64")

        X_symmetric = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_Expm = spd_linalg.ExpmSymmetric.apply(X_symmetric)
        X_expm_scipy = torch.from_numpy(
            expm(X_symmetric.numpy()),
        ).to(dtype=dtype)
        assert_close(X_Expm, X_expm_scipy)

    @pytest.mark.parametrize("n_features", [100])
    def test_identity_matrix(self, n_features, device, dtype):
        """
        Test that the matrix exponential of the identity is the identity
        """
        I = torch.eye(n_features, device=device, dtype=dtype)
        I_Expm = spd_linalg.ExpmSymmetric.apply(I)
        assert_close(
            I_Expm,
            torch.diag(
                torch.exp(torch.ones((n_features,), device=device, dtype=dtype))
            ),
        )

    @pytest.mark.parametrize("n_features", [100])
    def test_diagonal_matrix(self, n_features, device, dtype, generator):
        """
        Test of the matrix exponential of a diagonal matrix
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        D = torch.diag(diag_vals)
        D_Expm = spd_linalg.ExpmSymmetric.apply(D)
        expected = torch.diag(torch.exp(diag_vals))
        assert_close(D_Expm, expected)

    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_eigenvalues(self, n_features, cond, device, dtype, generator):
        """
        Test that eigenvalues of the matrix exponential of X are the exponential of the eigenvalues of X
        """
        X = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        eigvals_X = torch.linalg.eigvalsh(X)
        X_Expm = spd_linalg.ExpmSymmetric.apply(X)
        eigvals_X_Expm = torch.linalg.eigvalsh(X_Expm)
        # Sort for comparison
        expected = (torch.exp(eigvals_X)).sort()[0]
        actual = eigvals_X_Expm.sort()[0]
        assert_close(actual, expected)

    @pytest.mark.parametrize("n_matrices, n_features", [(1, 100), (50, 100)])
    def test_backward(self, n_matrices, n_features, device, dtype, generator):
        """
        Test of backward of ExpmSymmetric Function class
        """
        X_symmetric = spd_linalg.symmetrize(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        X_manual = X_symmetric.clone().detach()
        X_manual.requires_grad = True
        X_auto = X_symmetric.clone().detach()
        X_auto.requires_grad = True

        X_manual_expm = spd_linalg.ExpmSymmetric.apply(X_manual)
        X_auto_expm, _, _ = spd_linalg.expm_symmetric(X_auto)

        loss_manual = torch.norm(X_manual_expm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_expm)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# --------
# EighReLu
# --------
class TestEighReLu:
    """
    Test suite for ReLu activation functin on eigenvalues of SPD matrices
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, n_small_eigvals, eps", [(100, 30, 1e-4)])
    def test_forward(
        self, n_matrices, n_features, n_small_eigvals, eps, device, dtype, generator
    ):
        """
        Test of eigh_relu function and forward of EighReLu Function class
        """
        # Generate random SPD matrices with some small enough eigenvalues
        U, _, _ = torch.svd(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        L = (
            torch.randn(
                (n_matrices, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            ** 2
        )
        L[:, :n_small_eigvals] = eps / 100
        X = (U * L.unsqueeze(-2)) @ U.transpose(-1, -2)

        Z, _, _ = spd_linalg.eigh_relu(X, eps)
        Y = spd_linalg.EighReLu.apply(X, eps)

        assert Z.shape == X.shape
        assert Z.device == X.device
        assert Z.dtype == X.dtype
        assert is_spd(Z)
        assert bool((torch.linalg.eigvalsh(Z) >= eps - eps / 10).all())

        assert Y.shape == X.shape
        assert Y.device == X.device
        assert Y.dtype == X.dtype
        assert is_spd(Y)
        assert bool((torch.linalg.eigvalsh(Y) >= eps - eps / 10).all())

    @pytest.mark.parametrize("n_features, n_small_eigvals, eps", [(100, 30, 1e-4)])
    def test_rectification_diagonal(
        self, n_features, n_small_eigvals, eps, device, dtype, generator
    ):
        """
        Test that values of diagonal matrix below eps are actually set to eps
        """
        diag_vals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        diag_vals[:n_small_eigvals] = eps / 100
        diag_vals = diag_vals[torch.randperm(n_features)]
        D = torch.diag(diag_vals)
        Z = spd_linalg.EighReLu.apply(D, eps)
        eigvals_out = torch.linalg.eigvalsh(Z)
        expected = torch.clamp(diag_vals, min=eps)
        assert_close(eigvals_out.sort()[0], expected.sort()[0])

    @pytest.mark.parametrize("n_features, n_small_eigvals, eps", [(100, 30, 1e-4)])
    def test_rectification_eigenvalues(
        self, n_features, n_small_eigvals, eps, device, dtype, generator
    ):
        """
        Test that eigenvalues below eps are actually set to eps
        """
        U, _, _ = torch.svd(
            torch.randn(
                (n_features, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        eigvals = (
            torch.randn((n_features,), device=device, dtype=dtype, generator=generator)
            ** 2
        )
        eigvals[:n_small_eigvals] = eps / 100
        eigvals = eigvals[torch.randperm(n_features)]
        X = (U * eigvals.unsqueeze(-2)) @ U.transpose(-1, -2)
        Z = spd_linalg.EighReLu.apply(X, eps)
        eigvals_out = torch.linalg.eigvalsh(Z)
        expected = torch.clamp(eigvals, min=eps)
        assert_close(eigvals_out.sort()[0], expected.sort()[0])

    @pytest.mark.parametrize("n_features", [100])
    def test_no_rectification_needed(self, n_features, device, dtype, generator):
        """
        Test when all eigenvalues are already above eps
        """
        # selection of eps and cond ensures eigenvalues are all above eps
        # one needs 1/sqrt(cond) > eps
        eps = 1e-4
        X = random_SPD(
            n_features, cond=100, device=device, dtype=dtype, generator=generator
        )
        # Ensure all eigenvalues > eps
        eigvals_before = torch.linalg.eigvalsh(X)
        if eigvals_before.min() < eps:
            pytest.skip("Generated matrix has eigenvalues below eps")
        # apply EighReLu
        Z = spd_linalg.EighReLu.apply(X, eps)
        # Output should be identical to input
        assert_close(Z, X)

    @pytest.mark.parametrize("n_features", [100])
    def test_all_eigenvalues_below_eps(self, n_features, device, dtype, generator):
        """
        Test when all eigenvalues need rectification
        """
        eps = 20.0  # Large threshold
        X = random_SPD(
            n_features, cond=100, device=device, dtype=dtype, generator=generator
        )
        # Apply EighReLu
        Z = spd_linalg.EighReLu.apply(X, eps)
        # Z should be eps * I
        I = torch.eye(n_features, device=device, dtype=dtype)
        assert_close(Z, eps * I)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, n_small_eigvals, eps", [(100, 30, 1e-4)])
    def test_backward(
        self, n_matrices, n_features, n_small_eigvals, eps, device, dtype, generator
    ):
        """
        Test of backward of EighReLu Function class
        """
        # Generate random SPD matrices with some small enough eigenvalues
        U, _, _ = torch.svd(
            torch.squeeze(
                torch.randn(
                    (n_matrices, n_features, n_features),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        )
        L = (
            torch.randn(
                (n_matrices, n_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            ** 2
        )
        L[:, :n_small_eigvals] = eps / 100
        X = (U * L.unsqueeze(-2)) @ U.transpose(-1, -2)

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        Y_manual = spd_linalg.EighReLu.apply(X_manual, eps)
        Y_auto, _, _ = spd_linalg.eigh_relu(X_auto, eps)

        loss_manual = torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(Y_auto)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        # need to lower tolerances to get True
        assert_close(X_manual.grad, X_auto.grad, atol=1e-4, rtol=1e-6)


# -----------
# Congruences
# -----------
class TestCongruenceSPD:
    """
    Test suite for congruence of SPD matrices with an SPD matrix
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of congruence_SPD function and forward of CongruenceSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )

        Y = spd_linalg.CongruenceSPD.apply(X, G)
        Z = spd_linalg.congruence_SPD(X, G)

        X_class = spd_linalg.CongruenceSPD.apply(
            Y, torch.cholesky_inverse(torch.linalg.cholesky(G))
        )
        X_function = spd_linalg.congruence_SPD(
            Z, torch.cholesky_inverse(torch.linalg.cholesky(G))
        )

        assert Y.shape == X.shape
        assert Y.device == X.device
        assert Y.dtype == X.dtype
        assert Z.shape == X.shape
        assert Z.device == X.device
        assert Z.dtype == X.dtype
        assert is_spd(Y)
        assert is_spd(Z)
        assert_close(Y, Z)
        assert_close(X_class, X)
        assert_close(X_function, X)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of CongruenceSPD Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        G_manual = G.clone().detach()
        G_manual.requires_grad = True
        G_auto = G.clone().detach()
        G_auto.requires_grad = True

        X_manual_cong = spd_linalg.CongruenceSPD.apply(X_manual, G_manual)
        X_auto_cong = spd_linalg.congruence_SPD(X_auto, G_auto)

        loss_manual = torch.norm(X_manual_cong)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_cong)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert G_manual.grad is not None
        assert G_auto.grad is not None
        assert torch.isfinite(G_manual.grad).all()
        assert torch.isfinite(G_auto.grad).all()
        assert is_symmetric(G_manual.grad)
        assert is_symmetric(G_auto.grad)
        assert_close(G_manual.grad, G_auto.grad)


class TestWhitening:
    """
    Test suite for whitening of a batch of SPD matrices with a SPD matrix
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_forward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of whitening function and forward of Whitening Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )

        Y = spd_linalg.Whitening.apply(X, G)
        Z = spd_linalg.whitening(X, G)

        X_class = spd_linalg.Whitening.apply(
            Y, torch.cholesky_inverse(torch.linalg.cholesky(G))
        )
        X_function = spd_linalg.whitening(
            Z, torch.cholesky_inverse(torch.linalg.cholesky(G))
        )

        assert Y.shape == X.shape
        assert Y.device == X.device
        assert Y.dtype == X.dtype
        assert Z.shape == X.shape
        assert Z.device == X.device
        assert Z.dtype == X.dtype
        assert is_spd(Y)
        assert is_spd(Z)
        assert_close(Y, Z)
        assert_close(X_class, X)
        assert_close(X_function, X)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_features, cond", [(100, 1000)])
    def test_backward(self, n_matrices, n_features, cond, device, dtype, generator):
        """
        Test of backward of Whitening Function class
        """
        X = random_SPD(
            n_features,
            n_matrices,
            cond=cond,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        G = random_SPD(
            n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator
        )

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        G_manual = G.clone().detach()
        G_manual.requires_grad = True
        G_auto = G.clone().detach()
        G_auto.requires_grad = True

        X_manual_white = spd_linalg.Whitening.apply(X_manual, G_manual)
        X_auto_white = spd_linalg.whitening(X_auto, G_auto)

        loss_manual = torch.norm(X_manual_white)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_white)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert G_manual.grad is not None
        assert G_auto.grad is not None
        assert torch.isfinite(G_manual.grad).all()
        assert torch.isfinite(G_auto.grad).all()
        assert is_symmetric(G_manual.grad)
        assert is_symmetric(G_auto.grad)
        assert_close(G_manual.grad, G_auto.grad)


class TestCongruenceRectangular:
    """
    Test suite for congruence of a batch of SPD matrices with a rectangular matrix (dimensionality reduction)
    """

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out, cond", [(100, 70, 1000)])
    def test_forward(self, n_matrices, n_in, n_out, cond, device, dtype, generator):
        """
        Test of congruence_rectangular function and forward of CongruenceSPD Function class
        """
        # Generate random SPD matrices
        X = random_SPD(
            n_in, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator
        )
        # Generate random Stiefel matrix
        W = torch.empty((n_in, n_out), device=device, dtype=dtype)
        _init_weights_stiefel(W, generator=generator)

        Y = spd_linalg.CongruenceRectangular.apply(X, W)
        Z = spd_linalg.CongruenceRectangular.apply(X, W)

        assert Y.dim() == X.dim()
        if n_matrices > 1:
            assert Y.shape[0] == X.shape[0]
        assert Y.shape[-2] == n_out
        assert Y.shape[-1] == n_out
        assert Y.device == X.device
        assert Y.dtype == X.dtype
        assert is_spd(Y)

        assert Z.dim() == X.dim()
        if n_matrices > 1:
            assert Z.shape[0] == X.shape[0]
        assert Z.shape[-2] == n_out
        assert Z.shape[-1] == n_out
        assert Z.device == X.device
        assert Z.dtype == X.dtype
        assert is_spd(Z)

    @pytest.mark.parametrize("n_matrices", [1, 50])
    @pytest.mark.parametrize("n_in, n_out, cond", [(100, 70, 1000)])
    def test_backward(self, n_matrices, n_in, n_out, cond, device, dtype, generator):
        """
        Test of backward of CongruenceRectangular Function class
        """
        # Generate random SPD matrices
        X = random_SPD(
            n_in, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator
        )
        # Generate random Stiefel matrix
        W = torch.empty((n_in, n_out), device=device, dtype=dtype)
        _init_weights_stiefel(W, generator=generator)

        X_manual = X.clone().detach()
        X_manual.requires_grad = True
        X_auto = X.clone().detach()
        X_auto.requires_grad = True

        W_manual = W.clone().detach()
        W_manual.requires_grad = True
        W_auto = W.clone().detach()
        W_auto.requires_grad = True

        X_manual_cong = spd_linalg.CongruenceRectangular.apply(X_manual, W_manual)
        X_auto_cong = spd_linalg.congruence_rectangular(X_auto, W_auto)

        loss_manual = torch.norm(X_manual_cong)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_cong)
        loss_auto.backward()

        assert X_manual.grad is not None
        assert X_auto.grad is not None
        assert torch.isfinite(X_manual.grad).all()
        assert torch.isfinite(X_auto.grad).all()
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)

        assert W_manual.grad is not None
        assert W_auto.grad is not None
        assert torch.isfinite(W_manual.grad).all()
        assert torch.isfinite(W_auto.grad).all()
        assert_close(W_manual.grad, W_auto.grad)
