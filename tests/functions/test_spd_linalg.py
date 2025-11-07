import pytest
import torch
from scipy.linalg import expm, logm, solve_sylvester, sqrtm
from torch.testing import assert_close

import yetanotherspdnet.functions.spd_linalg as spd_linalg
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.random.stiefel import _init_weights_stiefel


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



def is_symmetric(X: torch.Tensor) -> bool:
    """
    Check whether matrices in a batch are symmetric

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrices

    Returns
    -------
    bool
        boolean
    """
    is_square = bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    return bool(torch.allclose(X, X.transpose(-1, -2)))


def is_spd(X: torch.Tensor) -> bool:
    """
    Check whether matrices in a batch are SPD

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        batch of matrices

    Returns
    -------
    bool
        boolean
    """
    is_square = bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    is_symmetric = bool(torch.allclose(X, X.transpose(-1, -2)))
    is_positivedefinite = bool((torch.linalg.eigvalsh(X) > 0).all())
    return is_symmetric and is_positivedefinite


# ----------
# Symmetrize
# ----------
@pytest.mark.parametrize("n_matrices, n_features", [(1,100), (50,100)])
def test_symmetrize(n_matrices, n_features, device, dtype, generator):
    """
    Test of symmetrize function
    """
    X = torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    X_sym = spd_linalg.symmetrize(X)
    X_skew = X - X_sym
    assert_close(X_sym, X_sym.transpose(-1,-2))
    assert_close(X_skew, -X_skew.transpose(-1,-2))
    assert_close(X, X_sym+X_skew)


# -------------
# Vectorization
# -------------
@pytest.mark.parametrize("n_matrices, n_rows, n_columns", [(1, 100, 100), (1, 100, 30), (50,100,100), (50, 100, 30)])
def test_vec_batch(n_matrices, n_rows, n_columns, device, dtype, generator):
    """
    Test of vec_batch function
    """
    X = torch.squeeze(torch.randn((n_matrices, n_rows, n_columns), device=device, dtype=dtype, generator=generator))
    X_vec = spd_linalg.vec_batch(X)
    assert X_vec.dim() == X.dim() - 1
    if n_matrices > 1 :
        assert X_vec.shape[0] == n_matrices
    assert X_vec.shape[-1] == n_rows * n_columns
    assert_close(spd_linalg.unvec_batch(X_vec, n_rows), X)


@pytest.mark.parametrize("n_matrices, n_rows, n_columns", [(1, 100, 100), (1, 100, 30), (50,100,100), (50, 100, 30)])
def test_unvec_batch(n_matrices, n_rows, n_columns, device, dtype, generator):
    """
    Test of unvec_batch function
    """
    X_vec = torch.squeeze(torch.randn((n_matrices, n_rows * n_columns), device=device, dtype=dtype, generator=generator))
    X_unvec = spd_linalg.unvec_batch(X_vec, n_rows)
    assert X_unvec.dim() == X_vec.dim() + 1
    if n_matrices > 1 :
        assert X_unvec.shape[0] == n_matrices
    assert X_unvec.shape[-2] == n_rows
    assert X_unvec.shape[-1] == n_columns
    assert_close(spd_linalg.vec_batch(X_unvec), X_vec)


@pytest.mark.parametrize("n_matrices, n_features", [(1,100), (50,100)])
def test_vech_batch(n_matrices, n_features, device, dtype, generator):
    """
    Test of vech_batch function
    """
    # random batch of symmetric matrices
    X = spd_linalg.symmetrize(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    X_vech = spd_linalg.vech_batch(X)
    assert X_vech.dim() == X.dim() - 1
    if n_matrices > 1 :
        assert X_vech.shape[0] == n_matrices
    assert X_vech.shape[-1] == n_features * (n_features + 1) // 2
    assert_close(spd_linalg.unvech_batch(X_vech), X)


@pytest.mark.parametrize("n_matrices, n_features", [(1,100), (50,100)])
def test_unvech_batch(n_matrices, n_features, device, dtype, generator):
    """
    Test of unvech_batch function
    """
    dim = n_features * (n_features + 1) // 2
    X_vech = torch.squeeze(torch.randn((n_matrices, dim), device=device, dtype=dtype, generator=generator))
    X_unvech = spd_linalg.unvech_batch(X_vech)
    assert X_unvech.dim() == X_vech.dim() + 1
    if n_matrices > 1 :
        assert X_unvech.shape[0] == n_matrices
    assert X_unvech.shape[-2] == n_features
    assert X_unvech.shape[-1] == n_features
    assert_close(spd_linalg.vech_batch(X_unvech), X_vech)


# ----------------------------------------------------------
# Remark:
# -------
# No testing of eigh_operation and eigh_operation_grad,
# it is done indirectly when testing sqrtm, inv_sqrtm, etc.
# ---------------------------------------------------------


# -------------------
# Solve Sylvester SPD
# -------------------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_sylvester_SPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of solve_sylvester_SPD function
    """
    A = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    Q = spd_linalg.symmetrize(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    eigvals, eigvecs = torch.linalg.eigh(A)
    X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)
    assert X.shape == A.shape
    assert is_symmetric(X)
    assert_close(A @ X + X @ A, Q)

@pytest.mark.parametrize("n_features, cond", [(100,1000)])
def test_sylvester_scipy_comparison(n_features, cond, device, dtype, generator):
    """
    Comparison of solve_sylvester_SPD and scipy.linalg.solve_sylvester
    """
    A = random_SPD(n_features, cond=cond, device=device, dtype=dtype, generator=generator)
    Q = spd_linalg.symmetrize(
        torch.randn((n_features, n_features), device=device, dtype=dtype, generator=generator)
    )
    eigvals, eigvecs = torch.linalg.eigh(A)
    X = spd_linalg.solve_sylvester_SPD(eigvals, eigvecs, Q)
    X_scipy = torch.tensor(
        solve_sylvester(A.detach().numpy(), A.detach().numpy(), Q.detach().numpy()),
        dtype = A.dtype
    )
    assert_close(X, X_scipy)


# ---------
# Sqrtm SPD
# ---------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_SqrtmSPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of sqrtm_SPD function and forward of SqrtmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    X_sqrtm, _, _ = spd_linalg.sqrtm_SPD(X)
    X_Sqrtm = spd_linalg.SqrtmSPD.apply(X)
    assert X_sqrtm.shape == X.shape
    assert is_spd(X_sqrtm)
    assert_close(X_sqrtm @ X_sqrtm, X)
    assert X_Sqrtm.shape == X.shape
    assert is_spd(X_Sqrtm)
    assert_close(X_Sqrtm @ X_Sqrtm, X)
    assert_close(X_Sqrtm, X_sqrtm)

@pytest.mark.parametrize("n_features, cond", [(100,1000)])
def test_SqrtmSPD_scipy_comparison(n_features, cond, device, dtype, generator):
    """
    Comparison of SqrtmSPD and scipy.linalg.sqrtm
    """
    X = random_SPD(n_features, cond=cond, device=device, dtype=dtype, generator=generator)
    X_Sqrtm = spd_linalg.SqrtmSPD.apply(X)
    X_sqrtm_scipy = torch.tensor(
        sqrtm(X.detach().numpy()),
        dtype = X.dtype
    )
    assert_close(X_Sqrtm, X_sqrtm_scipy)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_SqrtmSPD_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of SqrtmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)


# ------------
# InvSqrtm SPD
# ------------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_InvSqrtmSPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of inv_sqrtm_SPD function and forward of InvSqrtmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    X_inv_sqrtm, _, _ = spd_linalg.inv_sqrtm_SPD(X)
    X_InvSqrtm = spd_linalg.InvSqrtmSPD.apply(X)
    assert X_inv_sqrtm.shape == X.shape
    assert is_spd(X_inv_sqrtm)
    assert_close(
        X_inv_sqrtm @ X_inv_sqrtm,
        torch.cholesky_inverse(torch.linalg.cholesky(X)),
    )
    assert X_InvSqrtm.shape == X.shape
    assert is_spd(X_InvSqrtm)
    assert_close(
        X_InvSqrtm @ X_InvSqrtm,
        torch.cholesky_inverse(torch.linalg.cholesky(X)),
    )
    assert_close(X_InvSqrtm, X_inv_sqrtm)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_InvSqrtmSPD_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of SqrtmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)

# --------
# Powm SPD
# --------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_PowmSPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of powm_spd function and forward of PowmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    exponent = torch.randn(1, device=device, dtype=dtype, generator=generator)

    X_powm, _, _ = spd_linalg.powm_SPD(X, exponent)
    X_Powm = spd_linalg.PowmSPD.apply(X, exponent)

    X_orig, _, _ = spd_linalg.powm_SPD(X_powm, 1/exponent)
    X_Orig = spd_linalg.PowmSPD.apply(X_Powm, 1/exponent)

    assert X_powm.shape == X.shape
    assert is_spd(X_powm)
    assert_close(X_orig, X)
    assert X_Powm.shape == X.shape
    assert is_spd(X_Powm)
    assert_close(X_Orig, X)
    assert_close(X_Powm, X_powm)

    # compare with some specific powers
    assert_close(spd_linalg.powm_SPD(X,2)[0], X@X)
    assert_close(spd_linalg.PowmSPD.apply(X,2), X@X)
    assert_close(spd_linalg.powm_SPD(X,-1)[0], torch.cholesky_inverse(torch.linalg.cholesky(X)))
    assert_close(spd_linalg.PowmSPD.apply(X,-1), torch.cholesky_inverse(torch.linalg.cholesky(X)))
    assert_close(spd_linalg.powm_SPD(X,0.5)[0], spd_linalg.sqrtm_SPD(X)[0])
    assert_close(spd_linalg.PowmSPD.apply(X,0.5), spd_linalg.sqrtm_SPD(X)[0])


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_PowmSPD_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of PowmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    X_manual = X.clone().detach()
    X_manual.requires_grad = True
    X_auto = X.clone().detach()
    X_auto.requires_grad = True

    #exponent = torch.tensor(0.05)
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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)


# --------
# Logm SPD
# --------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_LogmSPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of logm_SPD function and forward of LogmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    X_Logm = spd_linalg.LogmSPD.apply(X)
    X_logm, _, _ = spd_linalg.logm_SPD(X)

    assert X_Logm.shape == X.shape
    assert is_symmetric(X_Logm)
    assert_close(spd_linalg.expm_symmetric(X_Logm)[0], X)

    assert X_logm.shape == X.shape
    assert is_symmetric(X_logm)
    assert_close(spd_linalg.expm_symmetric(X_logm)[0], X)
    assert_close(X_Logm, X_logm)


@pytest.mark.parametrize("n_features, cond", [(100,1000)])
def test_LogmSPD_scipy_comparison(n_features, cond, device, dtype, generator):
    """
    Comparison of LogmSPD with scipy.linalg.logm
    """
    X = random_SPD(n_features, cond=cond, device=device, dtype=dtype, generator=generator)
    X_Logm = spd_linalg.LogmSPD.apply(X)
    X_logm_scipy = torch.tensor(
        logm(X.detach().numpy()),
        dtype=X.dtype
    )
    assert_close(X_Logm, X_logm_scipy)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_LogmSPD_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of LogmSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)


# --------------
# Expm Symmetric
# --------------
@pytest.mark.parametrize("n_matrices, n_features", [(1,100), (50,100)])
def test_ExpmSymmetric(n_matrices, n_features, device, dtype, generator):
    """
    Test of expm_symmetric function and forward of ExpmSymmetric Function class
    """
    X_symmetric = spd_linalg.symmetrize(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    X_expm, _, _ = spd_linalg.expm_symmetric(X_symmetric)
    X_Expm = spd_linalg.ExpmSymmetric.apply(X_symmetric)

    assert X_expm.shape == X_symmetric.shape
    assert is_spd(X_expm)
    assert_close(spd_linalg.logm_SPD(X_expm)[0], X_symmetric, atol=1e-5, rtol=1e-6)

    assert X_Expm.shape == X_symmetric.shape
    assert is_spd(X_Expm)
    assert_close(spd_linalg.logm_SPD(X_Expm)[0], X_symmetric, atol=1e-5, rtol=1e-6)
    assert_close(X_Expm, X_expm)


@pytest.mark.parametrize("n_features", [100])
def test_ExpmSymmetric_scipy_comparison(n_features, device, dtype, generator):
    """
    Comparison of ExpmSymmetric with scipy.linalg.expm
    """
    X_symmetric = spd_linalg.symmetrize(
        torch.squeeze(torch.randn((n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    X_Expm = spd_linalg.ExpmSymmetric.apply(X_symmetric)
    X_expm_scipy = torch.tensor(
        expm(X_symmetric.detach().numpy()),
        dtype=X_symmetric.dtype
    )
    assert_close(X_Expm, X_expm_scipy)


@pytest.mark.parametrize("n_matrices, n_features", [(1,100), (50,100)])
def test_ExpmSymmetric_backward(n_matrices, n_features, device, dtype, generator):
    """
    Test of backward of ExpmSymmetric Function class
    """
    X_symmetric = spd_linalg.symmetrize(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)


# --------
# EighReLu
# --------
@pytest.mark.parametrize("n_matrices, n_features, n_small_eigvals, eps", [(1,100,30, 1e-4), (50,100,30,1e-4)])
def test_EighReLu(n_matrices, n_features, n_small_eigvals, eps, device, dtype, generator):
    """
    Test of eigh_relu function and forward of EighReLu Function class
    """
    # Generate random SPD matrices with some small enough eigenvalues
    U, _, _ = torch.svd(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    L = torch.randn((n_matrices, n_features), device=device, dtype=dtype, generator=generator) ** 2
    L[:, : n_small_eigvals] = eps / 100
    X = (U * L.unsqueeze(-2)) @ U.transpose(-1, -2)

    Z, _, _ = spd_linalg.eigh_relu(X, eps)
    Y = spd_linalg.EighReLu.apply(X, eps)

    assert Z.shape == X.shape
    assert is_spd(Z)
    assert bool((torch.linalg.eigvalsh(Z) >= eps - eps / 10).all())

    assert Y.shape == X.shape
    assert is_spd(Y)
    assert bool((torch.linalg.eigvalsh(Y) >= eps - eps / 10).all())


@pytest.mark.parametrize("n_matrices, n_features, n_small_eigvals, eps", [(1,100,30, 1e-4), (50,100,30,1e-4)])
def test_EighReLu_backward(n_matrices, n_features, n_small_eigvals, eps, device, dtype, generator):
    """
    Test of backward of EighReLu Function class
    """
    # Generate random SPD matrices with some small enough eigenvalues
    U, _, _ = torch.svd(
        torch.squeeze(torch.randn((n_matrices, n_features, n_features), device=device, dtype=dtype, generator=generator))
    )
    L = torch.randn((n_matrices, n_features), device=device, dtype=dtype, generator=generator) ** 2
    L[:, : n_small_eigvals] = eps / 100
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

    assert X_manual.grad.shape == X_manual.shape
    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    # need to lower tolerances to get True
    assert_close(X_manual.grad, X_auto.grad, atol=1e-4, rtol=1e-6)


# -----------
# Congruences
# -----------
@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_CongruenceSPD(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of congruence_SPD function and forward of CongruenceSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    G = random_SPD(n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator)

    Y = spd_linalg.CongruenceSPD.apply(X, G)
    Z = spd_linalg.congruence_SPD(X, G)

    X_class = spd_linalg.CongruenceSPD.apply(
        Y, torch.cholesky_inverse(torch.linalg.cholesky(G))
    )
    X_function = spd_linalg.congruence_SPD(
        Z, torch.cholesky_inverse(torch.linalg.cholesky(G))
    )

    assert Y.shape == X.shape
    assert Z.shape == X.shape
    assert is_spd(Y)
    assert is_spd(Z)
    assert_close(Y, Z)
    assert_close(X_class, X)
    assert_close(X_function, X)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_CongruenceSPD_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of CongruenceSPD Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    G = random_SPD(n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator)

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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)

    assert is_symmetric(G_manual.grad)
    assert is_symmetric(G_auto.grad)
    assert_close(G_manual.grad, G_auto.grad)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_Whitening(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of whitening function and forward of Whitening Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    G = random_SPD(n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator)

    Y = spd_linalg.Whitening.apply(X, G)
    Z = spd_linalg.whitening(X, G)

    X_class = spd_linalg.Whitening.apply(
        Y, torch.cholesky_inverse(torch.linalg.cholesky(G))
    )
    X_function = spd_linalg.whitening(
        Z, torch.cholesky_inverse(torch.linalg.cholesky(G))
    )

    assert Y.shape == X.shape
    assert Z.shape == X.shape
    assert is_spd(Y)
    assert is_spd(Z)
    assert_close(Y, Z)
    assert_close(X_class, X)
    assert_close(X_function, X)


@pytest.mark.parametrize("n_matrices, n_features, cond", [(1,100,1000), (50,100,1000)])
def test_Whitening_backward(n_matrices, n_features, cond, device, dtype, generator):
    """
    Test of backward of Whitening Function class
    """
    X = random_SPD(n_features, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    G = random_SPD(n_features, 1, cond=cond, device=device, dtype=dtype, generator=generator)

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

    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)

    assert is_symmetric(G_manual.grad)
    assert is_symmetric(G_auto.grad)
    assert_close(G_manual.grad, G_auto.grad)


@pytest.mark.parametrize("n_matrices, n_in, n_out, cond", [(1, 100, 70, 1000), (50, 100, 70, 1000)])
def test_CongruenceRectangular(n_matrices, n_in, n_out, cond, device, dtype, generator):
    """
    Test of congruence_rectangular function and forward of CongruenceSPD Function class
    """
    # Generate random SPD matrices
    X = random_SPD(n_in, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    # Generate random Stiefel matrix
    W = torch.empty((n_out, n_in), device=device, dtype=dtype)
    _init_weights_stiefel(W, generator=generator)

    Y = spd_linalg.CongruenceRectangular.apply(X, W)
    Z = spd_linalg.CongruenceRectangular.apply(X, W)

    assert Y.dim() == X.dim()
    if n_matrices > 1 :
        assert Y.shape[0] == X.shape[0]
    assert Y.shape[-2] == n_out
    assert Y.shape[-1] == n_out
    assert is_spd(Y)

    assert Z.dim() == X.dim()
    if n_matrices > 1 :
        assert Z.shape[0] == X.shape[0]
    assert Z.shape[-2] == n_out
    assert Z.shape[-1] == n_out
    assert is_spd(Z)


@pytest.mark.parametrize("n_matrices, n_in, n_out, cond", [(1, 100, 70, 1000), (50, 100, 70, 1000)])
def test_CongruenceRectangular_backward(n_matrices, n_in, n_out, cond, device, dtype, generator):
    """
    Test of backward of CongruenceRectangular Function class
    """
    # Generate random SPD matrices
    X = random_SPD(n_in, n_matrices, cond=cond, device=device, dtype=dtype, generator=generator)
    # Generate random Stiefel matrix
    W = torch.empty((n_out, n_in), device=device, dtype=dtype)
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

    assert X_manual.grad.shape == X_manual.shape
    assert is_symmetric(X_manual.grad)
    assert is_symmetric(X_auto.grad)
    assert_close(X_manual.grad, X_auto.grad)

    assert W_manual.grad.shape == W_manual.shape
    assert_close(W_manual.grad, W_auto.grad)


