from unittest import TestCase

import torch
from torch.testing import assert_close

from scipy.linalg import (
    solve_sylvester,
    sqrtm,
    logm,
    expm
)

import yetanotherspdnet.spd as spd
from yetanotherspdnet.stiefel import _init_weights_stiefel

seed = 777
torch.manual_seed(seed)

# "Fun" fact: float64 seems really important for most functions...
# float32 not that close result, seems true with scipy
# as well.....
torch.set_default_dtype(torch.float64)


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
    is_square =  bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    return bool(torch.allclose(X, X.transpose(-1,-2)))


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
    is_square =  bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    is_symmetric = bool(torch.allclose(X, X.transpose(-1,-2)))
    is_positivedefinite = bool((torch.linalg.eigvalsh(X)>0).all())
    return is_symmetric and is_positivedefinite


# ----------
# Symmetrize
# ----------
class TestSymmetrize(TestCase):
    """
    Test of symmetrize function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # random batch of matrices
        self.X = torch.randn(
            (self.n_matrices, self.n_features, self.n_features)
        )
    
    def test_symmetrize(self):
        X_sym = spd.symmetrize(self.X)
        X_skew = self.X - X_sym
        assert_close(X_sym, X_sym.transpose(-1,-2))
        assert_close(X_skew, -X_skew.transpose(-1,-2))
        assert_close(self.X, X_sym + X_skew)
    
# -------------
# Vectorization
# -------------
class TestVec(TestCase):
    """
    Test of vec_batch and unvec_batch functions
    """

    def setUp(self):
        self.n_rows = 50
        self.n_columns = 30
        self.n_matrices = 20

        # random batch of square matrices
        self.X_square = torch.randn(
            (self.n_matrices, self.n_rows, self.n_rows)
        )

        # random batch of rectangular matrices
        self.X_rectangular = torch.randn(
            (self.n_matrices, self.n_rows, self.n_columns)
        )

        # random batch of n_rows*n_rows vectors
        self.X_square_vec = torch.randn(
            (self.n_matrices, self.n_rows * self.n_rows)
        )

        # random batch of n_rows*n_columns vectors
        self.X_rectangular_vec = torch.randn(
            (self.n_matrices, self.n_rows * self.n_columns)
        )
    
    def test_vec_square(self):
        X_vec = spd.vec_batch(self.X_square)
        assert X_vec.dim() == self.X_square.dim() - 1
        assert X_vec.shape[0] == self.n_matrices
        assert X_vec.shape[1] == self.n_rows * self.n_rows
        assert_close(
            spd.unvec_batch(X_vec, self.n_rows), self.X_square
        )

    def test_unvec_square(self):
        X_unvec = spd.unvec_batch(self.X_square_vec, self.n_rows)
        assert X_unvec.dim() == self.X_square_vec.dim() + 1
        assert X_unvec.shape[0] == self.n_matrices
        assert X_unvec.shape[1] == self.n_rows
        assert X_unvec.shape[2] == self.n_rows
        assert_close(spd.vec_batch(X_unvec), self.X_square_vec)
    
    def test_vec_rectangular(self):
        X_vec = spd.vec_batch(self.X_rectangular)
        assert X_vec.dim() == self.X_rectangular.dim() - 1
        assert X_vec.shape[0] == self.n_matrices
        assert X_vec.shape[1] == self.n_rows * self.n_columns
        assert_close(
            spd.unvec_batch(X_vec, self.n_rows), self.X_rectangular
        )
    
    def test_unvec_rectangular(self):
        X_unvec = spd.unvec_batch(self.X_rectangular_vec, self.n_rows)
        assert X_unvec.dim() == self.X_rectangular_vec.dim() + 1
        assert X_unvec.shape[0] == self.n_matrices
        assert X_unvec.shape[1] == self.n_rows
        assert X_unvec.shape[2] == self.n_columns
        assert_close(spd.vec_batch(X_unvec), self.X_rectangular_vec)


class TestVech(TestCase):
    """
    Test of vech_batch and unvech_batch functions
    """

    def setUp(self):
        self.n_features = 50
        self.dim = self.n_features * (self.n_features+1) // 2
        self.n_matrices = 20

        # random batch of symmetric matrices
        self.X = spd.symmetrize(torch.randn(
            (self.n_matrices, self.n_features, self.n_features)
        ))

        # random batch of dim vectors
        self.X_vech = torch.randn(
            (self.n_matrices, self.dim)
        )

    def test_vech(self):
        X_vech = spd.vech_batch(self.X)
        assert X_vech.dim() == self.X.dim() - 1
        assert X_vech.shape[0] == self.n_matrices
        assert X_vech.shape[1] == self.dim
        assert_close(spd.unvech_batch(X_vech), self.X)
    
    def test_unvech(self):
        X_unvech = spd.unvech_batch(self.X_vech)
        assert X_unvech.dim() == self.X_vech.dim() + 1
        assert X_unvech.shape[0] == self.n_matrices
        assert X_unvech.shape[1] == self.n_features
        assert X_unvech.shape[2] == self.n_features
        assert_close(spd.vech_batch(X_unvech), self.X_vech)



#----------------------------------------------------------
# Remark: 
# No testing of eigh_operation and eigh_operation_grad, 
# it is done indirectly when testing sqrtm, inv_sqrtm, etc.
# ---------------------------------------------------------


# -------------------
# Solve Sylvester SPD
# -------------------
class TestSylvesterSPD(TestCase):
    """ 
    Test of solve_sylvester_SPD function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.A = spd.random_SPD(self.n_features, self.n_matrices, cond=100)

        # Generate random symmetric matrices
        self.Q = spd.symmetrize(torch.randn(
            (self.n_matrices, self.n_features, self.n_features)
        ))

    def test_sylvester_SPD_2D(self):
        A = self.A[0]
        Q = self.Q[0]
        eigvals, eigvecs = torch.linalg.eigh(A)
        X = spd.solve_sylvester_SPD(eigvals, eigvecs, Q)
        assert X.shape == A.shape
        assert is_symmetric(X)
        assert_close(A @ X + X @ A, Q)

    def test_sylvester_SPD(self):
        eigvals, eigvecs = torch.linalg.eigh(self.A)
        X = spd.solve_sylvester_SPD(eigvals, eigvecs, self.Q)
        assert X.shape == self.A.shape
        assert is_symmetric(X)
        assert_close(self.A @ X + X @ self.A, self.Q)
    
    def test_scipy_comparison(self):
        eigvals, eigvecs = torch.linalg.eigh(self.A)
        X = spd.solve_sylvester_SPD(eigvals, eigvecs, self.Q)
        X_scipy = torch.zeros(
            (self.n_matrices, self.n_features, self.n_features),
            dtype=self.A.dtype
        )
        for i in range(self.n_matrices):
            X_scipy[i] = torch.tensor(solve_sylvester(
                self.A[i].detach().numpy(), 
                self.A[i].detach().numpy(),
                self.Q[i].detach().numpy()
            ), dtype=self.A.dtype)
        # assert_close(self.A @ X_scipy + X_scipy @ self.A, self.Q)
        assert_close(X, X_scipy)



# ---------
# Sqrtm SPD
# ---------
class TestSqrtmSPD(TestCase):
    """
    Test of SqrtmSPD Function class and sqrtm_SPD function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_features, self.n_matrices, cond=100)
    
    def test_SqrtmSPD_2D(self):
        X = self.X[0]
        X_Sqrtm = spd.SqrtmSPD.apply(X)
        X_sqrtm = spd.sqrtm_SPD(X)
        assert X_Sqrtm.shape == X.shape
        assert is_spd(X_Sqrtm)
        assert_close(X_Sqrtm @ X_Sqrtm, X)
        assert X_sqrtm.shape == X.shape
        assert is_spd(X_sqrtm)
        assert_close(X_sqrtm @ X_sqrtm, X)
        assert_close(X_Sqrtm, X_sqrtm)

    def test_SqrtmSPD(self):
        X_Sqrtm = spd.SqrtmSPD.apply(self.X)
        X_sqrtm = spd.sqrtm_SPD(self.X)
        assert X_sqrtm.shape == self.X.shape
        assert is_spd(X_sqrtm)
        assert_close(X_sqrtm @ X_sqrtm, self.X)
        assert X_Sqrtm.shape == self.X.shape
        assert is_spd(X_Sqrtm)
        assert_close(X_Sqrtm @ X_Sqrtm, self.X)
        assert_close(X_Sqrtm, X_sqrtm)
    
    def test_scipy_comparison(self):
        X_Sqrtm = spd.SqrtmSPD.apply(self.X)
        X_sqrtm = spd.sqrtm_SPD(self.X)
        X_sqrtm_scipy = torch.zeros(
            (self.n_matrices, self.n_features, self.n_features),
            dtype=self.X.dtype
        )
        for i in range(self.n_matrices):
            X_sqrtm_scipy[i] = torch.tensor(sqrtm(
                self.X[i].detach().numpy(),
            ), dtype=self.X.dtype)
        assert_close(X_Sqrtm, X_sqrtm_scipy)
        assert_close(X_sqrtm, X_sqrtm_scipy)
    
    def test_SqrtmSPD_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        X_manual_sqrtm = spd.SqrtmSPD.apply(X_manual)
        X_auto_sqrtm = spd.sqrtm_SPD(X_auto)

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
class TestInvSqrtmSPD(TestCase):
    """
    Test of InvSqrtm Function class and inv_sqrtm_SPD function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_features, self.n_matrices, cond=100)

    def test_InvSqrtmSPD_2D(self):
        X = self.X[0]
        X_InvSqrtm = spd.InvSqrtmSPD.apply(X)
        X_inv_sqrtm = spd.inv_sqrtm_SPD(X)
        assert X_InvSqrtm.shape == X.shape
        assert is_spd(X_InvSqrtm)
        assert_close(
            X_InvSqrtm @ X_InvSqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(X))
        )
        assert X_inv_sqrtm.shape == X.shape
        assert is_spd(X_inv_sqrtm)
        assert_close(
            X_inv_sqrtm @ X_inv_sqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(X))
        )
        assert_close(X_InvSqrtm, X_inv_sqrtm)
    
    def test_InvSqrtmSPD(self):
        X_InvSqrtm = spd.InvSqrtmSPD.apply(self.X)
        X_inv_sqrtm = spd.inv_sqrtm_SPD(self.X)
        assert X_inv_sqrtm.shape == self.X.shape
        assert is_spd(X_inv_sqrtm)
        assert_close(
            X_inv_sqrtm @ X_inv_sqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(self.X))
        )
        assert X_InvSqrtm.shape == self.X.shape
        assert is_spd(X_InvSqrtm)
        assert_close(
            X_InvSqrtm @ X_InvSqrtm,
            torch.cholesky_inverse(torch.linalg.cholesky(self.X))
        )
        assert_close(X_InvSqrtm, X_inv_sqrtm)
    
    def test_InvSqrtmSPD_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        X_manual_inv_sqrtm = spd.InvSqrtmSPD.apply(X_manual)
        X_auto_inv_sqrtm = spd.inv_sqrtm_SPD(X_auto)

        loss_manual = torch.norm(X_manual_inv_sqrtm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_inv_sqrtm)
        loss_auto.backward()

        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)


# ---------------------------
# Logm SPD and Expm Symmetric
# ---------------------------
class TestLogmSPDExpmSymmetric(TestCase):
    """
    Test of LogmSPD Function class, logm_SPD function,
    ExpmSymmetric Function class and expm_symmetric function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_features, self.n_matrices, cond=100)

        # Generate random symmetric matrices
        self.X_symmetric = spd.symmetrize(torch.randn(
            (self.n_matrices, self.n_features, self.n_features)
        ))

    
    def test_LogmSPD(self):
        X_Logm = spd.LogmSPD.apply(self.X)
        X_logm = spd.logm_SPD(self.X)

        assert X_Logm.shape == self.X.shape
        assert is_symmetric(X_Logm)
        assert_close(spd.expm_symmetric(X_Logm), self.X)

        assert X_logm.shape == self.X.shape
        assert is_symmetric(X_logm)
        assert_close(spd.expm_symmetric(X_logm), self.X)
        assert_close(X_Logm, X_logm)
    
    def test_ExpmSymmetric(self):
        X_Expm = spd.ExpmSymmetric.apply(self.X_symmetric)
        X_expm = spd.expm_symmetric(self.X_symmetric)

        assert X_Expm.shape == self.X_symmetric.shape
        assert is_spd(X_Expm)
        assert_close(spd.logm_SPD(X_Expm), self.X_symmetric)

        assert X_expm.shape == self.X_symmetric.shape
        assert is_spd(X_expm)
        assert_close(spd.logm_SPD(X_expm), self.X_symmetric)
        assert_close(X_Expm, X_expm)
    
    def test_logm_scipy_comparison(self):
        X_Logm = spd.LogmSPD.apply(self.X)
        X_logm = spd.logm_SPD(self.X)
        X_logm_scipy = torch.zeros(
            (self.n_matrices, self.n_features, self.n_features),
            dtype=self.X.dtype
        )
        for i in range(self.n_matrices):
            X_logm_scipy[i] = torch.tensor(logm(
                self.X[i].detach().numpy(),
            ), dtype=self.X.dtype)
        assert_close(X_Logm, X_logm_scipy)
        assert_close(X_logm, X_logm_scipy)

    def test_expm_scipy_comparison(self):
        X_Expm = spd.ExpmSymmetric.apply(self.X_symmetric)
        X_expm = spd.expm_symmetric(self.X_symmetric)
        X_expm_scipy = torch.zeros(
            (self.n_matrices, self.n_features, self.n_features),
            dtype=self.X_symmetric.dtype
        )
        for i in range(self.n_matrices):
            X_expm_scipy[i] = torch.tensor(expm(
                self.X_symmetric[i].detach().numpy(),
            ), dtype=self.X_symmetric.dtype)
        assert_close(X_Expm, X_expm_scipy)
        assert_close(X_expm, X_expm_scipy)
    
    def test_logm_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        X_manual_logm = spd.LogmSPD.apply(X_manual)
        X_auto_logm = spd.logm_SPD(X_auto)

        loss_manual = torch.norm(X_manual_logm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_logm)
        loss_auto.backward()

        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)
    
    def test_expm_backward(self):
        X_manual = self.X_symmetric.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X_symmetric.clone().detach()
        X_auto.requires_grad = True

        X_manual_expm = spd.ExpmSymmetric.apply(X_manual)
        X_auto_expm = spd.expm_symmetric(X_auto)

        loss_manual = torch.norm(X_manual_expm)
        loss_manual.backward()
        loss_auto = torch.norm(X_auto_expm)
        loss_auto.backward()

        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        assert_close(X_manual.grad, X_auto.grad)
    

# -----------
# Congruences
# -----------
class TestCongruenceSPD(TestCase):
    """
    Test CongruenceSPD Function class
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_features, self.n_matrices, cond=100)

        # Generate random SPD matrix
        self.G = spd.random_SPD(self.n_features, 1, cond=100)

    def test_congruenceSPD(self):
        Y = spd.CongruenceSPD.apply(self.X, self.G)
        Z = spd.congruence_SPD(self.X, self.G)

        X_class = spd.CongruenceSPD.apply(
            Y, torch.cholesky_inverse(torch.linalg.cholesky(self.G))
        )
        X_function = spd.congruence_SPD(
            Z, torch.cholesky_inverse(torch.linalg.cholesky(self.G))
        )

        assert Y.shape == self.X.shape
        assert Z.shape == self.X.shape
        assert is_spd(Y)
        assert is_spd(Z)
        assert_close(Y, Z)
        assert_close(X_class, self.X)
        assert_close(X_function, self.X)
    
    def test_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        G_manual = self.G.clone().detach()
        G_manual.requires_grad = True
        G_auto = self.G.clone().detach()
        G_auto.requires_grad = True

        X_manual_cong = spd.CongruenceSPD.apply(X_manual, G_manual)
        X_auto_cong = spd.congruence_SPD(X_auto, G_auto)

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


class TestWhitening(TestCase):
    """
    Test of Whitening Function class and whitening function
    """

    def setUp(self):
        self.n_features = 50
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_features, self.n_matrices, cond=100)

        # Generate random SPD matrix
        self.G = spd.random_SPD(self.n_features, 1, cond=100)
    
    def test_Whitening(self):
        Y = spd.Whitening.apply(self.X, self.G)
        Z = spd.whitening(self.X, self.G)

        X_class = spd.Whitening.apply(
            Y, torch.cholesky_inverse(torch.linalg.cholesky(self.G))
        )
        X_function = spd.whitening(
            Z, torch.cholesky_inverse(torch.linalg.cholesky(self.G))
        )

        assert Y.shape == self.X.shape
        assert Z.shape == self.X.shape
        assert is_spd(Y)
        assert is_spd(Z)
        assert_close(Y, Z)
        assert_close(X_class, self.X)
        assert_close(X_function, self.X)

    def test_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        G_manual = self.G.clone().detach()
        G_manual.requires_grad = True
        G_auto = self.G.clone().detach()
        G_auto.requires_grad = True

        X_manual_white = spd.Whitening.apply(X_manual, G_manual)
        X_auto_white = spd.whitening(X_auto, G_auto)

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


class TestCongruenceRectangular(TestCase):
    """
    Test of CongruenceRectangular Function class and
    congruence_rectangular function
    """

    def setUp(self):
        self.n_in = 50
        self.n_out = 30
        self.n_matrices = 20

        # Generate random SPD matrices
        self.X = spd.random_SPD(self.n_in, self.n_matrices, cond=100)

        # Generate random Stiefel matrix
        self.W = torch.empty((self.n_out, self.n_in))
        _init_weights_stiefel(self.W)
    
    def test_CongruenceRectangular(self):
        Y = spd.CongruenceRectangular.apply(self.X, self.W)
        Z = spd.CongruenceRectangular.apply(self.X, self.W)

        assert Y.dim() == self.X.dim()
        assert Y.shape[0] == self.X.shape[0]
        assert Y.shape[1] == self.n_out
        assert Y.shape[2] == self.n_out
        assert is_spd(Y)

        assert Z.dim() == self.X.dim()
        assert Z.shape[0] == self.X.shape[0]
        assert Z.shape[1] == self.n_out
        assert Z.shape[2] == self.n_out
        assert is_spd(Z)

    def test_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        W_manual = self.W.clone().detach()
        W_manual.requires_grad = True
        W_auto = self.W.clone().detach()
        W_auto.requires_grad = True

        X_manual_cong = spd.CongruenceRectangular.apply(X_manual, W_manual)
        X_auto_cong = spd.congruence_rectangular(X_auto, W_auto)

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


class TestEighReLu(TestCase):
    """
    Test of EighReLu Function class and eigh_relu function
    """

    def setUp(self):
        self.n_features = 50
        self.n_small_eigvals = 10
        self.n_matrices = 20

        # Generate random SPD matrices with some small enough eigenvalues
        U, _, _ = torch.svd(torch.randn(
            (self.n_matrices, self.n_features, self.n_features)
        ))

        l = torch.randn((self.n_matrices, self.n_features))**2
        l[:,:self.n_small_eigvals] = 1e-6

        self.X = (U*l.unsqueeze(-2))@U.transpose(-1,-2)

        self.eps = 1e-4
        # to account for numerical errors in eigvalsh
        self.tol = self.eps/10
    
    def test_EighReLu(self):
        Y = spd.EighReLu.apply(self.X, self.eps)
        Z = spd.eigh_relu(self.X, self.eps)

        assert Y.shape == self.X.shape
        assert is_spd(Y)
        assert bool((torch.linalg.eigvalsh(Y) >= self.eps-self.tol).all())

        assert Z.shape == self.X.shape
        assert is_spd(Z)
        assert bool((torch.linalg.eigvalsh(Z) >= self.eps-self.tol).all())

    def test_backward(self):
        X_manual = self.X.clone().detach()
        X_manual.requires_grad = True
        X_auto = self.X.clone().detach()
        X_auto.requires_grad = True

        Y_manual = spd.EighReLu.apply(X_manual, self.eps)
        Y_auto = spd.eigh_relu(X_auto, self.eps)

        loss_manual = torch.norm(Y_manual)
        loss_manual.backward()
        loss_auto = torch.norm(Y_auto)
        loss_auto.backward()

        assert X_manual.grad.shape == X_manual.shape
        assert is_symmetric(X_manual.grad)
        assert is_symmetric(X_auto.grad)
        # need to lower tolerances to get True
        assert_close(X_manual.grad, X_auto.grad, atol=1e-5, rtol=1e-6)