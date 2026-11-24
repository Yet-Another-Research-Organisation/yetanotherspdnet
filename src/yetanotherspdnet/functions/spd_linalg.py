"""Core SPD linear algebra: matrix functions, congruences and vectorization."""

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from typing import overload

import torch
from torch.autograd import Function

from yetanotherspdnet._deprecation import _deprecated
from yetanotherspdnet.functions._constants import (
    EIGVAL_TIE_GENERIC_FACTOR,
    EIGVAL_TIE_RTOL,
    SPD_EIGVAL_FLOOR,
    SPD_FLOOR_RTOL,
)
from yetanotherspdnet.functions.scalar_functions import (
    exp_divided_difference,
    inv,
    inv_derivative,
    inv_divided_difference,
    inv_scaled_softplus,
    inv_scaled_softplus_derivative,
    inv_sqrt,
    inv_sqrt_derivative,
    inv_sqrt_divided_difference,
    log_divided_difference,
    pow_divided_difference,
    pow_exponent_derivative,
    scaled_softplus,
    scaled_softplus_derivative,
    sqrt_derivative,
    sqrt_divided_difference,
)


def symmetrize(data: torch.Tensor) -> torch.Tensor:
    r"""
    Symmetrize a tensor along the last two dimensions.

    .. math:: \operatorname{sym}(A) = \frac{1}{2}\big(A + A^\top\big)

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of square matrices

    Returns
    -------
    sym_data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetrized matrices
    """
    return 0.5 * (data + data.transpose(-1, -2))


def skew_symmetrize(data: torch.Tensor) -> torch.Tensor:
    """
    Skew-symmetrize a tensor along its two last dimensions.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of square matrices

    Returns
    -------
    skew_data : torch.Tensor of shape (..., n_features, n_features)
        Batch of skew-symmetrized matrices
    """
    return 0.5 * (data - data.transpose(-1, -2))


# -----------------------
# Vectorization operators
# -----------------------
def vec_batch(data: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a batch of tensors along last two dimensions.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_rows, n_columns)
        Batch of matrices

    Returns
    -------
    data_vec : torch.Tensor of shape (..., n_rows*n_columns)
        Batch of vectorized matrices
    """
    return data.reshape(*data.shape[:-2], -1)


def unvec_batch(data_vec: torch.Tensor, n_rows: int) -> torch.Tensor:
    """
    Unvectorize a batch of tensors along last dimension.

    Parameters
    ----------
    data_vec : torch.Tensor of shape (..., n_rows*n_columns)
        Batch of vectorized matrices

    n_rows : int
        Number of rows of the matrices

    Returns
    -------
    data : torch.Tensor of shape (..., n_rows, n_columns)
        Batch of matrices
    """
    return data_vec.reshape(*data_vec.shape[:-1], n_rows, -1)


class VecBatch(Function):
    """
    Vectorize a batch of matrices along their last two dimensions.

    Matrices are assumed to be symmetric (for the backward).
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vectorization of a batch of matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        vec_data : torch.Tensor of shape (..., n_features ** 2)
        """
        ctx.n_rows = data.shape[-2]
        return vec_batch(data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the vectorization of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features ** 2)
            Gradient of the loss with respect to vectorized input batch of matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of matrices
        """
        return symmetrize(unvec_batch(grad_output, ctx.n_rows))


def vech_batch(data: torch.Tensor) -> torch.Tensor:
    """
    Vectorize the lower triangular part of a batch of square matrices.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrices

    Returns
    -------
    data_vech : torch.Tensor of shape (..., n_features*(n_features+1)//2)
        Batch of vectorized matrices
    """
    indices = torch.tril_indices(*data.shape[-2:])
    return data[..., indices[0], indices[1]]


def unvech_batch(data_vech: torch.Tensor, n_features: int) -> torch.Tensor:
    """
    Unvectorize a batch of tensors along their last dimension.

    Matrices are assumed to be symmetric.

    Parameters
    ----------
    data_vech : torch.Tensor of shape (..., n_features*(n_features+1)//2)
        Batch of vectorized matrices

    n_features : int
        number of features

    Returns
    -------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices
    """
    indices_l = torch.tril_indices(n_features, n_features)
    indices_u = torch.triu_indices(n_features, n_features)
    data = torch.zeros(
        *data_vech.shape[:-1],
        n_features,
        n_features,
        dtype=data_vech.dtype,
        device=data_vech.device,
    )
    # fill lower triangular
    data[..., indices_l[0], indices_l[1]] = data_vech
    # fill upper triangular
    data[..., indices_u[0], indices_u[1]] = data.transpose(-1, -2)[
        ..., indices_u[0], indices_u[1]
    ]
    data = symmetrize(data)
    return data


class VechBatch(Function):
    """Half vectorize a batch of symmetric matrices along last two dimensions."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the half vectorization of a batch of matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        vech_data : torch.Tensor of shape (..., n_features*(n_features+1) // 2)
        """
        ctx.n_features = data.shape[-1]
        return vech_batch(data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the vectorization of a batch of matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features*(n_features+1) // 2)
            Gradient of the loss with respect to half vectorized input batch of symmetric matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of symmetric matrices
        """
        return unvech_batch(grad_output, ctx.n_features)


# -----------------------------------------
# Eigendecompositions (optional upcasting)
# -----------------------------------------
_EIGH_DTYPE: contextvars.ContextVar[torch.dtype | None] = contextvars.ContextVar(
    "eigh_dtype", default=None
)


def get_eigh_dtype() -> torch.dtype | None:
    """
    Return the dtype of the eigendecompositions set by :func:`eigh_precision`.

    Returns
    -------
    torch.dtype or None
        The decomposition dtype, None for the dtype of the input
    """
    return _EIGH_DTYPE.get()


@contextlib.contextmanager
def eigh_precision(dtype: torch.dtype | None) -> Iterator[None]:
    r"""
    Compute the eigendecompositions of the library in *dtype* (mixed precision).

    Inside the context, every symmetric eigendecomposition of the library
    (:func:`symmetric_eigh`, :func:`symmetric_eigvalsh`) of a tensor of another
    floating dtype is computed on the input cast to *dtype*, and its
    eigenvalues and eigenvectors are cast back: with ``dtype=torch.float64`` a
    float32 model gets the eigenvalues of its float32 matrices to float64
    accuracy (float32 ``eigh`` resolves the eigenvalues only to about
    :math:`\varepsilon_{32} \lVert X \rVert`, so the small eigenvalues of
    ill-conditioned matrices lose all their digits, or turn negative), while
    everything else stays in float32. The layers and models expose it as their
    ``eigh_dtype`` option. The setting is read when a decomposition is computed;
    the manual backwards reuse the decompositions of their forward.

    Parameters
    ----------
    dtype : torch.dtype or None
        dtype of the decompositions (None: the dtype of each input)

    Yields
    ------
    None
    """
    token = _EIGH_DTYPE.set(dtype)
    try:
        yield
    finally:
        _EIGH_DTYPE.reset(token)


def eigh_context(
    dtype: torch.dtype | None,
) -> contextlib.AbstractContextManager[None]:
    """
    Context of :func:`eigh_precision` for *dtype*, or a no-op for None.

    Unlike ``eigh_precision(None)``, None keeps an enclosing setting (a layer
    without ``eigh_dtype`` inside a model with one).

    Parameters
    ----------
    dtype : torch.dtype or None
        dtype of the decompositions

    Returns
    -------
    contextlib.AbstractContextManager
        The context manager
    """
    if dtype is None:
        return contextlib.nullcontext()
    return eigh_precision(dtype)


def symmetric_eigh(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition of symmetric matrices, in the precision of :func:`eigh_precision`.

    ``torch.linalg.eigh`` (ascending eigenvalues), computed in the dtype set by
    :func:`eigh_precision` when there is one and cast back to the dtype of
    *data*.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues
    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors
    """
    dtype = _EIGH_DTYPE.get()
    if dtype is None or dtype == data.dtype:
        return torch.linalg.eigh(data)
    eigvals, eigvecs = torch.linalg.eigh(data.to(dtype))
    return eigvals.to(data.dtype), eigvecs.to(data.dtype)


def symmetric_eigvalsh(data: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalues of symmetric matrices, in the precision of :func:`eigh_precision`.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues, ascending
    """
    dtype = _EIGH_DTYPE.get()
    if dtype is None or dtype == data.dtype:
        return torch.linalg.eigvalsh(data)
    return torch.linalg.eigvalsh(data.to(dtype)).to(data.dtype)


# -------------------------
# Operations on eigenvalues
# -------------------------
def eigh_operation(
    eigvals: torch.Tensor, eigvecs: torch.Tensor, operation: Callable
) -> torch.Tensor:
    r"""
    Apply a function on the eigenvalues of a batch of symmetric matrices.

    .. math::

        f(A) = V \operatorname{diag}\big(f(\lambda_1), \dots, f(\lambda_n)\big) V^\top

    given the eigendecomposition :math:`A = V \operatorname{diag}(\lambda) V^\top`.
    This is the core primitive behind every matrix function in this module
    (:func:`sqrtm_SPD`, :func:`inv_sqrtm_SPD`, :func:`powm_SPD`,
    :func:`logm_SPD`, :func:`expm_symmetric`, …): each just picks a
    different scalar ``operation`` applied eigenvalue-wise.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the corresponding batch of symmetric matrices

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the corresponding batch of symmetric matrices

    operation : Callable
        Function to apply on eigenvalues

    Returns
    -------
    result : torch.Tensor of shape (..., n_features, n_features)
        Resulting symmetric matrices with operation applied to eigenvalues
    """
    _eigvals = operation(eigvals)
    result = (eigvecs * _eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return result


# closed-form divided differences of the matrix functions of this module,
# looked up by the scalar ``operation`` when no ``divided_difference`` is given
_DIVIDED_DIFFERENCES: dict[Callable, Callable] = {
    torch.sqrt: sqrt_divided_difference,
    inv_sqrt: inv_sqrt_divided_difference,
    inv: inv_divided_difference,
    torch.log: log_divided_difference,
    torch.exp: exp_divided_difference,
}


def _aux_eigh_operation_grad(
    eigvals: torch.Tensor,
    operation: Callable,
    operation_deriv: Callable,
    divided_difference: Callable | None = None,
) -> torch.Tensor:
    r"""
    Build the Daleckii-Krein (Loewner) matrix of a function applied on eigenvalues.

    It is multiplied (Hadamard product) with the transformed output gradient to get the
    input gradient of a function applied on the eigenvalues of a symmetric matrix:
    :math:`L_{ij} = (f(\lambda_i) - f(\lambda_j)) / (\lambda_i - \lambda_j)`, and
    :math:`f'` at ties.

    Two regimes:

    - a closed-form, non-cancelling divided difference is known (given as
      ``divided_difference``, or looked up for ``torch.sqrt``, ``inv_sqrt``, ``inv``,
      ``torch.log``, ``torch.exp``): it is used for every pair, and the derivative
      only replaces it at rounding-level ties,
      :math:`|\lambda_i - \lambda_j| \le c\, n\, \varepsilon \max_k |\lambda_k|`
      (:math:`c` = ``EIGVAL_TIE_RTOL``, :math:`\varepsilon` the machine epsilon of
      the dtype);
    - otherwise the quotient is used down to the gap
      :math:`(24\varepsilon)^{1/3} \max(|\lambda_i|, |\lambda_j|)` where its
      cancellation error equals the error of the derivative at the midpoint, and
      :math:`f'((\lambda_i + \lambda_j) / 2)` below it.

    In both cases the near-tie value is the derivative at the midpoint, so the
    matrix is exactly symmetric.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices

    operation : Callable
        Function to apply on eigenvalues

    operation_deriv : Callable
        Derivative of the function to apply on eigenvalues

    divided_difference : Callable, optional
        ``divided_difference(a, b)`` returning :math:`(f(a) - f(b)) / (a - b)`
        without cancellation (see the kernels of
        :mod:`~yetanotherspdnet.functions.scalar_functions`). Default: looked up
        from ``operation``, else the generic quotient.

    Returns
    -------
    aux_mat : torch.Tensor of shape (..., n_features, n_features)
        Matrix to be multiplied (Hadamard product) with transformed output gradient
    """
    if divided_difference is None:
        divided_difference = _DIVIDED_DIFFERENCES.get(operation)
    finfo = torch.finfo(eigvals.dtype)
    lam_i = eigvals.unsqueeze(-1)
    lam_j = eigvals.unsqueeze(-2)
    gap = torch.abs(lam_i - lam_j)
    midpoint_deriv = operation_deriv((lam_i + lam_j) / 2)

    if divided_difference is not None:
        n_features = eigvals.shape[-1]
        scale = eigvals.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        tie_tol = EIGVAL_TIE_RTOL * n_features * finfo.eps * scale.clamp_min(finfo.tiny)
        tie = gap <= tie_tol
        return torch.where(tie, midpoint_deriv, divided_difference(lam_i, lam_j))

    pair_scale = torch.maximum(lam_i.abs(), lam_j.abs()).clamp_min(finfo.tiny)
    tie = gap <= (EIGVAL_TIE_GENERIC_FACTOR * finfo.eps) ** (1 / 3) * pair_scale
    transformed = operation(eigvals)
    numerator = transformed.unsqueeze(-1) - transformed.unsqueeze(-2)
    denominator = torch.where(tie, torch.ones_like(gap), lam_i - lam_j)
    return torch.where(tie, midpoint_deriv, numerator / denominator)


def eigh_operation_grad(
    grad_output: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    operation: Callable,
    operation_deriv: Callable,
    divided_difference: Callable | None = None,
) -> torch.Tensor:
    """
    Backpropagate through a function applied on the eigenvalues.

    Gradient of a function applied on the eigenvalues of a batch of symmetric matrices.

    Parameters
    ----------
    grad_output : torch.Tensor of shape (..., n_features, n_features)
        Gradient of the loss with respect to the output of the operation on eigenvalues

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the corresponding batch of symmetric matrices

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the corresponding batch of symmetric matrices

    operation : Callable
        Function to apply on eigenvalues

    operation_deriv : Callable
        Derivative of the function to apply on eigenvalues

    divided_difference : Callable, optional
        Closed-form divided difference of the function, see
        :func:`_aux_eigh_operation_grad`

    Returns
    -------
    grad_input : torch.Tensor of shape (..., n_features, n_features)
        Gradient of the loss with respect to the input batch of symmetric matrices
    """
    aux_mat = _aux_eigh_operation_grad(
        eigvals, operation, operation_deriv, divided_difference
    )
    middle_term = aux_mat * (
        eigvecs.transpose(-1, -2) @ symmetrize(grad_output) @ eigvecs
    )
    return eigvecs @ middle_term @ eigvecs.transpose(-1, -2)


# ------------------
# Sylvester equation
# ------------------
def solve_sylvester_SPD(
    eigvals: torch.Tensor, eigvecs: torch.Tensor, mat: torch.Tensor
) -> torch.Tensor:
    r"""
    Solve Sylvester equations AX + XA = mat for SPD matrices A.

    Relies on the eigendecomposition of A.

    Given :math:`A = V \operatorname{diag}(\lambda) V^\top` (via
    ``eigvals``, ``eigvecs``), solves :math:`AX + XA = \text{mat}` for
    :math:`X` in closed form:

    .. math::

        X = V\left[\frac{1}{\lambda_i + \lambda_j}
            \,(V^\top \,\text{mat}\, V)_{ij}\right]_{ij} V^\top

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of a batch of SPD matrices
    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of a batch of SPD matrices
    mat : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrices on the right side of Sylvester equations.
        If matrices are symmetric then the solutions will be symmetric.
        If they are skew-symmetric, then the results will be skew-symmetric.

    Returns
    -------
    result : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices solutions to Sylvester equations
    """
    K = 1 / (eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2))
    middle_term = K * (eigvecs.transpose(-1, -2) @ mat @ eigvecs)
    return eigvecs @ middle_term @ eigvecs.transpose(-1, -2)


# ------------------------------------
# Generalized eigenvalue decomposition
# ------------------------------------
def gevd(
    point1: torch.Tensor, point2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute the generalized eigendecomposition of (point1, point2).

    point1 is an SPD matrix and point2 a symmetric matrix.

    We have :math: `point1 = general_eigvecs^\top general_eigvecs, point2 = general_eigvecs^\top general_eigvals general_eigvecs`.

    Parameters
    ----------
    point1 : torch.Tensor of shape (n_features, n_features) or (..., n_features, n_features)
        SPD matrix (or batch of same shape as Q)

    point2 : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (..., n_features, n_features)
        Inverses of general_eigvecs
    """
    L = torch.linalg.cholesky(point1)
    L_inv = torch.inverse(L)
    general_eigvals, U = symmetric_eigh(L_inv @ point2 @ L_inv.transpose(-2, -1))
    return general_eigvals, (L @ U).transpose(-2, -1), L_inv.transpose(-2, -1) @ U


def gevd_operation(
    general_eigvals: torch.Tensor, general_eigvecs: torch.Tensor, operation: Callable
) -> torch.Tensor:
    r"""
    Apply a function on generalized eigenvalues.

    Generalized eigenvalue decomposition of a symmetric matrix point2 and an SPD matrix
    point1.

    We have :math: `point1 = general_eigvecs^\top general_eigvecs, point2 = general_eigvecs^\top general_eigvals general_eigvecs`.

    It returns :math: `general_eigvecs^\top f(general_eigvals) general_eigvecs`.

    Parameters
    ----------
    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    operation : Callable
        Function to apply on generalized eigenvalues

    Returns
    -------
    result : torch.Tensor of shape (..., n_features, n_features)
        Batch of resulting symmetric matrices with operation applied to generalized eigenvalues
    """
    _Lambda = operation(general_eigvals)
    result = (
        general_eigvecs.transpose(-1, -2) * _Lambda.unsqueeze(-2)
    ) @ general_eigvecs
    return result


def gevd_operation_grad(
    grad_output: torch.Tensor,
    general_eigvals: torch.Tensor,
    general_eigvecs: torch.Tensor,
    general_eigvecs_inv: torch.Tensor,
    operation: Callable,
    operation_derivative: Callable,
    divided_difference: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Backpropagate through a function applied on generalized eigenvalues.

    Generalized eigenvalue decomposition of a symmetric matrix point2 and an SPD matrix
    point1.

    We have :math: `point1 = general_eigvecs^\top general_eigvecs, point2 = general_eigvecs^\top general_eigvals general_eigvecs`.

    Parameters
    ----------
    grad_output : torch.Tensor of shape (..., n_features, n_features)
        Gradient of the loss with respect to the output of the operation on the batch of generalized eigenvalues

    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (..., n_features, n_features)
        Inverses of general_eigvecs

    operation : Callable
        Function to apply on generalized eigenvalues

    operation_derivative : Callable
        Derivative of the function to apply on generalized eigenvalues

    divided_difference : Callable, optional
        Closed-form divided difference of the function, see
        :func:`_aux_eigh_operation_grad`

    Returns
    -------
    grad_point1 : torch.Tensor of shape (n_features, n_features) or (..., n_features, n_features)
        Gradient w.r.t. SPD matrix (or batch) point1

    grad_point2 : torch.Tensor of shape (..., n_features, n_features)
        Gradient w.r.t. batch of symmetric matrices point2
    """
    grad_out_cong = (
        general_eigvecs @ symmetrize(grad_output) @ general_eigvecs.transpose(-1, -2)
    )
    aux_mat_point2 = _aux_eigh_operation_grad(
        general_eigvals, operation, operation_derivative, divided_difference
    )
    middle_term_point2 = aux_mat_point2 * grad_out_cong
    grad_point2 = (
        general_eigvecs_inv @ middle_term_point2 @ general_eigvecs_inv.transpose(-1, -2)
    )
    _Lambda = operation(general_eigvals)
    middle_term_point1 = symmetrize(
        grad_out_cong * _Lambda.unsqueeze(-2)
        - middle_term_point2 * general_eigvals.unsqueeze(-2)
    )
    grad_point1 = (
        general_eigvecs_inv @ middle_term_point1 @ general_eigvecs_inv.transpose(-1, -2)
    )
    return grad_point1, grad_point2


# ------------------
# SPD matrix inverse
# ------------------
def inv_SPD(data: torch.Tensor) -> torch.Tensor:
    """
    Inverse of a batch of SPD matrices, from their Cholesky factors.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    inv_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix inverses of the input batch of SPD matrices
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(data))


def _invm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix inverse of a batch of SPD matrices, from their eigendecomposition.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    invm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix inverses of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, inv), eigvals, eigvecs


class _InvmSPD(Function):
    """Matrix inverse of a batch of SPD matrices (eigendecomposition)."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix inverse of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        invm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix inverses of the input batch of SPD matrices
        """
        invm_data, eigvals, eigvecs = _invm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return invm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix inverse of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the matrix inverses

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(grad_output, eigvals, eigvecs, inv, inv_derivative)


# eigendecomposition-based inverse, unused by the library: kept with its
# (inverse, eigvals, eigvecs) outputs for backward compatibility
invm_SPD = _deprecated(
    "invm_SPD",
    _invm_SPD,
    note="inv_SPD returns the inverse only (Cholesky-based).",
    replacement="yetanotherspdnet.functions.spd_linalg.inv_SPD",
)
InvmSPD = _deprecated(
    "InvmSPD",
    _InvmSPD,
    note="inv_SPD is differentiated by autograd through the Cholesky factor.",
    replacement="yetanotherspdnet.functions.spd_linalg.inv_SPD",
)


# ----------------------
# SPD matrix square root
# ----------------------
def sqrtm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Matrix square root of a batch of SPD matrices.

    .. math:: P^{1/2} = V \operatorname{diag}(\sqrt{\lambda}) V^\top

    via :func:`eigh_operation` with ``operation=torch.sqrt``.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix square roots of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.sqrt), eigvals, eigvecs


class SqrtmSPD(Function):
    """Matrix square root of a batch of SPD matrices (relies on eigenvalue decomposition)."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix square root of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix square roots of the input batch of SPD matrices
        """
        sqrtm_data, eigvals, eigvecs = sqrtm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return sqrtm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix square root of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix square roots of the input batch of SPD matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, torch.sqrt, sqrt_derivative
        )


# ------------------------------
# SPD matrix inverse square root
# ------------------------------
def inv_sqrtm_SPD(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Inverse matrix square root of a batch of SPD matrices.

    .. math:: P^{-1/2} = V \operatorname{diag}(\lambda^{-1/2}) V^\top

    via :func:`eigh_operation` with ``operation=inv_sqrt``.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    inv_sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
        Inverse matrix square roots of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, inv_sqrt), eigvals, eigvecs


class InvSqrtmSPD(Function):
    """
    Matrix inverse square root of a batch of SPD matrices.

    Relies on the eigenvalue decomposition.
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix inverse square root of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        inv_sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix inverse square roots of the input batch of SPD matrices
        """
        inv_sqrtm_data, eigvals, eigvecs = inv_sqrtm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return inv_sqrtm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix inverse square root of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix square roots of the input batch of SPD matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, inv_sqrt, inv_sqrt_derivative
        )


def sqrtm_and_inv_sqrtm_SPD(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix square root and inverse square root of a batch of SPD matrices.

    Relies on the eigenvalue decomposition (one for both).

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix square roots of the input batch of SPD matrices

    inv_sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix inverse square root of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return (
        eigh_operation(eigvals, eigvecs, torch.sqrt),
        eigh_operation(eigvals, eigvecs, inv_sqrt),
        eigvals,
        eigvecs,
    )


class SqrtmAndInvSqrtmSPD(Function):
    """
    Matrix square roots and inverse square roots of a batch of SPD matrices.

    Relies on the eigenvalue decomposition (one for both).
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the matrix square roots and inverse square roots.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix square roots of the input batch of SPD matrices

        inv_sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix inverse square root of the input batch of SPD matrices
        """
        data_sqrtm, data_inv_sqrtm, eigvals, eigvecs = sqrtm_and_inv_sqrtm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return data_sqrtm, data_inv_sqrtm

    @staticmethod
    def backward(
        ctx, grad_output_sqrtm: torch.Tensor, grad_output_inv_sqrtm: torch.Tensor
    ) -> torch.Tensor:
        """
        Backward pass of the matrix square roots and inverse square roots.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_sqrtm : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss w.r.t. matrix square roots of the input batch of SPD matrices

        grad_output_inv_sqrtm : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss w.r.t. matrix inverse square roots of the input batch of SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss w.r.t. the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(
            grad_output_sqrtm, eigvals, eigvecs, torch.sqrt, sqrt_derivative
        ) + eigh_operation_grad(
            grad_output_inv_sqrtm, eigvals, eigvecs, inv_sqrt, inv_sqrt_derivative
        )


# ----------------
# SPD matrix power
# ----------------
def powm_SPD(
    data: torch.Tensor, exponent: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Matrix power of a batch of SPD matrices.

    .. math:: P^{p} = V \operatorname{diag}(\lambda^{p}) V^\top

    via :func:`eigh_operation` with ``operation=lambda x: x**exponent``.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    exponent : torch.float
        Power exponent

    Returns
    -------
    powm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix powers of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    pow_fun = lambda x: torch.pow(x, exponent)
    return eigh_operation(eigvals, eigvecs, pow_fun), eigvals, eigvecs


class PowmSPD(Function):
    """Matrix power of a batch of SPD matrices (relies on eigenvalue decomposition)."""

    @staticmethod
    def forward(
        ctx, data: torch.Tensor, exponent: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the matrix power of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        exponent : float | torch.Tensor
            Power exponent

        Returns
        -------
        powm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix powers of the input batch of SPD matrices
        """
        powm_data, eigvals, eigvecs = powm_SPD(data, exponent)
        # save_for_backward only accepts tensors: a python exponent is stored
        # on ctx and gets no gradient
        ctx.exponent_is_tensor = isinstance(exponent, torch.Tensor)
        if ctx.exponent_is_tensor:
            ctx.save_for_backward(eigvals, eigvecs, exponent)
        else:
            ctx.save_for_backward(eigvals, eigvecs)
            ctx.exponent = exponent
        return powm_data

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Backward pass of the matrix power of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix powers of the input batch of SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices

        grad_input_exponent : torch.Tensor | None
            Gradient of the loss with respect to the power exponent
            (None if the exponent is a python number)
        """
        if ctx.exponent_is_tensor:
            eigvals, eigvecs, exponent = ctx.saved_tensors
        else:
            eigvals, eigvecs = ctx.saved_tensors
            exponent = ctx.exponent
        pow_fun = lambda x: torch.pow(x, exponent)
        pow_deriv = lambda x: exponent * torch.pow(x, exponent - 1)
        pow_dd = lambda a, b: pow_divided_difference(a, b, exponent)
        grad_input_data = eigh_operation_grad(
            grad_output, eigvals, eigvecs, pow_fun, pow_deriv, pow_dd
        )
        grad_input_exponent = None
        if ctx.exponent_is_tensor and ctx.needs_input_grad[1]:
            # the exponent is a scalar shared by the whole batch:
            # d/dp tr(G^T P^p) = tr(G^T U diag(lambda^p log lambda) U^T),
            # summed over the batch
            exponent_deriv_fun = lambda x: pow_exponent_derivative(x, exponent)
            grad_input_exponent = (
                (grad_output @ eigh_operation(eigvals, eigvecs, exponent_deriv_fun))
                .diagonal(offset=0, dim1=-1, dim2=-2)
                .sum()
                .reshape(exponent.shape)
            )
        return grad_input_data, grad_input_exponent


# --------------------
# SPD matrix logarithm
# --------------------
def logm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Matrix logarithm of a batch of SPD matrices.

    .. math:: \log(P) = V \operatorname{diag}(\log\lambda) V^\top

    via :func:`eigh_operation` with ``operation=torch.log``. Maps the SPD
    manifold to the vector space of symmetric matrices — the basis of the
    Log-Euclidean geometry (see
    :mod:`~yetanotherspdnet.functions.spd_geometries.log_euclidean`).

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    logm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix logarithms of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.log), eigvals, eigvecs


class LogmSPD(Function):
    """Matrix logarithm of a batch of SPD matrices (relies on eigenvalue decomposition)."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix logarithm of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        logm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix logarithms of the input batch of SPD matrices
        """
        logm_data, eigvals, eigvecs = logm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return logm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix logarithm of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix logarithms of the input batch of SPD matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(grad_output, eigvals, eigvecs, torch.log, inv)


# ----------------------------
# Symmetric matrix exponential
# ----------------------------
def expm_symmetric(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Matrix exponential of a batch of symmetric matrices.

    .. math:: \exp(S) = V \operatorname{diag}(\exp\lambda) V^\top

    via :func:`eigh_operation` with ``operation=torch.exp``. The result is
    always SPD (eigenvalues :math:`\exp\lambda > 0`); inverse of
    :func:`logm_SPD`.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    expm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix exponentials of the input batch of symmetric matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.exp), eigvals, eigvecs


class ExpmSymmetric(Function):
    """
    Matrix exponential of a batch of symmetric matrices.

    Relies on the eigenvalue decomposition.
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix exponential of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        expm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix exponentials of the input batch of symmetric matrices
        """
        expm_data, eigvals, eigvecs = expm_symmetric(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return expm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix exponential of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix exponentials of the input batch of symmetric matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of symmetric matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(grad_output, eigvals, eigvecs, torch.exp, torch.exp)


# -------------------------
# Scaled SoftPlus Symmetric
# -------------------------
def scaled_softplus_symmetric(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Scaled matrix SoftPlus of a batch of symmetric matrices.

    .. math:: f(S) = V \operatorname{diag}\big(\log_2(1 + 2^{\lambda})\big) V^\top

    via :func:`eigh_operation` with
    ``operation=`` :func:`~yetanotherspdnet.functions.scalar_functions.scaled_softplus`.
    Maps any symmetric matrix to an SPD matrix (eigenvalues strictly
    positive) — used to parametrize BiMap weights or BatchNorm scale so
    they stay on the SPD manifold under unconstrained optimization.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    softplus_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix SoftPlus of the input batch of symmetric matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, scaled_softplus), eigvals, eigvecs


class ScaledSoftPlusSymmetric(Function):
    """
    Scaled matrix SoftPlus of a batch of symmetric matrices.

    It is scaled so that f(0) = 1, f(x) -> 0 as x -> -inf and f'(x) -> 1 as x -> +inf.
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scaled matrix SoftPlus of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        softplus_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix SoftPlus of the input batch of symmetric matrices
        """
        softplus_data, eigvals, eigvecs = scaled_softplus_symmetric(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return softplus_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the scaled matrix SoftPlus of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix SoftPlus of the input batch of symmetric matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of symmetric matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, scaled_softplus, scaled_softplus_derivative
        )


# ---------------------------
# Inverse Scaled SoftPlus SPD
# ---------------------------
def inv_scaled_softplus_SPD(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Inverse scaled SoftPlus of a batch of SPD matrices.

    .. math:: f^{-1}(P) = V \operatorname{diag}\big(\log_2(2^{\lambda} - 1)\big) V^\top

    via :func:`eigh_operation`. Inverse of :func:`scaled_softplus_symmetric`.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    inv_softplus_data : torch.Tensor of shape (..., n_features, n_features)
        Inverse scaled SoftPlus of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    return eigh_operation(eigvals, eigvecs, inv_scaled_softplus), eigvals, eigvecs


class InvScaledSoftPlusSPD(Function):
    """Matrix inverse scaled SoftPlus of a batch of SPD matrices."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix inverse scaled SoftPlus of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        inv_softplus_data : torch.Tensor of shape (..., n_features, n_features)
            Inverse scaled SoftPlus of the input batch of SPD matrices
        """
        inv_softplus_data, eigvals, eigvecs = inv_scaled_softplus_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return inv_softplus_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix inverse scaled SoftPlus of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to matrix inverse SoftPlus of the input batch of SPD matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eigvals, eigvecs = ctx.saved_tensors
        return eigh_operation_grad(
            grad_output,
            eigvals,
            eigvecs,
            inv_scaled_softplus,
            inv_scaled_softplus_derivative,
        )


# -----------------------------------------------------
# ReLu activation function on eigenvalues of SPD matrix
# -----------------------------------------------------
# a bound of the eigenvalues: a number, a tensor broadcasting against the
# eigenvalues, or a callable computing it from the (detached) eigenvalues
EigvalBound = float | torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None


@overload
def spd_floor(
    eigvals: torch.Tensor, floor: float = ..., rtol: float = ...
) -> torch.Tensor: ...


@overload
def spd_floor(eigvals: torch.dtype, floor: float = ..., rtol: float = ...) -> float: ...


def spd_floor(
    eigvals: torch.Tensor | torch.dtype,
    floor: float = SPD_EIGVAL_FLOOR,
    rtol: float = SPD_FLOOR_RTOL,
) -> torch.Tensor | float:
    r"""
    Dtype-aware lower bound for the eigenvalues of SPD matrices.

    .. math:: \max\big(\text{floor},\; c\, \varepsilon \max_k |\lambda_k|\big)

    per matrix, with :math:`\varepsilon` the machine epsilon of the dtype and
    :math:`c` = ``rtol`` (``SPD_FLOOR_RTOL`` = 1): rounding the matrix to the
    dtype already moves its eigenvalues by about :math:`\varepsilon \lVert X
    \rVert`, so an eigenvalue below that level carries no information and may
    come out of the decomposition negative. In float64 the relative term
    exceeds the default absolute floor 1e-8 only for :math:`\max \lambda > 4.5
    \cdot 10^{7}`; in float32 it bounds the condition number by
    :math:`1 / \varepsilon_{32} \approx 8 \cdot 10^{6}`.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features) or torch.dtype
        Eigenvalues (the result is per matrix, of shape (..., 1), detached), or a
        dtype (the result is the float :math:`\max(\text{floor}, c\,\varepsilon)`,
        for unit-scale matrices)
    floor : float, optional
        Absolute floor. Default is ``SPD_EIGVAL_FLOOR`` (1e-8)
    rtol : float, optional
        Factor :math:`c`. Default is ``SPD_FLOOR_RTOL`` (1)

    Returns
    -------
    torch.Tensor of shape (..., 1) or float
        Lower bound of the eigenvalues
    """
    if isinstance(eigvals, torch.dtype):
        return max(floor, rtol * torch.finfo(eigvals).eps)
    eigvals = eigvals.detach()
    eps = torch.finfo(eigvals.dtype).eps
    relative = rtol * eps * eigvals.abs().amax(-1, keepdim=True)
    return relative.clamp_min(floor)


def _resolve_bound(bound: EigvalBound, eigvals: torch.Tensor):
    """Evaluate a callable eigenvalue bound on the (detached) eigenvalues."""
    if callable(bound):
        return bound(eigvals.detach())
    return bound


def _pairwise_bound(bound):
    """Per-matrix bound of shape (..., 1) -> (..., 1, 1), for Loewner-shaped inputs."""
    return bound.unsqueeze(-1) if isinstance(bound, torch.Tensor) else bound


def _clamp(
    eigvals: torch.Tensor,
    min_eigval: float | torch.Tensor | None,
    max_eigval: float | torch.Tensor | None,
) -> torch.Tensor:
    """torch.clamp accepting a number and a tensor bound together."""
    if isinstance(min_eigval, torch.Tensor) or isinstance(max_eigval, torch.Tensor):
        as_tensor = lambda b: (
            None
            if b is None
            else torch.as_tensor(b, dtype=eigvals.dtype, device=eigvals.device)
        )
        return torch.clamp(
            eigvals, min=as_tensor(min_eigval), max=as_tensor(max_eigval)
        )
    return torch.clamp(eigvals, min=min_eigval, max=max_eigval)


def eigh_relu(
    data: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    ReLu activation function on the eigenvalues of SPD matrices.

    .. math::

        \operatorname{ReEig}_\epsilon(P) = V \operatorname{diag}\big(
            \max(\lambda, \epsilon)\big) V^\top

    via :func:`eigh_operation`. This is the ReEig layer's core operation
    (:class:`~yetanotherspdnet.nn.base.ReEig`): clamps small/negative
    eigenvalues to :math:`\epsilon > 0` to keep the result SPD, the
    manifold analogue of ReLU rectification.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    eps : float | torch.Tensor | Callable
        Value for the rectification of the eigenvalues: a number, a tensor of
        shape (..., 1), or a callable of the eigenvalues (e.g. :func:`spd_floor`)

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices with rectified eigenvalues

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    eps = _resolve_bound(eps, eigvals)
    operation = lambda x: torch.clamp(x, min=eps)
    return eigh_operation(eigvals, eigvecs, operation), eigvals, eigvecs


class EighReLu(Function):
    """ReLu activation function on the eigenvalues of SPD matrices."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Forward pass of the ReLu activation function on the eigenvalues of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        eps : float | torch.Tensor | Callable
            Value for the rectification of the eigenvalues (see :func:`eigh_relu`)

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues
        """
        data_transformed, eigvals, eigvecs = eigh_relu(data, eps)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = _resolve_bound(eps, eigvals)
        return data_transformed

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass of the ReLu activation on the eigenvalues.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output batch of SPD matrices

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        eps = ctx.eps
        eigvals, eigvecs = ctx.saved_tensors
        operation = lambda x: torch.clamp(x, min=eps)
        # the derivative is evaluated on (..., n, n) midpoints
        eps_pairwise = _pairwise_bound(eps)
        operation_deriv = lambda x: (x > eps_pairwise).type(x.dtype)
        return (
            eigh_operation_grad(
                grad_output, eigvals, eigvecs, operation, operation_deriv
            ),
            None,
        )


def eigh_clamp(
    data: torch.Tensor,
    min_eigval: float | torch.Tensor | None = None,
    max_eigval: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Clamp the eigenvalues of symmetric matrices to :math:`[\lambda_-, \lambda_+]`.

    .. math::

        U \operatorname{diag}\big(\min(\max(\lambda, \lambda_-), \lambda_+)\big) U^\top

    A projection onto the SPD matrices with bounded spectrum (with
    ``max_eigval=None`` it is :func:`eigh_relu`).

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    min_eigval, max_eigval : float | torch.Tensor | Callable | None
        Bounds of the eigenvalues (None: no bound); tensors broadcast against
        the eigenvalues, of shape (..., n_features); a callable is evaluated on
        the detached eigenvalues (e.g. :func:`spd_floor`)

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrices with clamped eigenvalues

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    min_eigval = _resolve_bound(min_eigval, eigvals)
    max_eigval = _resolve_bound(max_eigval, eigvals)
    operation = lambda x: _clamp(x, min_eigval, max_eigval)
    return eigh_operation(eigvals, eigvecs, operation), eigvals, eigvecs


def _clamp_derivative(
    eigvals: torch.Tensor,
    min_eigval: float | torch.Tensor | None,
    max_eigval: float | torch.Tensor | None,
) -> torch.Tensor:
    """1 strictly inside the bounds, 0 outside (subgradient 0 at a bound)."""
    inside = torch.ones_like(eigvals, dtype=torch.bool)
    if min_eigval is not None:
        inside = inside & (eigvals > min_eigval)
    if max_eigval is not None:
        inside = inside & (eigvals < max_eigval)
    return inside.to(eigvals.dtype)


class EighClamp(Function):
    """Clamp the eigenvalues of symmetric matrices, with a Daleckii-Krein backward."""

    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        min_eigval: float | torch.Tensor | None = None,
        max_eigval: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the eigenvalue clamp.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        min_eigval, max_eigval : float | torch.Tensor | Callable | None
            Bounds of the eigenvalues (see :func:`eigh_clamp`), not differentiated

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of matrices with clamped eigenvalues
        """
        data_transformed, eigvals, eigvecs = eigh_clamp(data, min_eigval, max_eigval)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.bounds = (
            _resolve_bound(min_eigval, eigvals),
            _resolve_bound(max_eigval, eigvals),
        )
        return data_transformed

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """
        Backward pass of the eigenvalue clamp.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output batch

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch
        """
        eigvals, eigvecs = ctx.saved_tensors
        min_eigval, max_eigval = ctx.bounds
        operation = lambda x: _clamp(x, min_eigval, max_eigval)
        # the derivative is evaluated on (..., n, n) midpoints
        bounds_pairwise = (_pairwise_bound(min_eigval), _pairwise_bound(max_eigval))
        operation_deriv = lambda x: _clamp_derivative(x, *bounds_pairwise)
        grad_input = eigh_operation_grad(
            grad_output, eigvals, eigvecs, operation, operation_deriv
        )
        return grad_input, None, None


def eigvalsh_symmetric(data: torch.Tensor) -> torch.Tensor:
    r"""
    Eigenvalues of symmetric matrices, in ascending order.

    ``torch.linalg.eigvalsh``, differentiated by autograd. Its gradient
    :math:`U \operatorname{diag}(\bar\lambda) U^\top` has no eigenvalue gap, but
    see :class:`EigvalshSymmetric` for the manual-backward twin.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    Returns
    -------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues, ascending
    """
    return symmetric_eigvalsh(data)


class EigvalshSymmetric(Function):
    r"""
    Eigenvalues of symmetric matrices, with the backward :math:`U \operatorname{diag}(\bar\lambda) U^\top`.

    The derivative of a simple eigenvalue is :math:`d\lambda_i = u_i^\top dX u_i`;
    for repeated eigenvalues the formula gives the gradient of their (symmetric)
    sum, the only differentiable combination.
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalues of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        eigvals : torch.Tensor of shape (..., n_features)
            Eigenvalues, ascending
        """
        eigvals, eigvecs = symmetric_eigh(data)
        ctx.save_for_backward(eigvecs)
        return eigvals

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Gradient with respect to the matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        grad_output : torch.Tensor of shape (..., n_features)
            Gradient of the loss with respect to the eigenvalues

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the matrices
        """
        (eigvecs,) = ctx.saved_tensors
        return (eigvecs * grad_output.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def _clamped_bias_operation(bias: torch.Tensor, eps: float) -> Callable:
    """Eigenvalue map of ReEigBias: shift by ``bias`` then clamp to [eps, 1/eps]."""
    return lambda x: torch.clamp(x + bias, min=eps, max=1 / eps)


def _clamped_bias_operation_deriv(bias: torch.Tensor, eps: float) -> Callable:
    """Compute the derivative of :func:`_clamped_bias_operation` (eigenvalues)."""
    return lambda x: ((x + bias > eps) & (x + bias < 1 / eps)).type(x.dtype)


def eigh_relu_bias(
    data: torch.Tensor, bias: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Eigenvalue rectification with a learnable shift of the eigenvalues.

    .. math::

        \operatorname{ReEigBias}_{\epsilon, b}(P) = V \operatorname{diag}\big(
            \operatorname{clamp}(\lambda_i + b_i,\ \epsilon,\ 1/\epsilon)
        \big) V^\top

    with :math:`\lambda_1 \le \dots \le \lambda_n` the eigenvalues in ascending
    order (as returned by ``torch.linalg.eigh``) and :math:`b` a bias vector
    indexed by eigenvalue rank. The upper clamp bounds the condition number of
    the output by :math:`\epsilon^{-2}`.

    Unlike :func:`eigh_relu`, this is not a spectral function when two
    eigenvalues coincide while their biases differ: the output then depends on
    the arbitrary eigenbasis of the repeated eigenvalue, and gradients with
    respect to ``data`` are only defined for distinct eigenvalues.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    bias : torch.Tensor of shape (n_features,)
        Shift added to the (ascending) eigenvalues

    eps : float
        Lower clamping value; the upper one is ``1 / eps``

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = symmetric_eigh(data)
    operation = _clamped_bias_operation(bias, eps)
    return eigh_operation(eigvals, eigvecs, operation), eigvals, eigvecs


class EighReLuBias(Function):
    """Eigenvalue rectification with a learnable shift, with a hand-written backward."""

    @staticmethod
    def forward(
        ctx, data: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """
        Forward pass of :func:`eigh_relu_bias`.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        bias : torch.Tensor of shape (n_features,)
            Shift added to the (ascending) eigenvalues

        eps : float
            Lower clamping value; the upper one is ``1 / eps``

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices
        """
        data_transformed, eigvals, eigvecs = eigh_relu_bias(data, bias, eps)
        ctx.save_for_backward(eigvals, eigvecs, bias)
        ctx.eps = eps
        return data_transformed

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        r"""
        Backward pass of :func:`eigh_relu_bias`.

        The gradient with respect to ``data`` is the Daleckii-Krein formula of
        :func:`eigh_operation_grad`. Only the eigenvalues depend on the bias, so
        its gradient is the diagonal of :math:`V^\top G V` masked by the
        derivative of the clamp, summed over the batch dimensions.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output batch

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch

        grad_bias : torch.Tensor of shape (n_features,)
            Gradient of the loss with respect to the bias
        """
        eps = ctx.eps
        eigvals, eigvecs, bias = ctx.saved_tensors
        operation = _clamped_bias_operation(bias, eps)
        operation_deriv = _clamped_bias_operation_deriv(bias, eps)
        grad_input = eigh_operation_grad(
            grad_output, eigvals, eigvecs, operation, operation_deriv
        )
        rotated = eigvecs.transpose(-1, -2) @ symmetrize(grad_output) @ eigvecs
        grad_eigvals = torch.diagonal(rotated, dim1=-2, dim2=-1) * operation_deriv(
            eigvals
        )
        grad_bias = grad_eigvals.reshape(-1, grad_eigvals.shape[-1]).sum(dim=0)
        return grad_input, grad_bias, None


# -----------------------------------
# Various congruences of SPD matrices
# -----------------------------------
def congruence_SPD(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Congruence of a batch of SPD matrices with an SPD matrix.

    .. math:: P' = A\, P\, A

    (here :math:`A` is itself SPD, hence symmetric, so :math:`A^\top = A`
    and there is no separate transpose). Congruence by an SPD matrix
    preserves the SPD manifold.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    matrix : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Transformed batch of SPD matrices
    """
    return matrix @ data @ matrix


class CongruenceSPD(Function):
    """Congruence of a batch of SPD matrices with an SPD matrix."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the congruence of a batch of SPD matrices with an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        ctx.save_for_backward(data, matrix)
        return congruence_SPD(data, matrix)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the congruence of a batch of SPD matrices with an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
           Gradient of the loss with respect to the batch of transformed SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices

        grad_input_bias : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the SPD matrix used for congruence
        """
        data, matrix = ctx.saved_tensors
        return (
            matrix @ symmetrize(grad_output) @ matrix,
            2
            * symmetrize(
                torch.einsum(
                    "...ik,kl,...lj->ij", symmetrize(grad_output), matrix, data
                )
            ),
        )


def congruence_SPD_sqrtm(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Congruence of a batch of SPD matrices with the matrix square root of an SPD matrix.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    matrix : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Transformed batch of SPD matrices
    """
    matrix_sqrtm = sqrtm_SPD(matrix)[0]
    return congruence_SPD(data, matrix_sqrtm)


class CongruenceSPDSqrtm(Function):
    """Congruence of a batch of SPD matrices with the matrix square root of an SPD matrix."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the congruence with the square root of an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        matrix_sqrtm, eigvals_matrix, eigvecs_matrix = sqrtm_SPD(matrix)
        ctx.save_for_backward(data, eigvals_matrix, eigvecs_matrix, matrix_sqrtm)
        return congruence_SPD(data, matrix_sqrtm)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the congruence with the square root of an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the batch of whitened SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices

        grad_input_matrix : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the SPD matrix used for whitening
        """
        data, eigvals_matrix, eigvecs_matrix, matrix_sqrtm = ctx.saved_tensors
        grad_input_data = matrix_sqrtm @ symmetrize(grad_output) @ matrix_sqrtm
        syl_right = 2 * symmetrize(
            torch.einsum("...ik, kl, ...lj->ij", data, matrix_sqrtm, grad_output)
        )
        grad_input_matrix = solve_sylvester_SPD(
            torch.sqrt(eigvals_matrix), eigvecs_matrix, syl_right
        )
        return grad_input_data, grad_input_matrix


def whitening(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Whitening of a batch of SPD matrices with an SPD matrix.

    .. math:: P' = A^{-1/2}\, P\, A^{-1/2}

    i.e. :func:`congruence_SPD` with :math:`A^{-1/2}` (see
    :func:`inv_sqrtm_SPD`) — transforms data so that :math:`A` itself maps
    to the identity, the SPD analogue of standardizing by the covariance.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    matrix : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Transformed batch of SPD matrices
    """
    inv_sqrtm_matrix, _, _ = inv_sqrtm_SPD(matrix)
    return congruence_SPD(data, inv_sqrtm_matrix)


class Whitening(Function):
    """Whitening of a batch of SPD matrices with an SPD matrix."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the whitening of a batch of SPD matrices with an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        inv_sqrtm_matrix, eigvals_matrix, eigvecs_matrix = inv_sqrtm_SPD(matrix)
        ctx.save_for_backward(data, eigvals_matrix, eigvecs_matrix, inv_sqrtm_matrix)
        return congruence_SPD(data, inv_sqrtm_matrix)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the whitening of a batch of SPD matrices with an SPD matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the batch of whitened SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices

        grad_input_matrix : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the SPD matrix used for whitening
        """
        data, eigvals_matrix, eigvecs_matrix, inv_sqrtm_matrix = ctx.saved_tensors
        grad_input_data = inv_sqrtm_matrix @ symmetrize(grad_output) @ inv_sqrtm_matrix
        syl_right = -2 * symmetrize(
            torch.einsum("...ik,...kl,lj->ij", grad_input_data, data, inv_sqrtm_matrix)
        )
        grad_input_matrix = solve_sylvester_SPD(
            torch.sqrt(eigvals_matrix), eigvecs_matrix, syl_right
        )
        return grad_input_data, grad_input_matrix


def congruence_rectangular(data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    r"""
    Forward pass of the congruence with a full-rank rectangular matrix.

    .. math:: P' = W^\top P\, W, \qquad W \in \mathbb{R}^{n_{in} \times n_{out}}

    with :math:`n_{in} \geq n_{out}`. This is the BiMap layer's core
    operation (:class:`~yetanotherspdnet.nn.base.BiMap`): a dimension
    reduction that keeps the result SPD as long as :math:`W` has full
    column rank.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_in, n_in)
        Batch of SPD matrices

    weight : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (e.g., weights),
        n_in > n_out is expected

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_out, n_out)
        Transformed batch of SPD matrices
    """
    if weight.shape[-2] < weight.shape[-1]:
        raise ValueError(
            "weight must reduce the dimension of data, i.e., n_in >= n_out"
        )
    return weight.transpose(-1, -2) @ data @ weight


class CongruenceRectangular(Function):
    """Congruence of a batch of SPD matrices with a (full-rank) rectangular matrix."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the congruence with a full-rank rectangular matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_in, n_in)
            Batch of SPD matrices

        weight : torch.Tensor of shape (n_out, n_in)
            Rectangular matrix (e.g., weights),
            n_in > n_out is expected

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_out, n_out)
            Transformed batch of SPD matrices
        """
        ctx.save_for_backward(data, weight)
        return congruence_rectangular(data, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the congruence with a full-rank rectangular matrix.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_out, n_out)
            Gradient of the loss with respect to the batch of transformed SPD matrices

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_in, n_in)
            Gradient of the loss with respect to the input batch of SPD matrices

        grad_input_W : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the (full-rank) rectangular matrix W
        """
        data, weight = ctx.saved_tensors
        grad_input_data = weight @ grad_output @ weight.transpose(-1, -2)
        grad_input_W = 2 * torch.einsum("...ik,kl,...lj->ij", data, weight, grad_output)
        return grad_input_data, grad_input_W
