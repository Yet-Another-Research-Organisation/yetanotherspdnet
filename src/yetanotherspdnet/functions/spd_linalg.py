from collections.abc import Callable

import torch
from torch.autograd import Function


def symmetrize(data: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize a tensor along the last two dimensions

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of square matrices

    Returns
    -------
    sym_data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetrized matrices
    """
    return torch.real(0.5 * (data + data.transpose(-1, -2)))


# -----------------------
# Vectorization operators
# -----------------------
def vec_batch(data: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a batch of tensors along last two dimensions

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
    Unvectorize a batch of tensors along last dimension

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


def vech_batch(data: torch.Tensor) -> torch.Tensor:
    """
    Vectorize the lower triangular part of a batch of square matrices

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


def unvech_batch(data_vech: torch.Tensor) -> torch.Tensor:
    """
    Unvectorize a batch of tensors along last dimension,
    assuming that matrices are symmetric

    Parameters
    ----------
    X_vech : torch.Tensor of shape (..., n_features*(n_features+1)//2)
        Batch of vectorized matrices

    Returns
    -------
    X : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices
    """
    n_features = 0.5 * (-1 + torch.sqrt(torch.Tensor([1 + 8 * data_vech.shape[-1]])))
    n_features = int(torch.round(n_features))
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


# -------------------------
# Operations on eigenvalues
# -------------------------
def eigh_operation(
    eigvals: torch.Tensor, eigvecs: torch.Tensor, operation: Callable
) -> torch.Tensor:
    """
    Applies a function on the eigenvalues of a batch of symmetric matrices

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
    eigvals, eigvecs = (
        torch.real(eigvals),
        torch.real(eigvecs),
    )  # to correct eventual numerical errors...
    _eigvals = operation(eigvals)
    result = (eigvecs * _eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return result


def _aux_eigh_operation_grad(
    eigvals: torch.Tensor,
    operation: Callable,
    operation_deriv: Callable,
) -> torch.Tensor:
    """
    Constructs matrix to be multiplied (Hadamard product) with transformed output gradient
    to get the input gradient for a function applied on the eigenvalues of a symmetric matrix

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices

    operation : Callable
        Function to apply on eigenvalues

    operation_deriv : Callable
        Derivative of the function to apply on eigenvalues

    Returns
    -------
    aux_mat : torch.Tensor of shape (..., n_features, n_features)
        Matrix to be multiplied (Hadamard product) with transformed output gradient
    """
    eigvals_transformed = operation(eigvals)
    eigvals_deriv = operation_deriv(eigvals)

    denominator = eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2)
    numerator = eigvals_transformed.unsqueeze(-1) - eigvals_transformed.unsqueeze(-2)

    null_denominator = (
        torch.abs(denominator) < 1e-6
    )  # arbitrary value, probably to be changed to be proper... Ammar ?

    numerator = torch.where(null_denominator, eigvals_deriv.unsqueeze(-1), numerator)
    denominator = torch.where(null_denominator, torch.ones_like(numerator), denominator)

    return numerator / denominator


def eigh_operation_grad(
    grad_output: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    operation: Callable,
    operation_deriv: Callable,
) -> torch.Tensor:
    """
    Computes the backpropagation of the gradient for a function applied on
    the eigenvalues of a batch of symmetric matrices

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

    Returns
    -------
    grad_input : torch.Tensor of shape (..., n_features, n_features)
        Gradient of the loss with respect to the input batch of symmetric matrices
    """
    aux_mat = _aux_eigh_operation_grad(eigvals, operation, operation_deriv)
    middle_term = aux_mat * (eigvecs.transpose(-1, -2) @ grad_output @ eigvecs)
    return eigvecs @ middle_term @ eigvecs.transpose(-1, -2)


# ------------------
# Sylvester equation
# ------------------
def solve_sylvester_SPD(
    eigvals: torch.Tensor, eigvecs: torch.Tensor, mat: torch.Tensor
) -> torch.Tensor:
    """
    Solve Sylvester equations in the context of SPD matrices
    relying on eigenvalue decomposition

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
    # should those corrections be here though ?
    eigvals, eigvecs = (
        torch.real(eigvals),
        torch.real(eigvecs),
    )  # to correct eventual numerical errors...
    K = 1 / (eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2))
    middle_term = K * (eigvecs.transpose(-1, -2) @ mat @ eigvecs)
    return eigvecs @ middle_term @ eigvecs.transpose(-1, -2)


# ----------------------
# SPD matrix square root
# ----------------------
def sqrtm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix logarithm of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    Returns
    -------
    sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
        Matrix logarithms of the input batch of SPD matrices

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.sqrt), eigvals, eigvecs


class SqrtmSPD(Function):
    """
    Matrix square root of a batch of SPD matrices
    (relies on eigenvalue decomposition)
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix square root of a batch of SPD matrices

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
        Backward pass of the matrix square root of a batch of SPD matrices

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
        operation_derivative = lambda x: 0.5 / torch.sqrt(x)
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, torch.sqrt, operation_derivative
        )


# ------------------------------
# SPD matrix inverse square root
# ------------------------------
def inv_sqrtm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix logarithm of a batch of SPD matrices

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
    eigvals, eigvecs = torch.linalg.eigh(data)
    inv_sqrt = lambda x: 1 / torch.sqrt(x)
    return eigh_operation(eigvals, eigvecs, inv_sqrt), eigvals, eigvecs


class InvSqrtmSPD(Function):
    """
    Matrix inverse square root of a batch of SPD matrices
    (relies on eigenvalue decomposition)
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix inverse square root of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        inv_sqrtm_data : torch.Tensor of shape (..., n_features, n_features)
            Matrix square roots of the input batch of SPD matrices
        """
        inv_sqrtm_data, eigvals, eigvecs = inv_sqrtm_SPD(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return inv_sqrtm_data

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the matrix inverse square root of a batch of SPD matrices

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
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        operation_derivative = lambda x: -0.5 / (x**1.5)
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, inv_sqrt, operation_derivative
        )


# --------------------
# SPD matrix logarithm
# --------------------
def logm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix logarithm of a batch of SPD matrices

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
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.log), eigvals, eigvecs


class LogmSPD(Function):
    """
    Matrix logarithm of a batch of SPD matrices
    (relies on eigenvalue decomposition)
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix logarithm of a batch of SPD matrices

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
        Backward pass of the matrix logarithm of a batch of SPD matrices

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
        operation_derivative = lambda x: 1 / x
        return eigh_operation_grad(
            grad_output, eigvals, eigvecs, torch.log, operation_derivative
        )


# ----------------------------
# Symmetric matrix exponential
# ----------------------------
def expm_symmetric(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix exponential of a batch of symmetric matrices

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
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.exp), eigvals, eigvecs


class ExpmSymmetric(Function):
    """
    Matrix exponential of a batch of symmetric matrices
    (relies on eigenvalue decomposition)
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix exponential of a batch of symmetric matrices

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
        Backward pass of the matrix exponential of a batch of symmetric matrices

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


# -----------------------------------------------------
# ReLu activation function on eigenvalues of SPD matrix
# -----------------------------------------------------
def eigh_relu(data: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ReLu activation function on the eigenvalues of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    eps : float
        Value for the rectification of the eigenvalues

    Returns
    -------
    data_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices with rectified eigenvalues

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of matrices in data

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of matrices in data
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
    return eigh_operation(eigvals, eigvecs, operation), eigvals, eigvecs


class EighReLu(Function):
    """
    ReLu activation function on the eigenvalues of SPD matrices
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Forward pass of the ReLu activation function on the eigenvalues of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        eps : float
            Value for the rectification of the eigenvalues

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues
        """
        data_transformed, eigvals, eigvecs = eigh_relu(data,eps)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = eps
        return data_transformed

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass of the ReLu activation function on the eigenvalues of SPD matrices

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
        operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
        operation_deriv = lambda x: (x > eps).type(x.dtype)

        return (
            eigh_operation_grad(
                grad_output, eigvals, eigvecs, operation, operation_deriv
            ),
            None,
        )


# -----------------------------------
# Various congruences of SPD matrices
# -----------------------------------
def congruence_SPD(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Congruence of a batch of SPD matrices with an SPD matrix

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
    """
    Congruence of a batch of SPD matrices with an SPD matrix
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the congruence of a batch of SPD matrices with an SPD matrix

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
        Backward pass of the congruence of a batch of SPD matrices with an SPD matrix

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
        #return matrix @ grad_output @ matrix, torch.sum(
        #    2 * symmetrize(grad_output @ matrix @ data), dim=tuple(range(data.ndim - 2))
        #)
        return (
            matrix @ grad_output @ matrix,
            2 * symmetrize( torch.einsum('...ik,kl,...lj->ij', grad_output, matrix, data) )
        )

def whitening(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Whitening of a batch of SPD matrices with an SPD matrix

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
    """
    Whitening of a batch of SPD matrices with an SPD matrix
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the whitening of a batch of SPD matrices with an SPD matrix

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
        Backward pass of the whitening of a batch of SPD matrices with an SPD matrix

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
        grad_input_data = inv_sqrtm_matrix @ grad_output @ inv_sqrtm_matrix
        #syl_right = -torch.sum(
        #    2 * symmetrize(grad_input_data @ data @ inv_sqrtm_matrix),
        #    dim=tuple(range(data.ndim - 2)),
        #)
        syl_right = -2 * symmetrize(
            torch.einsum('...ik,...kl,lj->ij', grad_input_data, data, inv_sqrtm_matrix)
        )
        grad_input_matrix = solve_sylvester_SPD(
            torch.sqrt(eigvals_matrix), eigvecs_matrix, syl_right
        )
        return grad_input_data, grad_input_matrix


def congruence_rectangular(data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

    Parameters
    ----------
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
    assert weight.shape[-1] > weight.shape[-2], (
        "weight must reduce the dimension of data, i.e., n_in > n_out"
    )
    return weight @ data @ weight.transpose(-1, -2)


class CongruenceRectangular(Function):
    """
    Congruence of a batch of SPD matrices with a (full-rank) rectangular matrix
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

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
        Backward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

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

        grad_input_W : torch.Tensor of shape (n_out, n_in)
            Gradient of the loss with respect to the (full-rank) rectangular matrix W
        """
        data, W = ctx.saved_tensors
        grad_input_data = W.transpose(-1, -2) @ grad_output @ W
        #grad_input_W = 2 * torch.sum(
        #    grad_output @ W @ data, dim=tuple(range(data.ndim - 2))
        #)
        grad_input_W = 2 * torch.einsum('...ik,kl,...lj->ij', grad_output, W, data)
        return grad_input_data, grad_input_W


