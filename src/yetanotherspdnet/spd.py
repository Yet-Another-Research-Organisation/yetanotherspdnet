from collections.abc import Callable

import torch
from torch.autograd import Function


def random_SPD(
    n_features: int,
    n_matrices: int = 1,
    cond: float = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate a batch of random SPD matrices

    Parameters
    ----------
    n_features : int
        Number of features

    n_matrices : int, optional
        Number of matrices. Default is 1

    cond : float, optional
        Condition number w.r.t. inversion of SPD matrices.
        Default is 10

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.

    Returns
    -------
    torch.Tensor of shape (n_matrices, n_features, n_features)
        Batch of SPD matrices
    """
    assert n_features >= 2, "Expected n_features>=2 to form matrices"
    # Generate random orthogonal matrices
    tmp = torch.randn((n_matrices, n_features, n_features), generator=generator)
    U, _, V = torch.svd(tmp)
    eigvecs = U @ V.transpose(-1, -2)

    eigvals = torch.zeros(n_matrices, n_features)
    eigvals[:, 0] = torch.sqrt(torch.tensor(cond))
    eigvals[:, 1] = 1 / torch.sqrt(torch.tensor(cond))
    if n_features > 2:
        eigvals[:, 2:] = torch.rand((n_matrices, n_features - 2), generator=generator)

    return torch.squeeze((eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2))


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    eigvals: torch.Tensor, eigvecs: torch.Tensor, sym_mat: torch.Tensor
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
    sym_mat : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices on the right side of Sylvester equations

    Returns
    -------
    result : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices solutions to Sylvester equations
    """
    eigvals, eigvecs = (
        torch.real(eigvals),
        torch.real(eigvecs),
    )  # to correct eventual numerical errors...
    K = 1 / (eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2))
    middle_term = K * (eigvecs.transpose(-1, -2) @ sym_mat @ eigvecs)
    return eigvecs @ middle_term @ eigvecs.transpose(-1, -2)


# ----------------------
# SPD matrix square root
# ----------------------
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
        eigvals, eigvecs = torch.linalg.eigh(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return eigh_operation(eigvals, eigvecs, torch.sqrt)

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


def sqrtm_SPD(data: torch.Tensor) -> torch.Tensor:
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
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.sqrt)


# ------------------------------
# SPD matrix inverse square root
# ------------------------------
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
        eigvals, eigvecs = torch.linalg.eigh(data)
        ctx.save_for_backward(eigvals, eigvecs)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        return eigh_operation(eigvals, eigvecs, inv_sqrt)

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


def inv_sqrtm_SPD(data: torch.Tensor) -> torch.Tensor:
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
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    inv_sqrt = lambda x: 1 / torch.sqrt(x)
    return eigh_operation(eigvals, eigvecs, inv_sqrt)


# --------------------
# SPD matrix logarithm
# --------------------
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
        eigvals, eigvecs = torch.linalg.eigh(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return eigh_operation(eigvals, eigvecs, torch.log)

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


def logm_SPD(data: torch.Tensor) -> torch.Tensor:
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
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.log)


# ----------------------------
# Symmetric matrix exponential
# ----------------------------
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
        eigvals, eigvecs = torch.linalg.eigh(data)
        ctx.save_for_backward(eigvals, eigvecs)
        return eigh_operation(eigvals, eigvecs, torch.exp)

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


def expm_symmetric(data: torch.Tensor) -> torch.Tensor:
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
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    return eigh_operation(eigvals, eigvecs, torch.exp)


# -----------------------------------
# Various congruences of SPD matrices
# -----------------------------------
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
        return matrix @ data @ matrix

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
        return matrix @ grad_output @ matrix, torch.sum(
            2 * symmetrize(grad_output @ matrix @ data), dim=tuple(range(data.ndim - 2))
        )


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
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        eigvals_matrix, eigvecs_matrix = torch.linalg.eigh(matrix)
        matrix_inv_sqrtm = eigh_operation(eigvals_matrix, eigvecs_matrix, inv_sqrt)
        ctx.save_for_backward(data, eigvals_matrix, eigvecs_matrix, matrix_inv_sqrtm)
        return matrix_inv_sqrtm @ data @ matrix_inv_sqrtm

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
        data, eigvals_matrix, eigvecs_matrix, matrix_inv_sqrtm = ctx.saved_tensors
        grad_input_data = matrix_inv_sqrtm @ grad_output @ matrix_inv_sqrtm
        syl_right = -torch.sum(
            2 * symmetrize(grad_input_data @ data @ matrix_inv_sqrtm),
            dim=tuple(range(data.ndim - 2)),
        )
        grad_input_matrix = solve_sylvester_SPD(
            torch.sqrt(eigvals_matrix), eigvecs_matrix, syl_right
        )
        return grad_input_data, grad_input_matrix


def whitening(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Whitening of a batch of SPD matrices with an SPD matrix

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
    matrix_inv_sqrtm = inv_sqrtm_SPD(matrix)
    return matrix_inv_sqrtm @ data @ matrix_inv_sqrtm


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
        assert weight.shape[-1] > weight.shape[-2], (
            "weight must reduce the dimension of data, i.e., n_in > n_out"
        )
        ctx.save_for_backward(data, weight)
        return weight @ data @ weight.transpose(-1, -2)

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
        grad_input_W = 2 * torch.sum(
            grad_output @ W @ data, dim=tuple(range(data.ndim - 2))
        )
        return grad_input_data, grad_input_W


def congruence_rectangular(data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
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
    assert weight.shape[-1] > weight.shape[-2], (
        "weight must reduce the dimension of data, i.e., n_in > n_out"
    )
    return weight @ data @ weight.transpose(-1, -2)


# -----------------------------------------------------
# ReLu activation function on eigenvalues of SPD matrix
# -----------------------------------------------------
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
        eigvals, eigvecs = torch.linalg.eigh(data)
        operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = eps
        return eigh_operation(eigvals, eigvecs, operation)

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


def eigh_relu(data: torch.Tensor, eps: float) -> torch.Tensor:
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
    """
    eigvals, eigvecs = torch.linalg.eigh(data)
    operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
    return eigh_operation(eigvals, eigvecs, operation)
