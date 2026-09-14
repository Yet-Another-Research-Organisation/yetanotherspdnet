from collections.abc import Callable

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.spd_linalg import (
    CongruenceRectangular,
    congruence_rectangular,
    symmetrize,
    _aux_eigh_operation_grad,
)
from yetanotherspdnet.functions.scalar_functions import inv, sqrt_derivative


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
    return torch.real(0.5 * (data - data.transpose(-1, -2)))


def gevd(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the generalized eigenvalue decomposition of an SPD matrix P and a symmetric matrix Q

    We have :math: `P = A^\top A, Q = A^\top Lambda A`.

    Parameters
    ----------
    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        Symmetric matrix

    Returns
    -------
    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    L = torch.linalg.cholesky(P)
    L_inv = torch.inverse(L)
    Lambda, U = torch.linalg.eigh(L_inv @ Q @ L_inv.transpose(-2, -1))
    return Lambda, (L @ U).transpose(-2, -1), L_inv.transpose(-2, -1) @ U


def gevd_operation(
    Lambda: torch.Tensor, A: torch.Tensor, operation: Callable
) -> torch.Tensor:
    """
    Applies a function on the generalized eigenvalues of the generalized eigenvalue decomposition
    of a symmetric matrix Q and a SPD matrix P.

    We have :math: `P = A^\top A, Q = A^\top Lambda A`.

    It returns :math: `A^\top f(Lambda) A`.

    Parameters
    ----------
    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    operation : Callable
        Function to apply on generalized eigenvalues

    Returns
    -------
    result : torch.Tensor of shape (n_features, n_features)
        Resulting symmetric matrix with operation applied to generalized eigenvalues
    """
    Lambda, A = (
        torch.real(Lambda),
        torch.real(A),
    )  # to correct eventual numerical errors...
    _Lambda = operation(Lambda)
    result = (A.transpose(-1, -2) * _Lambda.unsqueeze(-2)) @ A
    return result


def gevd_operation_grad(
    grad_output: torch.Tensor,
    Lambda: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    operation: Callable,
    operation_derivative: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backpropagation of the gradient for a function applied on the
    generalized eigenvalues of the generalized eigenvalue decomposition of a
    symmetric matrix Q and a SPD matrix P.

    We have :math: `P = A^\top A, Q = A^\top Lambda A, A^{-1} = B`.

    Parameters
    ----------
    grad_output : torch.Tensor of shape (n_features, n_features)
        Gradient of the loss with respect to the output of the operation on generalized eigenvalues

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A

    operation : Callable
        Function to apply on generalized eigenvalues

    operation_derivative : Callable
        Derivative of the function to apply on generalized eigenvalues

    Returns
    -------
    grad_P : torch.Tensor of shape (n_features, n_features)
        Gradient w.r.t. SPD matrix P

    grad_Q : torch.Tensor of shape (n_features, n_features)
        Gradient w.r.t. symmetric matrix Q
    """
    grad_out_cong = A @ symmetrize(grad_output) @ A.transpose(-1, -2)
    aux_mat_Q = _aux_eigh_operation_grad(Lambda, operation, operation_derivative)
    middle_term_Q = aux_mat_Q * grad_out_cong
    grad_Q = B @ middle_term_Q @ B.transpose(-1, -2)
    _Lambda = operation(Lambda)
    middle_term_P = symmetrize(
        grad_out_cong * _Lambda.unsqueeze(-2) - middle_term_Q * Lambda.unsqueeze(-2)
    )
    grad_P = B @ middle_term_P @ B.T
    return grad_P, grad_Q


# ----------------------------------------
# Affine-invariant geodesics based on GEVD
# ----------------------------------------
def affine_invariant_geodesic_gevd(
    P: torch.Tensor, Q: torch.Tensor, t: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

    Parameters
    ----------
    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    S : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of Q w.r.t. P

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    Lambda, A, B = gevd(P, Q)
    pow_t = lambda x: torch.pow(x, t)
    return gevd_operation(Lambda, A, pow_t), Lambda, A, B


class AffineInvariantGeodesicGEVD(Function):
    """
    Affine-invariant geodesic based on generalized eigenvalue decomposition
    """

    @staticmethod
    def forward(
        ctx, P: torch.Tensor, Q: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        P : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Q : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        t : float | torch.Tensor
            parameter on the path, should be in [0,1]

        Returns
        -------
        S : torch.Tensor of shape (n_features, n_features)
            SPD matrix
        """
        S, Lambda, A, B = affine_invariant_geodesic_gevd(P, Q, t)
        ctx.t = t
        ctx.save_for_backward(Lambda, A, B)
        return S

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the output of the affine-invariant geodesic between P and Q

        Returns
        -------
        grad_P : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. P

        grad_Q : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. Q
        """
        t = ctx.t
        Lambda, A, B = ctx.saved_tensors
        pow_t = lambda x: torch.pow(x, t)
        pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
        grad_P, grad_Q = gevd_operation_grad(
            grad_output, Lambda, A, B, pow_t, pow_t_deriv
        )
        return grad_P, grad_Q, None


# ---------------------------------------
# Affine-invariant Riemannian exponential
# ---------------------------------------
def affine_invariant_exp_gevd(
    xi: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant exponential mapping based on generalized eigenvalue decomposition.

    Parameters
    ----------
    xi : torch.Tensor of shape (n_features, n_features)
        Symmetric matrix, tangent vector at P

    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    S : torch.Tensor of shape (n_features, n_features)
        SPD matrix, exponential of xi at P

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of xi w.r.t. P

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    Lambda, A, B = gevd(P, xi)
    return gevd_operation(Lambda, A, torch.exp), Lambda, A, B


class AffineInvariantExpGEVD(Function):
    """
    Affine-invariant exponential mapping based on generalized eigenvalue decomposition.
    """

    @staticmethod
    def forward(ctx, xi: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the affine-invariant exponential mapping based on generalized eigenvalue decomposition.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        xi : torch.Tensor of shape (n_features, n_features)
            Symmetric matrix, tangent vector at P

        P : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        S : torch.Tensor of shape (n_features, n_features)
            SPD matrix, exponential of xi at P
        """
        S, Lambda, A, B = affine_invariant_exp_gevd(xi, P)
        ctx.save_for_backward(Lambda, A, B)
        return S

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the affine-invariant exponential mapping based on generalized eigenvalue decomposition.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the output of the Riemannian exponential mapping

        Returns
        -------
        grad_xi : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. xi

        grad_P : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. P
        """
        Lambda, A, B = ctx.saved_tensors
        grad_P, grad_xi = gevd_operation_grad(
            grad_output, Lambda, A, B, torch.exp, torch.exp
        )
        return grad_xi, grad_P, None


# -------------------------------------
# Affine-invariant Riemannian logarithm
# -------------------------------------
def affine_invariant_log_gevd(
    Q: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant Riemannian logarithm mapping based on generalized eigenvalue decomposition.

    Parameters
    ----------
    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    xi :torch.Tensor of shape (n_features, n_features)
        Symmetric matrix, logarithm of Q at P

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of Q w.r.t. P

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    Lambda, A, B = gevd(P, Q)
    return gevd_operation(Lambda, A, torch.log), Lambda, A, B


class AffineInvariantLogGEVD(Function):
    """
    Affine-invariant logarithm mapping based on generalized eigenvalue decomposition.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the affine-invariant logarithm mapping based on generalized eigenvalue decomposition.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        Q : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        P : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        xi : torch.Tensor of shape (n_features, n_features)
            symmetric matrix, logarithm of Q at P
        """
        xi, Lambda, A, B = affine_invariant_log_gevd(Q, P)
        ctx.save_for_backward(Lambda, A, B)
        return xi

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the affine-invariant logarithm mapping based on generalized eigenvalue decomposition.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the output of the Riemannian logarithm mapping

        Returns
        -------
        grad_Q : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. Q

        grad_P : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. P
        """
        Lambda, A, B = ctx.saved_tensors
        grad_P, grad_Q = gevd_operation_grad(grad_output, Lambda, A, B, torch.log, inv)
        return grad_Q, grad_P, None


# -----------------------------------
# Affine-invariant parallel transport
# -----------------------------------
def affine_invariant_parallel_transport_transformation_gevd(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant transformation matrix for parallel transport.

    Parameters
    ----------
    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    M : torch.Tensor of shape (n_features, n_features)
        Non-singular matrix

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of Q w.r.t. P

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    Lambda, A, B = gevd(P, Q)
    return (B * torch.sqrt(Lambda).unsqueeze(-2)) @ A, Lambda, A, B


class AffineInvariantParallelTransportTransformationGEVD(Function):
    """
    Affine-invariant transformation matrix for parallel transport.
    """

    @staticmethod
    def forward(ctx, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the affine-invariant transformation matrix for parallel transport.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        P : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Q : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        M : torch.Tensor of shape (n_features, n_features)
            Non-singular matrix
        """
        M, Lambda, A, B = affine_invariant_parallel_transport_transformation_gevd(P, Q)
        ctx.save_for_backward(Lambda, A, B)
        return M

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the affine-invariant transformation matrix for parallel transport.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the affine-invariant transformation matrix for parallel transport

        Returns
        -------
        grad_P : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. input SPD matrix P

        grad_Q : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. input SPD matrix Q
        """
        Lambda, A, B = ctx.saved_tensors
        grad_out_transf = A @ grad_output.transpose(-1, -2) @ B
        grad_out_transf_sym = symmetrize(grad_out_transf)
        grad_out_transf_skew = skew_symmetrize(grad_out_transf)
        aux_mat_Q = _aux_eigh_operation_grad(Lambda, torch.sqrt, sqrt_derivative)
        middle_term_Q = aux_mat_Q * grad_out_transf_sym
        grad_Q = B @ middle_term_Q @ B.transpose(-1, -2)
        _Lambda = torch.sqrt(Lambda)
        middle_term_P = symmetrize(
            grad_out_transf_skew * _Lambda.unsqueeze(-2)
            - middle_term_Q * Lambda.unsqueeze(-2)
        )
        grad_P = B @ middle_term_P @ B.T
        return grad_P, grad_Q


def affine_invariant_parallel_transport_gevd(
    S: torch.Tensor, P: torch.Tensor, Q: torch.Tensor
) -> torch.Tensor:
    """
    Affine-invariant parallel transport of some symmetric (or SPD) matrix S from P to Q.

    Parameters
    ----------
    S : torch.Tensor of shape (n_features, n_features)
        Symmetric (or SPD) matrix

    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    S_transp : torch.Tensor of shape (n_features, n_features)
        Transported symmetric (or SPD) matrix
    """
    M = affine_invariant_parallel_transport_transformation_gevd(P, Q)[0]
    return congruence_rectangular(S, M.transpose(-2, -1))


def AffineInvariantParallelTransportGEVD(
    S: torch.Tensor, P: torch.Tensor, Q: torch.Tensor
) -> torch.Tensor:
    """
    Affine-invariant parallel transport of some symmetric (or SPD) matrix S from P to Q.

    Parameters
    ----------
    S : torch.Tensor of shape (n_features, n_features)
        Symmetric (or SPD) matrix

    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    S_transp : torch.Tensor of shape (n_features, n_features)
        Transported symmetric (or SPD) matrix
    """
    M = AffineInvariantParallelTransportTransformationGEVD.apply(P, Q)
    return CongruenceRectangular.apply(S, M.transpose(-2, -1))
