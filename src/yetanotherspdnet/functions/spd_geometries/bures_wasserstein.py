"""Bures-Wasserstein geometry: geodesic, mean, standard deviation, and transport maps.

Implements the Bures-Wasserstein (BW) metric, geodesic, Frechet mean (barycenter),
scalar standard deviation, and related operations (log/exp maps, parallel transport)
on the manifold of symmetric positive definite matrices.

References
----------
[1] Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between positive
    definite matrices." Expositiones Mathematicae, 2019.
[2] Kobler et al. "Controlling the Fréchet Variance Improves Batch
    Normalization on the Symmetric Positive Definite Manifold." CVPR, 2022.
"""

import math

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.scalar_functions import inv_sqrt

from ..spd_linalg import (
    eigh_operation,
    inv_sqrtm_SPD,
    solve_sylvester_SPD,
    sqrtm_SPD,
)
from .kullback_leibler import arithmetic_mean


# ----------------------------------------
# Bures-Wasserstein squared distance
# ----------------------------------------
def bures_wasserstein_distance_squared(
    point1: torch.Tensor, point2: torch.Tensor
) -> torch.Tensor:
    """
    Squared Bures-Wasserstein distance between SPD matrices.

    d_BW^2(X1, X2) = tr(X1) + tr(X2)
                      - 2 tr( (X1^{1/2} X2 X1^{1/2})^{1/2} )

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    dist_sq : torch.Tensor of shape (...)
        Squared BW distances
    """
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    p1_sqrt = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    inner = p1_sqrt @ point2 @ p1_sqrt
    inner_sqrt = sqrtm_SPD(inner)[0]
    trace_p1 = torch.diagonal(point1, dim1=-2, dim2=-1).sum(-1)
    trace_p2 = torch.diagonal(point2, dim1=-2, dim2=-1).sum(-1)
    trace_inner = torch.diagonal(inner_sqrt, dim1=-2, dim2=-1).sum(-1)
    return trace_p1 + trace_p2 - 2 * trace_inner


# ----------------------------------------
# Log / Exp maps at identity
# ----------------------------------------
def bures_wasserstein_log_identity(
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Logarithmic map at the identity under Bures-Wasserstein geometry.

    Log_I(X) = 2 (X^{1/2} - I)

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    S : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices in the tangent space at I
    """
    X_sqrt = sqrtm_SPD(X)[0]
    eye = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
    return 2 * (X_sqrt - eye)


def bures_wasserstein_exp_identity(
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Exponential map at the identity under Bures-Wasserstein geometry.

    Exp_I(S) = (I + S/2)^2

    Parameters
    ----------
    S : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices in the tangent space at I

    Returns
    -------
    X : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    half = eye + S / 2
    return half @ half


# ----------------------------------------
# Log / Exp maps at general base point
# ----------------------------------------
def bures_wasserstein_log(X: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map at a general base point under BW geometry.

    Log_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2 B

    Uses the identity  (BX)^{1/2} = B^{1/2} (B^{1/2} X B^{1/2})^{1/2} B^{-1/2}
    and (XB)^{1/2} = [(BX)^{1/2}]^T.

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    base : torch.Tensor of shape (..., n_features, n_features) or (n_features, n_features)
        Base point (SPD matrix)

    Returns
    -------
    tangent : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at *base*
    """
    eigvals_B, eigvecs_B = torch.linalg.eigh(base)
    B_sqrt = eigh_operation(eigvals_B, eigvecs_B, torch.sqrt)
    B_inv_sqrt = eigh_operation(eigvals_B, eigvecs_B, inv_sqrt)
    S = B_sqrt @ X @ B_sqrt  # B^{1/2} X B^{1/2}  (SPD)
    S_sqrt = sqrtm_SPD(S)[0]
    BX_sqrt = B_sqrt @ S_sqrt @ B_inv_sqrt  # (BX)^{1/2}
    XB_sqrt = BX_sqrt.transpose(-2, -1)  # (XB)^{1/2}
    return XB_sqrt + BX_sqrt - 2 * base


def _bures_wasserstein_exp(
    tangent_vec: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    """
    Exponential map at a general base point under BW geometry.

    Exp_B(V) = B + V + Z^2
    where Z solves the Sylvester equation  B^{1/2} Z + Z B^{1/2} = V.

    Parameters
    ----------
    tangent_vec : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at *base*

    base : torch.Tensor of shape (..., n_features, n_features) or (n_features, n_features)
        Base point (SPD matrix)

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    eigvals_B, eigvecs_B = torch.linalg.eigh(base)
    Z = solve_sylvester_SPD(torch.sqrt(eigvals_B), eigvecs_B, tangent_vec)
    return base + tangent_vec + Z @ Z


# ----------------------------------------
# Parallel transport
# ----------------------------------------
def bures_wasserstein_parallel_transport_to_identity(
    tangent_vec: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    Parallel transport from *source* to the identity under BW geometry.

    Given source = V diag(lambda) V^T:
        Gamma_{source -> I}(S)
            = V [ sqrt(2 / (lambda_i + lambda_j))
                  * (V^T S V)_{ij} ] V^T

    Parameters
    ----------
    tangent_vec : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at *source*

    source : torch.Tensor of shape (..., n_features, n_features) or (n_features, n_features)
        Source point (SPD matrix)

    Returns
    -------
    transported : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at identity
    """
    eigvals, eigvecs = torch.linalg.eigh(source)
    lam_sum = eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)
    scale = torch.sqrt(2.0 / lam_sum)
    rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
    return eigvecs @ (scale * rotated) @ eigvecs.transpose(-2, -1)


def bures_wasserstein_parallel_transport_from_identity(
    tangent_vec: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Parallel transport from the identity to *target* under BW geometry.

    Given target = U diag(delta) U^T:
        Gamma_{I -> target}(S)
            = U [ sqrt((delta_i + delta_j) / 2)
                  * (U^T S U)_{ij} ] U^T

    Parameters
    ----------
    tangent_vec : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at identity

    target : torch.Tensor of shape (..., n_features, n_features) or (n_features, n_features)
        Target point (SPD matrix)

    Returns
    -------
    transported : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at *target*
    """
    eigvals, eigvecs = torch.linalg.eigh(target)
    delta_sum = eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)
    scale = torch.sqrt(delta_sum / 2.0)
    rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
    return eigvecs @ (scale * rotated) @ eigvecs.transpose(-2, -1)


# ----------------------------------------
# Bures-Wasserstein geodesic (2-sample)
# ----------------------------------------
def bures_wasserstein_geodesic(
    point1: torch.Tensor,
    point2: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """
    Bures-Wasserstein geodesic (closed-form 2-sample weighted mean).

    E_2(X1, X2; t) = (1-t)^2 X1 + t^2 X2
                      + t(1-t) [(X2 X1)^{1/2} + (X1 X2)^{1/2}]

    where (X1 X2)^{1/2} = X1^{1/2} (X1^{1/2} X2 X1^{1/2})^{1/2} X1^{-1/2}.

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float | torch.Tensor
        Interpolation parameter in [0, 1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices on the geodesic
    """
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    p1_sqrt = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    p1_inv_sqrt = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    M = p1_sqrt @ point2 @ p1_sqrt  # X1^{1/2} X2 X1^{1/2}
    M_sqrt = sqrtm_SPD(M)[0]
    cross = p1_sqrt @ M_sqrt @ p1_inv_sqrt  # (X1 X2)^{1/2}
    cross_T = cross.transpose(-2, -1)  # (X2 X1)^{1/2}
    return (1 - t) ** 2 * point1 + t**2 * point2 + t * (1 - t) * (cross + cross_T)


# ----------------------------------------
# Bures-Wasserstein mean (barycenter)
# ----------------------------------------
def bures_wasserstein_mean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
    """
    Bures-Wasserstein barycenter (Frechet mean) via fixed-point iteration.

    Update rule:
        G_{k+1} = G^{-1/2}
                   ( (1/N) sum_i (G^{1/2} X_i G^{1/2})^{1/2} )^2
                   G^{-1/2}

    The initial estimate is the arithmetic mean of *data*.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ``...`` axes.

    n_iterations : int
        Number of fixed-point iterations, by default 1

    Returns
    -------
    barycenter : torch.Tensor of shape (n_features, n_features)
        BW barycenter
    """
    if data.ndim == 2:
        return data
    G = arithmetic_mean(data)
    for _ in range(n_iterations):
        eigvals_G, eigvecs_G = torch.linalg.eigh(G)
        G_sqrt = eigh_operation(eigvals_G, eigvecs_G, torch.sqrt)
        G_inv_sqrt = eigh_operation(eigvals_G, eigvecs_G, inv_sqrt)
        whitened = G_sqrt @ data @ G_sqrt
        whitened_sqrt = sqrtm_SPD(whitened)[0]
        T = arithmetic_mean(whitened_sqrt)
        G = G_inv_sqrt @ (T @ T) @ G_inv_sqrt
    return G


def BuresWassersteinMean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
    """
    Bures-Wasserstein barycenter (Frechet mean) -- CamelCase wrapper.

    Since the fixed-point iteration composes differentiable operations
    (eigh, matmul, sqrtm), autograd handles the backward automatically.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ``...`` axes.

    n_iterations : int
        Number of fixed-point iterations, by default 1

    Returns
    -------
    barycenter : torch.Tensor of shape (n_features, n_features)
        BW barycenter
    """
    return bures_wasserstein_mean(data, n_iterations=n_iterations)


# ----------------------------------------
# Scalar standard deviation
# ----------------------------------------
def bures_wasserstein_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    """
    Scalar standard deviation under the Bures-Wasserstein distance.

    std = sqrt( (1/N) sum_i d_BW^2(B, X_i) )

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    reference_point : torch.Tensor of shape (n_features, n_features)
        SPD matrix (some kind of mean of data)

    Returns
    -------
    scalar_std : torch.Tensor of shape ()
        Scalar standard deviation
    """
    n_matrices = math.prod(data.shape[:-2])
    eigvals_B, eigvecs_B = torch.linalg.eigh(reference_point)
    B_sqrt = eigh_operation(eigvals_B, eigvecs_B, torch.sqrt)
    inner = B_sqrt @ data @ B_sqrt
    inner_sqrt = sqrtm_SPD(inner)[0]
    trace_B = torch.diagonal(reference_point, dim1=-2, dim2=-1).sum(-1)
    trace_X = torch.diagonal(data, dim1=-2, dim2=-1).sum(-1)
    trace_inner = torch.diagonal(inner_sqrt, dim1=-2, dim2=-1).sum(-1)
    dist_sq = trace_B + trace_X - 2 * trace_inner
    variance = dist_sq.sum() / n_matrices
    return torch.sqrt(variance)


class BuresWassersteinStdScalar(Function):
    """
    Scalar standard deviation under the Bures-Wasserstein distance
    (manual backward).
    """

    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        reference_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the BW scalar standard deviation

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        reference_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix (some kind of mean of data)

        Returns
        -------
        scalar_std : torch.Tensor of shape ()
            Scalar standard deviation
        """
        n_matrices = math.prod(data.shape[:-2])

        eigvals_B, eigvecs_B = torch.linalg.eigh(reference_point)
        B_sqrt = eigh_operation(eigvals_B, eigvecs_B, torch.sqrt)

        inner = B_sqrt @ data @ B_sqrt
        eigvals_inner, eigvecs_inner = torch.linalg.eigh(inner)
        inner_sqrt = eigh_operation(eigvals_inner, eigvecs_inner, torch.sqrt)

        trace_B = torch.diagonal(reference_point, dim1=-2, dim2=-1).sum(-1)
        trace_X = torch.diagonal(data, dim1=-2, dim2=-1).sum(-1)
        trace_inner = torch.diagonal(inner_sqrt, dim1=-2, dim2=-1).sum(-1)

        dist_sq = trace_B + trace_X - 2 * trace_inner
        variance = dist_sq.sum() / n_matrices
        scalar_std = torch.sqrt(variance)

        ctx.n_matrices = n_matrices
        ctx.save_for_backward(
            data,
            reference_point,
            B_sqrt,
            eigvals_inner,
            eigvecs_inner,
            scalar_std,
        )
        return scalar_std

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the BW scalar standard deviation

        Uses:
            d(d_BW^2(B,X)) / dX = I - B^{1/2} (B^{1/2} X B^{1/2})^{-1/2} B^{1/2}
            d(d_BW^2(B,X)) / dB = I - X^{1/2} (X^{1/2} B X^{1/2})^{-1/2} X^{1/2}

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape ()
            Gradient of the loss with respect to the scalar std

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input data

        grad_input_reference_point : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the reference point
        """
        n_matrices = ctx.n_matrices
        (
            data,
            reference_point,
            B_sqrt,
            eigvals_inner,
            eigvecs_inner,
            scalar_std,
        ) = ctx.saved_tensors

        eye = torch.eye(
            data.shape[-1],
            dtype=data.dtype,
            device=data.device,
        )

        # (B^{1/2} X B^{1/2})^{-1/2}
        inner_inv_sqrt = eigh_operation(eigvals_inner, eigvecs_inner, inv_sqrt)

        # --- gradient wrt data ---
        # d(d^2)/dX = I - B^{1/2} inner^{-1/2} B^{1/2}
        grad_dist_sq_data = eye - B_sqrt @ inner_inv_sqrt @ B_sqrt
        grad_input_data = (
            grad_output * grad_dist_sq_data / (2 * n_matrices * scalar_std)
        )

        # --- gradient wrt reference_point ---
        # d(d^2)/dB = I - X^{1/2} (X^{1/2} B X^{1/2})^{-1/2} X^{1/2}
        data_sqrtm = sqrtm_SPD(data)[0]
        inner_B = data_sqrtm @ reference_point @ data_sqrtm
        inner_B_inv_sqrt = inv_sqrtm_SPD(inner_B)[0]
        grad_dist_sq_B = eye - data_sqrtm @ inner_B_inv_sqrt @ data_sqrtm
        grad_input_G = grad_output * arithmetic_mean(grad_dist_sq_B) / (2 * scalar_std)

        return grad_input_data, grad_input_G


# ----------------------------------------
# Centering / Scaling / Biasing pipelines
# ----------------------------------------
def bures_wasserstein_center(
    data: torch.Tensor, barycenter: torch.Tensor
) -> torch.Tensor:
    """
    Center SPD data by parallel-transporting from the barycenter to the identity.

    X_centered = Exp_I( Gamma_{B -> I}( Log_B(X) ) )

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    barycenter : torch.Tensor of shape (n_features, n_features)
        BW barycenter

    Returns
    -------
    centered : torch.Tensor of shape (..., n_features, n_features)
        Centered SPD matrices (around the identity)
    """
    log_at_bary = bures_wasserstein_log(data, barycenter)
    transported = bures_wasserstein_parallel_transport_to_identity(
        log_at_bary, barycenter
    )
    return bures_wasserstein_exp_identity(transported)


def bures_wasserstein_scale(
    data: torch.Tensor,
    variance: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Scale centered SPD data at the identity.

    X_scaled = Exp_I( s / sqrt(var + eps) * Log_I(X) )
             = ( I + s / sqrt(var + eps) * (X^{1/2} - I) )^2

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Centered SPD matrices (around the identity)

    variance : torch.Tensor of shape ()
        Frechet variance

    shift : torch.Tensor of shape ()
        Learnable scaling parameter

    eps : float
        Small constant for numerical stability, by default 1e-5

    Returns
    -------
    scaled : torch.Tensor of shape (..., n_features, n_features)
        Scaled SPD matrices
    """
    factor = shift / torch.sqrt(variance + eps)
    log_at_I = bures_wasserstein_log_identity(data)
    return bures_wasserstein_exp_identity(factor * log_at_I)


def bures_wasserstein_bias(
    data: torch.Tensor, bias_point: torch.Tensor
) -> torch.Tensor:
    """
    Bias SPD data by parallel-transporting from the identity to *bias_point*.

    X_biased = Exp_G( Gamma_{I -> G}( Log_I(X) ) )

    where Exp_G(V) = G + V + Z^2 with G^{1/2} Z + Z G^{1/2} = V.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices around the identity

    bias_point : torch.Tensor of shape (n_features, n_features)
        Learned bias (SPD matrix)

    Returns
    -------
    biased : torch.Tensor of shape (..., n_features, n_features)
        Biased SPD matrices
    """
    log_at_I = bures_wasserstein_log_identity(data)
    transported = bures_wasserstein_parallel_transport_from_identity(
        log_at_I, bias_point
    )
    return _bures_wasserstein_exp(transported, bias_point)
