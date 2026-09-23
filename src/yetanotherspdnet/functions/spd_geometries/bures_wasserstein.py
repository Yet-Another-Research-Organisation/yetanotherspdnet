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
    r"""
    Squared Bures-Wasserstein distance between SPD matrices.

    .. math::

        d_{BW}^2(X_1, X_2) = \operatorname{tr}(X_1) + \operatorname{tr}(X_2)
            - 2\operatorname{tr}\!\Big(\big(X_1^{1/2} X_2 X_1^{1/2}\big)^{1/2}\Big)

    This is the (squared) 2-Wasserstein distance between the zero-mean
    Gaussian distributions with covariances :math:`X_1` and :math:`X_2`.

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
    r"""
    Logarithmic map at the identity under Bures-Wasserstein geometry.

    .. math::

        \mathrm{Log}_I(X) = 2\big(X^{1/2} - I\big)

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
    r"""
    Exponential map at the identity under Bures-Wasserstein geometry.

    .. math::

        \mathrm{Exp}_I(S) = \left(I + \frac{S}{2}\right)^2

    Inverse of :func:`bures_wasserstein_log_identity`.

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
    r"""
    Logarithmic map at a general base point under BW geometry.

    .. math::

        \mathrm{Log}_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2B

    computed via the identity :math:`(BX)^{1/2} = B^{1/2}
    (B^{1/2} X B^{1/2})^{1/2} B^{-1/2}` and
    :math:`(XB)^{1/2} = \big[(BX)^{1/2}\big]^\top`.

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


def _sum_to_shape(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Sum a broadcast gradient back to the shape of the input it came from."""
    while grad.ndim > len(shape):
        grad = grad.sum(dim=0)
    for dim, size in enumerate(shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


class LyapunovSolveSPD(Function):
    r"""
    Solution :math:`Z` of :math:`BZ + ZB = V` for SPD :math:`B`, with an
    implicit backward.

    Differentiating the equation gives :math:`B\,dZ + dZ\,B = dV - (dB\,Z +
    Z\,dB)`, so with :math:`W` solving :math:`BW + WB = \bar Z`:
    :math:`\bar V = W` and :math:`\bar B = -(WZ + ZW)`. Only solves with the
    eigendecomposition of :math:`B` are needed, never the derivative of
    ``torch.linalg.eigh``, so the gradient stays finite when :math:`B` has
    repeated eigenvalues (e.g. a parameter initialized at the identity).
    """

    @staticmethod
    def forward(ctx, base: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        base : torch.Tensor of shape (..., n_features, n_features)
            SPD matrices :math:`B` (broadcast against ``rhs``)

        rhs : torch.Tensor of shape (..., n_features, n_features)
            Right-hand sides :math:`V`

        Returns
        -------
        solution : torch.Tensor of shape (..., n_features, n_features)
            Solutions :math:`Z`
        """
        eigvals, eigvecs = torch.linalg.eigh(base)
        solution = solve_sylvester_SPD(eigvals, eigvecs, rhs)
        ctx.save_for_backward(eigvals, eigvecs, solution)
        ctx.base_shape = base.shape
        return solution

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eigvals, eigvecs, solution = ctx.saved_tensors
        adjoint = solve_sylvester_SPD(eigvals, eigvecs, grad_output)
        solution_t = solution.transpose(-2, -1)
        grad_base = -(adjoint @ solution_t + solution_t @ adjoint)
        grad_base = 0.5 * (grad_base + grad_base.transpose(-2, -1))
        return _sum_to_shape(grad_base, ctx.base_shape), adjoint


def _bures_wasserstein_exp(
    tangent_vec: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    r"""
    Exponential map at a general base point under BW geometry.

    .. math::

        \mathrm{Exp}_B(V) = B + V + Z B Z, \quad\text{where } Z \text{ solves }
        B Z + Z B = V

    (a Lyapunov equation, solved by
    :func:`~yetanotherspdnet.functions.spd_linalg.solve_sylvester_SPD`).
    Inverse of :func:`bures_wasserstein_log`: with :math:`T` the optimal
    transport map from :math:`B` to :math:`X`, :math:`\mathrm{Log}_B(X) =
    TB + BT - 2B` gives :math:`Z = T - I` and :math:`\mathrm{Exp}_B = TBT = X`.
    At :math:`B = I` it reduces to :func:`bures_wasserstein_exp_identity`.

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
    Z = LyapunovSolveSPD.apply(base, tangent_vec)
    return base + tangent_vec + Z @ base @ Z


# ----------------------------------------
# Parallel transport
# ----------------------------------------
def bures_wasserstein_parallel_transport_to_identity(
    tangent_vec: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    r"""
    Parallel transport from *source* to the identity under BW geometry.

    Given :math:`\text{source} = V \operatorname{diag}(\lambda) V^\top`:

    .. math::

        \Gamma_{\text{source}\to I}(S) = V\left[
            \sqrt{\frac{2}{\lambda_i + \lambda_j}} \,(V^\top S V)_{ij}
            \right]_{ij} V^\top

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
    r"""
    Parallel transport from the identity to *target* under BW geometry.

    Given :math:`\text{target} = U \operatorname{diag}(\delta) U^\top`:

    .. math::

        \Gamma_{I\to\text{target}}(S) = U\left[
            \sqrt{\frac{\delta_i + \delta_j}{2}} \,(U^\top S U)_{ij}
            \right]_{ij} U^\top

    Inverse of :func:`bures_wasserstein_parallel_transport_to_identity`.

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


class ParallelTransportFromIdentityBW(Function):
    r"""
    BW transport from the identity, :math:`\Gamma_{I\to G} = \mathcal{A}_G^{1/2}`,
    with a backward that stays finite for repeated eigenvalues of :math:`G`.

    :math:`\mathcal{A}_G(S) = (GS + SG)/2` is the Lyapunov operator; in the
    eigenbasis :math:`u_i` of :math:`G` it is diagonal with entries
    :math:`a_{ij} = (\delta_i + \delta_j)/2`, and the transport multiplies
    :math:`(U^\top S U)_{ij}` by :math:`b_{ij} = \sqrt{a_{ij}}`. Its derivative
    with respect to :math:`G` follows from the Sylvester equation
    :math:`\mathcal{B}\,d\mathcal{B} + d\mathcal{B}\,\mathcal{B} = d\mathcal{A}`
    between operators (:math:`\mathcal{B} = \mathcal{A}^{1/2}`):

    .. math::

        d\tilde W_{ij} = \frac12 \sum_k \frac{\tilde E_{ik} \tilde S_{kj}}{b_{ij} + b_{kj}}
            + \frac12 \sum_k \frac{\tilde S_{ik} \tilde E_{kj}}{b_{ij} + b_{ik}},
        \qquad \tilde E = U^\top dG\, U,

    whose denominators are positive, unlike the :math:`1/(\delta_i - \delta_j)`
    terms of the autograd path through ``torch.linalg.eigh``.
    """

    @staticmethod
    def forward(ctx, tangent_vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tangent_vec : torch.Tensor of shape (..., n_features, n_features)
            Tangent vectors :math:`S` at the identity

        target : torch.Tensor of shape (n_features, n_features)
            Target point :math:`G` (SPD)

        Returns
        -------
        transported : torch.Tensor of shape (..., n_features, n_features)
            Tangent vectors at *target*
        """
        if target.ndim != 2:
            raise ValueError(
                "ParallelTransportFromIdentityBW expects a single target matrix, "
                f"got shape {tuple(target.shape)}"
            )
        eigvals, eigvecs = torch.linalg.eigh(target)
        scale = torch.sqrt((eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)) / 2.0)
        rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
        ctx.save_for_backward(eigvecs, scale, rotated)
        return eigvecs @ (scale * rotated) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eigvecs, scale, rotated = ctx.saved_tensors
        grad_rot = eigvecs.transpose(-2, -1) @ grad_output @ eigvecs
        grad_tangent = eigvecs @ (scale * grad_rot) @ eigvecs.transpose(-2, -1)
        n = scale.shape[-1]
        grad_rot = grad_rot.reshape(-1, n, n)
        rotated = rotated.expand_as(grad_output).reshape(-1, n, n)
        # kernel[p, q, j] = 1 / (b_pj + b_qj)
        kernel = 1.0 / (scale.unsqueeze(1) + scale.unsqueeze(0))
        first = torch.einsum("bpj,bqj->pqj", grad_rot, rotated)
        second = torch.einsum("bip,biq->pqi", rotated, grad_rot)
        grad_rot_target = 0.5 * ((first + second) * kernel).sum(dim=-1)
        grad_target = eigvecs @ grad_rot_target @ eigvecs.transpose(-2, -1)
        grad_target = 0.5 * (grad_target + grad_target.transpose(-2, -1))
        return grad_tangent, grad_target


# ----------------------------------------
# Bures-Wasserstein geodesic (2-sample)
# ----------------------------------------
def bures_wasserstein_geodesic(
    point1: torch.Tensor,
    point2: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    r"""
    Bures-Wasserstein geodesic (closed-form 2-sample weighted mean).

    .. math::

        E_2(X_1, X_2; t) = (1-t)^2 X_1 + t^2 X_2
            + t(1-t)\Big[(X_2 X_1)^{1/2} + (X_1 X_2)^{1/2}\Big]

    where :math:`(X_1 X_2)^{1/2} = X_1^{1/2}
    (X_1^{1/2} X_2 X_1^{1/2})^{1/2} X_1^{-1/2}` and :math:`t \in [0, 1]`.

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
    r"""
    Bures-Wasserstein barycenter (Fréchet mean) via fixed-point iteration.

    .. math::

        G_{k+1} = G_k^{-1/2}\left(\frac{1}{N}\sum_{i=1}^{N}
            \big(G_k^{1/2} X_i G_k^{1/2}\big)^{1/2}\right)^2 G_k^{-1/2}

    the unique fixed point of this map is the barycenter minimizing
    :math:`\sum_i d_{BW}(G, X_i)^2` (see
    :func:`bures_wasserstein_distance_squared`). The initial estimate
    :math:`G_0` is the arithmetic mean of *data*.

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
    r"""
    Scalar standard deviation under the Bures-Wasserstein distance.

    .. math::

        \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_{BW}^2(G, X_i)}

    where :math:`G` is the reference point (typically the BW barycenter)
    and :math:`d_{BW}` is the Bures-Wasserstein distance (see
    :func:`bures_wasserstein_distance_squared`).

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
    r"""
    Center SPD data by parallel-transporting from the barycenter to the identity.

    .. math::

        X_{\text{centered}} = \mathrm{Exp}_I\big(\Gamma_{B\to I}(\mathrm{Log}_B(X))\big)

    composing :func:`bures_wasserstein_log`,
    :func:`bures_wasserstein_parallel_transport_to_identity`, and
    :func:`bures_wasserstein_exp_identity` — the BatchNorm analogue of
    subtracting the mean, but on the SPD manifold.

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
    r"""
    Scale centered SPD data at the identity.

    .. math::

        X_{\text{scaled}} = \mathrm{Exp}_I\!\left(
            \frac{s}{\sqrt{\text{var} + \epsilon}}\, \mathrm{Log}_I(X)\right)
        = \left(I + \frac{s}{\sqrt{\text{var} + \epsilon}}\,
            \big(X^{1/2} - I\big)\right)^2

    the BatchNorm analogue of dividing by the standard deviation and
    multiplying by a learnable scale :math:`s`.

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
    r"""
    Bias SPD data by parallel-transporting from the identity to *bias_point*.

    .. math::

        X_{\text{biased}} = \mathrm{Exp}_G\big(\Gamma_{I\to G}(\mathrm{Log}_I(X))\big)

    where :math:`\mathrm{Exp}_G(V) = G + V + Z G Z` with :math:`G Z + Z G = V`
    (see :func:`bures_wasserstein_parallel_transport_from_identity`).
    The BatchNorm analogue of adding a learnable bias :math:`G`.

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
    # degenerate-safe backward: bias_point is typically a learnable parameter
    # initialized at the identity (all eigenvalues equal)
    transported = ParallelTransportFromIdentityBW.apply(log_at_I, bias_point)
    return _bures_wasserstein_exp(transported, bias_point)
