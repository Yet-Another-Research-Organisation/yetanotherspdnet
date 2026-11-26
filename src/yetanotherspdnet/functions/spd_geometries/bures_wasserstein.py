r"""Bures-Wasserstein geometry: geodesic, mean, standard deviation, and transport maps.

Implements the Bures-Wasserstein (BW) metric, geodesic, Frechet mean (barycenter),
scalar standard deviation, and related operations (log/exp maps, parallel transport)
on the manifold of symmetric positive definite matrices.

References
----------
[1] Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between positive
    definite matrices." Expositiones Mathematicae, 2019.
[2] Kobler et al. "Controlling the Fréchet Variance Improves Batch
    Normalization on the Symmetric Positive Definite Manifold." CVPR, 2022.

**Cross terms through the polar factor.** Every operation of this geometry needs the cross term
:math:`(B^{1/2} X B^{1/2})^{1/2}`. It is not computed by an eigendecomposition
of :math:`B^{1/2} X B^{1/2}`, whose condition number is up to
:math:`\kappa(X)\kappa(B)` (the small eigenvalues lose all their digits beyond
:math:`\kappa \approx 10^{8}` in float64 and :math:`10^{4}` in float32), but
from the SVD of :math:`A = X^{1/2} B^{1/2} = U \Sigma V^\top`, whose condition
number is at most :math:`\sqrt{\kappa(X)\kappa(B)}`:
:math:`(B^{1/2} X B^{1/2})^{1/2} = V \Sigma V^\top = W^\top A` and
:math:`\operatorname{tr}\big((B^{1/2} X B^{1/2})^{1/2}\big) = \sum_i \sigma_i`,
with :math:`W = U V^\top` the orthogonal polar factor of :math:`A`
(:func:`~yetanotherspdnet.functions.spd_linalg.polar_factor`). The formulas
are then written without any inverse square root:

- :math:`(BX)^{1/2} = B^{1/2} W^\top X^{1/2}` (so
  :math:`\mathrm{Log}_B(X) = B^{1/2} W^\top X^{1/2} + X^{1/2} W B^{1/2} - 2B`);
- :math:`d_{BW}^2(X, B) = \lVert X^{1/2} W - B^{1/2} \rVert_F^2` (the
  Procrustes form: no cancellation when :math:`X \approx B`);
- the geodesic is :math:`Y_t Y_t^\top` with
  :math:`Y_t = (1 - t) X_1^{1/2} + t X_2^{1/2} W`;
- the fixed-point step of the mean is :math:`G \leftarrow C C^\top` with
  :math:`C = \frac1N \sum_i X_i^{1/2} W_i`.

The manual backward of :math:`W` (:class:`~yetanotherspdnet.functions.spd_linalg.PolarFactor`)
divides by sums of singular values, so it stays finite at repeated singular values.
"""

import math
from collections.abc import Callable

import torch
from torch.autograd import Function

from yetanotherspdnet.functions._constants import BW_SCALE_EPS
from yetanotherspdnet.functions.scalar_functions import (
    sqrt_derivative,
    std_from_variance,
    zero_where_zero_std,
)

from ..spd_linalg import (
    EighClamp,
    PolarFactor,
    SqrtmSPD,
    eigh_clamp,
    eigh_operation_grad,
    polar_factor,
    solve_sylvester_SPD,
    sqrtm_SPD,
    symmetric_eigh,
    symmetric_eigvalsh,
    symmetrize,
)
from .euclidean import arithmetic_mean


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

    Notes
    -----
    Computed in the Procrustes form
    :math:`\lVert X_2^{1/2} W - X_1^{1/2} \rVert_F^2`, with :math:`W` the
    polar factor of :math:`X_2^{1/2} X_1^{1/2}` (see the module docstring):
    nonnegative by construction and free of the cancellation of the trace
    form when the points are close. :math:`W` minimizes
    :math:`\lVert X_2^{1/2} Q - X_1^{1/2} \rVert_F` over orthogonal
    :math:`Q`, so its derivative does not contribute to the gradient
    (envelope theorem): it is computed without gradient.
    """
    p1_sqrt = sqrtm_SPD(point1)[0]
    p2_sqrt = sqrtm_SPD(point2)[0]
    with torch.no_grad():
        orthogonal = polar_factor(p2_sqrt @ p1_sqrt)[0]
    residual = p2_sqrt @ orthogonal - p1_sqrt
    return (residual**2).sum(dim=(-2, -1))


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

    computed as :math:`(BX)^{1/2} = B^{1/2} W^\top X^{1/2}` with :math:`W`
    the polar factor of :math:`X^{1/2} B^{1/2}` (see the module docstring),
    and :math:`(XB)^{1/2} = \big[(BX)^{1/2}\big]^\top`.

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
    return _log_from_roots(sqrtm_SPD(X)[0], sqrtm_SPD(base)[0], base, polar_factor)


def _log_from_roots(
    X_sqrt: torch.Tensor,
    B_sqrt: torch.Tensor,
    base: torch.Tensor,
    polar: Callable,
) -> torch.Tensor:
    """Log_B(X) = B^{1/2} W^T X^{1/2} + transpose - 2B, W = polar(X^{1/2} B^{1/2})."""
    orthogonal = _first(polar(X_sqrt @ B_sqrt))
    BX_sqrt = B_sqrt @ orthogonal.transpose(-2, -1) @ X_sqrt  # (BX)^{1/2}
    return BX_sqrt + BX_sqrt.transpose(-2, -1) - 2 * base


def _first(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """First output of polar_factor (a tuple) or PolarFactor.apply (a tensor)."""
    return output[0] if isinstance(output, tuple) else output


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
    Solve the Lyapunov equation BZ + ZB = V, with an implicit backward.

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
        Solve :math:`BZ + ZB = V`.

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
        eigvals, eigvecs = symmetric_eigh(base)
        solution = solve_sylvester_SPD(eigvals, eigvecs, rhs)
        ctx.save_for_backward(eigvals, eigvecs, solution)
        ctx.base_shape = base.shape
        return solution

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradients with respect to the base B and the right-hand sides V."""
        eigvals, eigvecs, solution = ctx.saved_tensors
        adjoint = solve_sylvester_SPD(eigvals, eigvecs, grad_output)
        solution_t = solution.transpose(-2, -1)
        grad_base = -(adjoint @ solution_t + solution_t @ adjoint)
        grad_base = symmetrize(grad_base)
        return _sum_to_shape(grad_base, ctx.base_shape), adjoint


def _bures_wasserstein_exp(
    tangent_vec: torch.Tensor,
    base: torch.Tensor,
    min_transport_eigval: float | None = None,
    use_autograd: bool = False,
) -> torch.Tensor:
    r"""
    Exponential map at a general base point under BW geometry.

    .. math::

        \mathrm{Exp}_B(V) = B + V + Z B Z = (I + Z) B (I + Z), \quad\text{where }
        Z \text{ solves } B Z + Z B = V

    (a Lyapunov equation, solved by
    :func:`~yetanotherspdnet.functions.spd_linalg.solve_sylvester_SPD`).
    Inverse of :func:`bures_wasserstein_log`: with :math:`T` the optimal
    transport map from :math:`B` to :math:`X`, :math:`\mathrm{Log}_B(X) =
    TB + BT - 2B` gives :math:`Z = T - I` and :math:`\mathrm{Exp}_B = TBT = X`.
    At :math:`B = I` it reduces to :func:`bures_wasserstein_exp_identity`.

    **Domain.** :math:`T = I + Z` is the optimal (Brenier) transport map from
    :math:`B` to :math:`\mathrm{Exp}_B(V)`, which is positive definite: the BW
    exponential is a geodesic, and a diffeomorphism onto the SPD cone, only on
    :math:`\{V : I + Z \succ 0\}`. Outside, :math:`TBT` folds back into the
    cone, and on the boundary it is singular. With ``min_transport_eigval``
    :math:`\tau > 0`, :math:`T` is projected onto :math:`\{T \succeq \tau I\}`
    by clamping its eigenvalues,
    :math:`T_+ = T + U \operatorname{diag}\big((\tau - \lambda)_+\big) U^\top`,
    so that the result stays in the open cone with
    :math:`\lambda_{\min}(T_+ B T_+) \ge \tau^2 \lambda_{\min}(B)`. The
    correction is exactly zero (the result is unchanged, bit for bit) when
    :math:`T \succ \tau I`.

    Parameters
    ----------
    tangent_vec : torch.Tensor of shape (..., n_features, n_features)
        Tangent vectors at *base*

    base : torch.Tensor of shape (..., n_features, n_features) or (n_features, n_features)
        Base point (SPD matrix)

    min_transport_eigval : float or None, optional
        Lower bound :math:`\tau` of the eigenvalues of the transport map
        :math:`I + Z` (dimensionless). Default is None (no projection)

    use_autograd : bool, optional
        Differentiate the projection with autograd through ``torch.linalg.eigh``
        rather than with the manual backward of
        :class:`~yetanotherspdnet.functions.spd_linalg.EighClamp`.
        Default is False

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    Z = LyapunovSolveSPD.apply(base, tangent_vec)
    point = base + tangent_vec + Z @ base @ Z
    if min_transport_eigval is None:
        return point
    eye = torch.eye(Z.shape[-1], dtype=Z.dtype, device=Z.device)
    # (tau - lambda)_+ of T = I + Z as the ReLU of tau I - T: exactly zero
    # (no rounding) when every eigenvalue of T is above tau
    deficit = symmetrize((min_transport_eigval - 1.0) * eye - Z)
    if use_autograd:
        # autograd through eigh is undefined at repeated eigenvalues even where
        # the correction vanishes: only differentiated when the projection acts
        with torch.no_grad():
            active = bool((symmetric_eigvalsh(deficit)[..., -1] > 0).any())
        if not active:
            return point
        correction = eigh_clamp(deficit, 0.0)[0]
    else:
        correction = EighClamp.apply(deficit, 0.0, None)
    # T_+ B T_+ = T B T + D B T + T B D + D B D, with D the correction
    transport = eye + Z
    cross = correction @ base @ transport
    return point + cross + cross.transpose(-2, -1) + correction @ base @ correction


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
    eigvals, eigvecs = symmetric_eigh(source)
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
    eigvals, eigvecs = symmetric_eigh(target)
    delta_sum = eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)
    scale = torch.sqrt(delta_sum / 2.0)
    rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
    return eigvecs @ (scale * rotated) @ eigvecs.transpose(-2, -1)


class ParallelTransportFromIdentityBW(Function):
    r"""
    Bures-Wasserstein parallel transport from the identity.

    :math:`\Gamma_{I\to G} = \mathcal{A}_G^{1/2}`, with a backward that stays finite for
    repeated eigenvalues of :math:`G`.

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
        Transport tangent vectors from the identity to *target*.

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
        eigvals, eigvecs = symmetric_eigh(target)
        scale = torch.sqrt((eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)) / 2.0)
        rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
        ctx.save_for_backward(eigvecs, scale, rotated)
        return eigvecs @ (scale * rotated) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradients with respect to the tangent vectors and the target."""
        eigvecs, scale, rotated = ctx.saved_tensors
        # d sqrt(a) between Lyapunov eigenvalues: 1 / (b_p + b_q)
        kernel = 1.0 / (scale.unsqueeze(1) + scale.unsqueeze(0))
        return _lyapunov_transport_backward(
            grad_output, eigvecs, scale, rotated, kernel
        )


def _lyapunov_transport_backward(
    grad_output: torch.Tensor,
    eigvecs: torch.Tensor,
    multiplier: torch.Tensor,
    rotated: torch.Tensor,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Backward of a transport :math:`S \mapsto \varphi(\mathcal{A}_P)(S)`.

    :math:`\mathcal{A}_P(S) = (PS + SP)/2` has eigenvalues
    :math:`a_{ij} = (\lambda_i + \lambda_j)/2` in the eigenbasis :math:`U` of
    :math:`P`; the transport multiplies :math:`\tilde S = U^\top S U` entrywise by
    ``multiplier`` :math:`= \varphi(a)`, and its derivative with respect to
    :math:`P` is the Daleckii-Krein formula on the operator :math:`\mathcal{A}_P`,
    with ``kernel[p, q, j]`` the divided difference
    :math:`\varphi[a_{pj}, a_{qj}]`.

    Parameters
    ----------
    grad_output : torch.Tensor of shape (..., n, n)
        Gradient with respect to the transported vectors
    eigvecs : torch.Tensor of shape (n, n)
        Eigenvectors :math:`U` of :math:`P`
    multiplier : torch.Tensor of shape (n, n)
        :math:`\varphi(a_{ij})`
    rotated : torch.Tensor of shape (..., n, n)
        :math:`U^\top S U`
    kernel : torch.Tensor of shape (n, n, n)
        :math:`\varphi[a_{pj}, a_{qj}]`

    Returns
    -------
    grad_tangent : torch.Tensor of shape (..., n, n)
        Gradient with respect to the tangent vectors
    grad_point : torch.Tensor of shape (n, n)
        Gradient with respect to :math:`P`
    """
    grad_rot = eigvecs.transpose(-2, -1) @ grad_output @ eigvecs
    grad_tangent = eigvecs @ (multiplier * grad_rot) @ eigvecs.transpose(-2, -1)
    n = multiplier.shape[-1]
    grad_rot = grad_rot.reshape(-1, n, n)
    rotated = rotated.expand_as(grad_output).reshape(-1, n, n)
    first = torch.einsum("bpj,bqj->pqj", grad_rot, rotated)
    second = torch.einsum("bip,biq->pqi", rotated, grad_rot)
    grad_rot_point = 0.5 * ((first + second) * kernel).sum(dim=-1)
    grad_point = eigvecs @ grad_rot_point @ eigvecs.transpose(-2, -1)
    return grad_tangent, symmetrize(grad_point)


class ParallelTransportToIdentityBW(Function):
    r"""
    Bures-Wasserstein parallel transport to the identity, with a manual backward.

    :math:`\Gamma_{B\to I} = \mathcal{A}_B^{-1/2}`, the inverse of
    :class:`ParallelTransportFromIdentityBW`: in the eigenbasis of :math:`B` it
    multiplies :math:`(U^\top S U)_{ij}` by :math:`1 / b_{ij}`,
    :math:`b_{ij} = \sqrt{(\lambda_i + \lambda_j)/2}`. Its derivative with
    respect to :math:`B` uses the divided difference of :math:`a^{-1/2}`,
    :math:`-1 / (b_p b_q (b_p + b_q))`, whose denominators are positive (no
    :math:`1/(\lambda_i - \lambda_j)` term, finite at repeated eigenvalues).
    """

    @staticmethod
    def forward(ctx, tangent_vec: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Transport tangent vectors from *source* to the identity.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        tangent_vec : torch.Tensor of shape (..., n_features, n_features)
            Tangent vectors at *source*

        source : torch.Tensor of shape (n_features, n_features)
            Source point :math:`B` (SPD)

        Returns
        -------
        transported : torch.Tensor of shape (..., n_features, n_features)
            Tangent vectors at the identity
        """
        if source.ndim != 2:
            raise ValueError(
                "ParallelTransportToIdentityBW expects a single source matrix, "
                f"got shape {tuple(source.shape)}"
            )
        eigvals, eigvecs = symmetric_eigh(source)
        scale = torch.sqrt((eigvals.unsqueeze(-1) + eigvals.unsqueeze(-2)) / 2.0)
        rotated = eigvecs.transpose(-2, -1) @ tangent_vec @ eigvecs
        ctx.save_for_backward(eigvecs, scale, rotated)
        return eigvecs @ (rotated / scale) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradients with respect to the tangent vectors and the source."""
        eigvecs, scale, rotated = ctx.saved_tensors
        b_p, b_q = scale.unsqueeze(1), scale.unsqueeze(0)
        kernel = -1.0 / (b_p * b_q * (b_p + b_q))
        return _lyapunov_transport_backward(
            grad_output, eigvecs, 1.0 / scale, rotated, kernel
        )


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

    where :math:`(X_1 X_2)^{1/2} = X_1^{1/2} W^\top X_2^{1/2}` and
    :math:`t \in [0, 1]`, :math:`W` the polar factor of
    :math:`X_2^{1/2} X_1^{1/2}`. Computed as :math:`Y_t Y_t^\top` with
    :math:`Y_t = (1 - t) X_1^{1/2} + t X_2^{1/2} W` (positive semidefinite by
    construction).

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
    return _geodesic_from_roots(
        sqrtm_SPD(point1)[0], sqrtm_SPD(point2)[0], t, polar_factor
    )


def _geodesic_from_roots(
    p1_sqrt: torch.Tensor,
    p2_sqrt: torch.Tensor,
    t: float | torch.Tensor,
    polar: Callable,
) -> torch.Tensor:
    """Y_t Y_t^T, Y_t = (1 - t) X1^{1/2} + t X2^{1/2} W, W = polar(X2^{1/2} X1^{1/2})."""
    orthogonal = _first(polar(p2_sqrt @ p1_sqrt))
    factor = (1 - t) * p1_sqrt + t * (p2_sqrt @ orthogonal)
    return symmetrize(factor @ factor.transpose(-2, -1))


# ----------------------------------------
# Bures-Wasserstein mean (barycenter)
# ----------------------------------------
def bures_wasserstein_mean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
    r"""
    Bures-Wasserstein barycenter (Fréchet mean) via fixed-point iteration.

    .. math::

        G_{k+1} = G_k^{-1/2}\left(\frac{1}{N}\sum_{i=1}^{N}
            \big(G_k^{1/2} X_i G_k^{1/2}\big)^{1/2}\right)^2 G_k^{-1/2}
            = C_k C_k^\top, \qquad
            C_k = \frac{1}{N}\sum_{i=1}^{N} X_i^{1/2} W_i

    with :math:`W_i` the polar factor of :math:`X_i^{1/2} G_k^{1/2}` (see the
    module docstring; the second form needs no inverse square root). The
    unique fixed point of this map is the barycenter minimizing
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
    return _mean_fixed_point(
        data,
        sqrtm_SPD(data)[0],
        n_iterations,
        lambda x: sqrtm_SPD(x)[0],
        polar_factor,
    )


def _mean_fixed_point(
    data: torch.Tensor,
    data_sqrt: torch.Tensor,
    n_iterations: int,
    sqrtm: Callable,
    polar: Callable,
) -> torch.Tensor:
    """Fixed-point iterations G <- C C^T, C = mean_i X_i^{1/2} W_i, from the arithmetic mean."""
    G = arithmetic_mean(data)
    for _ in range(n_iterations):
        orthogonal = _first(polar(data_sqrt @ sqrtm(G)))
        # not arithmetic_mean, which symmetrizes: C is not symmetric
        C = torch.mean(data_sqrt @ orthogonal, dim=tuple(range(data.ndim - 2)))
        G = symmetrize(C @ C.transpose(-2, -1))
    return G


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
    and :math:`d_{BW}` is the Bures-Wasserstein distance, computed in the
    Procrustes form of :func:`bures_wasserstein_distance_squared`.

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
    dist_sq = bures_wasserstein_distance_squared(reference_point, data)
    return std_from_variance(dist_sq.sum() / n_matrices)


class BuresWassersteinStdScalar(Function):
    r"""
    Scalar standard deviation under the Bures-Wasserstein distance (manual backward).

    Forward as :func:`bures_wasserstein_std_scalar`:
    :math:`\sigma^2 = \frac1N \sum_i \lVert R_i \rVert_F^2`, with
    :math:`R_i = X_i^{1/2} W_i - B^{1/2}` and :math:`W_i` the polar factor of
    :math:`X_i^{1/2} B^{1/2}`. Since :math:`W_i` minimizes
    :math:`\lVert X_i^{1/2} Q - B^{1/2} \rVert_F` over orthogonal :math:`Q`, its
    variation does not contribute (envelope theorem), and

    .. math::

        \frac{\partial \sigma}{\partial X_i^{1/2}} = \frac{R_i W_i^\top}{N\sigma},
        \qquad
        \frac{\partial \sigma}{\partial B^{1/2}} = -\frac{\sum_i R_i}{N\sigma},

    pulled back through the square roots with the Daleckii-Krein formula
    (divided differences :math:`1 / (\sqrt{\lambda_i} + \sqrt{\lambda_j})`).
    This equals the closed form :math:`\partial d^2 / \partial X = I -
    B^{1/2} (B^{1/2} X B^{1/2})^{-1/2} B^{1/2}` without inverting anything.
    """

    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        reference_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the BW scalar standard deviation.

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
        X_sqrt, eigvals_X, eigvecs_X = sqrtm_SPD(data)
        B_sqrt, eigvals_B, eigvecs_B = sqrtm_SPD(reference_point)
        orthogonal = polar_factor(X_sqrt @ B_sqrt)[0]
        residual = X_sqrt @ orthogonal - B_sqrt
        scalar_std = std_from_variance((residual**2).sum() / n_matrices)
        ctx.n_matrices = n_matrices
        ctx.save_for_backward(
            eigvals_X, eigvecs_X, eigvals_B, eigvecs_B, orthogonal, residual, scalar_std
        )
        return scalar_std

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the BW scalar standard deviation.

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
        (
            eigvals_X,
            eigvecs_X,
            eigvals_B,
            eigvecs_B,
            orthogonal,
            residual,
            scalar_std,
        ) = ctx.saved_tensors
        n_features = residual.shape[-1]
        factor = grad_output / (ctx.n_matrices * scalar_std)
        grad_X_sqrt = factor * residual @ orthogonal.transpose(-2, -1)
        grad_B_sqrt = -factor * residual.reshape(-1, n_features, n_features).sum(0)
        grad_input_data = eigh_operation_grad(
            grad_X_sqrt, eigvals_X, eigvecs_X, torch.sqrt, sqrt_derivative
        )
        grad_input_B = eigh_operation_grad(
            grad_B_sqrt, eigvals_B, eigvecs_B, torch.sqrt, sqrt_derivative
        )
        # std = 0: zero subgradient instead of 0/0
        return (
            zero_where_zero_std(grad_input_data, scalar_std),
            zero_where_zero_std(grad_input_B, scalar_std),
        )


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
    eps: float = BW_SCALE_EPS,
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
    data: torch.Tensor,
    bias_point: torch.Tensor,
    min_transport_eigval: float | None = None,
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

    min_transport_eigval : float or None, optional
        Lower bound of the eigenvalues of the transport map of
        :math:`\mathrm{Exp}_G` (see :func:`_bures_wasserstein_exp`): the
        transported vector :math:`\Gamma_{I\to G}(\mathrm{Log}_I(X))` can leave
        the domain of :math:`\mathrm{Exp}_G` (the transport is an isometry, but
        the domain is not invariant: it is scaled by about
        :math:`\sqrt{\lambda_{\min}(G)}` in the directions of the small
        eigenvalues of :math:`G`). Default is None (no projection)

    Returns
    -------
    biased : torch.Tensor of shape (..., n_features, n_features)
        Biased SPD matrices
    """
    log_at_I = bures_wasserstein_log_identity(data)
    # degenerate-safe backward: bias_point is typically a learnable parameter
    # initialized at the identity (all eigenvalues equal)
    transported = ParallelTransportFromIdentityBW.apply(log_at_I, bias_point)
    return _bures_wasserstein_exp(
        transported, bias_point, min_transport_eigval, use_autograd=True
    )


# ----------------------------------------------------------------
# Manual-backward data path (use_autograd=False)
# ----------------------------------------------------------------
# The functions below compose the manual-backward square roots (SqrtmSPD:
# Daleckii-Krein backward), polar factors (PolarFactor), the Lyapunov transports
# and the Lyapunov solve with products and sums differentiated by autograd: no matrix
# decomposition is differentiated by autograd, so the gradients stay finite at
# repeated eigenvalues (a batch of scalar matrices cI, parameters at the identity).


def _sqrtm(data: torch.Tensor) -> torch.Tensor:
    return SqrtmSPD.apply(data)


def BuresWassersteinLogIdentity(X: torch.Tensor) -> torch.Tensor:
    r"""
    Logarithmic map at the identity, :math:`2(X^{1/2} - I)` (manual backward).

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    S : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices in the tangent space at I
    """
    eye = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
    return 2 * (_sqrtm(X) - eye)


def BuresWassersteinLog(X: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    r"""
    Logarithmic map at *base*, :math:`(XB)^{1/2} + (BX)^{1/2} - 2B` (manual backward).

    Same formula as :func:`bures_wasserstein_log`.

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
    return _log_from_roots(_sqrtm(X), _sqrtm(base), base, PolarFactor.apply)


def BuresWassersteinGeodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Bures-Wasserstein geodesic :math:`E_2(X_1, X_2; t)` (manual backward).

    Same formula as :func:`bures_wasserstein_geodesic`.

    Parameters
    ----------
    point1, point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float | torch.Tensor
        Interpolation parameter in [0, 1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices on the geodesic
    """
    return _geodesic_from_roots(_sqrtm(point1), _sqrtm(point2), t, PolarFactor.apply)


def BuresWassersteinMean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor:
    r"""
    Bures-Wasserstein barycenter by fixed-point iteration (manual backward).

    Same iteration as :func:`bures_wasserstein_mean`, built on the
    manual-backward square roots. The gradient is the exact gradient of the
    ``n_iterations`` steps that are computed (the default single step is not
    a converged fixed point, so implicit differentiation of the fixed-point
    equation would differentiate another function).

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
    return _mean_fixed_point(
        data, _sqrtm(data), n_iterations, _sqrtm, PolarFactor.apply
    )


def BuresWassersteinCenter(
    data: torch.Tensor, barycenter: torch.Tensor
) -> torch.Tensor:
    r"""
    Center SPD data at the identity (manual backward).

    :math:`\mathrm{Exp}_I\big(\Gamma_{B\to I}(\mathrm{Log}_B(X))\big)`, as
    :func:`bures_wasserstein_center`, with :class:`ParallelTransportToIdentityBW`.

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
    log_at_bary = BuresWassersteinLog(data, barycenter)
    transported = ParallelTransportToIdentityBW.apply(log_at_bary, barycenter)
    return bures_wasserstein_exp_identity(transported)


def BuresWassersteinScale(
    data: torch.Tensor,
    variance: torch.Tensor,
    shift: torch.Tensor,
    eps: float = BW_SCALE_EPS,
) -> torch.Tensor:
    r"""
    Scale centered SPD data at the identity (manual backward).

    :math:`\big(I + \frac{s}{\sqrt{\text{var} + \epsilon}}(X^{1/2} - I)\big)^2`, as
    :func:`bures_wasserstein_scale`.

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
    return bures_wasserstein_exp_identity(factor * BuresWassersteinLogIdentity(data))


def BuresWassersteinBias(
    data: torch.Tensor,
    bias_point: torch.Tensor,
    min_transport_eigval: float | None = None,
) -> torch.Tensor:
    r"""
    Bias SPD data by transporting from the identity to *bias_point* (manual backward).

    As :func:`bures_wasserstein_bias`.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices around the identity

    bias_point : torch.Tensor of shape (n_features, n_features)
        Learned bias (SPD matrix)

    min_transport_eigval : float or None, optional
        Lower bound of the eigenvalues of the transport map of
        :math:`\mathrm{Exp}_G`, see :func:`bures_wasserstein_bias`.
        Default is None (no projection)

    Returns
    -------
    biased : torch.Tensor of shape (..., n_features, n_features)
        Biased SPD matrices
    """
    log_at_I = BuresWassersteinLogIdentity(data)
    transported = ParallelTransportFromIdentityBW.apply(log_at_I, bias_point)
    return _bures_wasserstein_exp(transported, bias_point, min_transport_eigval)
