r"""Robust covariance estimation: sample covariance and M-estimators.

An M-estimator of scatter is a fixed point of

.. math::

    \Sigma = F(\Sigma) = \frac{1}{n} \sum_{i=1}^{n}
        u\big(x_i^\top \Sigma^{-1} x_i\big)\, x_i x_i^\top

for a weight function :math:`u` that down-weights samples with a large
Mahalanobis distance: Tyler (:math:`u(q) = p/q`), Student-t
(:math:`u(q) = (p + \nu)/(\nu + q)`) or Huber. The sample covariance matrix
corresponds to :math:`u \equiv 1`.

Two gradient paths are available, as elsewhere in the library:

- :func:`m_estimator` unrolls the fixed-point iterations and lets autograd
  differentiate through them (memory grows with the number of iterations);
- :class:`MEstimator` differentiates the fixed point implicitly: the backward
  solves the adjoint equation :math:`w = g + J_\Sigma^\top w` by iteration and
  returns :math:`J_X^\top w`, with memory independent of the number of
  iterations.

Tyler's weight is scale-invariant (:math:`F(c\Sigma) = cF(\Sigma)`), so its
fixed point is only defined up to scale: use it with ``normalize="trace"`` or
``normalize="determinant"``, which is applied at every iteration and pins the
scale down.
"""

import math
from collections.abc import Callable
from functools import partial

import torch
from torch.autograd import Function


# ----------------
# Weight functions
# ----------------
def tyler_function(quadratic: torch.Tensor, n_features: int) -> torch.Tensor:
    r"""
    Tyler weight :math:`u(q) = p / q`.

    Parameters
    ----------
    quadratic : torch.Tensor of shape (..., n_samples)
        Squared Mahalanobis distances :math:`q_i = x_i^\top \Sigma^{-1} x_i`

    n_features : int
        Dimension :math:`p` of the samples

    Returns
    -------
    weights : torch.Tensor of shape (..., n_samples)
        Sample weights
    """
    return n_features / quadratic


def student_function(
    quadratic: torch.Tensor, n_features: int, nu: float
) -> torch.Tensor:
    r"""
    Student-t weight :math:`u(q) = (p + \nu) / (\nu + q)`.

    Maximum-likelihood weight for a multivariate Student-t distribution with
    :math:`\nu` degrees of freedom; tends to the sample covariance when
    :math:`\nu \to \infty` and to Tyler's weight when :math:`\nu \to 0`.

    Parameters
    ----------
    quadratic : torch.Tensor of shape (..., n_samples)
        Squared Mahalanobis distances

    n_features : int
        Dimension :math:`p` of the samples

    nu : float
        Degrees of freedom

    Returns
    -------
    weights : torch.Tensor of shape (..., n_samples)
        Sample weights
    """
    return (n_features + nu) / (nu + quadratic)


def huber_function(quadratic: torch.Tensor, delta: float, beta: float) -> torch.Tensor:
    r"""
    Huber weight: :math:`u(q) = 1/\beta` if :math:`q \le \delta`, else
    :math:`\delta / (\beta q)`.

    Parameters
    ----------
    quadratic : torch.Tensor of shape (..., n_samples)
        Squared Mahalanobis distances

    delta : float
        Threshold above which samples are down-weighted

    beta : float
        Scaling factor (makes the estimator consistent for Gaussian data)

    Returns
    -------
    weights : torch.Tensor of shape (..., n_samples)
        Sample weights
    """
    return torch.where(quadratic <= delta, 1 / beta, delta / (beta * quadratic))


# -------------
# Normalization
# -------------
def normalize_trace(data: torch.Tensor) -> torch.Tensor:
    r"""
    Scale SPD matrices so that :math:`\operatorname{tr}(\Sigma) = p`.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    normalized : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices with trace equal to ``n_features``
    """
    trace = torch.diagonal(data, dim1=-2, dim2=-1).sum(dim=-1)
    return data.shape[-1] * data / trace[..., None, None]


def normalize_determinant(data: torch.Tensor) -> torch.Tensor:
    r"""
    Scale SPD matrices so that :math:`\det(\Sigma) = 1`.

    The determinant is computed through its logarithm to avoid overflow.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    normalized : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices with unit determinant
    """
    logdet = torch.linalg.slogdet(data).logabsdet
    return data * torch.exp(-logdet / data.shape[-1])[..., None, None]


_NORMALIZATIONS: dict[str, Callable] = {
    "trace": normalize_trace,
    "determinant": normalize_determinant,
}


def _get_normalization(normalize: str | None) -> Callable | None:
    if normalize is None:
        return None
    if normalize not in _NORMALIZATIONS:
        raise ValueError(
            f"normalize must be None or one of {list(_NORMALIZATIONS)}, got {normalize}"
        )
    return _NORMALIZATIONS[normalize]


# -----------------
# Sample covariance
# -----------------
def sample_covariance(
    data: torch.Tensor, assume_centered: bool = False
) -> torch.Tensor:
    r"""
    Sample covariance matrix (SCM).

    .. math::

        \hat\Sigma = \frac{1}{n'} \sum_{i=1}^{n} (x_i - \bar x)(x_i - \bar x)^\top

    with :math:`n' = n - 1` when the data are centered here, :math:`n'= n`
    (and :math:`\bar x = 0`) when ``assume_centered``.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Samples

    assume_centered : bool, optional
        Whether the data are already centered. Default is False

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Sample covariance matrices
    """
    n_samples = data.shape[-2]
    if not assume_centered:
        data = data - data.mean(dim=-2, keepdim=True)
        n_samples = n_samples - 1
    covariance = data.transpose(-2, -1) @ data / n_samples
    return 0.5 * (covariance + covariance.transpose(-2, -1))


# -------------
# M-estimators
# -------------
def m_estimator_step(
    covariance: torch.Tensor,
    data: torch.Tensor,
    weight_function: Callable,
    shrinkage: float | None = None,
    normalize: Callable | None = None,
) -> torch.Tensor:
    r"""
    One fixed-point iteration :math:`\Sigma \mapsto F(\Sigma)` of an M-estimator.

    .. math::

        F(\Sigma) = \frac{1}{n} \sum_{i=1}^{n}
            u\big(x_i^\top \Sigma^{-1} x_i\big)\, x_i x_i^\top

    followed, if requested, by the shrinkage
    :math:`\beta F(\Sigma) + (1 - \beta) I` and by a normalization. The
    Mahalanobis distances are computed with a Cholesky factorization.

    Parameters
    ----------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Current estimate

    data : torch.Tensor of shape (..., n_samples, n_features)
        Centered samples

    weight_function : Callable
        Weight :math:`u`, called on the tensor of squared distances of shape
        ``(..., n_samples)``

    shrinkage : float | None, optional
        Shrinkage coefficient :math:`\beta \in (0, 1]` towards the identity.
        Default is None (no shrinkage)

    normalize : Callable | None, optional
        Normalization applied to the result (e.g. :func:`normalize_trace`).
        Default is None

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Updated estimate
    """
    cholesky = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(
        cholesky, data.transpose(-2, -1), upper=False
    )  # L^{-1} x_i as columns
    quadratic = (whitened**2).sum(dim=-2)
    weights = weight_function(quadratic)
    weighted = data * weights.unsqueeze(-1)
    updated = weighted.transpose(-2, -1) @ data / data.shape[-2]
    updated = 0.5 * (updated + updated.transpose(-2, -1))
    if shrinkage is not None:
        eye = torch.eye(data.shape[-1], dtype=data.dtype, device=data.device)
        updated = shrinkage * updated + (1 - shrinkage) * eye
    if normalize is not None:
        updated = normalize(updated)
    return updated


def _initial_covariance(data: torch.Tensor, init: torch.Tensor | None) -> torch.Tensor:
    n_features = data.shape[-1]
    batch_shape = data.shape[:-2]
    if init is None:
        init = torch.eye(n_features, dtype=data.dtype, device=data.device)
    if init.shape[-1] != n_features:
        raise ValueError(
            f"init of size {tuple(init.shape)} incompatible with data "
            f"of size {tuple(data.shape)}"
        )
    return init.expand(*batch_shape, n_features, n_features)


def _relative_change(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Largest relative Frobenius change over the batch."""
    return (torch.linalg.matrix_norm(new - old) / torch.linalg.matrix_norm(old)).max()


def m_estimator(
    data: torch.Tensor,
    weight_function: Callable,
    n_iterations: int = 30,
    tol: float = 1e-6,
    assume_centered: bool = False,
    init: torch.Tensor | None = None,
    shrinkage: float | None = None,
    normalize: str | None = None,
) -> torch.Tensor:
    r"""
    M-estimator of scatter by fixed-point iterations (autograd path).

    Iterates :func:`m_estimator_step` from ``init`` until the largest relative
    Frobenius change over the batch falls below ``tol`` or ``n_iterations`` is
    reached. Gradients flow through the unrolled iterations.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Samples

    weight_function : Callable
        Weight :math:`u` of squared Mahalanobis distances, e.g.
        ``functools.partial(student_function, n_features=p, nu=3.0)``

    n_iterations : int, optional
        Maximum number of iterations. Default is 30

    tol : float, optional
        Stopping tolerance on the relative change. Default is 1e-6

    assume_centered : bool, optional
        Whether the data are already centered. Default is False

    init : torch.Tensor of shape (n_features, n_features) or (..., n_features, n_features), optional
        Initial estimate. Default is the identity

    shrinkage : float | None, optional
        Shrinkage coefficient towards the identity, applied at each iteration.
        Default is None

    normalize : str | None, optional
        ``"trace"``, ``"determinant"`` or None, applied at each iteration.
        Required for scale-invariant weights such as Tyler's. Default is None

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Estimated scatter matrices
    """
    normalization = _get_normalization(normalize)
    if not assume_centered:
        data = data - data.mean(dim=-2, keepdim=True)
    covariance = _initial_covariance(data, init)
    for _ in range(n_iterations):
        updated = m_estimator_step(
            covariance, data, weight_function, shrinkage, normalization
        )
        converged = _relative_change(updated.detach(), covariance.detach()) < tol
        covariance = updated
        if converged:
            break
    return covariance


class MEstimator(Function):
    r"""
    M-estimator of scatter with an implicit (fixed-point) backward.

    The forward pass iterates without building a graph. At the fixed point
    :math:`\Sigma^\star = F(\Sigma^\star, X)`, the implicit function theorem
    gives :math:`\partial L / \partial X = J_X^\top w` where :math:`w` solves
    :math:`w = g + J_\Sigma^\top w` (:math:`g` the incoming gradient). The
    adjoint equation is solved by fixed-point iteration, which converges at
    the rate of the forward iteration since :math:`J_\Sigma` has spectral
    radius below one at an attracting fixed point. The vector-Jacobian
    products of one step are obtained with autograd.

    Use as ``MEstimator.apply(data, weight_function, n_iterations, tol,
    assume_centered, init, shrinkage, normalize)`` with the arguments of
    :func:`m_estimator`.
    """

    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        weight_function: Callable,
        n_iterations: int = 30,
        tol: float = 1e-6,
        assume_centered: bool = False,
        init: torch.Tensor | None = None,
        shrinkage: float | None = None,
        normalize: str | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: fixed-point iterations without graph

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data, weight_function, n_iterations, tol, assume_centered, init, shrinkage, normalize
            See :func:`m_estimator`

        Returns
        -------
        covariance : torch.Tensor of shape (..., n_features, n_features)
            Estimated scatter matrices
        """
        with torch.no_grad():
            covariance = m_estimator(
                data,
                weight_function,
                n_iterations,
                tol,
                assume_centered,
                init,
                shrinkage,
                normalize,
            )
        ctx.save_for_backward(data, covariance)
        ctx.weight_function = weight_function
        ctx.n_iterations = n_iterations
        ctx.tol = tol
        ctx.assume_centered = assume_centered
        ctx.shrinkage = shrinkage
        ctx.normalization = _get_normalization(normalize)
        return covariance

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass: implicit differentiation at the fixed point

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object with the saved tensors

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the estimate

        Returns
        -------
        grad_data : torch.Tensor of shape (..., n_samples, n_features)
            Gradient of the loss with respect to the samples; None for the
            other arguments
        """
        data, covariance = ctx.saved_tensors
        with torch.enable_grad():
            data_leaf = data.detach().requires_grad_(True)
            covariance_leaf = covariance.detach().requires_grad_(True)
            centered = (
                data_leaf
                if ctx.assume_centered
                else data_leaf - data_leaf.mean(dim=-2, keepdim=True)
            )
            fixed_point = m_estimator_step(
                covariance_leaf,
                centered,
                ctx.weight_function,
                ctx.shrinkage,
                ctx.normalization,
            )
        grad_output = 0.5 * (grad_output + grad_output.transpose(-2, -1))
        adjoint = grad_output
        for _ in range(ctx.n_iterations):
            (vjp_covariance,) = torch.autograd.grad(
                fixed_point, covariance_leaf, adjoint, retain_graph=True
            )
            updated = grad_output + vjp_covariance
            converged = (
                _relative_change(updated, adjoint)
                if torch.linalg.matrix_norm(adjoint).max() > 0
                else torch.tensor(0.0)
            ) < ctx.tol
            adjoint = updated
            if converged:
                break
        (grad_data,) = torch.autograd.grad(fixed_point, data_leaf, adjoint)
        return grad_data, None, None, None, None, None, None, None


def tyler_estimator(
    data: torch.Tensor,
    n_iterations: int = 30,
    tol: float = 1e-6,
    assume_centered: bool = False,
    normalize: str = "trace",
    use_autograd: bool = False,
) -> torch.Tensor:
    r"""
    Tyler's M-estimator of scatter, normalized to fix its scale.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Samples (at least ``n_features + 1`` of them)

    n_iterations : int, optional
        Maximum number of iterations. Default is 30

    tol : float, optional
        Stopping tolerance. Default is 1e-6

    assume_centered : bool, optional
        Whether the data are already centered. Default is False

    normalize : str, optional
        ``"trace"`` or ``"determinant"``. Default is ``"trace"``

    use_autograd : bool, optional
        Unrolled autograd path (True) or implicit backward (False, default)

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Tyler estimates
    """
    if normalize is None:
        raise ValueError("Tyler's estimator is scale-invariant: normalize is required")
    weight = partial(tyler_function, n_features=data.shape[-1])
    if use_autograd:
        return m_estimator(
            data, weight, n_iterations, tol, assume_centered, normalize=normalize
        )
    return MEstimator.apply(
        data, weight, n_iterations, tol, assume_centered, None, None, normalize
    )


def student_estimator(
    data: torch.Tensor,
    nu: float,
    n_iterations: int = 30,
    tol: float = 1e-6,
    assume_centered: bool = False,
    use_autograd: bool = False,
) -> torch.Tensor:
    r"""
    Student-t M-estimator of scatter.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Samples

    nu : float
        Degrees of freedom (:math:`\nu > 0`)

    n_iterations : int, optional
        Maximum number of iterations. Default is 30

    tol : float, optional
        Stopping tolerance. Default is 1e-6

    assume_centered : bool, optional
        Whether the data are already centered. Default is False

    use_autograd : bool, optional
        Unrolled autograd path (True) or implicit backward (False, default)

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Student-t estimates
    """
    if not nu > 0 or math.isinf(nu):
        raise ValueError(f"nu must be a finite positive number, got {nu}")
    weight = partial(student_function, n_features=data.shape[-1], nu=nu)
    if use_autograd:
        return m_estimator(data, weight, n_iterations, tol, assume_centered)
    return MEstimator.apply(
        data, weight, n_iterations, tol, assume_centered, None, None, None
    )
