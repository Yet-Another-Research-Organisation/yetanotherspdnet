"""Log-Euclidean Riemannian geometry: geodesic, exp / log maps, mean, std, transport."""

import math
import warnings

import torch
from torch.autograd import Function

from yetanotherspdnet._deprecation import _mark_deprecated
from yetanotherspdnet.functions.scalar_functions import (
    inv,
    std_from_variance,
    zero_where_zero_std,
)

from ..spd_linalg import (
    ExpmSymmetric,
    LogmSPD,
    eigh_operation,
    eigh_operation_grad,
    expm_symmetric,
    logm_SPD,
    symmetric_eigh,
)
from .euclidean import (
    ArithmeticMean,
    EuclideanGeodesic,
    arithmetic_mean,
    euclidean_geodesic,
)


# ----------------------
# Log-Euclidean geodesic
# ----------------------
def log_euclidean_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Log-Euclidean geodesic between two SPD matrices.

    .. math::

        \gamma(t) = \exp\big((1-t)\,\log(P_1) + t\,\log(P_2)\big)

    where :math:`\log` and :math:`\exp` are the matrix logarithm and
    exponential (:func:`~yetanotherspdnet.functions.spd_linalg.logm_SPD`,
    :func:`~yetanotherspdnet.functions.spd_linalg.expm_symmetric`). This is
    simply the Euclidean geodesic in the tangent space obtained by taking
    the matrix log, mapped back to the SPD manifold.

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point1_logm = logm_SPD(point1)[0]
    point2_logm = logm_SPD(point2)[0]
    point_logm = euclidean_geodesic(point1_logm, point2_logm, t)
    return expm_symmetric(point_logm)[0]


def LogEuclideanGeodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Log-Euclidean geodesic between two batches of SPD matrices.

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., nfeatures, nfeatures)
        SPD matrices

    point2 : torch.Tensor of shape (..., nfeatures, nfeatures)
        SPD matrices

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point1_logm = LogmSPD.apply(point1)
    point2_logm = LogmSPD.apply(point2)
    point_logm = EuclideanGeodesic.apply(point1_logm, point2_logm, t)
    return ExpmSymmetric.apply(point_logm)


# ------------------
# Log-Euclidean mean
# ------------------
def log_euclidean_mean(data: torch.Tensor) -> torch.Tensor:
    r"""
    Log-Euclidean mean of a batch of SPD matrices.

    .. math::

        \bar{P} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \log(P_i)\right)

    i.e. the arithmetic mean of the matrix logarithms, mapped back to the
    SPD manifold with the matrix exponential. Unlike the affine-invariant
    or Bures-Wasserstein means, this has a closed form (no fixed-point
    iteration needed).

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Log-Euclidean mean
    """
    if data.ndim == 2:
        return data
    log_data = logm_SPD(data)[0]
    log_mean = arithmetic_mean(log_data)
    return expm_symmetric(log_mean)[0]


def LogEuclideanMean(data: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean mean.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Log-Euclidean mean
    """
    if data.ndim == 2:
        return data
    log_data = LogmSPD.apply(data)
    log_mean = ArithmeticMean.apply(log_data)
    mean = ExpmSymmetric.apply(log_mean)
    return mean


# ---------------
# Scalar variance
# ---------------
def log_euclidean_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    r"""
    Scalar standard deviation with respect to the Log-Euclidean distance.

    .. math::

        \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
            \lVert \log(P_i) - \log(G) \rVert_F^2}

    where :math:`G` is the reference point (typically the Log-Euclidean
    mean) and :math:`\lVert \cdot \rVert_F` is the Frobenius norm — the
    Log-Euclidean distance is simply the Euclidean distance between matrix
    logarithms.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    reference_point : torch.Tensor of shape (n_features, n_features)
        SPD matrix (some kind of mean of data)

    Returns
    -------
    scalar_std : torch.Tensor of shape ()
        scalar standard deviation
    """
    return _log_euclidean_std_scalar(data, reference_point)[0]


def _log_euclidean_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Log-Euclidean scalar std and the logarithms / eigendecompositions it uses."""
    n_matrices = math.prod(data.shape[:-2])
    logm_data, eigvals_data, eigvecs_data = logm_SPD(data)
    logm_G, eigvals_G, eigvecs_G = logm_SPD(reference_point)
    variance = torch.sum((logm_data - logm_G) ** 2) / n_matrices
    std_scalar = std_from_variance(variance)
    return (
        std_scalar,
        logm_data,
        logm_G,
        eigvals_data,
        eigvecs_data,
        eigvals_G,
        eigvecs_G,
    )


class LogEuclideanStdScalar(Function):
    """Scalar standard deviation with respect to the Log-Euclidean distance."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the log-Euclidean scalar standard deviation.

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
            scalar standard deviation
        """
        (
            std_scalar,
            logm_data,
            logm_G,
            eigvals_data,
            eigvecs_data,
            eigvals_G,
            eigvecs_G,
        ) = _log_euclidean_std_scalar(data, reference_point)
        ctx.n_matrices = math.prod(data.shape[:-2])
        ctx.save_for_backward(
            logm_data,
            logm_G,
            eigvals_data,
            eigvecs_data,
            eigvals_G,
            eigvecs_G,
            std_scalar,
        )
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the log-Euclidean scalar standard deviation.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape ()
            Gradient of the loss with respect to the output of the scalar standard deviation Function

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            gradient of the loss with respect to the input data

        grad_input_reference_point : torch.Tensor of shape (n_features, n_features)
            gradient of the loss with respect to the input reference point
        """
        n_matrices = ctx.n_matrices
        (
            logm_data,
            logm_G,
            eigvals_data,
            eigvecs_data,
            eigvals_G,
            eigvecs_G,
            std_scalar,
        ) = ctx.saved_tensors
        grad_logm_data = grad_output * (logm_data - logm_G) / std_scalar / n_matrices
        grad_logm_G = grad_output * (logm_G - arithmetic_mean(logm_data)) / std_scalar
        grad_input_data = eigh_operation_grad(
            grad_logm_data, eigvals_data, eigvecs_data, torch.log, inv
        )
        grad_input_G = eigh_operation_grad(
            grad_logm_G, eigvals_G, eigvecs_G, torch.log, inv
        )
        # std = 0: zero subgradient instead of 0/0
        return (
            zero_where_zero_std(grad_input_data, std_scalar),
            zero_where_zero_std(grad_input_G, std_scalar),
        )


# ------------------------------
# Log-Euclidean exp and log maps
# ------------------------------
def log_euclidean_exp(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    r"""
    Log-Euclidean exponential map on the SPD manifold.

    .. math::

        \mathrm{Exp}^{LE}_X(V) = \exp\big(\log X + D\log_X[V]\big),
        \qquad
        D\log_X[V] = U \big(L^{\log}(\lambda) \circ (U^\top V U)\big) U^\top

    with :math:`X = U \operatorname{diag}(\lambda) U^\top` and the
    Daleckii-Krein matrix :math:`L^{\log}_{ij} = (\log\lambda_i -
    \log\lambda_j) / (\lambda_i - \lambda_j)` (:math:`1/\lambda_i` on the
    diagonal). The log-Euclidean metric is the pull-back of the Euclidean one by
    :math:`\log`, :math:`g_X(V, W) = \langle D\log_X[V], D\log_X[W]
    \rangle_F`, so its geodesics are straight lines in log coordinates.

    The residual blocks do not use this map: their log-Euclidean step
    :math:`\exp(\log X + V)` (as the reference RResNet code) is a translation
    in log coordinates, i.e. :math:`\mathrm{Exp}^{LE}_X(V')` for the tangent
    vector :math:`V' = D\exp_{\log X}[V]` (:math:`V` read as a displacement of
    :math:`\log X`). Differentiated by autograd (metric function, no layer
    uses it).

    Parameters
    ----------
    base : torch.Tensor of shape (..., n_features, n_features)
        SPD base points
    tangent : torch.Tensor of shape (..., n_features, n_features)
        Symmetric tangent vectors at *base*

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    eigvals, eigvecs = symmetric_eigh(base)
    log_base = eigh_operation(eigvals, eigvecs, torch.log)
    # D log_X[V]: the Daleckii-Krein operator is self-adjoint, so the backward
    # kernel of logm applied to V is the differential
    log_tangent = eigh_operation_grad(tangent, eigvals, eigvecs, torch.log, inv)
    return expm_symmetric(log_base + log_tangent)[0]


def log_euclidean_log(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    r"""
    Log-Euclidean logarithmic map on the SPD manifold.

    .. math::

        \mathrm{Log}^{LE}_X(Y) = D\exp_{\log X}\big[\log Y - \log X\big]
        = U \big(L^{\exp}(\log\lambda) \circ
          (U^\top (\log Y - \log X) U)\big) U^\top

    the inverse of :func:`log_euclidean_exp` (:math:`L^{\exp}_{ij} =
    (\lambda_i - \lambda_j) / (\log\lambda_i - \log\lambda_j)`, the
    elementwise inverse of :math:`L^{\log}`). Differentiated by autograd
    (metric function).

    Parameters
    ----------
    base : torch.Tensor of shape (..., n_features, n_features)
        SPD base points
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    Returns
    -------
    tangent : torch.Tensor of shape (..., n_features, n_features)
        Symmetric tangent vectors at *base*
    """
    eigvals, eigvecs = symmetric_eigh(base)
    log_eigvals = torch.log(eigvals)
    log_base = eigh_operation(log_eigvals, eigvecs, lambda x: x)
    difference = logm_SPD(point)[0] - log_base
    return eigh_operation_grad(difference, log_eigvals, eigvecs, torch.exp, torch.exp)


# ------------------
# Parallel transport
# ------------------
def log_euclidean_parallel_transport_SPD(
    data: torch.Tensor, current_reference: torch.Tensor, new_reference: torch.Tensor
) -> torch.Tensor:
    """
    Parallel transport with respect to the log-Euclidean geometry.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    current_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    new_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transp : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices, moved from current_reference to new_reference
    """
    return expm_symmetric(
        logm_SPD(data)[0] - logm_SPD(current_reference)[0] + logm_SPD(new_reference)[0]
    )[0]


def LogEuclideanParallelTransportSPD(
    data: torch.Tensor, current_reference: torch.Tensor, new_reference: torch.Tensor
) -> torch.Tensor:
    """
    Parallel transport with respect to the log-Euclidean geometry.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    current_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    new_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transp : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices, moved from current_reference to new_reference
    """
    return ExpmSymmetric.apply(
        LogmSPD.apply(data)
        - LogmSPD.apply(current_reference)
        + LogmSPD.apply(new_reference)
    )


def LogEuclideanParalleTransportSPD(
    data: torch.Tensor, current_reference: torch.Tensor, new_reference: torch.Tensor
) -> torch.Tensor:
    """
    Use :func:`LogEuclideanParallelTransportSPD` instead (deprecated alias).

    Misspelled name, kept for backward compatibility.
    """
    warnings.warn(
        "LogEuclideanParalleTransportSPD is deprecated, use "
        "LogEuclideanParallelTransportSPD instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return LogEuclideanParallelTransportSPD(data, current_reference, new_reference)


# marker read by the documentation (docs/_ext/dualpath_pairing.py)
_mark_deprecated(
    LogEuclideanParalleTransportSPD, "use LogEuclideanParallelTransportSPD instead"
)
