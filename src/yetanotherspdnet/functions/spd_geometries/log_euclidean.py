"""Log-Euclidean Riemannian geometry: geodesic, mean, and standard deviation."""

import math
import warnings

import torch
from torch.autograd import Function

from yetanotherspdnet._deprecation import _mark_deprecated
from yetanotherspdnet.functions.scalar_functions import inv, zero_where_zero_std

from ..spd_linalg import (
    ExpmSymmetric,
    LogmSPD,
    eigh_operation_grad,
    expm_symmetric,
    logm_SPD,
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
    std_scalar = torch.sqrt(torch.clamp(variance, min=0.0))
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
