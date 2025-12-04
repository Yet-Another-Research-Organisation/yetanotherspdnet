import torch

from ..spd_linalg import ExpmSymmetric, LogmSPD, expm_symmetric, logm_SPD
from .kullback_leibler import (
    euclidean_geodesic,
    ArithmeticMean,
    arithmetic_mean,
)


# --------------------------
# Affine-invariant geodesics
# --------------------------
def log_euclidean_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Log-Euclidean geodesic:
    expm((1-t)*logm(point1) + t*logm(point2))

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


# ------------------
# Log-Euclidean mean
# ------------------
def log_euclidean_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean mean of a batch of SPD matrices

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
    Log-Euclidean mean

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
