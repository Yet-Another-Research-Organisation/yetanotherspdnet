import torch

from .affine_invariant import (
    affine_invariant_mean_2points,
    AffineInvariantMean2Points,
)

from .kullback_leibler import (
    euclidean_geodesic,
    arithmetic_mean,
    ArithmeticMean,
    harmonic_curve,
    harmonic_mean,
    HarmonicMean,
)


# ----------------------------------------------------------------------------
# Curve for adaptive update of geometric mean of arithmetic and harmonic means
# ----------------------------------------------------------------------------
def adaptive_update_geometric_arithmetic_harmonic(
    point1_arithmetic: torch.Tensor,
    point1_harmonic: torch.Tensor,
    point2_arithmetic: torch.Tensor,
    point2_harmonic: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Path for adaptive computation of the geometric mean of arithmetic and harmonic means:
    ( (1-t)*point1_arithmetic + t*point2_arithmetic ) #_{1/2} ( (1-t)*point1_harmonic^{-1} + t*point2_harmonic^{-1} )^{-1}

    Parameters
    ----------
    point1_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point1_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point_arithmetic = euclidean_geodesic(point1_arithmetic, point2_arithmetic, t)
    point_harmonic = harmonic_curve(point1_harmonic, point2_harmonic, t)
    return (
        affine_invariant_mean_2points(point_arithmetic, point_harmonic),
        point_arithmetic,
        point_harmonic,
    )


# -----------------------------------------------
# Geometric mean of arithmetic and harmonic means
# -----------------------------------------------
def geometric_arithmetic_harmonic_mean(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Geometric mean of the arithmetic and harmonic means of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    return_arithmetic_harmonic : bool, optional
        Whether to also return arithmetic and harmonic means (for adptative mean update reasons), by default False

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Geometric mean of the arithmetic and harmonic means

    mean_arithmetic : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean

    mean_harmonic : torch.Tensor of shape (n_features, n_features)
        Harmonic mean (for adptative mean update reasons)
    """
    mean_arithmetic = arithmetic_mean(data)
    mean_harmonic = harmonic_mean(data)
    mean = affine_invariant_mean_2points(mean_arithmetic, mean_harmonic)
    return mean, mean_arithmetic, mean_harmonic


def GeometricArithmeticHarmonicMean(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Geometric mean of the arithmetic and harmonic means

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Geometric mean of the arithmetic and harmonic means

    mean_arithmetic : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean

    mean_harmonic : torch.Tensor of shape (n_features, n_features)
        Harmonic mean (for adptative mean update reasons)
    """
    mean_arithmetic = ArithmeticMean.apply(data)
    mean_harmonic = HarmonicMean.apply(data)
    mean = AffineInvariantMean2Points.apply(mean_arithmetic, mean_harmonic)
    return mean, mean_arithmetic, mean_harmonic
