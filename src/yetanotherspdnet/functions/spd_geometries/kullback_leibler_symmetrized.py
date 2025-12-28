import torch
from torch.autograd import Function

from .affine_invariant import (
    AffineInvariantGeodesic,
    AffineInvariantMean2Points,
    affine_invariant_geodesic,
    affine_invariant_mean_2points,
)
from .kullback_leibler import (
    ArithmeticMean,
    EuclideanGeodesic,
    HarmonicCurve,
    HarmonicMean,
    arithmetic_mean,
    euclidean_geodesic,
    harmonic_curve,
    harmonic_mean,
)


def geometric_euclidean_harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Curve corresponding to the geometric mean of the Euclidean geodesic and the harmonic curve

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
    if t == 0.0:
        return point1
    if t == 1.0:
        return point2
    point_euclidean = euclidean_geodesic(point1, point2, t)
    point_harmonic = harmonic_curve(point1, point2, t)
    return affine_invariant_geodesic(point_euclidean, point_harmonic, 0.5)


def GeometricEuclideanHarmonicCurve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Curve corresponding to the geometric mean of the Euclidean geodesic and the harmonic curve

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
    if t == 0.0:
        return point1
    if t == 1.0:
        return point2
    point_euclidean = EuclideanGeodesic.apply(point1, point2, t)
    point_harmonic = HarmonicCurve.apply(point1, point2, t)
    return AffineInvariantGeodesic.apply(point_euclidean, point_harmonic, 0.5)


# -----------------------------------------------
# Geometric mean of arithmetic and harmonic means
# -----------------------------------------------
def geometric_arithmetic_harmonic_mean(
    data: torch.Tensor,
) -> torch.Tensor:
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
    """
    if data.ndim == 2:
        return data
    mean_arithmetic = arithmetic_mean(data)
    mean_harmonic = harmonic_mean(data)
    return affine_invariant_mean_2points(mean_arithmetic, mean_harmonic)


def GeometricArithmeticHarmonicMean(
    data: torch.Tensor,
) -> torch.Tensor:
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
    """
    if data.ndim == 2:
        return data
    mean_arithmetic = ArithmeticMean.apply(data)
    mean_harmonic = HarmonicMean.apply(data)
    return AffineInvariantMean2Points.apply(mean_arithmetic, mean_harmonic)


# ---------------
# Scalar variance
# ---------------
def symmetrized_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    """
    Scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence

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
    n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
    n_features = data.shape[-1]
    G_inv = torch.cholesky_inverse(torch.linalg.cholesky(reference_point))
    data_inv = torch.cholesky_inverse(torch.linalg.cholesky(data))
    term1 = torch.einsum("ik,...ki->", G_inv, data)
    term2 = torch.einsum("...ik,ki->", data_inv, reference_point)
    # clamp to avoid small numerical errors yielding small negative variance
    return torch.sqrt(
        torch.clamp((term1 + term2) / 2 / n_matrices - n_features, min=0.0)
    )


class SymmetrizedKullbackLeiblerStdScalar(Function):
    """
    Scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence

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
        n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
        n_features = data.shape[-1]
        G_inv = torch.cholesky_inverse(torch.linalg.cholesky(reference_point))
        data_inv = torch.cholesky_inverse(torch.linalg.cholesky(data))
        term1 = torch.einsum("ik,...ki->", G_inv, data)
        term2 = torch.einsum("...ik,ki->", data_inv, reference_point)
        # clamp to avoid small numerical errors yielding small negative variance
        std_scalar = torch.sqrt(
            torch.clamp((term1 + term2) / 2 / n_matrices - n_features, min=0.0)
        )
        ctx.n_matrices = n_matrices
        ctx.save_for_backward(data, reference_point, data_inv, G_inv, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence

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
        data, reference_point, data_inv, G_inv, std_scalar = ctx.saved_tensors
        grad_input_data = (
            grad_output
            * (G_inv - data_inv @ reference_point @ data_inv)
            / 4
            / std_scalar
            / n_matrices
        )
        grad_input_G = (
            grad_output
            * arithmetic_mean(data_inv - G_inv @ data @ G_inv)
            / 4
            / std_scalar
        )
        return grad_input_data, grad_input_G
