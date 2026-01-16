import torch
from torch.autograd import Function

from yetanotherspdnet.functions.scalar_functions import inv_sqrt
from yetanotherspdnet.functions.spd_linalg import eigh_operation, eigh_operation_grad

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


# ---------------------------------------------------------------------------
# Adaptive Geometric mean of arithmetic and harmonic means (learnable t)
# ---------------------------------------------------------------------------
def adaptive_geometric_arithmetic_harmonic_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Adaptive geodesic between harmonic (point1) and arithmetic (point2) means:
    point1^{1/2} ( point1^{-1/2} point2 point1^{-1/2} )^t point1^{1/2}

    This is the same formula as the affine-invariant geodesic, but intended
    for use with point1 = harmonic mean and point2 = arithmetic mean, with
    learnable parameter t.

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices (typically harmonic mean)

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices (typically arithmetic mean)

    t : float | torch.Tensor
        parameter on the path, should be in [0,1].
        t=0.5 gives the standard geometric mean.

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
        point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
    )
    pow_t = lambda x: torch.pow(x, t)
    middle_term1 = eigh_operation(eigvals_middle_term1, eigvecs_middle_term1, pow_t)
    return point1_sqrtm @ middle_term1 @ point1_sqrtm


class AdaptiveGeometricArithmeticHarmonicGeodesic(Function):
    """
    Adaptive geodesic between harmonic and arithmetic means with learnable parameter t.
    Includes gradient with respect to t for learning.
    """

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the adaptive GAH geodesic

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point1 : torch.Tensor of shape (..., nfeatures, nfeatures)
            SPD matrices (harmonic mean)

        point2 : torch.Tensor of shape (..., nfeatures, nfeatures)
            SPD matrices (arithmetic mean)

        t : torch.Tensor of shape ()
            Learnable parameter on the geodesic, should be in [0,1]

        Returns
        -------
        point : torch.Tensor of shape (..., n_features, n_features)
            SPD matrices
        """
        eigvals1, eigvecs1 = torch.linalg.eigh(point1)
        point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
        point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
        eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
            point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
        )
        pow_t = lambda x: torch.pow(x, t)
        middle_term1 = eigh_operation(eigvals_middle_term1, eigvecs_middle_term1, pow_t)
        ctx.save_for_backward(
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
            t,
        )
        return point1_sqrtm @ middle_term1 @ point1_sqrtm

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of the adaptive GAH geodesic

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the output

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point1 (harmonic mean)

        grad_input2 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point2 (arithmetic mean)

        grad_t : torch.Tensor of shape ()
            Gradient of the loss with respect to t
        """
        (
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
            t,
        ) = ctx.saved_tensors

        # Gradient with respect to point2
        pow_t = lambda x: torch.pow(x, t)
        pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
        grad_input2 = (
            point1_inv_sqrtm
            @ eigh_operation_grad(
                point1_sqrtm @ grad_output @ point1_sqrtm,
                eigvals_middle_term1,
                eigvecs_middle_term1,
                pow_t,
                pow_t_deriv,
            )
            @ point1_inv_sqrtm
        )

        # Gradient with respect to point1
        eigvals2, eigvecs2 = torch.linalg.eigh(point2)
        point2_sqrtm = eigh_operation(eigvals2, eigvecs2, torch.sqrt)
        point2_inv_sqrtm = eigh_operation(eigvals2, eigvecs2, inv_sqrt)
        eigvals_middle_term2, eigvecs_middle_term2 = torch.linalg.eigh(
            point2_inv_sqrtm @ point1 @ point2_inv_sqrtm
        )
        pow_1_t = lambda x: torch.pow(x, 1 - t)
        pow_1_t_deriv = lambda x: (1 - t) * torch.pow(x, -t)
        grad_input1 = (
            point2_inv_sqrtm
            @ eigh_operation_grad(
                point2_sqrtm @ grad_output @ point2_sqrtm,
                eigvals_middle_term2,
                eigvecs_middle_term2,
                pow_1_t,
                pow_1_t_deriv,
            )
            @ point2_inv_sqrtm
        )

        # Gradient with respect to t
        # d/dt [V D^t V^T] = V (D^t * log(D)) V^T
        log_eigvals = torch.log(eigvals_middle_term1)
        pow_t_eigvals = torch.pow(eigvals_middle_term1, t)
        # middle_term_deriv = V @ diag(D^t * log(D)) @ V^T
        middle_term_deriv = (
            eigvecs_middle_term1
            @ torch.diag_embed(pow_t_eigvals * log_eigvals)
            @ eigvecs_middle_term1.transpose(-1, -2)
        )
        # d(output)/dt = point1_sqrtm @ middle_term_deriv @ point1_sqrtm
        # grad_t = sum(grad_output * d(output)/dt)
        deriv_output_t = point1_sqrtm @ middle_term_deriv @ point1_sqrtm
        grad_t = torch.sum(grad_output * deriv_output_t)

        return grad_input1, grad_input2, grad_t


def adaptive_geometric_arithmetic_harmonic_mean(
    data: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Adaptive geometric mean of the arithmetic and harmonic means of a batch of SPD matrices
    with learnable parameter t.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    t : float | torch.Tensor
        parameter on the geodesic between harmonic (t=0) and arithmetic (t=1) means.
        t=0.5 gives the standard geometric mean of arithmetic and harmonic means.

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Adaptive geometric mean of the arithmetic and harmonic means
    """
    if data.ndim == 2:
        return data
    mean_arithmetic = arithmetic_mean(data)
    mean_harmonic = harmonic_mean(data)
    return adaptive_geometric_arithmetic_harmonic_geodesic(
        mean_harmonic, mean_arithmetic, t
    )


def AdaptiveGeometricArithmeticHarmonicMean(
    data: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Adaptive geometric mean of the arithmetic and harmonic means with learnable t
    (using custom autograd Function for efficient gradients)

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    t : torch.Tensor of shape ()
        Learnable parameter on the geodesic between harmonic (t=0) and arithmetic (t=1) means.
        t=0.5 gives the standard geometric mean of arithmetic and harmonic means.

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Adaptive geometric mean of the arithmetic and harmonic means
    """
    if data.ndim == 2:
        return data
    mean_arithmetic = ArithmeticMean.apply(data)
    mean_harmonic = HarmonicMean.apply(data)
    return AdaptiveGeometricArithmeticHarmonicGeodesic.apply(
        mean_harmonic, mean_arithmetic, t
    )
