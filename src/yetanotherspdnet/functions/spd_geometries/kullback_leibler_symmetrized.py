"""Symmetrized KL geometry: GAH curve, GAH and adaptive GAH means, Jeffreys std."""

import math

import torch
from torch.autograd import Function

from yetanotherspdnet._deprecation import _deprecated
from yetanotherspdnet.functions.scalar_functions import zero_where_zero_std
from yetanotherspdnet.functions.spd_linalg import inv_SPD

from .affine_invariant import (
    AffineInvariantGeodesic,
    affine_invariant_geodesic,
)
from .euclidean import (
    ArithmeticMean,
    EuclideanGeodesic,
    arithmetic_mean,
    euclidean_geodesic,
)
from .kullback_leibler import (
    HarmonicCurve,
    HarmonicMean,
    harmonic_curve,
    harmonic_mean,
)


def geometric_euclidean_harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    GAH curve: geometric mean of the Euclidean geodesic and the harmonic curve.

    .. math::

        \gamma(t) = G\big(E(t),\ H(t)\big), \qquad
        E(t) = (1-t)P_1 + tP_2, \qquad
        H(t) = \big((1-t)P_1^{-1} + tP_2^{-1}\big)^{-1}

    where :math:`E` is the Euclidean geodesic
    (:func:`~yetanotherspdnet.functions.spd_geometries.euclidean.euclidean_geodesic`),
    :math:`H` is the harmonic curve
    (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_curve`),
    and :math:`G` is the affine-invariant geometric mean of two points, the
    midpoint of the affine-invariant geodesic
    (:func:`~yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_geodesic`
    at :math:`t = 1/2`).

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
    GAH curve: geometric mean of the Euclidean geodesic and the harmonic curve.

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
    r"""
    GAH mean: geometric mean of the arithmetic and harmonic means of a batch.

    .. math::

        \bar{P}_{GAH} = G\big(\bar{P}_{arith},\ \bar{P}_{harm}\big)

    with :math:`\bar{P}_{arith}` the arithmetic mean
    (:func:`~yetanotherspdnet.functions.spd_geometries.euclidean.arithmetic_mean`),
    :math:`\bar{P}_{harm}` the harmonic mean
    (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_mean`),
    and :math:`G` the affine-invariant geometric mean of two points.

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
    mean_arithmetic = arithmetic_mean(data)
    mean_harmonic = harmonic_mean(data)
    return affine_invariant_geodesic(mean_arithmetic, mean_harmonic, 0.5)


def GeometricArithmeticHarmonicMean(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Geometric mean of the arithmetic and harmonic means.

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
    return AffineInvariantGeodesic.apply(mean_arithmetic, mean_harmonic, 0.5)


# ---------------
# Scalar variance
# ---------------
def symmetrized_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    r"""
    Scalar standard deviation with respect to the Jeffreys divergence.

    The Jeffreys divergence is the symmetrized Kullback-Leibler divergence.

    .. math::

        \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}
            \frac{\operatorname{tr}(G^{-1}P_i) + \operatorname{tr}(P_i^{-1}G)}{2}
            - n

    the average Jeffreys divergence between the batch's covariances and the
    reference point :math:`G` — symmetrizing
    :func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.left_kullback_leibler_std_scalar`
    and
    :func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.right_kullback_leibler_std_scalar`
    (the :math:`\log\det` terms cancel out).

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
    return _symmetrized_kullback_leibler_std_scalar(data, reference_point)[0]


def _symmetrized_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetrized KL scalar std and the inverses of data and G."""
    n_matrices = math.prod(data.shape[:-2])
    n_features = data.shape[-1]
    G_inv = inv_SPD(reference_point)
    data_inv = inv_SPD(data)
    term1 = torch.einsum("ik,...ki->", G_inv, data)
    term2 = torch.einsum("...ik,ki->", data_inv, reference_point)
    # clamp to avoid small numerical errors yielding small negative variance
    std_scalar = torch.sqrt(
        torch.clamp((term1 + term2) / 2 / n_matrices - n_features, min=0.0)
    )
    return std_scalar, data_inv, G_inv


class SymmetrizedKullbackLeiblerStdScalar(Function):
    """Scalar standard deviation with respect to the symmetrized KL divergence."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the symmetrized KL scalar standard deviation.

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
        std_scalar, data_inv, G_inv = _symmetrized_kullback_leibler_std_scalar(
            data, reference_point
        )
        ctx.n_matrices = math.prod(data.shape[:-2])
        ctx.save_for_backward(data, reference_point, data_inv, G_inv, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the symmetrized KL scalar standard deviation.

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
        # std = 0: zero subgradient instead of 0/0
        return (
            zero_where_zero_std(grad_input_data, std_scalar),
            zero_where_zero_std(grad_input_G, std_scalar),
        )


# ---------------------------------------------------------------------------
# Adaptive Geometric mean of arithmetic and harmonic means (learnable t)
# ---------------------------------------------------------------------------
# the adaptive GAH geodesic is the affine-invariant geodesic, whose manual
# backward also gives the gradient with respect to t
adaptive_geometric_arithmetic_harmonic_geodesic = _deprecated(
    "adaptive_geometric_arithmetic_harmonic_geodesic", affine_invariant_geodesic
)
AdaptiveGeometricArithmeticHarmonicGeodesic = _deprecated(
    "AdaptiveGeometricArithmeticHarmonicGeodesic", AffineInvariantGeodesic
)


def adaptive_geometric_arithmetic_harmonic_mean(
    data: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Adaptive GAH mean, with a learnable position t on the geodesic.

    Point at :math:`t` on the affine-invariant geodesic from the harmonic to the
    arithmetic mean of a batch of SPD matrices.

    .. math::

        \bar{P}_{t} = \gamma_{AI}\big(\bar{P}_{harm}, \bar{P}_{arith}, t\big)

    where :math:`\gamma_{AI}` is the affine-invariant geodesic
    (:func:`~yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_geodesic`).
    Unlike
    :func:`geometric_arithmetic_harmonic_mean` (fixed at :math:`t=0.5`),
    :math:`t` here can be learned, letting the model pick where between the
    harmonic and arithmetic means the effective "center" of BatchNorm sits.

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
    return affine_invariant_geodesic(mean_harmonic, mean_arithmetic, t)


def AdaptiveGeometricArithmeticHarmonicMean(
    data: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Adaptive GAH mean, with a learnable position t (manual backward).

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
    return AffineInvariantGeodesic.apply(mean_harmonic, mean_arithmetic, t)
