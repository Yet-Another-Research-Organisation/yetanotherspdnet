"""Kullback-Leibler geometries: harmonic curve and mean, left / right KL scalar std.

The Euclidean geodesic and the arithmetic mean (the left KL mean) live in
:mod:`~yetanotherspdnet.functions.spd_geometries.euclidean`.
"""

import math

import torch
from torch.autograd import Function

from ..._deprecation import _deprecated
from ..scalar_functions import std_from_variance, zero_where_zero_std
from ..spd_linalg import (
    inv_SPD,
    symmetrize,
)
from . import euclidean


# ---------------
# Scalar variance
# ---------------
def left_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    r"""
    Scalar standard deviation with respect to the left Kullback-Leibler divergence.

    .. math::

        \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}\Big[
            \operatorname{tr}(G^{-1} P_i) - \log\det(P_i)\Big]
            + \log\det(G) - n

    where :math:`n` is the matrix dimension and :math:`G` is the reference
    point. Up to a factor 2, this is the average Kullback-Leibler
    divergence :math:`\mathrm{KL}\big(\mathcal{N}(0, P_i) \,\|\,
    \mathcal{N}(0, G)\big)` between zero-mean Gaussians with the batch's
    covariances and the reference covariance — hence "left" (:math:`G` is
    the second argument of the KL divergence).

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
    return _left_kullback_leibler_std_scalar(data, reference_point)[0]


def _left_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left KL scalar std, the inverse of G and the Cholesky factors of data."""
    n_matrices = math.prod(data.shape[:-2])
    n_features = data.shape[-1]
    L_G = torch.linalg.cholesky(reference_point)
    G_inv = torch.cholesky_inverse(L_G)
    L_data = torch.linalg.cholesky(data)
    trace = torch.einsum("ik,...ki->", G_inv, data)
    logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
    logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
    var = (trace - logdet_data) / n_matrices + logdet_G - n_features
    # clamp to avoid small numerical errors yielding small negative variance
    return std_from_variance(var), G_inv, L_data


class LeftKullbackLeiblerStdScalar(Function):
    """Scalar standard deviation with respect to the left Kullback-Leibler divergence."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the left Kullback-Leibler scalar standard deviation.

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
        std_scalar, G_inv, L_data = _left_kullback_leibler_std_scalar(
            data, reference_point
        )
        ctx.n_matrices = math.prod(data.shape[:-2])
        ctx.save_for_backward(data, reference_point, G_inv, L_data, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the left Kullback-Leibler scalar standard deviation.

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
        data, reference_point, G_inv, L_data, std_scalar = ctx.saved_tensors
        data_inv = torch.cholesky_inverse(L_data)
        grad_input_data = grad_output * (G_inv - data_inv) / 2 / std_scalar / n_matrices
        grad_input_G = (
            grad_output
            * G_inv
            @ (reference_point - euclidean.arithmetic_mean(data))
            @ G_inv
            / 2
            / std_scalar
        )
        # std = 0: zero subgradient instead of 0/0
        return (
            zero_where_zero_std(grad_input_data, std_scalar),
            zero_where_zero_std(grad_input_G, std_scalar),
        )


# --------------
# Harmonic curve
# --------------
def harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Curve for adaptive harmonic mean computation.

    .. math::

        \gamma(t) = \big((1-t)\,P_1^{-1} + t\,P_2^{-1}\big)^{-1}

    the harmonic analogue of
    :func:`~yetanotherspdnet.functions.spd_geometries.euclidean.euclidean_geodesic`: a Euclidean
    geodesic between the matrix inverses, inverted back.

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    return _harmonic_curve(point1, point2, t)[0]


def _harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Harmonic curve at t and the inverses of point1 and point2."""
    point1_inv = inv_SPD(point1)
    point2_inv = inv_SPD(point2)
    point_inv = euclidean.euclidean_geodesic(point1_inv, point2_inv, t)
    return inv_SPD(point_inv), point1_inv, point2_inv


class HarmonicCurve(Function):
    """Harmonic curve of two batches of SPD matrices."""

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the harmonic curve.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

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
        point, point1_inv, point2_inv = _harmonic_curve(point1, point2, t)
        ctx.t = t
        ctx.save_for_backward(point1_inv, point2_inv, point)
        return point

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the harmonic curve.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., n_features, n_features)

        grad_input2 : torch.Tensor of shape (..., n_features, n_features)
        """
        t = ctx.t
        point1_inv, point2_inv, point = ctx.saved_tensors
        return (
            (1 - t) * point1_inv @ point @ symmetrize(grad_output) @ point @ point1_inv,
            t * point2_inv @ point @ symmetrize(grad_output) @ point @ point2_inv,
            None,
        )


# -------------
# Harmonic mean
# -------------
def harmonic_mean(data: torch.Tensor) -> torch.Tensor:
    r"""
    Harmonic mean of a batch of SPD matrices.

    .. math::

        \bar{P} = \left(\frac{1}{N}\sum_{i=1}^{N} P_i^{-1}\right)^{-1}

    the inverse of the arithmetic mean of the matrix inverses.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Harmonic mean
    """
    if data.ndim == 2:
        return data
    return _harmonic_mean(data)[0]


def _harmonic_mean(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic mean of a batch (ndim > 2) and the inverses of the batch."""
    inv_data = inv_SPD(data)
    return inv_SPD(euclidean.arithmetic_mean(inv_data)), inv_data


class HarmonicMean(Function):
    """Harmonic mean."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the harmonic mean of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Harmonic mean
        """
        ctx.shape = data.shape
        if data.ndim == 2:
            return data
        mean, inv_data = _harmonic_mean(data)
        ctx.save_for_backward(inv_data, mean)
        return mean

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the harmonic mean of a batch of SPD matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the harmonic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        shape = ctx.shape
        if len(shape) == 2:
            return symmetrize(grad_output)
        n_matrices = math.prod(shape[:-2])
        inv_data, mean = ctx.saved_tensors
        tmp = symmetrize(mean @ grad_output @ mean)
        grad_input = symmetrize(inv_data @ tmp @ inv_data / n_matrices)
        return grad_input


# ---------------
# Scalar variance
# ---------------
def right_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    r"""
    Scalar standard deviation with respect to the right KL divergence.

    .. math::

        \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}\Big[
            \operatorname{tr}(P_i^{-1} G) + \log\det(P_i)\Big]
            - \log\det(G) - n

    Up to a factor 2, this is the average reverse Kullback-Leibler
    divergence :math:`\mathrm{KL}\big(\mathcal{N}(0, G) \,\|\,
    \mathcal{N}(0, P_i)\big)` — the roles of :math:`G` and :math:`P_i` are
    swapped relative to :func:`left_kullback_leibler_std_scalar`.

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
    return _right_kullback_leibler_std_scalar(data, reference_point)[0]


def _right_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right KL scalar std, the Cholesky factor of G and the inverses of data."""
    n_matrices = math.prod(data.shape[:-2])
    n_features = data.shape[-1]
    L_G = torch.linalg.cholesky(reference_point)
    L_data = torch.linalg.cholesky(data)
    data_inv = torch.cholesky_inverse(L_data)
    trace = torch.einsum("...ik,ki->", data_inv, reference_point)
    logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
    logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
    var = (trace + logdet_data) / n_matrices - logdet_G - n_features
    # clamp to avoid small numerical errors yielding small negative variance
    return std_from_variance(var), L_G, data_inv


class RightKullbackLeiblerStdScalar(Function):
    """Scalar standard deviation with respect to the right Kullback-Leibler divergence."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the right Kullback-Leibler scalar standard deviation.

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
        std_scalar, L_G, data_inv = _right_kullback_leibler_std_scalar(
            data, reference_point
        )
        ctx.n_matrices = math.prod(data.shape[:-2])
        ctx.save_for_backward(data, reference_point, L_G, data_inv, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the right Kullback-Leibler scalar standard deviation.

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
        data, reference_point, L_G, data_inv, std_scalar = ctx.saved_tensors
        G_inv = torch.cholesky_inverse(L_G)
        grad_input_data = (
            grad_output
            * data_inv
            @ (data - reference_point)
            @ data_inv
            / 2
            / std_scalar
            / n_matrices
        )
        grad_input_G = (
            grad_output * (euclidean.arithmetic_mean(data_inv) - G_inv) / 2 / std_scalar
        )
        # std = 0: zero subgradient instead of 0/0
        return (
            zero_where_zero_std(grad_input_data, std_scalar),
            zero_where_zero_std(grad_input_G, std_scalar),
        )


# moved to spd_geometries.euclidean
euclidean_geodesic = _deprecated("euclidean_geodesic", euclidean.euclidean_geodesic)
EuclideanGeodesic = _deprecated("EuclideanGeodesic", euclidean.EuclideanGeodesic)
arithmetic_mean = _deprecated("arithmetic_mean", euclidean.arithmetic_mean)
ArithmeticMean = _deprecated("ArithmeticMean", euclidean.ArithmeticMean)
