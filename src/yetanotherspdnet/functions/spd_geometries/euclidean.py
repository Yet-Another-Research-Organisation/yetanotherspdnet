"""Euclidean geometry of symmetric matrices: geodesic and arithmetic mean.

The Euclidean geodesic and the arithmetic mean are the building blocks of the
left Kullback-Leibler (arithmetic) and harmonic geometries and of the
log-Euclidean one (in the logarithm domain).
"""

import math

import torch
from torch.autograd import Function

from ..spd_linalg import symmetrize


# ------------------
# Euclidean geodesic
# ------------------
def euclidean_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Euclidean geodesic between two symmetric matrices.

    .. math::

        \gamma(t) = (1-t)\,P_1 + t\,P_2, \quad t \in [0, 1]

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices
    """
    return symmetrize((1 - t) * point1 + t * point2)


class EuclideanGeodesic(Function):
    """Euclidean geodesic of a batch of symmetric matrices."""

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Euclidean geodesic.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point1 : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices

        point2 : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices

        t : float | torch.Tensor
            parameter on the path, should be in [0,1]

        Returns
        -------
        point : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices
        """
        ctx.t = t
        return euclidean_geodesic(point1, point2, t)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the Euclidean geodesic.

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
        return symmetrize((1 - t) * grad_output), symmetrize(t * grad_output), None


# ---------------
# Arithmetic mean
# ---------------
def arithmetic_mean(data: torch.Tensor) -> torch.Tensor:
    r"""
    Arithmetic (Euclidean) mean of a batch of symmetric matrices.

    .. math::

        \bar{P} = \frac{1}{N}\sum_{i=1}^{N} P_i

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean
    """
    if data.ndim == 2:
        return data
    return symmetrize(torch.mean(data, dim=tuple(range(data.ndim - 2))))


class ArithmeticMean(Function):
    """Arithmetic mean."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the arithmetic mean of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Arithmetic mean
        """
        ctx.shape = data.shape
        if data.ndim == 2:
            return symmetrize(data)
        return arithmetic_mean(data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the arithmetic mean of a batch of symmetric matrices.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the arithmetic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of symmetric matrices
        """
        shape = ctx.shape
        if len(shape) == 2:
            return symmetrize(grad_output)
        n_matrices = math.prod(shape[:-2])
        grad_input = torch.zeros(
            shape, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_input[..., :, :] = symmetrize(grad_output / n_matrices)
        return grad_input
