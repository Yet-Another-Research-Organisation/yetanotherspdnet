import torch
from torch.autograd import Function

from ..spd_linalg import symmetrize


# ------------------
# Euclidean geodesic
# ------------------
def euclidean_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Euclidean geodesic:
    (1-t)*point1 + t*point2

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
    return symmetrize((1 - t) * point1 + t * point2)


# ---------------
# Arithmetic mean
# ---------------
def arithmetic_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Arithmetic mean of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean
    """
    if data.ndim == 2:
        return data
    return symmetrize(torch.real(torch.mean(data, dim=tuple(range(data.ndim - 2)))))


class ArithmeticMean(Function):
    """
    Arithmetic mean
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the arithmetic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Arithmetic mean
        """
        ctx.shape = data.shape
        if data.ndim == 2:
            return symmetrize(data)
        return symmetrize(torch.mean(data, dim=tuple(range(data.ndim - 2))))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the arithmetic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the arithmetic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        shape = ctx.shape
        if len(shape) == 2:
            return symmetrize(grad_output)
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        grad_input = torch.zeros(shape, device=grad_output.device)
        grad_input[..., :, :] = symmetrize(grad_output / n_matrices)
        return grad_input


# --------------
# Harmonic curve
# --------------
def harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Curve for adaptive harmonic mean computation:
    ((1-t)*point1^{-1} + t*point2^{-1})^{-1}

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
    point1_inv = torch.cholesky_inverse(torch.linalg.cholesky(point1))
    point2_inv = torch.cholesky_inverse(torch.linalg.cholesky(point2))
    point_inv = euclidean_geodesic(point1_inv, point2_inv, t)
    return torch.cholesky_inverse(torch.linalg.cholesky(point_inv))


# -------------
# Harmonic mean
# -------------
def harmonic_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Harmonic mean of a batch of SPD matrices

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
    inv_data = torch.cholesky_inverse(torch.linalg.cholesky(data))
    inv_mean = arithmetic_mean(inv_data)
    return torch.cholesky_inverse(torch.linalg.cholesky(inv_mean))


class HarmonicMean(Function):
    """
    Harmonic mean
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the harmonic mean of a batch of SPD matrices

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
        inv_data = torch.cholesky_inverse(torch.linalg.cholesky(data))
        inv_mean = arithmetic_mean(inv_data)
        mean = torch.cholesky_inverse(torch.linalg.cholesky(inv_mean))
        ctx.save_for_backward(inv_data, mean)
        return mean

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the harmonic mean of a batch of SPD matrices

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
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        inv_data, mean = ctx.saved_tensors
        tmp = symmetrize(mean @ grad_output @ mean)
        grad_input = symmetrize(inv_data @ tmp @ inv_data / n_matrices)
        return grad_input
