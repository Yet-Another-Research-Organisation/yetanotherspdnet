import torch
from torch.autograd import Function

from .spd_linalg import symmetrize

def projection_stiefel(point: torch.Tensor) -> torch.Tensor:
    """
    Projection from the ambient space onto the Stiefel manifold
    (relies on the polar decomposition)

    Parameters
    ----------
    point : torch.Tensor of shape (n_out, n_in)
        Rectangular matrix (with n_out < n_in)

    Returns
    -------
    projected_point : torch.Tensor of shape (n_out, n_in)
        Orthogonal matrix
    """
    U, _, Vh = torch.linalg.svd(point, full_matrices=False)
    return U @ Vh


def projection_tangent_stiefel(vector: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection from the ambient space onto the tangent space
    of the Stiefel manifold at point

    Parameters
    ----------
    vector : torch.Tensor of shape (n_out, n_in)
        Ambient vector (rectangular matrix)

    point : torch.Tensor of shape (n_out, n_in)
        Orthogonal matrix (with n_out < n_in)

    Returns
    -------
    tangent_vector : torch.Tensor of shape (n_out, n_in)
        Tangent vector at point
    """
    return vector - symmetrize(vector @ point.transpose(-2, -1)) @ point


class StiefelProjection(Function):
    """
    Projection from the ambient space onto the Stiefel manifold
    (relies on the polar decomposition)
    """

