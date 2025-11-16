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


class ProjectionStiefel(Function):
    """
    Projection from the ambient space onto the Stiefel manifold
    (relies on the polar decomposition)
    """
    @staticmethod
    def forward(ctx, point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection onto the Stiefel manifold

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point : torch.Tensor of shape (n_out, n_in)
            Rectangular matrix (with n_out < n_in)

        Returns
        -------
        projected_point : torch.Tensor of shape (n_out, n_in)
            Orthogonal matrix
        """
        projected_point = projection_stiefel(point)
        ctx.save_for_backward(projected_point)
        return projected_point

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the projection onto the Stiefel manifold

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_out, n_in)
            Gradient of the loss with respect to the projected orthogonal matrix

        Returns
        -------
        grad_input : torch.Tensor of shape (n_out, n_in)
            Gradient of the loss with respect to the input rectangular matrix
        """
        projected_point, = ctx.saved_tensors
        return projection_tangent_stiefel(grad_output, projected_point)
