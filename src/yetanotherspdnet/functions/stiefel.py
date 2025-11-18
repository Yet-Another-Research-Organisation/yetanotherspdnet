import torch
from torch.autograd import Function

from .spd_linalg import symmetrize

def stiefel_projection_polar(point: torch.Tensor) -> torch.Tensor:
    """
    Projection from the ambient space onto the Stiefel manifold based on the polar decomposition

    Parameters
    ----------
    point : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (with n_out <= n_in)

    Returns
    -------
    projected_point : torch.Tensor of shape (n_in, n_out)
        Orthogonal matrix
    """
    U, _, Vh = torch.linalg.svd(point, full_matrices=False)
    return U @ Vh


def stiefel_projection_tangent_orthogonal(vector: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection from the ambient space onto the tangent space
    of the Stiefel manifold at point

    Note that this also corresponds to both the differential and differential adjoint
    of projection_stiefel_polar

    Parameters
    ----------
    vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (direction)

    point : torch.Tensor of shape (n_in, n_out)
        Orthogonal matrix (with n_out <= n_in)

    Returns
    -------
    tangent_vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix
    """
    return vector - point @ symmetrize(point.transpose(-2, -1) @ vector)


class StiefelProjectionPolar(Function):
    """
    Projection from the ambient space onto the Stiefel manifold based on polar decomposition
    """
    @staticmethod
    def forward(ctx, point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection onto the Stiefel manifold

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix (with n_out <= n_in)

        Returns
        -------
        projected_point : torch.Tensor of shape (n_in, n_out)
            Orthogonal matrix
        """
        projected_point = stiefel_projection_polar(point)
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

        grad_output : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the projected orthogonal matrix

        Returns
        -------
        grad_input : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the input rectangular matrix
        """
        projected_point, = ctx.saved_tensors
        return stiefel_projection_tangent_orthogonal(grad_output, projected_point)


def stiefel_projection_qr(point: torch.Tensor) -> torch.Tensor:
    """
    Projection from the ambient space onto the Stiefel manifold based on the QR decomposition

    Parameters
    ----------
    point : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (with n_out <= n_in)

    Returns
    -------
    projected_point : torch.Tensor of shape (n_in, n_out)
        Orthogonal matrix
    """
    Q, _ = torch.linalg.qr(point)
    return Q


def stiefel_differential_projection_qr(vector: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Differential of the projection on Stiefel based on QR decomposition

    Parameters
    ----------
    vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (direction)

    Q : torch.Tensor of shape (n_in, n_out)
        Q factor of RQ decomposition of point (with n_out <= n_in)

    R : torch.Tensor of shape (n_out, n_out)
        R factor of QR decomposition of point (upper triangular)

    Returns
    -------
    tangent_vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix
    """
    tmp = torch.tril(torch.linalg.solve_triangular(R, Q.transpose(-2,-1) @ vector, upper=True, left=False))
    tmp = tmp - tmp.transpose(-2, -1)
    return vector - Q @ Q.transpose(-2, -1) @ vector + Q @ tmp


def stiefel_adjoint_differential_projection_qr(vector: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Adjoint of the differential projection on Stiefel based on QR decomposition

    Parameters
    ----------
    vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (direction)

    Q : torch.Tensor of shape (n_in, n_out)
        Q factor of RQ decomposition of point (with n_out <= n_in)

    R : torch.Tensor of shape (n_out, n_out)
        R factor of QR decomposition of point (upper triangular)

    Returns
    -------
    transformed_vector : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix
    """
    tmp = Q.transpose(-2,-1) @ vector
    tmp = torch.tril(tmp - tmp.transpose(-2,-1))
    return (
        vector - Q @ Q.transpose(-2,-1) @ vector
        + torch.linalg.solve_triangular(R.transpose(-2,-1), Q @ tmp, upper=False, left=False)
    )


class StiefelProjectionQR(Function):
    """
    Projection from the ambient space onto the Stiefel manifold based on QR decomposition
    """
    @staticmethod
    def forward(ctx, point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection onto the Stiefel manifold

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix (with n_out <= n_in)

        Returns
        -------
        projected_point : torch.Tensor of shape (n_in, n_out)
            Orthogonal matrix
        """
        Q, R = torch.linalg.qr(point)
        ctx.save_for_backward(Q, R)
        return Q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the projection onto the Stiefel manifold

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the projected orthogonal matrix

        Returns
        -------
        grad_input : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the input rectangular matrix
        """
        Q, R = ctx.saved_tensors
        return stiefel_adjoint_differential_projection_qr(grad_output, Q, R)

