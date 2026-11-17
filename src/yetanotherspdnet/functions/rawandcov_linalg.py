"""Linear algebra for the two-stream (covariance + raw data) SPDnet layers."""

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.spd_linalg import symmetrize


# ------------------------------------------
# Two-stream BiMap: congruence + projection
# ------------------------------------------
def congruence_and_projection(
    data_cov: torch.Tensor, data_raw: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream BiMap: the same rectangular matrix is
    applied to a batch of SPD matrices (congruence) and to the associated
    raw data (projection).

    .. math::
        P = W^\top \Sigma\, W, \qquad Y = W^\top X,
        \qquad W \in \mathbb{R}^{n_{in} \times n_{out}}

    with :math:`n_{in} \geq n_{out}`. If :math:`\Sigma = \frac{1}{n} X X^\top`,
    then :math:`P = \frac{1}{n} Y Y^\top`: the link between both streams is kept.

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_in, n_in)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_in, n_samples)
        Batch of raw data associated with the SPD matrices

    weight : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (e.g., weights), n_in >= n_out is expected

    Returns
    -------
    data_cov_transformed : torch.Tensor of shape (..., n_out, n_out)
        Transformed batch of SPD matrices

    data_raw_transformed : torch.Tensor of shape (..., n_out, n_samples)
        Projected batch of raw data
    """
    weight_t = weight.transpose(-1, -2)
    return weight_t @ data_cov @ weight, weight_t @ data_raw


class CongruenceAndProjection(Function):
    """
    Two-stream BiMap: congruence of a batch of SPD matrices and projection of
    the associated raw data with the same (full-rank) rectangular matrix
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream BiMap

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data_cov : torch.Tensor of shape (..., n_in, n_in)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_in, n_samples)
            Batch of raw data

        weight : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix (e.g., weights), n_in >= n_out is expected

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_out, n_out)
            Transformed batch of SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_out, n_samples)
            Projected batch of raw data
        """
        ctx.save_for_backward(data_cov, data_raw, weight)
        return congruence_and_projection(data_cov, data_raw, weight)

    @staticmethod
    def backward(
        ctx, grad_output_cov: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Backward pass of the two-stream BiMap.

        With :math:`E_k = \partial \ell / \partial P_k` (symmetric) and
        :math:`F_k = \partial \ell / \partial Y_k`:

        .. math::
            \frac{\partial \ell}{\partial W} = \sum_k 2\, \Sigma_k W E_k + X_k F_k^\top,
            \qquad
            \frac{\partial \ell}{\partial \Sigma_k} = W E_k W^\top,
            \qquad
            \frac{\partial \ell}{\partial X_k} = W F_k

        (``weight`` here is the transpose of :math:`W` in the two-stream BiMap note.)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_cov : torch.Tensor of shape (..., n_out, n_out)
            Gradient of the loss with respect to the transformed SPD matrices

        grad_output_raw : torch.Tensor of shape (..., n_out, n_samples)
            Gradient of the loss with respect to the projected raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_in, n_in)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_in, n_samples)
            Gradient of the loss with respect to the input raw data

        grad_input_W : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the rectangular matrix
        """
        # retrieve inputs saved during the forward pass
        data_cov, data_raw, weight = ctx.saved_tensors

        # error transmitted to Sigma_k: W E_k W^T
        grad_input_cov = weight @ grad_output_cov @ weight.transpose(-1, -2)

        # error transmitted to X_k: W F_k
        grad_input_raw = weight @ grad_output_raw

        # error to update W (transposed since weight stores W^T of the note):
        # sum over the batch of 2 Sigma_k W E_k (SPD stream) + X_k F_k^T (raw stream).
        # E_k is symmetrized because the factor 2 assumes E_k symmetric
        grad_input_W = 2 * torch.einsum(
            "...ik,kl,...lj->ij", data_cov, weight, symmetrize(grad_output_cov)
        ) + torch.einsum("...ik,...jk->ij", data_raw, grad_output_raw)

        # one gradient per input of forward, in the same order
        return grad_input_cov, grad_input_raw, grad_input_W
