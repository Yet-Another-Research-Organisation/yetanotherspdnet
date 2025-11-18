import torch

from ..functions.stiefel import stiefel_projection_qr


def _init_weights_stiefel(
    weight: torch.Tensor, generator: torch.Generator | None = None
) -> None:
    """
    Initialize weights on the Stiefel manifold.

    Parameters
    ----------
    weight : torch.Tensor of shape (..., n_in, n_out)
        Weights to be initialized in-place.
        We must have n_out > n_in

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.
    """
    if weight.shape[-1] > weight.shape[-2]:
        raise ValueError("Must have n_out < n_in")

    weight.data.copy_(
        stiefel_projection_qr(
            torch.randn(weight.shape, dtype=weight.dtype, device=weight.device, generator=generator)
        )
    )
