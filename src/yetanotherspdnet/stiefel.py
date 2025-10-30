from typing import Optional

import torch

def _init_weights_stiefel(weight: torch.Tensor, generator: Optional[torch.Generator] = None) -> None:
    """
    Initialize weights on the Stiefel manifold.

    Parameters
    ----------
    weight : torch.Tensor of shape (..., n_out, n_in)
        Weights to be initialized in-place.
        We must have n_out > n_in

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.
    """
    if weight.shape[-2] > weight.shape[-1]:
        raise ValueError("Must have n_out < n_in")
    
    U,_,V = torch.svd(torch.randn(
        weight.shape,
        dtype=weight.dtype,
        device=weight.device,
        generator=generator
    ))
    weight.data.copy_(U@V.transpose(-1,-2))
