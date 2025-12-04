import torch

from ..functions.stiefel import stiefel_projection_qr


def random_stiefel(
    n_in: int,
    n_out: int,
    n_matrices: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Random point on Stiefel manifold

    Parameters
    ----------
    n_in: int
        Number of rows

    n_out: int
        Number of columns

    n_matrices: int, optional
        Number of matrices. Default is 1

    device : torch.device, optional
        Torch device. Default is None

    dtype : torch.dtype, optional
        Torch dtype. Default is None

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None
    """
    return stiefel_projection_qr(
        torch.squeeze(
            torch.randn(
                (n_matrices, n_in, n_out),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    )


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
            torch.randn(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device,
                generator=generator,
            )
        )
    )
