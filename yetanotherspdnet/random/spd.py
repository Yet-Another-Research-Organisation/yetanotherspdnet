import torch

from .stiefel import random_stiefel


def random_DPD(
    n_features: int,
    n_matrices: int = 1,
    cond: float = 10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate a batch of random diagonal positive definite matrices

    Parameters
    ----------
    n_features : int
        Number of features

    n_matrices : int, optional
        Number of matrices. Default is 1

    cond : float, optional
        Condition number w.r.t. inversion of SPD matrices.
        Default is 10

    device : torch.device, optional
        Torch device. Default is None.

    dtype : torch.dtype, optional
        Torch dtype. Default is None.

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.

    Returns
    -------
    torch.Tensor of shape (n_matrices, n_features, n_features)
        Batch of DPD matrices
    """
    assert n_features >= 2, "Expected n_features>=2 to form matrices"

    diagvals = torch.zeros(n_matrices, n_features, device=device, dtype=dtype)
    sqrt_cond = torch.sqrt(torch.tensor(cond, device=device, dtype=dtype))
    diagvals[:, 0] = sqrt_cond
    diagvals[:, 1] = 1 / sqrt_cond
    if n_features > 2:
        diagvals[:, 2:] = 1 / sqrt_cond + (sqrt_cond - 1 / sqrt_cond) * torch.rand(
            (n_matrices, n_features - 2),
            device=device,
            dtype=dtype,
            generator=generator,
        )
    # randomly permute elements in each row of diagvals
    dumb_random_values = torch.rand(
        diagvals.shape, device=device, dtype=dtype, generator=generator
    )
    randperm_indices = torch.argsort(dumb_random_values, dim=1)
    return torch.squeeze(torch.diag_embed(torch.gather(diagvals, 1, randperm_indices)))


def random_SPD(
    n_features: int,
    n_matrices: int = 1,
    cond: float = 10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate a batch of random SPD matrices

    Parameters
    ----------
    n_features : int
        Number of features

    n_matrices : int, optional
        Number of matrices. Default is 1

    cond : float, optional
        Condition number w.r.t. inversion of SPD matrices.
        Default is 10

    device : torch.device, optional
        Torch device. Default is None.

    dtype : torch.dtype, optional
        Torch dtype. Default is None.

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.

    Returns
    -------
    torch.Tensor of shape (n_matrices, n_features, n_features)
        Batch of SPD matrices
    """
    assert n_features >= 2, "Expected n_features>=2 to form matrices"

    eigvecs = random_stiefel(
        n_features,
        n_features,
        n_matrices,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    eigvals_mats = random_DPD(
        n_features,
        n_matrices,
        cond=cond,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return torch.squeeze(eigvecs @ eigvals_mats @ eigvecs.transpose(-1, -2))
