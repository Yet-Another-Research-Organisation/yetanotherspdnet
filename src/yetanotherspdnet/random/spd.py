import torch


def random_SPD(
    n_features: int,
    n_matrices: int = 1,
    cond: float = 10,
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

    generator : torch.Generator, optional
        Generator to ensure reproducibility. Default is None.

    Returns
    -------
    torch.Tensor of shape (n_matrices, n_features, n_features)
        Batch of SPD matrices
    """
    assert n_features >= 2, "Expected n_features>=2 to form matrices"
    # Generate random orthogonal matrices
    tmp = torch.randn((n_matrices, n_features, n_features), generator=generator)
    U, _, V = torch.svd(tmp)
    eigvecs = U @ V.transpose(-1, -2)

    eigvals = torch.zeros(n_matrices, n_features)
    eigvals[:, 0] = torch.sqrt(torch.tensor(cond))
    eigvals[:, 1] = 1 / torch.sqrt(torch.tensor(cond))
    if n_features > 2:
        eigvals[:, 2:] = torch.rand((n_matrices, n_features - 2), generator=generator)

    return torch.squeeze((eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2))
