import torch



def is_symmetric(X: torch.Tensor) -> bool:
    """
    Check whether matrices in a batch are symmetric

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrices

    Returns
    -------
    bool
        boolean
    """
    is_square = bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    return bool(torch.allclose(X, X.transpose(-1, -2)))


def is_spd(X: torch.Tensor) -> bool:
    """
    Check whether matrices in a batch are SPD

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_features, n_features)
        batch of matrices

    Returns
    -------
    bool
        boolean
    """
    is_square = bool(X.shape[-1] == X.shape[-2])
    if not is_square:
        return False
    is_symmetric = bool(torch.allclose(X, X.transpose(-1, -2)))
    is_positivedefinite = bool((torch.linalg.eigvalsh(X) > 0).all())
    return is_symmetric and is_positivedefinite
