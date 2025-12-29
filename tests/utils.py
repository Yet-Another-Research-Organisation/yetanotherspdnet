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


def is_orthogonal(W: torch.Tensor) -> bool:
    """
    Check whether a matrix is orthogonal

    Parameters
    ----------
    W : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (n_out <= n_in)

    Returns
    -------
    bool
        boolean
    """
    assert W.dim() == 2
    assert W.shape[1] <= W.shape[0]
    WWT = W.transpose(-1, -2) @ W
    identity_matrix = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    return bool(torch.allclose(WWT, identity_matrix))
