import torch


def assert_spd(X: torch.Tensor) -> None:
    torch.testing.assert_close(X, X.T)
    eigvals = torch.linalg.eigvals(X).real
    assert torch.all(eigvals > 0), "Eigenvalues not positive !"
