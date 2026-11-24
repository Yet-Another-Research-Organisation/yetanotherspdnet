"""Two-stream (covariance + raw data) SPDnet layers."""

import torch
from torch import nn
from torch.nn.utils import parametrizations

from ..functions.rawandcov_linalg import (
    CongruenceAndProjection,
    EighReLuAndProjection,
    EighReLuAndWhitening,
)
from ..random.stiefel import _init_weights_stiefel


class BiMapTwoStream(nn.Module):
    r"""
    Two-stream BiMap layer: :math:`(\Sigma, X) \mapsto (W^\top \Sigma W, W^\top X)`.

    The same weight :math:`W \in \mathbb{R}^{n_{in} \times n_{out}}`, on the
    Stiefel manifold, is applied to the SPD matrices (congruence) and to the
    associated raw data (projection). If :math:`\Sigma = \frac{1}{n} X X^\top`,
    the outputs satisfy :math:`P = \frac{1}{n} Y Y^\top`.

    The orthogonality constraint :math:`W^\top W = I` is enforced with the
    orthogonal parametrization of torch.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        parametrized: bool = True,
        parametrization_options: dict | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Build a two-stream BiMap layer.

        Parameters
        ----------
        n_in : int
            Number of input features (p1 in the note)

        n_out : int
            Number of output features (p2 in the note), n_out <= n_in

        parametrized : bool, optional
            Whether to enforce the orthogonality constraint W^T W = I with
            the orthogonal parametrization of torch. Default is True

        parametrization_options : dict, optional
            Options for torch.nn.utils.parametrizations.orthogonal
            (e.g., orthogonal_map). Default is None

        device : torch.device, optional
            Device on which the layer is initialized.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None
        """
        super().__init__()
        assert n_out <= n_in, "must have n_out <= n_in"
        self.n_in = n_in
        self.n_out = n_out
        self.parametrized = parametrized
        self.parametrization_options = parametrization_options
        self.device = device
        self.dtype = dtype
        self.generator = generator

        # create the weight W (n_in x n_out) and initialize it on the Stiefel manifold
        self.weight = nn.Parameter(
            torch.empty((n_in, n_out), dtype=dtype, device=device), requires_grad=True
        )
        with torch.no_grad():
            _init_weights_stiefel(self.weight, generator=self.generator)

        # constraint W^T W = I with the orthogonal parametrization of torch
        if self.parametrized:
            parametrizations.orthogonal(
                module=self, name="weight", **(self.parametrization_options or {})
            )

        # two-stream BiMap function, with the hand-derived backward
        self.bimap_fun = CongruenceAndProjection.apply

    def forward(
        self, data_cov: torch.Tensor, data_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream BiMap layer

        Parameters
        ----------
        data_cov : torch.Tensor of shape (..., n_in, n_in)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_in, n_samples)
            Batch of raw data associated with the SPD matrices

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_out, n_out)
            Batch of transformed SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_out, n_samples)
            Batch of projected raw data
        """
        # apply the same W to both streams
        return self.bimap_fun(data_cov, data_raw, self.weight)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BiMapTwoStream(n_in={self.n_in}, n_out={self.n_out}, "
            f"parametrized={self.parametrized}, "
            f"parametrization_options={self.parametrization_options}, "
            f"device={self.device}, dtype={self.dtype}, generator={self.generator})"
        )


class ReEigTwoStream(nn.Module):
    r"""
    Two-stream ReEig layer: :math:`(\Sigma, X) \mapsto (Q, V)` with
    :math:`Q = U \max(\varepsilon I, \Lambda) U^\top` and, depending on
    ``mode``,

    - ``"projection"``: :math:`V = U\, \mathrm{ReLU}(U^\top X)`;
    - ``"whitening_Q"``: :math:`V = Q^{1/2}\, \mathrm{ReLU}(Q^{-1/2} X)`;
    - ``"whitening_Sigma"``:
      :math:`V = \Sigma^{1/2}\, \mathrm{ReLU}(\Sigma^{-1/2} X)`,

    where :math:`\Sigma = U \Lambda U^\top`. The SPD stream is the classical
    ReEig layer. The layer has no parameter.
    """

    _MODES = ("projection", "whitening_Q", "whitening_Sigma")

    def __init__(
        self, eps: float = 1e-4, mode: str = "whitening_Q", dim: int | None = None
    ) -> None:
        """
        Build a two-stream ReEig layer.

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-4

        mode : str, optional
            Transformation of the raw data: "projection", "whitening_Q" or
            "whitening_Sigma". Default is "whitening_Q"

        dim : int, optional
            Dimension of the SPD matrices. Default is None.
            Used for logging purposes.
        """
        super().__init__()
        assert mode in self._MODES, f"mode must be one of {self._MODES}"
        self.eps = eps
        self.mode = mode
        self.dim = dim

    def forward(
        self, data_cov: torch.Tensor, data_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream ReEig layer

        Parameters
        ----------
        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data associated with the SPD matrices

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of transformed raw data
        """
        if self.mode == "projection":
            return EighReLuAndProjection.apply(data_cov, data_raw, self.eps)
        whitening = "Q" if self.mode == "whitening_Q" else "Sigma"
        return EighReLuAndWhitening.apply(data_cov, data_raw, self.eps, whitening)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"ReEigTwoStream(eps={self.eps}, mode={self.mode!r}, dim={self.dim})"
