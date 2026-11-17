"""Two-stream (covariance + raw data) SPDnet layers."""

import torch
from torch import nn
from torch.nn.utils import parametrizations

from ..functions.rawandcov_linalg import CongruenceAndProjection
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
