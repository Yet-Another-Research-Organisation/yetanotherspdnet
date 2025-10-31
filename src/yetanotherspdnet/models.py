# FileName: models.py
# Date: 29 juin 2023 - 18:13
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Models used in the project
# =========================================


import zlib

import torch
from torch import nn

from yetanotherspdnet.nn import BatchNormSPDMean, BiMap, LogEig, ReEig, Vec


class SPDNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers_size: list[int],
        output_dim: int,
        softmax: bool = False,
        eps: float = 1e-3,
        batchnorm: bool = False,
        batchnorm_method: str = "geometric_arithmetic_harmonic",
        seed: int | None = None,
        device: torch.device | None = None,
        precision: torch.dtype | None = torch.float64,
        max_iter_batchnorm: int = 10,
        orthogonal_map: str | None = None,
        momentum: float = 0.01,
    ) -> None:
        """Standard SPDNet model with hidden layers

        Parameters
        ----------
        input_dim : int
            Input dimension of SPDNet

        hidden_layers_size : List[int]
            List of hidden layer sizes

        output_dim : int
            Output dimension of SPDNet

        softmax : bool, optional
            Whether to apply softmax to output, by default False

        eps : float, optional
            Regularization value for ReEig, by default 1e-3.

        batchnorm : bool, optional
            Whether to apply batchnorm to hidden layers, by default False

        batchnorm_method : str, optional
            Method for SPD mean computation in batchnorm, by default 'geometric_arithmetic_harmonic'

        seed : int, optional
            Random seed for layers initialization, by default None

        device : torch.device, optional
            Device to run model on, by default None = torch.device('cpu')

        precision : torch.dtype, optional
            Precision of model, by default torch.float64

        max_iter_batchnorm : int, optional
            Maximum number of iterations for mean computation batchnorm, by default 10.

        orthogonal_map : str, optional
            Method for orthogonal parametrization. Options include 'matrix_exp',
            'cayley', 'householder', etc. Default is None (uses PyTorch default).

        momentum : float, optional
            Momentum for running mean update in batchnorm, by default 0.01.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim
        self.softmax = softmax
        self.eps = eps
        self.seed = seed
        self.device = device if device is not None else torch.device("cpu")
        self.precision = precision
        self.batchnorm = batchnorm
        self.batchnorm_method = batchnorm_method
        self.max_iter_batchnorm = max_iter_batchnorm
        self.orthogonal_map = orthogonal_map
        self.momentum = momentum

        # Initialize seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Create layers
        spdnet_layers = [
            BiMap(
                n_in=input_dim,
                n_out=hidden_layers_size[0],
                seed=self.seed,
                device=self.device,
                dtype=self.precision,
                parametrization_options={"orthogonal_map": self.orthogonal_map}
                if self.orthogonal_map
                else None,
            ),
        ]

        spdnet_layers.append(ReEig(eps=eps, dim=hidden_layers_size[0]))

        if batchnorm:
            spdnet_layers.append(
                BatchNormSPDMean(
                    n_features=hidden_layers_size[0],
                    mean_type=self.batchnorm_method,
                    momentum=self.momentum,
                )
            )
        for i in range(1, len(hidden_layers_size)):
            spdnet_layers.append(
                BiMap(
                    n_in=hidden_layers_size[i - 1],
                    n_out=hidden_layers_size[i],
                    seed=self.seed,
                    device=self.device,
                    dtype=self.precision,
                    parametrization_options={"orthogonal_map": self.orthogonal_map}
                    if self.orthogonal_map
                    else None,
                )
            )
            spdnet_layers.append(ReEig(eps=eps, dim=hidden_layers_size[i]))

            if batchnorm:
                spdnet_layers.append(
                    BatchNormSPDMean(
                        n_features=hidden_layers_size[i],
                        mean_type=self.batchnorm_method,
                        momentum=self.momentum,
                    )
                )
        spdnet_layers.append(LogEig())

        self.spdnet_layers = nn.Sequential(*spdnet_layers)

        # Create final layer(s)
        self.vectorization = Vec()
        self.linear = nn.Linear(
            hidden_layers_size[-1] ** 2,
            output_dim,
            dtype=self.precision,
            device=self.device,
        )

        if self.softmax:
            self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDNet

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (..., input_dim, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim)
        """
        # Run through SPDNet layers
        X = self.spdnet_layers(X)
        # Run through final layer(s)
        X = self.vectorization(X)
        X = self.linear(X)
        # Apply softmax if required
        if self.softmax:
            X = self.softmax_layer(X)
        return X

    def __repr__(self) -> str:
        """Representation of SPDNet
        Returns
        -------
        str
            String representation of SPDNet
        """
        return (
            f"SPDNet(input_dim={self.input_dim}, "
            f"hidden_layers_size={self.hidden_layers_size}, "
            f"output_dim={self.output_dim}, softmax={self.softmax}, "
            f"eps={self.eps}, seed={self.seed}, device={self.device}, "
            f"precision={self.precision}, batchnorm={self.batchnorm}, "
            f"batchnorm_method={self.batchnorm_method}, "
            f"max_iter_batchnorm={self.max_iter_batchnorm}, "
            f"momentum={self.momentum}, orthogonal_map={self.orthogonal_map})"
        )

    def __str__(self) -> str:
        """String representation of SPDNet

        Returns
        -------
        str
            String representation of SPDNet
        """
        return self.__repr__()

    def show_layers(self) -> str:
        """Prints out the layers of SPDNet"""
        string = self.__repr__()
        string += "\n\nSPDNet Layers:\n"
        string += "---------------\n"
        for i, layer in enumerate(self.spdnet_layers):
            string += f"    ({i}). {layer}\n"
        string += f"    ({len(self.spdnet_layers) + 1}). LogEig()\n"
        string += f"    ({len(self.spdnet_layers) + 2}). {self.linear}\n"
        if self.softmax:
            string += f"    ({len(self.spdnet_layers) + 3}). Softmax()\n"
        print(string)
        return string

    def get_last_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Returns the last tensor of SPDNet rather than the output of the
        final layer

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (..., input_dim, input_dim)

        Returns
        -------
        torch.Tensor
            Last tensor of SPDNet
        """
        X = self.spdnet_layers(X)
        return X

    def create_model_name_hash(self) -> str:
        """Creates a very short hash of the model name based on the model parameters
        Returns
        -------
        str
            Short hash of model name (8 characters)
        """
        # CRC32 hash - 8 characters, very short and fast
        crc_hash = zlib.crc32(self.__str__().encode("utf-8")) & 0xFFFFFFFF
        self.model_hash = f"{crc_hash:08x}"  # 8 hex characters
        return self.model_hash

    def get_model_hash(self) -> str:
        """Returns the model hash
        Returns
        -------
        str
            Model hash
        """
        if not hasattr(self, "model_hash"):
            self.create_model_name_hash()
        return self.model_hash
