import zlib

from collections.abc import Callable

import torch
from torch import nn
from torch.nn.utils import parametrizations

from yetanotherspdnet.nn.base import BiMap, LogEig, ReEig, Vec, Vech
from yetanotherspdnet.nn.batchnorm import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
)


class SPDnet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers_size: list[int],
        output_dim: int,
        softmax: bool = False,
        reeig_eps: float = 1e-3,
        bimap_parametrized: bool = True,
        bimap_parametrization: type[nn.Module] | Callable = parametrizations.orthogonal,
        bimap_parametrization_options: dict | None = None,
        batchnorm: bool = False,
        batchnorm_type: str = "mean_only",
        batchnorm_mean_type: str = "geometric_arithmetic_harmonic",
        batchnorm_mean_options: dict | None = None,
        batchnorm_momentum: float = 0.01,
        batchnorm_norm_strategy: str = "classical",
        batchnorm_minibatch_mode: str = "constant",
        batchnorm_minibatch_momentum: float = 0.01,
        batchnorm_minibatch_maxstep: int = 100,
        batchnorm_parametrization: str = "softplus",
        vec_type: str = "vec",
        use_logeig: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
        use_autograd: bool | dict = False,
    ) -> None:
        """
        Standard SPDnet model with hidden layers

        Parameters
        ----------
        input_dim : int
            Input dimension of SPDNet

        hidden_layers_size : List[int]
            List of hidden layer sizes

        output_dim : int
            Output dimension of SPDNet

        softmax : bool, optional
            Whether to apply softmax to output. Default is False

        reeig_eps : float, optional
            Regularization value for ReEig. Default is 1e-3

        bimap_parametrized : bool, optional
            Whether to apply parametrization to enforce manifold constraints in BiMap.
            Default is True

        bimap_parametrization : nn.Module or Callable, optional
            Parametrization to apply in BiMap if bimap_parametrized is True.
            If not nn.Module, only parametrization.orthogonal is supported.
            Default is parametrization.orthogonal

        bimap_parametrization_options : dict, optional
            Options for the parametrization function in BiMap.
            Default is None

        batchnorm : bool, optional
            Whether to apply BatchNormSPDMean to hidden layers. Default is False

        batchnorm_type : str, optional
            The type of batch normalization layer to use.
            Default is "mean_only".
            Choices are: "mean_only" and "mean_var_scalar"

        batchnorm_mean_type : str, optional
            Choice of SPD mean in BatchNormSPDMean. Default is "affine_invariant".
            Choices are: "affine_invariant", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic"

        batchnorm_mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        batchnorm_momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        batchnorm_norm_strategy : str, optional
            Strategy for normalization.
            Default is "classical".
            Choices are: "classical" and "minibatch"

        batchnorm_minibatch_mode : str, optional
            How the minibatch momentum behaves during the training.
            Default is "constant".
            Choices are: "constant", "decay", "growth"

        batchnorm_minibatch_momentum : float, optional
            Momentum for mean regularization in minibatch normalization strategy
            Default is 0.01

        batchnorm_minibatch_maxstep : int, optional
            If minibatch_mode is "decay" or "growth", this is the training step at which the minibatch momentum
            attains its final value.
            Default is 100

        batchnorm_parametrization : str, optional
            Parametrization to apply on covariance bias.
            Default is "softplus".
            Choices are: "softplus", "exp"

        vec_type : str, optional
            Whether to use Vec or Vech module.
            Default is "vec".
            Choices are: "vec", "vech"

        use_logeig : bool, optional
            Whether to apply LogEig layer before vectorization.
            Default is True

        device : torch.device, optional
            Device to run model on. Default is torch.device('cpu')

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None

        use_autograd : bool | dict, optional
            Use torch autograd for gradient computation. Can be bool for all layers,
            or dict with keys: 'bimap', 'reeig', 'logeig', 'batchnorm', 'vec'.
            Note that Vech module always uses manual gradient.
            Default is False
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim
        self.softmax = softmax

        self.reeig_eps = reeig_eps

        self.bimap_parametrized = bimap_parametrized
        self.bimap_parametrization = bimap_parametrization
        self.bimap_parametrization_options = bimap_parametrization_options

        self.batchnorm = batchnorm
        self.batchnorm_type = batchnorm_type
        assert self.batchnorm_type in ["mean_only", "mean_var_scalar"], (
            f"expected formula in ['mean_only', 'mean_var_scalar'], got {self.batchnorm_type}"
        )
        self.batchnorm_mean_type = batchnorm_mean_type
        self.batchnorm_mean_options = batchnorm_mean_options
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_norm_strategy = batchnorm_norm_strategy
        self.batchnorm_minibatch_mode = batchnorm_minibatch_mode
        self.batchnorm_minibatch_momentum = batchnorm_minibatch_momentum
        self.batchnorm_minibatch_maxstep = batchnorm_minibatch_maxstep
        self.batchnorm_parametrization = batchnorm_parametrization

        self.vec_type = vec_type
        assert self.vec_type in ["vec", "vech"], (
            f"vec_type must be 'vec' or 'vech', got {self.vec_type}"
        )

        self.use_logeig = use_logeig
        self.device = device
        self.dtype = dtype
        self.generator = generator

        # Handle use_autograd as bool or dict
        if isinstance(use_autograd, bool):
            self.use_autograd = {
                "bimap": use_autograd,
                "reeig": use_autograd,
                "logeig": use_autograd,
                "batchnorm": use_autograd,
                "vec": use_autograd,
            }
        else:
            # Default all to False, then update with provided values
            self.use_autograd = {
                "bimap": False,
                "reeig": False,
                "logeig": False,
                "batchnorm": False,
                "vec": False,
            }
            self.use_autograd.update(use_autograd)

        # Store original for compatibility
        self._use_autograd_original = use_autograd

        # Create layers
        spdnet_layers: list[nn.Module] = [
            BiMap(
                n_in=self.input_dim,
                n_out=self.hidden_layers_size[0],
                parametrized=self.bimap_parametrized,
                parametrization=self.bimap_parametrization,
                parametrization_options=self.bimap_parametrization_options,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
                use_autograd=self.use_autograd["bimap"],
            )
        ]

        spdnet_layers.append(
            ReEig(
                eps=self.reeig_eps,
                dim=self.hidden_layers_size[0],
                use_autograd=self.use_autograd["reeig"],
            )
        )

        if batchnorm:
            if self.batchnorm_type == "mean_only":
                spdnet_layers.append(
                    BatchNormSPDMean(
                        n_features=self.hidden_layers_size[0],
                        mean_type=self.batchnorm_mean_type,
                        mean_options=self.batchnorm_mean_options,
                        momentum=self.batchnorm_momentum,
                        norm_strategy=self.batchnorm_norm_strategy,
                        minibatch_mode=self.batchnorm_minibatch_mode,
                        minibatch_momentum=self.batchnorm_minibatch_momentum,
                        minibatch_maxstep=self.batchnorm_minibatch_maxstep,
                        parametrization=self.batchnorm_parametrization,
                        use_autograd=self.use_autograd["batchnorm"],
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            elif self.batchnorm_type == "mean_var_scalar":
                spdnet_layers.append(
                    BatchNormSPDMeanScalarVariance(
                        n_features=self.hidden_layers_size[0],
                        mean_type=self.batchnorm_mean_type,
                        mean_options=self.batchnorm_mean_options,
                        momentum=self.batchnorm_momentum,
                        norm_strategy=self.batchnorm_norm_strategy,
                        minibatch_mode=self.batchnorm_minibatch_mode,
                        minibatch_momentum=self.batchnorm_minibatch_momentum,
                        minibatch_maxstep=self.batchnorm_minibatch_maxstep,
                        parametrization=self.batchnorm_parametrization,
                        use_autograd=self.use_autograd["batchnorm"],
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
        for i in range(1, len(hidden_layers_size)):
            spdnet_layers.append(
                BiMap(
                    n_in=self.hidden_layers_size[i - 1],
                    n_out=self.hidden_layers_size[i],
                    parametrized=self.bimap_parametrized,
                    parametrization=self.bimap_parametrization,
                    parametrization_options=self.bimap_parametrization_options,
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.generator,
                    use_autograd=self.use_autograd["bimap"],
                )
            )
            spdnet_layers.append(
                ReEig(
                    eps=self.reeig_eps,
                    dim=self.hidden_layers_size[i],
                    use_autograd=self.use_autograd["reeig"],
                )
            )

            if batchnorm:
                if self.batchnorm_type == "mean_only":
                    spdnet_layers.append(
                        BatchNormSPDMean(
                            n_features=self.hidden_layers_size[i],
                            mean_type=self.batchnorm_mean_type,
                            mean_options=self.batchnorm_mean_options,
                            momentum=self.batchnorm_momentum,
                            norm_strategy=self.batchnorm_norm_strategy,
                            minibatch_mode=self.batchnorm_minibatch_mode,
                            minibatch_momentum=self.batchnorm_minibatch_momentum,
                            minibatch_maxstep=self.batchnorm_minibatch_maxstep,
                            parametrization=self.batchnorm_parametrization,
                            use_autograd=self.use_autograd["batchnorm"],
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )
                elif self.batchnorm_type == "mean_var_scalar":
                    spdnet_layers.append(
                        BatchNormSPDMeanScalarVariance(
                            n_features=self.hidden_layers_size[i],
                            mean_type=self.batchnorm_mean_type,
                            mean_options=self.batchnorm_mean_options,
                            momentum=self.batchnorm_momentum,
                            norm_strategy=self.batchnorm_norm_strategy,
                            minibatch_mode=self.batchnorm_minibatch_mode,
                            minibatch_momentum=self.batchnorm_minibatch_momentum,
                            minibatch_maxstep=self.batchnorm_minibatch_maxstep,
                            parametrization=self.batchnorm_parametrization,
                            use_autograd=self.use_autograd["batchnorm"],
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )

        # Conditionally add LogEig layer
        if self.use_logeig:
            spdnet_layers.append(LogEig(use_autograd=self.use_autograd["logeig"]))

        self.spdnet_layers = nn.Sequential(*spdnet_layers)

        # Create final layer(s)
        if self.vec_type == "vec":
            self.vectorization = Vec(use_autograd=self.use_autograd["vec"])
            self.linear = nn.Linear(
                self.hidden_layers_size[-1] ** 2,
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )
        elif self.vec_type == "vech":
            self.vectorization = Vech()  # Vech always uses manual gradient
            self.linear = nn.Linear(
                self.hidden_layers_size[-1] * (self.hidden_layers_size[-1] + 1) // 2,
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )

        if self.softmax:
            self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDnet

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
        """
        String representation of SPDnet
        """
        return (
            f"SPDnet(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_layers_size={self.hidden_layers_size},\n"
            f"  output_dim={self.output_dim},\n"
            f"  softmax={self.softmax},\n"
            f"  reeig_eps={self.reeig_eps},\n"
            f"  bimap_parametrized={self.bimap_parametrized},\n"
            f"  bimap_parametrization={self.bimap_parametrization},\n"
            f"  bimap_parametrization_options={self.bimap_parametrization_options},\n"
            f"  batchnorm={self.batchnorm},\n"
            f"  batchnorm_mean_type='{self.batchnorm_mean_type}',\n"
            f"  batchnorm_mean_options={self.batchnorm_mean_options},\n"
            f"  batchnorm_momentum={self.batchnorm_momentum},\n"
            f"  batchnorm_norm_strategy={self.batchnorm_norm_strategy}, \n"
            f"  batchnorm_minibatch_momentum={self.batchnorm_minibatch_momentum}, \n"
            f"  vec_type='{self.vec_type}',\n"
            f"  use_logeig={self.use_logeig},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  generator={self.generator},\n"
            f"  use_autograd={self._use_autograd_original}\n"
            f")"
        )

    def __str__(self) -> str:
        """
        String representation of SPDnet

        Returns
        -------
        str
            String representation of SPDnet
        """
        return self.__repr__()

    def layers_str(self) -> str:
        """Return a formatted string listing the layers of SPDnet."""
        string = self.__repr__() + "\n\nSPDnet Layers:\n"
        string += "---------------\n"

        # SPDNet feature layers (BiMap / ReEig / BatchNormSPDMean / LogEig)
        for i, layer in enumerate(self.spdnet_layers):
            string += f"  ({i}). {layer}\n"

        # Vectorization layer
        string += f"  ({len(self.spdnet_layers)}). {self.vectorization}\n"

        # Final linear layer
        string += f"  ({len(self.spdnet_layers) + 1}). {self.linear}\n"

        # Optional softmax layer
        if self.softmax:
            string += f"  ({len(self.spdnet_layers) + 2}). Softmax(dim=-1)\n"

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
            Last tensor of SPDnet
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
