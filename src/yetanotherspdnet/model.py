"""SPD network model definitions: SPDnet, GBWBNRResNet, and RResNet."""

import zlib
from typing import Any

import torch
from torch import nn

from yetanotherspdnet.nn.base import BiMap, LogEig, ReEig, Vec, Vech
from yetanotherspdnet.nn.batchnorm import (
    BatchNormSPDMean,
    BatchNormSPDMeanScalarVariance,
)
from yetanotherspdnet.nn.rresnet_layers import ResidualBlock


class _ModelMixin(nn.Module):
    """
    Helpers shared by the models: optimizer hooks, model hash, layer listing.

    A model sets ``_use_autograd_keys`` (its layer kinds), ``_layers_attr`` (the
    ``nn.Sequential`` of its SPD layers) and ``_layers_header`` (title of
    :meth:`layers_str`), and calls :meth:`_set_use_autograd` in ``__init__``.
    The model hash is the CRC32 of ``__repr__``, which each model defines.
    """

    _use_autograd_keys: tuple[str, ...] = ()
    _layers_attr: str = "spd_layers"
    _layers_header: str = "Layers:\n-------\n"

    # set in the __init__ of each model (declared for type checkers)
    vectorization: nn.Module
    linear: nn.Linear
    softmax: bool

    def _set_use_autograd(self, use_autograd: bool | dict) -> None:
        """
        Expand ``use_autograd`` into one flag per layer kind.

        Parameters
        ----------
        use_autograd : bool | dict
            Same flag for every layer kind, or a dict of flags (missing kinds
            default to False).
        """
        if isinstance(use_autograd, bool):
            self.use_autograd = dict.fromkeys(self._use_autograd_keys, use_autograd)
        else:
            self.use_autograd = dict.fromkeys(self._use_autograd_keys, False)
            self.use_autograd.update(use_autograd)
        # kept for __repr__ (and so for the model hash)
        self._use_autograd_original = use_autograd

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Register the optimizer hooks of every layer with a dynamic parametrization.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer used for training
        """
        for module in self.modules():
            if (
                hasattr(module, "register_optimizer_hook")
                and module is not self
                and hasattr(module, "is_dynamic")
                and module.is_dynamic is True
            ):
                module.register_optimizer_hook(optimizer)

    def get_last_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return the last SPD (or LogEig) tensor, before vectorization.

        Parameters
        ----------
        X : torch.Tensor of shape (..., input_dim, input_dim)
            Input SPD matrices

        Returns
        -------
        torch.Tensor
            Output of the SPD layers
        """
        return getattr(self, self._layers_attr)(X)

    def layers_str(self) -> str:
        """Return a formatted string listing the layers of the model."""
        layers = getattr(self, self._layers_attr)
        string = self.__repr__() + "\n\n" + self._layers_header
        for i, layer in enumerate(layers):
            string += f"  ({i}). {layer}\n"
        string += f"  ({len(layers)}). {self.vectorization}\n"
        string += f"  ({len(layers) + 1}). {self.linear}\n"
        if self.softmax:
            string += f"  ({len(layers) + 2}). Softmax(dim=-1)\n"
        return string

    def create_model_name_hash(self) -> str:
        """
        Create a short hash of the model configuration.

        Returns
        -------
        str
            CRC32 of the model representation (8 hexadecimal characters)
        """
        crc_hash = zlib.crc32(self.__str__().encode("utf-8")) & 0xFFFFFFFF
        self.model_hash = f"{crc_hash:08x}"
        return self.model_hash

    def get_model_hash(self) -> str:
        """
        Return the model hash, creating it if needed.

        Returns
        -------
        str
            Model hash
        """
        if not hasattr(self, "model_hash"):
            self.create_model_name_hash()
        return self.model_hash


class SPDnet(_ModelMixin, nn.Module):
    r"""
    SPDNet classifier: ``[BiMap -> ReEig -> (BatchNorm)] x L -> LogEig -> Vec -> Linear``.

    Implements the architecture of Huang & Van Gool (AAAI 2017), with optional
    Riemannian batch normalization after each ReEig. (The SPDNetBN paper of
    Brooks et al., NeurIPS 2019, places it between the BiMap and the ReEig.)

    """

    _use_autograd_keys = ("bimap", "reeig", "logeig", "batchnorm", "vec")
    _layers_attr = "spdnet_layers"
    _layers_header = "SPDnet Layers:\n---------------\n"

    def __init__(
        self,
        input_dim: int,
        hidden_layers_size: list[int],
        output_dim: int,
        softmax: bool = False,
        reeig_eps: float = 1e-3,
        bimap_parametrized: bool = True,
        bimap_parametrization_mode: str = "static",
        bimap_parametrization_options: dict | None = None,
        bimap_n_steps_ref_update: int = 100,
        batchnorm: bool = False,
        batchnorm_type: str = "mean_only",
        batchnorm_mean_type: str = "geometric_arithmetic_harmonic",
        batchnorm_mean_options: dict | None = None,
        batchnorm_momentum: float = 0.01,
        batchnorm_norm_and_bias_mode: str = "whitening",
        batchnorm_norm_strategy: str = "classical",
        batchnorm_minibatch_mode: str = "constant",
        batchnorm_minibatch_momentum: float = 0.01,
        batchnorm_minibatch_maxstep: int = 100,
        batchnorm_parametrization: str = "softplus",
        batchnorm_parametrization_mode: str = "static",
        batchnorm_n_steps_ref_update: int = 100,
        batchnorm_bw_options: dict | None = None,
        vec_type: str = "vec",
        use_logeig: bool = True,
        use_autograd: bool | dict = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Build an SPDnet model with hidden layers.

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

        bimap_parametrization_mode : str, optional
            Parametrization mode of BiMap if bimap_parametrized is True.
            Default is "static".
            Choices are: "static" and "dynamic"

        bimap_parametrization_options : dict, optional
            Options for the parametrization function in BiMap.
            Default is None

        bimap_n_steps_ref_update : int, optional
            If bimap_parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        batchnorm : bool, optional
            Whether to apply BatchNormSPDMean to hidden layers. Default is False

        batchnorm_type : str, optional
            The type of batch normalization layer to use.
            Default is "mean_only".
            Choices are: "mean_only" and "mean_var_scalar"

        batchnorm_mean_type : str, optional
            Choice of SPD mean of the batch normalization.
            Default is "geometric_arithmetic_harmonic".
            Choices are: "affine_invariant", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic",
            "adaptive_geometric_arithmetic_harmonic", "bures_wasserstein"

        batchnorm_mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        batchnorm_momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        batchnorm_norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening", "parallel_transport"

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

        batchnorm_parametrization_mode : str, optional
            Parametrization mode.
            Default is "static".
            Choices are: "static" and "dynamic"

        batchnorm_n_steps_ref_update : int, optional
            If parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        batchnorm_bw_options : dict | None, optional
            GBWBN keyword arguments of :class:`BatchNormSPDMeanScalarVariance`
            (``bw_theta``, ``bw_batch_stats_grad``), used when
            ``batchnorm_type="mean_var_scalar"``.
            Default is None (the layer defaults)

        vec_type : str, optional
            Whether to use Vec or Vech module.
            Default is "vec".
            Choices are: "vec", "vech"

        use_logeig : bool, optional
            Whether to apply LogEig layer before vectorization.
            Default is True

        use_autograd : bool | dict, optional
            Use torch autograd for gradient computation. Can be bool for all layers,
            or dict with keys: 'bimap', 'reeig', 'logeig', 'batchnorm', 'vec'.
            Note that Vech module always uses manual gradient.
            Default is False

        device : torch.device, optional
            Device to run model on. Default is torch.device('cpu')

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None

        Attributes
        ----------
        spdnet_layers : torch.nn.Sequential
            The SPD part of the network (BiMap, ReEig, BatchNorm and LogEig layers).
        vectorization : Vec or Vech
            Flattens the final symmetric matrices.
        linear : torch.nn.Linear
            Euclidean classification head.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim
        self.softmax = softmax

        self.reeig_eps = reeig_eps

        self.bimap_parametrized = bimap_parametrized
        self.bimap_parametrization_mode = bimap_parametrization_mode
        self.bimap_parametrization_options = bimap_parametrization_options
        self.bimap_n_steps_ref_update = bimap_n_steps_ref_update

        self.batchnorm = batchnorm
        self.batchnorm_type = batchnorm_type
        if self.batchnorm_type not in ("mean_only", "mean_var_scalar"):
            raise ValueError(
                f"batchnorm_type must be in ['mean_only', 'mean_var_scalar'], got {self.batchnorm_type}"
            )
        self.batchnorm_mean_type = batchnorm_mean_type
        self.batchnorm_mean_options = batchnorm_mean_options
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_norm_and_bias_mode = batchnorm_norm_and_bias_mode
        self.batchnorm_norm_strategy = batchnorm_norm_strategy
        self.batchnorm_minibatch_mode = batchnorm_minibatch_mode
        self.batchnorm_minibatch_momentum = batchnorm_minibatch_momentum
        self.batchnorm_minibatch_maxstep = batchnorm_minibatch_maxstep
        self.batchnorm_parametrization = batchnorm_parametrization
        self.batchnorm_parametrization_mode = batchnorm_parametrization_mode
        self.batchnorm_n_steps_ref_update = batchnorm_n_steps_ref_update
        self.batchnorm_bw_options = batchnorm_bw_options

        self.vec_type = vec_type
        if self.vec_type not in ("vec", "vech"):
            raise ValueError(f"vec_type must be 'vec' or 'vech', got {self.vec_type}")

        self.use_logeig = use_logeig
        self.device = device
        self.dtype = dtype
        self.generator = generator

        self._set_use_autograd(use_autograd)

        def make_batchnorm(n_features: int) -> nn.Module:
            return _make_batchnorm(
                batchnorm_type=self.batchnorm_type,
                n_features=n_features,
                mean_type=self.batchnorm_mean_type,
                mean_options=self.batchnorm_mean_options,
                momentum=self.batchnorm_momentum,
                norm_and_bias_mode=self.batchnorm_norm_and_bias_mode,
                norm_strategy=self.batchnorm_norm_strategy,
                minibatch_mode=self.batchnorm_minibatch_mode,
                minibatch_momentum=self.batchnorm_minibatch_momentum,
                minibatch_maxstep=self.batchnorm_minibatch_maxstep,
                parametrization=self.batchnorm_parametrization,
                parametrization_mode=self.batchnorm_parametrization_mode,
                n_steps_ref_update=self.batchnorm_n_steps_ref_update,
                use_autograd=self.use_autograd["batchnorm"],
                device=self.device,
                dtype=self.dtype,
                bw_options=self.batchnorm_bw_options,
            )

        # Create layers
        spdnet_layers: list[nn.Module] = [
            BiMap(
                n_in=self.input_dim,
                n_out=self.hidden_layers_size[0],
                parametrized=self.bimap_parametrized,
                parametrization_mode=self.bimap_parametrization_mode,
                parametrization_options=self.bimap_parametrization_options,
                n_steps_ref_update=self.bimap_n_steps_ref_update,
                use_autograd=self.use_autograd["bimap"],
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
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
            spdnet_layers.append(make_batchnorm(self.hidden_layers_size[0]))
        for i in range(1, len(hidden_layers_size)):
            spdnet_layers.append(
                BiMap(
                    n_in=self.hidden_layers_size[i - 1],
                    n_out=self.hidden_layers_size[i],
                    parametrized=self.bimap_parametrized,
                    parametrization_mode=self.bimap_parametrization_mode,
                    parametrization_options=self.bimap_parametrization_options,
                    n_steps_ref_update=self.bimap_n_steps_ref_update,
                    use_autograd=self.use_autograd["bimap"],
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.generator,
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
                spdnet_layers.append(make_batchnorm(self.hidden_layers_size[i]))

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
        """Forward pass of SPDnet.

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
        """Return the string representation of SPDnet."""
        return (
            f"SPDnet(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_layers_size={self.hidden_layers_size},\n"
            f"  output_dim={self.output_dim},\n"
            f"  softmax={self.softmax},\n"
            f"  reeig_eps={self.reeig_eps},\n"
            f"  bimap_parametrized={self.bimap_parametrized},\n"
            f"  bimap_parametrization_mode={self.bimap_parametrization_mode},\n"
            f"  bimap_parametrization_options={self.bimap_parametrization_options},\n"
            f"  bimap_n_steps_ref_update={self.bimap_n_steps_ref_update},\n"
            f"  batchnorm={self.batchnorm},\n"
            f"  batchnorm_type={self.batchnorm_type},\n"
            f"  batchnorm_mean_type='{self.batchnorm_mean_type}',\n"
            f"  batchnorm_mean_options={self.batchnorm_mean_options},\n"
            f"  batchnorm_momentum={self.batchnorm_momentum},\n"
            f"  batchnorm_norm_and_bias_mode={self.batchnorm_norm_and_bias_mode},\n"
            f"  batchnorm_norm_strategy={self.batchnorm_norm_strategy},\n"
            f"  batchnorm_minibatch_mode={self.batchnorm_minibatch_mode},\n"
            f"  batchnorm_minibatch_momentum={self.batchnorm_minibatch_momentum},\n"
            f"  batchnorm_minibatch_maxstep={self.batchnorm_minibatch_maxstep},\n"
            f"  batchnorm_parametrization={self.batchnorm_parametrization},\n"
            f"  batchnorm_parametrization_mode={self.batchnorm_parametrization_mode},\n"
            f"  batchnorm_n_steps_ref_update={self.batchnorm_n_steps_ref_update},\n"
            f"  batchnorm_bw_options={self.batchnorm_bw_options},\n"
            f"  vec_type='{self.vec_type}',\n"
            f"  use_logeig={self.use_logeig},\n"
            f"  use_autograd={self._use_autograd_original},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )


def _make_batchnorm(
    batchnorm_type: str,
    n_features: int,
    mean_type: str,
    mean_options: dict | None,
    momentum: float,
    norm_strategy: str,
    minibatch_mode: str,
    minibatch_momentum: float,
    minibatch_maxstep: int,
    parametrization: str,
    parametrization_mode: str,
    n_steps_ref_update: int,
    use_autograd: bool,
    device: torch.device,
    dtype: torch.dtype,
    bw_options: dict | None = None,
    norm_and_bias_mode: str = "whitening",
) -> nn.Module:
    """
    Create the batch normalization layer of a model.

    ``norm_and_bias_mode`` is only used by ``batchnorm_type="mean_only"``
    (:class:`BatchNormSPDMean`) and ``bw_options`` only by
    ``batchnorm_type="mean_var_scalar"`` (:class:`BatchNormSPDMeanScalarVariance`).
    """
    kwargs: dict[str, Any] = {
        "n_features": n_features,
        "mean_type": mean_type,
        "mean_options": mean_options,
        "momentum": momentum,
        "norm_strategy": norm_strategy,
        "minibatch_mode": minibatch_mode,
        "minibatch_momentum": minibatch_momentum,
        "minibatch_maxstep": minibatch_maxstep,
        "parametrization": parametrization,
        "parametrization_mode": parametrization_mode,
        "n_steps_ref_update": n_steps_ref_update,
        "use_autograd": use_autograd,
        "device": device,
        "dtype": dtype,
    }
    if batchnorm_type == "mean_only":
        return BatchNormSPDMean(**kwargs, norm_and_bias_mode=norm_and_bias_mode)
    elif batchnorm_type == "mean_var_scalar":
        return BatchNormSPDMeanScalarVariance(**kwargs, **(bw_options or {}))
    else:
        raise ValueError(
            f"Unknown batchnorm_type '{batchnorm_type}', "
            "expected 'mean_only' or 'mean_var_scalar'"
        )


class GBWBNRResNet(_ModelMixin, nn.Module):
    """
    Riemannian Residual Network faithful to the GBWBN experiment architecture.

    Architecture:
        BiMap(input_dim -> hidden_dim)
        -> [BatchNorm]
        -> ResidualBlock (spectral vector field + exp map)
        -> LogEig
        -> Vec/Vech
        -> Linear(hidden_dim^2 -> output_dim)
        -> [Softmax]

    This architecture matches the paper experiments (HDM05, NTU60) which use:
    - A single BiMap for dimension reduction
    - A single residual block with spectral vector field
    - No ReEig (eigenvalue rectification not needed before residual block)

    The residual block applies a geodesic step on the SPD manifold:
        X_new = Exp_X(Q diag(f(spec(X))) Q^T)

    where Q is a Stiefel-parametrized orthogonal matrix and f is a
    learnable spectrum mapping (Conv1d or MLP on eigenvalues).

    Parameters
    ----------
    input_dim : int
        Input SPD matrix dimension

    hidden_dim : int
        Dimension after BiMap (also the residual block dimension)

    output_dim : int
        Number of output classes

    softmax : bool, optional
        Apply softmax to output. Default is False

    bimap_parametrized : bool, optional
        Enforce Stiefel constraints on BiMap. Default is True

    bimap_parametrization_mode : str, optional
        "static" or "dynamic" parametrization. Default is "static"

    bimap_parametrization_options : dict | None, optional
        Options for BiMap parametrization. Default is None

    bimap_n_steps_ref_update : int, optional
        Steps between reference updates for dynamic BiMap. Default is 100

    batchnorm : bool, optional
        Apply batch normalization. Default is True

    batchnorm_type : str, optional
        "mean_only" or "mean_var_scalar". Default is "mean_var_scalar"

    batchnorm_mean_type : str, optional
        SPD mean type for batchnorm. Default is "bures_wasserstein"

    batchnorm_mean_options : dict | None, optional
        Options for mean computation. Default is None

    batchnorm_momentum : float, optional
        Running mean momentum. Default is 0.1

    batchnorm_norm_strategy : str, optional
        "classical" or "minibatch". Default is "classical"

    batchnorm_minibatch_mode : str, optional
        "constant", "decay", or "growth". Default is "constant"

    batchnorm_minibatch_momentum : float, optional
        Minibatch momentum. Default is 0.01

    batchnorm_minibatch_maxstep : int, optional
        Max step for momentum schedule. Default is 100

    batchnorm_parametrization : str, optional
        "softplus" or "exp". Default is "softplus"

    batchnorm_parametrization_mode : str, optional
        "static" or "dynamic". Default is "static"

    batchnorm_n_steps_ref_update : int, optional
        Steps between reference updates for BN. Default is 100

    batchnorm_bw_options : dict | None, optional
        GBWBN keyword arguments of BatchNormSPDMeanScalarVariance
        (``bw_theta``, ``bw_batch_stats_grad``). Default is None

    spectrum_type : str, optional
        "conv1d" or "mlp" for spectral vector field. Default is "conv1d"

    spectrum_hidden_dim : int, optional
        Hidden dimension for spectrum network. Default is 3

    spectrum_n_layers : int, optional
        Number of hidden layers in spectrum network. Default is 2

    spectrum_kernel_size : int, optional
        Kernel size for Conv1d spectrum. Default is 5

    stiefel_parametrization_mode : str, optional
        Parametrization mode for Q matrix. Default is "static"

    stiefel_n_steps_ref_update : int, optional
        Steps between Q reference updates. Default is 100

    residual_metric : str, optional
        Residual step of the ResidualBlocks: "affine_invariant" (unit-length
        exponential-map step) or "log_euclidean" (exp(log X + V)).
        Default is "affine_invariant"

    vec_type : str, optional
        "vec" or "vech". Default is "vec"

    use_logeig : bool, optional
        Apply LogEig before vectorization. Default is True

    use_autograd : bool | dict, optional
        Autograd control. Bool for all, dict with keys:
        'bimap', 'logeig', 'batchnorm', 'vec', 'residual'.
        Default is False

    device : torch.device, optional
        Device. Default is torch.device("cpu")

    dtype : torch.dtype, optional
        Data type. Default is torch.float64

    generator : torch.Generator | None, optional
        Generator for reproducibility. Default is None
    """

    _use_autograd_keys = ("bimap", "logeig", "batchnorm", "vec", "residual")

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        softmax: bool = False,
        bimap_parametrized: bool = True,
        bimap_parametrization_mode: str = "static",
        bimap_parametrization_options: dict | None = None,
        bimap_n_steps_ref_update: int = 100,
        batchnorm: bool = True,
        batchnorm_type: str = "mean_var_scalar",
        batchnorm_mean_type: str = "bures_wasserstein",
        batchnorm_mean_options: dict | None = None,
        batchnorm_momentum: float = 0.1,
        batchnorm_norm_strategy: str = "classical",
        batchnorm_minibatch_mode: str = "constant",
        batchnorm_minibatch_momentum: float = 0.01,
        batchnorm_minibatch_maxstep: int = 100,
        batchnorm_parametrization: str = "softplus",
        batchnorm_parametrization_mode: str = "static",
        batchnorm_n_steps_ref_update: int = 100,
        batchnorm_bw_options: dict | None = None,
        spectrum_type: str = "conv1d",
        spectrum_hidden_dim: int = 3,
        spectrum_n_layers: int = 2,
        spectrum_kernel_size: int = 5,
        stiefel_parametrization_mode: str = "static",
        stiefel_n_steps_ref_update: int = 100,
        residual_metric: str = "affine_invariant",
        vec_type: str = "vec",
        use_logeig: bool = True,
        use_autograd: bool | dict = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.softmax = softmax

        self.batchnorm = batchnorm
        self.batchnorm_type = batchnorm_type
        self.batchnorm_mean_type = batchnorm_mean_type
        # remaining hyper-parameters, kept for __repr__ and the model hash
        self.bimap_parametrized = bimap_parametrized
        self.bimap_parametrization_mode = bimap_parametrization_mode
        self.bimap_parametrization_options = bimap_parametrization_options
        self.bimap_n_steps_ref_update = bimap_n_steps_ref_update
        self.batchnorm_mean_options = batchnorm_mean_options
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_norm_strategy = batchnorm_norm_strategy
        self.batchnorm_minibatch_mode = batchnorm_minibatch_mode
        self.batchnorm_minibatch_momentum = batchnorm_minibatch_momentum
        self.batchnorm_minibatch_maxstep = batchnorm_minibatch_maxstep
        self.batchnorm_parametrization = batchnorm_parametrization
        self.batchnorm_parametrization_mode = batchnorm_parametrization_mode
        self.batchnorm_n_steps_ref_update = batchnorm_n_steps_ref_update
        self.batchnorm_bw_options = batchnorm_bw_options
        self.spectrum_type = spectrum_type
        self.spectrum_hidden_dim = spectrum_hidden_dim
        self.spectrum_n_layers = spectrum_n_layers
        self.spectrum_kernel_size = spectrum_kernel_size
        self.stiefel_parametrization_mode = stiefel_parametrization_mode
        self.stiefel_n_steps_ref_update = stiefel_n_steps_ref_update
        self.residual_metric = residual_metric

        self.vec_type = vec_type
        if vec_type not in ("vec", "vech"):
            raise ValueError(f"vec_type must be 'vec' or 'vech', got {vec_type}")

        self.use_logeig = use_logeig
        self.device = device
        self.dtype = dtype
        self.generator = generator

        self._set_use_autograd(use_autograd)

        # Build layers
        layers: list[nn.Module] = []

        # BiMap: dimension reduction
        layers.append(
            BiMap(
                n_in=input_dim,
                n_out=hidden_dim,
                parametrized=bimap_parametrized,
                parametrization_mode=bimap_parametrization_mode,
                parametrization_options=bimap_parametrization_options,
                n_steps_ref_update=bimap_n_steps_ref_update,
                use_autograd=self.use_autograd["bimap"],
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )

        # BatchNorm (optional)
        if batchnorm:
            layers.append(
                _make_batchnorm(
                    batchnorm_type=batchnorm_type,
                    n_features=hidden_dim,
                    mean_type=batchnorm_mean_type,
                    mean_options=batchnorm_mean_options,
                    momentum=batchnorm_momentum,
                    norm_strategy=batchnorm_norm_strategy,
                    minibatch_mode=batchnorm_minibatch_mode,
                    minibatch_momentum=batchnorm_minibatch_momentum,
                    minibatch_maxstep=batchnorm_minibatch_maxstep,
                    parametrization=batchnorm_parametrization,
                    parametrization_mode=batchnorm_parametrization_mode,
                    n_steps_ref_update=batchnorm_n_steps_ref_update,
                    use_autograd=self.use_autograd["batchnorm"],
                    bw_options=batchnorm_bw_options,
                    device=device,
                    dtype=dtype,
                )
            )

        # Residual block
        layers.append(
            ResidualBlock(
                n_features=hidden_dim,
                spectrum_type=spectrum_type,
                spectrum_hidden_dim=spectrum_hidden_dim,
                spectrum_n_layers=spectrum_n_layers,
                spectrum_kernel_size=spectrum_kernel_size,
                stiefel_parametrization_mode=stiefel_parametrization_mode,
                stiefel_n_steps_ref_update=stiefel_n_steps_ref_update,
                metric=residual_metric,
                use_autograd=self.use_autograd["residual"],
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )

        # LogEig
        if use_logeig:
            layers.append(LogEig(use_autograd=self.use_autograd["logeig"]))

        self.spd_layers = nn.Sequential(*layers)

        # Vectorization + classification head
        if vec_type == "vec":
            self.vectorization = Vec(use_autograd=self.use_autograd["vec"])
            linear_in = hidden_dim**2
        else:
            self.vectorization = Vech()
            linear_in = hidden_dim * (hidden_dim + 1) // 2

        self.linear = nn.Linear(linear_in, output_dim, dtype=dtype, device=device)

        if softmax:
            self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor of shape (..., input_dim, input_dim)
            Input SPD matrices

        Returns
        -------
        torch.Tensor of shape (..., output_dim)
            Output predictions
        """
        X = self.spd_layers(X)
        X = self.vectorization(X)
        X = self.linear(X)
        if self.softmax:
            X = self.softmax_layer(X)
        return X

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        return (
            f"GBWBNRResNet(\n"
            f"  input_dim={self.input_dim!r},\n"
            f"  hidden_dim={self.hidden_dim!r},\n"
            f"  output_dim={self.output_dim!r},\n"
            f"  softmax={self.softmax!r},\n"
            f"  bimap_parametrized={self.bimap_parametrized!r},\n"
            f"  bimap_parametrization_mode={self.bimap_parametrization_mode!r},\n"
            f"  bimap_parametrization_options={self.bimap_parametrization_options!r},\n"
            f"  bimap_n_steps_ref_update={self.bimap_n_steps_ref_update!r},\n"
            f"  batchnorm={self.batchnorm!r},\n"
            f"  batchnorm_type={self.batchnorm_type!r},\n"
            f"  batchnorm_mean_type={self.batchnorm_mean_type!r},\n"
            f"  batchnorm_mean_options={self.batchnorm_mean_options!r},\n"
            f"  batchnorm_momentum={self.batchnorm_momentum!r},\n"
            f"  batchnorm_norm_strategy={self.batchnorm_norm_strategy!r},\n"
            f"  batchnorm_minibatch_mode={self.batchnorm_minibatch_mode!r},\n"
            f"  batchnorm_minibatch_momentum={self.batchnorm_minibatch_momentum!r},\n"
            f"  batchnorm_minibatch_maxstep={self.batchnorm_minibatch_maxstep!r},\n"
            f"  batchnorm_parametrization={self.batchnorm_parametrization!r},\n"
            f"  batchnorm_parametrization_mode={self.batchnorm_parametrization_mode!r},\n"
            f"  batchnorm_n_steps_ref_update={self.batchnorm_n_steps_ref_update!r},\n"
            f"  batchnorm_bw_options={self.batchnorm_bw_options!r},\n"
            f"  spectrum_type={self.spectrum_type!r},\n"
            f"  spectrum_hidden_dim={self.spectrum_hidden_dim!r},\n"
            f"  spectrum_n_layers={self.spectrum_n_layers!r},\n"
            f"  spectrum_kernel_size={self.spectrum_kernel_size!r},\n"
            f"  stiefel_parametrization_mode={self.stiefel_parametrization_mode!r},\n"
            f"  stiefel_n_steps_ref_update={self.stiefel_n_steps_ref_update!r},\n"
            f"  residual_metric={self.residual_metric!r},\n"
            f"  vec_type={self.vec_type!r},\n"
            f"  use_logeig={self.use_logeig!r},\n"
            f"  use_autograd={self._use_autograd_original!r},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )


class RResNet(_ModelMixin, nn.Module):
    """
    Flexible multi-stage Riemannian Residual Network on SPD manifold.

    Architecture (for each stage i):
        BiMap(d_{i-1} -> d_i)
        -> [ReEig]
        -> [BatchNorm]
        -> ResidualBlock x n_residual_blocks[i]
    then:
        -> LogEig -> Vec/Vech -> Linear -> [Softmax]

    Inspired by classical ResNet (multi-stage with dimension changes at each
    stage boundary), this architecture is more flexible than GBWBNRResNet:
    it supports multiple BiMap stages, optional ReEig at each stage,
    and multiple residual blocks per stage.

    Parameters
    ----------
    input_dim : int
        Input SPD matrix dimension

    hidden_layers_size : list[int]
        Dimensions at each stage (after each BiMap)

    n_residual_blocks : list[int]
        Number of residual blocks at each stage

    output_dim : int
        Number of output classes

    softmax : bool, optional
        Apply softmax. Default is False

    reeig : bool, optional
        Apply ReEig after each BiMap. Default is False

    reeig_eps : float, optional
        Minimum eigenvalue for ReEig. Default is 1e-3

    bimap_parametrized : bool, optional
        Enforce Stiefel on BiMap. Default is True

    bimap_parametrization_mode : str, optional
        "static" or "dynamic". Default is "static"

    bimap_parametrization_options : dict | None, optional
        Options for BiMap parametrization. Default is None

    bimap_n_steps_ref_update : int, optional
        Steps between reference updates. Default is 100

    batchnorm : bool, optional
        Apply batchnorm at each stage. Default is False

    batchnorm_type : str, optional
        "mean_only" or "mean_var_scalar". Default is "mean_only"

    batchnorm_mean_type : str, optional
        SPD mean type. Default is "affine_invariant"

    batchnorm_mean_options : dict | None, optional
        Options for mean computation. Default is None

    batchnorm_momentum : float, optional
        Running mean momentum. Default is 0.01

    batchnorm_norm_strategy : str, optional
        "classical" or "minibatch". Default is "classical"

    batchnorm_minibatch_mode : str, optional
        "constant", "decay", or "growth". Default is "constant"

    batchnorm_minibatch_momentum : float, optional
        Minibatch momentum. Default is 0.01

    batchnorm_minibatch_maxstep : int, optional
        Max step for momentum schedule. Default is 100

    batchnorm_parametrization : str, optional
        "softplus" or "exp". Default is "softplus"

    batchnorm_parametrization_mode : str, optional
        "static" or "dynamic". Default is "static"

    batchnorm_n_steps_ref_update : int, optional
        Steps between BN reference updates. Default is 100

    batchnorm_bw_options : dict | None, optional
        GBWBN keyword arguments of BatchNormSPDMeanScalarVariance
        (``bw_theta``, ``bw_batch_stats_grad``). Default is None

    spectrum_type : str, optional
        "conv1d" or "mlp". Default is "conv1d"

    spectrum_hidden_dim : int, optional
        Hidden dimension for spectrum network. Default is 3

    spectrum_n_layers : int, optional
        Hidden layers in spectrum network. Default is 2

    spectrum_kernel_size : int, optional
        Kernel size for Conv1d. Default is 5

    stiefel_parametrization_mode : str, optional
        Parametrization mode for Q matrices. Default is "static"

    stiefel_n_steps_ref_update : int, optional
        Steps between Q reference updates. Default is 100

    residual_metric : str, optional
        Residual step of the ResidualBlocks: "affine_invariant" (unit-length
        exponential-map step) or "log_euclidean" (exp(log X + V)).
        Default is "affine_invariant"

    vec_type : str, optional
        "vec" or "vech". Default is "vec"

    use_logeig : bool, optional
        Apply LogEig before vectorization. Default is True

    use_autograd : bool | dict, optional
        Autograd control. Bool for all, dict with keys:
        'bimap', 'reeig', 'logeig', 'batchnorm', 'vec', 'residual'.
        Default is False

    device : torch.device, optional
        Device. Default is torch.device("cpu")

    dtype : torch.dtype, optional
        Data type. Default is torch.float64

    generator : torch.Generator | None, optional
        Generator for reproducibility. Default is None
    """

    _use_autograd_keys = ("bimap", "reeig", "logeig", "batchnorm", "vec", "residual")

    def __init__(
        self,
        input_dim: int,
        hidden_layers_size: list[int],
        n_residual_blocks: list[int],
        output_dim: int,
        softmax: bool = False,
        reeig: bool = False,
        reeig_eps: float = 1e-3,
        bimap_parametrized: bool = True,
        bimap_parametrization_mode: str = "static",
        bimap_parametrization_options: dict | None = None,
        bimap_n_steps_ref_update: int = 100,
        batchnorm: bool = False,
        batchnorm_type: str = "mean_only",
        batchnorm_mean_type: str = "affine_invariant",
        batchnorm_mean_options: dict | None = None,
        batchnorm_momentum: float = 0.01,
        batchnorm_norm_strategy: str = "classical",
        batchnorm_minibatch_mode: str = "constant",
        batchnorm_minibatch_momentum: float = 0.01,
        batchnorm_minibatch_maxstep: int = 100,
        batchnorm_parametrization: str = "softplus",
        batchnorm_parametrization_mode: str = "static",
        batchnorm_n_steps_ref_update: int = 100,
        batchnorm_bw_options: dict | None = None,
        spectrum_type: str = "conv1d",
        spectrum_hidden_dim: int = 3,
        spectrum_n_layers: int = 2,
        spectrum_kernel_size: int = 5,
        stiefel_parametrization_mode: str = "static",
        stiefel_n_steps_ref_update: int = 100,
        residual_metric: str = "affine_invariant",
        vec_type: str = "vec",
        use_logeig: bool = True,
        use_autograd: bool | dict = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if len(hidden_layers_size) != len(n_residual_blocks):
            raise ValueError(
                f"hidden_layers_size and n_residual_blocks must have same length, "
                f"got {len(hidden_layers_size)} and {len(n_residual_blocks)}"
            )
        if any(n < 0 for n in n_residual_blocks):
            raise ValueError("n_residual_blocks must be non-negative")

        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.n_residual_blocks = n_residual_blocks
        self.output_dim = output_dim
        self.softmax = softmax
        self.reeig = reeig
        self.reeig_eps = reeig_eps

        self.batchnorm = batchnorm
        self.batchnorm_type = batchnorm_type
        self.batchnorm_mean_type = batchnorm_mean_type
        # remaining hyper-parameters, kept for __repr__ and the model hash
        self.bimap_parametrized = bimap_parametrized
        self.bimap_parametrization_mode = bimap_parametrization_mode
        self.bimap_parametrization_options = bimap_parametrization_options
        self.bimap_n_steps_ref_update = bimap_n_steps_ref_update
        self.batchnorm_mean_options = batchnorm_mean_options
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_norm_strategy = batchnorm_norm_strategy
        self.batchnorm_minibatch_mode = batchnorm_minibatch_mode
        self.batchnorm_minibatch_momentum = batchnorm_minibatch_momentum
        self.batchnorm_minibatch_maxstep = batchnorm_minibatch_maxstep
        self.batchnorm_parametrization = batchnorm_parametrization
        self.batchnorm_parametrization_mode = batchnorm_parametrization_mode
        self.batchnorm_n_steps_ref_update = batchnorm_n_steps_ref_update
        self.batchnorm_bw_options = batchnorm_bw_options
        self.spectrum_type = spectrum_type
        self.spectrum_hidden_dim = spectrum_hidden_dim
        self.spectrum_n_layers = spectrum_n_layers
        self.spectrum_kernel_size = spectrum_kernel_size
        self.stiefel_parametrization_mode = stiefel_parametrization_mode
        self.stiefel_n_steps_ref_update = stiefel_n_steps_ref_update
        self.residual_metric = residual_metric

        self.vec_type = vec_type
        if vec_type not in ("vec", "vech"):
            raise ValueError(f"vec_type must be 'vec' or 'vech', got {vec_type}")

        self.use_logeig = use_logeig
        self.device = device
        self.dtype = dtype
        self.generator = generator

        self._set_use_autograd(use_autograd)

        # Build stages
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_layers_size]

        for stage_idx in range(len(hidden_layers_size)):
            d_in = dims[stage_idx]
            d_out = dims[stage_idx + 1]

            # BiMap: dimension change
            layers.append(
                BiMap(
                    n_in=d_in,
                    n_out=d_out,
                    parametrized=bimap_parametrized,
                    parametrization_mode=bimap_parametrization_mode,
                    parametrization_options=bimap_parametrization_options,
                    n_steps_ref_update=bimap_n_steps_ref_update,
                    use_autograd=self.use_autograd["bimap"],
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )

            # ReEig (optional)
            if reeig:
                layers.append(
                    ReEig(
                        eps=reeig_eps,
                        dim=d_out,
                        use_autograd=self.use_autograd["reeig"],
                    )
                )

            # BatchNorm (optional)
            if batchnorm:
                layers.append(
                    _make_batchnorm(
                        batchnorm_type=batchnorm_type,
                        n_features=d_out,
                        mean_type=batchnorm_mean_type,
                        mean_options=batchnorm_mean_options,
                        momentum=batchnorm_momentum,
                        norm_strategy=batchnorm_norm_strategy,
                        minibatch_mode=batchnorm_minibatch_mode,
                        minibatch_momentum=batchnorm_minibatch_momentum,
                        minibatch_maxstep=batchnorm_minibatch_maxstep,
                        parametrization=batchnorm_parametrization,
                        parametrization_mode=batchnorm_parametrization_mode,
                        n_steps_ref_update=batchnorm_n_steps_ref_update,
                        use_autograd=self.use_autograd["batchnorm"],
                        bw_options=batchnorm_bw_options,
                        device=device,
                        dtype=dtype,
                    )
                )

            # Residual blocks (at same dimension)
            for _ in range(n_residual_blocks[stage_idx]):
                layers.append(
                    ResidualBlock(
                        n_features=d_out,
                        spectrum_type=spectrum_type,
                        spectrum_hidden_dim=spectrum_hidden_dim,
                        spectrum_n_layers=spectrum_n_layers,
                        spectrum_kernel_size=spectrum_kernel_size,
                        stiefel_parametrization_mode=stiefel_parametrization_mode,
                        stiefel_n_steps_ref_update=stiefel_n_steps_ref_update,
                        metric=residual_metric,
                        use_autograd=self.use_autograd["residual"],
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                )

        # LogEig
        if use_logeig:
            layers.append(LogEig(use_autograd=self.use_autograd["logeig"]))

        self.spd_layers = nn.Sequential(*layers)

        # Vectorization + classification head
        last_dim = hidden_layers_size[-1]
        if vec_type == "vec":
            self.vectorization = Vec(use_autograd=self.use_autograd["vec"])
            linear_in = last_dim**2
        else:
            self.vectorization = Vech()
            linear_in = last_dim * (last_dim + 1) // 2

        self.linear = nn.Linear(linear_in, output_dim, dtype=dtype, device=device)

        if softmax:
            self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor of shape (..., input_dim, input_dim)
            Input SPD matrices

        Returns
        -------
        torch.Tensor of shape (..., output_dim)
            Output predictions
        """
        X = self.spd_layers(X)
        X = self.vectorization(X)
        X = self.linear(X)
        if self.softmax:
            X = self.softmax_layer(X)
        return X

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        return (
            f"RResNet(\n"
            f"  input_dim={self.input_dim!r},\n"
            f"  hidden_layers_size={self.hidden_layers_size!r},\n"
            f"  n_residual_blocks={self.n_residual_blocks!r},\n"
            f"  output_dim={self.output_dim!r},\n"
            f"  softmax={self.softmax!r},\n"
            f"  reeig={self.reeig!r},\n"
            f"  reeig_eps={self.reeig_eps!r},\n"
            f"  bimap_parametrized={self.bimap_parametrized!r},\n"
            f"  bimap_parametrization_mode={self.bimap_parametrization_mode!r},\n"
            f"  bimap_parametrization_options={self.bimap_parametrization_options!r},\n"
            f"  bimap_n_steps_ref_update={self.bimap_n_steps_ref_update!r},\n"
            f"  batchnorm={self.batchnorm!r},\n"
            f"  batchnorm_type={self.batchnorm_type!r},\n"
            f"  batchnorm_mean_type={self.batchnorm_mean_type!r},\n"
            f"  batchnorm_mean_options={self.batchnorm_mean_options!r},\n"
            f"  batchnorm_momentum={self.batchnorm_momentum!r},\n"
            f"  batchnorm_norm_strategy={self.batchnorm_norm_strategy!r},\n"
            f"  batchnorm_minibatch_mode={self.batchnorm_minibatch_mode!r},\n"
            f"  batchnorm_minibatch_momentum={self.batchnorm_minibatch_momentum!r},\n"
            f"  batchnorm_minibatch_maxstep={self.batchnorm_minibatch_maxstep!r},\n"
            f"  batchnorm_parametrization={self.batchnorm_parametrization!r},\n"
            f"  batchnorm_parametrization_mode={self.batchnorm_parametrization_mode!r},\n"
            f"  batchnorm_n_steps_ref_update={self.batchnorm_n_steps_ref_update!r},\n"
            f"  batchnorm_bw_options={self.batchnorm_bw_options!r},\n"
            f"  spectrum_type={self.spectrum_type!r},\n"
            f"  spectrum_hidden_dim={self.spectrum_hidden_dim!r},\n"
            f"  spectrum_n_layers={self.spectrum_n_layers!r},\n"
            f"  spectrum_kernel_size={self.spectrum_kernel_size!r},\n"
            f"  stiefel_parametrization_mode={self.stiefel_parametrization_mode!r},\n"
            f"  stiefel_n_steps_ref_update={self.stiefel_n_steps_ref_update!r},\n"
            f"  residual_metric={self.residual_metric!r},\n"
            f"  vec_type={self.vec_type!r},\n"
            f"  use_logeig={self.use_logeig!r},\n"
            f"  use_autograd={self._use_autograd_original!r},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )
