"""SPD batch normalization layers with configurable Riemannian geometry."""

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from yetanotherspdnet.functions.spd_geometries.registry import (
    BURES_WASSERSTEIN_STEPS,
    get_geometry,
)

from ..functions._constants import STD_FLOOR
from ..functions.dispatch import select_path
from ..functions.spd_linalg import (
    CongruenceSPD,
    InvSqrtmSPD,
    PowmSPD,
    SqrtmSPD,
    Whitening,
    congruence_SPD,
    eigh_context,
    inv_sqrtm_SPD,
    powm_SPD,
    sqrtm_SPD,
    whitening,
)
from .parametrizations import (
    ScalarSigmoidParametrization,
    ScalarSoftPlusParametrization,
    SPDAdaptiveParametrization,
    SPDParametrization,
    _parametrization_of,
)


def _use_batch_statistics(module: nn.Module, data: torch.Tensor) -> bool:
    """
    Whether batch statistics can be computed for this forward pass.

    Batch statistics need at least two SPD matrices: with a single one the batch
    mean is the matrix itself and the batch dispersion is zero, so normalization
    maps every input to the identity, erasing all information and feeding exactly
    repeated eigenvalues to downstream eigendecompositions. In that case (as for
    an unbatched ``(n_features, n_features)`` input) the running statistics are
    used instead and left unchanged, as in evaluation mode.

    Parameters
    ----------
    module : nn.Module
        Batch normalization layer (its ``training`` flag is read)
    data : torch.Tensor of shape (..., n_features, n_features)
        Input SPD matrices

    Returns
    -------
    bool
        True if the layer is training and the batch holds at least two matrices
    """
    if not module.training:
        return False
    if data.shape[:-2].numel() < 2:
        warnings.warn(
            "SPD batch normalization received a single matrix in training mode; "
            "batch statistics are undefined, so running statistics are used and "
            "not updated.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return True


def _check_bw_theta(theta: float) -> float:
    r"""
    Validate the GBWBN power deformation theta.

    Parameters
    ----------
    theta : float
        Power of the deformation :math:`X \mapsto X^\theta`

    Returns
    -------
    float
        theta as a float

    Raises
    ------
    ValueError
        If theta is not a finite, nonzero real number: :math:`X^0 = I` erases
        the data and the variance is divided by :math:`\theta^2`.
    """
    try:
        value = float(theta)
    except (TypeError, ValueError):
        raise ValueError(f"bw_theta must be a real number, got {theta!r}") from None
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(
            f"bw_theta must be finite and nonzero (X^0 = I erases the data and the "
            f"variance is divided by theta^2), got {theta!r}"
        )
    return value


class BatchNormSPDBase(ABC, nn.Module):
    """
    Common machinery of the SPD batch normalization layers.

    Selects the mean (and its running/regularization updates) from
    ``mean_type`` and runs the training/evaluation logic in :meth:`forward`;
    subclasses define the normalization parameters, the bias and how they are
    applied (``_get_norm_parameters``, ``_update_running_parameters``,
    ``_get_running_parameters``, ``_norm_and_bias``).
    """

    # buffers registered in __init__ (declared for type checkers)
    running_mean: torch.Tensor
    mean_regularizer: torch.Tensor

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Build the common part of the SPD batch normalization layers.

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "affine_invariant_gevd",
            "log_euclidean", "arithmetic", "harmonic",
            "geometric_arithmetic_harmonic",
            "adaptive_geometric_arithmetic_harmonic", "bures_wasserstein"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            For adaptive_geometric_arithmetic_harmonic, {'t_init': t} sets the
            initial value of the learnable t in [0, 1] (0: harmonic,
            1: arithmetic). Default t_init is 0.5 (the GAH mean).
            For bures_wasserstein, one can set {'n_iterations': 1}.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_strategy : str, optional
            Strategy for normalization.
            Default is "classical".
            Choices are: "classical" and "minibatch"

        minibatch_mode : str, optional
            How the minibatch momentum behaves during the training.
            Default is "constant".
            Choices are: "constant", "decay", "growth"

        minibatch_momentum : float, optional
            Momentum for mean regularization in minibatch normalization strategy.
            If minibatch_mode is "decay", this momentum corresponds to the minimum momentum that is reached.
            If minibatch_mode is "growth", this momentum corresponds to the initial momentum.
            Default is 0.01

        minibatch_maxstep : int, optional
            If minibatch_mode is "decay" or "growth", this is the training step at which the minibatch momentum
            attains its final value.
            Default is 100

        parametrization : str, optional
            Parametrization to apply on covariance bias.
            Default is "softplus".
            Choices are: "softplus", "exp"

        parametrization_mode : str, optional
            Parametrization mode.
            Default is "static".
            Choices are: "static" and "dynamic"

        n_steps_ref_update : int, optional
            If parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula.
            Default is False

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
        """
        super().__init__()

        # register all input variables
        self.n_features = n_features
        self.mean_type = mean_type
        # mean, geodesic, std and transport of the geometry (ValueError if unknown)
        self.geometry = get_geometry(mean_type)
        self.mean_options = mean_options
        self.momentum = momentum
        self.norm_strategy = norm_strategy
        if self.norm_strategy not in ("classical", "minibatch"):
            raise ValueError(
                f"norm_strategy must be in ['classical', 'minibatch'], got {self.norm_strategy}"
            )
        self.minibatch_mode = minibatch_mode
        if norm_strategy == "minibatch" and self.minibatch_mode not in (
            "constant",
            "decay",
            "growth",
        ):
            raise ValueError(
                f"minibatch_mode must be in ['constant', 'decay', 'growth'], got {self.minibatch_mode}"
            )
        self.minibatch_momentum = minibatch_momentum
        self.minibatch_maxstep = minibatch_maxstep
        self.parametrization = parametrization
        if self.parametrization not in ("softplus", "exp"):
            raise ValueError(
                f"parametrization must be in ['softplus', 'exp'], got {self.parametrization}"
            )
        self.parametrization_mode = parametrization_mode
        if self.parametrization_mode not in ("static", "dynamic"):
            raise ValueError(
                f"parametrization_mode must be in ['static', 'dynamic'], got {self.parametrization_mode}"
            )
        self.n_steps_ref_update = n_steps_ref_update
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype

        # initialize training_step counter
        self.training_step = 0

        # deal with parametrization_mode
        if parametrization_mode == "static":
            self.is_dynamic = False
        elif parametrization_mode == "dynamic":
            self.is_dynamic = True
            self.current_ref_step = 0

        # initialize normalization parameters and running parameters
        self._init_norm_parameters()

        # initialize bias parameters
        self._init_bias_parameters()

        # initialize normalization and bias functions
        self._init_norm_and_bias()

    def _init_norm_parameters(self) -> None:
        """Initialize the batch, normalization and running parameters."""
        self._init_mean()
        self._init_adaptive_mean_fun()
        self._init_norm_strategy_mean()

        # set running mean
        self.register_buffer(
            "running_mean",
            torch.eye(self.n_features, dtype=self.dtype, device=self.device),
        )

    @abstractmethod
    def _init_bias_parameters(self) -> None:
        """Initialize bias parameters and their parametrization."""
        pass

    @abstractmethod
    def _init_norm_and_bias(self) -> None:
        """Initialize normalization and bias functions."""
        pass

    def _init_mean(self) -> None:
        """Select the mean function of the geometry."""
        mean = self.geometry.mean.select(self.use_autograd)
        if self.geometry.learnable_t:
            # Register learnable parameter t for adaptive GAH mean
            # Use sigmoid parametrization to constrain t in [0, 1]
            t_init = (self.mean_options or {}).get("t_init", 0.5)
            if not 0.0 <= t_init <= 1.0:
                raise ValueError(f"t_init must lie in [0, 1], got {t_init}")
            self.t_gah = torch.nn.Parameter(
                torch.tensor(t_init, dtype=self.dtype, device=self.device)
            )
            register_parametrization(self, "t_gah", ScalarSigmoidParametrization())
            self.mean_fun = lambda data: mean(data, self.t_gah)
        else:
            mean_kwargs = self.geometry.mean_kwargs(self.mean_options)
            self.mean_fun = partial(mean, **mean_kwargs) if mean_kwargs else mean

    def _init_adaptive_mean_fun(self) -> None:
        """
        Select the running mean update, a step along the geodesic.

        It is computed without gradient, so through the autograd path.
        """
        # every registered geometry has an autograd geodesic (DualPath.autograd
        # is only None for some parallel transports)
        self.adaptive_mean_fun = cast(Callable, self.geometry.geodesic.autograd)

    def _init_norm_strategy_mean(self) -> None:
        """Auxiliary function to select normalization strategy."""
        if self.norm_strategy == "classical":
            self.get_norm_mean = lambda mean: mean
        elif self.norm_strategy == "minibatch":
            self.get_norm_mean = self._handle_minibatch_mean
            self.register_buffer(
                "mean_regularizer",
                torch.eye(self.n_features, dtype=self.dtype, device=self.device),
            )
            self.regularize_mean_fun = self.geometry.geodesic.select(self.use_autograd)
            self._init_minibatch_mode()

    def _init_minibatch_mode(self) -> None:
        """Auxiliary function to deal with initialization of minibatch mode."""
        if self.minibatch_mode == "constant":
            self.get_minibatch_momentum = lambda: self.minibatch_momentum
        elif self.minibatch_mode == "decay":
            self.get_minibatch_momentum = self._minibatch_momentum_decay
        elif self.minibatch_mode == "growth":
            self.get_minibatch_momentum = self._minibatch_momentum_growth

    def _minibatch_momentum_decay(self) -> float:
        """
        Compute the minibatch momentum for minibatch_mode == "decay".

        Returns
        -------
        minibatch_momentum : float
            decreased minibatch momentum
        """
        return (
            1
            - self.minibatch_momentum
            ** (
                max(self.minibatch_maxstep - self.training_step, 0)
                / self.minibatch_maxstep
            )
            + self.minibatch_momentum
        )

    def _minibatch_momentum_growth(self) -> float:
        """
        Compute the minibatch momentum for minibatch_mode == "growth".

        Returns
        -------
        minibatch_momentum : float
            increased minibatch momentum
        """
        if self.training_step == 0:
            return self.minibatch_momentum
        return self.minibatch_momentum + (1 - self.minibatch_momentum) ** (
            self.minibatch_maxstep / min(self.training_step, self.minibatch_maxstep)
        )

    def _handle_minibatch_mean(self, mean_batch: torch.Tensor) -> torch.Tensor:
        """
        Regularize the batch mean in the minibatch normalization strategy.

        Parameters
        ----------
        mean_batch : torch.Tensor of shape (n_features, n_features)
            SPD mean of the batch

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization
        """
        mean = self.regularize_mean_fun(
            self.mean_regularizer, mean_batch, self.get_minibatch_momentum()
        )
        with torch.no_grad():
            self.mean_regularizer = mean.detach()
        return mean

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchNormSPD layer.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed (normalized then biased) SPD matrices
        """
        with eigh_context(getattr(self, "eigh_dtype", None)):
            if _use_batch_statistics(self, data):
                parameters, parameters_batch = self._get_norm_parameters(data)
                with torch.no_grad():
                    self._update_running_parameters(parameters_batch)
                self.training_step = self.training_step + 1
                if self.is_dynamic:
                    self.current_ref_step += 1
            else:
                # training over, use overall parameters learnt on all batches
                parameters = self._get_running_parameters()
            return self._norm_and_bias(data, parameters)

    @abstractmethod
    def _get_norm_parameters(self, data: torch.Tensor) -> tuple[Any, Any]:
        """
        Get parameters used to normalize data.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        parameters
            Parameters used for normalization

        parameters_batch
            Parameters of the batch of data
            (different from parameters in minibatch mode)
        """
        pass

    @abstractmethod
    def _update_running_parameters(self, parameters_batch) -> None:
        """
        Update running parameters.

        Parameters
        ----------
        parameters_batch
            Parameters of the batch of data
        """
        pass

    @abstractmethod
    def _get_running_parameters(self) -> Any:
        """
        Get running parameters for normalization in inference.

        Returns
        -------
        running_parameters
            Running parameters
        """
        pass

    @abstractmethod
    def _norm_and_bias(self, data: torch.Tensor, parameters) -> torch.Tensor:
        """
        Apply normalization and add bias.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Parameters
        ----------
            Parameters used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        pass

    @abstractmethod
    def _post_optimizer_hook(
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
    ) -> None:
        """
        Update the dynamic parametrization after ``optimizer.step()``.

        Moves the reference point. See ``torch.optim.Optimizer.register_step_post_hook``
        for the signature.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        pass

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Register the post-step hook with the optimizer.

        If dynamic parametrization, it needs to be called once after creating the
        optimizer for dynamic parametrization to actually work as expected

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if self.is_dynamic:
            optimizer.register_step_post_hook(self._post_optimizer_hook)

    def __repr__(self) -> str:
        """
        Representation of the layer.

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BatchNormSPDBase(\n"
            f"  n_features={self.n_features},\n"
            f"  mean_type={self.mean_type},\n"
            f"  mean_options={self.mean_options},\n"
            f"  momentum={self.momentum},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  use_autograd={self.use_autograd},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )

    def __str__(self) -> str:
        """
        Return the string representation of the layer.

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class BatchNormSPDMean(BatchNormSPDBase):
    r"""
    Riemannian batch normalization of SPD matrices: mean centring and bias.

    In training mode each batch is centred on its SPD mean :math:`\bar X`
    (computed in the geometry selected by ``mean_type``) and re-biased by a
    learnable SPD matrix :math:`G`; in evaluation mode the running mean replaces
    the batch mean. A batch holding a single matrix has no batch statistics and
    is normalized with the running mean, which is then left unchanged.

    Centring and biasing are either a whitening by the mean followed by a
    congruence by the bias (``norm_and_bias_mode="whitening"``, default) or a
    parallel transport from the mean to the bias
    (``norm_and_bias_mode="parallel_transport"``).

    """

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_and_bias_mode: str = "whitening",
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        eigh_dtype: torch.dtype | None = None,
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean.

        Only the SPD mean is normalized

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "affine_invariant_gevd",
            "log_euclidean", "arithmetic", "harmonic",
            "geometric_arithmetic_harmonic",
            "adaptive_geometric_arithmetic_harmonic", "bures_wasserstein"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            For adaptive_geometric_arithmetic_harmonic, {'t_init': t} sets the
            initial value of the learnable t in [0, 1] (0: harmonic,
            1: arithmetic). Default t_init is 0.5 (the GAH mean).
            For bures_wasserstein, one can set {'n_iterations': 1}.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening" (whitening by the mean, then congruence
            by the bias) and "parallel_transport" (parallel transport from the
            mean to the bias, in the geometry of mean_type; the arithmetic,
            harmonic and (adaptive) GAH means use the affine-invariant
            transport; not available for "bures_wasserstein"; with
            use_autograd=True, not available for the affine-invariant and
            log-Euclidean means, whose transport has NaN autograd gradients at
            the identity initialization of the bias)

        norm_strategy : str, optional
            Strategy for normalization.
            Default is "classical".
            Choices are: "classical" and "minibatch"

        minibatch_mode : str, optional
            How the minibatch momentum behaves during the training.
            Default is "constant".
            Choices are: "constant", "decay", "growth"

        minibatch_momentum : float, optional
            Momentum for mean regularization in minibatch normalization strategy.
            If minibatch_mode is "decay", this momentum corresponds to the minimum momentum that is reached.
            If minibatch_mode is "growth", this momentum corresponds to the initial momentum.
            Default is 0.01

        minibatch_maxstep : int, optional
            If minibatch_mode is "decay" or "growth", this is the training step at which the minibatch momentum
            attains its final value.
            Default is 100

        parametrization : str, optional
            Parametrization to apply on covariance bias.
            Default is "softplus".
            Choices are: "softplus", "exp"

        parametrization_mode : str, optional
            Parametrization mode.
            Default is "static".
            Choices are: "static" and "dynamic"

        n_steps_ref_update : int, optional
            If parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula.
            Default is False

        eigh_dtype : torch.dtype or None, optional
            dtype of the eigendecompositions of the forward pass (e.g.
            ``torch.float64`` in a float32 model: mixed precision, see
            :func:`~yetanotherspdnet.functions.spd_linalg.eigh_precision`).
            Default is None (the dtype of the data).

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
        """
        self.norm_and_bias_mode = norm_and_bias_mode
        if self.norm_and_bias_mode not in ("whitening", "parallel_transport"):
            raise ValueError(
                f"norm_and_bias_mode must be in ['whitening', 'parallel_transport'], got {self.norm_and_bias_mode}"
            )
        super().__init__(
            n_features,
            mean_type,
            mean_options,
            momentum,
            norm_strategy,
            minibatch_mode,
            minibatch_momentum,
            minibatch_maxstep,
            parametrization,
            parametrization_mode,
            n_steps_ref_update,
            use_autograd,
            device,
            dtype,
        )
        self.eigh_dtype = eigh_dtype

    def _init_bias_parameters(self) -> None:
        """Initialize bias parameters and their parametrization."""
        self.Covbias = torch.nn.Parameter(
            torch.eye(self.n_features, dtype=self.dtype, device=self.device)
        )
        # deal with parametrization
        self.spd_parametrization: SPDParametrization | SPDAdaptiveParametrization
        if self.parametrization_mode == "static":
            self.spd_parametrization = SPDParametrization(mapping=self.parametrization)
        elif self.parametrization_mode == "dynamic":
            self.spd_parametrization = SPDAdaptiveParametrization(
                self.n_features,
                initial_reference=self.Covbias.clone().detach(),
                mapping=self.parametrization,
            )
        register_parametrization(self, "Covbias", self.spd_parametrization)

    def _init_norm_and_bias(self) -> None:
        """Initialize normalization and bias functions."""
        if self.norm_and_bias_mode == "whitening":
            self.normalize_mean = select_path(
                whitening, Whitening.apply, self.use_autograd
            )
            self.add_bias_mean = select_path(
                congruence_SPD, CongruenceSPD.apply, self.use_autograd
            )
            self._norm_and_bias_fun = self._norm_and_bias_whitening
        elif self.norm_and_bias_mode == "parallel_transport":
            if self.geometry.transport is None:
                raise ValueError(
                    f"norm_and_bias_mode='parallel_transport' is not available "
                    f"for mean_type={self.mean_type!r}"
                )
            self.parallel_transport = self.geometry.transport.select(self.use_autograd)
            self._norm_and_bias_fun = self._norm_and_bias_transport

    def _norm_and_bias_whitening(
        self, data: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply normalization and add bias.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        parameters : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        return self.add_bias_mean(self.normalize_mean(data, parameters), self.Covbias)

    def _norm_and_bias_transport(
        self, data: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply normalization and add bias.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        parameters : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        return self.parallel_transport(data, parameters, self.Covbias)

    def _norm_and_bias(self, data: torch.Tensor, parameters) -> torch.Tensor:
        """
        Apply normalization and add bias.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        parameters : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        return self._norm_and_bias_fun(data, parameters)

    # the parameters are Any (as in BatchNormSPDBase), not Tensor: the subclass
    # BatchNormSPDMeanScalarVariance passes (mean, std) tuples instead
    def _get_norm_parameters(self, data: torch.Tensor) -> tuple[Any, Any]:
        """
        Get parameters used to normalize data.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization

        mean_batch : torch.Tensor of shape (n_features, n_features)
            SPD mean of data
            (different from mean in minibatch mode)
        """
        mean_batch = self.mean_fun(data)
        mean = self.get_norm_mean(mean_batch)
        return mean, mean_batch

    def _update_running_parameters(self, parameters_batch: Any) -> None:
        """
        Update running parameters.

        Parameters
        ----------
        parameters_batch : torch.Tensor of shape (n_features, n_features)
            SPD mean of the batch of data
        """
        self.running_mean = self.adaptive_mean_fun(
            self.running_mean.to(parameters_batch.device),
            parameters_batch,
            self.momentum,
        )

    def _get_running_parameters(self) -> Any:
        """
        Get running parameters for normalization in inference.

        Returns
        -------
        running_parameters
            Running mean
        """
        return self.running_mean

    def _post_optimizer_hook(
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
    ) -> None:
        """
        Update the dynamic parametrization after ``optimizer.step()``.

        Moves the reference point. See ``torch.optim.Optimizer.register_step_post_hook``
        for the signature.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if (
            self.training
            and self.is_dynamic
            and self.current_ref_step >= self.n_steps_ref_update
        ):
            with torch.no_grad():
                # update reference point
                # is_dynamic: the parametrization is the adaptive one
                cast(
                    SPDAdaptiveParametrization, self.spd_parametrization
                ).update_reference_point()
                # reset tangent vector to zero
                _parametrization_of(self, "Covbias").original.zero_()
                # reset counter
                self.current_ref_step = 0

    def __repr__(self) -> str:
        """
        Representation of the layer.

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BatchNormSPDMean(\n"
            f"  n_features={self.n_features},\n"
            f"  mean_type={self.mean_type},\n"
            f"  mean_options={self.mean_options},\n"
            f"  momentum={self.momentum},\n"
            f"  norm_and_bias_mode={self.norm_and_bias_mode},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  use_autograd={self.use_autograd},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )


class BatchNormSPDMeanScalarVariance(BatchNormSPDMean):
    r"""
    Riemannian batch normalization of SPD matrices: mean, scalar dispersion, bias.

    Extends :class:`BatchNormSPDMean` with a scalar dispersion: centred matrices
    are rescaled along geodesics from the identity by
    :math:`s / \sigma`, where :math:`\sigma` is the batch (or running) standard
    deviation and :math:`s` a learnable positive scalar. With
    ``mean_type="bures_wasserstein"`` the layer implements GBWBN (generalized
    Bures-Wasserstein batch normalization) with learnable pre/post transforms.

    """

    # buffers registered in __init__ (declared for type checkers)
    running_std_scalar: torch.Tensor
    std_regularizer: torch.Tensor
    bw_metric_initialized: torch.Tensor

    # version 2: the GBWBN running mean is stored in the input space (version 1
    # stored it in the transformed space of M, see _load_from_state_dict)
    _version = 2

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        bw_theta: float = 0.5,
        bw_batch_stats_grad: bool = True,
        bw_init_metric: str = "identity",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        eigh_dtype: torch.dtype | None = None,
    ) -> None:
        r"""
        Batch normalization layer for SPDnet relying on a SPD mean.

        Both the SPD mean and scalar variance are normalized

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "affine_invariant_gevd",
            "log_euclidean", "arithmetic", "harmonic",
            "geometric_arithmetic_harmonic",
            "adaptive_geometric_arithmetic_harmonic", "bures_wasserstein"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            For adaptive_geometric_arithmetic_harmonic, {'t_init': t} sets the
            initial value of the learnable t in [0, 1] (0: harmonic,
            1: arithmetic). Default t_init is 0.5 (the GAH mean).
            For bures_wasserstein, one can set {'n_iterations': 1}.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_strategy : str, optional
            Strategy for normalization.
            Default is "classical".
            Choices are: "classical" and "minibatch"

        minibatch_mode : str, optional
            How the minibatch momentum behaves during the training.
            Default is "constant".
            Choices are: "constant", "decay", "growth"

        minibatch_momentum : float, optional
            Momentum for mean regularization in minibatch normalization strategy.
            If minibatch_mode is "decay", this momentum corresponds to the minimum momentum that is reached.
            If minibatch_mode is "growth", this momentum corresponds to the initial momentum.
            Default is 0.01

        minibatch_maxstep : int, optional
            If minibatch_mode is "decay" or "growth", this is the training step at which the minibatch momentum
            attains its final value.
            Default is 100

        parametrization : str, optional
            Parametrization to apply on covariance bias.
            Default is "softplus".
            Choices are: "softplus", "exp"

        parametrization_mode : str, optional
            Parametrization mode.
            Default is "static".
            Choices are: "static" and "dynamic"

        n_steps_ref_update : int, optional
            If parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula.
            Default is False

        eigh_dtype : torch.dtype or None, optional
            dtype of the eigendecompositions of the forward pass (e.g.
            ``torch.float64`` in a float32 model: mixed precision, see
            :func:`~yetanotherspdnet.functions.spd_linalg.eigh_precision`).
            Default is None (the dtype of the data).

        bw_theta : float, optional
            Power deformation :math:`\theta` of the generalized Bures-Wasserstein
            metric (GBWBN, Wang et al. 2025): data are mapped by
            :math:`X \mapsto M^{-1/2} X^\theta M^{-1/2}` and the variance is
            measured with the deformed metric (divided by :math:`\theta^2`).
            Only used when mean_type="bures_wasserstein". theta=1.0 is the
            plain BW metric; 0.5 is the best value of the paper's ablation
            (HDM05: 69.1% vs 62.0% for theta=1) and the reference code's default.
            Must be finite and nonzero (ValueError otherwise); a negative theta is
            allowed (the variance is divided by :math:`\theta^2`, the std by
            :math:`|\theta|`).
            Default is 0.5

        bw_batch_stats_grad : bool, optional
            GBWBN only. If True (as in the paper), gradients flow through the
            batch mean and variance; if False (as in the reference code), they
            are computed without gradient, as constants.
            Default is True

        bw_init_metric : str, optional
            GBWBN only. Initialization of :math:`M` and :math:`G`.
            ``"identity"``: :math:`M = G = I`. ``"trace_scale"``: on the first
            training batch, :math:`M = cI` and :math:`G = c^{1/\theta} I` with
            :math:`c = \operatorname{tr}(\bar B) / n`, :math:`\bar B` the BW
            mean of the :math:`X_i^\theta` (and the running mean is set to
            :math:`c^{1/\theta} I`, the identity of the transformed space).
            The transformed batch then has unit scale, which removes the
            folding of :math:`\mathrm{Exp}_I` caused by the data scale (the BW
            metric is not scale invariant), and the layer outputs keep the
            scale of the data instead of being centred at :math:`I`.
            Default is "identity"

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64

        Attributes
        ----------
        Covbias : torch.Tensor of shape (n_features, n_features)
            Learnable SPD bias (not created for mean_type="bures_wasserstein",
            which uses bw_G).
        stdScalarbias : torch.Tensor
            Learnable positive scalar scale :math:`s`.
        running_mean : torch.Tensor of shape (n_features, n_features)
            Buffer: running SPD mean. For GBWBN, it is stored in the input
            space, :math:`B_r = (M^{1/2} \hat B_r M^{1/2})^{1/\theta}` where
            :math:`\hat B_r` is the running mean in the transformed space, and
            mapped with the current :math:`M` when used (see the batch
            normalization guide for what that guarantees).
        running_std_scalar : torch.Tensor
            Buffer: running scalar standard deviation (for GBWBN, in the deformed
            metric of the :math:`M` of each update).
        bw_M, bw_G : torch.Tensor of shape (n_features, n_features)
            GBWBN only: learnable SPD pre-transform :math:`M` and bias
            :math:`G`, applied in the transformed space as
            :math:`\hat G = M^{-1/2} G^\theta M^{-1/2}`.
        bw_theta : float
            GBWBN only: power of the pre-transform :math:`X \mapsto X^\theta`.
        """
        super().__init__(
            n_features=n_features,
            mean_type=mean_type,
            mean_options=mean_options,
            momentum=momentum,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=minibatch_momentum,
            minibatch_maxstep=minibatch_maxstep,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            n_steps_ref_update=n_steps_ref_update,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            eigh_dtype=eigh_dtype,
        )
        self.eigh_dtype = eigh_dtype

        # Bures-Wasserstein specific parameters
        if self.mean_type == "bures_wasserstein":
            self.bw_theta = _check_bw_theta(bw_theta)
            self.bw_batch_stats_grad = bw_batch_stats_grad
            if bw_init_metric not in ("identity", "trace_scale"):
                raise ValueError(
                    "bw_init_metric must be in ['identity', 'trace_scale'], "
                    f"got {bw_init_metric!r}"
                )
            self.bw_init_metric = bw_init_metric
            if bw_init_metric == "trace_scale":
                # persistent: a resumed or loaded layer is not initialized again
                self.register_buffer(
                    "bw_metric_initialized", torch.tensor(False, device=self.device)
                )
            # M: learnable SPD metric (n x n); G: learnable bias on the input
            # manifold (n x n), mapped to the transformed space as
            # M^{-1/2} G^theta M^{-1/2} like the data. Both start at I, with the
            # parametrization of Covbias (static, or dynamic: tangent vector at
            # a reference point moved every n_steps_ref_update steps)
            self.bw_parametrizations: dict[
                str, SPDParametrization | SPDAdaptiveParametrization
            ] = {}
            for name in ("bw_M", "bw_G"):
                eye = torch.eye(n_features, dtype=self.dtype, device=self.device)
                setattr(self, name, torch.nn.Parameter(eye))
                spd_parametrization: SPDParametrization | SPDAdaptiveParametrization
                if self.is_dynamic:
                    spd_parametrization = SPDAdaptiveParametrization(
                        n_features,
                        initial_reference=eye.clone(),
                        mapping=self.parametrization,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    spd_parametrization = SPDParametrization(
                        mapping=self.parametrization
                    )
                self.bw_parametrizations[name] = spd_parametrization
                register_parametrization(self, name, spd_parametrization)

    def _init_norm_parameters(self) -> None:
        """Initialize the batch, normalization and running parameters."""
        # init mean parameters from parent class
        super()._init_norm_parameters()
        # init scalar std parameters and functions
        self._init_std()
        self._init_norm_strategy_std()
        # set running std
        self.register_buffer(
            "running_std_scalar",
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )

    def _init_bias_parameters(self) -> None:
        """Initialize bias parameters and their parametrization."""
        # init cov bias from parent class (GBWBN uses its own bias bw_G instead)
        if self.mean_type != "bures_wasserstein":
            super()._init_bias_parameters()
        # scalar variance bias parameter
        self.stdScalarbias = torch.nn.Parameter(
            torch.ones((), dtype=self.dtype, device=self.device)
        )
        register_parametrization(self, "stdScalarbias", ScalarSoftPlusParametrization())

    def _init_norm_and_bias(self) -> None:
        """Initialize normalization and bias functions."""
        if self.mean_type == "bures_wasserstein":
            # GBWBN has its own pipeline (see _forward_bures_wasserstein)
            return
        # init parent class norm and bias functions
        super()._init_norm_and_bias()
        # normalize and add bias scalar variance function
        self.norm_and_bias_var = select_path(
            lambda data, exponent: powm_SPD(data, exponent)[0],
            PowmSPD.apply,
            self.use_autograd,
        )

    def _init_std(self) -> None:
        """Select the scalar standard deviation of the geometry."""
        self.std_fun = self.geometry.std.select(self.use_autograd)

    def _init_norm_strategy_std(self) -> None:
        """Auxiliary function to select norm strategy."""
        if self.norm_strategy == "classical":
            self.get_norm_std = lambda std: std
        elif self.norm_strategy == "minibatch":
            self.get_norm_std = self._handle_minibatch_std
            self.register_buffer(
                "std_regularizer",
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
            )

    def _handle_minibatch_std(self, std_scalar_batch: torch.Tensor) -> torch.Tensor:
        """Auxiliary function to handle var used for normalization with minibatch strategy."""
        std_scalar = self.adaptive_std_fun(
            self.std_regularizer, std_scalar_batch, self.get_minibatch_momentum()
        )
        with torch.no_grad():
            self.std_regularizer = std_scalar.detach()
        return std_scalar

    def adaptive_std_fun(
        self,
        running_std_scalar: torch.Tensor,
        std_scalar_batch: torch.Tensor,
        momentum: float,
    ) -> torch.Tensor:
        r"""
        Combine two scalar standard deviations (running update of the variance).

        Parameters
        ----------
        running_std_scalar : torch.Tensor of shape ()
            Current (running or regularized) scalar standard deviation

        std_scalar_batch : torch.Tensor of shape ()
            Scalar standard deviation of the batch

        momentum : float
            Weight of the batch

        Returns
        -------
        std_scalar : torch.Tensor of shape ()
            :math:`\sqrt{(1 - m)\,\sigma_r^2 + m\,\sigma_b^2}`
        """
        std_scalar = torch.sqrt(
            (1 - momentum) * running_std_scalar**2 + momentum * std_scalar_batch**2
        )
        return std_scalar

    def _get_norm_parameters(
        self, data: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Get parameters used to normalize data.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        parameters : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            SPD mean and scalar std used for normalization

        parameters_batch : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            SPD mean and scalar std of data
            (different from mean in minibatch mode)
        """
        mean, mean_batch = super()._get_norm_parameters(data)
        std_batch = self.std_fun(data, mean)
        std = self.get_norm_std(std_batch)
        return (mean, std), (mean_batch, std_batch)

    def _update_running_parameters(
        self, parameters_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """
        Update running parameters.

        Parameters
        ----------
        parameters_batch : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            SPD mean and scalar std of the batch of data
        """
        super()._update_running_parameters(parameters_batch[0])
        self.running_std_scalar = self.adaptive_std_fun(
            self.running_std_scalar.to(parameters_batch[1].device),
            parameters_batch[1],
            self.momentum,
        )

    def _get_running_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get running parameters for normalization in inference.

        Returns
        -------
        running_parameters : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            Running mean and scalar std
        """
        return (self.running_mean, self.running_std_scalar)

    def _post_optimizer_hook(
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
    ) -> None:
        """
        Update the dynamic parametrization after ``optimizer.step()``.

        For GBWBN (no Covbias), moves the reference points of :math:`M` and
        :math:`G` to their current values (computed from the parameters after
        the step, so the values are continuous across the update) and resets
        their tangent vectors to zero. See
        ``torch.optim.Optimizer.register_step_post_hook`` for the signature.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if self.mean_type != "bures_wasserstein":
            super()._post_optimizer_hook(_optimizer, *_args, **_kwargs)
            return
        if not (
            self.training
            and self.is_dynamic
            and self.current_ref_step >= self.n_steps_ref_update
        ):
            return
        with torch.no_grad():
            for name in ("bw_M", "bw_G"):
                self._move_reference_point(name, getattr(self, name))
            self.current_ref_step = 0

    def _init_metric_trace_scale(self, data: torch.Tensor) -> None:
        r"""
        Set M = cI and G = c^(1/theta) I from a batch (``bw_init_metric="trace_scale"``).

        :math:`c = \operatorname{tr}(\bar B) / n` with :math:`\bar B` the BW mean
        of the :math:`X_i^\theta`: the transformed data
        :math:`X_i^\theta / c` have unit scale and the transformed bias
        :math:`\hat G = G^\theta / c` is the identity. The running mean is set
        to :math:`c^{1/\theta} I` (the identity of the transformed space).
        """
        theta = self.bw_theta
        with torch.no_grad():
            data_pow = powm_SPD(data, theta)[0] if theta != 1.0 else data
            mean = self.mean_fun(data_pow.detach())
            scale = torch.diagonal(mean, dim1=-2, dim2=-1).sum() / self.n_features
            eye = torch.eye(self.n_features, dtype=self.dtype, device=data.device)
            self._move_reference_point("bw_M", scale * eye)
            self._move_reference_point("bw_G", scale ** (1.0 / theta) * eye)
            self.running_mean = scale ** (1.0 / theta) * eye
            self.bw_metric_initialized.fill_(True)

    def _move_reference_point(self, name: str, value: torch.Tensor) -> None:
        """
        Set the GBWBN SPD parameter *name* to *value*, as the reference point if dynamic.

        Dynamic: the reference point becomes *value* and the tangent vector is
        reset to zero. Static: the unconstrained tensor is set to the inverse
        of the parametrization at *value*.
        """
        parametrization = self.bw_parametrizations[name]
        with torch.no_grad():
            if isinstance(parametrization, SPDAdaptiveParametrization):
                parametrization.last_spd_value.copy_(value)
                parametrization.update_reference_point()
                _parametrization_of(self, name).original.zero_()
            else:
                _parametrization_of(self, name).original.copy_(
                    parametrization.right_inverse(value)
                )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        r"""
        Load GBWBN checkpoints of earlier versions.

        - GBWBN layers saved an unused Covbias before it was removed from that
          mode: it is dropped.
        - Version 1 (``local_metadata["version"]`` missing or 1, as for every
          checkpoint saved before the running mean moved to the input space)
          stored the GBWBN running mean :math:`\hat B_r` in the transformed
          space of the :math:`M` of the time; it is converted to the input space,
          :math:`B_r = (M^{1/2} \hat B_r M^{1/2})^{1/\theta}`, with the
          checkpoint's :math:`M` and the layer's :math:`\theta`, so the layer
          gives the same outputs as before. Keep the ``_metadata`` attribute of
          the state dict (``torch.save(module.state_dict())`` does) so that a
          version-2 checkpoint is not converted twice.
        - Version-1 checkpoints of ``parametrization_mode="dynamic"`` had static
          parametrizations of :math:`M` and :math:`G` (reference point
          :math:`I`): the missing reference buffers are set to :math:`I`, which
          gives the same values.
        """
        if self.mean_type == "bures_wasserstein":
            stale = prefix + "parametrizations.Covbias."
            for key in [k for k in state_dict if k.startswith(stale)]:
                del state_dict[key]
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                self._upgrade_bw_state_dict_v1(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _upgrade_bw_state_dict_v1(self, state_dict: dict, prefix: str) -> None:
        """Convert a version-1 GBWBN state dict in place (see ``_load_from_state_dict``)."""
        eye = torch.eye(self.n_features, dtype=self.dtype, device=self.device)
        for name in ("bw_M", "bw_G"):
            parametrization = self.bw_parametrizations[name]
            if isinstance(parametrization, SPDAdaptiveParametrization):
                for buffer in (
                    "reference_point",
                    "reference_point_sqrtm",
                    "reference_point_inv_sqrtm",
                    "last_spd_value",
                ):
                    state_dict.setdefault(
                        f"{prefix}parametrizations.{name}.0.{buffer}", eye.clone()
                    )
        original_key = prefix + "parametrizations.bw_M.original"
        mean_key = prefix + "running_mean"
        if original_key not in state_dict or mean_key not in state_dict:
            return
        with torch.no_grad():
            # version 1 charts are static (reference I): M = mapping(original)
            mapping = SPDParametrization(mapping=self.parametrization)
            M = mapping(state_dict[original_key].to(self.dtype))
            M_sqrt = sqrtm_SPD(M)[0]
            mean = M_sqrt @ state_dict[mean_key].to(M) @ M_sqrt
            if self.bw_theta != 1.0:
                mean = powm_SPD(mean, 1.0 / self.bw_theta)[0]
            state_dict[mean_key] = mean

    def _norm_and_bias(
        self, data: torch.Tensor, parameters: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply normalization and add bias.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        parameters : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            SPD mean and scalar std used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """
        data_normalized = self.normalize_mean(data, parameters[0])
        # guard: a batch of identical matrices has std 0; the exponent s / std
        # is bounded by s / STD_FLOOR = s / sqrt(BW_SCALE_EPS), as in the GBWBN
        # scaling s / sqrt(var + eps) (no effect for std >= STD_FLOOR = 3.2e-3)
        std = parameters[1].clamp_min(STD_FLOOR)
        data_std_transf = self.norm_and_bias_var(
            data_normalized, self.stdScalarbias / std
        )
        data_transformed = self.add_bias_mean(data_std_transf, self.Covbias)
        return data_transformed

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of SPD matrices.

        GBWBN pipeline for ``mean_type="bures_wasserstein"``, otherwise the generic one.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed (normalized then biased) SPD matrices
        """
        with eigh_context(self.eigh_dtype):
            if self.mean_type == "bures_wasserstein":
                return self._forward_bures_wasserstein(data)
            return super().forward(data)

    def _forward_bures_wasserstein(self, data: torch.Tensor) -> torch.Tensor:
        """
        GBWBN forward pass (Generalized Bures-Wasserstein Batch Normalization).

        Pipeline: pre-transform -> BW center -> BW scale -> BW bias -> post-transform
        (Algorithm 1 of Wang et al., "GBWBN", 2025). The variance is the one of
        the theta-deformed metric, d_BW^2(X^theta, Y^theta) / theta^2.
        """
        theta = self.bw_theta
        if (
            self.bw_init_metric == "trace_scale"
            and not bool(self.bw_metric_initialized)
            and self.training
            and data.shape[:-2].numel() >= 2
        ):
            self._init_metric_trace_scale(data)
        M = self.bw_M

        # Pre-transform: X_hat = M^{-1/2} X^theta M^{-1/2}
        # bw_M is initialized at the identity: with use_autograd=False the
        # manual (Daleckii-Krein) backwards keep its gradient finite, whereas
        # autograd through torch.linalg.eigh is undefined for repeated
        # eigenvalues
        inv_sqrtm = select_path(
            lambda x: inv_sqrtm_SPD(x)[0], InvSqrtmSPD.apply, self.use_autograd
        )
        sqrtm = select_path(
            lambda x: sqrtm_SPD(x)[0], SqrtmSPD.apply, self.use_autograd
        )

        def _powm_manual(x: torch.Tensor, p: float) -> torch.Tensor:
            exponent = torch.tensor(p, dtype=x.dtype, device=x.device)
            return PowmSPD.apply(x, exponent)

        powm = select_path(
            lambda x, p: powm_SPD(x, p)[0], _powm_manual, self.use_autograd
        )
        center, scale, bias = (
            BURES_WASSERSTEIN_STEPS[step].select(self.use_autograd)
            for step in ("center", "scale", "bias")
        )

        M_isqrt = inv_sqrtm(M)
        M_sqrt = sqrtm(M)

        def to_transformed(point: torch.Tensor, power: Callable = powm) -> torch.Tensor:
            # input space -> transformed space: P -> M^{-1/2} P^theta M^{-1/2}
            return M_isqrt @ (power(point, theta) if theta != 1.0 else point) @ M_isqrt

        def to_input(point: torch.Tensor) -> torch.Tensor:
            # transformed space -> input space: P -> (M^{1/2} P M^{1/2})^{1/theta}
            result = M_sqrt @ point @ M_sqrt
            return powm(result, 1.0 / theta) if theta != 1.0 else result

        def powm_no_grad(x: torch.Tensor, p: float) -> torch.Tensor:
            return powm_SPD(x, p)[0]

        data_hat = to_transformed(data)
        G_hat = to_transformed(cast(torch.Tensor, self.bw_G))

        if _use_batch_statistics(self, data):
            with torch.set_grad_enabled(
                self.bw_batch_stats_grad and torch.is_grad_enabled()
            ):
                mean_batch = self.mean_fun(data_hat)
                mean = self.get_norm_mean(mean_batch)
                std_batch = self.std_fun(data_hat, mean) / abs(theta)
                std = self.get_norm_std(std_batch)
            with torch.no_grad():
                # the running mean is stored in the input space (see
                # _load_from_state_dict): pulled to the transformed space of the
                # current M, moved along the BW geodesic, pushed back
                running_mean = to_transformed(
                    self.running_mean.to(data.device), powm_no_grad
                )
                self.running_mean = to_input(
                    self.adaptive_mean_fun(running_mean, mean_batch, self.momentum)
                )
                self.running_std_scalar = self.adaptive_std_fun(
                    self.running_std_scalar.to(data.device), std_batch, self.momentum
                )
            self.training_step = self.training_step + 1
            if self.is_dynamic:
                self.current_ref_step += 1
        else:
            mean = to_transformed(self.running_mean.to(data.device), powm_no_grad)
            std = self.running_std_scalar.to(data.device)

        # BW center, scale, bias
        centered = center(data_hat, mean)
        scaled = scale(centered, std**2, self.stdScalarbias)
        biased = bias(scaled, G_hat)

        # Post-transform: output = (M^{1/2} X_tilde M^{1/2})^{1/theta}
        return to_input(biased)

    def __repr__(self) -> str:
        """
        Representation of the layer.

        Returns
        -------
        str
            Representation of the layer
        """
        bw_info = ""
        if self.mean_type == "bures_wasserstein":
            bw_info = (
                f"  bw_theta={self.bw_theta},\n"
                f"  bw_batch_stats_grad={self.bw_batch_stats_grad},\n"
            )
            # shown only when not the default, so that the model hashes of
            # default configurations are unchanged
            if self.bw_init_metric != "identity":
                bw_info += f"  bw_init_metric={self.bw_init_metric},\n"
        return (
            f"BatchNormSPDMeanScalarVariance(\n"
            f"  n_features={self.n_features},\n"
            f"  mean_type={self.mean_type},\n"
            f"  mean_options={self.mean_options},\n"
            f"  momentum={self.momentum},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  use_autograd={self.use_autograd},\n"
            f"{bw_info}"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )
