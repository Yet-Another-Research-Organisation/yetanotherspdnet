from functools import partial

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from yetanotherspdnet.functions.spd_geometries.affine_invariant_gevd import (
    AffineInvariantGeodesicGEVD,
    AffineInvariantMeanGEVD,
    AffineInvariantParallelTransportGEVD,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    EuclideanGeodesic,
    HarmonicCurve,
    HarmonicMean,
    LeftKullbackLeiblerStdScalar,
    RightKullbackLeiblerStdScalar,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    GeometricArithmeticHarmonicMean,
    GeometricEuclideanHarmonicCurve,
    SymmetrizedKullbackLeiblerStdScalar,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    LogEuclideanGeodesic,
    LogEuclideanMean,
    LogEuclideanParalleTransportSPD,
    LogEuclideanStdScalar,
)

from ..functions.spd_geometries.affine_invariant import (
    AffineInvariantCoordinatesTransport,
    AffineInvariantGeodesic,
    AffineInvariantMean,
    AffineInvariantParallelTransport,
    AffineInvariantStdScalar,
)
from .parametrizations import (
    SPDAdaptiveParametrization,
    ScalarSoftPlusParametrization,
    SPDParametrization,
)


# -------------------------------------------------- #
# Identity function to avoid lambdas, so that        #
# the module can be pickled with torch.save(model)   #
# -------------------------------------------------- #
def _identity(x):
    return x


# ---------------------------- #
# Minibatch momentum schedules #
# ---------------------------- #
def _minibatch_momentum_constant(momentum: float, maxstep: int, step: int) -> float:
    """Constant minibatch momentum."""
    return momentum


def _minibatch_momentum_decay(momentum: float, maxstep: int, step: int) -> float:
    """Decreasing minibatch momentum; `momentum` is the minimum reached."""
    return 1 - momentum ** (max(maxstep - step, 0) / maxstep) + momentum


def _minibatch_momentum_growth(momentum: float, maxstep: int, step: int) -> float:
    """Increasing minibatch momentum; `momentum` is the initial value."""
    if step == 0:
        return momentum
    return momentum + (1 - momentum) ** (maxstep / min(step, maxstep))


class BatchNormSPDBase(ABC, nn.Module):
    _NORM_AND_BIAS_MODES = ("whitening", "parallel_transport")
    _NORM_STRATEGIES = ("classical", "minibatch")
    _PARAMETRIZATIONS = ("softplus", "exp")
    _PARAMETRIZATION_MODES = ("static", "dynamic")

    _MINIBATCH_MOMENTUM_FUNS = {
        "constant": _minibatch_momentum_constant,
        "decay": _minibatch_momentum_decay,
        "growth": _minibatch_momentum_growth,
    }

    # Batch mean computation
    _MEAN_FUNS = {
        "affine_invariant": AffineInvariantMean,
        "affine_invariant_gevd": AffineInvariantMeanGEVD,
        "log_euclidean": LogEuclideanMean,
        "arithmetic": ArithmeticMean.apply,
        "harmonic": HarmonicMean.apply,
        "geometric_arithmetic_harmonic": GeometricArithmeticHarmonicMean,
    }

    # Geodesics / curve functions used for:
    #    - Running mean adaptive update
    #    - Regularization in minibatch strategy
    _CURVE_FUNS = {
        "affine_invariant": AffineInvariantGeodesic.apply,
        "affine_invariant_gevd": AffineInvariantGeodesicGEVD.apply,
        "log_euclidean": LogEuclideanGeodesic,
        "arithmetic": EuclideanGeodesic.apply,
        "harmonic": HarmonicCurve.apply,
        "geometric_arithmetic_harmonic": GeometricEuclideanHarmonicCurve,
    }

    # To check that all mean tables actually handle the same means
    _MEAN_LINKED_TABLES: tuple[str, ...] = (
        "_MEAN_FUNS",
        "_CURVE_FUNS",
    )

    # Table consistency check                                                 #
    @classmethod
    def _check_mean_tables(cls) -> None:
        """Ensure every mean-linked table covers every known mean type."""
        reference = set(cls._MEAN_FUNS)
        for name in cls._MEAN_LINKED_TABLES:
            missing = reference - set(getattr(cls, name))
            if missing:
                raise TypeError(
                    f"{cls.__name__}.{name} is missing entries for "
                    f"mean types {sorted(missing)}"
                )

    # also ensure check performed for subclasses at the right time
    # after init because more mean tables will be added in subclasses
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._check_mean_tables()

    # to check categorical arguments are correct
    @staticmethod
    def _check_choice(name: str, value, choices) -> None:
        if value not in choices:
            raise ValueError(f"{name} must be in {list(choices)}, got {value!r}")

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_only: bool = False,
        norm_and_bias_mode: str = "whitening",
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Base class for battch normalization layers for SPDnet

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "affine_invariant_gevd", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_only : bool, optional
            Whether to apply some bias or only perform normalization.
            Default is False

        norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening", "parallel_transport"

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

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
        """
        super().__init__()

        # validate categorical arguments
        self._check_choice("mean_type", mean_type, self._MEAN_FUNS)
        self._check_choice(
            "norm_and_bias_mode", norm_and_bias_mode, self._NORM_AND_BIAS_MODES
        )
        self._check_choice("norm_strategy", norm_strategy, self._NORM_STRATEGIES)
        if norm_strategy == "minibatch":
            self._check_choice(
                "minibatch_mode", minibatch_mode, self._MINIBATCH_MOMENTUM_FUNS
            )
        self._check_choice("parametrization", parametrization, self._PARAMETRIZATIONS)
        self._check_choice(
            "parametrization_mode", parametrization_mode, self._PARAMETRIZATION_MODES
        )

        # register all input variables
        self.n_features = n_features
        self.mean_type = mean_type
        self.mean_options = mean_options
        self.momentum = momentum
        self.norm_only = norm_only
        self.norm_and_bias_mode = norm_and_bias_mode
        self.norm_strategy = norm_strategy
        self.minibatch_mode = minibatch_mode
        self.minibatch_momentum = minibatch_momentum
        self.minibatch_maxstep = minibatch_maxstep
        self.parametrization = parametrization
        self.parametrization_mode = parametrization_mode
        self.n_steps_ref_update = n_steps_ref_update
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

        if self.norm_only:
            # no bias parameters, so no init of them
            # no bias, also ensure not entering _post_optimizer_hook
            self.is_dynamic = False
            self._init_neutral_bias()
        else:
            # initialize bias parameters
            self._init_bias_parameters()

        # initialize normalization and bias functions
        self._init_norm_and_bias()

    def _init_norm_parameters(self) -> None:
        """
        Initialize functions to compute batch parameters, normalization parameters,
        and running parameters
        """
        # select mean function
        fun = self._MEAN_FUNS[self.mean_type]
        options = self.mean_options or {}
        self.mean_fun = partial(fun, **options) if options else fun

        # select adaptive mean update function
        self.geodesic_fun = self._CURVE_FUNS[self.mean_type]

        self._init_norm_strategy_mean()

        # set running mean
        self.register_buffer(
            "running_mean",
            torch.eye(self.n_features, dtype=self.dtype, device=self.device),
        )

    def _init_norm_strategy_mean(self) -> None:
        """
        Auxiliary function to select normalization strategy
        """
        if self.norm_strategy == "classical":
            self.get_norm_mean = _identity
        elif self.norm_strategy == "minibatch":
            self.get_norm_mean = self._handle_minibatch_mean
            self.register_buffer(
                "mean_regularizer",
                torch.eye(self.n_features, dtype=self.dtype, device=self.device),
            )
            self._minibatch_momentum_fun = self._MINIBATCH_MOMENTUM_FUNS[
                self.minibatch_mode
            ]

    @abstractmethod
    def _init_neutral_bias(self) -> None:
        """
        Set bias parameters as neutral parameters.
        They need to be constant, i.e. require_grad must be False
        """

    @abstractmethod
    def _init_bias_parameters(self) -> None:
        """
        Initialize bias parameters and their parametrization
        """

    @abstractmethod
    def _init_norm_and_bias(self) -> None:
        """
        Initialize normalization and bias functions
        """

    def _get_minibatch_momentum(self) -> float:
        """
        Current minibatch momentum, according to minibatch_mode and training_step
        """
        return self._minibatch_momentum_fun(
            self.minibatch_momentum, self.minibatch_maxstep, self.training_step
        )

    def _handle_minibatch_mean(self, mean_batch: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to handle mean used for normalization with minibatch strategy

        Parameters
        ----------
        mean_batch : torch.Tensor of shape (n_features, n_features)
            SPD mean of the batch

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            SPD mean used for normalization
        """
        mean = self.geodesic_fun(
            self.mean_regularizer, mean_batch, self._get_minibatch_momentum()
        )
        with torch.no_grad():
            self.mean_regularizer = mean.detach()
        return mean

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchNormSPD layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed (normalized then biased) SPD matrices
        """
        if self.training:
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
        Get parameters used to normalize data

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

    @abstractmethod
    def _update_running_parameters(self, parameters_batch) -> None:
        """
        Update running parameters

        Parameters
        ----------
        parameters_batch
            Parameters of the batch of data
        """

    @abstractmethod
    def _get_running_parameters(self) -> Any:
        """
        Get running parameters for normalization in inference

        Returns
        -------
        running_parameters
            Running parameters
        """

    @abstractmethod
    def _norm_and_bias(self, data: torch.Tensor, parameters) -> torch.Tensor:
        """
        Apply normalization and add bias

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        parameters
            Parameters used for normalization

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Transformed batch of SPD matrices
        """

    @abstractmethod
    def _post_optimizer_hook(
        self, optimizer: torch.optim.Optimizer, *args, **kwargs
    ) -> None:
        """
        Hook that runs after optimizer.step() to handle dynamic parametrization reference point.
        See torch.optim.Optimizer.register_step_post_hook method for more details

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Register the post-step hook with the optimizer.
        If dynamic parametrization, it needs to be called once after creating
        the optimizer for dynamic parametrization to actually work as expected

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if self.is_dynamic:
            optimizer.register_step_post_hook(self._post_optimizer_hook)

    def __repr__(self) -> str:
        """
        Representation of the layer

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
            f"  norm_only={self.norm_only},\n"
            f"  norm_and_bias_mode={self.norm_and_bias_mode},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class BatchNormSPDMean(BatchNormSPDBase):
    # Transport functions used in "parallel_transport" mode.
    # It must cover all mean types
    _TRANSPORT_FUNS = {
        "affine_invariant": AffineInvariantParallelTransport,
        "affine_invariant_gevd": AffineInvariantParallelTransportGEVD,
        "log_euclidean": LogEuclideanParalleTransportSPD,
        "arithmetic": AffineInvariantParallelTransportGEVD,
        "harmonic": AffineInvariantParallelTransportGEVD,
        "geometric_arithmetic_harmonic": AffineInvariantParallelTransportGEVD,
    }
    # It must be added to _MEAN_LINKED_TABLES to check that it's ok
    _MEAN_LINKED_TABLES = BatchNormSPDBase._MEAN_LINKED_TABLES + ("_TRANSPORT_FUNS",)

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_only: bool = False,
        norm_and_bias_mode: str = "whitening",
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
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
            Choices are: "affine_invariant", "affine_invariant_gevd", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_only : bool, optional
            Whether to apply some bias or only perform normalization.
            Default is False

        norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening", "parallel_transport", "norm_only"

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

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
        """
        super().__init__(
            n_features=n_features,
            mean_type=mean_type,
            mean_options=mean_options,
            momentum=momentum,
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=minibatch_momentum,
            minibatch_maxstep=minibatch_maxstep,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            n_steps_ref_update=n_steps_ref_update,
            device=device,
            dtype=dtype,
        )

    def _init_neutral_bias(self) -> None:
        """
        Set bias parameters as neutral parameters
        """
        self.Covbias = torch.eye(
            self.n_features, dtype=self.dtype, device=self.device, requires_grad=False
        )

    def _init_bias_parameters(self) -> None:
        """
        Initialize bias parameters and their parametrization
        """
        self.Covbias = torch.nn.Parameter(
            torch.eye(self.n_features, dtype=self.dtype, device=self.device)
        )
        # deal with parametrization
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
        """
        Initialize normalization and bias functions
        """
        if self.norm_and_bias_mode == "whitening":
            self.transport_fun = AffineInvariantCoordinatesTransport
        elif self.norm_and_bias_mode == "parallel_transport":
            self.transport_fun = self._TRANSPORT_FUNS[self.mean_type]

    def _norm_and_bias(self, data: torch.Tensor, parameters) -> torch.Tensor:
        """
        Apply normalization and add bias

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
        return self.transport_fun(data, parameters, self.Covbias)

    def _get_norm_parameters(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters used to normalize data

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

    def _update_running_parameters(self, parameters_batch: torch.Tensor) -> None:
        """
        Update running parameters

        Parameters
        ----------
        parameters_batch : torch.Tensor of shape (n_features, n_features)
            SPD mean of the batch of data
        """
        self.running_mean = self.geodesic_fun(
            self.running_mean.to(parameters_batch.device),
            parameters_batch,
            self.momentum,
        )

    def _get_running_parameters(self) -> torch.Tensor:
        """
        Get running parameters for normalization in inference

        Returns
        -------
        running_parameters
            Running mean
        """
        return self.running_mean

    def _post_optimizer_hook(
        self, optimizer: torch.optim.Optimizer, *args, **kwargs
    ) -> None:
        """
        Hook that runs after optimizer.step() to handle dynamic parametrization reference point.
        See torch.optim.Optimizer.register_step_post_hook method for more details

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
                self.spd_parametrization.update_reference_point()
                # reset tangent vector to zero
                self.parametrizations.Covbias.original.zero_()
                # reset counter
                self.current_ref_step = 0

    def __repr__(self) -> str:
        """
        Representation of the layer

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
            f"  norm_only={self.norm_only},\n"
            f"  norm_and_bias_mode={self.norm_and_bias_mode},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )


class BatchNormSPDMeanScalarVariance(BatchNormSPDMean):
    # Transport functions used in "parallel_transport" mode.
    # It must cover all mean types
    _STD_FUNS = {
        "affine_invariant": AffineInvariantStdScalar.apply,
        "affine_invariant_gevd": AffineInvariantStdScalar.apply,  # TODO: GEVD version of StdScalar
        "log_euclidean": LogEuclideanStdScalar.apply,
        "arithmetic": LeftKullbackLeiblerStdScalar.apply,
        "harmonic": RightKullbackLeiblerStdScalar.apply,
        "geometric_arithmetic_harmonic": SymmetrizedKullbackLeiblerStdScalar.apply,
    }
    # It must be added to _MEAN_LINKED_TABLES to check that it's ok
    _MEAN_LINKED_TABLES = BatchNormSPDMean._MEAN_LINKED_TABLES + ("_STD_FUNS",)

    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        norm_only: bool = False,
        norm_and_bias_mode: str = "whitening",
        norm_strategy: str = "classical",
        minibatch_mode: str = "constant",
        minibatch_momentum: float = 0.01,
        minibatch_maxstep: int = 100,
        parametrization: str = "softplus",
        parametrization_mode: str = "static",
        n_steps_ref_update: int = 100,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean.
        Both the SPD mean and scalar variance are normalized

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        momentum : float, optional
            Momentum for running mean update.
            Default is 0.01

        norm_only : bool, optional
            Whether to apply some bias or only perform normalization.
            Default is False

        norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening", "parallel_transport", "norm_only"

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

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
        """
        super().__init__(
            n_features=n_features,
            mean_type=mean_type,
            mean_options=mean_options,
            momentum=momentum,
            norm_only=norm_only,
            norm_and_bias_mode=norm_and_bias_mode,
            norm_strategy=norm_strategy,
            minibatch_mode=minibatch_mode,
            minibatch_momentum=minibatch_momentum,
            minibatch_maxstep=minibatch_maxstep,
            parametrization=parametrization,
            parametrization_mode=parametrization_mode,
            n_steps_ref_update=n_steps_ref_update,
            device=device,
            dtype=dtype,
        )

    def _init_norm_parameters(self) -> None:
        """
        Initialize functions to compute batch parameters, normalization parameters,
        and running parameters
        """
        # init mean parameters from parent class
        super()._init_norm_parameters()
        # init scalar std parameters and functions
        self.std_fun = self._STD_FUNS[self.mean_type]
        self._init_norm_strategy_std()
        # set running std
        self.register_buffer(
            "running_std_scalar",
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )

    def _init_norm_strategy_std(self) -> None:
        """
        Auxiliary function to select norm strategy
        """
        if self.norm_strategy == "classical":
            self.get_norm_std = _identity
        elif self.norm_strategy == "minibatch":
            self.get_norm_std = self._handle_minibatch_std
            self.register_buffer(
                "std_regularizer",
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
            )

    def _init_neutral_bias(self) -> None:
        """
        Set bias parameters as neutral parameters
        """
        # init cov neutral bias from parent class
        super()._init_neutral_bias()
        # scalar variance bias to 1
        self.stdScalarbias = torch.ones(
            (), dtype=self.dtype, device=self.device, requires_grad=False
        )

    def _init_bias_parameters(self) -> None:
        """
        Initialize bias parameters and their parametrization
        """
        # init cov bias from parent class
        super()._init_bias_parameters()
        # scalar variance bias parameter
        self.stdScalarbias = torch.nn.Parameter(
            torch.ones((), dtype=self.dtype, device=self.device)
        )
        register_parametrization(self, "stdScalarbias", ScalarSoftPlusParametrization())

    # No need to change  _init_norm_and_bias, only _norm_and_bias
    def _norm_and_bias(self, data: torch.Tensor, parameters) -> torch.Tensor:
        """
        Apply normalization and add bias

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
        # TODO This won't work because...pas de backprop par rapport au parametre t des functions
        # Il faut changer pour que ça fonctionne...
        data_rescaled = self.geodesic_fun(
            parameters[0], data, self.stdScalarbias / parameters[1]
        )
        return self.transport_fun(data_rescaled, parameters[0], self.Covbias)

    def _handle_minibatch_std(self, std_scalar_batch: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to handle var used for normalization with minibatch strategy
        """
        std_scalar = self.adaptive_std_fun(
            self.std_regularizer, std_scalar_batch, self.minibatch_momentum
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
        """ """
        std_scalar = torch.sqrt(
            (1 - momentum) * running_std_scalar**2 + momentum * std_scalar_batch**2
        )
        return std_scalar

    def _get_norm_parameters(
        self, data: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Get parameters used to normalize data

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
        Update running parameters

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
        Get running parameters for normalization in inference

        Returns
        -------
        running_parameters : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and ()
            Running mean and scalar std
        """
        return (self.running_mean, self.running_std_scalar)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BatchNormSPDMeanScalarVariance(\n"
            f"  n_features={self.n_features},\n"
            f"  mean_type={self.mean_type},\n"
            f"  mean_options={self.mean_options},\n"
            f"  momentum={self.momentum},\n"
            f"  norm_only={self.norm_only},\n"
            f"  norm_and_bias_mode={self.norm_and_bias_mode},\n"
            f"  norm_strategy={self.norm_strategy},\n"
            f"  minibatch_mode={self.minibatch_mode},\n"
            f"  minibatch_momentum={self.minibatch_momentum},\n"
            f"  minibatch_maxstep={self.minibatch_maxstep},\n"
            f"  parametrization={self.parametrization},\n"
            f"  parametrization_mode={self.parametrization_mode},\n"
            f"  n_steps_ref_update={self.n_steps_ref_update},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f")"
        )


class BatchNormSPDMeanMultivariateVariance(BatchNormSPDBase):
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
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean.
        Both the SPD mean and tangent variance are normalized

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "affine_invariant".
            Choices are: "affine_invariant", "log_euclidean",
            "arithmetic", "harmonic", "geometric_arithmetic_harmonic"

        mean_options : dict | None, optional
            Options for the SPD mean computation.
            For affine-invariant mean, one can typically set {'n_iterations': 5}.
            Currently, for others, no options available.
            Default is None

        momentum : float, optional
            Momentum for running mean and variance update.
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

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64
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
            device=device,
            dtype=dtype,
        )
