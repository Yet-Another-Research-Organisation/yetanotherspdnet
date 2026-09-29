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
    affine_invariant_geodesic_gevd,
    affine_invariant_mean_gevd,
    affine_invariant_parallel_transport_gevd,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    EuclideanGeodesic,
    HarmonicCurve,
    HarmonicMean,
    LeftKullbackLeiblerStdScalar,
    RightKullbackLeiblerStdScalar,
    arithmetic_mean,
    euclidean_geodesic,
    harmonic_curve,
    harmonic_mean,
    left_kullback_leibler_std_scalar,
    right_kullback_leibler_std_scalar,
)
from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    GeometricArithmeticHarmonicMean,
    GeometricEuclideanHarmonicCurve,
    SymmetrizedKullbackLeiblerStdScalar,
    geometric_arithmetic_harmonic_mean,
    geometric_euclidean_harmonic_curve,
    symmetrized_kullback_leibler_std_scalar,
)
from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    LogEuclideanGeodesic,
    LogEuclideanMean,
    LogEuclideanParalleTransportSPD,
    LogEuclideanStdScalar,
    log_euclidean_geodesic,
    log_euclidean_mean,
    log_euclidean_parallel_transport_SPD,
    log_euclidean_std_scalar,
)

from ..functions.spd_geometries.affine_invariant import (
    AffineInvariantGeodesic,
    AffineInvariantMean,
    AffineInvariantParallelTransport,
    AffineInvariantStdScalar,
    affine_invariant_geodesic,
    affine_invariant_mean,
    affine_invariant_parallel_transport,
    affine_invariant_std_scalar,
)
from ..functions.spd_linalg import (
    CongruenceSPD,
    PowmSPD,
    Whitening,
    congruence_SPD,
    powm_SPD,
    whitening,
)
from .parametrizations import (
    SPDAdaptiveParametrization,
    ScalarSoftPlusParametrization,
    SPDParametrization,
)


class BatchNormSPDBase(ABC, nn.Module):
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
        assert self.mean_type in [
            "affine_invariant",
            "affine_invariant_gevd",
            "log_euclidean",
            "arithmetic",
            "harmonic",
            "geometric_arithmetic_harmonic",
        ], (
            f"formula must be in ['affine_invariant', 'affine_invariant_gevd', 'log_euclidean', "
            f"'arithmetic', 'harmonic', 'geometric_arithmetic_harmonic'], "
            f"got {self.mean_type}"
        )
        self.mean_options = mean_options
        self.momentum = momentum
        self.norm_strategy = norm_strategy
        assert self.norm_strategy in [
            "classical",
            "minibatch",
        ], f"formula must be in ['classical', 'minibatch'], got {self.norm_strategy}"
        self.minibatch_mode = minibatch_mode
        if norm_strategy == "minibatch":
            assert self.minibatch_mode in [
                "constant",
                "decay",
                "growth",
            ], (
                f"formula must be in ['constant', 'decay', 'growth'], got {self.minibatch_mode}"
            )
        self.minibatch_momentum = minibatch_momentum
        self.minibatch_maxstep = minibatch_maxstep
        self.parametrization = parametrization
        assert self.parametrization in [
            "softplus",
            "exp",
        ], f"parametrization must be in ['softplus', 'exp'], got {self.parametrization}"
        self.parametrization_mode = parametrization_mode
        assert self.parametrization_mode in ["static", "dynamic"], (
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
        """
        Initialize functions to compute batch parameters, normalization parameters,
        and running parameters
        """
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
        """
        Initialize bias parameters and their parametrization
        """
        pass

    @abstractmethod
    def _init_norm_and_bias(self) -> None:
        """
        Initialize normalization and bias functions
        """
        pass

    def _init_mean(self) -> None:
        """
        Auxiliary function to select mean function
        """
        if self.mean_type == "affine_invariant":
            if self.mean_options is not None and "n_iterations" in self.mean_options:
                if self.use_autograd:
                    self.mean_fun = partial(
                        affine_invariant_mean,
                        n_iterations=self.mean_options["n_iterations"],
                    )
                else:
                    self.mean_fun = partial(
                        AffineInvariantMean,
                        n_iterations=self.mean_options["n_iterations"],
                    )
            else:
                self.mean_fun = (
                    affine_invariant_mean if self.use_autograd else AffineInvariantMean
                )
        elif self.mean_type == "affine_invariant_gevd":
            if self.mean_options is not None and "n_iterations" in self.mean_options:
                if self.use_autograd:
                    self.mean_fun = partial(
                        affine_invariant_mean_gevd,
                        n_iterations=self.mean_options["n_iterations"],
                    )
                else:
                    self.mean_fun = partial(
                        AffineInvariantMeanGEVD,
                        n_iterations=self.mean_options["n_iterations"],
                    )
            else:
                self.mean_fun = (
                    affine_invariant_mean_gevd
                    if self.use_autograd
                    else AffineInvariantMeanGEVD
                )
        elif self.mean_type == "log_euclidean":
            self.mean_fun = (
                log_euclidean_mean if self.use_autograd else LogEuclideanMean
            )
        elif self.mean_type == "arithmetic":
            self.mean_fun = (
                arithmetic_mean if self.use_autograd else ArithmeticMean.apply
            )
        elif self.mean_type == "harmonic":
            self.mean_fun = harmonic_mean if self.use_autograd else HarmonicMean.apply
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.mean_fun = (
                geometric_arithmetic_harmonic_mean
                if self.use_autograd
                else GeometricArithmeticHarmonicMean
            )

    def _init_adaptive_mean_fun(self) -> None:
        """
        Auxiliary function to select adaptive mean update function
        """
        if self.mean_type == "affine_invariant":
            self.adaptive_mean_fun = affine_invariant_geodesic
        elif self.mean_type == "affine_invariant_gevd":
            self.adaptive_mean_fun = lambda x, y, t: affine_invariant_geodesic_gevd(
                x, y, t
            )[0]
        elif self.mean_type == "log_euclidean":
            self.adaptive_mean_fun = log_euclidean_geodesic
        elif self.mean_type == "arithmetic":
            self.adaptive_mean_fun = euclidean_geodesic
        elif self.mean_type == "harmonic":
            self.adaptive_mean_fun = harmonic_curve
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.adaptive_mean_fun = geometric_euclidean_harmonic_curve

    def _init_norm_strategy_mean(self) -> None:
        """
        Auxiliary function to select normalization strategy
        """
        if self.norm_strategy == "classical":
            self.get_norm_mean = lambda mean: mean
        elif self.norm_strategy == "minibatch":
            self.get_norm_mean = self._handle_minibatch_mean
            self.register_buffer(
                "mean_regularizer",
                torch.eye(self.n_features, dtype=self.dtype, device=self.device),
            )
            if self.mean_type == "affine_invariant":
                self.regularize_mean_fun = (
                    affine_invariant_geodesic
                    if self.use_autograd
                    else AffineInvariantGeodesic.apply
                )
            elif self.mean_type == "affine_invariant_gevd":
                self.regularize_mean_fun = (
                    (lambda x, y, t: affine_invariant_geodesic_gevd(x, y, t)[0])
                    if self.use_autograd
                    else AffineInvariantGeodesicGEVD.apply
                )
            elif self.mean_type == "log_euclidean":
                self.regularize_mean_fun = (
                    log_euclidean_geodesic
                    if self.use_autograd
                    else LogEuclideanGeodesic
                )
            elif self.mean_type == "arithmetic":
                self.regularize_mean_fun = (
                    euclidean_geodesic if self.use_autograd else EuclideanGeodesic.apply
                )
            elif self.mean_type == "harmonic":
                self.regularize_mean_fun = (
                    harmonic_curve if self.use_autograd else HarmonicCurve.apply
                )
            elif self.mean_type == "geometric_arithmetic_harmonic":
                self.regularize_mean_fun = (
                    geometric_euclidean_harmonic_curve
                    if self.use_autograd
                    else GeometricEuclideanHarmonicCurve
                )
            self._init_minibatch_mode()

    def _init_minibatch_mode(self) -> None:
        """
        Auxiliary function to deal with initialization of minibatch mode
        """
        if self.minibatch_mode == "constant":
            self.get_minibatch_momentum = lambda: self.minibatch_momentum
        elif self.minibatch_mode == "decay":
            self.get_minibatch_momentum = self._minibatch_momentum_decay
        elif self.minibatch_mode == "growth":
            self.get_minibatch_momentum = self._minibatch_momentum_growth

    def _minibatch_momentum_decay(self) -> float:
        """
        Function to compute the minibatch momentum with minibatch_mode == "decay"

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
        Function to compute the minibatch momentum for minibatch_mode == "growth"

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
        mean = self.regularize_mean_fun(
            self.mean_regularizer, mean_batch, self.get_minibatch_momentum()
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
        pass

    @abstractmethod
    def _update_running_parameters(self, parameters_batch) -> None:
        """
        Update running parameters

        Parameters
        ----------
        parameters_batch
            Parameters of the batch of data
        """
        pass

    @abstractmethod
    def _get_running_parameters(self) -> Any:
        """
        Get running parameters for normalization in inference

        Returns
        -------
        running_parameters
            Running parameters
        """
        pass

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
        pass

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
        pass

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
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class BatchNormSPDMean(BatchNormSPDBase):
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

        norm_and_bias_mode : str, optional
            How normalization and bias is applied. Default is "whitening"
            Choices are: "whitening", "parallel_transport"
        #TODO: Add "norm_only" mode

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
        self.norm_and_bias_mode = norm_and_bias_mode
        assert self.norm_and_bias_mode in [
            "whitening",
            "parallel_transport",
        ], (
            f"formula must be in ['whitening', 'parallel_transport'], got {self.norm_and_bias_mode}"
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
            self.normalize_mean = whitening if self.use_autograd else Whitening.apply
            self.add_bias_mean = (
                congruence_SPD if self.use_autograd else CongruenceSPD.apply
            )
            self._norm_and_bias_fun = self._norm_and_bias_whitening
        elif self.norm_and_bias_mode == "parallel_transport":
            if self.mean_type == "affine_invariant":
                self.parallel_transport = (
                    affine_invariant_parallel_transport
                    if self.use_autograd
                    else AffineInvariantParallelTransport
                )
            elif self.mean_type == "affine_invariant_gevd":
                self.parallel_transport = (
                    affine_invariant_parallel_transport_gevd
                    if self.use_autograd
                    else AffineInvariantParallelTransportGEVD
                )
            elif self.mean_type == "log_euclidean":
                self.parallel_transport = (
                    log_euclidean_parallel_transport_SPD
                    if self.use_autograd
                    else LogEuclideanParalleTransportSPD
                )
            elif (
                self.mean_type == "geometric_arithmetic_harmonic"
                or self.mean_type == "arithmetic"
                or self.mean_type == "harmonic"
            ):
                self.parallel_transport = (
                    affine_invariant_parallel_transport_gevd
                    if self.use_autograd
                    else AffineInvariantParallelTransportGEVD
                )
            self._norm_and_bias_fun = self._norm_and_bias_transport

    def _norm_and_bias_whitening(
        self, data: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
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
        return self.add_bias_mean(self.normalize_mean(data, parameters), self.Covbias)

    def _norm_and_bias_transport(
        self, data: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
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
        return self.parallel_transport(data, parameters, self.Covbias)

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
        return self._norm_and_bias_fun(data, parameters)

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
        self.running_mean = self.adaptive_mean_fun(
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
        )

    def _init_norm_parameters(self) -> None:
        """
        Initialize functions to compute batch parameters, normalization parameters,
        and running parameters
        """
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

    def _init_norm_and_bias(self) -> None:
        """
        Initialize normalization and bias functions
        """
        # init parent class norm and bias functions
        super()._init_norm_and_bias()
        # normalize and add bias scalar variance function
        self.norm_and_bias_var = (
            (lambda data, exponent: powm_SPD(data, exponent)[0])
            if self.use_autograd
            else PowmSPD.apply
        )

    def _init_std(self) -> None:
        """
        Auxiliray function to select std function
        """
        if self.mean_type == "affine_invariant":
            self.std_fun = (
                affine_invariant_std_scalar
                if self.use_autograd
                else AffineInvariantStdScalar.apply
            )
        elif self.mean_type == "log_euclidean":
            self.std_fun = (
                log_euclidean_std_scalar
                if self.use_autograd
                else LogEuclideanStdScalar.apply
            )
        elif self.mean_type == "arithmetic":
            self.std_fun = (
                left_kullback_leibler_std_scalar
                if self.use_autograd
                else LeftKullbackLeiblerStdScalar.apply
            )
        elif self.mean_type == "harmonic":
            self.std_fun = (
                right_kullback_leibler_std_scalar
                if self.use_autograd
                else RightKullbackLeiblerStdScalar.apply
            )
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.std_fun = (
                symmetrized_kullback_leibler_std_scalar
                if self.use_autograd
                else SymmetrizedKullbackLeiblerStdScalar.apply
            )

    def _init_norm_strategy_std(self) -> None:
        """
        Auxiliary function to select norm strategy
        """
        if self.norm_strategy == "classical":
            self.get_norm_std = lambda std: std
        elif self.norm_strategy == "minibatch":
            self.get_norm_std = self._handle_minibatch_std
            self.register_buffer(
                "std_regularizer",
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
            )

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

    def _norm_and_bias(
        self, data: torch.Tensor, parameters: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply normalization and add bias

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
        data_std_transf = self.norm_and_bias_var(
            data_normalized, self.stdScalarbias / parameters[1]
        )
        data_transformed = self.add_bias_mean(data_std_transf, self.Covbias)
        return data_transformed

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
        use_autograd: bool = False,
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
        )
