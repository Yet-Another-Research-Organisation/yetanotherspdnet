import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from functools import partial

from ..functions.spd_linalg import (
    CongruenceSPD,
    PowmSPD,
    Whitening,
    congruence_SPD,
    powm_SPD,
    whitening,
)

from ..functions.spd_geometries.affine_invariant import (
    AffineInvariantGeodesic,
    AffineInvariantMean,
    AffineInvariantStdScalar,
    affine_invariant_mean,
    affine_invariant_geodesic,
    affine_invariant_std_scalar,
)

from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    LogEuclideanGeodesic,
    LogEuclideanMean,
    LogEuclideanStdScalar,
    log_euclidean_geodesic,
    log_euclidean_mean,
    log_euclidean_std_scalar,
)

from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    EuclideanGeodesic,
    HarmonicCurve,
    HarmonicMean,
    LeftKullbackLeiblerStdScalar,
    RightKullbackLeiblerStdScalar,
    arithmetic_mean,
    harmonic_mean,
    euclidean_geodesic,
    harmonic_curve,
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

from .parametrizations import (
    SPDAdaptiveParametrization,
    SPDParametrization,
    ScalarSoftPlusParametrization,
)


class BatchNormSPDMean(nn.Module):
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
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean.
        Here, only the SPD mean is normalized

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

        self.n_features = n_features
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype

        # deal with mean_type and adaptive_mean_type
        self.mean_type = mean_type
        self.mean_options = mean_options
        self.momentum = momentum
        self._init_mean()
        self._init_adaptive_mean_fun()

        self.norm_strategy = norm_strategy
        self.minibatch_mode = minibatch_mode
        self.minibatch_momentum = minibatch_momentum
        self.minibatch_maxstep = minibatch_maxstep
        self._init_norm_strategy()

        self.training_step = 0

        # bias parameter
        self.parametrization = parametrization
        assert self.parametrization in ["softplus", "exp"], (
            f"formula must be in ['softplus', 'exp'], got {self.parametrization}"
        )
        self.Covbias = torch.nn.Parameter(
            torch.eye(n_features, dtype=self.dtype, device=self.device)
        )
        register_parametrization(
            self, "Covbias", SPDParametrization(mapping=self.parametrization)
        )

        # normalize and add bias functions
        self.normalize_mean = whitening if self.use_autograd else Whitening.apply
        self.add_bias_mean = (
            congruence_SPD if self.use_autograd else CongruenceSPD.apply
        )
        # set running mean
        self.running_mean = torch.eye(
            self.n_features, dtype=self.dtype, device=self.device
        )
        self.running_mean.requires_grad = False

    def _init_mean(self) -> None:
        """
        Auxiliary function to select mean function
        """
        assert self.mean_type in [
            "affine_invariant",
            "log_euclidean",
            "arithmetic",
            "harmonic",
            "geometric_arithmetic_harmonic",
        ], (
            f"formula must be in ['affine_invariant', 'log_euclidean', "
            f"'arithmetic', 'harmonic', 'geometric_arithmetic_harmonic'], "
            f"got {self.mean_type}"
        )

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
        elif self.mean_type == "log_euclidean":
            self.adaptive_mean_fun = log_euclidean_geodesic
        elif self.mean_type == "arithmetic":
            self.adaptive_mean_fun = euclidean_geodesic
        elif self.mean_type == "harmonic":
            self.adaptive_mean_fun = harmonic_curve
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.adaptive_mean_fun = geometric_euclidean_harmonic_curve

    def _init_norm_strategy(self) -> None:
        """
        Auxiliary function to select normalization strategy
        """
        assert self.norm_strategy in [
            "classical",
            "minibatch",
        ], f"formula must be in ['classical', 'minibatch'], got {self.mean_type}"

        if self.norm_strategy == "classical":
            self.get_norm_mean = lambda mean: mean
        elif self.norm_strategy == "minibatch":
            self.get_norm_mean = self._handle_mean_minibatch
            self.mean_regularizer = torch.eye(
                self.n_features, device=self.device, dtype=self.dtype
            )
            self.mean_regularizer.requires_grad = False
            if self.mean_type == "affine_invariant":
                self.regularize_mean_fun = (
                    affine_invariant_geodesic
                    if self.use_autograd
                    else AffineInvariantGeodesic.apply
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
        assert self.minibatch_mode in ["constant", "decay", "growth"], (
            f"formula must be in ['constant', 'decay', 'growth'], got {self.minibatch_mode}"
        )
        if self.minibatch_mode == "constant":
            self.get_minibatch_momentum = lambda: self.minibatch_momentum
        elif self.minibatch_mode == "decay":
            self.get_minibatch_momentum = self.minibatch_momentum_decay
        elif self.minibatch_mode == "growth":
            self.get_minibatch_momentum = self.minibatch_momentum_growth

    def minibatch_momentum_decay(self) -> float:
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

    def minibatch_momentum_growth(self) -> float:
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

    def _handle_mean_minibatch(self, mean_batch: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to handle mean used for normalization with minibatch strategy
        """
        mean = self.regularize_mean_fun(
            self.mean_regularizer, mean_batch, self.get_minibatch_momentum()
        )
        with torch.no_grad():
            self.mean_regularizer = mean.detach()
        return mean

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchNormSPDMean layer

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
            mean_batch = self.mean_fun(data)
            mean = self.get_norm_mean(mean_batch)
            # update running mean
            with torch.no_grad():
                self.running_mean = self.adaptive_mean_fun(
                    self.running_mean, mean_batch, self.momentum
                )
            self.training_step = self.training_step + 1
        else:
            # training over, use overall mean learnt on all batches
            mean = self.running_mean

        # Normalize data and add bias
        data_normalized = self.normalize_mean(data, mean)
        data_transformed = self.add_bias_mean(data_normalized, self.Covbias)

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
            f"BatchNormSPDMean(n_features={self.n_features}, "
            f"mean_type={self.mean_type}, mean_options={self.mean_options}, "
            f"momentum={self.momentum}, "
            f"norm_strategy={self.norm_strategy}, minibatch_mode={self.minibatch_mode}"
            f"minibatch_momentum={self.minibatch_momentum}, "
            f"minibatch_maxstep={self.minibatch_maxstep}"
            f"use_autograd={self.use_autograd}, device={self.device}, "
            f"dtype={self.dtype})"
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


class BatchNormSPDMeanScalarVariance(nn.Module):
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
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean.
        Here, the SPD mean and scalar variance are normalized

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

        self.n_features = n_features
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype

        # deal with mean_type and adaptive_mean_type
        self.mean_type = mean_type
        self.mean_options = mean_options
        self.momentum = momentum
        self._init_mean_and_var()
        self._init_adaptive_mean_fun()

        self.norm_strategy = norm_strategy
        self.minibatch_mode = minibatch_mode
        self.minibatch_momentum = minibatch_momentum
        self.minibatch_maxstep = minibatch_maxstep
        self._init_norm_strategy()

        self.training_step = 0

        # mean covariance bias parameters
        self.parametrization = parametrization
        assert self.parametrization in ["softplus", "exp"], (
            f"formula must be in ['softplus', 'exp'], got {self.parametrization}"
        )
        self.Covbias = torch.nn.Parameter(
            torch.eye(n_features, dtype=self.dtype, device=self.device)
        )
        register_parametrization(
            self, "Covbias", SPDParametrization(mapping=self.parametrization)
        )
        # scalar variance bias parameter
        self.stdScalarbias = torch.nn.Parameter(
            torch.ones((), dtype=self.dtype, device=self.device)
        )
        register_parametrization(self, "stdScalarbias", ScalarSoftPlusParametrization())
        # normalize and add bias functions
        self.normalize_mean = whitening if self.use_autograd else Whitening.apply
        self.add_bias_mean = (
            congruence_SPD if self.use_autograd else CongruenceSPD.apply
        )
        self.norm_and_bias_var = (
            (lambda data, exponent: powm_SPD(data, exponent)[0])
            if self.use_autograd
            else PowmSPD.apply
        )
        # set running mean
        self.running_mean = torch.eye(
            self.n_features, dtype=self.dtype, device=self.device
        )
        self.running_mean.requires_grad = False
        self.running_std_scalar = torch.tensor(
            1.0, dtype=self.dtype, device=self.device
        )
        self.running_std_scalar.requires_grad = False

    def _init_mean_and_var(self) -> None:
        """
        Auxiliary function to select mean and var functions
        """
        assert self.mean_type in [
            "affine_invariant",
            "log_euclidean",
            "arithmetic",
            "harmonic",
            "geometric_arithmetic_harmonic",
        ], (
            f"formula must be in ['affine_invariant', 'log_euclidean', "
            f"'arithmetic', 'harmonic', 'geometric_arithmetic_harmonic'],"
            f"got {self.mean_type}"
        )

        if self.mean_type == "affine_invariant":
            if self.mean_options is not None and "n_iterations" in self.mean_options:
                if self.use_autograd:
                    self.mean_fun = partial(
                        affine_invariant_mean,
                        n_iterations=self.mean_options["n_iterations"],
                    )
                    self.std_fun = affine_invariant_std_scalar
                else:
                    self.mean_fun = partial(
                        AffineInvariantMean,
                        n_iterations=self.mean_options["n_iterations"],
                    )
                    self.std_fun = AffineInvariantStdScalar.apply
            else:
                self.mean_fun = (
                    affine_invariant_mean if self.use_autograd else AffineInvariantMean
                )
                self.std_fun = (
                    affine_invariant_std_scalar
                    if self.use_autograd
                    else AffineInvariantStdScalar.apply
                )
        elif self.mean_type == "log_euclidean":
            self.mean_fun = (
                log_euclidean_mean if self.use_autograd else LogEuclideanMean
            )
            self.std_fun = (
                log_euclidean_std_scalar
                if self.use_autograd
                else LogEuclideanStdScalar.apply
            )
        elif self.mean_type == "arithmetic":
            self.mean_fun = (
                arithmetic_mean if self.use_autograd else ArithmeticMean.apply
            )
            self.std_fun = (
                left_kullback_leibler_std_scalar
                if self.use_autograd
                else LeftKullbackLeiblerStdScalar.apply
            )
        elif self.mean_type == "harmonic":
            self.mean_fun = harmonic_mean if self.use_autograd else HarmonicMean.apply
            self.std_fun = (
                right_kullback_leibler_std_scalar
                if self.use_autograd
                else RightKullbackLeiblerStdScalar.apply
            )
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.mean_fun = (
                geometric_arithmetic_harmonic_mean
                if self.use_autograd
                else GeometricArithmeticHarmonicMean
            )
            self.std_fun = (
                symmetrized_kullback_leibler_std_scalar
                if self.use_autograd
                else SymmetrizedKullbackLeiblerStdScalar.apply
            )

    def _init_adaptive_mean_fun(self) -> None:
        """
        Auxiliary function to select adaptive mean update function
        """
        if self.mean_type == "affine_invariant":
            self.adaptive_mean_fun = affine_invariant_geodesic
        elif self.mean_type == "log_euclidean":
            self.adaptive_mean_fun = log_euclidean_geodesic
        elif self.mean_type == "arithmetic":
            self.adaptive_mean_fun = euclidean_geodesic
        elif self.mean_type == "harmonic":
            self.adaptive_mean_fun = harmonic_curve
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.adaptive_mean_fun = geometric_euclidean_harmonic_curve

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

    def _init_norm_strategy(self) -> None:
        """
        Auxiliary function to select normalization strategy
        """
        assert self.norm_strategy in [
            "classical",
            "minibatch",
        ], f"formula must be in ['classical', 'minibatch'], got {self.mean_type}"

        if self.norm_strategy == "classical":
            self.get_norm_mean = lambda mean: mean
            self.get_norm_std = lambda std: std
        elif self.norm_strategy == "minibatch":
            self.get_norm_mean = self._handle_mean_minibatch
            self.mean_regularizer = torch.eye(
                self.n_features, device=self.device, dtype=self.dtype
            )
            self.mean_regularizer.requires_grad = False
            self.get_norm_std = self._handle_std_minibatch
            self.std_regularizer = torch.ones((), device=self.device, dtype=self.dtype)
            self.std_regularizer.requires_grad = False
            if self.mean_type == "affine_invariant":
                self.regularize_mean_fun = (
                    affine_invariant_geodesic
                    if self.use_autograd
                    else AffineInvariantGeodesic.apply
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
        assert self.minibatch_mode in ["constant", "decay", "growth"], (
            f"formula must be in ['constant', 'decay', 'growth'], got {self.minibatch_mode}"
        )
        if self.minibatch_mode == "constant":
            self.get_minibatch_momentum = lambda: self.minibatch_momentum
        elif self.minibatch_mode == "decay":
            self.get_minibatch_momentum = self.minibatch_momentum_decay
        elif self.minibatch_mode == "growth":
            self.get_minibatch_momentum = self.minibatch_momentum_growth

    def minibatch_momentum_decay(self) -> float:
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

    def minibatch_momentum_growth(self) -> float:
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

    def _handle_mean_minibatch(self, mean_batch: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to handle mean used for normalization with minibatch strategy
        """
        mean = self.regularize_mean_fun(
            self.mean_regularizer, mean_batch, self.minibatch_momentum
        )
        with torch.no_grad():
            self.mean_regularizer = mean.detach()
        return mean

    def _handle_std_minibatch(self, std_scalar_batch: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to handle var used for normalization with minibatch strategy
        """
        std_scalar = self.adaptive_std_fun(
            self.std_regularizer, std_scalar_batch, self.minibatch_momentum
        )
        with torch.no_grad():
            self.std_regularizer = std_scalar.detach()
        return std_scalar

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchNormSPDMeanScalarVariance layer

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
            mean_batch = self.mean_fun(data)
            mean = self.get_norm_mean(mean_batch)
            std_batch = self.std_fun(data, mean)
            std = self.get_norm_std(std_batch)
            # update running mean
            with torch.no_grad():
                self.running_mean = self.adaptive_mean_fun(
                    self.running_mean, mean_batch, self.momentum
                )
                self.running_std_scalar = self.adaptive_std_fun(
                    self.running_std_scalar, std_batch, self.momentum
                )
            self.training_step = self.training_step + 1
        else:
            # training over, use overall mean learnt on all batches
            mean = self.running_mean
            std = self.running_std_scalar

        # Normalize data and add bias
        data_normalized = self.normalize_mean(data, mean)
        data_std_transf = self.norm_and_bias_var(
            data_normalized, self.stdScalarbias / std
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
            f"BatchNormSPDMeanScalarVariance(n_features={self.n_features}, "
            f"mean_type={self.mean_type}, mean_options={self.mean_options}, "
            f"momentum={self.momentum}, "
            f"norm_strategy={self.norm_strategy}, minibatch_mode={self.minibatch_mode}"
            f"minibatch_momentum={self.minibatch_momentum}, "
            f"minibatch_maxstep={self.minibatch_maxstep}"
            f"use_autograd={self.use_autograd}, device={self.device}, "
            f"dtype={self.dtype})"
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
