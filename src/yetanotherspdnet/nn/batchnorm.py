"""SPD batch normalization layers with configurable Riemannian geometry."""

import warnings
from abc import ABC, abstractmethod
from functools import partial
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
from yetanotherspdnet.functions.spd_geometries.bures_wasserstein import (
    BuresWassersteinMean,
    BuresWassersteinStdScalar,
    bures_wasserstein_bias,
    bures_wasserstein_center,
    bures_wasserstein_geodesic,
    bures_wasserstein_mean,
    bures_wasserstein_scale,
    bures_wasserstein_std_scalar,
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
    AdaptiveGeometricArithmeticHarmonicGeodesic,
    AdaptiveGeometricArithmeticHarmonicMean,
    GeometricArithmeticHarmonicMean,
    GeometricEuclideanHarmonicCurve,
    SymmetrizedKullbackLeiblerStdScalar,
    adaptive_geometric_arithmetic_harmonic_geodesic,
    adaptive_geometric_arithmetic_harmonic_mean,
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
    InvSqrtmSPD,
    PowmSPD,
    SqrtmSPD,
    Whitening,
    congruence_SPD,
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


class BatchNormSPDBase(ABC, nn.Module):
    """
    Common machinery of the SPD batch normalization layers.

    Selects the mean (and its running/regularization updates) from
    ``mean_type`` and runs the training/evaluation logic in :meth:`forward`;
    subclasses define the normalization parameters, the bias and how they are
    applied (``_get_norm_parameters``, ``_update_running_parameters``,
    ``_get_running_parameters``, ``_norm_and_bias``).
    """

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
        assert self.mean_type in [
            "affine_invariant",
            "affine_invariant_gevd",
            "log_euclidean",
            "arithmetic",
            "harmonic",
            "geometric_arithmetic_harmonic",
            "adaptive_geometric_arithmetic_harmonic",
            "bures_wasserstein",
        ], (
            f"formula must be in ['affine_invariant', 'affine_invariant_gevd', "
            f"'log_euclidean', 'arithmetic', 'harmonic', "
            f"'geometric_arithmetic_harmonic', "
            f"'adaptive_geometric_arithmetic_harmonic', 'bures_wasserstein'], "
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
        elif self.mean_type == "adaptive_geometric_arithmetic_harmonic":
            # Register learnable parameter t for adaptive GAH mean
            # Use sigmoid parametrization to constrain t in [0, 1]
            t_init = (self.mean_options or {}).get("t_init", 0.5)
            if not 0.0 <= t_init <= 1.0:
                raise ValueError(f"t_init must lie in [0, 1], got {t_init}")
            self.t_gah = torch.nn.Parameter(
                torch.tensor(t_init, dtype=self.dtype, device=self.device)
            )
            register_parametrization(self, "t_gah", ScalarSigmoidParametrization())
            if self.use_autograd:
                self.mean_fun = lambda data: (
                    adaptive_geometric_arithmetic_harmonic_mean(data, self.t_gah)
                )
            else:
                self.mean_fun = lambda data: AdaptiveGeometricArithmeticHarmonicMean(
                    data, self.t_gah
                )
        elif self.mean_type == "bures_wasserstein":
            n_iter = 1
            if self.mean_options is not None and "n_iterations" in self.mean_options:
                n_iter = self.mean_options["n_iterations"]
            self.mean_fun = (
                partial(bures_wasserstein_mean, n_iterations=n_iter)
                if self.use_autograd
                else partial(BuresWassersteinMean, n_iterations=n_iter)
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
        elif self.mean_type == "adaptive_geometric_arithmetic_harmonic":
            # Use the adaptive geodesic with learnable t for running mean update
            if self.use_autograd:
                self.adaptive_mean_fun = lambda p1, p2, t: (
                    adaptive_geometric_arithmetic_harmonic_geodesic(p1, p2, t)
                )
            else:
                self.adaptive_mean_fun = lambda p1, p2, t: (
                    AdaptiveGeometricArithmeticHarmonicGeodesic.apply(
                        p1, p2, torch.tensor(t, dtype=self.dtype, device=self.device)
                    )
                )
        elif self.mean_type == "bures_wasserstein":
            self.adaptive_mean_fun = bures_wasserstein_geodesic

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
            elif self.mean_type == "adaptive_geometric_arithmetic_harmonic":
                if self.use_autograd:
                    self.regularize_mean_fun = (
                        adaptive_geometric_arithmetic_harmonic_geodesic
                    )
                else:
                    self.regularize_mean_fun = lambda p1, p2, t: (
                        AdaptiveGeometricArithmeticHarmonicGeodesic.apply(
                            p1,
                            p2,
                            torch.tensor(t, dtype=self.dtype, device=self.device),
                        )
                    )
            elif self.mean_type == "bures_wasserstein":
                self.regularize_mean_fun = bures_wasserstein_geodesic
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
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
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
            if self.use_autograd and self.mean_type in (
                "affine_invariant",
                "log_euclidean",
            ):
                raise ValueError(
                    "norm_and_bias_mode='parallel_transport' with use_autograd=True "
                    f"is not supported for mean_type={self.mean_type!r}: the "
                    "transport takes square roots or logarithms of the bias, which "
                    "starts at the identity, where autograd through "
                    "torch.linalg.eigh gives NaN gradients (repeated eigenvalues). "
                    "Use use_autograd=False (hand-written backward)."
                )
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
            elif self.mean_type in (
                "arithmetic",
                "harmonic",
                "geometric_arithmetic_harmonic",
                "adaptive_geometric_arithmetic_harmonic",
            ):
                self.parallel_transport = (
                    affine_invariant_parallel_transport_gevd
                    if self.use_autograd
                    else AffineInvariantParallelTransportGEVD
                )
            else:
                raise ValueError(
                    f"norm_and_bias_mode='parallel_transport' is not available "
                    f"for mean_type={self.mean_type!r}"
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
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
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
    r"""
    Riemannian batch normalization of SPD matrices: mean, scalar dispersion, bias.

    Extends :class:`BatchNormSPDMean` with a scalar dispersion: centred matrices
    are rescaled along geodesics from the identity by
    :math:`s / \sigma`, where :math:`\sigma` is the batch (or running) standard
    deviation and :math:`s` a learnable positive scalar. With
    ``mean_type="bures_wasserstein"`` the layer implements GBWBN (generalized
    Bures-Wasserstein batch normalization) with learnable pre/post transforms.

    """

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

        bw_theta : float, optional
            Power deformation :math:`\theta` of the generalized Bures-Wasserstein
            metric (GBWBN, Wang et al. 2025): data are mapped by
            :math:`X \\mapsto M^{-1/2} X^\theta M^{-1/2}` and the variance is
            measured with the deformed metric (divided by :math:`\theta^2`).
            Only used when mean_type="bures_wasserstein". theta=1.0 is the
            plain BW metric; 0.5 is the best value of the paper's ablation
            (HDM05: 69.1% vs 62.0% for theta=1) and the reference code's default.
            Default is 0.5

        bw_batch_stats_grad : bool, optional
            GBWBN only. If True (as in the paper), gradients flow through the
            batch mean and variance; if False (as in the reference code), they
            are computed without gradient, as constants.
            Default is True

        device: torch.device, optional
            Device on which to store the parameters.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer.
            Default is torch.float64

        Attributes
        ----------
        Covbias : torch.Tensor of shape (n_features, n_features)
            Learnable SPD bias.
        stdScalarbias : torch.Tensor
            Learnable positive scalar scale :math:`s`.
        running_mean : torch.Tensor of shape (n_features, n_features)
            Buffer: running SPD mean.
        running_std_scalar : torch.Tensor
            Buffer: running scalar standard deviation.
        bw_M, bw_G : torch.Tensor of shape (n_features, n_features)
            GBWBN only: learnable SPD pre-transform :math:`M` and bias
            :math:`G`, applied in the transformed space as
            :math:`\\hat G = M^{-1/2} G^\theta M^{-1/2}`.
        bw_theta : float
            GBWBN only: power of the pre-transform :math:`X \\mapsto X^\theta`.
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

        # Bures-Wasserstein specific parameters
        if self.mean_type == "bures_wasserstein":
            self.bw_theta = bw_theta
            self.bw_batch_stats_grad = bw_batch_stats_grad
            # M: learnable SPD metric (n x n)
            self.bw_M = torch.nn.Parameter(
                torch.eye(n_features, dtype=self.dtype, device=self.device)
            )
            register_parametrization(
                self, "bw_M", SPDParametrization(mapping=self.parametrization)
            )
            # G: learnable bias on the input manifold (n x n), mapped to the
            # transformed space as M^{-1/2} G^theta M^{-1/2} like the data
            self.bw_G = torch.nn.Parameter(
                torch.eye(n_features, dtype=self.dtype, device=self.device)
            )
            register_parametrization(
                self, "bw_G", SPDParametrization(mapping=self.parametrization)
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
        elif self.mean_type == "bures_wasserstein":
            self.std_fun = (
                bures_wasserstein_std_scalar
                if self.use_autograd
                else BuresWassersteinStdScalar.apply
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of SPD matrices (GBWBN pipeline for
        ``mean_type="bures_wasserstein"``, otherwise the generic one)

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed (normalized then biased) SPD matrices
        """
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
        M = self.bw_M

        # Pre-transform: X_hat = M^{-1/2} X^theta M^{-1/2}
        # bw_M is initialized at the identity: with use_autograd=False the
        # manual (Daleckii-Krein) backwards keep its gradient finite, whereas
        # autograd through torch.linalg.eigh is undefined for repeated
        # eigenvalues
        if self.use_autograd:
            inv_sqrtm = lambda x: inv_sqrtm_SPD(x)[0]  # noqa: E731
            sqrtm = lambda x: sqrtm_SPD(x)[0]  # noqa: E731
            powm = lambda x, p: powm_SPD(x, p)[0]  # noqa: E731
        else:
            inv_sqrtm, sqrtm = InvSqrtmSPD.apply, SqrtmSPD.apply

            def powm(x: torch.Tensor, p: float) -> torch.Tensor:
                exponent = torch.tensor(p, dtype=x.dtype, device=x.device)
                return PowmSPD.apply(x, exponent)

        M_isqrt = inv_sqrtm(M)
        data_pow = powm(data, theta) if theta != 1.0 else data
        data_hat = M_isqrt @ data_pow @ M_isqrt
        G = self.bw_G
        G_hat = M_isqrt @ (powm(G, theta) if theta != 1.0 else G) @ M_isqrt

        if _use_batch_statistics(self, data):
            with torch.set_grad_enabled(
                self.bw_batch_stats_grad and torch.is_grad_enabled()
            ):
                mean_batch = self.mean_fun(data_hat)
                mean = self.get_norm_mean(mean_batch)
                std_batch = self.std_fun(data_hat, mean) / theta
                std = self.get_norm_std(std_batch)
            with torch.no_grad():
                self.running_mean = self.adaptive_mean_fun(
                    self.running_mean.to(data.device), mean_batch, self.momentum
                )
                self.running_std_scalar = self.adaptive_std_fun(
                    self.running_std_scalar.to(data.device), std_batch, self.momentum
                )
            self.training_step = self.training_step + 1
            if self.is_dynamic:
                self.current_ref_step += 1
        else:
            mean = self.running_mean.to(data.device)
            std = self.running_std_scalar.to(data.device)

        # BW center, scale, bias
        centered = bures_wasserstein_center(data_hat, mean)
        scaled = bures_wasserstein_scale(centered, std**2, self.stdScalarbias)
        biased = bures_wasserstein_bias(scaled, G_hat)

        # Post-transform: output = (M^{1/2} X_tilde M^{1/2})^{1/theta}
        M_sqrt = sqrtm(M)
        result = M_sqrt @ biased @ M_sqrt
        return powm(result, 1.0 / theta) if theta != 1.0 else result

    def __repr__(self) -> str:
        """
        Representation of the layer

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
