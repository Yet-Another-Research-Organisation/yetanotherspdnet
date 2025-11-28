import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from functools import partial

from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    GeometricArithmeticHarmonicMean,
    geometric_arithmetic_harmonic_mean,
    geometric_euclidean_harmonic_curve,
)

from ..functions.spd_linalg import (
    CongruenceSPD,
    Whitening,
    congruence_SPD,
    whitening,
)

from ..functions.spd_geometries.affine_invariant import (
    AffineInvariantMean,
    affine_invariant_mean,
    affine_invariant_geodesic,
)

from yetanotherspdnet.functions.spd_geometries.log_euclidean import (
    LogEuclideanMean,
    log_euclidean_geodesic,
    log_euclidean_mean,
)

from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    HarmonicMean,
    arithmetic_mean,
    harmonic_mean,
    euclidean_geodesic,
    harmonic_curve,
)

from .base import SPDLogEuclideanParametrization


class BatchNormSPDMean(nn.Module):
    def __init__(
        self,
        n_features: int,
        mean_type: str = "affine_invariant",
        mean_options: dict | None = None,
        momentum: float = 0.01,
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Batch normalization layer for SPDnet relying on a SPD mean

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

        self.norm_type = "classical"

        # bias parameter
        self.Covbias = torch.nn.Parameter(
            torch.zeros(n_features, n_features, dtype=self.dtype, device=self.device)
        )
        register_parametrization(self, "Covbias", SPDLogEuclideanParametrization())
        # normalize and add bias functions
        self.normalize = whitening if self.use_autograd else Whitening.apply
        self.add_bias = congruence_SPD if self.use_autograd else CongruenceSPD.apply
        # set running mean
        self.running_mean = torch.eye(
            self.n_features, dtype=self.dtype, device=self.device
        )

    def _init_mean(self) -> None:
        """
        Auxiliary function to select mean and adaptive mean functions
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
                else:
                    self.mean_fun = partial(
                        AffineInvariantMean,
                        n_iterations=self.mean_options["n_iterations"],
                    )
            else:
                self.mean_fun = (
                    affine_invariant_mean if self.use_autograd else AffineInvariantMean
                )
            self.adaptive_fun = affine_invariant_geodesic
        elif self.mean_type == "log_euclidean":
            self.mean_fun = (
                log_euclidean_mean if self.use_autograd else LogEuclideanMean
            )
            self.adaptive_fun = log_euclidean_geodesic
        elif self.mean_type == "arithmetic":
            self.mean_fun = (
                arithmetic_mean if self.use_autograd else ArithmeticMean.apply
            )
            self.adaptive_fun = euclidean_geodesic
        elif self.mean_type == "harmonic":
            self.mean_fun = harmonic_mean if self.use_autograd else HarmonicMean.apply
            self.adaptive_fun = harmonic_curve
        elif self.mean_type == "geometric_arithmetic_harmonic":
            self.mean_fun = (
                geometric_arithmetic_harmonic_mean
                if self.use_autograd
                else GeometricArithmeticHarmonicMean
            )
            self.adaptive_fun = geometric_euclidean_harmonic_curve

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
            if self.norm_type == "classical":
                mean = mean_batch
            elif self.norm_type == "minibatch":
                raise ValueError("not implemented yet")
            with torch.no_grad():
                self.running_mean = self.adaptive_fun(
                    self.running_mean, mean_batch, self.momentum
                )
        else:
            # training over, use overall mean learnt on all batches
            mean = self.running_mean

        # Normalize data and add bias
        data_normalized = self.normalize(data, mean)
        data_transformed = self.add_bias(data_normalized, self.Covbias)

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


# class BatchNormSPDMean_old(nn.Module):
#     def __init__(
#         self,
#         n_features: int,
#         mean_type: str = "geometric_arithmetic_harmonic",
#         momentum: float = 0.01,
#         use_autograd: bool = False,
#         dtype: torch.dtype = torch.float64,
#         max_iter: int = 5,
#         device: torch.device = torch.device("cpu"),
#     ) -> None:
#         """
#         Batch normalization layer for SPDnet relying on a SPD mean
#
#         Parameters
#         ----------
#         n_features : int
#             Number of features
#
#         mean_type : str, optional
#             Choice of SPD mean. Default is "geometric_arithmetic_harmonic".
#             Choices are : "arithmetic", "harmonic", "geometric_arithmetic_harmonic",
#             "logEuclidean" and "geometric"
#
#         momentum : float, optional
#             Momentum for running mean update. Default is 0.01.
#
#         use_autograd : bool, optional
#             Use torch autograd for the computation of the gradient rather than
#             the analytical formula. Default is False.
#
#         dtype : torch.dtype, optional
#             Data type of the layer. Default is torch.float64.
#
#         max_iter: int
#             Number of maximum iterations for geometric case
#
#         device: torch.device
#             Device on which to store the parameters
#         """
#         super().__init__()
#
#         assert mean_type in [
#             "arithmetic",
#             "harmonic",
#             "geometric_arithmetic_harmonic",
#             "logEuclidean",
#             "geometric",
#         ], (
#             f"formula must be in ['arithmetic', 'harmonic', "
#             f"'geometric_arithmetic_harmonic', 'logEuclidean', 'geometric'],"
#             f"got {mean_type}"
#         )
#
#         self.n_features = n_features
#         self.mean_type = mean_type
#         self.momentum = momentum
#         self.use_autograd = use_autograd
#         self.dtype = dtype
#         self.max_iter = max_iter
#         self.device = device
#
#         # SPD bias parameter
#         self.Covbias = torch.nn.Parameter(
#             torch.zeros(n_features, n_features, dtype=self.dtype, device=self.device)
#         )
#         register_parametrization(self, "Covbias", SPDLogEuclideanParametrization())
#
#         # Deal with selected SPD mean method
#         ## running parameters and function to access running mean
#         if self.mean_type == "geometric_arithmetic_harmonic":
#             self.running_param = (
#                 torch.eye(n_features, dtype=self.dtype, device=self.device),
#                 torch.eye(n_features, dtype=self.dtype, device=self.device),
#                 torch.eye(n_features, dtype=self.dtype, device=self.device),
#             )
#             self.get_running_mean = self._get_mean_gar
#         else:
#             self.running_param = torch.eye(
#                 n_features, dtype=self.dtype, device=self.device
#             )
#             self.get_running_mean = lambda x: x
#         ## mean and adaptive update functions
#         if self.mean_type == "arithmetic":
#             self.mean_fun = (
#                 arithmetic_mean if self.use_autograd else ArithmeticMean.apply
#             )
#             self.adaptive_fun = adaptive_update_arithmetic
#         elif self.mean_type == "harmonic":
#             self.mean_fun = harmonic_mean if self.use_autograd else HarmonicMean.apply
#             self.adaptive_fun = adaptive_update_harmonic
#         elif self.mean_type == "geometric_arithmetic_harmonic":
#             mean_fun = (
#                 geometric_arithmetic_harmonic_mean
#                 if self.use_autograd
#                 else GeometricArithmeticHarmonicMean
#             )
#             self.mean_fun = lambda x: mean_fun(x, True)
#             self.adaptive_fun = self._aux_adaptive_gar
#         elif self.mean_type == "logEuclidean":
#             self.mean_fun = logEuclidean_mean if self.use_autograd else LogEuclideanMean
#             self.adaptive_fun = adaptive_update_logEuclidean
#         elif self.mean_type == "geometric":
#             _geometric_mean = lambda x: geometric_mean(x, n_iterations=self.max_iter)
#             _GeometricMean = lambda x: GeometricMean(x, n_iterations=self.max_iter)
#             self.mean_fun = _geometric_mean if self.use_autograd else _GeometricMean
#             self.adaptive_fun = adaptive_update_geometric
#
#         # Normalize and add bias functions
#         self.normalize = whitening if self.use_autograd else Whitening.apply
#         self.add_bias = congruence_SPD if self.use_autograd else CongruenceSPD.apply
#
#     def _get_mean_gar(
#         self, mean_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#     ) -> torch.Tensor:
#         """
#         Auxiliary function to get actual mean in the case of geometric
#         mean of arithmetic and harmonic means
#
#         Parameters
#         ----------
#         mean_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#             Tuple of geometric mean of arithmetic and harmonic means,
#             and corresponding arithmetic and harmonic means
#
#         Returns
#         -------
#         mean : torch.Tensor of shape (n_features, n_features)
#             Geometric mean of arithmetic and harmonic means
#         """
#         return mean_param[0]
#
#     def _aux_adaptive_gar(
#         self,
#         running_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
#         batch_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
#         momentum: float,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Auxiliary function for the adaptive update of the geometric
#         mean of arithmetic and harmonic means
#
#         Parameters
#         ----------
#         running_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#             Tuple of running geometric mean of arithmetic and harmonic means,
#             and corresponding arithmetic and harmonic means
#
#         batch_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#             Tuple of batch geometric mean of arithmetic and harmonic means,
#             and corresponding arithmetic and harmonic means
#
#         momentum : float
#             Momentum for running mean update
#
#         Returns
#         -------
#         running_param_updated : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#             Updated tuple of running geometric mean of arithmetic and
#             harmonic means, and corresponding arithmetic and harmonic means
#         """
#         return adaptive_update_geometric_arithmetic_harmonic(
#             running_param[1], running_param[2], batch_param[1], batch_param[2], momentum
#         )
#
#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the BatchNorm layer
#
#         Parameters
#         ----------
#         data : torch.Tensor of shape (..., n_features, n_features)
#             Batch of SPD matrices
#
#         Returns
#         -------
#         data_transformed : torch.Tensor of shape (..., n_features, n_features)
#             Batch of transformed (normalized then biased) SPD matrices
#         """
#         if self.training:
#             mean_param = self.mean_fun(data)
#             with torch.no_grad():
#                 self.running_param = self.adaptive_fun(
#                     self.running_param, mean_param, self.momentum
#                 )
#         else:
#             # training over, use overall mean learnt on all batches
#             mean_param = self.running_param
#
#         # get mean from mean parameters
#         mean = self.get_running_mean(mean_param)
#
#         # Normalize data and add bias
#         data_normalized = self.normalize(data, mean)
#         data_transformed = self.add_bias(data_normalized, self.Covbias)
#
#         return data_transformed
#
#     def __repr__(self) -> str:
#         """Representation of the layer
#
#         Returns
#         -------
#         str
#             Representation of the layer
#         """
#         return (
#             f"BatchNormSPDMean(n_features={self.n_features}, "
#             f"mean_type={self.mean_type}, momentum={self.momentum}, "
#             f"use_autograd={self.use_autograd}, dtype={self.dtype}, "
#             f"max_iter={self.max_iter})"
#         )
#
#     def __str__(self) -> str:
#         """String representation of the layer
#
#         Returns
#         -------
#         str
#             String representation of the layer
#         """
#         return self.__repr__()
