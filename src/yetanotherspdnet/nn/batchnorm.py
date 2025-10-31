import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from ..functions.spd_geometries import (
    ArithmeticMean,
    GeometricArithmeticHarmonicMean,
    GeometricMean,
    HarmonicMean,
    LogEuclideanMean,
    adaptive_update_arithmetic,
    adaptive_update_geometric,
    adaptive_update_geometric_arithmetic_harmonic,
    adaptive_update_harmonic,
    adaptive_update_logEuclidean,
    arithmetic_mean,
    geometric_arithmetic_harmonic_mean,
    geometric_mean,
    harmonic_mean,
    logEuclidean_mean,
)
from ..functions.spd_linalg import (
    CongruenceSPD,
    Whitening,
    congruence_SPD,
    whitening,
)
from .base import SPDLogEuclideanParametrization


class BatchNormSPDMean(nn.Module):
    def __init__(
        self,
        n_features: int,
        mean_type: str = "geometric_arithmetic_harmonic",
        momentum: float = 0.01,
        use_autograd: bool = False,
        dtype: torch.dtype = torch.float64,
        max_iter: int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Batch normalization layer for SPDnet relying on a SPD mean

        Parameters
        ----------
        n_features : int
            Number of features

        mean_type : str, optional
            Choice of SPD mean. Default is "geometric_arithmetic_harmonic".
            Choices are : "arithmetic", "harmonic", "geometric_arithmetic_harmonic",
            "logEuclidean" and "geometric"

        momentum : float, optional
            Momentum for running mean update. Default is 0.01.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64.

        max_iter: int
            Number of maximum iterations for geometric case

        device: torch.device
            Device on which to store the parameters
        """
        super().__init__()

        assert mean_type in [
            "arithmetic",
            "harmonic",
            "geometric_arithmetic_harmonic",
            "logEuclidean",
            "geometric",
        ], (
            f"formula must be in ['arithmetic', 'harmonic', "
            f"'geometric_arithmetic_harmonic', 'logEuclidean', 'geometric'],"
            f"got {mean_type}"
        )

        self.n_features = n_features
        self.mean_type = mean_type
        self.momentum = momentum
        self.use_autograd = use_autograd
        self.dtype = dtype
        self.max_iter = max_iter
        self.device = device

        # SPD bias parameter
        self.Covbias = torch.nn.Parameter(
            torch.zeros(n_features, n_features, dtype=self.dtype, device=self.device)
        )
        register_parametrization(self, "Covbias", SPDLogEuclideanParametrization())

        # Deal with selected SPD mean method
        ## running parameters and function to access running mean
        if self.mean_type == "geometric_arithmetic_harmonic":
            self.running_param = (
                torch.eye(n_features, dtype=self.dtype, device=self.device),
                torch.eye(n_features, dtype=self.dtype, device=self.device),
                torch.eye(n_features, dtype=self.dtype, device=self.device),
            )
            self.get_running_mean = self._get_mean_gar
        else:
            self.running_param = torch.eye(
                n_features, dtype=self.dtype, device=self.device
            )
            self.get_running_mean = lambda x: x
        ## mean and adaptive update functions
        if self.mean_type == "arithmetic":
            self.mean_fun = (
                arithmetic_mean if self.use_autograd else ArithmeticMean.apply
            )
            self.adaptive_fun = adaptive_update_arithmetic
        elif self.mean_type == "harmonic":
            self.mean_fun = harmonic_mean if self.use_autograd else HarmonicMean.apply
            self.adaptive_fun = adaptive_update_harmonic
        elif self.mean_type == "geometric_arithmetic_harmonic":
            mean_fun = (
                geometric_arithmetic_harmonic_mean
                if self.use_autograd
                else GeometricArithmeticHarmonicMean
            )
            self.mean_fun = lambda x: mean_fun(x, True)
            self.adaptive_fun = self._aux_adaptive_gar
        elif self.mean_type == "logEuclidean":
            self.mean_fun = logEuclidean_mean if self.use_autograd else LogEuclideanMean
            self.adaptive_fun = adaptive_update_logEuclidean
        elif self.mean_type == "geometric":
            _geometric_mean = lambda x: geometric_mean(x, n_iterations=self.max_iter)
            _GeometricMean = lambda x: GeometricMean(x, n_iterations=self.max_iter)
            self.mean_fun = _geometric_mean if self.use_autograd else _GeometricMean
            self.adaptive_fun = adaptive_update_geometric

        # Normalize and add bias functions
        self.normalize = whitening if self.use_autograd else Whitening.apply
        self.add_bias = congruence_SPD if self.use_autograd else CongruenceSPD.apply

    def _get_mean_gar(
        self, mean_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Auxiliary function to get actual mean in the case of geometric
        mean of arithmetic and harmonic means

        Parameters
        ----------
        mean_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of geometric mean of arithmetic and harmonic means,
            and corresponding arithmetic and harmonic means

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Geometric mean of arithmetic and harmonic means
        """
        return mean_param[0]

    def _aux_adaptive_gar(
        self,
        running_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_param: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        momentum: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auxiliary function for the adaptive update of the geometric
        mean of arithmetic and harmonic means

        Parameters
        ----------
        running_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of running geometric mean of arithmetic and harmonic means,
            and corresponding arithmetic and harmonic means

        batch_param : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of batch geometric mean of arithmetic and harmonic means,
            and corresponding arithmetic and harmonic means

        momentum : float
            Momentum for running mean update

        Returns
        -------
        running_param_updated : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated tuple of running geometric mean of arithmetic and
            harmonic means, and corresponding arithmetic and harmonic means
        """
        return adaptive_update_geometric_arithmetic_harmonic(
            running_param[1], running_param[2], batch_param[1], batch_param[2], momentum
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchNorm layer

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
            mean_param = self.mean_fun(data)
            with torch.no_grad():
                self.running_param = self.adaptive_fun(
                    self.running_param, mean_param, self.momentum
                )
        else:
            # training over, use overall mean learnt on all batches
            mean_param = self.running_param

        # get mean from mean parameters
        mean = self.get_running_mean(mean_param)

        # Normalize data and add bias
        data_normalized = self.normalize(data, mean)
        data_transformed = self.add_bias(data_normalized, self.Covbias)

        return data_transformed

    def __repr__(self) -> str:
        """Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BatchNormSPDMean(n_features={self.n_features}, "
            f"mean_type={self.mean_type}, momentum={self.momentum}, "
            f"use_autograd={self.use_autograd}, dtype={self.dtype}, "
            f"max_iter={self.max_iter})"
        )

    def __str__(self) -> str:
        """String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()
