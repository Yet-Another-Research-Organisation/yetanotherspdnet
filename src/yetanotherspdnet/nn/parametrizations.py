import torch
from torch import nn

from ..functions.spd_linalg import (
    ExpmSymmetric,
    ScaledSoftPlusSymmetric,
    expm_symmetric,
    scaled_softplus_symmetric,
)
from ..functions.stiefel import (
    StiefelProjectionPolar,
    stiefel_projection_polar,
    StiefelProjectionQR,
    stiefel_projection_qr,
)


class ScalarSoftPlusParametrization(nn.Module):
    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Positive definite scalars parametrization using the SoftPlus function
        (rescaled so that f(0) = 1 as compared to default torch function)

        Parameters
        ----------
        scalar : torch.Tensor of shape ()
            Real number
        """
        return torch.log(
            torch.tensor(1.0) + torch.pow(torch.tensor(2.0), scalar)
        ) / torch.log(torch.tensor(2.0))


class SPDSoftPlusParametrization(nn.Module):
    def __init__(self, use_autograd: bool = False):
        """
        SPD parametrization using the SoftPlus map

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.softplusSymmetric = (
            (lambda data: scaled_softplus_symmetric(data)[0])
            if self.use_autograd
            else ScaledSoftPlusSymmetric.apply
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SPDLogEuclideanParametrization layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        data_expm : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices
        """
        return self.softplusSymmetric(data)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"SPDSoftPlusParametrization(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class SPDLogEuclideanParametrization(nn.Module):
    def __init__(self, use_autograd: bool = False):
        """
        SPD parametrization using the log-Euclidean exponential mapping

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.expmSymmetric = (
            (lambda data: expm_symmetric(data)[0])
            if self.use_autograd
            else ExpmSymmetric.apply
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SPDLogEuclideanParametrization layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        data_expm : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices
        """
        return self.expmSymmetric(data)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"SPDLogEuclideanParametrization(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class StiefelProjectionPolarParametrization(nn.Module):
    def __init__(self, use_autograd: bool = False):
        """
        Stiefel parametrization using the projection based on the polar decomposition

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.projectionStiefel = (
            stiefel_projection_polar
            if self.use_autograd
            else StiefelProjectionPolar.apply
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StiefelProjectionParametrization layer

        Parameters
        ----------
        weight : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix

        Returns
        -------
        projected_weight : torch.Tensor (n_in, n_out)
            Orthogonal matrix
        """
        return self.projectionStiefel(weight)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"StiefelProjectionParametrization(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class StiefelProjectionQRParametrization(nn.Module):
    def __init__(self, use_autograd: bool = False):
        """
        Stiefel parametrization using the projection based on the QR decomposition

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.projectionStiefel = (
            stiefel_projection_qr if self.use_autograd else StiefelProjectionQR.apply
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StiefelProjectionParametrization layer

        Parameters
        ----------
        weight : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix

        Returns
        -------
        projected_weight : torch.Tensor (n_in, n_out)
            Orthogonal matrix
        """
        return self.projectionStiefel(weight)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"StiefelProjectionParametrization(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()
