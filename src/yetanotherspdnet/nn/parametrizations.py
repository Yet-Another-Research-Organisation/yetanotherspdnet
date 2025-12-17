import torch
from torch import nn

from yetanotherspdnet.random.stiefel import random_stiefel

from ..functions.spd_linalg import (
    ExpmSymmetric,
    InvSqrtmSPD,
    ScaledSoftPlusSymmetric,
    SqrtmSPD,
    expm_symmetric,
    inv_sqrtm_SPD,
    scaled_softplus_symmetric,
    sqrtm_SPD,
    symmetrize,
)
from ..functions.stiefel import (
    StiefelProjectionPolar,
    StiefelProjectionTangentOrthogonal,
    stiefel_projection_polar,
    StiefelProjectionQR,
    stiefel_projection_qr,
    stiefel_projection_tangent_orthogonal,
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
            1.0 + torch.pow(2.0, scalar)
        ) / torch.log(torch.as_tensor(2.0, dtype=scalar.dtype, device=scalar.device))


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


class SPDAdaptiveParametrization(nn.Module):
    def __init__(
        self,
        n_features: int,
        initial_reference: torch.Tensor | None = None,
        mapping: str = "softplus",
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Adaptive SPD Parametrization

        Parameters
        ----------
        n_features : int
            Number of features

        initial_reference : torch.Tensor | None, optional
            Initial reference point.
            If None, the identity matrix is selected.
            Default is None

        mapping : str, optional
            Mapping to obtain a SPD point from a tangent vector.
            Default is "softplus".
            Choices are: "softplus" and "exp"

        use_autograd : bool | dict, optional
            Use torch autograd for gradient computation. Can be bool for all layers,
            or dict with keys: 'bimap', 'reeig', 'logeig', 'batchnorm', 'vec'.
            Note that Vech module always uses manual gradient.
            Default is False

        device : torch.device, optional
            Device to run model on. Default is torch.device('cpu')

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64
        """
        super().__init__()

        self.n_features = n_features
        self.initial_reference = initial_reference
        assert mapping in ["softplus", "exp"], (
            f"mapping must be in ['softplus', 'exp'], got {mapping}"
        )
        self.mapping = mapping
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype

        # deal with initial_reference
        if self.initial_reference is None:
            # initialize reference_point with identity
            self.register_buffer(
                "reference_point", torch.eye(n_features, device=device, dtype=dtype)
            )
            self.register_buffer(
                "reference_point_sqrtm",
                torch.eye(n_features, device=device, dtype=dtype),
            )
            self.register_buffer(
                "reference_point_inv_sqrtm",
                torch.eye(n_features, device=device, dtype=dtype),
            )
        else:
            assert isinstance(self.initial_reference, torch.Tensor) and (
                self.initial_reference.shape == (n_features, n_features)
            ), (
                "Got incoherent initial_reference, either it is not a torch.Tensor or its shape is not (n_features, n_features)"
            )
            self.register_buffer("reference_point", self.initial_reference.clone())
            self.register_buffer(
                "reference_point_sqrtm", sqrtm_SPD(self.initial_reference.clone())[0]
            )
            self.register_buffer(
                "reference_point_inv_sqrtm",
                inv_sqrtm_SPD(self.initial_reference.clone())[0],
            )

        # Track epoch changes
        self.register_buffer(
            "current_epoch", torch.tensor(0, dtype=torch.long, device=self.device)
        )
        self.register_buffer("last_spd_value", self.reference_point.detach())
        self._epoch_updated = False

        # Deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = (
                (lambda data: scaled_softplus_symmetric(data)[0])
                if self.use_autograd
                else ScaledSoftPlusSymmetric.apply
            )
        elif self.mapping == "exp":
            self.spd_fun = (
                (lambda data: expm_symmetric(data)[0])
                if self.use_autograd
                else ExpmSymmetric.apply
            )

    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the tangent space at reference_point onto the SPD manifold

        Parameters
        ----------
        tangent_vector : torch.Tensor of shape (n_features, n_features)
            Symmetric matrix

        Returns
        -------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
        """
        # ensure tangent_vector symmetric
        tangent_vector = symmetrize(tangent_vector)

        spd_matrix = (
            self.reference_point_sqrtm
            @ self.spd_fun(
                self.reference_point_inv_sqrtm
                @ tangent_vector
                @ self.reference_point_inv_sqrtm
            )
            @ self.reference_point_sqrtm
        )
        # store spd value during training (for reference update)
        if self.training and not self._epoch_updated:
            self.last_spd_value.copy_(spd_matrix.detach())

        return spd_matrix

    # TODO: for right inverse, need to implement Inverse SoftPlus function in spd_linalg


class StiefelProjectionAdaptiveParametrization(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        initial_reference: torch.Tensor | None = None,
        mapping: str = "QR",
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Adaptive Stiefel Parametrization

        Parameters
        ----------
        n_in : int
            Number of rows

        n_out : int
            Number of columns

        initial_reference : torch.Tensor | None, optional
            Initial reference point.
            If None, a random point on Stiefel is generated.
            Default is None

        mapping : str, optional
            Mapping to obtain a point on Stiefel from a tangent vector.
            Default is "QR".
            Choices are: "QR" and "polar"

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
        """
        super().__init__()

        assert n_in >= n_out, f"Must have n_in >= n_out, got n_in={n_in}, n_out={n_out}"
        self.n_in = n_in
        self.n_out = n_out
        self.initial_reference = initial_reference
        assert mapping in ["QR", "polar"], (
            f"mapping must be in ['QR', 'polar'], got {mapping}"
        )
        self.mapping = mapping
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype
        self.generator = generator

        # deal with initial_reference
        if self.initial_reference is None:
            # initialize reference_point with random point on Stiefel
            self.register_buffer(
                "reference_point",
                random_stiefel(
                    self.n_in,
                    self.n_out,
                    n_matrices=1,
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.generator,
                ),
            )
        else:
            assert isinstance(self.initial_reference, torch.Tensor) and (
                self.initial_reference.shape == (n_in, n_out)
            ), (
                "Got incoherent initial_reference, either it is not a torch.Tensor or its shape is not (n_in, n_out)"
            )
            self.register_buffer("reference_point", self.initial_reference.clone())

        # Track epoch changes
        self.register_buffer(
            "current_epoch", torch.tensor(0, dtype=torch.long, device=self.device)
        )
        self.register_buffer("last_stiefel_value", self.reference_point.detach())
        self._epoch_updated = False

        # Deal with retraction and tangent projection functions
        self.projectionTangent = (
            stiefel_projection_tangent_orthogonal
            if self.use_autograd
            else StiefelProjectionTangentOrthogonal.apply
        )
        if self.mapping == "QR":
            self.projectionStiefel = (
                stiefel_projection_qr
                if self.use_autograd
                else StiefelProjectionQR.apply
            )
        elif self.mapping == "polar":
            self.projectionStiefel = (
                stiefel_projection_polar
                if self.use_autograd
                else StiefelProjectionPolar.apply
            )

    def forward(self, weight_tangent: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the tangent space of reference_point to the Stiefel manifold

        Parameters
        ----------
        weight_tangent : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix, tangent vector at reference_point

        Returns
        -------
        weight : torch.Tensor of shape (n_in, n_out)
            Orthogonal matrix
        """
        # ensure weight_tangent is on the tangent space
        weight_tangent = self.projectionTangent(weight_tangent, self.reference_point)
        # map weight_tangent on the manifold
        weight = self.projectionStiefel(self.reference_point + weight_tangent)
        # store weight value during training (for reference update)
        if self.training and not self._epoch_updated:
            self.last_stiefel_value.copy_(weight.detach())
        return weight

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Mapping from Stiefel manifold to the tangent space at reference_point
        (achieved through orthogonal projection)

        Parameters
        ----------
        weight : torch.Tensor of shape (n_in, n_out)
            Orthogonal matrix

        Returns
        -------
        weight_tangent : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix, tangent vector at reference_point
        """
        return self.projectionTangent(
            weight - self.reference_point, self.reference_point
        )

    def on_epoch_end(self) -> None:
        """
        Called at the end of an epoch to update the reference point
        """
        if self.training:
            self.reference_point.copy_(self.last_stiefel_value)
            self.current_epoch += 1
            self._epoch_updated = True

    def on_epoch_start(self) -> None:
        """
        Called at the start of an epoch to reset the update flag
        """
        self._epoch_updated = False
