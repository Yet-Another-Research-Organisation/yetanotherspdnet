import torch
from torch import nn

from yetanotherspdnet.functions.scalar_functions import (
    inv_scaled_softplus,
    scaled_softplus,
)
from yetanotherspdnet.random.stiefel import random_stiefel

from ..functions.spd_linalg import (
    ExpmSymmetric,
    InvScaledSoftPlusSPD,
    LogmSPD,
    ScaledSoftPlusSymmetric,
    expm_symmetric,
    inv_scaled_softplus_SPD,
    inv_sqrtm_SPD,
    logm_SPD,
    scaled_softplus_symmetric,
    sqrtm_SPD,
    symmetrize,
)
from ..functions.stiefel import (
    StiefelProjectionPolar,
    StiefelProjectionQR,
    StiefelProjectionTangentOrthogonal,
    stiefel_projection_polar,
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
            Real scalar

        Returns
        -------
        scalar_pd : torch.Tensor of shape ()
            Positive definite scalar
        """
        return scaled_softplus(scalar)

    def right_inverse(self, scalar_pd: torch.Tensor) -> torch.Tensor:
        """
        Mapping from positive definite scalar onto real scalars through
        the inverse SoftPlus function

        Parameters
        ----------
        scalar_pd : torch.Tensor of shape ()
            Positive definite scalar

        Returns
        -------
        scalar : torch.Tensor of shape ()
            Real scalar
        """
        return inv_scaled_softplus(scalar_pd)


class ScalarSigmoidParametrization(nn.Module):
    """
    Parametrization to constrain a scalar to the interval [0, 1] using sigmoid.
    """

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Mapping from real scalar to [0, 1] through sigmoid function

        Parameters
        ----------
        scalar : torch.Tensor of shape ()
            Real scalar (unconstrained)

        Returns
        -------
        scalar_constrained : torch.Tensor of shape ()
            Scalar in [0, 1]
        """
        return torch.sigmoid(scalar)

    def right_inverse(self, scalar_constrained: torch.Tensor) -> torch.Tensor:
        """
        Mapping from [0, 1] to real scalars through inverse sigmoid (logit)

        Parameters
        ----------
        scalar_constrained : torch.Tensor of shape ()
            Scalar in [0, 1]

        Returns
        -------
        scalar : torch.Tensor of shape ()
            Real scalar (unconstrained)
        """
        # Clamp to avoid numerical issues at boundaries
        eps = 1e-7
        scalar_clamped = torch.clamp(scalar_constrained, eps, 1 - eps)
        return torch.log(scalar_clamped / (1 - scalar_clamped))


class SPDParametrization(nn.Module):
    def __init__(self, mapping: str = "softplus", use_autograd: bool = False) -> None:
        """
        SPD Parametrization

        Parameters
        ----------
        mapping : str, optional
            Mapping to obtain a SPD point from a symmetric matrix.
            Default is "softplus".
            Choices are: "softplus" and "exp"

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        assert mapping in [
            "softplus",
            "exp",
        ], f"mapping must be in ['softplus', 'exp'], got {mapping}"
        self.mapping = mapping
        self.use_autograd = use_autograd

        # deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = (
                (lambda data: scaled_softplus_symmetric(data)[0])
                if self.use_autograd
                else ScaledSoftPlusSymmetric.apply
            )
            self.tangent_fun = (
                (lambda data: inv_scaled_softplus_SPD(data)[0])
                if self.use_autograd
                else InvScaledSoftPlusSPD.apply
            )
        elif self.mapping == "exp":
            self.spd_fun = (
                (lambda data: expm_symmetric(data)[0])
                if self.use_autograd
                else ExpmSymmetric.apply
            )
            self.tangent_fun = (
                (lambda data: logm_SPD(data)[0]) if self.use_autograd else LogmSPD.apply
            )

    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the tangent space at identity to the SPD manifold

        Parameters
        ----------
        tangent_vector : torch.Tensor of shape (n_features, n_features)
            Symmetric matrix

        Returns
        -------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix
        """
        return self.spd_fun(tangent_vector)

    def right_inverse(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the SPD manifold to the tangent space at identity

        Parameters
        ----------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        tangent_vector : torch.Tensor of shape (n_features, n_features)
            Symmetric matrix
        """
        return self.tangent_fun(spd_matrix)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"SPDParametrization(mapping={self.mapping}, use_autograd={self.use_autograd})"

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
        assert mapping in [
            "softplus",
            "exp",
        ], f"mapping must be in ['softplus', 'exp'], got {mapping}"
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

        # Last SPD value (for reference point update)
        self.register_buffer("last_spd_value", self.reference_point.detach())

        # Deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = (
                (lambda data: scaled_softplus_symmetric(data)[0])
                if self.use_autograd
                else ScaledSoftPlusSymmetric.apply
            )
            self.tangent_fun = (
                (lambda data: inv_scaled_softplus_SPD(data)[0])
                if self.use_autograd
                else InvScaledSoftPlusSPD.apply
            )
        elif self.mapping == "exp":
            self.spd_fun = (
                (lambda data: expm_symmetric(data)[0])
                if self.use_autograd
                else ExpmSymmetric.apply
            )
            self.tangent_fun = (
                (lambda data: logm_SPD(data)[0]) if self.use_autograd else LogmSPD.apply
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
        if self.training:
            self.last_spd_value.copy_(spd_matrix.detach())

        return spd_matrix

    def right_inverse(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        Mapping from SPD manifold onto the tangent space at reference_point

        Parameters
        ----------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        tangent_vector : torch.Tensor of shape (n_features, n_features)
            Symmetric matrix
        """
        return (
            self.reference_point_sqrtm
            @ self.tangent_fun(
                self.reference_point_inv_sqrtm
                @ spd_matrix
                @ self.reference_point_inv_sqrtm
            )
            @ self.reference_point_sqrtm
        )

    def update_reference_point(self) -> None:
        """
        Update reference point with last SPD value
        """
        self.reference_point.copy_(self.last_spd_value)
        self.reference_point_sqrtm.copy_(sqrtm_SPD(self.last_spd_value)[0])
        self.reference_point_inv_sqrtm.copy_(inv_sqrtm_SPD(self.last_spd_value)[0])

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"SPDAdaptiveParametrization(n_features={self.n_features}, "
            f"initial_reference={self.initial_reference}, mapping={self.mapping}, "
            f"use_autograd={self.use_autograd}), "
            f"device={self.device}, dtype={self.dtype})"
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


class StiefelAdaptiveParametrization(nn.Module):
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
            WARNING: with "polar", use_autograd needs to be False

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
        assert mapping in [
            "QR",
            "polar",
        ], f"mapping must be in ['QR', 'polar'], got {mapping}"
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

        # Last Stiefel value (for reference point update)
        self.register_buffer("last_stiefel_value", self.reference_point.detach())

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
        # Need this to have a correct computation graph and gradient computation
        ref_point = self.reference_point.detach().clone()
        # ensure weight_tangent is on the tangent space
        weight_tangent = self.projectionTangent(weight_tangent, ref_point)
        # map weight_tangent on the manifold
        weight = self.projectionStiefel(ref_point + weight_tangent)
        # store weight value during training (for reference update)
        if self.training:
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

    def update_reference_point(self) -> None:
        """
        Update reference point with last Stiefel value
        """
        self.reference_point.copy_(self.last_stiefel_value)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"StiefelAdaptiveParametrization(n_in={self.n_in}, n_out={self.n_out}, "
            f"initial_reference={self.initial_reference}, mapping={self.mapping}, "
            f"use_autograd={self.use_autograd}), "
            f"device={self.device}, dtype={self.dtype}, generator={self.generator})"
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
