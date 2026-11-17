import torch
from torch import nn

from yetanotherspdnet.functions.scalar_functions import (
    inv_scaled_softplus,
    scaled_softplus,
)
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    AffineInvariantParallelTransportTangentTensor,
)
from yetanotherspdnet.random.stiefel import random_stiefel

from ..functions.spd_linalg import (
    ExpmSymmetric,
    InvScaledSoftPlusSPD,
    LogmSPD,
    ScaledSoftPlusSymmetric,
    SymCoordinatesToMatrix,
    SymMatrixToCoordinates,
    inv_sqrtm_SPD,
    sqrtm_SPD,
    symmetrize,
)
from ..functions.stiefel import (
    StiefelProjectionPolar,
    StiefelProjectionQR,
    StiefelProjectionTangentOrthogonal,
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


class SPDParametrization(nn.Module):
    def __init__(self, n_features: int, mapping: str = "softplus") -> None:
        """
        SPD Parametrization

        Parameters
        ----------
        n_features : int
            number of features

        mapping : str, optional
            Mapping to obtain a SPD point from a symmetric matrix.
            Default is "softplus".
            Choices are: "softplus" and "exp"
        """
        super().__init__()
        self.n_features = n_features
        assert mapping in [
            "softplus",
            "exp",
        ], f"mapping must be in ['softplus', 'exp'], got {mapping}"
        self.mapping = mapping

        # deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = ScaledSoftPlusSymmetric.apply
            self.tangent_fun = InvScaledSoftPlusSPD.apply
        elif self.mapping == "exp":
            self.spd_fun = ExpmSymmetric.apply
            self.tangent_fun = LogmSPD.apply

    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the tangent space at identity to the SPD manifold

        Parameters
        ----------
        tangent_vector : torch.Tensor of shape (n_features*(n_features+1)//2,)
            Coordinates vector

        Returns
        -------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix
        """
        return self.spd_fun(
            SymCoordinatesToMatrix.apply(tangent_vector, self.n_features)
        )

    def right_inverse(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        Mapping from the SPD manifold to the tangent space at identity

        Parameters
        ----------
        spd_matrix : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        tangent_vector : torch.Tensor of shape (n_features*(n_features+1)//2,)
            Coordinates vector
        """
        return SymMatrixToCoordinates.apply(self.tangent_fun(spd_matrix))

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"SPDParametrization(n_features={self.n_features}, mapping={self.mapping})"
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


class SPDAndTensorParametrization(nn.Module):
    def __init__(
        self,
        n_features: int,
        mapping: str = "softplus",
        tensor_format: str = "diagonal",
    ) -> None:
        """
        SPD and metric tensor parametrization

        Parameters
        ----------
        n_features : int
            Number of features.

        mapping : str, optional
            Mapping to obtain a SPD point from a symmetric matrix.
            Default is "softplus".
            Choices are: "softplus" and "exp".

        tensor_format : str, optional
            Structure of the metric tensor at identity point.
            Default is "diagonal".
            Choices are: "diagonal" and "full".
        """
        super().__init__()
        self.n_features = n_features
        self.dim = n_features * (n_features + 1) // 2
        assert mapping in [
            "softplus",
            "exp",
        ], f"mapping must be in ['softplus', 'exp'], got {mapping}"
        self.mapping = mapping
        assert tensor_format in ["diagonal", "full"], (
            f"tensor_format must be in ['diagonal', 'full'], got {tensor_format}"
        )
        self.tensor_format = tensor_format

        # deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = ScaledSoftPlusSymmetric.apply
            self.tangent_fun = InvScaledSoftPlusSPD.apply
        elif self.mapping == "exp":
            self.spd_fun = ExpmSymmetric.apply
            self.tangent_fun = LogmSPD.apply

        # deal with tensor_format
        if self.tensor_format == "diagonal":
            self.tensor_fun_fwd = self._tensor_fun_fwd_diagonal
            self.tensor_fun_bkwd = self._tensor_fun_bkwd_diagonal
        elif self.tensor_format == "full":
            self.tensor_fun_fwd = self._tensor_fun_fwd_full
            self.tensor_fun_bkwd = self._tensor_fun_bkwd_full

    def _tensor_fun_fwd_diagonal(
        self, tangent_tensor: torch.Tensor, spd_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary function to compute metric tensor at spd_point.

        Parameters
        ----------
        tangent_tensor : torch.Tensor of shape (dim,) with dim = n_features*(n_features+1)//2
            Vector encoding the diagonal of tangent vector of the metric tensor at the identity matrix.

        spd_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix, reference_point for metric tensor.

        Returns
        -------
        tensor : torch.Tensor of shape (dim, dim)
            Metric tensor at spd_point, transported from diagonal metric tensor at identity.
        """
        identity = torch.eye(
            spd_point.shape[-1], dtype=spd_point.dtype, device=spd_point.device
        )
        # TODO Vector version of that to do manual backprop ?
        # + more efficient computing of // transp in that case (no longer diag_embbed) possible ?
        tensor_at_identity = torch.diag_embed(scaled_softplus(tangent_tensor))
        return AffineInvariantParallelTransportTangentTensor(
            tensor_at_identity, identity, spd_point
        )

    def _tensor_fun_bkwd_diagonal(
        self, tensor: torch.Tensor, spd_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary function to compute the tangent vector of the tensor at identity
        from the tensor at spd_point.

        Parameters
        ----------
        tensor : torch.Tensor of shape (dim, dim) with dim = n_features*(n_features+1)//2
            Metric tensor at spd_point, transported from diagonal metric tensor at identity.

        spd_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix, reference_point for metric tensor.

        Returns
        -------
        tangent_tensor : torch.Tensor of shape (dim,)
            Vector encoding the diagonal of tangent vector of the metric tensor at the identity matrix.
        """
        identity = torch.eye(
            spd_point.shape[-1], dtype=spd_point.dtype, device=spd_point.device
        )
        tensor_at_identity = AffineInvariantParallelTransportTangentTensor(
            tensor, spd_point, identity
        )
        return inv_scaled_softplus(tensor_at_identity.diagonal(dim1=-1, dim2=-2))

    def _tensor_fun_fwd_full(
        self, tangent_tensor: torch.Tensor, spd_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary function to compute metric tensor at spd_point.

        Parameters
        ----------
        tangent_tensor : torch.Tensor of shape (dim*(dim+1)//2,) with dim = n_features*(n_features+1)//2
            Vector (in coordinates) encoding the diagonal of tangent vector
            of the metric tensor at the identity matrix.

        spd_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix, reference_point for metric tensor.

        Returns
        -------
        tensor : torch.Tensor of shape (dim, dim)
            Metric tensor at spd_point, transported from diagonal metric tensor at identity.
        """
        identity = torch.eye(
            spd_point.shape[-1], dtype=spd_point.dtype, device=spd_point.device
        )
        tensor_at_identity = ScaledSoftPlusSymmetric.apply(
            SymCoordinatesToMatrix.apply(tangent_tensor, self.dim)
        )
        return AffineInvariantParallelTransportTangentTensor(
            tensor_at_identity, identity, spd_point
        )

    def _tensor_fun_bkwd_full(
        self, tensor: torch.Tensor, spd_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary function to compute the tangent vector of the tensor at identity
        from the tensor at spd_point.

        Parameters
        ----------
        tensor : torch.Tensor of shape (dim, dim) with dim = n_features*(n_features+1)//2
            Metric tensor at spd_point, transported from diagonal metric tensor at identity.

        spd_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix, reference_point for metric tensor.

        Returns
        -------
        tangent_tensor : torch.Tensor of shape (dim*(dim+1)//2,)
            Vector encoding the diagonal of tangent vector of the metric tensor at the identity matrix.
        """
        identity = torch.eye(
            spd_point.shape[-1], dtype=spd_point.dtype, device=spd_point.device
        )
        tensor_at_identity = AffineInvariantParallelTransportTangentTensor(
            tensor, spd_point, identity
        )
        return SymMatrixToCoordinates.apply(
            InvScaledSoftPlusSPD.apply(tensor_at_identity)
        )

    def forward(
        self, tangent_vector: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For SPD matrix: mapping from the tangent space at identity to the SPD manifold.
        For metric tensor: mapping from symmetric matrices to SPD matrices (tensor at identity),
        followed by parallel transport to get tensor at spd_point.

        Parameters
        ----------
        tangent_vector : tuple[torch.Tensor, torch.Tensor] of shapes (dim,) and (dim*(dim+1)//2,),
        where dim = n_features*(n_features+1)//2
            Coordinates of symmetric matrix and of tangent of metric tensor at identity

        Returns
        -------
        point : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and (dim, dim)
            SPD matrix and metric tensor
        """
        spd_point = self.spd_fun(
            SymCoordinatesToMatrix.apply(tangent_vector[0], self.n_features)
        )
        tensor = self.tensor_fun_fwd(tangent_vector[1], spd_point)
        return (spd_point, tensor)

    def right_inverse(
        self, point: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For SPD matrix: mapping from SPD manifold to tangent space at identity.
        For metric tensor: tensor at spd_point transported at identity then
        mapped to symmetric matrices.

        Parameters
        ----------
        point : tuple[torch.Tensor, torch.Tensor] of shapes (n_features, n_features) and (dim, dim),
        where dim = n_features*(n_features+1)//2
            SPD matrix and metric tensor

        Returns
        -------
        tangent_vector : tuple[torch.Tensor, torch.Tensor] of shapes (dim,) and (dim*(dim+1)//2,),
            Coordinates of symmetric matrix and of tangent of metric tensor at identity
        """
        tangent_matrix = SymMatrixToCoordinates.apply(self.tangent_fun(point[0]))
        tangent_tensor = self.tensor_fun_bkwd(point[1], point[0])
        return (tangent_matrix, tangent_tensor)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"SPDAndTensorParametrization(\n"
            f"n_features={self.n_features},\n"
            f"mapping={self.mapping},\n"
            f"tensor_format={self.tensor_format},\n"
            ")"
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


class SPDAdaptiveParametrization(nn.Module):
    def __init__(
        self,
        n_features: int,
        initial_reference: torch.Tensor | None = None,
        mapping: str = "softplus",
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
            self.spd_fun = ScaledSoftPlusSymmetric.apply
            self.tangent_fun = InvScaledSoftPlusSPD.apply
        elif self.mapping == "exp":
            self.spd_fun = ExpmSymmetric.apply
            self.tangent_fun = LogmSPD.apply

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
        self.projectionTangent = StiefelProjectionTangentOrthogonal.apply
        if self.mapping == "QR":
            self.projectionStiefel = StiefelProjectionQR.apply
        elif self.mapping == "polar":
            self.projectionStiefel = StiefelProjectionPolar.apply

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
