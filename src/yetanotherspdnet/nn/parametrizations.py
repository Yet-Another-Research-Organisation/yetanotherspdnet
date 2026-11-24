"""SPD and Stiefel manifold parametrizations for constrained optimization."""

from typing import cast

import torch
from torch import nn
from torch.nn.utils.parametrize import ParametrizationList

from yetanotherspdnet.functions._constants import SIGMOID_CLAMP_EPS
from yetanotherspdnet.functions.scalar_functions import (
    inv_scaled_softplus,
    scaled_softplus,
)
from yetanotherspdnet.random.stiefel import random_stiefel

from ..functions.dispatch import select_path
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


def _parametrization_of(module: nn.Module, tensor_name: str) -> ParametrizationList:
    """
    Return the parametrizations registered on ``module.<tensor_name>``.

    Same as ``module.parametrizations.<tensor_name>``, typed for type checkers
    (``nn.Module.__getattr__`` is typed ``Tensor | Module``).

    Parameters
    ----------
    module : nn.Module
        Module with a parametrized tensor.
    tensor_name : str
        Name of the parametrized tensor.

    Returns
    -------
    ParametrizationList
        Its parametrizations; ``.original`` is the unconstrained tensor.
    """
    return cast(ParametrizationList, getattr(module.parametrizations, tensor_name))


def recenter_dynamic_parametrization(module: nn.Module, tensor_name: str) -> None:
    """
    Move the reference point of a dynamic parametrization to the current value.

    Used by the post-step optimizer hooks of the layers with
    ``parametrization_mode="dynamic"`` (BiMap, SpectralVectorField): the
    reference point becomes the value of the parametrized tensor *after* the
    optimizer step, and the unconstrained tangent vector (``.original``) is
    reset to zero, so that the parametrized tensor does not move (it was
    already at the reference point: the retraction of a zero tangent vector
    is the reference point itself).

    Parameters
    ----------
    module : nn.Module
        Module whose ``tensor_name`` carries a single parametrization with an
        ``update_reference_point(value)`` method
        (:class:`StiefelAdaptiveParametrization`)
    tensor_name : str
        Name of the parametrized tensor
    """
    parametrization_list = _parametrization_of(module, tensor_name)
    with torch.no_grad():
        current = getattr(module, tensor_name).detach().clone()
        cast(StiefelAdaptiveParametrization, parametrization_list[0]).update_reference_point(
            current
        )
        parametrization_list.original.zero_()


class ScalarSoftPlusParametrization(nn.Module):
    """Parametrization constraining a scalar to be positive with a softplus."""

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Positive scalar parametrization through the scaled SoftPlus function.

        Rescaled so that f(0) = 1, as compared to the default torch function.

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
        Map a positive scalar to a real scalar (inverse SoftPlus).

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
    """Parametrization to constrain a scalar to the interval [0, 1] using sigmoid."""

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Map a real scalar to [0, 1] through the sigmoid function.

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
        Map [0, 1] to real scalars through the inverse sigmoid (logit).

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
        eps = SIGMOID_CLAMP_EPS
        scalar_clamped = torch.clamp(scalar_constrained, eps, 1 - eps)
        return torch.log(scalar_clamped / (1 - scalar_clamped))


class SPDParametrization(nn.Module):
    """
    Parametrization mapping a symmetric matrix to an SPD matrix.

    The eigenvalues of the unconstrained symmetric matrix go through a softplus
    (``mapping="softplus"``) or an exponential (``mapping="exp"``). Used for
    the SPD biases of batch normalization.
    """

    def __init__(self, mapping: str = "softplus", use_autograd: bool = False) -> None:
        """
        SPD Parametrization.

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
        if mapping not in ("softplus", "exp"):
            raise ValueError(f"mapping must be in ['softplus', 'exp'], got {mapping}")
        self.mapping = mapping
        self.use_autograd = use_autograd

        # deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = select_path(
                lambda data: scaled_softplus_symmetric(data)[0],
                ScaledSoftPlusSymmetric.apply,
                self.use_autograd,
            )
            self.tangent_fun = select_path(
                lambda data: inv_scaled_softplus_SPD(data)[0],
                InvScaledSoftPlusSPD.apply,
                self.use_autograd,
            )
        elif self.mapping == "exp":
            self.spd_fun = select_path(
                lambda data: expm_symmetric(data)[0],
                ExpmSymmetric.apply,
                self.use_autograd,
            )
            self.tangent_fun = select_path(
                lambda data: logm_SPD(data)[0], LogmSPD.apply, self.use_autograd
            )

    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map the tangent space at identity to the SPD manifold.

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
        Map the SPD manifold to the tangent space at identity.

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
        Representation of the layer.

        Returns
        -------
        str
            Representation of the layer
        """
        return f"SPDParametrization(mapping={self.mapping}, use_autograd={self.use_autograd})"


class SPDAdaptiveParametrization(nn.Module):
    """
    SPD parametrization around a moving reference point.

    The trainable tensor is a tangent vector at the current reference point.
    Calling ``update_reference_point`` moves the reference to the current value,
    which keeps the chart well conditioned during long trainings
    (``parametrization_mode="dynamic"`` in the layers).
    """

    # buffers registered in __init__ (declared for type checkers)
    reference_point: torch.Tensor
    reference_point_sqrtm: torch.Tensor
    reference_point_inv_sqrtm: torch.Tensor
    last_spd_value: torch.Tensor

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
        Adaptive SPD Parametrization.

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

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        device : torch.device, optional
            Device to run model on. Default is torch.device('cpu')

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64
        """
        super().__init__()

        self.n_features = n_features
        self.initial_reference = initial_reference
        if mapping not in ("softplus", "exp"):
            raise ValueError(f"mapping must be in ['softplus', 'exp'], got {mapping}")
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
            if not (
                isinstance(self.initial_reference, torch.Tensor)
                and self.initial_reference.shape == (n_features, n_features)
            ):
                raise ValueError(
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
        # clone: a plain .detach() would share storage with reference_point, so
        # every training forward would silently move the reference point
        self.register_buffer("last_spd_value", self.reference_point.detach().clone())

        # Deal with mapping
        if self.mapping == "softplus":
            self.spd_fun = select_path(
                lambda data: scaled_softplus_symmetric(data)[0],
                ScaledSoftPlusSymmetric.apply,
                self.use_autograd,
            )
            self.tangent_fun = select_path(
                lambda data: inv_scaled_softplus_SPD(data)[0],
                InvScaledSoftPlusSPD.apply,
                self.use_autograd,
            )
        elif self.mapping == "exp":
            self.spd_fun = select_path(
                lambda data: expm_symmetric(data)[0],
                ExpmSymmetric.apply,
                self.use_autograd,
            )
            self.tangent_fun = select_path(
                lambda data: logm_SPD(data)[0], LogmSPD.apply, self.use_autograd
            )

    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map the tangent space at reference_point onto the SPD manifold.

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
        Map the SPD manifold onto the tangent space at reference_point.

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
        """Update reference point with last SPD value."""
        self.reference_point.copy_(self.last_spd_value)
        self.reference_point_sqrtm.copy_(sqrtm_SPD(self.last_spd_value)[0])
        self.reference_point_inv_sqrtm.copy_(inv_sqrtm_SPD(self.last_spd_value)[0])

    def __repr__(self) -> str:
        """
        Representation of the layer.

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


class StiefelAdaptiveParametrization(nn.Module):
    """
    Stiefel parametrization around a moving reference point.

    The trainable tensor is a tangent vector at the current reference point,
    mapped back to the Stiefel manifold by a QR or polar retraction. Used by
    BiMap with ``parametrization_mode="dynamic"``.
    """

    # buffers registered in __init__ (declared for type checkers)
    reference_point: torch.Tensor
    last_stiefel_value: torch.Tensor

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
        Adaptive Stiefel Parametrization.

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

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        device : torch.device, optional
            Device to run model on. Default is torch.device('cpu')

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None
        """
        super().__init__()

        if n_in < n_out:
            raise ValueError(f"Must have n_in >= n_out, got n_in={n_in}, n_out={n_out}")
        self.n_in = n_in
        self.n_out = n_out
        self.initial_reference = initial_reference
        if mapping not in ("QR", "polar"):
            raise ValueError(f"mapping must be in ['QR', 'polar'], got {mapping}")
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
            if not (
                isinstance(self.initial_reference, torch.Tensor)
                and self.initial_reference.shape == (n_in, n_out)
            ):
                raise ValueError(
                    "Got incoherent initial_reference, either it is not a torch.Tensor or its shape is not (n_in, n_out)"
                )
            self.register_buffer("reference_point", self.initial_reference.clone())

        # Last Stiefel value (for reference point update)
        # clone: a plain .detach() would share storage with reference_point, so
        # every training forward would silently move the reference point
        self.register_buffer(
            "last_stiefel_value", self.reference_point.detach().clone()
        )

        # Deal with retraction and tangent projection functions
        self.projectionTangent = select_path(
            stiefel_projection_tangent_orthogonal,
            StiefelProjectionTangentOrthogonal.apply,
            self.use_autograd,
        )
        if self.mapping == "QR":
            self.projectionStiefel = select_path(
                stiefel_projection_qr, StiefelProjectionQR.apply, self.use_autograd
            )
        elif self.mapping == "polar":
            self.projectionStiefel = select_path(
                stiefel_projection_polar,
                StiefelProjectionPolar.apply,
                self.use_autograd,
            )

    def forward(self, weight_tangent: torch.Tensor) -> torch.Tensor:
        """
        Map the tangent space at reference_point to the Stiefel manifold.

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
        Map the Stiefel manifold to the tangent space at reference_point.

        Achieved through the orthogonal projection.

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

    def update_reference_point(self, value: torch.Tensor | None = None) -> None:
        """
        Move the reference point.

        Parameters
        ----------
        value : torch.Tensor of shape (n_in, n_out) or None, optional
            New reference point (a point of the Stiefel manifold). Default is
            None: the value of the last training forward.
        """
        if value is not None:
            self.last_stiefel_value.copy_(value)
        self.reference_point.copy_(self.last_stiefel_value)

    def __repr__(self) -> str:
        """
        Representation of the layer.

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
