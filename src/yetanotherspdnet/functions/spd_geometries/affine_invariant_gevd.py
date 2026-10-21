import torch
from torch.autograd import Function

from yetanotherspdnet.functions.scalar_functions import inv, sqrt_derivative
from yetanotherspdnet.functions.spd_geometries.kullback_leibler import (
    ArithmeticMean,
    arithmetic_mean,
)
from yetanotherspdnet.functions.spd_linalg import (
    CongruenceRectangular,
    _aux_eigh_operation_grad,
    congruence_rectangular,
    gevd,
    gevd_operation,
    gevd_operation_grad,
    skew_symmetrize,
    symmetrize,
)


# --------------------------
# affine-invariant geodesics
# --------------------------
def affine_invariant_geodesic_gevd(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant geodesic:
    point1^{1/2} ( point1^{-1/2} point2 point1^{-1/2} )^t point1^{1/2}

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (..., n_features, n_features)
        Inverses of general_eigvecs
    """
    general_eigvals, general_eigvecs, general_eigvecs_inv = gevd(point1, point2)
    pow_t = lambda x: torch.pow(x, t)
    return (
        gevd_operation(general_eigvals, general_eigvecs, pow_t),
        general_eigvals,
        general_eigvecs,
        general_eigvecs_inv,
    )


class AffineInvariantGeodesicGEVD(Function):
    """
    Affine-invariant geodesic between two batches of SPD matrices
    """

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
    ):
        """
        Forward pass of the affine-invariant geodesic between two batches of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point1 : torch.Tensor of shape (..., nfeatures, nfeatures)
            SPD matrices

        point2 : torch.Tensor of shape (..., nfeatures, nfeatures)
            SPD matrices

        Returns
        -------
        point : torch.Tensor of shape (..., n_features, n_features)
            SPD matrices
        """
        point, general_eigvals, general_eigvecs, general_eigvecs_inv = (
            affine_invariant_geodesic_gevd(point1, point2, t)
        )
        ctx.t = t
        ctx.save_for_backward(general_eigvals, general_eigvecs, general_eigvecs_inv)
        return point

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the affine-invariant geodesic

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the geodesic between two SPD matrices

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point1

        grad_input2 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point2
        """
        # TODO: We need to add the backward with respect to t
        # We need to add conditions to only compute required gradients
        # condition is made possible by ctx.needs_input_grad
        # This should be done everywhere
        # Also, here, probably better to switch roles of point1 and point2
        # so that we can get the grad of point1 directly with computations from
        # forward (and set grad2 to None). Maybe more logical, to discuss with Ammar
        t = ctx.t
        general_eigvals, general_eigvecs, general_eigvecs_inv = ctx.saved_tensors
        pow_t = lambda x: torch.pow(x, t)
        pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
        grad_input1, grad_input2 = gevd_operation_grad(
            grad_output,
            general_eigvals,
            general_eigvecs,
            general_eigvecs_inv,
            pow_t,
            pow_t_deriv,
        )
        return grad_input1, grad_input2, None


# ------------------------------------
# affine-invariant exponential mapping
# ------------------------------------
def affine_invariant_exponential_gevd(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant exponential mapping

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices, in the tangent space of reference_point

    reference_point : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_exp : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (..., n_features, n_features)
        Inverses of general_eigvecs
    """
    general_eigvals, general_eigvecs, general_eigvecs_inv = gevd(reference_point, data)
    return (
        gevd_operation(general_eigvals, general_eigvecs, torch.exp),
        general_eigvals,
        general_eigvecs,
        general_eigvecs_inv,
    )


class AffineInvariantExponentialGEVD(Function):
    """
    Affine-invariant exponential mapping
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Affine-invariant exponential mapping

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., nfeatures, nfeatures)
            Batch of symmetric matrices, in the tangent space of reference_point

        reference_point : torch.Tensor of shape (nfeatures, nfeatures)
            SPD matrix

        Returns
        -------
        data_exp : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices
        """
        data_exp, general_eigvals, general_eigvecs, general_eigvecs_inv = (
            affine_invariant_exponential_gevd(data, reference_point)
        )
        ctx.save_for_backward(general_eigvals, general_eigvecs, general_eigvecs_inv)
        return data_exp

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the affine-invariant logarithm mapping

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the affine-invariant exponential mapping

        Returns
        -------
        grad_data : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to data

        grad_reference : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to reference_point
        """
        general_eigvals, general_eigvecs, general_eigvecs_inv = ctx.saved_tensors
        grad_reference, grad_data = gevd_operation_grad(
            grad_output,
            general_eigvals,
            general_eigvecs,
            general_eigvecs_inv,
            torch.exp,
            torch.exp,
        )
        return grad_data, grad_reference


# ----------------------------------
# affine-invariant logarithm mapping
# ----------------------------------
def affine_invariant_logarithm_gevd(
    data: torch.Tensor, reference_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant logarithm mapping

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    reference_point : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_log : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices, in the tangent space of reference_point

    general_eigvals : torch.Tensor of shape (..., n_features)
        Batch of generalized eigenvalues

    general_eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Batch of generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (..., n_features, n_features)
        Inverses of general_eigvecs
    """
    general_eigvals, general_eigvecs, general_eigvecs_inv = gevd(reference_point, data)
    return (
        gevd_operation(general_eigvals, general_eigvecs, torch.log),
        general_eigvals,
        general_eigvecs,
        general_eigvecs_inv,
    )


class AffineInvariantLogarithmGEVD(Function):
    """
    Affine-invariant logarithm mapping
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Affine-invariant logarithm mapping

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., nfeatures, nfeatures)
            Batch of SPD matrices

        reference_point : torch.Tensor of shape (nfeatures, nfeatures)
            SPD matrix

        Returns
        -------
        data_log : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices, in the tangent space of reference_point
        """
        data_log, general_eigvals, general_eigvecs, general_eigvecs_inv = (
            affine_invariant_logarithm_gevd(data, reference_point)
        )
        ctx.save_for_backward(general_eigvals, general_eigvecs, general_eigvecs_inv)
        return data_log

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the affine-invariant logarithm mapping

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the affine-invariant logarithm mapping

        Returns
        -------
        grad_data : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to data

        grad_reference : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to reference_point
        """
        general_eigvals, general_eigvecs, general_eigvecs_inv = ctx.saved_tensors
        grad_reference, grad_data = gevd_operation_grad(
            grad_output,
            general_eigvals,
            general_eigvecs,
            general_eigvecs_inv,
            torch.log,
            inv,
        )
        return grad_data, grad_reference


# ---------------------
# Affine-invariant mean
# ---------------------
def affine_invariant_mean_gevd(
    data: torch.Tensor, n_iterations: int = 5
) -> torch.Tensor:
    """
    Affine-invariant (geometric) mean computed with fixed-point algorithm

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    n_iterations : int
        Number of iterations to perform to estimate the geometric mean, by default 5

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        SPD matrix
    """
    if data.ndim == 2:
        return data
    n_features = data.shape[-1]
    mean = torch.eye(
        n_features, dtype=data.dtype, device=data.device
    )  # initialize with identity to ensure correct manual backpropagation
    for it in range(n_iterations):
        log_data = affine_invariant_logarithm_gevd(data, mean)[0]
        log_mean = arithmetic_mean(log_data)
        stepsize = 0.95**it
        mean = affine_invariant_exponential_gevd(stepsize * log_mean, mean)[0]
    return mean


def AffineInvariantMeanGEVD(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor:
    """
    Affine-invariant (geometric) mean computed with fixed-point algorithm

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    n_iterations : int
        Number of iterations to perform to estimate the geometric mean, by default 5

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        SPD matrix
    """
    if data.ndim == 2:
        return data
    n_features = data.shape[-1]
    mean = torch.eye(n_features, dtype=data.dtype, device=data.device)
    for it in range(n_iterations):
        log_data = AffineInvariantLogarithmGEVD.apply(data, mean)
        log_mean = ArithmeticMean.apply(log_data)
        stepsize = 0.95**it
        mean = AffineInvariantExponentialGEVD.apply(stepsize * log_mean, mean)
    return mean


# -----------------------------------
# Affine-invariant parallel transport
# -----------------------------------
def affine_invariant_parallel_transport_transformation_gevd(
    current_reference: torch.Tensor, new_reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant transformation matrix for parallel transport.

    Parameters
    ----------
    current_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    new_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    transformation_matrix : torch.Tensor of shape (n_features, n_features)
        Non-singular matrix

    general_eigvals : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of new_reference w.r.t. current_reference

    general_eigvecs : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    general_eigvecs_inv : torch.Tensor of shape (n_features, n_features)
        Inverse of general_eigvecs
    """
    general_eigvals, general_eigvecs, general_eigvecs_inv = gevd(
        current_reference, new_reference
    )
    return (
        (general_eigvecs_inv * torch.sqrt(general_eigvals).unsqueeze(-2))
        @ general_eigvecs,
        general_eigvals,
        general_eigvecs,
        general_eigvecs_inv,
    )


class AffineInvariantParallelTransportTransformationGEVD(Function):
    """
    Affine-invariant transformation matrix for parallel transport
    """

    @staticmethod
    def forward(
        ctx, current_reference: torch.Tensor, new_reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the affine-invariant transformation matrix for parallel transport.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        current_reference : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        new_reference : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Returns
        -------
        transformation_matrix : torch.Tensor of shape (n_features, n_features)
            Non-singular matrix
        """
        transformation_matrix, general_eigvals, general_eigvecs, general_eigvecs_inv = (
            affine_invariant_parallel_transport_transformation_gevd(
                current_reference, new_reference
            )
        )
        ctx.save_for_backward(general_eigvals, general_eigvecs, general_eigvecs_inv)
        return transformation_matrix

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the affine-invariant transformation matrix for parallel transport.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the affine-invariant transformation matrix for parallel transport

        Returns
        -------
        grad_current_ref : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. input SPD matrix current_reference

        grad_new_ref : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. input SPD matrix new_reference
        """
        general_eigvals, general_eigvecs, general_eigvecs_inv = ctx.saved_tensors
        grad_out_transf = (
            general_eigvecs @ grad_output.transpose(-1, -2) @ general_eigvecs_inv
        )
        grad_out_transf_sym = symmetrize(grad_out_transf)
        grad_out_transf_skew = skew_symmetrize(grad_out_transf)
        aux_mat_new = _aux_eigh_operation_grad(
            general_eigvals, torch.sqrt, sqrt_derivative
        )
        middle_term_new = aux_mat_new * grad_out_transf_sym
        grad_new_ref = (
            general_eigvecs_inv
            @ middle_term_new
            @ general_eigvecs_inv.transpose(-1, -2)
        )
        _Lambda = torch.sqrt(general_eigvals)
        middle_term_current = symmetrize(
            grad_out_transf_skew * _Lambda.unsqueeze(-2)
            - middle_term_new * general_eigvals.unsqueeze(-2)
        )
        grad_current_ref = (
            general_eigvecs_inv
            @ middle_term_current
            @ general_eigvecs_inv.transpose(-1, -2)
        )
        return grad_current_ref, grad_new_ref


def affine_invariant_parallel_transport_gevd(
    data: torch.Tensor, current_reference: torch.Tensor, new_reference: torch.Tensor
) -> torch.Tensor:
    """
    Affine-invariant parallel transport of some symmetric (or SPD) matrices data from current_reference to new_reference.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric (or SPD) matrices

    current_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    new_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transp : torch.Tensor of shape (..., n_features, n_features)
        Batch of transported symmetric (or SPD) matrices
    """
    transformation_matrix = affine_invariant_parallel_transport_transformation_gevd(
        current_reference, new_reference
    )[0]
    return congruence_rectangular(data, transformation_matrix)


def AffineInvariantParallelTransportGEVD(
    data: torch.Tensor, current_reference: torch.Tensor, new_reference: torch.Tensor
) -> torch.Tensor:
    """
    Affine-invariant parallel transport of some symmetric (or SPD) matrices data from current_reference to new_reference.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric (or SPD) matrices

    current_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    new_reference : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Returns
    -------
    data_transp : torch.Tensor of shape (..., n_features, n_features)
        Batch of transported symmetric (or SPD) matrices
    """
    transformation_matrix = AffineInvariantParallelTransportTransformationGEVD.apply(
        current_reference, new_reference
    )
    return CongruenceRectangular.apply(data, transformation_matrix)
