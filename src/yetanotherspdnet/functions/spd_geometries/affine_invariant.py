from numpy.linalg import eigvals
import torch
from torch.autograd import Function

from ..spd_linalg import (
    eigh_operation,
    eigh_operation_grad,
    sqrtm_SPD,
    expm_symmetric,
    logm_SPD,
    solve_sylvester_SPD,
)
from .kullback_leibler import arithmetic_mean


# --------------------------
# Affine-invariant geodesics
# --------------------------
def affine_invariant_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
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
    """
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    inv_sqrt = lambda x: 1 / torch.sqrt(x)
    point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
        point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
    )
    pow_t = lambda x: torch.pow(x, t)
    middle_term1 = eigh_operation(eigvals_middle_term1, eigvecs_middle_term1, pow_t)
    return point1_sqrtm @ middle_term1 @ point1_sqrtm


class AffineInvariantGeodesic(Function):
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
        eigvals1, eigvecs1 = torch.linalg.eigh(point1)
        point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
        eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
            point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
        )
        pow_t = lambda x: torch.pow(x, t)
        middle_term1 = eigh_operation(eigvals_middle_term1, eigvecs_middle_term1, pow_t)
        ctx.save_for_backward(
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
        )
        ctx.t = t
        return point1_sqrtm @ middle_term1 @ point1_sqrtm

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
            Gradient of the loss with respect to the geometric mean of two SPD matrices

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
        (
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
        ) = ctx.saved_tensors
        t = ctx.t
        eigvals2, eigvecs2 = torch.linalg.eigh(point2)
        point2_sqrtm = eigh_operation(eigvals2, eigvecs2, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        point2_inv_sqrtm = eigh_operation(eigvals2, eigvecs2, inv_sqrt)
        eigvals_middle_term2, eigvecs_middle_term2 = torch.linalg.eigh(
            point2_inv_sqrtm @ point1 @ point2_inv_sqrtm
        )
        pow_t = lambda x: torch.pow(x, t)
        pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
        pow_1_t = lambda x: torch.pow(x, 1 - t)
        pow_1_t_deriv = lambda x: (1 - t) * torch.pow(x, -t)
        grad_input1 = (
            point2_inv_sqrtm
            @ eigh_operation_grad(
                point2_sqrtm @ grad_output @ point2_sqrtm,
                eigvals_middle_term2,
                eigvecs_middle_term2,
                pow_1_t,
                pow_1_t_deriv,
            )
            @ point2_inv_sqrtm
        )
        grad_input2 = (
            point1_inv_sqrtm
            @ eigh_operation_grad(
                point1_sqrtm @ grad_output @ point1_sqrtm,
                eigvals_middle_term1,
                eigvecs_middle_term1,
                pow_t,
                pow_t_deriv,
            )
            @ point1_inv_sqrtm
        )
        return grad_input1, grad_input2, None


# -----------------------------------
# Affine-invariant mean of two points
# -----------------------------------
# TODO: These are probably to be removed when a Function class for affine-invariant geodesics
# will be implemented (needed for minibatch batchnorm approach)
def affine_invariant_mean_2points(
    point1: torch.Tensor, point2: torch.Tensor
) -> torch.Tensor:
    """
    Affine-invariant (geometric) mean of two SPD matrices

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., nfeatures, nfeatures)
        SPD matrices

    point2 : torch.Tensor of shape (..., nfeatures, nfeatures)
        SPD matrices

    Returns
    -------
    mean : torch.Tensor of shape (..., nfeatures, nfeatures)
        Geometric means of point1 and point2
    """
    # we don't use sqrtm_SPD and inv_sqrtm_SPD here to avoid an unnecessary evd
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    inv_sqrt = lambda x: 1 / torch.sqrt(x)
    point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    middle_term1 = sqrtm_SPD(point1_inv_sqrtm @ point2 @ point1_inv_sqrtm)[0]
    return point1_sqrtm @ middle_term1 @ point1_sqrtm


class AffineInvariantMean2Points(Function):
    """
    Affine-invariant (geometric) mean of two SPD matrices
    """

    @staticmethod
    def forward(ctx, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the geometric mean of two SPD matrices

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
        mean : torch.Tensor of shape (..., nfeatures, nfeatures)
            Geometric means of point1 and point2
        """
        eigvals1, eigvecs1 = torch.linalg.eigh(point1)
        point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
        eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
            point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
        )
        middle_term1 = eigh_operation(
            eigvals_middle_term1, eigvecs_middle_term1, torch.sqrt
        )
        ctx.save_for_backward(
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
        )
        return point1_sqrtm @ middle_term1 @ point1_sqrtm

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the geometric mean of two SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the geometric mean of two SPD matrices

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point1

        grad_input2 : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to point2
        """
        (
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
        ) = ctx.saved_tensors
        eigvals2, eigvecs2 = torch.linalg.eigh(point2)
        point2_sqrtm = eigh_operation(eigvals2, eigvecs2, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        point2_inv_sqrtm = eigh_operation(eigvals2, eigvecs2, inv_sqrt)
        eigvals_middle_term2, eigvecs_middle_term2 = torch.linalg.eigh(
            point2_inv_sqrtm @ point1 @ point2_inv_sqrtm
        )
        syl_sol1 = solve_sylvester_SPD(
            torch.sqrt(eigvals_middle_term2),
            eigvecs_middle_term2,
            point2_sqrtm @ grad_output @ point2_sqrtm,
        )
        syl_sol2 = solve_sylvester_SPD(
            torch.sqrt(eigvals_middle_term1),
            eigvecs_middle_term1,
            point1_sqrtm @ grad_output @ point1_sqrtm,
        )
        return (
            point2_inv_sqrtm @ syl_sol1 @ point2_inv_sqrtm,
            point1_inv_sqrtm @ syl_sol2 @ point1_inv_sqrtm,
        )


# ---------------------
# Affine-invariant mean
# ---------------------
def affine_invariant_mean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor:
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
        # sqrtm and inverse sqrtm of mean
        eigvals_mean, eigvecs_mean = torch.linalg.eigh(mean)
        mean_sqrtm = eigh_operation(eigvals_mean, eigvecs_mean, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        mean_inv_sqrtm = eigh_operation(eigvals_mean, eigvecs_mean, inv_sqrt)
        # transform data
        transformed_data = mean_inv_sqrtm @ data @ mean_inv_sqrtm
        # compute descent direction
        logm_transformed_data = logm_SPD(transformed_data)[0]
        logm_mean = arithmetic_mean(logm_transformed_data)
        # step-size to stabilize algorithm
        stepsize = 0.95**it
        # stepsize = 1
        # compute new iterate
        expm_logm_mean = expm_symmetric(stepsize * logm_mean)[0]
        mean = mean_sqrtm @ expm_logm_mean @ mean_sqrtm
    return mean


class AffineInvariantMeanIteration(Function):
    """
    One iteration of the fixed-point algorithm computing the affine-invariant (geometric) mean
    """

    @staticmethod
    def forward(
        ctx, mean_iterate: torch.Tensor, data: torch.Tensor, stepsize: float
    ) -> torch.Tensor:
        """
        Forward pass of one iteration of the fixed-point algorithm for the affine-invariant (geometric) mean

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        mean_iterate : torch.Tensor of shape (n_features, n_features)
            Current iterate of the affine-invariant mean

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        stepsize : float
            step-size to stabilize the fixed-point algorithm

        Returns
        -------
        mean_iterate_new : torch.Tensor of shape (n_features, n_features)
            New iterate of the affine-invariant mean
        """
        eigvals_mean_iterate, eigvecs_mean_iterate = torch.linalg.eigh(mean_iterate)
        mean_iterate_sqrtm = eigh_operation(
            eigvals_mean_iterate, eigvecs_mean_iterate, torch.sqrt
        )
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        mean_iterate_inv_sqrtm = eigh_operation(
            eigvals_mean_iterate, eigvecs_mean_iterate, inv_sqrt
        )
        transformed_data = mean_iterate_inv_sqrtm @ data @ mean_iterate_inv_sqrtm
        eigvals_transformed_data, eigvecs_transformed_data = torch.linalg.eigh(
            transformed_data
        )
        log_transformed_data = eigh_operation(
            eigvals_transformed_data, eigvecs_transformed_data, torch.log
        )
        log_mean = stepsize * arithmetic_mean(log_transformed_data)
        eigvals_log_mean, eigvecs_log_mean = torch.linalg.eigh(log_mean)
        exp_log_mean = eigh_operation(eigvals_log_mean, eigvecs_log_mean, torch.exp)
        ctx.shape = data.shape
        ctx.stepsize = stepsize
        ctx.save_for_backward(
            eigvals_mean_iterate,
            eigvecs_mean_iterate,
            data,
            mean_iterate_sqrtm,
            mean_iterate_inv_sqrtm,
            eigvals_transformed_data,
            eigvecs_transformed_data,
            eigvals_log_mean,
            eigvecs_log_mean,
            exp_log_mean,
        )
        return mean_iterate_sqrtm @ exp_log_mean @ mean_iterate_sqrtm

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of one iteration of the fixed-point algorithm for the affine-invariant (geometric) mean

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to the new iterate of the affine-invariant mean

        Returns
        -------
        grad_input_mean : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to the current iterate of the affine-invariant mean

        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the data at the current iterate
        """
        shape = ctx.shape
        stepsize = ctx.stepsize
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        (
            eigvals_mean_iterate,
            eigvecs_mean_iterate,
            data,
            mean_iterate_sqrtm,
            mean_iterate_inv_sqrtm,
            eigvals_transformed_data,
            eigvecs_transformed_data,
            eigvals_log_mean,
            eigvecs_log_mean,
            exp_log_mean,
        ) = ctx.saved_tensors

        diff_exp = eigh_operation_grad(
            mean_iterate_sqrtm @ grad_output @ mean_iterate_sqrtm,
            eigvals_log_mean,
            eigvecs_log_mean,
            torch.exp,
            torch.exp,
        )
        inv = lambda x: 1 / x
        diff_log_data = eigh_operation_grad(
            diff_exp.expand(shape),
            eigvals_transformed_data,
            eigvecs_transformed_data,
            torch.log,
            inv,
        )

        syl2_right = stepsize * data @ mean_iterate_inv_sqrtm @ diff_log_data
        syl2_right = arithmetic_mean(syl2_right + syl2_right.transpose(-1, -2))
        syl2_sol = solve_sylvester_SPD(
            1 / torch.sqrt(eigvals_mean_iterate), eigvecs_mean_iterate, syl2_right
        )

        syl1_right = exp_log_mean @ mean_iterate_sqrtm @ grad_output
        syl1_right = syl1_right + syl1_right.transpose(-1, -2)
        syl1_sol = solve_sylvester_SPD(
            torch.sqrt(eigvals_mean_iterate), eigvecs_mean_iterate, syl1_right
        )

        mean_iterate_inv = eigh_operation(
            eigvals_mean_iterate, eigvecs_mean_iterate, inv
        )

        return (
            syl1_sol - mean_iterate_inv @ syl2_sol @ mean_iterate_inv,
            mean_iterate_inv_sqrtm
            @ (stepsize * diff_log_data)
            @ mean_iterate_inv_sqrtm
            / n_matrices,
            None,
        )


def AffineInvariantMean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor:
    """
    Affine-invariant (geometric) mean computed with fixed-point algorithm

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    n_iterations : int
        Number of iterations to perform to estimate the geometric mean.
        Default is 10

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
        stepsize = 0.95**it
        # stepsize = 1
        mean = AffineInvariantMeanIteration.apply(mean, data, stepsize)
    return mean
