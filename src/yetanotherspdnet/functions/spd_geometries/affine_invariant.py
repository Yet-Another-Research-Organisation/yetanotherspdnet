"""Affine-invariant Riemannian geometry: geodesic, exp/log maps, mean, and standard deviation."""

import math

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.scalar_functions import inv, inv_sqrt

from ..spd_linalg import (
    eigh_operation,
    eigh_operation_grad,
    expm_symmetric,
    inv_sqrtm_SPD,
    logm_SPD,
    solve_sylvester_SPD,
    sqrtm_SPD,
    symmetrize,
)
from .kullback_leibler import arithmetic_mean


# --------------------------
# Affine-invariant geodesics
# --------------------------
def affine_invariant_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Affine-invariant geodesic between two SPD matrices.

    .. math::

        \gamma(t) = P_1^{1/2}
            \big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{t}
            P_1^{1/2}

    with :math:`t \in [0, 1]` (:math:`\gamma(0) = P_1`,
    :math:`\gamma(1) = P_2`). This is the geodesic for the affine-invariant
    Riemannian metric on the SPD manifold, invariant under congruence
    transformations :math:`P \mapsto A P A^\top` for any invertible
    :math:`A`.

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
    point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    eigvals_middle_term1, eigvecs_middle_term1 = torch.linalg.eigh(
        point1_inv_sqrtm @ point2 @ point1_inv_sqrtm
    )
    pow_t = lambda x: torch.pow(x, t)
    middle_term1 = eigh_operation(eigvals_middle_term1, eigvecs_middle_term1, pow_t)
    return point1_sqrtm @ middle_term1 @ point1_sqrtm


class AffineInvariantGeodesic(Function):
    """
    Affine-invariant geodesic between two batches of SPD matrices.

    Computes: point1^{1/2} (point1^{-1/2} point2 point1^{-1/2})^t point1^{1/2}

    Supports gradients with respect to point1, point2, and optionally t
    (when t is a tensor with requires_grad=True).
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

        t : float | torch.Tensor
            Parameter on the geodesic path, should be in [0, 1]

        Returns
        -------
        point : torch.Tensor of shape (..., n_features, n_features)
            SPD matrices
        """
        eigvals1, eigvecs1 = torch.linalg.eigh(point1)
        point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Backward pass of the affine-invariant geodesic.

        Computes gradients with respect to point1, point2, and optionally t.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., nfeatures, nfeatures)
            Gradient of the loss with respect to the output

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., nfeatures, nfeatures) or None
            Gradient of the loss with respect to point1

        grad_input2 : torch.Tensor of shape (..., nfeatures, nfeatures) or None
            Gradient of the loss with respect to point2

        grad_t : torch.Tensor of shape () or None
            Gradient of the loss with respect to t (only when t requires grad)
        """
        (
            point1_sqrtm,
            point1_inv_sqrtm,
            eigvals_middle_term1,
            eigvecs_middle_term1,
            point1,
            point2,
        ) = ctx.saved_tensors
        t = ctx.t

        grad_input1 = None
        grad_input2 = None
        grad_t = None

        if ctx.needs_input_grad[1]:
            # Gradient with respect to point2
            pow_t = lambda x: torch.pow(x, t)
            pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
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

        if ctx.needs_input_grad[0]:
            # Gradient with respect to point1
            eigvals2, eigvecs2 = torch.linalg.eigh(point2)
            point2_sqrtm = eigh_operation(eigvals2, eigvecs2, torch.sqrt)
            point2_inv_sqrtm = eigh_operation(eigvals2, eigvecs2, inv_sqrt)
            eigvals_middle_term2, eigvecs_middle_term2 = torch.linalg.eigh(
                point2_inv_sqrtm @ point1 @ point2_inv_sqrtm
            )
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

        if ctx.needs_input_grad[2]:
            # Gradient with respect to t
            # d/dt [V D^t V^T] = V diag(D^t * log(D)) V^T
            # Using broadcasting: (V * (D^t * log(D)).unsqueeze(-2)) @ V^T
            log_eigvals = torch.log(eigvals_middle_term1)
            pow_t_eigvals = torch.pow(eigvals_middle_term1, t)
            d_eigvals = pow_t_eigvals * log_eigvals
            middle_term_deriv = (
                eigvecs_middle_term1 * d_eigvals.unsqueeze(-2)
            ) @ eigvecs_middle_term1.transpose(-1, -2)
            deriv_output_t = point1_sqrtm @ middle_term_deriv @ point1_sqrtm
            grad_t = torch.sum(grad_output * deriv_output_t)

        return grad_input1, grad_input2, grad_t


# -----------------------------------
# Affine-invariant mean of two points
# -----------------------------------
# TODO: These are probably to be removed when a Function class for affine-invariant geodesics
# will be implemented (needed for minibatch batchnorm approach)
def affine_invariant_mean_2points(
    point1: torch.Tensor, point2: torch.Tensor
) -> torch.Tensor:
    r"""
    Affine-invariant (geometric) mean of two SPD matrices.

    .. math::

        G(P_1, P_2) = P_1^{1/2}
            \big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{1/2}
            P_1^{1/2}

    the midpoint (:math:`t=1/2`) of the affine-invariant geodesic between
    :math:`P_1` and :math:`P_2` (see :func:`affine_invariant_geodesic`).

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
    r"""
    Affine-invariant (geometric/Fréchet) mean computed with a fixed-point
    (Karcher flow) algorithm.

    Starting from :math:`M_0 = I`, each iteration :math:`k` moves along the
    average tangent direction at :math:`M_k` and retracts back onto the
    manifold with the affine-invariant exponential map
    (:func:`affine_invariant_exp`):

    .. math::

        M_{k+1} = M_k^{1/2} \exp\!\left(\eta_k \cdot
            \frac{1}{N}\sum_{i=1}^{N} \log\big(M_k^{-1/2} P_i M_k^{-1/2}\big)
            \right) M_k^{1/2}

    with step size :math:`\eta_k = 0.95^k`. This converges to the unique
    minimizer of :math:`\sum_i d_{AI}(M, P_i)^2` for the affine-invariant
    distance :math:`d_{AI}`.

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
        n_matrices = math.prod(shape[:-2])
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


# ---------------
# Scalar variance
# ---------------
def affine_invariant_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    r"""
    Scalar standard deviation with respect to the affine-invariant distance.

    .. math::

        \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
            \big\lVert \log\big(G^{-1/2} P_i G^{-1/2}\big) \big\rVert_F^2}

    where :math:`G` is the reference point (typically the affine-invariant
    mean) — equivalently, the norm of :math:`\mathrm{Log}_G(P_i)` under the
    affine-invariant metric (:func:`affine_invariant_log`).

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    reference_point : torch.Tensor of shape (n_features, n_features)
        SPD matrix (some kind of mean of data)

    Returns
    -------
    scalar_std : torch.Tensor of shape ()
        scalar standard deviation
    """
    n_matrices = math.prod(data.shape[:-2])
    G_inv_sqrtm = inv_sqrtm_SPD(reference_point)[0]
    transformed_data = G_inv_sqrtm @ data @ G_inv_sqrtm
    eigvals = torch.linalg.eigvalsh(transformed_data)
    return torch.sqrt(torch.sum(torch.log(eigvals) ** 2) / n_matrices)


class AffineInvariantStdScalar(Function):
    """
    Scalar standard deviation with respect to the affine-invariant distance
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scalar standard deviation with respect to the affine-invariant distance

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        reference_point : torch.Tensor of shape (n_features, n_features)
            SPD matrix (some kind of mean of data)

        Returns
        -------
        scalar_std : torch.Tensor of shape ()
            scalar standard deviation
        """
        n_matrices = math.prod(data.shape[:-2])
        G_inv_sqrtm = inv_sqrtm_SPD(reference_point)[0]
        transformed_data = G_inv_sqrtm @ data @ G_inv_sqrtm
        eigvals_transdat, eigvecs_transdat = torch.linalg.eigh(transformed_data)
        scalar_std = torch.sqrt(
            torch.sum(torch.log(eigvals_transdat) ** 2) / n_matrices
        )
        ctx.n_matrices = n_matrices
        ctx.save_for_backward(
            data,
            reference_point,
            G_inv_sqrtm,
            eigvals_transdat,
            eigvecs_transdat,
            scalar_std,
        )
        return scalar_std

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the scalar standard deviation with respect to the affine-invariant distance

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape ()
            Gradient of the loss with respect to the output of the scalar standard deviation Function

        Returns
        -------
        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            gradient of the loss with respect to the input data

        grad_input_reference_point : torch.Tensor of shape (n_features, n_features)
            gradient of the loss with respect to the input reference point
        """
        n_matrices = ctx.n_matrices
        (
            data,
            reference_point,
            G_inv_sqrtm,
            eigvals_transdat,
            eigvecs_transdat,
            scalar_std,
        ) = ctx.saved_tensors

        data_inv_sqrtm = inv_sqrtm_SPD(data)[0]

        transformed_G = data_inv_sqrtm @ reference_point @ data_inv_sqrtm
        eigvals_transG, eigvecs_transG = torch.linalg.eigh(transformed_G)

        log_inv = lambda x: torch.log(x) / x
        middle_term_data = eigh_operation(eigvals_transdat, eigvecs_transdat, log_inv)
        middle_term_G = eigh_operation(eigvals_transG, eigvecs_transG, log_inv)

        grad_input_data = (
            grad_output
            * G_inv_sqrtm
            @ middle_term_data
            @ G_inv_sqrtm
            / n_matrices
            / scalar_std
        )
        grad_input_G = (
            grad_output
            * arithmetic_mean(data_inv_sqrtm @ middle_term_G @ data_inv_sqrtm)
            / scalar_std
        )
        return grad_input_data, grad_input_G


# --------------------------------
# Affine-invariant exponential map
# --------------------------------
def affine_invariant_exp(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    r"""
    Affine-invariant exponential map on the SPD manifold.

    .. math::

        \mathrm{Exp}_X(V) = X^{1/2} \exp\big(X^{-1/2} V X^{-1/2}\big) X^{1/2}

    Maps a tangent vector :math:`V` at base point :math:`X` (a symmetric
    matrix) to a point on the SPD manifold, by following the geodesic from
    :math:`X` in direction :math:`V` for unit time. Inverse of
    :func:`affine_invariant_log`.

    Parameters
    ----------
    base : torch.Tensor of shape (..., n, n)
        Base point(s) on the SPD manifold

    tangent : torch.Tensor of shape (..., n, n)
        Tangent vector(s) at base (symmetric matrices)

    Returns
    -------
    result : torch.Tensor of shape (..., n, n)
        Point(s) on the SPD manifold
    """
    eigvals, eigvecs = torch.linalg.eigh(base)
    base_sqrtm = eigh_operation(eigvals, eigvecs, torch.sqrt)
    base_inv_sqrtm = eigh_operation(eigvals, eigvecs, inv_sqrt)
    transformed = base_inv_sqrtm @ tangent @ base_inv_sqrtm
    exp_transformed = expm_symmetric(transformed)[0]
    return base_sqrtm @ exp_transformed @ base_sqrtm


class AffineInvariantExp(Function):
    """
    Affine-invariant exponential map with manual backward.

    Exp_X(V) = X^{1/2} expm(X^{-1/2} V X^{-1/2}) X^{1/2}
    """

    @staticmethod
    def forward(ctx, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the affine-invariant exponential map.

        Parameters
        ----------
        ctx : context
            Context for saving tensors for backward

        base : torch.Tensor of shape (..., n, n)
            Base point(s) on the SPD manifold

        tangent : torch.Tensor of shape (..., n, n)
            Tangent vector(s) at base (symmetric matrices)

        Returns
        -------
        result : torch.Tensor of shape (..., n, n)
            Point(s) on the SPD manifold
        """
        eigvals_base, eigvecs_base = torch.linalg.eigh(base)
        base_sqrtm = eigh_operation(eigvals_base, eigvecs_base, torch.sqrt)
        base_inv_sqrtm = eigh_operation(eigvals_base, eigvecs_base, inv_sqrt)
        transformed = base_inv_sqrtm @ tangent @ base_inv_sqrtm
        eigvals_trans, eigvecs_trans = torch.linalg.eigh(transformed)
        exp_transformed = eigh_operation(eigvals_trans, eigvecs_trans, torch.exp)
        result = base_sqrtm @ exp_transformed @ base_sqrtm
        ctx.save_for_backward(
            base_sqrtm,
            base_inv_sqrtm,
            eigvals_base,
            eigvecs_base,
            eigvals_trans,
            eigvecs_trans,
            base,
            tangent,
        )
        return result

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Backward pass of the affine-invariant exponential map.

        Parameters
        ----------
        ctx : context
            Context with saved tensors

        grad_output : torch.Tensor of shape (..., n, n)
            Gradient w.r.t. the output

        Returns
        -------
        grad_base : torch.Tensor of shape (..., n, n) or None
        grad_tangent : torch.Tensor of shape (..., n, n) or None
        """
        (
            base_sqrtm,
            base_inv_sqrtm,
            eigvals_base,
            eigvecs_base,
            eigvals_trans,
            eigvecs_trans,
            base,
            tangent,
        ) = ctx.saved_tensors

        grad_tangent = None
        grad_base = None

        if ctx.needs_input_grad[1]:
            # d(result)/d(tangent):
            # result = S @ expm(S^{-1} V S^{-1}) @ S with S = base^{1/2}
            # grad_tangent = S^{-1} @ d_expm(S^T @ grad @ S) @ S^{-1}
            grad_tangent = (
                base_inv_sqrtm
                @ eigh_operation_grad(
                    base_sqrtm @ grad_output @ base_sqrtm,
                    eigvals_trans,
                    eigvecs_trans,
                    torch.exp,
                    torch.exp,
                )
                @ base_inv_sqrtm
            )

        if ctx.needs_input_grad[0]:
            # Product rule for result = S @ M @ S where
            # S = base^{1/2}, M = expm(S^{-1} V S^{-1})
            exp_trans = eigh_operation(eigvals_trans, eigvecs_trans, torch.exp)

            # d/dS [S M S] with M=expm(T), S=base^{1/2}, G=grad_output:
            # gradient = M @ S @ G + G @ S @ M
            grad_through_sandwich = (
                exp_trans @ base_sqrtm @ grad_output
                + grad_output @ base_sqrtm @ exp_trans
            )
            grad_through_sqrtm = eigh_operation_grad(
                grad_through_sandwich,
                eigvals_base,
                eigvecs_base,
                torch.sqrt,
                lambda x: 0.5 / torch.sqrt(x),
            )

            # d/dR through expm argument T = R V R, where R = base^{-1/2}:
            # G_T = Daleckii-Krein gradient of expm at T w.r.t. S @ G @ S
            # d/dR = V @ R @ G_T + G_T @ R @ V
            inner_grad = eigh_operation_grad(
                base_sqrtm @ grad_output @ base_sqrtm,
                eigvals_trans,
                eigvecs_trans,
                torch.exp,
                torch.exp,
            )
            grad_through_inv_sqrtm = (
                tangent @ base_inv_sqrtm @ inner_grad
                + inner_grad @ base_inv_sqrtm @ tangent
            )
            grad_through_inv_sqrtm = eigh_operation_grad(
                grad_through_inv_sqrtm,
                eigvals_base,
                eigvecs_base,
                inv_sqrt,
                lambda x: -0.5 * torch.pow(x, -1.5),
            )

            grad_base = grad_through_sqrtm + grad_through_inv_sqrtm

        return grad_base, grad_tangent


# ----------------------------
# Affine-invariant log map
# ----------------------------
def affine_invariant_log(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    r"""
    Affine-invariant logarithmic map on the SPD manifold.

    .. math::

        \mathrm{Log}_X(Y) = X^{1/2} \log\big(X^{-1/2} Y X^{-1/2}\big) X^{1/2}

    Maps a point :math:`Y` on the manifold to a tangent vector at base
    :math:`X`. Inverse of :func:`affine_invariant_exp`.

    Parameters
    ----------
    base : torch.Tensor of shape (..., n, n)
        Base point(s) on the SPD manifold

    point : torch.Tensor of shape (..., n, n)
        Point(s) on the SPD manifold

    Returns
    -------
    tangent : torch.Tensor of shape (..., n, n)
        Tangent vector(s) at base (symmetric matrices)
    """
    eigvals, eigvecs = torch.linalg.eigh(base)
    base_sqrtm = eigh_operation(eigvals, eigvecs, torch.sqrt)
    base_inv_sqrtm = eigh_operation(eigvals, eigvecs, inv_sqrt)
    transformed = base_inv_sqrtm @ point @ base_inv_sqrtm
    log_transformed = logm_SPD(transformed)[0]
    return base_sqrtm @ log_transformed @ base_sqrtm


# -------------------------------------------
# Affine-invariant projection onto SPD
# -------------------------------------------
def affine_invariant_projx(data: torch.Tensor) -> torch.Tensor:
    """
    Project matrices onto the SPD manifold by symmetrizing and
    clamping eigenvalues to be strictly positive.

    Parameters
    ----------
    data : torch.Tensor of shape (..., n, n)
        Batch of matrices

    Returns
    -------
    projected : torch.Tensor of shape (..., n, n)
        Batch of SPD matrices
    """
    data = symmetrize(data)
    eigvals, eigvecs = torch.linalg.eigh(data)
    eigvals = eigvals.clamp(min=1e-8)
    return (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
