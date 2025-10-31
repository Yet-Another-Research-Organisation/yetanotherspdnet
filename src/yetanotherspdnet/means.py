import torch
from torch.autograd import Function

from .spd import (
    ExpmSymmetric,
    LogmSPD,
    eigh_operation,
    eigh_operation_grad,
    expm_symmetric,
    logm_SPD,
    solve_sylvester_SPD,
    sqrtm_SPD,
    symmetrize,
)


# ---------------
# Arithmetic mean
# ---------------
class ArithmeticMean(Function):
    """
    Arithmetic mean
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the arithmetic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Arithmetic mean
        """
        ctx.shape = data.shape
        return symmetrize(torch.mean(data, dim=tuple(range(data.ndim - 2))))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the arithmetic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the arithmetic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        shape = ctx.shape
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        grad_input = torch.zeros(shape, device=grad_output.device)
        grad_input[..., :, :] = symmetrize(grad_output / n_matrices)
        return grad_input


def arithmetic_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Arithmetic mean of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean
    """
    return symmetrize(torch.real(torch.mean(data, dim=tuple(range(data.ndim - 2)))))


def adaptive_update_arithmetic(
    point1: torch.Tensor, point2: torch.Tensor, t: int
) -> torch.Tensor:
    """
    Euclidean path for adaptive arithmetic mean computation:
    (1-t)*point1 + t*point2

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    return symmetrize((1 - t) * point1 + t * point2)


# -------------
# Harmonic mean
# -------------
class HarmonicMean(Function):
    """
    Harmonic mean
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the harmonic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Harmonic mean
        """
        ctx.shape = data.shape
        inv_data = torch.cholesky_inverse(torch.linalg.cholesky(data))
        inv_mean = arithmetic_mean(inv_data)
        mean = torch.cholesky_inverse(torch.linalg.cholesky(inv_mean))
        ctx.save_for_backward(inv_data, mean)
        return mean

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the harmonic mean of a batch of SPD matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the harmonic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of SPD matrices
        """
        shape = ctx.shape
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        inv_data, mean = ctx.saved_tensors
        tmp = symmetrize(mean @ grad_output @ mean)
        grad_input = symmetrize(inv_data @ tmp @ inv_data / n_matrices)
        return grad_input


def harmonic_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Harmonic mean of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Harmonic mean
    """
    inv_data = torch.cholesky_inverse(torch.linalg.cholesky(data))
    inv_mean = arithmetic_mean(inv_data)
    return torch.cholesky_inverse(torch.linalg.cholesky(inv_mean))


def adaptive_update_harmonic(
    point1: torch.Tensor, point2: torch.Tensor, t: int
) -> torch.Tensor:
    """
    Path for adaptive harmonic mean computation:
    ((1-t)*point1^{-1} + t*point2^{-1})^{-1}

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point1_inv = torch.cholesky_inverse(torch.linalg.cholesky(point1))
    point2_inv = torch.cholesky_inverse(torch.linalg.cholesky(point2))
    point_inv = adaptive_update_arithmetic(point1_inv, point2_inv, t)
    return torch.cholesky_inverse(torch.linalg.cholesky(point_inv))


# ------------------
# Log-Euclidean mean
# ------------------
def LogEuclideanMean(data: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean mean

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Log-Euclidean mean
    """
    log_data = LogmSPD.apply(data)
    log_mean = ArithmeticMean.apply(log_data)
    mean = ExpmSymmetric.apply(log_mean)
    return mean


def logEuclidean_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean mean of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Log-Euclidean mean
    """
    log_data = logm_SPD(data)
    log_mean = arithmetic_mean(log_data)
    return expm_symmetric(log_mean)


def adaptive_update_logEuclidean(
    point1: torch.Tensor, point2: torch.Tensor, t: int
) -> torch.Tensor:
    """
    Path for adaptive log-Euclidean mean computation:
    expm((1-t)*logm(point1) + t*logm(point2))

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point1_logm = logm_SPD(point1)
    point2_logm = logm_SPD(point2)
    point_logm = adaptive_update_arithmetic(point1_logm, point2_logm, t)
    return expm_symmetric(point_logm)


# -----------------------------------------------
# Geometric mean of arithmetic and harmonic means
# -----------------------------------------------
class GeometricMean2Points(Function):
    """
    Geometric mean of two SPD matrices
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


def geometric_mean_2points(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
    """
    Geometric mean of two SPD matrices

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
    eigvals1, eigvecs1 = torch.linalg.eigh(point1)
    point1_sqrtm = eigh_operation(eigvals1, eigvecs1, torch.sqrt)
    inv_sqrt = lambda x: 1 / torch.sqrt(x)
    point1_inv_sqrtm = eigh_operation(eigvals1, eigvecs1, inv_sqrt)
    middle_term1 = sqrtm_SPD(point1_inv_sqrtm @ point2 @ point1_inv_sqrtm)
    return point1_sqrtm @ middle_term1 @ point1_sqrtm


def GeometricArithmeticHarmonicMean(
    data: torch.Tensor, return_arithmetic_harmonic: bool = False
) -> torch.Tensor:
    """
    Geometric mean of the arithmetic and harmonic means

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    return_arithmetic_harmonic : bool, optional
        Whether to also return arithmetic and harmonic means (for adptative mean update reasons), by default False

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Geometric mean of the arithmetic and harmonic means

    mean_arithmetic : torch.Tensor of shape (n_features, n_features), optional
        Arithmetic mean

    mean_harmonic : torch.Tensor of shape (n_features, n_features), optional
        Harmonic mean (for adptative mean update reasons)
    """
    mean_arithmetic = ArithmeticMean.apply(data)
    mean_harmonic = HarmonicMean.apply(data)
    mean = GeometricMean2Points.apply(mean_arithmetic, mean_harmonic)
    if return_arithmetic_harmonic:
        return mean, mean_arithmetic, mean_harmonic
    else:
        return mean


def geometric_arithmetic_harmonic_mean(
    data: torch.Tensor, return_arithmetic_harmonic: bool = False
) -> torch.Tensor:
    """
    Geometric mean of the arithmetic and harmonic means of a batch of SPD matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    return_arithmetic_harmonic : bool, optional
        Whether to also return arithmetic and harmonic means (for adptative mean update reasons), by default False

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Geometric mean of the arithmetic and harmonic means

    mean_arithmetic : torch.Tensor of shape (n_features, n_features), optional
        Arithmetic mean

    mean_harmonic : torch.Tensor of shape (n_features, n_features), optional
        Harmonic mean (for adptative mean update reasons)
    """
    mean_arithmetic = arithmetic_mean(data)
    mean_harmonic = harmonic_mean(data)
    mean = geometric_mean_2points(mean_arithmetic, mean_harmonic)
    if return_arithmetic_harmonic:
        return mean, mean_arithmetic, mean_harmonic
    else:
        return mean


def adaptive_update_geometric_arithmetic_harmonic(
    point1_arithmetic: torch.Tensor,
    point1_harmonic: torch.Tensor,
    point2_arithmetic: torch.Tensor,
    point2_harmonic: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """
    Path for adaptive computation of the geometric mean of arithmetic and harmonic means:
    ( (1-t)*point1_arithmetic + t*point2_arithmetic ) #_{1/2} ( (1-t)*point1_harmonic^{-1} + t*point2_harmonic^{-1} )^{-1}

    Parameters
    ----------
    point1_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point1_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point_arithmetic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point_harmonic : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point_arithmetic = adaptive_update_arithmetic(
        point1_arithmetic, point2_arithmetic, t
    )
    point_harmonic = adaptive_update_harmonic(point1_harmonic, point2_harmonic, t)
    return (
        geometric_mean_2points(point_arithmetic, point_harmonic),
        point_arithmetic,
        point_harmonic,
    )


# --------------
# Geometric mean
# --------------
class GeometricMeanIteration(Function):
    """
    One iteration of the fixed-point algorithm computing the geometric mean
    """

    @staticmethod
    def forward(ctx, mean_iterate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of one iteration of the fixed-point algorithm for the geometric mean

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        mean_iterate : torch.Tensor of shape (n_features, n_features)
            Current iterate of the geometric mean

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices. The mean is computed along ... axes

        Returns
        -------
        mean_iterate_new : torch.Tensor of shape (n_features, n_features)
            New iterate of the geometric mean
        """
        ctx.shape = data.shape
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
        log_mean = arithmetic_mean(log_transformed_data)
        eigvals_log_mean, eigvecs_log_mean = torch.linalg.eigh(log_mean)
        exp_log_mean = eigh_operation(eigvals_log_mean, eigvecs_log_mean, torch.exp)
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
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of one iteration of the fixed-point algorithm for the geometric mean

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to the new iterate of the geometric mean

        Returns
        -------
        grad_input_mean : torch.Tensor of shape (nfeatures, nfeatures)
            Gradient of the loss with respect to the current iterate of the geometric mean

        grad_input_data : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the data at the current iterate
        """
        shape = ctx.shape
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

        syl2_right = data @ mean_iterate_inv_sqrtm @ diff_log_data
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
            @ diff_log_data
            @ mean_iterate_inv_sqrtm
            / n_matrices,
        )


def GeometricMean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor:
    """
    Geometric mean computed with fixed-point algorithm

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
        Geometric mean
    """
    n_features = data.shape[-1]
    mean = torch.eye(n_features, dtype=data.dtype, device=data.device)
    for _ in range(n_iterations):
        mean = GeometricMeanIteration.apply(mean, data)
    return mean


def geometric_mean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor:
    """
    Geometric mean computed with fixed-point algorithm

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices. The mean is computed along ... axes

    n_iterations : int
        Number of iterations to perform to estimate the geometric mean, by default 10

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Geometric mean
    """
    n_features = data.shape[-1]
    mean = torch.eye(
        n_features, dtype=data.dtype, device=data.device
    )  # initialize with identity to ensure correct manual backpropagation
    for _ in range(n_iterations):
        # sqrtm and inverse sqrtm of mean
        eigvals_mean, eigvecs_mean = torch.linalg.eigh(mean)
        mean_sqrtm = eigh_operation(eigvals_mean, eigvecs_mean, torch.sqrt)
        inv_sqrt = lambda x: 1 / torch.sqrt(x)
        mean_inv_sqrtm = eigh_operation(eigvals_mean, eigvecs_mean, inv_sqrt)
        # transform data
        transformed_data = mean_inv_sqrtm @ data @ mean_inv_sqrtm
        # compute descent direction
        logm_transformed_data = logm_SPD(transformed_data)
        logm_mean = arithmetic_mean(logm_transformed_data)
        # compute new iterate
        expm_logm_mean = expm_symmetric(logm_mean)
        mean = mean_sqrtm @ expm_logm_mean @ mean_sqrtm
    return mean


def adaptive_update_geometric(
    point1: torch.Tensor, point2: torch.Tensor, t: int
) -> torch.Tensor:
    """
    Path for adaptive geometric mean computation:
    point1^{1/2} ( point1^{-1/2} point2 point1^{-1/2} )^t point1^{-1/2}

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : int
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
