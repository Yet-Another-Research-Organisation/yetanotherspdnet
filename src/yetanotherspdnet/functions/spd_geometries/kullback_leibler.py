import torch
from torch.autograd import Function

from ..spd_linalg import symmetrize


# ------------------
# Euclidean geodesic
# ------------------
def euclidean_geodesic(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Euclidean geodesic:
    (1-t)*point1 + t*point2

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        Symmetric matrices
    """
    return symmetrize((1 - t) * point1 + t * point2)


class EuclideanGeodesic(Function):
    """
    Euclidean geodesic of a batch of symmetric matrices
    """

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Euclidean geodesic

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        point1 : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices

        point2 : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices

        t : float | torch.Tensor
            parameter on the path, should be in [0,1]

        Returns
        -------
        point : torch.Tensor of shape (..., n_features, n_features)
            Symmetric matrices
        """
        ctx.t = t
        return symmetrize((1 - t) * point1 + t * point2)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the Euclidean geodesic

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., n_features, n_features)

        grad_input2 : torch.Tensor of shape (..., n_features, n_features)
        """
        t = ctx.t
        return symmetrize((1 - t) * grad_output), symmetrize(t * grad_output), None


# ---------------
# Arithmetic mean
# ---------------
def arithmetic_mean(data: torch.Tensor) -> torch.Tensor:
    """
    Arithmetic mean of a batch of symmetric matrices

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices. The mean is computed along ... axes

    Returns
    -------
    mean : torch.Tensor of shape (n_features, n_features)
        Arithmetic mean
    """
    if data.ndim == 2:
        return data
    return symmetrize(torch.real(torch.mean(data, dim=tuple(range(data.ndim - 2)))))


class ArithmeticMean(Function):
    """
    Arithmetic mean
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the arithmetic mean of a batch of symmetric matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices. The mean is computed along ... axes

        Returns
        -------
        mean : torch.Tensor of shape (n_features, n_features)
            Arithmetic mean
        """
        ctx.shape = data.shape
        if data.ndim == 2:
            return symmetrize(data)
        return symmetrize(torch.mean(data, dim=tuple(range(data.ndim - 2))))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the arithmetic mean of a batch of symmetric matrices

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss with respect to the arithmetic mean

        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input batch of symmetric matrices
        """
        shape = ctx.shape
        if len(shape) == 2:
            return symmetrize(grad_output)
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        grad_input = torch.zeros(shape, device=grad_output.device)
        grad_input[..., :, :] = symmetrize(grad_output / n_matrices)
        return grad_input


# ---------------
# Scalar variance
# ---------------
def left_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    """
    Scalar standard deviation with respect to the left Kullback-Leibler divergence

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
    n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
    n_features = data.shape[-1]
    L_G = torch.linalg.cholesky(reference_point)
    G_inv = torch.cholesky_inverse(L_G)
    L_data = torch.linalg.cholesky(data)
    trace = torch.einsum("ik,...ki->", G_inv, data)
    logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
    logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
    var = (trace - logdet_data) / n_matrices + logdet_G - n_features
    # clamp to avoid small numerical errors yielding small negative variance
    return torch.sqrt(torch.clamp(var, min=0.0))


class LeftKullbackLeiblerStdScalar(Function):
    """
    Scalar standard deviation with respect to the left Kullback-Leibler divergence
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scalar standard deviation with respect to the left Kullback-Leibler divergence

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
        n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
        n_features = data.shape[-1]
        L_G = torch.linalg.cholesky(reference_point)
        G_inv = torch.cholesky_inverse(L_G)
        L_data = torch.linalg.cholesky(data)
        trace = torch.einsum("ik,...ki->", G_inv, data)
        logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
        logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
        # clamp to avoid small numerical errors yielding small negative variance
        std_scalar = torch.sqrt(
            torch.clamp(
                (trace - logdet_data) / n_matrices + logdet_G - n_features, min=0.0
            )
        )
        ctx.n_matrices = n_matrices
        ctx.save_for_backward(data, reference_point, G_inv, L_data, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the scalar standard deviation with respect to the left Kullback-Leibler divergence

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
        data, reference_point, G_inv, L_data, std_scalar = ctx.saved_tensors
        data_inv = torch.cholesky_inverse(L_data)
        grad_input_data = grad_output * (G_inv - data_inv) / 2 / std_scalar / n_matrices
        grad_input_G = (
            grad_output
            * G_inv
            @ (reference_point - arithmetic_mean(data))
            @ G_inv
            / 2
            / std_scalar
        )
        return grad_input_data, grad_input_G


# --------------
# Harmonic curve
# --------------
def harmonic_curve(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """
    Curve for adaptive harmonic mean computation:
    ((1-t)*point1^{-1} + t*point2^{-1})^{-1}

    Parameters
    ----------
    point1 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    point2 : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices

    t : float
        parameter on the path, should be in [0,1]

    Returns
    -------
    point : torch.Tensor of shape (..., n_features, n_features)
        SPD matrices
    """
    point1_inv = torch.cholesky_inverse(torch.linalg.cholesky(point1))
    point2_inv = torch.cholesky_inverse(torch.linalg.cholesky(point2))
    point_inv = euclidean_geodesic(point1_inv, point2_inv, t)
    return torch.cholesky_inverse(torch.linalg.cholesky(point_inv))


class HarmonicCurve(Function):
    """
    Harmonic curve of two batches of SPD matrices
    """

    @staticmethod
    def forward(
        ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the harmonic curve

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

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
        point1_inv = torch.cholesky_inverse(torch.linalg.cholesky(point1))
        point2_inv = torch.cholesky_inverse(torch.linalg.cholesky(point2))
        point_inv = euclidean_geodesic(point1_inv, point2_inv, t)
        point = torch.cholesky_inverse(torch.linalg.cholesky(point_inv))
        ctx.t = t
        ctx.save_for_backward(point1_inv, point2_inv, point)
        return point

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the harmonic curve

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)

        Returns
        -------
        grad_input1 : torch.Tensor of shape (..., n_features, n_features)

        grad_input2 : torch.Tensor of shape (..., n_features, n_features)
        """
        t = ctx.t
        point1_inv, point2_inv, point = ctx.saved_tensors
        return (
            (1 - t) * point1_inv @ point @ symmetrize(grad_output) @ point @ point1_inv,
            t * point2_inv @ point @ symmetrize(grad_output) @ point @ point2_inv,
            None,
        )


# -------------
# Harmonic mean
# -------------
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
    if data.ndim == 2:
        return data
    inv_data = torch.cholesky_inverse(torch.linalg.cholesky(data))
    inv_mean = arithmetic_mean(inv_data)
    return torch.cholesky_inverse(torch.linalg.cholesky(inv_mean))


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
        if data.ndim == 2:
            return data
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
        if len(shape) == 2:
            return symmetrize(grad_output)
        n_matrices = torch.prod(torch.tensor(shape[:-2]))
        inv_data, mean = ctx.saved_tensors
        tmp = symmetrize(mean @ grad_output @ mean)
        grad_input = symmetrize(inv_data @ tmp @ inv_data / n_matrices)
        return grad_input


# ---------------
# Scalar variance
# ---------------
def right_kullback_leibler_std_scalar(
    data: torch.Tensor, reference_point: torch.Tensor
) -> torch.Tensor:
    """
    Scalar standard deviation with respect to the left Kullback-Leibler divergence

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
    n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
    n_features = data.shape[-1]
    L_G = torch.linalg.cholesky(reference_point)
    L_data = torch.linalg.cholesky(data)
    data_inv = torch.cholesky_inverse(L_data)
    trace = torch.einsum("...ik,ki->", data_inv, reference_point)
    logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
    logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
    var = (trace + logdet_data) / n_matrices - logdet_G - n_features
    # clamp to avoid small numerical errors yielding small negative variance
    return torch.sqrt(torch.clamp(var, min=0.0))


class RightKullbackLeiblerStdScalar(Function):
    """
    Scalar standard deviation with respect to the left Kullback-Leibler divergence
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scalar standard deviation with respect to the right Kullback-Leibler divergence

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
        n_matrices = torch.prod(torch.tensor(data.shape[:-2]))
        n_features = data.shape[-1]
        L_G = torch.linalg.cholesky(reference_point)
        L_data = torch.linalg.cholesky(data)
        data_inv = torch.cholesky_inverse(L_data)
        trace = torch.einsum("...ik,ki->", data_inv, reference_point)
        logdet_data = 2 * L_data.diagonal(dim1=-2, dim2=-1).log().sum()
        logdet_G = 2 * L_G.diagonal(dim1=-2, dim2=-1).log().sum()
        # clamp to avoid small numerical errors yielding small negative variance
        std_scalar = torch.sqrt(
            torch.clamp(
                (trace + logdet_data) / n_matrices - logdet_G - n_features, min=0.0
            )
        )
        ctx.n_matrices = n_matrices
        ctx.save_for_backward(data, reference_point, L_G, data_inv, std_scalar)
        return std_scalar

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the scalar standard deviation with respect to the right Kullback-Leibler divergence

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
        data, reference_point, L_G, data_inv, std_scalar = ctx.saved_tensors
        G_inv = torch.cholesky_inverse(L_G)
        grad_input_data = (
            grad_output
            * data_inv
            @ (data - reference_point)
            @ data_inv
            / 2
            / std_scalar
            / n_matrices
        )
        grad_input_G = (
            grad_output * (arithmetic_mean(data_inv) - G_inv) / 2 / std_scalar
        )
        return grad_input_data, grad_input_G
