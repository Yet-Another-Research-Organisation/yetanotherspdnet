import torch
from torch.autograd import Function

from yetanotherspdnet.functions.spd_linalg import symmetrize


def sample_covariance(data: torch.Tensor, assume_centered: bool = True) -> torch.Tensor:
    """
    Sample covariance matrix

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Batch of data

    assume_centered : bool, optional
        Whether to recenter data. Default is True

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Batch of sample covariance matrices
    """
    if assume_centered:
        X = data
        n_samples = X.shape[-2]
    else:
        X = data - data.mean(dim=-2, keepdim=True)
        n_samples = X.shape[-2] - 1
    return symmetrize(X.transpose(-2, -1) @ X / n_samples)


class SampleCovariance(Function):
    """
    Sample covariance matrix
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, assume_centered: bool = True) -> torch.Tensor:
        """
        Forward pass of the sample covariance matrix

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_samples, n_features)
            Batch of data

        assume_centered : bool, optional
            Whether to recenter data. Default is True

        Returns
        -------
        covariance : torch.Tensor of shape (..., n_features, n_features)
            Batch of sample covariance matrices
        """
        if assume_centered:
            X = data
            n_samples = X.shape[-2]
        else:
            X = data - data.mean(dim=-2, keepdim=True)
            n_samples = X.shape[-2] - 1
        ctx.n_samples = n_samples
        ctx.save_for_backward(X)
        return symmetrize(X.transpose(-2, -1) @ X / n_samples)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass of the sample covariance matrix

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient with respect to sample covariance matrices

        Returns
        -------
        grad_input: torch.Tensor of shape (..., n_samples, n_features)
            Gradient with respect to input batch of data
        """
        n_samples = ctx.n_samples
        (X,) = ctx.saved_tensors
        return 2 * X @ grad_output / n_samples, None


def ledoit_wolf_covariance(
    data: torch.Tensor, assume_centered: bool = True
) -> torch.Tensor:
    """
    Ledoit-Wolf covariance estimator

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)
        Batch of data

    assume_centered : bool, optional
        Whether to recenter data. Default is True

    Returns
    -------
    covariance : torch.Tensor of shape (..., n_features, n_features)
        Batch of Ledoit-Wolf covariance matrices
    """
    n_features = data.shape[-1]
    if assume_centered:
        X = data
        n_samples = data.shape[-2]
    else:
        X = data - data.mean(dim=-2, keepdim=True)
        n_samples = X.shape[-2] - 1
    # Sample covariance matrix
    SCM = symmetrize(X.transpose(-2, -1) @ X / n_samples)
    # Target matrix
    nu = torch.sum(torch.diagonal(SCM, dim1=-1, dim2=-2), axis=-1) / n_features
    Target = torch.diag_embed(nu.unsqueeze(-1) * torch.ones(SCM.shape[:-1]))
    # Shrinkage parameter
    row_norms_squared = torch.sum(X**2, dim=-1)
    sum_fourth_powers = torch.sum(row_norms_squared**2, dim=-1)
    trace_SCM_squared = torch.sum(SCM**2, dim=(-2, -1))
    kappa = sum_fourth_powers / (n_samples**2) - trace_SCM_squared / n_samples
    tau = torch.norm(SCM - Target, dim=(-1, -2), p="fro") ** 2
    alpha = torch.clamp(kappa / tau, 0, 1)

    return (1 - alpha[..., None, None]) * SCM + alpha[..., None, None] * Target


class LedoitWolfCovariance(Function):
    """
    Ledoit-Wolf covariance estimator
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, assume_centered: bool = True):
        """
        Forward pass of the Ledoit-Wolf covariance estimator

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_samples, n_features)
            Batch of data

        assume_centered : bool, optional
            Whether to recenter data. Default is True

        Returns
        -------
        covariance : torch.Tensor of shape (..., n_features, n_features)
            Batch of Ledoit-Wolf covariance matrices
        """
        n_features = data.shape[-1]
        if assume_centered:
            X = data
            n_samples = data.shape[-2]
        else:
            X = data - data.mean(dim=-2, keepdim=True)
            n_samples = X.shape[-2] - 1
        # Sample covariance matrix
        SCM = symmetrize(X.transpose(-2, -1) @ X / n_samples)
        # Target matrix
        nu = torch.sum(torch.diagonal(SCM, dim1=-1, dim2=-2), axis=-1) / n_features
        Target = torch.diag_embed(
            nu.unsqueeze(-1)
            * torch.ones(SCM.shape[:-1], device=SCM.device, dtype=SCM.dtype)
        )
        # Shrinkage parameter
        row_norms_squared = torch.sum(X**2, dim=-1)
        sum_fourth_powers = torch.sum(row_norms_squared**2, dim=-1)
        trace_SCM_squared = torch.sum(SCM**2, dim=(-2, -1))
        kappa = sum_fourth_powers / (n_samples**2) - trace_SCM_squared / n_samples
        tau = torch.norm(SCM - Target, dim=(-1, -2), p="fro") ** 2
        alpha = torch.clamp(kappa / tau, 0, 1)

        ctx.assume_centered = assume_centered
        ctx.n_samples = n_samples
        ctx.n_features = n_features
        ctx.ones_shape = SCM.shape[:-1]
        ctx.save_for_backward(X, SCM, Target, nu, alpha, kappa, tau)
        return (1 - alpha[..., None, None]) * SCM + alpha[..., None, None] * Target

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass of the Ledoit-Wolf covariance estimator

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient with respect to Ledoit-Wolf covariance matrices

        Returns
        -------
        grad_input: torch.Tensor of shape (..., n_samples, n_features)
            Gradient with respect to input batch of data
        """
        assume_centered = ctx.assume_centered
        n_samples = ctx.n_samples
        n_features = ctx.n_features
        ones_shape = ctx.ones_shape
        X, SCM, Target, nu, alpha, kappa, tau = ctx.saved_tensors
        Identity = torch.diag_embed(
            torch.ones(ones_shape, device=X.device, dtype=X.dtype)
        )
        trace_grad_output = torch.sum(
            torch.diagonal(grad_output, dim1=-2, dim2=-1), dim=-1
        )
        SCM_Target = SCM - Target
        trace_SCM_Target = torch.sum(
            torch.diagonal(SCM_Target, dim1=-2, dim2=-1), dim=-1
        )
        trace_SCMgrad_output = torch.sum(
            torch.diagonal(SCM @ grad_output, dim1=-2, dim2=-1), dim=-1
        )
        grad_SCM = (1 - alpha[..., None, None]) * grad_output + (
            alpha * trace_grad_output
        )[..., None, None] * Identity / n_features
        grad_alpha_scaling = trace_grad_output * nu - trace_SCMgrad_output
        grad_alpha_clamp = (alpha > 0).to(alpha.dtype) * (alpha < 1).to(alpha.dtype)
        grad_alpha_SCM = -2 * SCM / tau[..., None, None] / n_samples - 2 * kappa[
            ..., None, None
        ] * (SCM_Target - trace_SCM_Target[..., None, None] * Identity) / (
            tau[..., None, None] ** 2
        )
        grad_alpha_SCM = (grad_alpha_clamp * grad_alpha_scaling)[
            ..., None, None
        ] * grad_alpha_SCM
        grad_alpha_X = (
            4
            * ((X * X).sum(axis=-1))[..., None]
            * X
            / tau[..., None, None]
            / n_samples**2
        )
        if not assume_centered:
            grad_alpha_X = grad_alpha_X - grad_alpha_X.mean(dim=-2, keepdim=True)
        grad_alpha_X = (grad_alpha_clamp * grad_alpha_scaling)[
            ..., None, None
        ] * grad_alpha_X
        grad_SCM = grad_SCM + grad_alpha_SCM
        return (
            2 * X @ grad_SCM / n_samples + grad_alpha_X,
            None,
        )


def sample_variance(data: torch.Tensor, assume_centered: bool = True) -> torch.Tensor:
    """
    Sample variance vector

    Parameters
    ----------
    data : torch.Tensor of shape (..., n_samples, n_features)FR7615135005000423745813470
        Batch of data

    assume_centered : bool, optional
        Whether to recenter data. Default is True

    Returns
    -------
    variance : torch.Tensor of shape (..., n_features)
        Batch of sample variance vectors
    """
    if assume_centered:
        X = data
        n_samples = X.shape[-2]
    else:
        X = data - data.mean(dim=-2, keepdim=True)
        n_samples = X.shape[-2] - 1
    return torch.sum(X * X, dim=-2) / n_samples


class SampleVariance(Function):
    """
    Sample variance vector
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, assume_centered: bool = True) -> torch.Tensor:
        """
        Forwad pass of the sample variance vector estimator

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data : torch.Tensor of shape (..., n_samples, n_features)
            Batch of data

        assume_centered : bool, optional
            Whether to recenter data. Default is True

        Returns
        -------
        variance : torch.Tensor of shape (..., n_features)
            Batch of sample variance vectors
        """
        if assume_centered:
            X = data
            n_samples = X.shape[-2]
        else:
            X = data - data.mean(dim=-2, keepdim=True)
            n_samples = X.shape[-2] - 1
        ctx.n_samples = n_samples
        ctx.save_for_backward(X)
        return torch.sum(X * X, dim=-2) / n_samples

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass of the sample variance vector estimator

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features)
            Gradient with respect to the sample variance vector

        Returns
        -------
        grad_input: torch.Tensor of shape (..., n_samples, n_features)
            Gradient with respect to input batch of data
        """
        n_samples = ctx.n_samples
        (X,) = ctx.saved_tensors
        return 2 * grad_output.unsqueeze(-2) * X / n_samples, None
