import torch


def sqrt_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of the square root function

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_sqrt_deriv : torch.Tensor
        Derivative of sqrt of x
    """
    return 0.5 / torch.sqrt(x)


def inv_sqrt(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the square root

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv_sqrt : torch.Tensor
        Inverse sqrt of x
    """
    return 1 / torch.sqrt(x)


def inv_sqrt_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of the inverse of the square root

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv_sqrt_deriv : torch.Tensor
        Derivative of the inverse sqrt of x
    """
    return -0.5 / torch.pow(x, 1.5)


def inv(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse function

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv : torch.Tensor
        Inverse of x
    """
    return 1 / x


def scaled_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Scaled SoftPlus function.
    It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
    f'(x) -> 1 as x -> +inf

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus : torch.Tensor
        SoftPlus of x
    """
    return torch.log2(
        1.0 + torch.pow(torch.tensor(2.0, dtype=x.dtype, device=x.device), x)
    )


def scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of the scaled SoftPlus function

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus_deriv : torch.Tensor
        Derivative of SoftPlus of x
    """
    return 1 / (1.0 + torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), -x))


def inv_scaled_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the scaled SoftPlus function

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv_softplus : torch.Tensor
        Inverse of SoftPlus of x
    """
    return torch.log2(
        torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), x) - 1.0
    )


def inv_scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of the inverse of the scaled SoftPlus function

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus_deriv : torch.Tensor
        Derivative of the inverse SoftPlus of x
    """
    return 1 / (1.0 - torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), -x))
