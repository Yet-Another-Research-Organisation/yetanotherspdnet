"""Scalar functions applied element-wise to eigenvalues: sqrt, inv_sqrt, softplus, and derivatives."""

import math

import torch


def sqrt_derivative(x: torch.Tensor) -> torch.Tensor:
    r"""
    Derivative of the square root function.

    .. math:: \frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}}

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
    r"""
    Inverse of the square root.

    .. math:: f(x) = \frac{1}{\sqrt{x}} = x^{-1/2}

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
    r"""
    Derivative of the inverse of the square root.

    .. math:: \frac{d}{dx} x^{-1/2} = -\frac{1}{2} x^{-3/2}

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
    r"""
    Inverse function.

    .. math:: f(x) = \frac{1}{x}

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
    r"""
    Scaled SoftPlus function.

    .. math:: f(x) = \log_2\big(1 + 2^{x}\big)

    Base-2 (rather than the usual base-:math:`e`) SoftPlus, chosen so that
    :math:`f(0) = 1`, :math:`f(x) \to 0` as :math:`x \to -\infty`, and
    :math:`f'(x) \to 1` as :math:`x \to +\infty`. Used (via
    :func:`~yetanotherspdnet.functions.spd_linalg.scaled_softplus_symmetric`)
    to reparametrize eigenvalues so BiMap weights stay strictly positive.

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus : torch.Tensor
        SoftPlus of x
    """
    return torch.log2(1.0 + torch.exp2(x))


def scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor:
    r"""
    Derivative of the scaled SoftPlus function.

    .. math:: f'(x) = \sigma(x \ln 2)

    where :math:`\sigma` is the logistic sigmoid.

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus_deriv : torch.Tensor
        Derivative of SoftPlus of x
    """
    return torch.sigmoid(x * math.log(2))


def inv_scaled_softplus(x: torch.Tensor) -> torch.Tensor:
    r"""
    Inverse of the scaled SoftPlus function.

    .. math:: f^{-1}(x) = \log_2\big(2^{x} - 1\big)

    Inverse of :func:`scaled_softplus`.

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv_softplus : torch.Tensor
        Inverse of SoftPlus of x
    """
    return torch.log2(torch.exp2(x) - 1.0)


def inv_scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor:
    r"""
    Derivative of the inverse of the scaled SoftPlus function.

    .. math:: (f^{-1})'(x) = \frac{1}{1 - 2^{-x}}

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_softplus_deriv : torch.Tensor
        Derivative of the inverse SoftPlus of x
    """
    return 1 / (1.0 - torch.exp2(-x))
