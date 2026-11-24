"""Scalar functions applied element-wise to eigenvalues, and their derivatives."""

import math

import torch


def sqrt_derivative(x: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the derivative of the square root function.

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
    Compute the derivative of the inverse of the square root.

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


def inv_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of the inverse function.

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    x_inv_deriv : torch.Tensor
        Derivative of the inverse of x
    """
    return -1 / torch.square(x)


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
    Compute the derivative of the scaled SoftPlus function.

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
    Compute the derivative of the inverse of the scaled SoftPlus function.

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


def pow_exponent_derivative(
    x: torch.Tensor, exponent: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the derivative of the power function with respect to its exponent.

    .. math:: \frac{\partial}{\partial t} x^{t} = x^{t} \log x

    Used for the gradient of matrix powers and of the affine-invariant geodesic
    with respect to the exponent :math:`t` (applied to eigenvalues).

    Parameters
    ----------
    x : torch.Tensor
        Scalar or array of positive scalars.

    exponent : float | torch.Tensor
        Power exponent :math:`t`.

    Returns
    -------
    x_pow_deriv : torch.Tensor
        :math:`x^t \log x`
    """
    return torch.pow(x, exponent) * torch.log(x)


def zero_where_zero_std(grad: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    r"""
    Zero a gradient where the scalar standard deviation it was divided by is 0.

    The scalar standard deviations :math:`\sigma = \sqrt{v}` are not
    differentiable at :math:`v = 0` (all the matrices equal the reference
    point, or a variance clamped at 0): their hand-written backwards divide by
    :math:`\sigma`, which gives 0/0 = NaN there. They return the zero
    subgradient instead.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient computed with a division by std

    std : torch.Tensor of shape ()
        Scalar standard deviation

    Returns
    -------
    grad : torch.Tensor
        grad where std > 0, zeros where std == 0
    """
    return torch.where(std > 0, grad, torch.zeros_like(grad))


# ---------------------------------------------------------------------------
# Divided differences (f(a) - f(b)) / (a - b) without cancellation
# ---------------------------------------------------------------------------
# Each kernel takes broadcastable eigenvalue tensors a, b and returns the divided
# difference of its function. It is symmetric in (a, b) by construction (it works
# on lo = min(a, b), hi = max(a, b)), accurate to a few ulps for any gap
# a - b != 0, and may be undefined (0 / 0) at exact ties, which
# spd_linalg._aux_eigh_operation_grad replaces with the derivative.


def _sorted_pair(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lo = min(a, b), hi = max(a, b) and the gap hi - lo >= 0."""
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return lo, hi, hi - lo


def sqrt_divided_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Divided difference of the square root, :math:`1 / (\sqrt a + \sqrt b)`.

    Exact also at ties (:math:`1 / (2\sqrt a)`).

    Parameters
    ----------
    a, b : torch.Tensor
        Positive eigenvalues (broadcastable)

    Returns
    -------
    torch.Tensor
        :math:`(\sqrt a - \sqrt b) / (a - b)`
    """
    return 1 / (torch.sqrt(a) + torch.sqrt(b))


def inv_sqrt_divided_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Divided difference of :math:`x^{-1/2}`, :math:`-1 / (\sqrt a \sqrt b (\sqrt a + \sqrt b))`.

    Exact also at ties.

    Parameters
    ----------
    a, b : torch.Tensor
        Positive eigenvalues (broadcastable)

    Returns
    -------
    torch.Tensor
        :math:`(a^{-1/2} - b^{-1/2}) / (a - b)`
    """
    sqrt_a, sqrt_b = torch.sqrt(a), torch.sqrt(b)
    return -1 / (sqrt_a * sqrt_b * (sqrt_a + sqrt_b))


def inv_divided_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Divided difference of :math:`x^{-1}`, :math:`-1 / (ab)`.

    Exact also at ties.

    Parameters
    ----------
    a, b : torch.Tensor
        Nonzero eigenvalues (broadcastable)

    Returns
    -------
    torch.Tensor
        :math:`(a^{-1} - b^{-1}) / (a - b)`
    """
    return -1 / (a * b)


def log_divided_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Divided difference of the logarithm, :math:`\operatorname{log1p}(d / l) / d`.

    With :math:`l = \min(a, b)` and :math:`d = |a - b|`; 0 / 0 at ties.

    Parameters
    ----------
    a, b : torch.Tensor
        Positive eigenvalues (broadcastable)

    Returns
    -------
    torch.Tensor
        :math:`(\log a - \log b) / (a - b)`
    """
    lo, _, gap = _sorted_pair(a, b)
    return torch.log1p(gap / lo) / gap


def exp_divided_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Divided difference of the exponential, :math:`e^{l} \operatorname{expm1}(d) / d`.

    With :math:`l = \min(a, b)` and :math:`d = |a - b|`; 0 / 0 at ties.

    Parameters
    ----------
    a, b : torch.Tensor
        Eigenvalues (broadcastable)

    Returns
    -------
    torch.Tensor
        :math:`(e^a - e^b) / (a - b)`
    """
    lo, _, gap = _sorted_pair(a, b)
    return torch.exp(lo) * torch.expm1(gap) / gap


def pow_divided_difference(
    a: torch.Tensor, b: torch.Tensor, exponent: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Divided difference of :math:`x^t`, :math:`l^t \operatorname{expm1}(t \operatorname{log1p}(d / l)) / d`.

    With :math:`l = \min(a, b)` and :math:`d = |a - b|`; 0 / 0 at ties.

    Parameters
    ----------
    a, b : torch.Tensor
        Positive eigenvalues (broadcastable)

    exponent : float | torch.Tensor
        Power exponent :math:`t`

    Returns
    -------
    torch.Tensor
        :math:`(a^t - b^t) / (a - b)`
    """
    lo, _, gap = _sorted_pair(a, b)
    return torch.pow(lo, exponent) * torch.expm1(exponent * torch.log1p(gap / lo)) / gap
