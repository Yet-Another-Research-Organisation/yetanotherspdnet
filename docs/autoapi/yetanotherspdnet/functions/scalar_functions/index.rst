yetanotherspdnet.functions.scalar_functions
===========================================

.. py:module:: yetanotherspdnet.functions.scalar_functions

.. autoapi-nested-parse::

   Scalar functions applied element-wise to eigenvalues: sqrt, inv_sqrt, softplus, and derivatives.



Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.scalar_functions.sqrt_derivative
   yetanotherspdnet.functions.scalar_functions.inv_sqrt
   yetanotherspdnet.functions.scalar_functions.inv_sqrt_derivative
   yetanotherspdnet.functions.scalar_functions.inv
   yetanotherspdnet.functions.scalar_functions.scaled_softplus
   yetanotherspdnet.functions.scalar_functions.scaled_softplus_derivative
   yetanotherspdnet.functions.scalar_functions.inv_scaled_softplus
   yetanotherspdnet.functions.scalar_functions.inv_scaled_softplus_derivative


Module Contents
---------------

.. py:function:: sqrt_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the square root function.

   .. math:: \frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}}

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_sqrt_deriv** -- Derivative of sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_sqrt(x: torch.Tensor) -> torch.Tensor

   Inverse of the square root.

   .. math:: f(x) = \frac{1}{\sqrt{x}} = x^{-1/2}

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_sqrt** -- Inverse sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_sqrt_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the inverse of the square root.

   .. math:: \frac{d}{dx} x^{-1/2} = -\frac{1}{2} x^{-3/2}

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_sqrt_deriv** -- Derivative of the inverse sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv(x: torch.Tensor) -> torch.Tensor

   Inverse function.

   .. math:: f(x) = \frac{1}{x}

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv** -- Inverse of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: scaled_softplus(x: torch.Tensor) -> torch.Tensor

   Scaled SoftPlus function.

   .. math:: f(x) = \log_2\big(1 + 2^{x}\big)

   Base-2 (rather than the usual base-:math:`e`) SoftPlus, chosen so that
   :math:`f(0) = 1`, :math:`f(x) \to 0` as :math:`x \to -\infty`, and
   :math:`f'(x) \to 1` as :math:`x \to +\infty`. Used (via
   :func:`~yetanotherspdnet.functions.spd_linalg.scaled_softplus_symmetric`)
   to reparametrize eigenvalues so BiMap weights stay strictly positive.

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus** -- SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the scaled SoftPlus function.

   .. math:: f'(x) = \sigma(x \ln 2)

   where :math:`\sigma` is the logistic sigmoid.

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus_deriv** -- Derivative of SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_scaled_softplus(x: torch.Tensor) -> torch.Tensor

   Inverse of the scaled SoftPlus function.

   .. math:: f^{-1}(x) = \log_2\big(2^{x} - 1\big)

   Inverse of :func:`scaled_softplus`.

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_softplus** -- Inverse of SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the inverse of the scaled SoftPlus function.

   .. math:: (f^{-1})'(x) = \frac{1}{1 - 2^{-x}}

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus_deriv** -- Derivative of the inverse SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


