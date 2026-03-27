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

   Derivative of the square root function

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_sqrt_deriv** -- Derivative of sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_sqrt(x: torch.Tensor) -> torch.Tensor

   Inverse of the square root

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_sqrt** -- Inverse sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_sqrt_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the inverse of the square root

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_sqrt_deriv** -- Derivative of the inverse sqrt of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv(x: torch.Tensor) -> torch.Tensor

   Inverse function

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv** -- Inverse of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: scaled_softplus(x: torch.Tensor) -> torch.Tensor

   Scaled SoftPlus function.
   It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
   f'(x) -> 1 as x -> +inf

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus** -- SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the scaled SoftPlus function

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus_deriv** -- Derivative of SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_scaled_softplus(x: torch.Tensor) -> torch.Tensor

   Inverse of the scaled SoftPlus function

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_inv_softplus** -- Inverse of SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


.. py:function:: inv_scaled_softplus_derivative(x: torch.Tensor) -> torch.Tensor

   Derivative of the inverse of the scaled SoftPlus function

   :param x: Scalar or array of scalars.
   :type x: :py:class:`torch.Tensor`

   :returns: **x_softplus_deriv** -- Derivative of the inverse SoftPlus of x
   :rtype: :py:class:`torch.Tensor`


