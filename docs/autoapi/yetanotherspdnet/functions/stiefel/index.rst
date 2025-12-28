yetanotherspdnet.functions.stiefel
==================================

.. py:module:: yetanotherspdnet.functions.stiefel


Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.stiefel.StiefelProjectionTangentOrthogonal
   yetanotherspdnet.functions.stiefel.StiefelProjectionPolar
   yetanotherspdnet.functions.stiefel.StiefelProjectionQR


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.stiefel.stiefel_projection_polar
   yetanotherspdnet.functions.stiefel.stiefel_projection_tangent_orthogonal
   yetanotherspdnet.functions.stiefel.stiefel_projection_qr
   yetanotherspdnet.functions.stiefel.stiefel_differential_projection_qr
   yetanotherspdnet.functions.stiefel.stiefel_adjoint_differential_projection_qr


Module Contents
---------------

.. py:function:: stiefel_projection_polar(point: torch.Tensor) -> torch.Tensor

   Projection from the ambient space onto the Stiefel manifold based on the polar decomposition

   :param point: Rectangular matrix (with n_out <= n_in)
   :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

   :returns: **projected_point** -- Orthogonal matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`


.. py:function:: stiefel_projection_tangent_orthogonal(vector: torch.Tensor, point: torch.Tensor) -> torch.Tensor

   Orthogonal projection from the ambient space onto the tangent space
   of the Stiefel manifold at point

   Note that this also corresponds to both the differential and differential adjoint
   of projection_stiefel_polar

   :param vector: Rectangular matrix (direction)
   :type vector: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
   :param point: Orthogonal matrix (with n_out <= n_in)
   :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

   :returns: **tangent_vector** -- Rectangular matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`


.. py:class:: StiefelProjectionTangentOrthogonal(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Orthogonal projection from the ambient space onto the tangent space
   of the Stiefel manifold at point


   .. py:method:: forward(ctx, vector: torch.Tensor, point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the orthogonal projection from the ambient space onto
      the tangent space of the Stiefel manifold at point

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param vector: Rectangular matrix (direction)
      :type vector: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
      :param point: Orthogonal matrix (with n_out <= n_in)
      :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **tangent_vector** -- Rectangular matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]
      :staticmethod:


      Backward pass of the orthogonal projection onto the tangent space
      of the Stiefel manifold at point

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output tangent vector
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input rectangular matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



.. py:class:: StiefelProjectionPolar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Projection from the ambient space onto the Stiefel manifold based on polar decomposition


   .. py:method:: forward(ctx, point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the projection onto the Stiefel manifold

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point: Rectangular matrix (with n_out <= n_in)
      :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **projected_point** -- Orthogonal matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the projection onto the Stiefel manifold

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the projected orthogonal matrix
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input rectangular matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



.. py:function:: stiefel_projection_qr(point: torch.Tensor) -> torch.Tensor

   Projection from the ambient space onto the Stiefel manifold based on the QR decomposition

   :param point: Rectangular matrix (with n_out <= n_in)
   :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

   :returns: **projected_point** -- Orthogonal matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`


.. py:function:: stiefel_differential_projection_qr(vector: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor

   Differential of the projection on Stiefel based on QR decomposition

   :param vector: Rectangular matrix (direction)
   :type vector: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
   :param Q: Q factor of RQ decomposition of point (with n_out <= n_in)
   :type Q: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
   :param R: R factor of QR decomposition of point (upper triangular)
   :type R: :py:class:`torch.Tensor` of :py:class:`shape (n_out`, :py:class:`n_out)`

   :returns: **tangent_vector** -- Rectangular matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`


.. py:function:: stiefel_adjoint_differential_projection_qr(vector: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor

   Adjoint of the differential projection on Stiefel based on QR decomposition

   :param vector: Rectangular matrix (direction)
   :type vector: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
   :param Q: Q factor of RQ decomposition of point (with n_out <= n_in)
   :type Q: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`
   :param R: R factor of QR decomposition of point (upper triangular)
   :type R: :py:class:`torch.Tensor` of :py:class:`shape (n_out`, :py:class:`n_out)`

   :returns: **transformed_vector** -- Rectangular matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`


.. py:class:: StiefelProjectionQR(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Projection from the ambient space onto the Stiefel manifold based on QR decomposition


   .. py:method:: forward(ctx, point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the projection onto the Stiefel manifold

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point: Rectangular matrix (with n_out <= n_in)
      :type point: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **projected_point** -- Orthogonal matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the projection onto the Stiefel manifold

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the projected orthogonal matrix
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input rectangular matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



