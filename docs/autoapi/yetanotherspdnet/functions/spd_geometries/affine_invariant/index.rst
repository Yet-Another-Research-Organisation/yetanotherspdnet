yetanotherspdnet.functions.spd_geometries.affine_invariant
==========================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.affine_invariant


Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantGeodesic
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMean2Points
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMeanIteration
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantStdScalar


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_geodesic
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_mean_2points
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_mean
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMean
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_std_scalar


Module Contents
---------------

.. py:function:: affine_invariant_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Affine-invariant geodesic:
   point1^{1/2} ( point1^{-1/2} point2 point1^{-1/2} )^t point1^{1/2}

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: AffineInvariantGeodesic(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Affine-invariant geodesic between two batches of SPD matrices


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor)
      :staticmethod:


      Forward pass of the affine-invariant geodesic between two batches of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: SPD matrices
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param point2: SPD matrices
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: **point** -- SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]
      :staticmethod:


      Backward pass of the affine-invariant geodesic

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the geometric mean of two SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point1
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point2



.. py:function:: affine_invariant_mean_2points(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor

   Affine-invariant (geometric) mean of two SPD matrices

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

   :returns: **mean** -- Geometric means of point1 and point2
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`


.. py:class:: AffineInvariantMean2Points(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Affine-invariant (geometric) mean of two SPD matrices


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the geometric mean of two SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: SPD matrices
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param point2: SPD matrices
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: **mean** -- Geometric means of point1 and point2
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the geometric mean of two SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the geometric mean of two SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point1
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point2



.. py:function:: affine_invariant_mean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor

   Affine-invariant (geometric) mean computed with fixed-point algorithm

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param n_iterations: Number of iterations to perform to estimate the geometric mean, by default 5
   :type n_iterations: :py:class:`int`

   :returns: **mean** -- SPD matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:class:: AffineInvariantMeanIteration(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   One iteration of the fixed-point algorithm computing the affine-invariant (geometric) mean


   .. py:method:: forward(ctx, mean_iterate: torch.Tensor, data: torch.Tensor, stepsize: float) -> torch.Tensor
      :staticmethod:


      Forward pass of one iteration of the fixed-point algorithm for the affine-invariant (geometric) mean

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param mean_iterate: Current iterate of the affine-invariant mean
      :type mean_iterate: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`
      :param data: Batch of SPD matrices. The mean is computed along ... axes
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param stepsize: step-size to stabilize the fixed-point algorithm
      :type stepsize: :py:class:`float`

      :returns: **mean_iterate_new** -- New iterate of the affine-invariant mean
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]
      :staticmethod:


      Backward pass of one iteration of the fixed-point algorithm for the affine-invariant (geometric) mean

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the new iterate of the affine-invariant mean
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (nfeatures`, :py:class:`nfeatures)`

      :returns: * **grad_input_mean** (:py:class:`torch.Tensor` of :py:class:`shape (nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to the current iterate of the affine-invariant mean
                * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the data at the current iterate



.. py:function:: AffineInvariantMean(data: torch.Tensor, n_iterations: int = 5) -> torch.Tensor

   Affine-invariant (geometric) mean computed with fixed-point algorithm

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param n_iterations: Number of iterations to perform to estimate the geometric mean.
                        Default is 10
   :type n_iterations: :py:class:`int`

   :returns: **mean** -- SPD matrix
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: affine_invariant_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation with respect to the affine-invariant distance

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: AffineInvariantStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation with respect to the affine-invariant distance


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scalar standard deviation with respect to the affine-invariant distance

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param reference_point: SPD matrix (some kind of mean of data)
      :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **scalar_std** -- scalar standard deviation
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the scalar standard deviation with respect to the affine-invariant distance

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output of the scalar standard deviation Function
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input data
                * **grad_input_reference_point** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input reference point



