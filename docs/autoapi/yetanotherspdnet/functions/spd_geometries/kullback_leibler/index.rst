yetanotherspdnet.functions.spd_geometries.kullback_leibler
==========================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.kullback_leibler


Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.kullback_leibler.EuclideanGeodesic
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.ArithmeticMean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.LeftKullbackLeiblerStdScalar
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.HarmonicCurve
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.HarmonicMean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.RightKullbackLeiblerStdScalar


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.kullback_leibler.euclidean_geodesic
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.arithmetic_mean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.left_kullback_leibler_std_scalar
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_curve
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_mean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler.right_kullback_leibler_std_scalar


Module Contents
---------------

.. py:function:: euclidean_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Euclidean geodesic:
   (1-t)*point1 + t*point2

   :param point1: Symmetric matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: Symmetric matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- Symmetric matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: EuclideanGeodesic(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Euclidean geodesic of a batch of symmetric matrices


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the Euclidean geodesic

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: Symmetric matrices
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param point2: Symmetric matrices
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param t: parameter on the path, should be in [0,1]
      :type t: :py:class:`float | torch.Tensor`

      :returns: **point** -- Symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]
      :staticmethod:


      Backward pass of the Euclidean geodesic

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output:
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`)
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`)



.. py:function:: arithmetic_mean(data: torch.Tensor) -> torch.Tensor

   Arithmetic mean of a batch of symmetric matrices

   :param data: Batch of symmetric matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **mean** -- Arithmetic mean
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:class:: ArithmeticMean(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Arithmetic mean


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the arithmetic mean of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of symmetric matrices. The mean is computed along ... axes
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **mean** -- Arithmetic mean
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the arithmetic mean of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the arithmetic mean
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: left_kullback_leibler_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation with respect to the left Kullback-Leibler divergence

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: LeftKullbackLeiblerStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation with respect to the left Kullback-Leibler divergence


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scalar standard deviation with respect to the left Kullback-Leibler divergence

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param reference_point: SPD matrix (some kind of mean of data)
      :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **scalar_std** -- scalar standard deviation
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the scalar standard deviation with respect to the left Kullback-Leibler divergence

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output of the scalar standard deviation Function
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input data
                * **grad_input_reference_point** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input reference point



.. py:function:: harmonic_curve(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Curve for adaptive harmonic mean computation:
   ((1-t)*point1^{-1} + t*point2^{-1})^{-1}

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: HarmonicCurve(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Harmonic curve of two batches of SPD matrices


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the harmonic curve

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: SPD matrices
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param point2: SPD matrices
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param t: parameter on the path, should be in [0,1]
      :type t: :py:class:`float | torch.Tensor`

      :returns: **point** -- SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]
      :staticmethod:


      Backward pass of the harmonic curve

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output:
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`)
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`)



.. py:function:: harmonic_mean(data: torch.Tensor) -> torch.Tensor

   Harmonic mean of a batch of SPD matrices

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **mean** -- Harmonic mean
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:class:: HarmonicMean(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Harmonic mean


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the harmonic mean of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices. The mean is computed along ... axes
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **mean** -- Harmonic mean
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the harmonic mean of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the harmonic mean
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: right_kullback_leibler_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation with respect to the left Kullback-Leibler divergence

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: RightKullbackLeiblerStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation with respect to the left Kullback-Leibler divergence


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scalar standard deviation with respect to the right Kullback-Leibler divergence

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param reference_point: SPD matrix (some kind of mean of data)
      :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **scalar_std** -- scalar standard deviation
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the scalar standard deviation with respect to the right Kullback-Leibler divergence

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output of the scalar standard deviation Function
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input data
                * **grad_input_reference_point** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input reference point



