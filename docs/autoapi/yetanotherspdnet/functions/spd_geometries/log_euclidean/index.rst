yetanotherspdnet.functions.spd_geometries.log_euclidean
=======================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.log_euclidean

.. autoapi-nested-parse::

   Log-Euclidean Riemannian geometry: geodesic, mean, and standard deviation.



Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.log_euclidean.LogEuclideanStdScalar


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.log_euclidean.log_euclidean_geodesic
   yetanotherspdnet.functions.spd_geometries.log_euclidean.LogEuclideanGeodesic
   yetanotherspdnet.functions.spd_geometries.log_euclidean.log_euclidean_mean
   yetanotherspdnet.functions.spd_geometries.log_euclidean.LogEuclideanMean
   yetanotherspdnet.functions.spd_geometries.log_euclidean.log_euclidean_std_scalar


Module Contents
---------------

.. py:function:: log_euclidean_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Log-Euclidean geodesic between two SPD matrices.

   .. math::

       \gamma(t) = \exp\big((1-t)\,\log(P_1) + t\,\log(P_2)\big)

   where :math:`\log` and :math:`\exp` are the matrix logarithm and
   exponential (:func:`~yetanotherspdnet.functions.spd_linalg.logm_SPD`,
   :func:`~yetanotherspdnet.functions.spd_linalg.expm_symmetric`). This is
   simply the Euclidean geodesic in the tangent space obtained by taking
   the matrix log, mapped back to the SPD manifold.

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: LogEuclideanGeodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Log-Euclidean geodesic between two batches of SPD matrices

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: log_euclidean_mean(data: torch.Tensor) -> torch.Tensor

   Log-Euclidean mean of a batch of SPD matrices.

   .. math::

       \bar{P} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \log(P_i)\right)

   i.e. the arithmetic mean of the matrix logarithms, mapped back to the
   SPD manifold with the matrix exponential. Unlike the affine-invariant
   or Bures-Wasserstein means, this has a closed form (no fixed-point
   iteration needed).

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **mean** -- Log-Euclidean mean
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: LogEuclideanMean(data: torch.Tensor) -> torch.Tensor

   Log-Euclidean mean

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **mean** -- Log-Euclidean mean
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: log_euclidean_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation with respect to the Log-Euclidean distance.

   .. math::

       \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
           \lVert \log(P_i) - \log(G) \rVert_F^2}

   where :math:`G` is the reference point (typically the Log-Euclidean
   mean) and :math:`\lVert \cdot \rVert_F` is the Frobenius norm — the
   Log-Euclidean distance is simply the Euclidean distance between matrix
   logarithms.

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: LogEuclideanStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation with respect to the Log-Euclidean distance


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scalar standard deviation with respect to the Log-Euclidean distance

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



