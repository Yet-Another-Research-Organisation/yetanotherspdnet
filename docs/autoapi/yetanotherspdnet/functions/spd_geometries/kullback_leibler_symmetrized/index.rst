yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized
======================================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized

.. autoapi-nested-parse::

   Symmetrized KL geometry: GAH curves/means and adaptive geodesic with learnable parameter.



Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.SymmetrizedKullbackLeiblerStdScalar
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.AdaptiveGeometricArithmeticHarmonicGeodesic


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.geometric_euclidean_harmonic_curve
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.GeometricEuclideanHarmonicCurve
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.geometric_arithmetic_harmonic_mean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.GeometricArithmeticHarmonicMean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.symmetrized_kullback_leibler_std_scalar
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.adaptive_geometric_arithmetic_harmonic_geodesic
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.adaptive_geometric_arithmetic_harmonic_mean
   yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized.AdaptiveGeometricArithmeticHarmonicMean


Module Contents
---------------

.. py:function:: geometric_euclidean_harmonic_curve(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Curve corresponding to the geometric mean of the Euclidean geodesic and
   the harmonic curve (the GAH — Geometric-Arithmetic-Harmonic — curve).

   .. math::

       \gamma(t) = G\big(E(t),\ H(t)\big), \qquad
       E(t) = (1-t)P_1 + tP_2, \qquad
       H(t) = \big((1-t)P_1^{-1} + tP_2^{-1}\big)^{-1}

   where :math:`E` is the Euclidean geodesic
   (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.euclidean_geodesic`),
   :math:`H` is the harmonic curve
   (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_curve`),
   and :math:`G` is the affine-invariant geometric mean of two points
   (:func:`~yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_mean_2points`).

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: GeometricEuclideanHarmonicCurve(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Curve corresponding to the geometric mean of the Euclidean geodesic and the harmonic curve

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: geometric_arithmetic_harmonic_mean(data: torch.Tensor) -> torch.Tensor

   Geometric mean of the arithmetic and harmonic means of a batch of SPD
   matrices (GAH mean).

   .. math::

       \bar{P}_{GAH} = G\big(\bar{P}_{arith},\ \bar{P}_{harm}\big)

   with :math:`\bar{P}_{arith}` the arithmetic mean
   (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.arithmetic_mean`),
   :math:`\bar{P}_{harm}` the harmonic mean
   (:func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.harmonic_mean`),
   and :math:`G` the affine-invariant geometric mean of two points.

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param return_arithmetic_harmonic: Whether to also return arithmetic and harmonic means (for adptative mean update reasons), by default False
   :type return_arithmetic_harmonic: :py:class:`bool`, *optional*

   :returns: **mean** -- Geometric mean of the arithmetic and harmonic means
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: GeometricArithmeticHarmonicMean(data: torch.Tensor) -> torch.Tensor

   Geometric mean of the arithmetic and harmonic means

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **mean** -- Geometric mean of the arithmetic and harmonic means
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: symmetrized_kullback_leibler_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation with respect to the symmetrized
   Kullback-Leibler (Jeffreys) divergence.

   .. math::

       \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}
           \frac{\operatorname{tr}(G^{-1}P_i) + \operatorname{tr}(P_i^{-1}G)}{2}
           - n

   the average Jeffreys divergence between the batch's covariances and the
   reference point :math:`G` — symmetrizing
   :func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.left_kullback_leibler_std_scalar`
   and
   :func:`~yetanotherspdnet.functions.spd_geometries.kullback_leibler.right_kullback_leibler_std_scalar`
   (the :math:`\log\det` terms cancel out).

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: SymmetrizedKullbackLeiblerStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence

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


      Backward pass of the scalar standard deviation with respect to the symmetrized Kullback-Leibler divergence

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output of the scalar standard deviation Function
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input data
                * **grad_input_reference_point** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- gradient of the loss with respect to the input reference point



.. py:function:: adaptive_geometric_arithmetic_harmonic_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Adaptive (AdaptiveGAH) geodesic between harmonic (point1) and arithmetic
   (point2) means, with a learnable position :math:`t` on the path.

   .. math::

       \gamma(t) = P_1^{1/2}
           \big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{t}
           P_1^{1/2}

   This is mathematically identical to :func:`~yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_geodesic`
   — the point of this alias is that here :math:`t` is treated as a
   learnable parameter (interpolating between the harmonic mean at
   :math:`t=0` and the arithmetic mean at :math:`t=1`) rather than a fixed
   schedule value.

   :param point1: SPD matrices (typically harmonic mean)
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices (typically arithmetic mean)
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the path, should be in [0,1].
             t=0.5 gives the standard geometric mean.
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: AdaptiveGeometricArithmeticHarmonicGeodesic(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Adaptive geodesic between harmonic and arithmetic means with learnable parameter t.

   This is mathematically identical to the affine-invariant geodesic.
   It delegates to AffineInvariantGeodesic and includes gradient with respect to t
   for learning.


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor, t: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the adaptive GAH geodesic.

      Delegates to AffineInvariantGeodesic.

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: SPD matrices (harmonic mean)
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param point2: SPD matrices (arithmetic mean)
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param t: Learnable parameter on the geodesic, should be in [0,1]
      :type t: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: **point** -- SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the adaptive GAH geodesic.

      Uses the same gradient logic as AffineInvariantGeodesic,
      with the addition of grad_t using broadcasting pattern.

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point1 (harmonic mean)
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`) -- Gradient of the loss with respect to point2 (arithmetic mean)
                * **grad_t** (:py:class:`torch.Tensor` of :py:class:`shape ()`) -- Gradient of the loss with respect to t



.. py:function:: adaptive_geometric_arithmetic_harmonic_mean(data: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Adaptive geometric mean (AdaptiveGAH) of the arithmetic and harmonic
   means of a batch of SPD matrices, with a learnable parameter :math:`t`.

   .. math::

       \bar{P}_{t} = \gamma_{AI}\big(\bar{P}_{harm}, \bar{P}_{arith}, t\big)

   where :math:`\gamma_{AI}` is the affine-invariant geodesic
   (:func:`adaptive_geometric_arithmetic_harmonic_geodesic`). Unlike
   :func:`geometric_arithmetic_harmonic_mean` (fixed at :math:`t=0.5`),
   :math:`t` here can be learned, letting the model pick where between the
   harmonic and arithmetic means the effective "center" of BatchNorm sits.

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: parameter on the geodesic between harmonic (t=0) and arithmetic (t=1) means.
             t=0.5 gives the standard geometric mean of arithmetic and harmonic means.
   :type t: :py:class:`float | torch.Tensor`

   :returns: **mean** -- Adaptive geometric mean of the arithmetic and harmonic means
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: AdaptiveGeometricArithmeticHarmonicMean(data: torch.Tensor, t: torch.Tensor) -> torch.Tensor

   Adaptive geometric mean of the arithmetic and harmonic means with learnable t
   (using custom autograd Function for efficient gradients)

   :param data: Batch of SPD matrices. The mean is computed along ... axes
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: Learnable parameter on the geodesic between harmonic (t=0) and arithmetic (t=1) means.
             t=0.5 gives the standard geometric mean of arithmetic and harmonic means.
   :type t: :py:class:`torch.Tensor` of :py:class:`shape ()`

   :returns: **mean** -- Adaptive geometric mean of the arithmetic and harmonic means
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


