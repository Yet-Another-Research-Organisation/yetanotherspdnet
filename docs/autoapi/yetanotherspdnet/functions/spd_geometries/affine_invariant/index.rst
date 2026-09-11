yetanotherspdnet.functions.spd_geometries.affine_invariant
==========================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.affine_invariant

.. autoapi-nested-parse::

   Affine-invariant Riemannian geometry: geodesic, exp/log maps, mean, and standard deviation.



Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantGeodesic
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMean2Points
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMeanIteration
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantStdScalar
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantExp


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_geodesic
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_mean_2points
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_mean
   yetanotherspdnet.functions.spd_geometries.affine_invariant.AffineInvariantMean
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_std_scalar
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_exp
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_log
   yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_projx


Module Contents
---------------

.. py:function:: affine_invariant_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Affine-invariant geodesic between two SPD matrices.

   .. math::

       \gamma(t) = P_1^{1/2}
           \big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{t}
           P_1^{1/2}

   with :math:`t \in [0, 1]` (:math:`\gamma(0) = P_1`,
   :math:`\gamma(1) = P_2`). This is the geodesic for the affine-invariant
   Riemannian metric on the SPD manifold, invariant under congruence
   transformations :math:`P \mapsto A P A^\top` for any invertible
   :math:`A`.

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


   Affine-invariant geodesic between two batches of SPD matrices.

   Computes: point1^{1/2} (point1^{-1/2} point2 point1^{-1/2})^t point1^{1/2}

   Supports gradients with respect to point1, point2, and optionally t
   (when t is a tensor with requires_grad=True).


   .. py:method:: forward(ctx, point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor)
      :staticmethod:


      Forward pass of the affine-invariant geodesic between two batches of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param point1: SPD matrices
      :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param point2: SPD matrices
      :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`
      :param t: Parameter on the geodesic path, should be in [0, 1]
      :type t: :py:class:`float | torch.Tensor`

      :returns: **point** -- SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
      :staticmethod:


      Backward pass of the affine-invariant geodesic.

      Computes gradients with respect to point1, point2, and optionally t.

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)`

      :returns: * **grad_input1** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)` or :py:obj:`None`) -- Gradient of the loss with respect to point1
                * **grad_input2** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`nfeatures`, :py:class:`nfeatures)` or :py:obj:`None`) -- Gradient of the loss with respect to point2
                * **grad_t** (:py:class:`torch.Tensor` of :py:class:`shape ()` or :py:obj:`None`) -- Gradient of the loss with respect to t (only when t requires grad)



.. py:function:: affine_invariant_mean_2points(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor

   Affine-invariant (geometric) mean of two SPD matrices.

   .. math::

       G(P_1, P_2) = P_1^{1/2}
           \big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{1/2}
           P_1^{1/2}

   the midpoint (:math:`t=1/2`) of the affine-invariant geodesic between
   :math:`P_1` and :math:`P_2` (see :func:`affine_invariant_geodesic`).

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

   Affine-invariant (geometric/Fréchet) mean computed with a fixed-point
   (Karcher flow) algorithm.

   Starting from :math:`M_0 = I`, each iteration :math:`k` moves along the
   average tangent direction at :math:`M_k` and retracts back onto the
   manifold with the affine-invariant exponential map
   (:func:`affine_invariant_exp`):

   .. math::

       M_{k+1} = M_k^{1/2} \exp\!\left(\eta_k \cdot
           \frac{1}{N}\sum_{i=1}^{N} \log\big(M_k^{-1/2} P_i M_k^{-1/2}\big)
           \right) M_k^{1/2}

   with step size :math:`\eta_k = 0.95^k`. This converges to the unique
   minimizer of :math:`\sum_i d_{AI}(M, P_i)^2` for the affine-invariant
   distance :math:`d_{AI}`.

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

   Scalar standard deviation with respect to the affine-invariant distance.

   .. math::

       \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
           \big\lVert \log\big(G^{-1/2} P_i G^{-1/2}\big) \big\rVert_F^2}

   where :math:`G` is the reference point (typically the affine-invariant
   mean) — equivalently, the norm of :math:`\mathrm{Log}_G(P_i)` under the
   affine-invariant metric (:func:`affine_invariant_log`).

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



.. py:function:: affine_invariant_exp(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor

   Affine-invariant exponential map on the SPD manifold.

   .. math::

       \mathrm{Exp}_X(V) = X^{1/2} \exp\big(X^{-1/2} V X^{-1/2}\big) X^{1/2}

   Maps a tangent vector :math:`V` at base point :math:`X` (a symmetric
   matrix) to a point on the SPD manifold, by following the geodesic from
   :math:`X` in direction :math:`V` for unit time. Inverse of
   :func:`affine_invariant_log`.

   :param base: Base point(s) on the SPD manifold
   :type base: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`
   :param tangent: Tangent vector(s) at base (symmetric matrices)
   :type tangent: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

   :returns: **result** -- Point(s) on the SPD manifold
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`


.. py:class:: AffineInvariantExp(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Affine-invariant exponential map with manual backward.

   Exp_X(V) = X^{1/2} expm(X^{-1/2} V X^{-1/2}) X^{1/2}


   .. py:method:: forward(ctx, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the affine-invariant exponential map.

      :param ctx: Context for saving tensors for backward
      :type ctx: :py:class:`context`
      :param base: Base point(s) on the SPD manifold
      :type base: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`
      :param tangent: Tangent vector(s) at base (symmetric matrices)
      :type tangent: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

      :returns: **result** -- Point(s) on the SPD manifold
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]
      :staticmethod:


      Backward pass of the affine-invariant exponential map.

      :param ctx: Context with saved tensors
      :type ctx: :py:class:`context`
      :param grad_output: Gradient w.r.t. the output
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

      :returns: * **grad_base** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)` or :py:obj:`None`)
                * **grad_tangent** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)` or :py:obj:`None`)



.. py:function:: affine_invariant_log(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor

   Affine-invariant logarithmic map on the SPD manifold.

   .. math::

       \mathrm{Log}_X(Y) = X^{1/2} \log\big(X^{-1/2} Y X^{-1/2}\big) X^{1/2}

   Maps a point :math:`Y` on the manifold to a tangent vector at base
   :math:`X`. Inverse of :func:`affine_invariant_exp`.

   :param base: Base point(s) on the SPD manifold
   :type base: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`
   :param point: Point(s) on the SPD manifold
   :type point: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

   :returns: **tangent** -- Tangent vector(s) at base (symmetric matrices)
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`


.. py:function:: affine_invariant_projx(data: torch.Tensor) -> torch.Tensor

   Project matrices onto the SPD manifold by symmetrizing and
   clamping eigenvalues to be strictly positive.

   :param data: Batch of matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

   :returns: **projected** -- Batch of SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`


