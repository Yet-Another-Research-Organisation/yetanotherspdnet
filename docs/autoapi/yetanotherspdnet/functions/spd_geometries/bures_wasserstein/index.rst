yetanotherspdnet.functions.spd_geometries.bures_wasserstein
===========================================================

.. py:module:: yetanotherspdnet.functions.spd_geometries.bures_wasserstein

.. autoapi-nested-parse::

   Bures-Wasserstein geometry: geodesic, mean, standard deviation, and transport maps.

   Implements the Bures-Wasserstein (BW) metric, geodesic, Frechet mean (barycenter),
   scalar standard deviation, and related operations (log/exp maps, parallel transport)
   on the manifold of symmetric positive definite matrices.

   .. admonition:: References

      [1] Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between positive
          definite matrices." Expositiones Mathematicae, 2019.
      [2] Kobler et al. "Controlling the Fréchet Variance Improves Batch
          Normalization on the Symmetric Positive Definite Manifold." CVPR, 2022.



Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.BuresWassersteinStdScalar


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_distance_squared
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_log_identity
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_exp_identity
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_log
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_parallel_transport_to_identity
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_parallel_transport_from_identity
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_geodesic
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_mean
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.BuresWassersteinMean
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_std_scalar
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_center
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_scale
   yetanotherspdnet.functions.spd_geometries.bures_wasserstein.bures_wasserstein_bias


Module Contents
---------------

.. py:function:: bures_wasserstein_distance_squared(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor

   Squared Bures-Wasserstein distance between SPD matrices.

   .. math::

       d_{BW}^2(X_1, X_2) = \operatorname{tr}(X_1) + \operatorname{tr}(X_2)
           - 2\operatorname{tr}\!\Big(\big(X_1^{1/2} X_2 X_1^{1/2}\big)^{1/2}\Big)

   This is the (squared) 2-Wasserstein distance between the zero-mean
   Gaussian distributions with covariances :math:`X_1` and :math:`X_2`.

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **dist_sq** -- Squared BW distances
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...)`


.. py:function:: bures_wasserstein_log_identity(X: torch.Tensor) -> torch.Tensor

   Logarithmic map at the identity under Bures-Wasserstein geometry.

   .. math::

       \mathrm{Log}_I(X) = 2\big(X^{1/2} - I\big)

   :param X: SPD matrices
   :type X: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **S** -- Symmetric matrices in the tangent space at I
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_exp_identity(S: torch.Tensor) -> torch.Tensor

   Exponential map at the identity under Bures-Wasserstein geometry.

   .. math::

       \mathrm{Exp}_I(S) = \left(I + \frac{S}{2}\right)^2

   Inverse of :func:`bures_wasserstein_log_identity`.

   :param S: Symmetric matrices in the tangent space at I
   :type S: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **X** -- SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_log(X: torch.Tensor, base: torch.Tensor) -> torch.Tensor

   Logarithmic map at a general base point under BW geometry.

   .. math::

       \mathrm{Log}_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2B

   computed via the identity :math:`(BX)^{1/2} = B^{1/2}
   (B^{1/2} X B^{1/2})^{1/2} B^{-1/2}` and
   :math:`(XB)^{1/2} = \big[(BX)^{1/2}\big]^\top`.

   :param X: SPD matrices
   :type X: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param base: Base point (SPD matrix)
   :type base: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)` or :py:class:`(n_features`, :py:class:`n_features)`

   :returns: **tangent** -- Tangent vectors at *base*
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_parallel_transport_to_identity(tangent_vec: torch.Tensor, source: torch.Tensor) -> torch.Tensor

   Parallel transport from *source* to the identity under BW geometry.

   Given :math:`\text{source} = V \operatorname{diag}(\lambda) V^\top`:

   .. math::

       \Gamma_{\text{source}\to I}(S) = V\left[
           \sqrt{\frac{2}{\lambda_i + \lambda_j}} \,(V^\top S V)_{ij}
           \right]_{ij} V^\top

   :param tangent_vec: Tangent vectors at *source*
   :type tangent_vec: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param source: Source point (SPD matrix)
   :type source: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)` or :py:class:`(n_features`, :py:class:`n_features)`

   :returns: **transported** -- Tangent vectors at identity
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_parallel_transport_from_identity(tangent_vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor

   Parallel transport from the identity to *target* under BW geometry.

   Given :math:`\text{target} = U \operatorname{diag}(\delta) U^\top`:

   .. math::

       \Gamma_{I\to\text{target}}(S) = U\left[
           \sqrt{\frac{\delta_i + \delta_j}{2}} \,(U^\top S U)_{ij}
           \right]_{ij} U^\top

   Inverse of :func:`bures_wasserstein_parallel_transport_to_identity`.

   :param tangent_vec: Tangent vectors at identity
   :type tangent_vec: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param target: Target point (SPD matrix)
   :type target: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)` or :py:class:`(n_features`, :py:class:`n_features)`

   :returns: **transported** -- Tangent vectors at *target*
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_geodesic(point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor

   Bures-Wasserstein geodesic (closed-form 2-sample weighted mean).

   .. math::

       E_2(X_1, X_2; t) = (1-t)^2 X_1 + t^2 X_2
           + t(1-t)\Big[(X_2 X_1)^{1/2} + (X_1 X_2)^{1/2}\Big]

   where :math:`(X_1 X_2)^{1/2} = X_1^{1/2}
   (X_1^{1/2} X_2 X_1^{1/2})^{1/2} X_1^{-1/2}` and :math:`t \in [0, 1]`.

   :param point1: SPD matrices
   :type point1: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param point2: SPD matrices
   :type point2: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param t: Interpolation parameter in [0, 1]
   :type t: :py:class:`float | torch.Tensor`

   :returns: **point** -- SPD matrices on the geodesic
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_mean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor

   Bures-Wasserstein barycenter (Fréchet mean) via fixed-point iteration.

   .. math::

       G_{k+1} = G_k^{-1/2}\left(\frac{1}{N}\sum_{i=1}^{N}
           \big(G_k^{1/2} X_i G_k^{1/2}\big)^{1/2}\right)^2 G_k^{-1/2}

   the unique fixed point of this map is the barycenter minimizing
   :math:`\sum_i d_{BW}(G, X_i)^2` (see
   :func:`bures_wasserstein_distance_squared`). The initial estimate
   :math:`G_0` is the arithmetic mean of *data*.

   :param data: Batch of SPD matrices. The mean is computed along ``...`` axes.
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param n_iterations: Number of fixed-point iterations, by default 1
   :type n_iterations: :py:class:`int`

   :returns: **barycenter** -- BW barycenter
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: BuresWassersteinMean(data: torch.Tensor, n_iterations: int = 1) -> torch.Tensor

   Bures-Wasserstein barycenter (Frechet mean) -- CamelCase wrapper.

   Since the fixed-point iteration composes differentiable operations
   (eigh, matmul, sqrtm), autograd handles the backward automatically.

   :param data: Batch of SPD matrices. The mean is computed along ``...`` axes.
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param n_iterations: Number of fixed-point iterations, by default 1
   :type n_iterations: :py:class:`int`

   :returns: **barycenter** -- BW barycenter
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_std_scalar(data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor

   Scalar standard deviation under the Bures-Wasserstein distance.

   .. math::

       \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_{BW}^2(G, X_i)}

   where :math:`G` is the reference point (typically the BW barycenter)
   and :math:`d_{BW}` is the Bures-Wasserstein distance (see
   :func:`bures_wasserstein_distance_squared`).

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param reference_point: SPD matrix (some kind of mean of data)
   :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **scalar_std** -- Scalar standard deviation
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`


.. py:class:: BuresWassersteinStdScalar(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scalar standard deviation under the Bures-Wasserstein distance
   (manual backward).


   .. py:method:: forward(ctx, data: torch.Tensor, reference_point: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the BW scalar standard deviation

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param reference_point: SPD matrix (some kind of mean of data)
      :type reference_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **scalar_std** -- Scalar standard deviation
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the BW scalar standard deviation

      Uses:
          d(d_BW^2(B,X)) / dX = I - B^{1/2} (B^{1/2} X B^{1/2})^{-1/2} B^{1/2}
          d(d_BW^2(B,X)) / dB = I - X^{1/2} (X^{1/2} B X^{1/2})^{-1/2} X^{1/2}

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the scalar std
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the input data
                * **grad_input_reference_point** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the reference point



.. py:function:: bures_wasserstein_center(data: torch.Tensor, barycenter: torch.Tensor) -> torch.Tensor

   Center SPD data by parallel-transporting from the barycenter to the identity.

   .. math::

       X_{\text{centered}} = \mathrm{Exp}_I\big(\Gamma_{B\to I}(\mathrm{Log}_B(X))\big)

   composing :func:`bures_wasserstein_log`,
   :func:`bures_wasserstein_parallel_transport_to_identity`, and
   :func:`bures_wasserstein_exp_identity` — the BatchNorm analogue of
   subtracting the mean, but on the SPD manifold.

   :param data: SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param barycenter: BW barycenter
   :type barycenter: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **centered** -- Centered SPD matrices (around the identity)
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_scale(data: torch.Tensor, variance: torch.Tensor, shift: torch.Tensor, eps: float = 1e-05) -> torch.Tensor

   Scale centered SPD data at the identity.

   .. math::

       X_{\text{scaled}} = \mathrm{Exp}_I\!\left(
           \frac{s}{\sqrt{\text{var} + \epsilon}}\, \mathrm{Log}_I(X)\right)
       = \left(I + \frac{s}{\sqrt{\text{var} + \epsilon}}\,
           \big(X^{1/2} - I\big)\right)^2

   the BatchNorm analogue of dividing by the standard deviation and
   multiplying by a learnable scale :math:`s`.

   :param data: Centered SPD matrices (around the identity)
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param variance: Frechet variance
   :type variance: :py:class:`torch.Tensor` of :py:class:`shape ()`
   :param shift: Learnable scaling parameter
   :type shift: :py:class:`torch.Tensor` of :py:class:`shape ()`
   :param eps: Small constant for numerical stability, by default 1e-5
   :type eps: :py:class:`float`

   :returns: **scaled** -- Scaled SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: bures_wasserstein_bias(data: torch.Tensor, bias_point: torch.Tensor) -> torch.Tensor

   Bias SPD data by parallel-transporting from the identity to *bias_point*.

   .. math::

       X_{\text{biased}} = \mathrm{Exp}_G\big(\Gamma_{I\to G}(\mathrm{Log}_I(X))\big)

   where :math:`\mathrm{Exp}_G(V) = G + V + Z^2` with :math:`G^{1/2} Z +
   Z G^{1/2} = V` (see :func:`bures_wasserstein_parallel_transport_from_identity`).
   The BatchNorm analogue of adding a learnable bias :math:`G`.

   :param data: SPD matrices around the identity
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param bias_point: Learned bias (SPD matrix)
   :type bias_point: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **biased** -- Biased SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


