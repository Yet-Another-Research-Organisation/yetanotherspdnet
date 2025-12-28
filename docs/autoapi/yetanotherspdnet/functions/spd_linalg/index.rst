yetanotherspdnet.functions.spd_linalg
=====================================

.. py:module:: yetanotherspdnet.functions.spd_linalg


Classes
-------

.. autoapisummary::

   yetanotherspdnet.functions.spd_linalg.VecBatch
   yetanotherspdnet.functions.spd_linalg.VechBatch
   yetanotherspdnet.functions.spd_linalg.SqrtmSPD
   yetanotherspdnet.functions.spd_linalg.InvSqrtmSPD
   yetanotherspdnet.functions.spd_linalg.PowmSPD
   yetanotherspdnet.functions.spd_linalg.LogmSPD
   yetanotherspdnet.functions.spd_linalg.ExpmSymmetric
   yetanotherspdnet.functions.spd_linalg.ScaledSoftPlusSymmetric
   yetanotherspdnet.functions.spd_linalg.InvScaledSoftPlusSPD
   yetanotherspdnet.functions.spd_linalg.EighReLu
   yetanotherspdnet.functions.spd_linalg.CongruenceSPD
   yetanotherspdnet.functions.spd_linalg.Whitening
   yetanotherspdnet.functions.spd_linalg.CongruenceRectangular


Functions
---------

.. autoapisummary::

   yetanotherspdnet.functions.spd_linalg.symmetrize
   yetanotherspdnet.functions.spd_linalg.vec_batch
   yetanotherspdnet.functions.spd_linalg.unvec_batch
   yetanotherspdnet.functions.spd_linalg.vech_batch
   yetanotherspdnet.functions.spd_linalg.unvech_batch
   yetanotherspdnet.functions.spd_linalg.eigh_operation
   yetanotherspdnet.functions.spd_linalg.eigh_operation_grad
   yetanotherspdnet.functions.spd_linalg.solve_sylvester_SPD
   yetanotherspdnet.functions.spd_linalg.sqrtm_SPD
   yetanotherspdnet.functions.spd_linalg.inv_sqrtm_SPD
   yetanotherspdnet.functions.spd_linalg.powm_SPD
   yetanotherspdnet.functions.spd_linalg.logm_SPD
   yetanotherspdnet.functions.spd_linalg.expm_symmetric
   yetanotherspdnet.functions.spd_linalg.scaled_softplus_symmetric
   yetanotherspdnet.functions.spd_linalg.inv_scaled_softplus_SPD
   yetanotherspdnet.functions.spd_linalg.eigh_relu
   yetanotherspdnet.functions.spd_linalg.congruence_SPD
   yetanotherspdnet.functions.spd_linalg.whitening
   yetanotherspdnet.functions.spd_linalg.congruence_rectangular


Module Contents
---------------

.. py:function:: symmetrize(data: torch.Tensor) -> torch.Tensor

   Symmetrize a tensor along the last two dimensions

   :param data: Batch of square matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **sym_data** -- Batch of symmetrized matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: vec_batch(data: torch.Tensor) -> torch.Tensor

   Vectorize a batch of tensors along last two dimensions

   :param data: Batch of matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows`, :py:class:`n_columns)`

   :returns: **data_vec** -- Batch of vectorized matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows*n_columns)`


.. py:function:: unvec_batch(data_vec: torch.Tensor, n_rows: int) -> torch.Tensor

   Unvectorize a batch of tensors along last dimension

   :param data_vec: Batch of vectorized matrices
   :type data_vec: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows*n_columns)`
   :param n_rows: Number of rows of the matrices
   :type n_rows: :py:class:`int`

   :returns: **data** -- Batch of matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows`, :py:class:`n_columns)`


.. py:class:: VecBatch(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Vectorize a batch of matrices along last two dimensions.
   Matrices are assumed to be symmetric (for backward)


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the vectorization of a batch of matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **vec_data**
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features ** 2)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the vectorization of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to vectorized input batch of matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features ** 2)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: vech_batch(data: torch.Tensor) -> torch.Tensor

   Vectorize the lower triangular part of a batch of square matrices

   :param data: Batch of matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **data_vech** -- Batch of vectorized matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features*(n_features+1)//2)`


.. py:function:: unvech_batch(data_vech: torch.Tensor, n_features: int) -> torch.Tensor

   Unvectorize a batch of tensors along last dimension,
   assuming that matrices are symmetric

   :param X_vech: Batch of vectorized matrices
   :type X_vech: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features*(n_features+1)//2)`
   :param n_features: number of features
   :type n_features: :py:class:`int`

   :returns: **X** -- Batch of symmetric matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: VechBatch(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Half vectorize a batch of symmetric matrices along last two dimensions


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the half vectorization of a batch of matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **vech_data**
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features*(n_features+1) // 2)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the vectorization of a batch of matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to half vectorized input batch of symmetric matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features*(n_features+1) // 2)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: eigh_operation(eigvals: torch.Tensor, eigvecs: torch.Tensor, operation: collections.abc.Callable) -> torch.Tensor

   Applies a function on the eigenvalues of a batch of symmetric matrices

   :param eigvals: Eigenvalues of the corresponding batch of symmetric matrices
   :type eigvals: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`
   :param eigvecs: Eigenvectors of the corresponding batch of symmetric matrices
   :type eigvecs: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param operation: Function to apply on eigenvalues
   :type operation: :py:class:`Callable`

   :returns: **result** -- Resulting symmetric matrices with operation applied to eigenvalues
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: eigh_operation_grad(grad_output: torch.Tensor, eigvals: torch.Tensor, eigvecs: torch.Tensor, operation: collections.abc.Callable, operation_deriv: collections.abc.Callable) -> torch.Tensor

   Computes the backpropagation of the gradient for a function applied on
   the eigenvalues of a batch of symmetric matrices

   :param grad_output: Gradient of the loss with respect to the output of the operation on eigenvalues
   :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param eigvals: Eigenvalues of the corresponding batch of symmetric matrices
   :type eigvals: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`
   :param eigvecs: Eigenvectors of the corresponding batch of symmetric matrices
   :type eigvecs: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param operation: Function to apply on eigenvalues
   :type operation: :py:class:`Callable`
   :param operation_deriv: Derivative of the function to apply on eigenvalues
   :type operation_deriv: :py:class:`Callable`

   :returns: **grad_input** -- Gradient of the loss with respect to the input batch of symmetric matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: solve_sylvester_SPD(eigvals: torch.Tensor, eigvecs: torch.Tensor, mat: torch.Tensor) -> torch.Tensor

   Solve Sylvester equations in the context of SPD matrices
   relying on eigenvalue decomposition

   :param eigvals: Eigenvalues of a batch of SPD matrices
   :type eigvals: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`
   :param eigvecs: Eigenvectors of a batch of SPD matrices
   :type eigvecs: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param mat: Batch of matrices on the right side of Sylvester equations.
               If matrices are symmetric then the solutions will be symmetric.
               If they are skew-symmetric, then the results will be skew-symmetric.
   :type mat: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: **result** -- Symmetric matrices solutions to Sylvester equations
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: sqrtm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Matrix logarithm of a batch of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **sqrtm_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix logarithms of the input batch of SPD matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: SqrtmSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix square root of a batch of SPD matrices
   (relies on eigenvalue decomposition)


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix square root of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **sqrtm_data** -- Matrix square roots of the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the matrix square root of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix square roots of the input batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: inv_sqrtm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Matrix logarithm of a batch of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **logm_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix logarithms of the input batch of SPD matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: InvSqrtmSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix inverse square root of a batch of SPD matrices
   (relies on eigenvalue decomposition)


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix inverse square root of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **inv_sqrtm_data** -- Matrix square roots of the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the matrix inverse square root of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix square roots of the input batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: powm_SPD(data: torch.Tensor, exponent: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Matrix power of a batch of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param exponent: Power exponent
   :type exponent: :py:class:`torch.float`

   :returns: * **powm_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix powers of the input batch of SPD matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: PowmSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix power of a batch of SPD matrices
   (relies on eigenvalue decomposition)


   .. py:method:: forward(ctx, data: torch.Tensor, exponent: float | torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix power of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param exponent: Power exponent
      :type exponent: :py:class:`torch.float`

      :returns: **powm_data** -- Matrix powers of the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the matrix power of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix powers of the input batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the input batch of SPD matrices
                * **grad_input_exponent** (:py:class:`torch.float`) -- Gradient of the loss with respect to the power exponent



.. py:function:: logm_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Matrix logarithm of a batch of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **logm_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix logarithms of the input batch of SPD matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: LogmSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix logarithm of a batch of SPD matrices
   (relies on eigenvalue decomposition)


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix logarithm of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **logm_data** -- Matrix logarithms of the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the matrix logarithm of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix logarithms of the input batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: expm_symmetric(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Matrix exponential of a batch of symmetric matrices

   :param data: Batch of symmetric matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **expm_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix exponentials of the input batch of symmetric matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: ExpmSymmetric(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix exponential of a batch of symmetric matrices
   (relies on eigenvalue decomposition)


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix exponential of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of symmetric matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **expm_data** -- Matrix exponentials of the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the matrix exponential of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix exponentials of the input batch of symmetric matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: scaled_softplus_symmetric(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Scaled matrix SoftPlus of a batch of symmetric matrices.
   It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
   f'(x) -> 1 as x -> +inf

   :param data: Batch of symmetric matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **softplus_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix SoftPlus of the input batch of symmetric matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: ScaledSoftPlusSymmetric(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Scaled matrix SoftPlus of a batch of symmetric matrices.
   It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
   f'(x) -> 1 as x -> +inf


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the scaled matrix SoftPlus of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of symmetric matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **softplus_data** -- Matrix SoftPlus of the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the scaled matrix SoftPlus of a batch of symmetric matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix SoftPlus of the input batch of symmetric matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: inv_scaled_softplus_SPD(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Inverse scaled SoftPlus of a batch of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

   :returns: * **inv_softplus_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Matrix logarithms of the input batch of SPD matrices
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: InvScaledSoftPlusSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Matrix inverse scaled SoftPlus of a batch of SPD matrices


   .. py:method:: forward(ctx, data: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the matrix inverse scaled SoftPlus of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **inv_softplus_data** -- Matrix logarithms of the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Backward pass of the matrix inverse scaled SoftPlus of a batch of SPD matrices

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to matrix inverse SoftPlus of the input batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: eigh_relu(data: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   ReLu activation function on the eigenvalues of SPD matrices

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param eps: Value for the rectification of the eigenvalues
   :type eps: :py:class:`float`

   :returns: * **data_transformed** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Batch of SPD matrices with rectified eigenvalues
             * **eigvals** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features)`) -- Eigenvalues of matrices in data
             * **eigvecs** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Eigenvectors of matrices in data


.. py:class:: EighReLu(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   ReLu activation function on the eigenvalues of SPD matrices


   .. py:method:: forward(ctx, data: torch.Tensor, eps: float) -> torch.Tensor
      :staticmethod:


      Forward pass of the ReLu activation function on the eigenvalues of SPD matrices

      :param ctx: Context object to save tensors for the backward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param eps: Value for the rectification of the eigenvalues
      :type eps: :py:class:`float`

      :returns: **data_transformed** -- Batch of SPD matrices with rectified eigenvalues
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]
      :staticmethod:


      Backward pass of the ReLu activation function on the eigenvalues of SPD matrices

      :param ctx: Context object to save tensors for the backward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the output batch of SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **grad_input** -- Gradient of the loss with respect to the input batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



.. py:function:: congruence_SPD(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor

   Congruence of a batch of SPD matrices with an SPD matrix

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param matrix: SPD matrix
   :type matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **data_transformed** -- Transformed batch of SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: CongruenceSPD(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Congruence of a batch of SPD matrices with an SPD matrix


   .. py:method:: forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the congruence of a batch of SPD matrices with an SPD matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param matrix: SPD matrix
      :type matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Transformed batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the congruence of a batch of SPD matrices with an SPD matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the batch of transformed SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the input batch of SPD matrices
                * **grad_input_bias** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the SPD matrix used for congruence



.. py:function:: whitening(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor

   Whitening of a batch of SPD matrices with an SPD matrix

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
   :param matrix: SPD matrix
   :type matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

   :returns: **data_transformed** -- Transformed batch of SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`


.. py:class:: Whitening(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Whitening of a batch of SPD matrices with an SPD matrix


   .. py:method:: forward(ctx, data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the whitening of a batch of SPD matrices with an SPD matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`
      :param matrix: SPD matrix
      :type matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Transformed batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the whitening of a batch of SPD matrices with an SPD matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the batch of whitened SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the input batch of SPD matrices
                * **grad_input_matrix** (:py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`) -- Gradient of the loss with respect to the SPD matrix used for whitening



.. py:function:: congruence_rectangular(data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor

   Forward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

   :param data: Batch of SPD matrices
   :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_in`, :py:class:`n_in)`
   :param weight: Rectangular matrix (e.g., weights),
                  n_in > n_out is expected
   :type weight: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

   :returns: **data_transformed** -- Transformed batch of SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_out`, :py:class:`n_out)`


.. py:class:: CongruenceRectangular(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Congruence of a batch of SPD matrices with a (full-rank) rectangular matrix


   .. py:method:: forward(ctx, data: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
      :staticmethod:


      Forward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_in`, :py:class:`n_in)`
      :param weight: Rectangular matrix (e.g., weights),
                     n_in > n_out is expected
      :type weight: :py:class:`torch.Tensor` of :py:class:`shape (n_out`, :py:class:`n_in)`

      :returns: **data_transformed** -- Transformed batch of SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_out`, :py:class:`n_out)`



   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
      :staticmethod:


      Backward pass of the congruence of a batch of SPD matrices with a (full-rank) rectangular matrix

      :param ctx: Context object to retrieve tensors saved during the forward pass
      :type ctx: :py:class:`torch.autograd.function._ContextMethodMixin`
      :param grad_output: Gradient of the loss with respect to the batch of transformed SPD matrices
      :type grad_output: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_out`, :py:class:`n_out)`

      :returns: * **grad_input_data** (:py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_in`, :py:class:`n_in)`) -- Gradient of the loss with respect to the input batch of SPD matrices
                * **grad_input_W** (:py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`) -- Gradient of the loss with respect to the (full-rank) rectangular matrix W



