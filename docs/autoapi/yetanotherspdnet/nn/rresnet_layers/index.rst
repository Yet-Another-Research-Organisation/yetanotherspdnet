yetanotherspdnet.nn.rresnet_layers
==================================

.. py:module:: yetanotherspdnet.nn.rresnet_layers

.. autoapi-nested-parse::

   Riemannian Residual Network layers: spectral vector field and residual block.



Classes
-------

.. autoapisummary::

   yetanotherspdnet.nn.rresnet_layers.SpectralVectorField
   yetanotherspdnet.nn.rresnet_layers.ResidualBlock


Module Contents
---------------

.. py:class:: SpectralVectorField(n_features: int, spectrum_type: str = 'conv1d', spectrum_hidden_dim: int = 3, spectrum_n_layers: int = 2, spectrum_kernel_size: int = 5, stiefel_parametrization_mode: str = 'static', stiefel_n_steps_ref_update: int = 100, use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Spectral vector field on SPD manifold.

   Computes a tangent vector at X as:
       V = Q diag(f(spec(X))) Q^T

   where:
   - spec(X): eigenvalues of the input SPD matrix
   - f: learnable spectrum mapping (Conv1d or MLP)
   - Q: learnable orthogonal matrix (Stiefel-parametrized)

   The output is a symmetric matrix in the tangent space at X.

   :param n_features: Dimension of SPD matrices (n x n)
   :type n_features: :py:class:`int`
   :param spectrum_type: Type of spectrum mapping network.
                         Default is "conv1d".
                         Choices are: "conv1d" and "mlp"
   :type spectrum_type: :py:class:`str`, *optional*
   :param spectrum_hidden_dim: Hidden dimension for spectrum mapping. Default is 3
   :type spectrum_hidden_dim: :py:class:`int`, *optional*
   :param spectrum_n_layers: Number of hidden layers in spectrum mapping. Default is 2
   :type spectrum_n_layers: :py:class:`int`, *optional*
   :param spectrum_kernel_size: Kernel size for Conv1d spectrum mapping. Default is 5
   :type spectrum_kernel_size: :py:class:`int`, *optional*
   :param stiefel_parametrization_mode: Parametrization mode for orthogonal matrix Q.
                                        Default is "static".
                                        Choices are: "static" and "dynamic"
   :type stiefel_parametrization_mode: :py:class:`str`, *optional*
   :param stiefel_n_steps_ref_update: Steps between reference point updates for dynamic parametrization.
                                      Default is 100
   :type stiefel_n_steps_ref_update: :py:class:`int`, *optional*
   :param use_autograd: Use torch autograd for gradient computation. Default is False
   :type use_autograd: :py:class:`bool`, *optional*
   :param device: Device. Default is torch.device("cpu")
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Data type. Default is torch.float64
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator for reproducibility. Default is None
   :type generator: :py:class:`torch.Generator | None`, *optional*


   .. py:attribute:: n_features


   .. py:attribute:: spectrum_type
      :value: 'conv1d'



   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: Q


   .. py:attribute:: is_dynamic


   .. py:attribute:: current_ref_step
      :value: 0



   .. py:attribute:: n_steps_ref_update
      :value: 100



   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Compute tangent vector from SPD matrix spectrum.

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

      :returns: **tangent** -- Symmetric tangent vectors at each input point
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register optimizer hook for dynamic Stiefel parametrization.



   .. py:method:: __repr__() -> str


.. py:class:: ResidualBlock(n_features: int, spectrum_type: str = 'conv1d', spectrum_hidden_dim: int = 3, spectrum_n_layers: int = 2, spectrum_kernel_size: int = 5, stiefel_parametrization_mode: str = 'static', stiefel_n_steps_ref_update: int = 100, use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Riemannian residual block on SPD manifold.

   Applies one geodesic step:
       X_new = projx(Exp_X(VF(X)))

   where VF is a SpectralVectorField and Exp is the affine-invariant
   exponential map.

   :param n_features: Dimension of SPD matrices
   :type n_features: :py:class:`int`
   :param spectrum_type: Type of spectrum mapping. Default is "conv1d"
   :type spectrum_type: :py:class:`str`, *optional*
   :param spectrum_hidden_dim: Hidden dimension for spectrum mapping. Default is 3
   :type spectrum_hidden_dim: :py:class:`int`, *optional*
   :param spectrum_n_layers: Hidden layers in spectrum mapping. Default is 2
   :type spectrum_n_layers: :py:class:`int`, *optional*
   :param spectrum_kernel_size: Kernel size for Conv1d. Default is 5
   :type spectrum_kernel_size: :py:class:`int`, *optional*
   :param stiefel_parametrization_mode: Parametrization mode for Q matrix. Default is "static"
   :type stiefel_parametrization_mode: :py:class:`str`, *optional*
   :param stiefel_n_steps_ref_update: Steps between reference updates. Default is 100
   :type stiefel_n_steps_ref_update: :py:class:`int`, *optional*
   :param use_autograd: Use autograd for exp map gradient. Default is False
   :type use_autograd: :py:class:`bool`, *optional*
   :param device: Device. Default is torch.device("cpu")
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Data type. Default is torch.float64
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator. Default is None
   :type generator: :py:class:`torch.Generator | None`, *optional*


   .. py:attribute:: n_features


   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: vector_field


   .. py:attribute:: exp_map


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass: geodesic residual step.

      X_new = projx(Exp_X(VF(X)))

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`

      :returns: **result** -- Updated SPD matrices after one residual step
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n`, :py:class:`n)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register optimizer hooks for dynamic parametrizations.



   .. py:method:: __repr__() -> str


