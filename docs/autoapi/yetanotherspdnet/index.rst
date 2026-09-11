yetanotherspdnet
================

.. py:module:: yetanotherspdnet

.. autoapi-nested-parse::

   Yet Another SPDNet - A robust and tested implementation of SPDNet learning models.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/yetanotherspdnet/functions/index
   /autoapi/yetanotherspdnet/model/index
   /autoapi/yetanotherspdnet/nn/index
   /autoapi/yetanotherspdnet/random/index


Attributes
----------

.. autoapisummary::

   yetanotherspdnet.__version__


Classes
-------

.. autoapisummary::

   yetanotherspdnet.GBWBNRResNet
   yetanotherspdnet.RResNet
   yetanotherspdnet.SPDnet


Package Contents
----------------

.. py:data:: __version__
   :value: '0.1.0'


.. py:class:: GBWBNRResNet(input_dim: int, hidden_dim: int, output_dim: int, softmax: bool = False, bimap_parametrized: bool = True, bimap_parametrization_mode: str = 'static', bimap_parametrization_options: dict | None = None, bimap_n_steps_ref_update: int = 100, batchnorm: bool = True, batchnorm_type: str = 'mean_var_scalar', batchnorm_mean_type: str = 'bures_wasserstein', batchnorm_mean_options: dict | None = None, batchnorm_momentum: float = 0.1, batchnorm_norm_strategy: str = 'classical', batchnorm_minibatch_mode: str = 'constant', batchnorm_minibatch_momentum: float = 0.01, batchnorm_minibatch_maxstep: int = 100, batchnorm_parametrization: str = 'softplus', batchnorm_parametrization_mode: str = 'static', batchnorm_n_steps_ref_update: int = 100, spectrum_type: str = 'conv1d', spectrum_hidden_dim: int = 3, spectrum_n_layers: int = 2, spectrum_kernel_size: int = 5, stiefel_parametrization_mode: str = 'static', stiefel_n_steps_ref_update: int = 100, vec_type: str = 'vec', use_logeig: bool = True, use_autograd: bool | dict = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Riemannian Residual Network faithful to the GBWBN experiment architecture.

   Architecture:
       BiMap(input_dim -> hidden_dim)
       -> [BatchNorm]
       -> ResidualBlock (spectral vector field + exp map)
       -> LogEig
       -> Vec/Vech
       -> Linear(hidden_dim^2 -> output_dim)
       -> [Softmax]

   This architecture matches the paper experiments (HDM05, NTU60) which use:
   - A single BiMap for dimension reduction
   - A single residual block with spectral vector field
   - No ReEig (eigenvalue rectification not needed before residual block)

   The residual block applies a geodesic step on the SPD manifold:
       X_new = Exp_X(Q diag(f(spec(X))) Q^T)

   where Q is a Stiefel-parametrized orthogonal matrix and f is a
   learnable spectrum mapping (Conv1d or MLP on eigenvalues).

   :param input_dim: Input SPD matrix dimension
   :type input_dim: :py:class:`int`
   :param hidden_dim: Dimension after BiMap (also the residual block dimension)
   :type hidden_dim: :py:class:`int`
   :param output_dim: Number of output classes
   :type output_dim: :py:class:`int`
   :param softmax: Apply softmax to output. Default is False
   :type softmax: :py:class:`bool`, *optional*
   :param bimap_parametrized: Enforce Stiefel constraints on BiMap. Default is True
   :type bimap_parametrized: :py:class:`bool`, *optional*
   :param bimap_parametrization_mode: "static" or "dynamic" parametrization. Default is "static"
   :type bimap_parametrization_mode: :py:class:`str`, *optional*
   :param bimap_parametrization_options: Options for BiMap parametrization. Default is None
   :type bimap_parametrization_options: :py:class:`dict | None`, *optional*
   :param bimap_n_steps_ref_update: Steps between reference updates for dynamic BiMap. Default is 100
   :type bimap_n_steps_ref_update: :py:class:`int`, *optional*
   :param batchnorm: Apply batch normalization. Default is True
   :type batchnorm: :py:class:`bool`, *optional*
   :param batchnorm_type: "mean_only" or "mean_var_scalar". Default is "mean_var_scalar"
   :type batchnorm_type: :py:class:`str`, *optional*
   :param batchnorm_mean_type: SPD mean type for batchnorm. Default is "bures_wasserstein"
   :type batchnorm_mean_type: :py:class:`str`, *optional*
   :param batchnorm_mean_options: Options for mean computation. Default is None
   :type batchnorm_mean_options: :py:class:`dict | None`, *optional*
   :param batchnorm_momentum: Running mean momentum. Default is 0.1
   :type batchnorm_momentum: :py:class:`float`, *optional*
   :param batchnorm_norm_strategy: "classical" or "minibatch". Default is "classical"
   :type batchnorm_norm_strategy: :py:class:`str`, *optional*
   :param batchnorm_minibatch_mode: "constant", "decay", or "growth". Default is "constant"
   :type batchnorm_minibatch_mode: :py:class:`str`, *optional*
   :param batchnorm_minibatch_momentum: Minibatch momentum. Default is 0.01
   :type batchnorm_minibatch_momentum: :py:class:`float`, *optional*
   :param batchnorm_minibatch_maxstep: Max step for momentum schedule. Default is 100
   :type batchnorm_minibatch_maxstep: :py:class:`int`, *optional*
   :param batchnorm_parametrization: "softplus" or "exp". Default is "softplus"
   :type batchnorm_parametrization: :py:class:`str`, *optional*
   :param batchnorm_parametrization_mode: "static" or "dynamic". Default is "static"
   :type batchnorm_parametrization_mode: :py:class:`str`, *optional*
   :param batchnorm_n_steps_ref_update: Steps between reference updates for BN. Default is 100
   :type batchnorm_n_steps_ref_update: :py:class:`int`, *optional*
   :param spectrum_type: "conv1d" or "mlp" for spectral vector field. Default is "conv1d"
   :type spectrum_type: :py:class:`str`, *optional*
   :param spectrum_hidden_dim: Hidden dimension for spectrum network. Default is 3
   :type spectrum_hidden_dim: :py:class:`int`, *optional*
   :param spectrum_n_layers: Number of hidden layers in spectrum network. Default is 2
   :type spectrum_n_layers: :py:class:`int`, *optional*
   :param spectrum_kernel_size: Kernel size for Conv1d spectrum. Default is 5
   :type spectrum_kernel_size: :py:class:`int`, *optional*
   :param stiefel_parametrization_mode: Parametrization mode for Q matrix. Default is "static"
   :type stiefel_parametrization_mode: :py:class:`str`, *optional*
   :param stiefel_n_steps_ref_update: Steps between Q reference updates. Default is 100
   :type stiefel_n_steps_ref_update: :py:class:`int`, *optional*
   :param vec_type: "vec" or "vech". Default is "vec"
   :type vec_type: :py:class:`str`, *optional*
   :param use_logeig: Apply LogEig before vectorization. Default is True
   :type use_logeig: :py:class:`bool`, *optional*
   :param use_autograd: Autograd control. Bool for all, dict with keys:
                        'bimap', 'logeig', 'batchnorm', 'vec', 'residual'.
                        Default is False
   :type use_autograd: :py:class:`bool | dict`, *optional*
   :param device: Device. Default is torch.device("cpu")
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Data type. Default is torch.float64
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator for reproducibility. Default is None
   :type generator: :py:class:`torch.Generator | None`, *optional*


   .. py:attribute:: input_dim


   .. py:attribute:: hidden_dim


   .. py:attribute:: output_dim


   .. py:attribute:: softmax
      :value: False



   .. py:attribute:: batchnorm
      :value: True



   .. py:attribute:: batchnorm_type
      :value: 'mean_var_scalar'



   .. py:attribute:: batchnorm_mean_type
      :value: 'bures_wasserstein'



   .. py:attribute:: vec_type
      :value: 'vec'



   .. py:attribute:: use_logeig
      :value: True



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: generator
      :value: None



   .. py:attribute:: spd_layers


   .. py:attribute:: linear


   .. py:method:: forward(X: torch.Tensor) -> torch.Tensor

      Forward pass.

      :param X: Input SPD matrices
      :type X: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`input_dim`, :py:class:`input_dim)`

      :returns: Output predictions
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`output_dim)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register optimizer hooks for all dynamic parametrizations.



   .. py:method:: get_last_tensor(X: torch.Tensor) -> torch.Tensor

      Return the last SPD tensor before vectorization.



   .. py:method:: __repr__() -> str


   .. py:method:: layers_str() -> str

      Return a formatted string listing the layers.



   .. py:method:: create_model_name_hash() -> str

      Create a short hash of the model configuration.



   .. py:method:: get_model_hash() -> str

      Return the model hash, creating it if needed.



.. py:class:: RResNet(input_dim: int, hidden_layers_size: list[int], n_residual_blocks: list[int], output_dim: int, softmax: bool = False, reeig: bool = False, reeig_eps: float = 0.001, bimap_parametrized: bool = True, bimap_parametrization_mode: str = 'static', bimap_parametrization_options: dict | None = None, bimap_n_steps_ref_update: int = 100, batchnorm: bool = False, batchnorm_type: str = 'mean_only', batchnorm_mean_type: str = 'affine_invariant', batchnorm_mean_options: dict | None = None, batchnorm_momentum: float = 0.01, batchnorm_norm_strategy: str = 'classical', batchnorm_minibatch_mode: str = 'constant', batchnorm_minibatch_momentum: float = 0.01, batchnorm_minibatch_maxstep: int = 100, batchnorm_parametrization: str = 'softplus', batchnorm_parametrization_mode: str = 'static', batchnorm_n_steps_ref_update: int = 100, spectrum_type: str = 'conv1d', spectrum_hidden_dim: int = 3, spectrum_n_layers: int = 2, spectrum_kernel_size: int = 5, stiefel_parametrization_mode: str = 'static', stiefel_n_steps_ref_update: int = 100, vec_type: str = 'vec', use_logeig: bool = True, use_autograd: bool | dict = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Flexible multi-stage Riemannian Residual Network on SPD manifold.

   Architecture (for each stage i):
       BiMap(d_{i-1} -> d_i)
       -> [ReEig]
       -> [BatchNorm]
       -> ResidualBlock x n_residual_blocks[i]
   then:
       -> LogEig -> Vec/Vech -> Linear -> [Softmax]

   Inspired by classical ResNet (multi-stage with dimension changes at each
   stage boundary), this architecture is more flexible than GBWBNRResNet:
   it supports multiple BiMap stages, optional ReEig at each stage,
   and multiple residual blocks per stage.

   :param input_dim: Input SPD matrix dimension
   :type input_dim: :py:class:`int`
   :param hidden_layers_size: Dimensions at each stage (after each BiMap)
   :type hidden_layers_size: :py:class:`list[int]`
   :param n_residual_blocks: Number of residual blocks at each stage
   :type n_residual_blocks: :py:class:`list[int]`
   :param output_dim: Number of output classes
   :type output_dim: :py:class:`int`
   :param softmax: Apply softmax. Default is False
   :type softmax: :py:class:`bool`, *optional*
   :param reeig: Apply ReEig after each BiMap. Default is False
   :type reeig: :py:class:`bool`, *optional*
   :param reeig_eps: Minimum eigenvalue for ReEig. Default is 1e-3
   :type reeig_eps: :py:class:`float`, *optional*
   :param bimap_parametrized: Enforce Stiefel on BiMap. Default is True
   :type bimap_parametrized: :py:class:`bool`, *optional*
   :param bimap_parametrization_mode: "static" or "dynamic". Default is "static"
   :type bimap_parametrization_mode: :py:class:`str`, *optional*
   :param bimap_parametrization_options: Options for BiMap parametrization. Default is None
   :type bimap_parametrization_options: :py:class:`dict | None`, *optional*
   :param bimap_n_steps_ref_update: Steps between reference updates. Default is 100
   :type bimap_n_steps_ref_update: :py:class:`int`, *optional*
   :param batchnorm: Apply batchnorm at each stage. Default is False
   :type batchnorm: :py:class:`bool`, *optional*
   :param batchnorm_type: "mean_only" or "mean_var_scalar". Default is "mean_only"
   :type batchnorm_type: :py:class:`str`, *optional*
   :param batchnorm_mean_type: SPD mean type. Default is "affine_invariant"
   :type batchnorm_mean_type: :py:class:`str`, *optional*
   :param batchnorm_mean_options: Options for mean computation. Default is None
   :type batchnorm_mean_options: :py:class:`dict | None`, *optional*
   :param batchnorm_momentum: Running mean momentum. Default is 0.01
   :type batchnorm_momentum: :py:class:`float`, *optional*
   :param batchnorm_norm_strategy: "classical" or "minibatch". Default is "classical"
   :type batchnorm_norm_strategy: :py:class:`str`, *optional*
   :param batchnorm_minibatch_mode: "constant", "decay", or "growth". Default is "constant"
   :type batchnorm_minibatch_mode: :py:class:`str`, *optional*
   :param batchnorm_minibatch_momentum: Minibatch momentum. Default is 0.01
   :type batchnorm_minibatch_momentum: :py:class:`float`, *optional*
   :param batchnorm_minibatch_maxstep: Max step for momentum schedule. Default is 100
   :type batchnorm_minibatch_maxstep: :py:class:`int`, *optional*
   :param batchnorm_parametrization: "softplus" or "exp". Default is "softplus"
   :type batchnorm_parametrization: :py:class:`str`, *optional*
   :param batchnorm_parametrization_mode: "static" or "dynamic". Default is "static"
   :type batchnorm_parametrization_mode: :py:class:`str`, *optional*
   :param batchnorm_n_steps_ref_update: Steps between BN reference updates. Default is 100
   :type batchnorm_n_steps_ref_update: :py:class:`int`, *optional*
   :param spectrum_type: "conv1d" or "mlp". Default is "conv1d"
   :type spectrum_type: :py:class:`str`, *optional*
   :param spectrum_hidden_dim: Hidden dimension for spectrum network. Default is 3
   :type spectrum_hidden_dim: :py:class:`int`, *optional*
   :param spectrum_n_layers: Hidden layers in spectrum network. Default is 2
   :type spectrum_n_layers: :py:class:`int`, *optional*
   :param spectrum_kernel_size: Kernel size for Conv1d. Default is 5
   :type spectrum_kernel_size: :py:class:`int`, *optional*
   :param stiefel_parametrization_mode: Parametrization mode for Q matrices. Default is "static"
   :type stiefel_parametrization_mode: :py:class:`str`, *optional*
   :param stiefel_n_steps_ref_update: Steps between Q reference updates. Default is 100
   :type stiefel_n_steps_ref_update: :py:class:`int`, *optional*
   :param vec_type: "vec" or "vech". Default is "vec"
   :type vec_type: :py:class:`str`, *optional*
   :param use_logeig: Apply LogEig before vectorization. Default is True
   :type use_logeig: :py:class:`bool`, *optional*
   :param use_autograd: Autograd control. Bool for all, dict with keys:
                        'bimap', 'reeig', 'logeig', 'batchnorm', 'vec', 'residual'.
                        Default is False
   :type use_autograd: :py:class:`bool | dict`, *optional*
   :param device: Device. Default is torch.device("cpu")
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Data type. Default is torch.float64
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator for reproducibility. Default is None
   :type generator: :py:class:`torch.Generator | None`, *optional*


   .. py:attribute:: input_dim


   .. py:attribute:: hidden_layers_size


   .. py:attribute:: n_residual_blocks


   .. py:attribute:: output_dim


   .. py:attribute:: softmax
      :value: False



   .. py:attribute:: reeig
      :value: False



   .. py:attribute:: reeig_eps
      :value: 0.001



   .. py:attribute:: batchnorm
      :value: False



   .. py:attribute:: batchnorm_type
      :value: 'mean_only'



   .. py:attribute:: batchnorm_mean_type
      :value: 'affine_invariant'



   .. py:attribute:: vec_type
      :value: 'vec'



   .. py:attribute:: use_logeig
      :value: True



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: generator
      :value: None



   .. py:attribute:: spd_layers


   .. py:attribute:: linear


   .. py:method:: forward(X: torch.Tensor) -> torch.Tensor

      Forward pass.

      :param X: Input SPD matrices
      :type X: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`input_dim`, :py:class:`input_dim)`

      :returns: Output predictions
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`output_dim)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register optimizer hooks for all dynamic parametrizations.



   .. py:method:: get_last_tensor(X: torch.Tensor) -> torch.Tensor

      Return the last SPD tensor before vectorization.



   .. py:method:: __repr__() -> str


   .. py:method:: layers_str() -> str

      Return a formatted string listing the layers.



   .. py:method:: create_model_name_hash() -> str

      Create a short hash of the model configuration.



   .. py:method:: get_model_hash() -> str

      Return the model hash, creating it if needed.



.. py:class:: SPDnet(input_dim: int, hidden_layers_size: list[int], output_dim: int, softmax: bool = False, reeig_eps: float = 0.001, bimap_parametrized: bool = True, bimap_parametrization_mode: str = 'static', bimap_parametrization_options: dict | None = None, bimap_n_steps_ref_update: int = 100, batchnorm: bool = False, batchnorm_type: str = 'mean_only', batchnorm_mean_type: str = 'geometric_arithmetic_harmonic', batchnorm_mean_options: dict | None = None, batchnorm_momentum: float = 0.01, batchnorm_norm_strategy: str = 'classical', batchnorm_minibatch_mode: str = 'constant', batchnorm_minibatch_momentum: float = 0.01, batchnorm_minibatch_maxstep: int = 100, batchnorm_parametrization: str = 'softplus', batchnorm_parametrization_mode: str = 'static', batchnorm_n_steps_ref_update: int = 100, vec_type: str = 'vec', use_logeig: bool = True, use_autograd: bool | dict = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: input_dim


   .. py:attribute:: hidden_layers_size


   .. py:attribute:: output_dim


   .. py:attribute:: softmax
      :value: False



   .. py:attribute:: reeig_eps
      :value: 0.001



   .. py:attribute:: bimap_parametrized
      :value: True



   .. py:attribute:: bimap_parametrization_mode
      :value: 'static'



   .. py:attribute:: bimap_parametrization_options
      :value: None



   .. py:attribute:: bimap_n_steps_ref_update
      :value: 100



   .. py:attribute:: batchnorm
      :value: False



   .. py:attribute:: batchnorm_type
      :value: 'mean_only'



   .. py:attribute:: batchnorm_mean_type
      :value: 'geometric_arithmetic_harmonic'



   .. py:attribute:: batchnorm_mean_options
      :value: None



   .. py:attribute:: batchnorm_momentum
      :value: 0.01



   .. py:attribute:: batchnorm_norm_strategy
      :value: 'classical'



   .. py:attribute:: batchnorm_minibatch_mode
      :value: 'constant'



   .. py:attribute:: batchnorm_minibatch_momentum
      :value: 0.01



   .. py:attribute:: batchnorm_minibatch_maxstep
      :value: 100



   .. py:attribute:: batchnorm_parametrization
      :value: 'softplus'



   .. py:attribute:: batchnorm_parametrization_mode
      :value: 'static'



   .. py:attribute:: batchnorm_n_steps_ref_update
      :value: 100



   .. py:attribute:: vec_type
      :value: 'vec'



   .. py:attribute:: use_logeig
      :value: True



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: generator
      :value: None



   .. py:attribute:: spdnet_layers


   .. py:method:: forward(X: torch.Tensor) -> torch.Tensor

      Forward pass of SPDnet

      :param X: Input tensor of shape (..., input_dim, input_dim)
      :type X: :py:class:`torch.Tensor`

      :returns: Output tensor of shape (..., output_dim)
      :rtype: :py:class:`torch.Tensor`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register optimizer hooks for all layers with dynamic parametrization.
      This method automatically finds all layers that use dynamic parametrization
      and registers the appropriate hooks

      :param optimizer: The optimizer used for training
      :type optimizer: :py:class:`torch.optim.Optimizer`



   .. py:method:: __repr__() -> str

      String representation of SPDnet



   .. py:method:: layers_str() -> str

      Return a formatted string listing the layers of SPDnet.



   .. py:method:: get_last_tensor(X: torch.Tensor) -> torch.Tensor

      Returns the last tensor of SPDNet rather than the output of the
      final layer

      :param X: Input tensor of shape (..., input_dim, input_dim)
      :type X: :py:class:`torch.Tensor`

      :returns: Last tensor of SPDnet
      :rtype: :py:class:`torch.Tensor`



   .. py:method:: create_model_name_hash() -> str

      Creates a very short hash of the model name based on the model parameters
      :returns: Short hash of model name (8 characters)
      :rtype: :py:class:`str`



   .. py:method:: get_model_hash() -> str

      Returns the model hash
      :returns: Model hash
      :rtype: :py:class:`str`



