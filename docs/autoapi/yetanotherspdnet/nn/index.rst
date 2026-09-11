yetanotherspdnet.nn
===================

.. py:module:: yetanotherspdnet.nn


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/yetanotherspdnet/nn/base/index
   /autoapi/yetanotherspdnet/nn/batchnorm/index
   /autoapi/yetanotherspdnet/nn/parametrizations/index
   /autoapi/yetanotherspdnet/nn/rresnet_layers/index


Classes
-------

.. autoapisummary::

   yetanotherspdnet.nn.BiMap
   yetanotherspdnet.nn.LogEig
   yetanotherspdnet.nn.ReEig
   yetanotherspdnet.nn.Vec
   yetanotherspdnet.nn.Vech
   yetanotherspdnet.nn.BatchNormSPDMean
   yetanotherspdnet.nn.BatchNormSPDMeanScalarVariance
   yetanotherspdnet.nn.ResidualBlock
   yetanotherspdnet.nn.SpectralVectorField


Package Contents
----------------

.. py:class:: BiMap(n_in: int, n_out: int, parametrized: bool = True, parametrization_mode: str = 'static', parametrization_options: dict | None = None, n_steps_ref_update: int = 100, use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

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


   .. py:attribute:: n_in


   .. py:attribute:: n_out


   .. py:attribute:: parametrized
      :value: True



   .. py:attribute:: parametrization_mode
      :value: 'static'



   .. py:attribute:: parametrization_options
      :value: None



   .. py:attribute:: n_steps_ref_update
      :value: 100



   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: generator
      :value: None



   .. py:attribute:: is_dynamic
      :value: False



   .. py:attribute:: weight


   .. py:attribute:: bimap_fun


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the BiMap layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_in`, :py:class:`n_in)`

      :returns: **data_transformed** -- Batch of transformed SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_out`, :py:class:`n_out)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register the post-step hook with the optimizer.
      If dynamic parametrization, it needs to be called once after creating
      the optimizer for dynamic parametrization to actually work as expected

      :param optimizer: Torch optimizer used for training
      :type optimizer: :py:class:`torch.optim.Optimizer`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



.. py:class:: LogEig(use_autograd: bool = False)

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


   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: logmSPD


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the LogEig layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Batch of transformed symmetric matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



.. py:class:: ReEig(eps: float = 0.0001, use_autograd: bool = False, dim: int | None = None)

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


   .. py:attribute:: eps
      :value: 0.0001



   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: dim
      :value: None



   .. py:attribute:: reeig_fun


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the ReEig layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Batch of transformed SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



.. py:class:: Vec(use_autograd: bool = False)

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


   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: vecBatch


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the Vec layer

      :param data: Batch of matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows`, :py:class:`n_columns)`

      :returns: **data_vec** -- Batch of vectorized matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_rows*n_columns)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



.. py:class:: Vech

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


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the Vech layer

      WARNING : no automatic differentiation available here because it fails

      :param data: Batch of symmetric matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_vech** -- Batch of vech matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features*(n_features+1)//2)`



.. py:class:: BatchNormSPDMean(n_features: int, mean_type: str = 'affine_invariant', mean_options: dict | None = None, momentum: float = 0.01, norm_strategy: str = 'classical', minibatch_mode: str = 'constant', minibatch_momentum: float = 0.01, minibatch_maxstep: int = 100, parametrization: str = 'softplus', parametrization_mode: str = 'static', n_steps_ref_update: int = 100, use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64)

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


   .. py:attribute:: n_features


   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: mean_type
      :value: 'affine_invariant'



   .. py:attribute:: mean_options
      :value: None



   .. py:attribute:: momentum
      :value: 0.01



   .. py:attribute:: norm_strategy
      :value: 'classical'



   .. py:attribute:: minibatch_mode
      :value: 'constant'



   .. py:attribute:: minibatch_momentum
      :value: 0.01



   .. py:attribute:: minibatch_maxstep
      :value: 100



   .. py:attribute:: training_step
      :value: 0



   .. py:attribute:: is_dynamic
      :value: False



   .. py:attribute:: parametrization
      :value: 'softplus'



   .. py:attribute:: parametrization_mode
      :value: 'static'



   .. py:attribute:: n_steps_ref_update
      :value: 100



   .. py:attribute:: Covbias


   .. py:attribute:: normalize_mean


   .. py:attribute:: add_bias_mean


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the BatchNormSPDMean layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Batch of transformed (normalized then biased) SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: register_optimizer_hook(optimizer: torch.optim.Optimizer) -> None

      Register the post-step hook with the optimizer.
      If dynamic parametrization, it needs to be called once after creating
      the optimizer for dynamic parametrization to actually work as expected

      :param optimizer: Torch optimizer used for training
      :type optimizer: :py:class:`torch.optim.Optimizer`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



.. py:class:: BatchNormSPDMeanScalarVariance(n_features: int, mean_type: str = 'affine_invariant', mean_options: dict | None = None, momentum: float = 0.01, norm_strategy: str = 'classical', minibatch_mode: str = 'constant', minibatch_momentum: float = 0.01, minibatch_maxstep: int = 100, parametrization: str = 'softplus', parametrization_mode: str = 'static', n_steps_ref_update: int = 100, use_autograd: bool = False, bw_theta: float = 1.0, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64)

   Bases: :py:obj:`BatchNormSPDMean`


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


   .. py:attribute:: stdScalarbias


   .. py:attribute:: norm_and_bias_var


   .. py:attribute:: running_std_scalar


   .. py:method:: adaptive_std_fun(running_std_scalar: torch.Tensor, std_scalar_batch: torch.Tensor, momentum: float) -> torch.Tensor


   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the BatchNormSPDMeanScalarVariance layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Batch of transformed (normalized then biased) SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



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


