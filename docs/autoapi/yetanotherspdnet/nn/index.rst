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


Classes
-------

.. autoapisummary::

   yetanotherspdnet.nn.BiMap
   yetanotherspdnet.nn.LogEig
   yetanotherspdnet.nn.ReEig
   yetanotherspdnet.nn.Vec
   yetanotherspdnet.nn.Vech
   yetanotherspdnet.nn.BatchNormSPDMean


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



