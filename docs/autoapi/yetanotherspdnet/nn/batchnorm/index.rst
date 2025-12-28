yetanotherspdnet.nn.batchnorm
=============================

.. py:module:: yetanotherspdnet.nn.batchnorm


Classes
-------

.. autoapisummary::

   yetanotherspdnet.nn.batchnorm.BatchNormSPDMean
   yetanotherspdnet.nn.batchnorm.BatchNormSPDMeanScalarVariance


Module Contents
---------------

.. py:class:: BatchNormSPDMean(n_features: int, mean_type: str = 'affine_invariant', mean_options: dict | None = None, momentum: float = 0.01, norm_strategy: str = 'classical', minibatch_mode: str = 'constant', minibatch_momentum: float = 0.01, minibatch_maxstep: int = 100, parametrization: str = 'softplus', use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
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

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

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



   .. py:attribute:: parametrization
      :value: 'softplus'



   .. py:attribute:: Covbias


   .. py:attribute:: normalize_mean


   .. py:attribute:: add_bias_mean


   .. py:attribute:: running_mean


   .. py:method:: minibatch_momentum_decay() -> float

      Function to compute the minibatch momentum with minibatch_mode == "decay"

      :returns: **minibatch_momentum** -- decreased minibatch momentum
      :rtype: :py:class:`float`



   .. py:method:: minibatch_momentum_growth() -> float

      Function to compute the minibatch momentum for minibatch_mode == "growth"

      :returns: **minibatch_momentum** -- increased minibatch momentum
      :rtype: :py:class:`float`



   .. py:method:: forward(data: torch.Tensor) -> torch.Tensor

      Forward pass of the BatchNormSPDMean layer

      :param data: Batch of SPD matrices
      :type data: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`

      :returns: **data_transformed** -- Batch of transformed (normalized then biased) SPD matrices
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (...`, :py:class:`n_features`, :py:class:`n_features)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



   .. py:method:: __str__() -> str

      String representation of the layer

      :returns: String representation of the layer
      :rtype: :py:class:`str`



.. py:class:: BatchNormSPDMeanScalarVariance(n_features: int, mean_type: str = 'affine_invariant', mean_options: dict | None = None, momentum: float = 0.01, norm_strategy: str = 'classical', minibatch_mode: str = 'constant', minibatch_momentum: float = 0.01, minibatch_maxstep: int = 100, parametrization: str = 'softplus', use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
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

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

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



   .. py:attribute:: parametrization
      :value: 'softplus'



   .. py:attribute:: Covbias


   .. py:attribute:: stdScalarbias


   .. py:attribute:: normalize_mean


   .. py:attribute:: add_bias_mean


   .. py:attribute:: norm_and_bias_var


   .. py:attribute:: running_mean


   .. py:attribute:: running_std_scalar


   .. py:method:: adaptive_std_fun(running_std_scalar: torch.Tensor, std_scalar_batch: torch.Tensor, momentum: float) -> torch.Tensor


   .. py:method:: minibatch_momentum_decay() -> float

      Function to compute the minibatch momentum with minibatch_mode == "decay"

      :returns: **minibatch_momentum** -- decreased minibatch momentum
      :rtype: :py:class:`float`



   .. py:method:: minibatch_momentum_growth() -> float

      Function to compute the minibatch momentum for minibatch_mode == "growth"

      :returns: **minibatch_momentum** -- increased minibatch momentum
      :rtype: :py:class:`float`



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



   .. py:method:: __str__() -> str

      String representation of the layer

      :returns: String representation of the layer
      :rtype: :py:class:`str`



