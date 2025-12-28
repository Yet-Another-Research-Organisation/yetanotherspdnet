yetanotherspdnet.model
======================

.. py:module:: yetanotherspdnet.model


Classes
-------

.. autoapisummary::

   yetanotherspdnet.model.SPDnet


Module Contents
---------------

.. py:class:: SPDnet(input_dim: int, hidden_layers_size: list[int], output_dim: int, softmax: bool = False, reeig_eps: float = 0.001, bimap_parametrized: bool = True, bimap_parametrization: type[torch.nn.Module] | collections.abc.Callable = parametrizations.orthogonal, bimap_parametrization_options: dict | None = None, batchnorm: bool = False, batchnorm_type: str = 'mean_only', batchnorm_mean_type: str = 'geometric_arithmetic_harmonic', batchnorm_mean_options: dict | None = None, batchnorm_momentum: float = 0.01, batchnorm_norm_strategy: str = 'classical', batchnorm_minibatch_mode: str = 'constant', batchnorm_minibatch_momentum: float = 0.01, batchnorm_minibatch_maxstep: int = 100, batchnorm_parametrization: str = 'softplus', vec_type: str = 'vec', use_logeig: bool = True, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None, use_autograd: bool | dict = False)

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


   .. py:attribute:: input_dim


   .. py:attribute:: hidden_layers_size


   .. py:attribute:: output_dim


   .. py:attribute:: softmax
      :value: False



   .. py:attribute:: reeig_eps
      :value: 0.001



   .. py:attribute:: bimap_parametrized
      :value: True



   .. py:attribute:: bimap_parametrization


   .. py:attribute:: bimap_parametrization_options
      :value: None



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



   .. py:method:: __repr__() -> str

      String representation of SPDnet



   .. py:method:: __str__() -> str

      String representation of SPDnet

      :returns: String representation of SPDnet
      :rtype: :py:class:`str`



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



