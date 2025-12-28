yetanotherspdnet.nn.parametrizations
====================================

.. py:module:: yetanotherspdnet.nn.parametrizations


Classes
-------

.. autoapisummary::

   yetanotherspdnet.nn.parametrizations.ScalarSoftPlusParametrization
   yetanotherspdnet.nn.parametrizations.SPDParametrization
   yetanotherspdnet.nn.parametrizations.SPDAdaptiveParametrization
   yetanotherspdnet.nn.parametrizations.StiefelAdaptiveParametrization


Module Contents
---------------

.. py:class:: ScalarSoftPlusParametrization(*args, **kwargs)

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


   .. py:method:: forward(scalar: torch.Tensor) -> torch.Tensor

      Positive definite scalars parametrization using the SoftPlus function
      (rescaled so that f(0) = 1 as compared to default torch function)

      :param scalar: Real scalar
      :type scalar: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: **scalar_pd** -- Positive definite scalar
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



   .. py:method:: right_inverse(scalar_pd: torch.Tensor) -> torch.Tensor

      Mapping from positive definite scalar onto real scalars through
      the inverse SoftPlus function

      :param scalar_pd: Positive definite scalar
      :type scalar_pd: :py:class:`torch.Tensor` of :py:class:`shape ()`

      :returns: **scalar** -- Real scalar
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape ()`



.. py:class:: SPDParametrization(mapping: str = 'softplus', use_autograd: bool = False)

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


   .. py:attribute:: mapping
      :value: 'softplus'



   .. py:attribute:: use_autograd
      :value: False



   .. py:method:: forward(tangent_vector: torch.Tensor) -> torch.Tensor

      Mapping from the tangent space at identity to the SPD manifold

      :param tangent_vector: Symmetric matrix
      :type tangent_vector: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **spd_matrix** -- SPD matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: right_inverse(spd_matrix: torch.Tensor) -> torch.Tensor

      Mapping from the SPD manifold to the tangent space at identity

      :param spd_matrix: SPD matrix
      :type spd_matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **tangent_vector** -- Symmetric matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



   .. py:method:: __str__() -> str

      String representation of the layer

      :returns: String representation of the layer
      :rtype: :py:class:`str`



.. py:class:: SPDAdaptiveParametrization(n_features: int, initial_reference: torch.Tensor | None = None, mapping: str = 'softplus', use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64)

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


   .. py:attribute:: initial_reference
      :value: None



   .. py:attribute:: mapping
      :value: 'softplus'



   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:method:: forward(tangent_vector: torch.Tensor) -> torch.Tensor

      Mapping from the tangent space at reference_point onto the SPD manifold

      :param tangent_vector: Symmetric matrix
      :type tangent_vector: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **spd_matrix**
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: right_inverse(spd_matrix: torch.Tensor) -> torch.Tensor

      Mapping from SPD manifold onto the tangent space at reference_point

      :param spd_matrix: SPD matrix
      :type spd_matrix: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`

      :returns: **tangent_vector** -- Symmetric matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_features`, :py:class:`n_features)`



   .. py:method:: update_reference_point() -> None

      Update reference point with last SPD value



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



   .. py:method:: __str__() -> str

      String representation of the layer

      :returns: String representation of the layer
      :rtype: :py:class:`str`



.. py:class:: StiefelAdaptiveParametrization(n_in: int, n_out: int, initial_reference: torch.Tensor | None = None, mapping: str = 'QR', use_autograd: bool = False, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64, generator: torch.Generator | None = None)

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


   .. py:attribute:: n_in


   .. py:attribute:: n_out


   .. py:attribute:: initial_reference
      :value: None



   .. py:attribute:: mapping
      :value: 'QR'



   .. py:attribute:: use_autograd
      :value: False



   .. py:attribute:: device


   .. py:attribute:: dtype
      :value: Ellipsis



   .. py:attribute:: generator
      :value: None



   .. py:attribute:: projectionTangent


   .. py:method:: forward(weight_tangent: torch.Tensor) -> torch.Tensor

      Mapping from the tangent space of reference_point to the Stiefel manifold

      :param weight_tangent: Rectangular matrix, tangent vector at reference_point
      :type weight_tangent: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **weight** -- Orthogonal matrix
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



   .. py:method:: right_inverse(weight: torch.Tensor) -> torch.Tensor

      Mapping from Stiefel manifold to the tangent space at reference_point
      (achieved through orthogonal projection)

      :param weight: Orthogonal matrix
      :type weight: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`

      :returns: **weight_tangent** -- Rectangular matrix, tangent vector at reference_point
      :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_in`, :py:class:`n_out)`



   .. py:method:: update_reference_point() -> None

      Update reference point with last Stiefel value



   .. py:method:: __repr__() -> str

      Representation of the layer

      :returns: Representation of the layer
      :rtype: :py:class:`str`



   .. py:method:: __str__() -> str

      String representation of the layer

      :returns: String representation of the layer
      :rtype: :py:class:`str`



