yetanotherspdnet.random.stiefel
===============================

.. py:module:: yetanotherspdnet.random.stiefel


Functions
---------

.. autoapisummary::

   yetanotherspdnet.random.stiefel.random_stiefel


Module Contents
---------------

.. py:function:: random_stiefel(n_in: int, n_out: int, n_matrices: int = 1, device: torch.device | None = None, dtype: torch.dtype | None = None, generator: torch.Generator | None = None) -> torch.Tensor

   Random point on Stiefel manifold

   :param n_in: Number of rows
   :type n_in: :py:class:`int`
   :param n_out: Number of columns
   :type n_out: :py:class:`int`
   :param n_matrices: Number of matrices. Default is 1
   :type n_matrices: :py:class:`int`, *optional*
   :param device: Torch device. Default is None
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Torch dtype. Default is None
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator to ensure reproducibility. Default is None
   :type generator: :py:class:`torch.Generator`, *optional*


