yetanotherspdnet.random.spd
===========================

.. py:module:: yetanotherspdnet.random.spd


Functions
---------

.. autoapisummary::

   yetanotherspdnet.random.spd.random_DPD
   yetanotherspdnet.random.spd.random_SPD


Module Contents
---------------

.. py:function:: random_DPD(n_features: int, n_matrices: int = 1, cond: float = 10, device: torch.device | None = None, dtype: torch.dtype | None = None, generator: torch.Generator | None = None) -> torch.Tensor

   Generate a batch of random diagonal positive definite matrices

   :param n_features: Number of features
   :type n_features: :py:class:`int`
   :param n_matrices: Number of matrices. Default is 1
   :type n_matrices: :py:class:`int`, *optional*
   :param cond: Condition number w.r.t. inversion of SPD matrices.
                Default is 10
   :type cond: :py:class:`float`, *optional*
   :param device: Torch device. Default is None.
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Torch dtype. Default is None.
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator to ensure reproducibility. Default is None.
   :type generator: :py:class:`torch.Generator`, *optional*

   :returns: Batch of DPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_matrices`, :py:class:`n_features`, :py:class:`n_features)`


.. py:function:: random_SPD(n_features: int, n_matrices: int = 1, cond: float = 10, device: torch.device | None = None, dtype: torch.dtype | None = None, generator: torch.Generator | None = None) -> torch.Tensor

   Generate a batch of random SPD matrices

   :param n_features: Number of features
   :type n_features: :py:class:`int`
   :param n_matrices: Number of matrices. Default is 1
   :type n_matrices: :py:class:`int`, *optional*
   :param cond: Condition number w.r.t. inversion of SPD matrices.
                Default is 10
   :type cond: :py:class:`float`, *optional*
   :param device: Torch device. Default is None.
   :type device: :py:class:`torch.device`, *optional*
   :param dtype: Torch dtype. Default is None.
   :type dtype: :py:class:`torch.dtype`, *optional*
   :param generator: Generator to ensure reproducibility. Default is None.
   :type generator: :py:class:`torch.Generator`, *optional*

   :returns: Batch of SPD matrices
   :rtype: :py:class:`torch.Tensor` of :py:class:`shape (n_matrices`, :py:class:`n_features`, :py:class:`n_features)`


