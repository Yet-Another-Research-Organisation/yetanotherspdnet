from collections.abc import Callable

import torch
from torch import nn
from torch.nn.utils import parametrizations

from ..functions.spd_linalg import (
    CongruenceRectangular,
    EighReLu,
    ExpmSymmetric,
    LogmSPD,
    VecBatch,
    VechBatch,
    congruence_rectangular,
    eigh_relu,
    logm_SPD,
    vec_batch,
)
from ..random.stiefel import _init_weights_stiefel


class BiMap(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        parametrized: bool = True,
        parametrization: Callable = parametrizations.orthogonal,
        parametrization_options: dict | None = None,
        init_method: Callable = _init_weights_stiefel,
        init_options: dict | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        use_autograd: bool = False,
    ) -> None:
        """
        BiMap layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        n_in : int
            Number of input features.

        n_out : int
            Number of output features.

        parametrized : bool, optional
            Whether to apply parametrization to enforce manifold constraints.
            Default is True.

        parametrization : Callable, optional
            Parametrization to apply if parametrized is True.
            Default is parametrizations.orthogonal.

        parametrization_options : dict, optional
            Options for the parametrization function.
            Default is None.

        init_method : Callable, optional
            Initialization method for the weight matrix.
            Default is _init_weights_stiefel.

        init_options : dict, optional
            Options for the init_method function.
            Default is None.

        seed : int, optional
            Seed for the initialization of the weight matrix. Default is None.

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64.

        device : torch.device, optional
            Device on which the layer is initialized.
            Default is torch.device("cpu").

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.seed = seed
        self.dtype = dtype
        self.use_autograd = use_autograd
        self.parametrized = parametrized
        self.parametrization = parametrization
        self.parametrization_options = parametrization_options
        self.init_method = init_method
        self.init_options = init_options
        self.dim = n_out
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        if n_out > n_in:
            raise ValueError("must have n_out < n_in")

        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")

        # Create weight parameter
        self.weight = nn.Parameter(
            torch.empty((n_out, n_in), dtype=dtype, device=device), requires_grad=True
        )
        # Initialize weights
        with torch.no_grad():
            if self.init_options is None:
                self.init_method(self.weight, generator=self.generator)
            else:
                self.init_method(
                    self.weight, generator=self.generator, **self.init_options
                )

        if self.parametrized:
            if self.parametrization_options is None:
                self.parametrization(module=self, name="weight")
            else:
                self.parametrization(
                    module=self, name="weight", **self.parametrization_options
                )

        self.bimap_fun = (
            congruence_rectangular if self.use_autograd else CongruenceRectangular.apply
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiMap layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_in, n_in)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_out, n_out)
            Batch of transformed SPD matrices
        """
        return self.bimap_fun(data, self.weight)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"BiMap(n_in={self.n_in}, n_out={self.n_out}, "
            f"seed={self.seed}, dtype={self.dtype}, device={self.device}, "
            f"use_autograd={self.use_autograd}, parametrized={self.parametrized}, "
            f"parametrization={self.parametrization}, "
            f"parametrization_options={self.parametrization_options}, "
            f"init_method={self.init_method}, init_options={self.init_options})"
        )

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class ReEig(nn.Module):
    def __init__(
        self, eps: float = 1e-4, use_autograd: bool = False, dim: int | None = None
    ) -> None:
        """
        ReEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-2.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        dim : int, optional
            Dimension of the SPD matrices. Default is None.
            Used for logging purposes.
        """
        super().__init__()
        self.eps = eps
        self.use_autograd = use_autograd
        self.dim = dim

        self.reeig_fun = (lambda data, eps: eigh_relu(data, eps)[0]) if self.use_autograd else EighReLu.apply

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ReEig layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed SPD matrices
        """
        return self.reeig_fun(data, self.eps)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"ReEig(eps={self.eps},use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class LogEig(nn.Module):
    def __init__(self, use_autograd: bool = False) -> None:
        """
        LogEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.logmSPD = (lambda data: logm_SPD(data)[0]) if self.use_autograd else LogmSPD.apply

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LogEig layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        Returns
        -------
        data_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of transformed symmetric matrices
        """
        return self.logmSPD(data)

    def __repr__(self) -> str:
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"LogEig(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class Vec(nn.Module):
    def __init__(self, use_autograd: bool = False):
        """
        Vectorization operator of a batch of matrices according to
        the last two dimensions

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.vecBatch = vec_batch if self.use_autograd else VecBatch.apply

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vec layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_rows, n_columns)
            Batch of matrices

        Returns
        -------
        data_vec : torch.Tensor of shape (..., n_rows*n_columns)
            Batch of vectorized matrices
        """
        return self.vecBatch(data)

    def __repr__(self) -> str:
        """Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f"Vec(use_autograd={self.use_autograd})"

    def __str__(self) -> str:
        """String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class Vech(nn.Module):
    def __init__(self) -> None:
        """
        Vech operator of a batch of matrices according to the last two dimensions

        WARNING : no automatic differentiation available here because it fails
        """
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vech layer

        WARNING : no automatic differentiation available here because it fails

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        Returns
        -------
        data_vech: torch.Tensor of shape (..., n_features*(n_features+1)//2)
            Batch of vech matrices
        """
        return VechBatch.apply(data)


class SPDLogEuclideanParametrization(torch.nn.Module):
    """
    SPD parametrization using Log-Euclidean metric.
    Input: Symmetric matrix V of shape (..., n, n)
    Output: exp(V) (guaranteed SPD) of shape (..., n, n)
    Works on both single matrices and batches.
    """
    def forward(self, symmetric_matrix):
        """
        Args:
            symmetric_matrix: Symmetric matrix parameter of shape (..., n, n)
        Returns:
            SPD matrix exp(symmetric_matrix) of shape (..., n, n)
        """
        return ExpmSymmetric.apply(symmetric_matrix)
