"""Core SPDNet layers: BiMap, ReEig, LogEig, Vec, Vech."""

import torch
from torch import nn
from torch.nn.utils import parametrizations
from torch.nn.utils.parametrize import register_parametrization

from yetanotherspdnet.nn.parametrizations import StiefelAdaptiveParametrization

from ..functions.spd_linalg import (
    CongruenceRectangular,
    EighReLu,
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
    r"""
    Bilinear mapping layer :math:`X \mapsto W^\top X W` reducing SPD dimension.

    With :math:`W \in \mathbb{R}^{n_{in} \times n_{out}}` on the Stiefel manifold
    (:math:`W^\top W = I`), a congruence maps SPD matrices to SPD matrices of
    smaller size. Introduced in Huang & Van Gool, *A Riemannian Network for SPD
    Matrix Learning*, AAAI 2017.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        parametrized: bool = True,
        parametrization_mode: str = "static",
        parametrization_options: dict | None = None,
        n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Build a BiMap layer.

        Parameters
        ----------
        n_in : int
            Number of input features

        n_out : int
            Number of output features

        parametrized : bool, optional
            Whether to apply parametrization to enforce orthogonal constraint.
            Default is True

        parametrization_mode : str, optional
            Parametrization mode.
            Default is "static".
            Choices are: "static" and "dynamic"

        parametrization_options : dict, optional
            Options for the parametrization function.
            Default is None

        n_steps_ref_update : int, optional
            If parametrization_mode is "dynamic",
            number of steps in between each reference point update.
            Default is 100

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False

        device : torch.device, optional
            Device on which the layer is initialized.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None

        Attributes
        ----------
        weight : torch.Tensor of shape (n_in, n_out)
            Orthonormal weight. When ``parametrized`` it is a
            ``torch.nn.utils.parametrize`` view whose trainable tensor is
            ``parametrizations.weight.original``.
        n_in, n_out : int
            Input and output matrix dimensions (``n_out <= n_in``).
        is_dynamic : bool
            Whether the Stiefel parametrization re-centres its reference point every
            ``n_steps_ref_update`` optimizer steps (``parametrization_mode="dynamic"``).
        use_autograd : bool
            Autograd path (True) or hand-written backward (False, default).
        """
        super().__init__()
        assert n_out <= n_in, "must have n_out <= n_in"
        self.n_in = n_in
        self.n_out = n_out
        self.parametrized = parametrized
        assert parametrization_mode in ["static", "dynamic"], (
            f"expected parametrization_mode in ['static', 'dynamic'], got {parametrization_mode}"
        )
        self.parametrization_mode = parametrization_mode
        self.parametrization_options = parametrization_options
        self.n_steps_ref_update = n_steps_ref_update
        self.use_autograd = use_autograd
        self.device = device
        self.dtype = dtype
        self.generator = generator

        # initialize dynamic parametrization bool
        self.is_dynamic = False
        # create and inintialize weight
        self.weight = nn.Parameter(
            torch.empty((n_in, n_out), dtype=dtype, device=device), requires_grad=True
        )
        with torch.no_grad():
            _init_weights_stiefel(self.weight, generator=self.generator)
        # deal with parametrization
        if self.parametrized and self.parametrization_mode == "static":
            if self.parametrization_options is None:
                parametrizations.orthogonal(module=self, name="weight")
            else:
                parametrizations.orthogonal(
                    module=self, name="weight", **self.parametrization_options
                )
        elif self.parametrized and self.parametrization_mode == "dynamic":
            self.is_dynamic = True
            self.current_ref_step = 0
            if self.parametrization_options is None:
                self.stiefel_parametrization = StiefelAdaptiveParametrization(
                    self.n_in,
                    self.n_out,
                    initial_reference=self.weight.clone().detach(),
                )
            else:
                self.stiefel_parametrization = StiefelAdaptiveParametrization(
                    self.n_in,
                    self.n_out,
                    initial_reference=self.weight.clone().detach(),
                    **self.parametrization_options,
                )
            register_parametrization(
                module=self,
                tensor_name="weight",
                parametrization=self.stiefel_parametrization,
            )
        # BiMap function
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
        if self.training and self.is_dynamic:
            self.current_ref_step += 1
        return self.bimap_fun(data, self.weight)

    def _post_optimizer_hook(
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
    ) -> None:
        """
        Hook that runs after optimizer.step() to handle dynamic parametrization reference point.
        See torch.optim.Optimizer.register_step_post_hook method for more details

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if (
            self.training
            and self.is_dynamic
            and self.current_ref_step >= self.n_steps_ref_update
        ):
            with torch.no_grad():
                # update reference point
                self.stiefel_parametrization.update_reference_point()
                # reset tangent vector to zero
                self.parametrizations.weight.original.zero_()
                # reset counter
                self.current_ref_step = 0

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Register the post-step hook with the optimizer.
        If dynamic parametrization, it needs to be called once after creating
        the optimizer for dynamic parametrization to actually work as expected

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Torch optimizer used for training
        """
        if self.is_dynamic:
            optimizer.register_step_post_hook(self._post_optimizer_hook)

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
            f"parametrized={self.parametrized}, parametrization_mode={self.parametrization_mode}, "
            f"parametrization_options={self.parametrization_options}, "
            f"n_steps_ref_update={self.n_steps_ref_update}, "
            f"use_autograd={self.use_autograd}, "
            f"device={self.device}, dtype={self.dtype}, generator={self.generator})"
        )


class ReEig(nn.Module):
    r"""
    Eigenvalue rectification :math:`X \mapsto U \max(\Lambda, \epsilon I) U^\top`.

    The SPD analogue of ReLU: eigenvalues below ``eps`` are clamped, which keeps
    the output well conditioned. Non-linear on the manifold, it plays the role of
    the activation in SPDNet.

    """

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
        Attributes
        ----------
        eps : float
            Rectification threshold on the eigenvalues.
        use_autograd : bool
            Autograd path (True) or hand-written Daleckii-Krein backward (False).
        """
        super().__init__()
        self.eps = eps
        self.use_autograd = use_autograd
        self.dim = dim

        self.reeig_fun = (
            (lambda data, eps: eigh_relu(data, eps)[0])
            if self.use_autograd
            else EighReLu.apply
        )

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
        return f"ReEig(eps={self.eps}, use_autograd={self.use_autograd})"


class LogEig(nn.Module):
    r"""
    Matrix logarithm :math:`X \mapsto U \log(\Lambda) U^\top`.

    Maps SPD matrices to the (flat) space of symmetric matrices, usually right
    before vectorization and a Euclidean classifier.

    """

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
        Attributes
        ----------
        use_autograd : bool
            Autograd path (True) or hand-written Daleckii-Krein backward (False).
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.logmSPD = (
            (lambda data: logm_SPD(data)[0]) if self.use_autograd else LogmSPD.apply
        )

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


class Vec(nn.Module):
    r"""
    Full vectorization of a batch of matrices: ``(..., n, n) -> (..., n * n)``.

    """

    def __init__(self, use_autograd: bool = False):
        """
        Vectorization operator of a batch of matrices according to
        the last two dimensions

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        Attributes
        ----------
        use_autograd : bool
            Autograd path (True) or hand-written backward (False).
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


class Vech(nn.Module):
    r"""
    Half vectorization of symmetric matrices: ``(..., n, n) -> (..., n (n + 1) / 2)``.

    Keeps only the upper-triangular part, which carries all the information of a
    symmetric matrix.
    """

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
