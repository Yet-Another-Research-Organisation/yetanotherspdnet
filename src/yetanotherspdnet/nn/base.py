import torch
from torch import nn
from torch.nn.utils import parametrizations
from torch.nn.utils.parametrize import register_parametrization

from yetanotherspdnet.nn.parametrizations import StiefelAdaptiveParametrization

from ..functions.spd_linalg import (
    CongruenceFullRank,
    EighReLu,
    LogmSPD,
    VecBatch,
    VechBatch,
)
from ..random.stiefel import _init_weights_stiefel


class BiMap(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        parametrized: bool = True,
        parametrization_mode: str = "static",
        parametrization_options: dict | None = None,
        n_steps_ref_update: int = 100,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        BiMap layer for the SPDnet architecture according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

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

        device : torch.device, optional
            Device on which the layer is initialized.
            Default is torch.device("cpu")

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64

        generator : torch.Generator, optional
            Generator to ensure reproducibility. Default is None
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
        self.bimap_fun = CongruenceFullRank.apply

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
        self, optimizer: torch.optim.Optimizer, *args, **kwargs
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
            f"device={self.device}, dtype={self.dtype}, generator={self.generator})"
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
    def __init__(self, eps: float = 1e-4, dim: int | None = None) -> None:
        """
        ReEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-2.

        dim : int, optional
            Dimension of the SPD matrices. Default is None.
            Used for logging purposes.
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.reeig_fun = EighReLu.apply

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
        return f"ReEig(eps={self.eps}, dim={self.dim})"

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
    def __init__(self) -> None:
        """
        LogEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017
        """
        super().__init__()
        self.logmSPD = LogmSPD.apply

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
        return "LogEig()"

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
    def __init__(self):
        """
        Vectorization operator of a batch of matrices according to
        the last two dimensions
        """
        super().__init__()
        self.vecBatch = VecBatch.apply

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
        """
        Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return "Vec()"

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
        """
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vech layer

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
