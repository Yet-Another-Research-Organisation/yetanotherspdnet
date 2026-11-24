"""Riemannian Residual Network layers: spectral vector field and residual block."""

import torch
from torch import nn

from yetanotherspdnet._deprecation import _deprecated
from yetanotherspdnet.functions._constants import (
    SPD_EIGVAL_CEILING,
    SPD_EIGVAL_FLOOR,
    TANGENT_NORM_FLOOR,
)
from yetanotherspdnet.functions.dispatch import select_path
from yetanotherspdnet.functions.spd_geometries import affine_invariant
from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    AffineInvariantExp,
    AffineInvariantNorm,
    affine_invariant_exp,
)
from yetanotherspdnet.functions.spd_linalg import (
    EighClamp,
    EighReLu,
    EigvalshSymmetric,
    ExpmSymmetric,
    LogmSPD,
    eigh_clamp,
    eigh_relu,
    eigvalsh_symmetric,
    expm_symmetric,
    logm_SPD,
    spd_floor,
    symmetrize,
)
from yetanotherspdnet.nn.parametrizations import (
    StiefelAdaptiveParametrization,
)
from yetanotherspdnet.random.stiefel import random_stiefel


# absolute eigenvalue floor applied to the input of ResidualBlock (lower bound of
# the eigenvalue clamp of its output, as affine_invariant_projx of the reference
# code); the floor actually used is spd_floor(eigenvalues, INPUT_EIGVAL_FLOOR),
# raised to the rounding level c n eps max(lambda) of the dtype
INPUT_EIGVAL_FLOOR = SPD_EIGVAL_FLOOR


def _input_floor(eigvals: torch.Tensor) -> torch.Tensor:
    """Dtype-aware floor of the ResidualBlock eigenvalues (see spd_floor)."""
    return spd_floor(eigvals, floor=INPUT_EIGVAL_FLOOR)


class SpectralVectorField(nn.Module):
    """
    Spectral vector field on SPD manifold.

    Computes a tangent vector at X as:
        V = Q diag(f(spec(X))) Q^T

    where:
    - spec(X): eigenvalues of the input SPD matrix
    - f: learnable spectrum mapping (Conv1d or MLP)
    - Q: learnable orthogonal matrix (Stiefel-parametrized)

    The output is a symmetric matrix in the tangent space at X.

    Parameters
    ----------
    n_features : int
        Dimension of SPD matrices (n x n)

    spectrum_type : str, optional
        Type of spectrum mapping network.
        Default is "conv1d".
        Choices are: "conv1d" and "mlp"

    spectrum_hidden_dim : int, optional
        Hidden dimension for spectrum mapping. Default is 3

    spectrum_n_layers : int, optional
        Number of hidden layers in spectrum mapping. Default is 2

    spectrum_kernel_size : int, optional
        Kernel size for Conv1d spectrum mapping. Default is 5

    stiefel_parametrization_mode : str, optional
        Parametrization mode for orthogonal matrix Q.
        Default is "static".
        Choices are: "static" and "dynamic"

    stiefel_n_steps_ref_update : int, optional
        Steps between reference point updates for dynamic parametrization.
        Default is 100

    use_autograd : bool, optional
        Use torch autograd for gradient computation. Default is False

    device : torch.device, optional
        Device. Default is torch.device("cpu")

    dtype : torch.dtype, optional
        Data type. Default is torch.float64

    generator : torch.Generator | None, optional
        Generator for reproducibility. Default is None
    """

    def __init__(
        self,
        n_features: int,
        spectrum_type: str = "conv1d",
        spectrum_hidden_dim: int = 3,
        spectrum_n_layers: int = 2,
        spectrum_kernel_size: int = 5,
        stiefel_parametrization_mode: str = "static",
        stiefel_n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.spectrum_type = spectrum_type
        if spectrum_type not in ("conv1d", "mlp"):
            raise ValueError(
                f"spectrum_type must be 'conv1d' or 'mlp', got {spectrum_type}"
            )
        self.use_autograd = use_autograd
        self.eigvalsh = select_path(
            eigvalsh_symmetric, EigvalshSymmetric.apply, use_autograd
        )

        # Learnable orthogonal matrix Q on Stiefel(n, n)
        self.Q = nn.Parameter(
            random_stiefel(
                n_features,
                n_features,
                n_matrices=1,
                device=device,
                dtype=dtype,
                generator=generator,
            ).squeeze(0)
        )

        # Stiefel parametrization
        self.is_dynamic = stiefel_parametrization_mode == "dynamic"
        self.current_ref_step = 0
        self.n_steps_ref_update = stiefel_n_steps_ref_update
        nn.utils.parametrize.register_parametrization(
            self,
            "Q",
            StiefelAdaptiveParametrization(
                n_in=n_features,
                n_out=n_features,
                mapping="QR",
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
                generator=generator,
            ),
        )

        # Spectrum mapping network
        padding = (spectrum_kernel_size - 1) // 2
        if spectrum_type == "conv1d":
            layers: list[nn.Module] = [
                nn.Conv1d(
                    1,
                    spectrum_hidden_dim,
                    spectrum_kernel_size,
                    padding=padding,
                    dtype=dtype,
                    device=device,
                ),
            ]
            for _ in range(spectrum_n_layers - 1):
                layers.extend(
                    [
                        nn.LeakyReLU(negative_slope=0.5),
                        nn.BatchNorm1d(spectrum_hidden_dim, dtype=dtype, device=device),
                        nn.Conv1d(
                            spectrum_hidden_dim,
                            spectrum_hidden_dim,
                            spectrum_kernel_size,
                            padding=padding,
                            dtype=dtype,
                            device=device,
                        ),
                    ]
                )
            layers.extend(
                [
                    nn.LeakyReLU(negative_slope=0.5),
                    nn.BatchNorm1d(spectrum_hidden_dim, dtype=dtype, device=device),
                    nn.Conv1d(
                        spectrum_hidden_dim,
                        1,
                        spectrum_kernel_size,
                        padding=padding,
                        dtype=dtype,
                        device=device,
                    ),
                ]
            )
            self.spectrum_map = nn.Sequential(*layers)
        else:
            layers_mlp: list[nn.Module] = [
                nn.Linear(n_features, spectrum_hidden_dim, dtype=dtype, device=device),
            ]
            for _ in range(spectrum_n_layers - 1):
                layers_mlp.extend(
                    [
                        nn.LeakyReLU(negative_slope=0.5),
                        nn.Linear(
                            spectrum_hidden_dim,
                            spectrum_hidden_dim,
                            dtype=dtype,
                            device=device,
                        ),
                    ]
                )
            layers_mlp.extend(
                [
                    nn.LeakyReLU(negative_slope=0.5),
                    nn.Linear(
                        spectrum_hidden_dim, n_features, dtype=dtype, device=device
                    ),
                ]
            )
            self.spectrum_map = nn.Sequential(*layers_mlp)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute tangent vector from SPD matrix spectrum.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n, n)
            Batch of SPD matrices

        Returns
        -------
        tangent : torch.Tensor of shape (..., n, n)
            Symmetric tangent vectors at each input point
        """
        if self.training and self.is_dynamic:
            self.current_ref_step += 1

        batch_shape = data.shape[:-2]
        n = self.n_features

        # Extract eigenvalues (sorted ascending)
        eigenvalues = self.eigvalsh(data)  # (..., n)

        # Apply spectrum mapping
        flat_eigs = eigenvalues.reshape(-1, n)  # (B, n)
        if self.spectrum_type == "conv1d":
            # Conv1d expects (B, C, L) with C=1
            f_eigs = self.spectrum_map(flat_eigs.unsqueeze(1)).squeeze(1)
        else:
            f_eigs = self.spectrum_map(flat_eigs)
        f_eigs = f_eigs.reshape(*batch_shape, n)  # (..., n)

        # Construct tangent vector: Q @ diag(f_eigs) @ Q^T
        Q = self.Q  # (n, n) orthogonal
        tangent = (Q * f_eigs.unsqueeze(-2)) @ Q.transpose(-1, -2)
        return tangent

    def _post_optimizer_hook(
        self, _optimizer: torch.optim.Optimizer, *_args, **_kwargs
    ) -> None:
        """
        Update the dynamic Stiefel reference point after ``optimizer.step()``.

        Same logic as BiMap. See ``torch.optim.Optimizer.register_step_post_hook`` for
        the signature.

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
                self.parametrizations.Q[0].update_reference_point()
                # reset tangent vector to zero, so that Q stays at the new
                # reference point instead of jumping by the old tangent vector
                self.parametrizations.Q.original.zero_()
                # reset counter
                self.current_ref_step = 0

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hook for dynamic Stiefel parametrization."""
        if self.is_dynamic:
            optimizer.register_step_post_hook(self._post_optimizer_hook)

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        return (
            f"SpectralVectorField(n_features={self.n_features}, "
            f"spectrum_type='{self.spectrum_type}', "
            f"use_autograd={self.use_autograd})"
        )


class ResidualBlock(nn.Module):
    """
    Riemannian residual block on SPD manifold.

    Applies one residual step along the vector field V = VF(X) (a
    SpectralVectorField), following Katsman et al., *Riemannian Residual
    Neural Networks* (NeurIPS 2023) and their reference implementation:

    - ``metric="affine_invariant"``: X_new = clamp(Exp_X(V / ||V||_X)), with the eigenvalues clamped to
      [1e-8, 1e8] (projx of the reference implementation), a step
      of unit affine-invariant length. Without this normalization,
      ||V||_X = ||X^{-1/2} V X^{-1/2}||_F grows like 1 / lambda_min(X) and the
      matrix exponential overflows on ill-conditioned inputs.
    - ``metric="log_euclidean"``: X_new = exp(log(X) + V).

    Parameters
    ----------
    n_features : int
        Dimension of SPD matrices

    spectrum_type : str, optional
        Type of spectrum mapping. Default is "conv1d"

    spectrum_hidden_dim : int, optional
        Hidden dimension for spectrum mapping. Default is 3

    spectrum_n_layers : int, optional
        Hidden layers in spectrum mapping. Default is 2

    spectrum_kernel_size : int, optional
        Kernel size for Conv1d. Default is 5

    stiefel_parametrization_mode : str, optional
        Parametrization mode for Q matrix. Default is "static"

    stiefel_n_steps_ref_update : int, optional
        Steps between reference updates. Default is 100

    use_autograd : bool, optional
        Use autograd for exp map gradient. Default is False

    metric : str, optional
        "affine_invariant" (default) or "log_euclidean", see above

    device : torch.device, optional
        Device. Default is torch.device("cpu")

    dtype : torch.dtype, optional
        Data type. Default is torch.float64

    generator : torch.Generator | None, optional
        Generator. Default is None
    """

    def __init__(
        self,
        n_features: int,
        spectrum_type: str = "conv1d",
        spectrum_hidden_dim: int = 3,
        spectrum_n_layers: int = 2,
        spectrum_kernel_size: int = 5,
        stiefel_parametrization_mode: str = "static",
        stiefel_n_steps_ref_update: int = 100,
        use_autograd: bool = False,
        metric: str = "affine_invariant",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if metric not in ("affine_invariant", "log_euclidean"):
            raise ValueError(
                f"metric must be 'affine_invariant' or 'log_euclidean', got {metric}"
            )
        self.n_features = n_features
        self.use_autograd = use_autograd
        self.metric = metric

        self.vector_field = SpectralVectorField(
            n_features=n_features,
            spectrum_type=spectrum_type,
            spectrum_hidden_dim=spectrum_hidden_dim,
            spectrum_n_layers=spectrum_n_layers,
            spectrum_kernel_size=spectrum_kernel_size,
            stiefel_parametrization_mode=stiefel_parametrization_mode,
            stiefel_n_steps_ref_update=stiefel_n_steps_ref_update,
            use_autograd=use_autograd,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Select implementations (autograd or manual backward)
        self.exp_map = select_path(
            affine_invariant_exp, AffineInvariantExp.apply, use_autograd
        )
        self.logm = select_path(lambda x: logm_SPD(x)[0], LogmSPD.apply, use_autograd)
        self.expm = select_path(
            lambda x: expm_symmetric(x)[0], ExpmSymmetric.apply, use_autograd
        )
        floor = select_path(
            lambda x, eps: eigh_relu(x, eps)[0], EighReLu.apply, use_autograd
        )
        self.floor = lambda x: floor(x, _input_floor)
        self.norm = select_path(
            affine_invariant.affine_invariant_norm,
            AffineInvariantNorm.apply,
            use_autograd,
        )
        self.clamp = select_path(
            lambda x, lo, hi: eigh_clamp(x, lo, hi)[0], EighClamp.apply, use_autograd
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: one residual step (see the class docstring).

        Parameters
        ----------
        data : torch.Tensor of shape (..., n, n)
            Batch of SPD matrices

        Returns
        -------
        result : torch.Tensor of shape (..., n, n)
            Updated SPD matrices after one residual step
        """
        # the input may be numerically singular (e.g. a Bures-Wasserstein batch
        # normalization folds matrices outside the injectivity domain of its
        # exponential): floor its eigenvalues, a no-op on well-conditioned
        # inputs, so that the Cholesky of the norm and the logarithm are defined
        data = self.floor(data)
        tangent = self.vector_field(data)
        if self.metric == "log_euclidean":
            return self.expm(self.logm(data) + tangent)
        # unit affine-invariant step (reference implementation)
        norm = self.norm(data, tangent).clamp_min(TANGENT_NORM_FLOOR)
        result = self.exp_map(data, tangent / norm[..., None, None])
        # projection onto the SPD matrices with spectrum in [1e-8, 1e8], as
        # projx of the reference implementation
        return self.clamp(symmetrize(result), _input_floor, SPD_EIGVAL_CEILING)

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hooks for dynamic parametrizations."""
        self.vector_field.register_optimizer_hook(optimizer)

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        return (
            f"ResidualBlock(n_features={self.n_features}, metric={self.metric}, "
            f"vector_field={self.vector_field})"
        )


# moved to functions.spd_geometries.affine_invariant
affine_invariant_norm = _deprecated(
    "affine_invariant_norm", affine_invariant.affine_invariant_norm
)
