"""Riemannian Residual Network layers: spectral vector field and residual block."""

import torch
from torch import nn

from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    AffineInvariantExp,
    affine_invariant_exp,
    affine_invariant_projx,
)
from yetanotherspdnet.functions.spd_linalg import (
    ExpmSymmetric,
    LogmSPD,
    expm_symmetric,
    logm_SPD,
)
from yetanotherspdnet.nn.parametrizations import (
    StiefelAdaptiveParametrization,
)
from yetanotherspdnet.random.stiefel import random_stiefel


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
        assert spectrum_type in ["conv1d", "mlp"], (
            f"spectrum_type must be 'conv1d' or 'mlp', got {spectrum_type}"
        )
        self.use_autograd = use_autograd

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
        eigenvalues = torch.linalg.eigvalsh(data)  # (..., n)

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

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hook for dynamic Stiefel parametrization."""
        if self.is_dynamic:
            stiefel_param = self.parametrizations.Q[0]

            def _hook(_opt):
                if self.current_ref_step >= self.n_steps_ref_update:
                    stiefel_param.update_reference_point()
                    self.current_ref_step = 0

            optimizer.register_step_post_hook(lambda _opt, _args: _hook(_opt))

    def __repr__(self) -> str:
        return (
            f"SpectralVectorField(n_features={self.n_features}, "
            f"spectrum_type='{self.spectrum_type}', "
            f"use_autograd={self.use_autograd})"
        )


def affine_invariant_norm(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    r"""
    Affine-invariant norm of tangent vectors.

    :math:`\lVert X^{-1/2} V X^{-1/2} \rVert_F = \lVert L^{-1} V L^{-\top} \rVert_F`
    with :math:`X = LL^\top`: Cholesky and triangular solves only, so the
    gradient does not go through an eigendecomposition.

    Parameters
    ----------
    base : torch.Tensor of shape (..., n, n)
        SPD base points

    tangent : torch.Tensor of shape (..., n, n)
        Symmetric tangent vectors at *base*

    Returns
    -------
    norm : torch.Tensor of shape (...)
        Affine-invariant norms
    """
    cholesky = torch.linalg.cholesky(base)
    half = torch.linalg.solve_triangular(cholesky, tangent, upper=False)
    whitened = torch.linalg.solve_triangular(cholesky, half.mT, upper=False)
    return torch.linalg.matrix_norm(whitened)


class ResidualBlock(nn.Module):
    """
    Riemannian residual block on SPD manifold.

    Applies one residual step along the vector field V = VF(X) (a
    SpectralVectorField), following Katsman et al., *Riemannian Residual
    Neural Networks* (NeurIPS 2023) and their reference implementation:

    - ``metric="affine_invariant"``: X_new = projx(Exp_X(V / ||V||_X)), a step
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
        assert metric in ["affine_invariant", "log_euclidean"], (
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
        if use_autograd:
            self.exp_map = affine_invariant_exp
            self.logm = lambda x: logm_SPD(x)[0]
            self.expm = lambda x: expm_symmetric(x)[0]
        else:
            self.exp_map = AffineInvariantExp.apply
            self.logm, self.expm = LogmSPD.apply, ExpmSymmetric.apply

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
        tangent = self.vector_field(data)
        if self.metric == "log_euclidean":
            return self.expm(self.logm(data) + tangent)
        # unit affine-invariant step (reference implementation)
        norm = affine_invariant_norm(data, tangent).clamp_min(1e-12)
        result = self.exp_map(data, tangent / norm[..., None, None])
        return affine_invariant_projx(result)

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hooks for dynamic parametrizations."""
        self.vector_field.register_optimizer_hook(optimizer)

    def __repr__(self) -> str:
        return (
            f"ResidualBlock(n_features={self.n_features}, metric={self.metric}, "
            f"vector_field={self.vector_field})"
        )
