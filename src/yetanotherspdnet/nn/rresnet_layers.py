r"""Riemannian Residual Network layers: spectral vector field and residual block.

Katsman et al., *Riemannian Residual Neural Networks* (NeurIPS 2023), as in
their reference implementation, with options from Res2SPDNet (Yue et al.,
CVPR 2026 Findings) for the residual connection; see
``docs/user_guide/residual.md`` for the formulas.
"""

from typing import cast

import torch
from torch import nn
from torch.autograd import Function

from yetanotherspdnet._deprecation import _deprecated
from yetanotherspdnet.functions._constants import (
    EIGVAL_TIE_RTOL,
    SPD_EIGVAL_CEILING,
    SPD_EIGVAL_FLOOR,
    TANGENT_NORM_FLOOR,
)
from yetanotherspdnet.functions.dispatch import select_path
from yetanotherspdnet.functions.spd_geometries import affine_invariant
from yetanotherspdnet.functions.spd_linalg import (
    EighClamp,
    EighReLu,
    EigvalshSymmetric,
    ExpmSymmetric,
    FlooredEighFunctions,
    InvSqrtmSPD,
    eigh_clamp,
    eigh_context,
    eigh_relu,
    eigvalsh_symmetric,
    expm_symmetric,
    floored_eigh_functions,
    inv_sqrtm_SPD,
    spd_floor,
    symmetrize,
)
from yetanotherspdnet.nn.base import BiMap, ReEig
from yetanotherspdnet.nn.parametrizations import (
    StiefelAdaptiveParametrization,
    recenter_dynamic_parametrization,
)
from yetanotherspdnet.random.stiefel import random_stiefel


# absolute eigenvalue floor applied to the input of ResidualBlock (lower bound
# of the optional output clamp, as affine_invariant_projx of the reference
# code); the floor actually used is spd_floor(eigenvalues, INPUT_EIGVAL_FLOOR),
# raised to the rounding level eps max(lambda) of the dtype
INPUT_EIGVAL_FLOOR = SPD_EIGVAL_FLOOR

METRICS = ("affine_invariant", "log_euclidean")
STEPS = ("unit", "none", "clip", "learned_scale")
RESIDUALS = ("exp", "branch_pt", "congruence")
BASES = ("learned_Q", "eigenbasis")
SPECTRUM_INPUTS = ("raw", "log")
LE_OUTPUTS = ("spd", "log")


def _input_floor(eigvals: torch.Tensor) -> torch.Tensor:
    """Dtype-aware floor of the ResidualBlock eigenvalues (see spd_floor)."""
    return spd_floor(eigvals, floor=INPUT_EIGVAL_FLOOR)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Raise a ValueError if *value* is not one of *choices*."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


# -------------------------------------------------------------
# Matrix with prescribed eigenvalues on the eigenvectors of X
# -------------------------------------------------------------
def _eigenbasis_matrix(
    data: torch.Tensor,
    values: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
) -> torch.Tensor:
    r"""
    :math:`U \operatorname{diag}(g) U^\top` on the eigenvectors of *data* (autograd).

    Autograd twin of :class:`_EigenbasisMatrix`: *eigvecs* must come from a
    decomposition of *data* tracked by autograd. *data* and *eigvals* are only
    used by the manual backward.
    """
    del data, eigvals
    return (eigvecs * values.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


class _EigenbasisMatrix(Function):
    r"""
    :math:`Y = U \operatorname{diag}(g) U^\top` with :math:`X = U \operatorname{diag}(\lambda) U^\top`.

    *g* is any (differentiable) function of the spectrum, not necessarily a
    spectral function :math:`g_i = \phi(\lambda_i)`. The backward splits
    :math:`dY = U\big(\operatorname{diag}(dg) + \Omega \circ (U^\top dX U)\big)U^\top`:

    .. math::

        \bar g = \operatorname{diag}(U^\top \bar Y U),
        \qquad
        \bar X = U \big(\Omega \circ (U^\top \operatorname{sym}(\bar Y) U)\big) U^\top,
        \qquad
        \Omega_{ij} = \frac{g_i - g_j}{\lambda_i - \lambda_j}\ (i \ne j)

    (the rotation of the eigenvectors; the dependence of :math:`g` on
    :math:`\lambda` flows through :math:`\bar g`). :math:`\Omega` is set to 0
    at rounding-level ties :math:`|\lambda_i - \lambda_j| \le n \varepsilon
    \max|\lambda|`: when :math:`g_i \ne g_j` for :math:`\lambda_i =
    \lambda_j` (a position-dependent map of the sorted spectrum, e.g. a
    Conv1d), :math:`Y` is not even defined, as the eigenvectors of a repeated
    eigenvalue are arbitrary.
    """

    @staticmethod
    def forward(ctx, data, values, eigvals, eigvecs):
        r"""
        Forward pass.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass
        data : torch.Tensor of shape (..., n, n)
            Matrices whose decomposition is (eigvals, eigvecs); receives the
            gradient of the rotation of the eigenvectors
        values : torch.Tensor of shape (..., n)
            New eigenvalues :math:`g`
        eigvals : torch.Tensor of shape (..., n)
            Eigenvalues of *data* (not differentiated)
        eigvecs : torch.Tensor of shape (..., n, n)
            Eigenvectors of *data* (not differentiated)

        Returns
        -------
        torch.Tensor of shape (..., n, n)
            :math:`U \\operatorname{diag}(g) U^\top`
        """
        ctx.save_for_backward(values, eigvals, eigvecs)
        return _eigenbasis_matrix(data, values, eigvals, eigvecs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (see the class docstring).

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object with the saved tensors
        grad_output : torch.Tensor of shape (..., n, n)
            Gradient with respect to the output

        Returns
        -------
        tuple
            Gradients with respect to data and values (None for the rest)
        """
        values, eigvals, eigvecs = ctx.saved_tensors
        middle = eigvecs.transpose(-1, -2) @ symmetrize(grad_output) @ eigvecs
        grad_values = torch.diagonal(middle, dim1=-2, dim2=-1)
        gap = eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2)
        scale = eigvals.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        finfo = torch.finfo(eigvals.dtype)
        tie = gap.abs() <= (
            EIGVAL_TIE_RTOL
            * eigvals.shape[-1]
            * finfo.eps
            * scale.clamp_min(finfo.tiny)
        )
        omega = torch.where(
            tie,
            torch.zeros_like(gap),
            (values.unsqueeze(-1) - values.unsqueeze(-2))
            / torch.where(tie, torch.ones_like(gap), gap),
        )
        grad_data = eigvecs @ (omega * middle) @ eigvecs.transpose(-1, -2)
        return grad_data, grad_values, None, None


class SpectralVectorField(nn.Module):
    r"""
    Spectral vector field on the SPD manifold.

    Computes a tangent vector at X as

    .. math:: V = Q \operatorname{diag}\big(f(s(X))\big) Q^\top

    where :math:`s(X)` is the spectrum of X in ascending order (or its
    logarithm, ``spectrum_input="log"``), :math:`f` a learned map
    :math:`\mathbb{R}^n \to \mathbb{R}^n` (Conv1d along the sorted spectrum, or
    MLP) and :math:`Q` a learned orthogonal matrix (Stiefel-parametrized), as
    the reference implementation. With ``basis="eigenbasis"``, :math:`Q` is
    replaced by the eigenvectors :math:`U` of X itself
    (:math:`V = U \operatorname{diag}(f) U^\top`): no learned rotation.

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

    eigh_dtype : torch.dtype or None, optional
        dtype of the eigendecompositions of the forward pass (e.g.
        ``torch.float64`` in a float32 model: mixed precision, see
        :func:`~yetanotherspdnet.functions.spd_linalg.eigh_precision`).
        Default is None (the dtype of the data).

    basis : str, optional
        "learned_Q" (default, the reference): :math:`V = Q \operatorname{diag}(f) Q^\top`;
        "eigenbasis": :math:`V = U \operatorname{diag}(f) U^\top` on the
        eigenvectors of X (no Q; a pure eigenvalue residual in ResidualBlock).

    spectrum_input : str, optional
        Input of :math:`f`: "raw" (default, the eigenvalues, as the reference)
        or "log" (their logarithms: invariant to a rescaling of the data up to
        a shift).

    field_batchnorm : bool, optional
        BatchNorm1d between the Conv1d layers of :math:`f` (the reference).
        Default is True. No effect with ``spectrum_type="mlp"``.
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
        eigh_dtype: torch.dtype | None = None,
        basis: str = "learned_Q",
        spectrum_input: str = "raw",
        field_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.eigh_dtype = eigh_dtype
        self.n_features = n_features
        self.spectrum_type = spectrum_type
        if spectrum_type not in ("conv1d", "mlp"):
            raise ValueError(
                f"spectrum_type must be 'conv1d' or 'mlp', got {spectrum_type}"
            )
        _check_choice("basis", basis, BASES)
        _check_choice("spectrum_input", spectrum_input, SPECTRUM_INPUTS)
        self.basis = basis
        self.spectrum_input = spectrum_input
        self.field_batchnorm = field_batchnorm
        self.use_autograd = use_autograd
        self.eigvalsh = select_path(
            eigvalsh_symmetric, EigvalshSymmetric.apply, use_autograd
        )
        self.decompose = select_path(
            floored_eigh_functions, FlooredEighFunctions.apply, use_autograd
        )
        self.eigenbasis_matrix = select_path(
            _eigenbasis_matrix, _EigenbasisMatrix.apply, use_autograd
        )

        self.is_dynamic = False
        self.current_ref_step = 0
        self.n_steps_ref_update = stiefel_n_steps_ref_update
        if basis == "learned_Q":
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
            self.is_dynamic = stiefel_parametrization_mode == "dynamic"
            nn.utils.parametrize.register_parametrization(
                self,
                "Q",
                StiefelAdaptiveParametrization(
                    n_in=n_features,
                    n_out=n_features,
                    # reference point = the sampled Q, as BiMap: the tangent
                    # vector starts at zero and Q at its sampled value
                    initial_reference=self.Q.clone().detach(),
                    mapping="QR",
                    use_autograd=use_autograd,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ),
            )

        # Spectrum mapping network
        padding = (spectrum_kernel_size - 1) // 2

        def batchnorm() -> list[nn.Module]:
            if not field_batchnorm:
                return []
            return [nn.BatchNorm1d(spectrum_hidden_dim, dtype=dtype, device=device)]

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
                        *batchnorm(),
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
                    *batchnorm(),
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

    def spectrum(self, eigvals: torch.Tensor) -> torch.Tensor:
        r"""
        Learned map of the spectrum, :math:`f(s(X))`.

        Parameters
        ----------
        eigvals : torch.Tensor of shape (..., n)
            Eigenvalues of the SPD matrices, ascending (positive)

        Returns
        -------
        torch.Tensor of shape (..., n)
            :math:`f(\lambda)` or :math:`f(\log\lambda)` (``spectrum_input``)
        """
        batch_shape = eigvals.shape[:-1]
        n = self.n_features
        if self.spectrum_input == "log":
            eigvals = torch.log(eigvals)
        flat_eigs = eigvals.reshape(-1, n)  # (B, n)
        if self.spectrum_type == "conv1d":
            # Conv1d expects (B, C, L) with C=1
            f_eigs = self.spectrum_map(flat_eigs.unsqueeze(1)).squeeze(1)
        else:
            f_eigs = self.spectrum_map(flat_eigs)
        return f_eigs.reshape(*batch_shape, n)

    def from_spectrum(self, f_eigs: torch.Tensor) -> torch.Tensor:
        r"""
        :math:`Q \operatorname{diag}(f) Q^\top` (``basis="learned_Q"``).

        Counts the training steps of the dynamic parametrization.

        Parameters
        ----------
        f_eigs : torch.Tensor of shape (..., n)
            Output of :meth:`spectrum`

        Returns
        -------
        torch.Tensor of shape (..., n, n)
            Symmetric tangent vectors
        """
        if self.training and self.is_dynamic:
            self.current_ref_step += 1
        Q = self.Q  # (n, n) orthogonal
        return (Q * f_eigs.unsqueeze(-2)) @ Q.transpose(-1, -2)

    def forward(
        self, data: torch.Tensor, eigvals: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute tangent vector from SPD matrix spectrum.

        Parameters
        ----------
        data : torch.Tensor of shape (..., n, n)
            Batch of SPD matrices

        eigvals : torch.Tensor of shape (..., n) or None, optional
            Precomputed (differentiable) eigenvalues of *data*, ascending, e.g.
            the floored spectrum of the decomposition of ResidualBlock. Default
            is None: computed here (one ``eigvalsh``). Not accepted with
            ``basis="eigenbasis"``, which needs the eigenvectors too.

        Returns
        -------
        tangent : torch.Tensor of shape (..., n, n)
            Symmetric tangent vectors at each input point
        """
        with eigh_context(self.eigh_dtype):
            if self.basis == "eigenbasis":
                if eigvals is not None:
                    raise ValueError(
                        "basis='eigenbasis' computes its own decomposition: "
                        "eigvals cannot be given"
                    )
                eigvals_floored, eigvals_raw, eigvecs = self.decompose(
                    data, _input_floor, ()
                )
                return self.eigenbasis_matrix(
                    data, self.spectrum(eigvals_floored), eigvals_raw, eigvecs
                )
            if eigvals is None:
                eigvals = self.eigvalsh(data)  # (..., n)
            return self.from_spectrum(self.spectrum(eigvals))

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
            # reference point <- Q after the step, tangent vector <- 0: Q is
            # continuous across the update
            recenter_dynamic_parametrization(self, "Q")
            self.current_ref_step = 0

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hook for dynamic Stiefel parametrization."""
        if self.is_dynamic:
            optimizer.register_step_post_hook(self._post_optimizer_hook)

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        options = ""
        if self.basis != "learned_Q":
            options += f", basis='{self.basis}'"
        if self.spectrum_input != "raw":
            options += f", spectrum_input='{self.spectrum_input}'"
        if not self.field_batchnorm:
            options += ", field_batchnorm=False"
        return (
            f"SpectralVectorField(n_features={self.n_features}, "
            f"spectrum_type='{self.spectrum_type}', "
            f"use_autograd={self.use_autograd}{options})"
        )


class ResidualBlock(nn.Module):
    r"""
    Riemannian residual block on the SPD manifold.

    One residual step along the vector field :math:`V = Q \operatorname{diag}(f(\lambda)) Q^\top`
    of a :class:`SpectralVectorField`, following Katsman et al., *Riemannian
    Residual Neural Networks* (NeurIPS 2023) and their reference
    implementation (the defaults reproduce it):

    - ``metric="affine_invariant"``:
      :math:`X_+ = \mathrm{Exp}_X(V / \lVert V \rVert_X)`, a step of unit
      affine-invariant length, :math:`\lVert V \rVert_X = \lVert X^{-1/2} V
      X^{-1/2} \rVert_F`. Without this normalization the norm grows like
      :math:`1 / \lambda_{\min}(X)` and the matrix exponential overflows on
      ill-conditioned inputs.
    - ``metric="log_euclidean"``: :math:`X_+ = \exp(\log X + V)`, a
      translation in log coordinates (not the log-Euclidean exponential
      :math:`\mathrm{Exp}^{LE}_X(V) = \exp(\log X + D\log_X[V])`, see
      :func:`~yetanotherspdnet.functions.spd_geometries.log_euclidean.log_euclidean_exp`).

    The input eigenvalues are first floored at
    :math:`\max(10^{-8}, \varepsilon \max\lambda)` (identity on
    well-conditioned inputs). One eigendecomposition
    :math:`X = U \Lambda U^\top` gives the spectrum of the field and
    :math:`X^{\pm 1/2}` (or :math:`\log X`), a second one the exponential:
    2 per block (1 with ``basis="eigenbasis"``).

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
        Use autograd (through ``torch.linalg.eigh``) instead of the manual
        backwards. Default is False

    metric : str, optional
        "affine_invariant" (default) or "log_euclidean", see above

    device : torch.device, optional
        Device. Default is torch.device("cpu")

    dtype : torch.dtype, optional
        Data type. Default is torch.float64

    generator : torch.Generator | None, optional
        Generator. Default is None

    eigh_dtype : torch.dtype or None, optional
        dtype of the eigendecompositions of the forward pass (e.g.
        ``torch.float64`` in a float32 model: mixed precision, see
        :func:`~yetanotherspdnet.functions.spd_linalg.eigh_precision`).
        Default is None (the dtype of the data).

    step : str or None, optional
        Length of the step, with :math:`\nu = \lVert V \rVert_X` (affine-
        invariant) or :math:`\lVert V \rVert_F` (log-Euclidean):
        "unit" (:math:`V / \nu`), "none" (:math:`V`), "clip"
        (:math:`V \min(1, r / \nu)`, :math:`r` = ``step_radius``) or
        "learned_scale" (:math:`\alpha V / \nu`, :math:`\alpha` learned,
        initialized at ``step_scale_init``). Default is None: the reference,
        "unit" for the affine-invariant metric, "none" for the log-Euclidean
        one. Not used with ``residual="congruence"``.

    step_radius : float, optional
        Radius :math:`r` of ``step="clip"``. Default is 1.0

    step_scale_init : float, optional
        Initial :math:`\alpha` of ``step="learned_scale"``: a small value
        starts the block near the identity (exactly at it for a zero field).
        Default is 1e-2

    output_clamp : bool or None, optional
        Clamp the output eigenvalues to :math:`[\max(10^{-8}, \varepsilon
        \max\lambda), 10^8]` (``projx`` of the reference code, one more
        eigendecomposition). Default is None: only for the affine-invariant
        steps "none" and "learned_scale". The unit (and clipped) steps do not
        need it: with :math:`\lVert W \rVert_F \le r`, the spectrum of
        :math:`X_+ = X^{1/2} e^{W} X^{1/2}` lies in
        :math:`[\lambda_{\min} e^{-r}, \lambda_{\max} e^{r}]`.

    residual : str, optional
        Residual connection: "exp" (default, :math:`\mathrm{Exp}_X(V(X))`),
        "branch_pt" (:math:`Z = F(X)`, :math:`X_+ = \mathrm{Exp}_X(\Gamma_{Z
        \to X} V(Z))`, the field computed at the branch output and
        parallel-transported to X, as Res2SPDNet) or "congruence"
        (:math:`X_+ = X^{1/2} F(X) X^{1/2} = \mathrm{Exp}_X(\Gamma_{I \to X}
        \log F(X))`, no vector field).

    branch : nn.Module or None, optional
        The branch :math:`F` (SPD to SPD, same size) of "branch_pt" and
        "congruence". Default is None: ``BiMap(n, n)`` (dynamic Stiefel
        parametrization, manual backward) then ``ReEig()``.

    le_output : str, optional
        Output of the log-Euclidean step: "spd" (default,
        :math:`\exp(\log X + V)`) or "log" (:math:`\log X + V`, a symmetric
        matrix, without the exponential; for the last block before the
        classification head, which then skips LogEig, as the reference code).

    spectrum_input, basis, field_batchnorm
        Options of the :class:`SpectralVectorField`. With
        ``basis="eigenbasis"`` and ``residual="exp"``, the step is an
        eigenvalue map (one eigendecomposition per block):
        :math:`\lambda_i \mapsto \lambda_i \exp(s f_i / \lambda_i)`
        (affine-invariant, :math:`\nu = \lVert f / \lambda \rVert_2`) or
        :math:`\lambda_i e^{s f_i}` (log-Euclidean, :math:`\nu = \lVert f
        \rVert_2`), with the step factor :math:`s` of ``step``.
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
        eigh_dtype: torch.dtype | None = None,
        step: str | None = None,
        step_radius: float = 1.0,
        step_scale_init: float = 1e-2,
        output_clamp: bool | None = None,
        residual: str = "exp",
        branch: nn.Module | None = None,
        le_output: str = "spd",
        spectrum_input: str = "raw",
        basis: str = "learned_Q",
        field_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.eigh_dtype = eigh_dtype
        _check_choice("metric", metric, METRICS)
        _check_choice("residual", residual, RESIDUALS)
        _check_choice("le_output", le_output, LE_OUTPUTS)
        if step is not None:
            _check_choice("step", step, STEPS)
            if residual == "congruence":
                raise ValueError("residual='congruence' has no step to choose")
        if le_output == "log" and (
            metric != "log_euclidean" or residual == "congruence"
        ):
            raise ValueError(
                "le_output='log' needs metric='log_euclidean' and a vector field "
                "(residual 'exp' or 'branch_pt')"
            )
        self.n_features = n_features
        self.use_autograd = use_autograd
        self.metric = metric
        self.residual = residual
        self.le_output = le_output
        self.step_option = step
        self.step = step or ("unit" if metric == "affine_invariant" else "none")
        self.step_radius = step_radius
        self.output_clamp_option = output_clamp
        if output_clamp is None:
            output_clamp = metric == "affine_invariant" and self.step in (
                "none",
                "learned_scale",
            )
        self.output_clamp = output_clamp

        if residual != "congruence":
            self.vector_field: SpectralVectorField | None = SpectralVectorField(
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
                basis=basis,
                spectrum_input=spectrum_input,
                field_batchnorm=field_batchnorm,
            )
        else:
            self.vector_field = None
        if self.step == "learned_scale" and residual != "congruence":
            self.step_scale = nn.Parameter(
                torch.tensor(step_scale_init, dtype=dtype, device=device)
            )
        self.branch: nn.Module | None = None
        if residual != "exp":
            if branch is None:
                # dynamic Stiefel parametrization: manual QR retraction (the
                # static one is torch's orthogonal, autograd only)
                branch = nn.Sequential(
                    BiMap(
                        n_features,
                        n_features,
                        parametrization_mode="dynamic",
                        use_autograd=use_autograd,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    ),
                    ReEig(use_autograd=use_autograd),
                )
            self.branch = branch

        # Select implementations (autograd or manual backward)
        self.decompose = select_path(
            floored_eigh_functions, FlooredEighFunctions.apply, use_autograd
        )
        self.expm = select_path(
            lambda x: expm_symmetric(x)[0], ExpmSymmetric.apply, use_autograd
        )
        self.inv_sqrtm = select_path(
            lambda x: inv_sqrtm_SPD(x)[0], InvSqrtmSPD.apply, use_autograd
        )
        self.eigenbasis_matrix = select_path(
            _eigenbasis_matrix, _EigenbasisMatrix.apply, use_autograd
        )
        self.clamp = select_path(
            lambda x, lo, hi: eigh_clamp(x, lo, hi)[0], EighClamp.apply, use_autograd
        )
        # the input floor as a matrix map (the forward applies it inside the
        # decomposition of X, without forming the floored matrix)
        floor = select_path(
            lambda x, eps: eigh_relu(x, eps)[0], EighReLu.apply, use_autograd
        )
        self.floor = lambda x: floor(x, _input_floor)

    def _step_factor(self, norm: torch.Tensor) -> torch.Tensor:
        """Factor s of the step s V for tangent vectors of norm *norm*, shape (...)."""
        safe = norm.clamp_min(TANGENT_NORM_FLOOR)
        if self.step == "unit":
            return 1 / safe
        if self.step == "none":
            return torch.ones_like(norm)
        if self.step == "clip":
            return torch.clamp(self.step_radius / safe, max=1.0)
        return self.step_scale / safe

    def _tangent(self, point, eigvals_floored, eigvals, eigvecs) -> torch.Tensor:
        """Vector field at *point* (whose decomposition is given), (..., n, n)."""
        assert self.vector_field is not None
        f_eigs = self.vector_field.spectrum(eigvals_floored)
        if self.vector_field.basis == "eigenbasis":
            return self.eigenbasis_matrix(point, f_eigs, eigvals, eigvecs)
        return self.vector_field.from_spectrum(f_eigs)

    def _output(self, result: torch.Tensor) -> torch.Tensor:
        """Apply the optional eigenvalue clamp of the output (projx of the reference)."""
        if self.output_clamp:
            return self.clamp(symmetrize(result), _input_floor, SPD_EIGVAL_CEILING)
        return result

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
            Updated SPD matrices after one residual step (with
            ``le_output="log"``: their logarithms)
        """
        with eigh_context(self.eigh_dtype):
            # the input may be numerically singular (e.g. a Bures-Wasserstein
            # batch normalization folds matrices outside the injectivity domain
            # of its exponential): its eigenvalues are floored, a no-op on
            # well-conditioned inputs
            if self.residual == "congruence":
                assert self.branch is not None
                _, _, _, sqrt = self.decompose(data, _input_floor, ("sqrt",))
                result = symmetrize(sqrt @ self.branch(data) @ sqrt)
                return self._output(result)
            if self.metric == "affine_invariant":
                return self._affine_invariant_step(data)
            return self._log_euclidean_step(data)

    def _affine_invariant_step(self, data: torch.Tensor) -> torch.Tensor:
        """Affine-invariant step Exp_X(s V) (residual "exp" or "branch_pt")."""
        assert self.vector_field is not None
        eigvals_floored, eigvals, eigvecs, sqrt, inv_sqrt = self.decompose(
            data, _input_floor, ("sqrt", "inv_sqrt")
        )
        if self.residual == "exp" and self.vector_field.basis == "eigenbasis":
            # V = U diag(f) U^T: X^{-1/2} V X^{-1/2} = U diag(f / lambda) U^T,
            # Exp_X(s V) = U diag(lambda exp(s f / lambda)) U^T
            whitened = self.vector_field.spectrum(eigvals_floored) / eigvals_floored
            factor = self._step_factor(torch.linalg.vector_norm(whitened, dim=-1))
            values = eigvals_floored * torch.exp(factor.unsqueeze(-1) * whitened)
            result = self.eigenbasis_matrix(data, values, eigvals, eigvecs)
            return self._output(result)
        if self.residual == "exp":
            tangent = self._tangent(data, eigvals_floored, eigvals, eigvecs)
            # W = X^{-1/2} V X^{-1/2}, ||V||_X = ||W||_F
            whitened = symmetrize(inv_sqrt @ tangent @ inv_sqrt)
        else:
            assert self.branch is not None
            branch = self.branch(data)
            branch_floored, branch_eigvals, branch_eigvecs = self.decompose(
                branch, _input_floor, ()
            )
            tangent = self._tangent(
                branch, branch_floored, branch_eigvals, branch_eigvecs
            )
            # Gamma_{Z -> X}(V) = E V E^T, E = (X Z^{-1})^{1/2}
            # = X^{1/2} N^{-1/2} X^{-1/2} with N = X^{-1/2} Z X^{-1/2}, so
            # X^{-1/2} Gamma(V) X^{-1/2} = N^{-1/2} X^{-1/2} V X^{-1/2} N^{-1/2};
            # normalized at X (= at Z: the transport is an isometry)
            relative = self.inv_sqrtm(symmetrize(inv_sqrt @ branch @ inv_sqrt))
            whitened = symmetrize(relative @ inv_sqrt @ tangent @ inv_sqrt @ relative)
        factor = self._step_factor(torch.linalg.matrix_norm(whitened))
        exp_whitened = self.expm(factor[..., None, None] * whitened)
        return self._output(symmetrize(sqrt @ exp_whitened @ sqrt))

    def _log_euclidean_step(self, data: torch.Tensor) -> torch.Tensor:
        """Log-Euclidean step exp(log X + s V) (residual "exp" or "branch_pt")."""
        assert self.vector_field is not None
        eigvals_floored, eigvals, eigvecs, log_data = self.decompose(
            data, _input_floor, ("log",)
        )
        if self.residual == "exp" and self.vector_field.basis == "eigenbasis":
            # log X + s V = U diag(log lambda + s f) U^T
            f_eigs = self.vector_field.spectrum(eigvals_floored)
            factor = self._step_factor(torch.linalg.vector_norm(f_eigs, dim=-1))
            values = torch.log(eigvals_floored) + factor.unsqueeze(-1) * f_eigs
            if self.le_output == "spd":
                values = torch.exp(values)
            result = self.eigenbasis_matrix(data, values, eigvals, eigvecs)
            return result if self.le_output == "log" else self._output(result)
        if self.residual == "exp":
            tangent = self._tangent(data, eigvals_floored, eigvals, eigvecs)
        else:
            # the log-Euclidean transport is the identity in log coordinates
            assert self.branch is not None
            branch = self.branch(data)
            branch_floored, branch_eigvals, branch_eigvecs = self.decompose(
                branch, _input_floor, ()
            )
            tangent = self._tangent(
                branch, branch_floored, branch_eigvals, branch_eigvecs
            )
        if self.step == "none":
            log_result = log_data + tangent
        else:
            factor = self._step_factor(torch.linalg.matrix_norm(tangent))
            log_result = log_data + factor[..., None, None] * tangent
        if self.le_output == "log":
            return log_result
        return self._output(self.expm(log_result))

    def register_optimizer_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Register optimizer hooks for dynamic parametrizations."""
        for module in self.modules():
            if (
                module is not self
                and getattr(module, "is_dynamic", False) is True
                and hasattr(module, "register_optimizer_hook")
            ):
                cast(SpectralVectorField | BiMap, module).register_optimizer_hook(
                    optimizer
                )

    def __repr__(self) -> str:
        """Return the representation of the module (its hyper-parameters)."""
        options = ""
        if self.step_option is not None:
            options += f", step={self.step_option}"
        if self.output_clamp_option is not None:
            options += f", output_clamp={self.output_clamp_option}"
        if self.residual != "exp":
            options += f", residual={self.residual}, branch={self.branch}"
        if self.le_output != "spd":
            options += f", le_output={self.le_output}"
        return (
            f"ResidualBlock(n_features={self.n_features}, metric={self.metric}, "
            f"vector_field={self.vector_field}{options})"
        )


# moved to functions.spd_geometries.affine_invariant
affine_invariant_norm = _deprecated(
    "affine_invariant_norm", affine_invariant.affine_invariant_norm
)
