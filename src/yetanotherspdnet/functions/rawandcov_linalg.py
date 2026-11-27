"""Linear algebra for the two-stream (covariance + raw data) SPDnet layers."""

from collections.abc import Callable

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.spd_linalg import (
    eigh_operation,
    eigh_operation_grad,
    sym_coordinates_to_matrix,
    sym_matrix_to_coordinates,
    symmetrize,
    unvec_batch,
    vec_batch,
)


# ------------------------------------------
# Two-stream BiMap: congruence + projection
# ------------------------------------------
def congruence_and_projection(
    data_cov: torch.Tensor, data_raw: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream BiMap: the same rectangular matrix is
    applied to a batch of SPD matrices (congruence) and to the associated
    raw data (projection).

    .. math::
        P = W^\top \Sigma\, W, \qquad Y = W^\top X,
        \qquad W \in \mathbb{R}^{n_{in} \times n_{out}}

    with :math:`n_{in} \geq n_{out}`. If :math:`\Sigma = \frac{1}{n} X X^\top`,
    then :math:`P = \frac{1}{n} Y Y^\top`: the link between both streams is kept.

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_in, n_in)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_in, n_samples)
        Batch of raw data associated with the SPD matrices

    weight : torch.Tensor of shape (n_in, n_out)
        Rectangular matrix (e.g., weights), n_in >= n_out is expected

    Returns
    -------
    data_cov_transformed : torch.Tensor of shape (..., n_out, n_out)
        Transformed batch of SPD matrices

    data_raw_transformed : torch.Tensor of shape (..., n_out, n_samples)
        Projected batch of raw data
    """
    weight_t = weight.transpose(-1, -2)
    return weight_t @ data_cov @ weight, weight_t @ data_raw


class CongruenceAndProjection(Function):
    """
    Two-stream BiMap: congruence of a batch of SPD matrices and projection of
    the associated raw data with the same (full-rank) rectangular matrix
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream BiMap

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        data_cov : torch.Tensor of shape (..., n_in, n_in)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_in, n_samples)
            Batch of raw data

        weight : torch.Tensor of shape (n_in, n_out)
            Rectangular matrix (e.g., weights), n_in >= n_out is expected

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_out, n_out)
            Transformed batch of SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_out, n_samples)
            Projected batch of raw data
        """
        ctx.save_for_backward(data_cov, data_raw, weight)
        return congruence_and_projection(data_cov, data_raw, weight)

    @staticmethod
    def backward(
        ctx, grad_output_cov: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Backward pass of the two-stream BiMap.

        With :math:`E_k = \partial \ell / \partial P_k` (symmetric) and
        :math:`F_k = \partial \ell / \partial Y_k`:

        .. math::
            \frac{\partial \ell}{\partial W} = \sum_k 2\, \Sigma_k W E_k + X_k F_k^\top,
            \qquad
            \frac{\partial \ell}{\partial \Sigma_k} = W E_k W^\top,
            \qquad
            \frac{\partial \ell}{\partial X_k} = W F_k

        (``weight`` here is the transpose of :math:`W` in the two-stream BiMap note.)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_cov : torch.Tensor of shape (..., n_out, n_out)
            Gradient of the loss with respect to the transformed SPD matrices

        grad_output_raw : torch.Tensor of shape (..., n_out, n_samples)
            Gradient of the loss with respect to the projected raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_in, n_in)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_in, n_samples)
            Gradient of the loss with respect to the input raw data

        grad_input_W : torch.Tensor of shape (n_in, n_out)
            Gradient of the loss with respect to the rectangular matrix
        """
        # retrieve inputs saved during the forward pass
        data_cov, data_raw, weight = ctx.saved_tensors

        # error transmitted to Sigma_k: W E_k W^T
        grad_input_cov = weight @ grad_output_cov @ weight.transpose(-1, -2)

        # error transmitted to X_k: W F_k
        grad_input_raw = weight @ grad_output_raw

        # error to update W (transposed since weight stores W^T of the note):
        # sum over the batch of 2 Sigma_k W E_k (SPD stream) + X_k F_k^T (raw stream).
        # E_k is symmetrized because the factor 2 assumes E_k symmetric
        grad_input_W = 2 * torch.einsum(
            "...ik,kl,...lj->ij", data_cov, weight, symmetrize(grad_output_cov)
        ) + torch.einsum("...ik,...jk->ij", data_raw, grad_output_raw)

        # one gradient per input of forward, in the same order
        return grad_input_cov, grad_input_raw, grad_input_W


# ------------------------------------------------------------
# Two-stream ReEig: rectification of the eigenvalues + ReLU on
# the raw data, expressed in a basis given by the SPD matrices
# ------------------------------------------------------------
def _eigh_sign_convention(eigvecs: torch.Tensor) -> torch.Tensor:
    """
    Fix the sign of the eigenvectors: the entry of largest magnitude of each
    eigenvector is made positive.

    Parameters
    ----------
    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors (columns) of a batch of symmetric matrices

    Returns
    -------
    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors with a deterministic sign
    """
    idx = eigvecs.abs().argmax(dim=-2, keepdim=True)
    sign = torch.sign(torch.gather(eigvecs, -2, idx))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return eigvecs * sign


def _inverse_eigvals_gaps(eigvals: torch.Tensor, tol: float = 1e-10) -> torch.Tensor:
    r"""
    Matrix :math:`F` with :math:`F_{ij} = 1 / (\lambda_j - \lambda_i)` for
    :math:`i \neq j` and :math:`F_{ii} = 0`, used in the backward pass of the
    eigenvectors. The gaps are bounded from below by ``tol`` in magnitude.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of a batch of symmetric matrices

    tol : float, optional
        Lower bound on the magnitude of the gaps. Default is 1e-10

    Returns
    -------
    F : torch.Tensor of shape (..., n_features, n_features)
        Antisymmetric matrix of inverse gaps
    """
    gaps = eigvals.unsqueeze(-2) - eigvals.unsqueeze(-1)  # lambda_j - lambda_i
    sign = torch.where(gaps >= 0, torch.ones_like(gaps), -torch.ones_like(gaps))
    gaps = torch.where(gaps.abs() < tol, tol * sign, gaps)
    eye = torch.eye(eigvals.shape[-1], dtype=eigvals.dtype, device=eigvals.device)
    return (1.0 / gaps) * (1 - eye)


def eigh_relu_and_projection(
    data_cov: torch.Tensor, data_raw: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream ReEig, projection version: the eigenvalues
    of the SPD matrices are rectified and the raw data are rectified in the
    eigenbasis.

    .. math::
        Q = U \max(\varepsilon I, \Lambda) U^\top, \qquad
        V = U\, \mathrm{ReLU}(U^\top X),
        \qquad \Sigma = U \Lambda U^\top

    The sign of the eigenvectors is fixed by :func:`_eigh_sign_convention`,
    since :math:`V` depends on it.

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data associated with the SPD matrices

    eps : float
        Value for the rectification of the eigenvalues

    Returns
    -------
    data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices with rectified eigenvalues

    data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
        Batch of rectified raw data

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the matrices in data_cov

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the matrices in data_cov
    """
    eigvals, eigvecs = torch.linalg.eigh(data_cov)
    eigvecs = _eigh_sign_convention(eigvecs)
    data_cov_transformed = eigh_operation(
        eigvals, eigvecs, lambda x: torch.clamp(x, min=eps)
    )
    data_raw_transformed = eigvecs @ torch.relu(eigvecs.transpose(-1, -2) @ data_raw)
    return data_cov_transformed, data_raw_transformed, eigvals, eigvecs


class EighReLuAndProjection(Function):
    """
    Two-stream ReEig, projection version: rectification of the eigenvalues of
    the SPD matrices and ReLU on the raw data projected onto the eigenvectors
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream ReEig, projection version

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        eps : float
            Value for the rectification of the eigenvalues

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of rectified raw data
        """
        (
            data_cov_transformed,
            data_raw_transformed,
            eigvals,
            eigvecs,
        ) = eigh_relu_and_projection(data_cov, data_raw, eps)
        ctx.save_for_backward(eigvals, eigvecs, data_raw)
        ctx.eps = eps
        return data_cov_transformed, data_raw_transformed

    @staticmethod
    def backward(
        ctx, grad_output_cov: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        r"""
        Backward pass of the two-stream ReEig, projection version.

        With :math:`E = \partial \ell / \partial Q`, :math:`G = \partial \ell /
        \partial V`, :math:`Z = U^\top X` and :math:`H = (U^\top G) \odot
        \mathbf{1}[Z > 0]`:

        .. math::
            \frac{\partial \ell}{\partial X} = U H, \qquad
            \frac{\partial \ell}{\partial \Sigma} = U \big[ K_f \odot
            (U^\top \mathrm{sym}(E) U) + F \odot \mathrm{skew}(M) \big] U^\top

        with :math:`M = U^\top (G\, \mathrm{ReLU}(Z)^\top + X H^\top)`,
        :math:`K_f` the matrix of divided differences of
        :math:`f = \max(\varepsilon, \cdot)` and
        :math:`F_{ij} = 1 / (\lambda_j - \lambda_i)`.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output SPD matrices

        grad_output_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the output raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the input raw data
        """
        eps = ctx.eps
        eigvals, eigvecs, data_raw = ctx.saved_tensors
        eigvecs_t = eigvecs.transpose(-1, -2)

        # SPD stream: U [K_f o (U^T sym(E) U)] U^T
        grad_input_cov = eigh_operation_grad(
            grad_output_cov,
            eigvals,
            eigvecs,
            lambda x: torch.clamp(x, min=eps),
            lambda x: (x > eps).type(x.dtype),
        )

        # raw stream: H = (U^T G) o g'(Z), dl/dX = U H
        projected = eigvecs_t @ data_raw
        H = (eigvecs_t @ grad_output_raw) * (projected > 0).type(projected.dtype)
        grad_input_raw = eigvecs @ H

        # dl/dU, then back to Sigma: U [F o skew(U^T dl/dU)] U^T
        grad_eigvecs = grad_output_raw @ torch.relu(projected).transpose(
            -1, -2
        ) + data_raw @ H.transpose(-1, -2)
        M = eigvecs_t @ grad_eigvecs
        middle_term = _inverse_eigvals_gaps(eigvals) * 0.5 * (M - M.transpose(-1, -2))
        grad_input_cov = grad_input_cov + eigvecs @ middle_term @ eigvecs_t

        return grad_input_cov, grad_input_raw, None


def _whitening_functions(
    eps: float, whitening: str
) -> tuple[Callable, Callable, Callable, Callable]:
    r"""
    Scalar functions :math:`h_1 = \sqrt{r}`, :math:`h_2 = 1 / \sqrt{r}` and
    their derivatives, where :math:`r(\lambda)` gives the eigenvalues of the
    whitening matrix :math:`R` from those of :math:`\Sigma`.

    Parameters
    ----------
    eps : float
        Value for the rectification of the eigenvalues

    whitening : str
        "Q" (:math:`R = Q`, :math:`r = \max(\varepsilon, \cdot)`) or
        "Sigma" (:math:`R = \Sigma`, :math:`r = \mathrm{id}`)

    Returns
    -------
    h1, h1_deriv, h2, h2_deriv : Callable
        Scalar functions and their derivatives
    """
    if whitening == "Q":
        r = lambda x: torch.clamp(x, min=eps)
        r_deriv = lambda x: (x > eps).type(x.dtype)
    elif whitening == "Sigma":
        r = lambda x: x
        r_deriv = lambda x: torch.ones_like(x)
    else:
        raise ValueError(f"whitening must be 'Q' or 'Sigma', got {whitening!r}")
    h1 = lambda x: torch.sqrt(r(x))
    h2 = lambda x: torch.rsqrt(r(x))
    h1_deriv = lambda x: 0.5 * r_deriv(x) * torch.rsqrt(r(x))
    h2_deriv = lambda x: -0.5 * r_deriv(x) * torch.pow(r(x), -1.5)
    return h1, h1_deriv, h2, h2_deriv


def eigh_relu_and_whitening(
    data_cov: torch.Tensor, data_raw: torch.Tensor, eps: float, whitening: str = "Q"
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""
    Forward pass of the two-stream ReEig, whitening version: the eigenvalues
    of the SPD matrices are rectified and the raw data are whitened, rectified
    and recoloured.

    .. math::
        Q = U \max(\varepsilon I, \Lambda) U^\top, \qquad
        V = R^{1/2}\, \mathrm{ReLU}(R^{-1/2} X),
        \qquad R \in \{Q, \Sigma\}

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data associated with the SPD matrices

    eps : float
        Value for the rectification of the eigenvalues

    whitening : str, optional
        "Q" to whiten with the rectified matrices Q, "Sigma" to whiten with the
        input matrices. Default is "Q"

    Returns
    -------
    data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices with rectified eigenvalues

    data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
        Batch of whitened, rectified and recoloured raw data

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the matrices in data_cov

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the matrices in data_cov

    sqrt_R : torch.Tensor of shape (..., n_features, n_features)
        Square roots of the whitening matrices

    whitened : torch.Tensor of shape (..., n_features, n_samples)
        Whitened raw data R^{-1/2} X
    """
    h1, _, h2, _ = _whitening_functions(eps, whitening)
    eigvals, eigvecs = torch.linalg.eigh(data_cov)
    data_cov_transformed = eigh_operation(
        eigvals, eigvecs, lambda x: torch.clamp(x, min=eps)
    )
    sqrt_R = eigh_operation(eigvals, eigvecs, h1)
    inv_sqrt_R = eigh_operation(eigvals, eigvecs, h2)
    whitened = inv_sqrt_R @ data_raw
    data_raw_transformed = sqrt_R @ torch.relu(whitened)
    return (
        data_cov_transformed,
        data_raw_transformed,
        eigvals,
        eigvecs,
        sqrt_R,
        whitened,
    )


class EighReLuAndWhitening(Function):
    """
    Two-stream ReEig, whitening version: rectification of the eigenvalues of
    the SPD matrices and ReLU on the raw data whitened by Q or Sigma
    """

    @staticmethod
    def forward(
        ctx,
        data_cov: torch.Tensor,
        data_raw: torch.Tensor,
        eps: float,
        whitening: str = "Q",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream ReEig, whitening version

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        eps : float
            Value for the rectification of the eigenvalues

        whitening : str, optional
            "Q" or "Sigma". Default is "Q"

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of whitened, rectified and recoloured raw data
        """
        (
            data_cov_transformed,
            data_raw_transformed,
            eigvals,
            eigvecs,
            sqrt_R,
            whitened,
        ) = eigh_relu_and_whitening(data_cov, data_raw, eps, whitening)
        ctx.save_for_backward(eigvals, eigvecs, data_raw, sqrt_R, whitened)
        ctx.eps = eps
        ctx.whitening = whitening
        return data_cov_transformed, data_raw_transformed

    @staticmethod
    def backward(
        ctx, grad_output_cov: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        r"""
        Backward pass of the two-stream ReEig, whitening version.

        With :math:`E = \partial \ell / \partial Q`, :math:`G = \partial \ell /
        \partial V`, :math:`W = R^{-1/2} X` and :math:`H = (R^{1/2} G) \odot
        \mathbf{1}[W > 0]`:

        .. math::
            \frac{\partial \ell}{\partial X} = R^{-1/2} H, \qquad
            \frac{\partial \ell}{\partial \Sigma} = U \big[ K_f \odot
            (U^\top \mathrm{sym}(E) U)
            + K_{h_1} \odot (U^\top \mathrm{sym}(G\, \mathrm{ReLU}(W)^\top) U)
            + K_{h_2} \odot (U^\top \mathrm{sym}(H X^\top) U) \big] U^\top

        where :math:`K_h` is the matrix of divided differences of :math:`h`.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output SPD matrices

        grad_output_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the output raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the input raw data
        """
        eps, whitening = ctx.eps, ctx.whitening
        eigvals, eigvecs, data_raw, sqrt_R, whitened = ctx.saved_tensors
        h1, h1_deriv, h2, h2_deriv = _whitening_functions(eps, whitening)
        inv_sqrt_R = eigh_operation(eigvals, eigvecs, h2)

        # "linear" part: V = A g(W), W = B X, with A = R^{1/2}, B = R^{-1/2}
        H = (sqrt_R @ grad_output_raw) * (whitened > 0).type(whitened.dtype)
        grad_input_raw = inv_sqrt_R @ H
        grad_sqrt_R = grad_output_raw @ torch.relu(whitened).transpose(-1, -2)
        grad_inv_sqrt_R = H @ data_raw.transpose(-1, -2)

        # back to Sigma through the three spectral functions Q, A and B
        # (eigh_operation_grad symmetrizes the gradients it receives)
        grad_input_cov = (
            eigh_operation_grad(
                grad_output_cov,
                eigvals,
                eigvecs,
                lambda x: torch.clamp(x, min=eps),
                lambda x: (x > eps).type(x.dtype),
            )
            + eigh_operation_grad(grad_sqrt_R, eigvals, eigvecs, h1, h1_deriv)
            + eigh_operation_grad(grad_inv_sqrt_R, eigvals, eigvecs, h2, h2_deriv)
        )

        return grad_input_cov, grad_input_raw, None, None


# ------------------------------------------------------------
# Two-stream LogEig: matrix logarithm of the SPD matrices, with
# the raw data unchanged (version 1) or whitened (version 2)
# ------------------------------------------------------------
def _log_deriv(x: torch.Tensor) -> torch.Tensor:
    """Derivative of the logarithm."""
    return 1 / x


def _inv_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Inverse square root."""
    return torch.rsqrt(x)


def _inv_sqrt_deriv(x: torch.Tensor) -> torch.Tensor:
    """Derivative of the inverse square root."""
    return -0.5 * torch.pow(x, -1.5)


def logm_two_stream(
    data_cov: torch.Tensor, data_raw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream LogEig layer, version 1: matrix logarithm of
    the SPD matrices, and raw data left unchanged.

    .. math::
        L = \log\Sigma, \qquad W = X

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data associated with the SPD matrices

    Returns
    -------
    data_log : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrix logarithms of the SPD matrices

    data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data (unchanged)

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the matrices in data_cov

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the matrices in data_cov
    """
    eigvals, eigvecs = torch.linalg.eigh(data_cov)
    data_log = eigh_operation(eigvals, eigvecs, torch.log)
    return data_log, data_raw, eigvals, eigvecs


class LogmTwoStream(Function):
    """
    Two-stream LogEig layer, version 1: matrix logarithm of the SPD matrices,
    and raw data left unchanged
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream LogEig layer, version 1

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        Returns
        -------
        data_log : torch.Tensor of shape (..., n_features, n_features)
            Batch of matrix logarithms of the SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data (unchanged)
        """
        data_log, _, eigvals, eigvecs = logm_two_stream(data_cov, data_raw)
        ctx.save_for_backward(eigvals, eigvecs)
        # clone, so that the output is not the input itself
        return data_log, data_raw.clone()

    @staticmethod
    def backward(
        ctx, grad_output_log: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Backward pass of the two-stream LogEig layer, version 1.

        With :math:`G_L = \partial \ell / \partial L` and
        :math:`G_W = \partial \ell / \partial W`:

        .. math::
            \frac{\partial \ell}{\partial \Sigma} = U \big[ K_{\log} \odot
            (U^\top \mathrm{sym}(G_L) U) \big] U^\top, \qquad
            \frac{\partial \ell}{\partial X} = G_W

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_log : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the matrix logarithms

        grad_output_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the output raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the input raw data
        """
        eigvals, eigvecs = ctx.saved_tensors
        grad_input_cov = eigh_operation_grad(
            grad_output_log, eigvals, eigvecs, torch.log, _log_deriv
        )
        return grad_input_cov, grad_output_raw


def logm_and_whitening(
    data_cov: torch.Tensor, data_raw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream LogEig layer, version 2: matrix logarithm of
    the SPD matrices, and raw data whitened by the SPD matrices.

    .. math::
        L = \log\Sigma, \qquad W = \Sigma^{-1/2} X

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data associated with the SPD matrices

    Returns
    -------
    data_log : torch.Tensor of shape (..., n_features, n_features)
        Batch of matrix logarithms of the SPD matrices

    data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
        Batch of whitened raw data

    eigvals : torch.Tensor of shape (..., n_features)
        Eigenvalues of the matrices in data_cov

    eigvecs : torch.Tensor of shape (..., n_features, n_features)
        Eigenvectors of the matrices in data_cov
    """
    eigvals, eigvecs = torch.linalg.eigh(data_cov)
    data_log = eigh_operation(eigvals, eigvecs, torch.log)
    data_raw_transformed = eigh_operation(eigvals, eigvecs, _inv_sqrt) @ data_raw
    return data_log, data_raw_transformed, eigvals, eigvecs


class LogmAndWhitening(Function):
    """
    Two-stream LogEig layer, version 2: matrix logarithm of the SPD matrices,
    and raw data whitened by the SPD matrices
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream LogEig layer, version 2

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        Returns
        -------
        data_log : torch.Tensor of shape (..., n_features, n_features)
            Batch of matrix logarithms of the SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of whitened raw data
        """
        data_log, data_raw_transformed, eigvals, eigvecs = logm_and_whitening(
            data_cov, data_raw
        )
        ctx.save_for_backward(eigvals, eigvecs, data_raw)
        return data_log, data_raw_transformed

    @staticmethod
    def backward(
        ctx, grad_output_log: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Backward pass of the two-stream LogEig layer, version 2.

        With :math:`G_L = \partial \ell / \partial L` and
        :math:`G_W = \partial \ell / \partial W`:

        .. math::
            \frac{\partial \ell}{\partial \Sigma} = U \big[ K_{\log} \odot
            (U^\top \mathrm{sym}(G_L) U) + K_{h_2} \odot
            (U^\top \mathrm{sym}(G_W X^\top) U) \big] U^\top, \qquad
            \frac{\partial \ell}{\partial X} = \Sigma^{-1/2} G_W

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_log : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the matrix logarithms

        grad_output_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the whitened raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the input raw data
        """
        eigvals, eigvecs, data_raw = ctx.saved_tensors

        # SPD stream: U [K_log o (U^T sym(G_L) U)] U^T
        grad_input_cov = eigh_operation_grad(
            grad_output_log, eigvals, eigvecs, torch.log, _log_deriv
        )

        # raw data stream: W = B X with B = Sigma^{-1/2}, so dl/dX = B G_W, and
        # dl/dB = G_W X^T goes back to Sigma through B
        inv_sqrt_cov = eigh_operation(eigvals, eigvecs, _inv_sqrt)
        grad_input_raw = inv_sqrt_cov @ grad_output_raw
        grad_input_cov = grad_input_cov + eigh_operation_grad(
            grad_output_raw @ data_raw.transpose(-1, -2),
            eigvals,
            eigvecs,
            _inv_sqrt,
            _inv_sqrt_deriv,
        )

        return grad_input_cov, grad_input_raw


# ------------------------------------------------------------
# Two-stream vectorization: orthonormal coordinates of the
# symmetric matrices, with or without the vectorized raw data
# ------------------------------------------------------------
def coordinates_and_vec(data_sym: torch.Tensor, data_raw: torch.Tensor) -> torch.Tensor:
    r"""
    Forward pass of the two-stream vectorization: the symmetric matrices are
    mapped to their Euclidean orthonormal coordinates, and concatenated with the
    vectorized raw data.

    .. math::
        z = \big[\mathrm{coord}(L)\,;\,\mathrm{vec}(W)\big]

    where :math:`\mathrm{coord}` keeps the lower triangular part, with the
    off-diagonal entries scaled by :math:`\sqrt{2}`.

    Parameters
    ----------
    data_sym : torch.Tensor of shape (..., n_features, n_features)
        Batch of symmetric matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data

    Returns
    -------
    data_vec : torch.Tensor of shape (..., n_features*(n_features+1)//2 + n_features*n_samples)
        Batch of concatenated vectors
    """
    return torch.cat([sym_matrix_to_coordinates(data_sym), vec_batch(data_raw)], dim=-1)


class CoordinatesAndVec(Function):
    """
    Two-stream vectorization: orthonormal coordinates of the symmetric matrices,
    concatenated with the vectorized raw data
    """

    @staticmethod
    def forward(ctx, data_sym: torch.Tensor, data_raw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the two-stream vectorization

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_sym : torch.Tensor of shape (..., n_features, n_features)
            Batch of symmetric matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        Returns
        -------
        data_vec : torch.Tensor of shape (..., n_features*(n_features+1)//2 + n_features*n_samples)
            Batch of concatenated vectors
        """
        ctx.n_features = data_sym.shape[-1]
        return coordinates_and_vec(data_sym, data_raw)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Backward pass of the two-stream vectorization.

        The gradient is split as the output, :math:`[g_1\,;\,g_2]`, and both
        parts are reshaped as matrices:

        .. math::
            \frac{\partial \ell}{\partial L} = \mathrm{coord}^{-1}(g_1), \qquad
            \frac{\partial \ell}{\partial W} = \mathrm{unvec}(g_2).

        The orthonormal coordinates are an isometry, so that their adjoint is
        their inverse.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (..., n_features*(n_features+1)//2 + n_features*n_samples)
            Gradient of the loss with respect to the output vectors

        Returns
        -------
        grad_input_sym : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the symmetric matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the raw data
        """
        n_features = ctx.n_features
        n_coord = n_features * (n_features + 1) // 2
        grad_input_sym = sym_coordinates_to_matrix(
            grad_output[..., :n_coord], n_features
        )
        grad_input_raw = unvec_batch(grad_output[..., n_coord:], n_features)
        return grad_input_sym, grad_input_raw


# ------------------------------------------------------------
# Two-stream Mixing: Euclidean mixing of the SPD matrices with
# the covariance of the raw data, to restore the link between
# both streams
# ------------------------------------------------------------
def euclidean_mixing(
    data_cov: torch.Tensor, data_raw: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Forward pass of the two-stream Mixing layer (Euclidean path): the SPD
    matrices are mixed with the covariance of the raw data, and the raw data are
    left unchanged.

    .. math::
        P = \alpha\,\Sigma + (1-\alpha)\,\tfrac{1}{n} X X^\top, \qquad Y = X

    Parameters
    ----------
    data_cov : torch.Tensor of shape (..., n_features, n_features)
        Batch of SPD matrices

    data_raw : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data associated with the SPD matrices

    alpha : float
        Fixed mixing weight, in (0, 1]

    Returns
    -------
    data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
        Batch of mixed SPD matrices

    data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
        Batch of raw data (unchanged)
    """
    n_samples = data_raw.shape[-1]
    data_cov_transformed = alpha * data_cov + (1 - alpha) * (
        data_raw @ data_raw.transpose(-1, -2) / n_samples
    )
    return data_cov_transformed, data_raw


class EuclideanMixing(Function):
    """
    Two-stream Mixing layer (Euclidean path): mixing of the SPD matrices with
    the covariance of the raw data
    """

    @staticmethod
    def forward(
        ctx, data_cov: torch.Tensor, data_raw: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the two-stream Mixing layer (Euclidean path)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass

        data_cov : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices

        data_raw : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data

        alpha : float
            Fixed mixing weight, in (0, 1]

        Returns
        -------
        data_cov_transformed : torch.Tensor of shape (..., n_features, n_features)
            Batch of mixed SPD matrices

        data_raw_transformed : torch.Tensor of shape (..., n_features, n_samples)
            Batch of raw data (unchanged)
        """
        data_cov_transformed, _ = euclidean_mixing(data_cov, data_raw, alpha)
        ctx.save_for_backward(data_raw)
        ctx.alpha = alpha
        # clone, so that the output is not the input itself
        return data_cov_transformed, data_raw.clone()

    @staticmethod
    def backward(
        ctx, grad_output_cov: torch.Tensor, grad_output_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        r"""
        Backward pass of the two-stream Mixing layer (Euclidean path).

        With :math:`E = \partial \ell / \partial P` and
        :math:`F = \partial \ell / \partial Y`:

        .. math::
            \frac{\partial \ell}{\partial \Sigma} = \alpha\,E, \qquad
            \frac{\partial \ell}{\partial X} = F + \frac{2(1-\alpha)}{n}\,E X

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output SPD matrices

        grad_output_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the output raw data

        Returns
        -------
        grad_input_cov : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input SPD matrices

        grad_input_raw : torch.Tensor of shape (..., n_features, n_samples)
            Gradient of the loss with respect to the input raw data
        """
        alpha = ctx.alpha
        (data_raw,) = ctx.saved_tensors
        n_samples = data_raw.shape[-1]
        E = symmetrize(grad_output_cov)

        # SPD stream: dl/dSigma = alpha E
        grad_input_cov = alpha * E

        # raw data stream: directly through Y = X, and through the covariance
        grad_input_raw = grad_output_raw + 2 * (1 - alpha) / n_samples * E @ data_raw

        return grad_input_cov, grad_input_raw, None
