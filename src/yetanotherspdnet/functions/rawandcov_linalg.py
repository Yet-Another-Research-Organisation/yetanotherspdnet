"""Linear algebra for the two-stream (covariance + raw data) SPDnet layers."""

from collections.abc import Callable

import torch
from torch.autograd import Function

from yetanotherspdnet.functions.spd_linalg import (
    eigh_operation,
    eigh_operation_grad,
    symmetrize,
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
