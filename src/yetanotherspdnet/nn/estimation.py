"""Covariance estimation layers: sample covariance and robust M-estimators."""

from collections.abc import Callable

import torch
from torch import nn

from ..functions.m_estimators import MEstimator, m_estimator, sample_covariance


class SampleCovariance(nn.Module):
    """
    Sample covariance matrix of a batch of sample sets.

    Maps samples ``(..., n_samples, n_features)`` to SPD matrices
    ``(..., n_features, n_features)``, typically as the first layer of an SPD
    network fed with raw signals.
    """

    def __init__(self, assume_centered: bool = False) -> None:
        """
        Build a SampleCovariance layer.

        Parameters
        ----------
        assume_centered : bool, optional
            Whether the samples are already centered. Default is False

        Attributes
        ----------
        assume_centered : bool
            Whether the samples are assumed centered.
        """
        super().__init__()
        self.assume_centered = assume_centered

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SampleCovariance layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_samples, n_features)
            Samples

        Returns
        -------
        covariance : torch.Tensor of shape (..., n_features, n_features)
            Sample covariance matrices
        """
        return sample_covariance(data, self.assume_centered)

    def __repr__(self) -> str:
        return f"SampleCovariance(assume_centered={self.assume_centered})"


class MEstimation(nn.Module):
    """
    Robust covariance estimation layer (M-estimator of scatter).

    See :mod:`yetanotherspdnet.functions.m_estimators` for the estimator and
    its two gradient paths. The layer has no trainable parameter; it makes the
    estimator usable in a ``torch.nn.Sequential`` and differentiable with
    respect to its input.
    """

    def __init__(
        self,
        weight_function: Callable,
        n_iterations: int = 30,
        tol: float = 1e-6,
        assume_centered: bool = False,
        shrinkage: float | None = None,
        normalize: str | None = None,
        use_autograd: bool = False,
    ) -> None:
        """
        Build an MEstimation layer.

        Parameters
        ----------
        weight_function : Callable
            Weight of squared Mahalanobis distances, e.g.
            ``functools.partial(student_function, n_features=p, nu=3.0)``

        n_iterations : int, optional
            Maximum number of fixed-point iterations. Default is 30

        tol : float, optional
            Stopping tolerance on the relative change. Default is 1e-6

        assume_centered : bool, optional
            Whether the samples are already centered. Default is False

        shrinkage : float | None, optional
            Shrinkage coefficient towards the identity. Default is None

        normalize : str | None, optional
            ``"trace"``, ``"determinant"`` or None, applied at each iteration;
            required for scale-invariant weights such as Tyler's.
            Default is None

        use_autograd : bool, optional
            Unrolled autograd path (True) or implicit backward (False, default)

        Attributes
        ----------
        weight_function : Callable
            Weight function of the estimator.
        n_iterations, tol : int, float
            Stopping criteria of the fixed-point iterations.
        use_autograd : bool
            Gradient path.
        """
        super().__init__()
        self.weight_function = weight_function
        self.n_iterations = n_iterations
        self.tol = tol
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.use_autograd = use_autograd

    def forward(
        self, data: torch.Tensor, init: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the MEstimation layer

        Parameters
        ----------
        data : torch.Tensor of shape (..., n_samples, n_features)
            Samples

        init : torch.Tensor, optional
            Initial estimate. Default is the identity

        Returns
        -------
        covariance : torch.Tensor of shape (..., n_features, n_features)
            Estimated scatter matrices
        """
        args = (
            data,
            self.weight_function,
            self.n_iterations,
            self.tol,
            self.assume_centered,
            init,
            self.shrinkage,
            self.normalize,
        )
        if self.use_autograd:
            return m_estimator(*args)
        return MEstimator.apply(*args)

    def __repr__(self) -> str:
        return (
            f"MEstimation(n_iterations={self.n_iterations}, tol={self.tol}, "
            f"normalize={self.normalize}, use_autograd={self.use_autograd})"
        )
