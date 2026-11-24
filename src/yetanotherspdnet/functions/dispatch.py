r"""Selection between the autograd and the manual-backward implementation of an operation.

Manual backward is the norm: every layer, model and parametrization defaults to
``use_autograd=False``, where every matrix decomposition (``eigh``,
``eigvalsh``, Cholesky, generalized eigenvalue decomposition) is differentiated
by a hand-written backward (a ``torch.autograd.Function``). Autograd still
differentiates the elementary operations around them (products, sums,
polynomials such as :math:`(I + S/2)^2`), which is exact.

Autograd through a decomposition is only used where no manual backward exists.
Those places are listed in :data:`AUTOGRAD_ONLY`; a layer that reaches one of
them with ``use_autograd=False`` emits a single :class:`AutogradOnlyWarning`
(raising would break a default configuration). The test suite checks that the
autograd graph of every layer and model built with ``use_autograd=False`` holds
no decomposition node besides the listed ones.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass


class AutogradOnlyWarning(UserWarning):
    """An operation without manual backward is differentiated by autograd."""


# operation -> why it has no manual backward (and the manual alternative, if any)
AUTOGRAD_ONLY: dict[str, str] = {
    "yetanotherspdnet.nn.BiMap(parametrization_mode='static')": (
        "the static Stiefel parametrization is torch.nn.utils.parametrizations."
        "orthogonal (Householder product / matrix exponential, differentiated by "
        "autograd; no eigenvalue gap appears in its backward). "
        "parametrization_mode='dynamic' uses the manual QR / polar retractions."
    ),
    "yetanotherspdnet.functions.spd_linalg.inv_SPD": (
        "inverse through the Cholesky factor, differentiated by autograd (stable: "
        "no eigenvalue gap); used inside the manual backwards of the KL "
        "geometries, never on a manual-mode autograd graph."
    ),
    "yetanotherspdnet.functions.spd_geometries.bures_wasserstein."
    "bures_wasserstein_distance_squared": (
        "metric only (no layer differentiates it); the batch normalization uses "
        "the manual BuresWassersteinStdScalar."
    ),
    "yetanotherspdnet.functions.spd_geometries.affine_invariant.affine_invariant_log": (
        "metric only (no layer differentiates it)."
    ),
}


# AUTOGRAD_ONLY keys that already warned in this process
_WARNED: set[str] = set()


def warn_autograd_only(name: str) -> None:
    """
    Warn (once per process) that the operation *name* is differentiated by autograd.

    Parameters
    ----------
    name : str
        Key of :data:`AUTOGRAD_ONLY`

    Raises
    ------
    KeyError
        If *name* is not listed in :data:`AUTOGRAD_ONLY`.
    """
    reason = AUTOGRAD_ONLY[name]
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"use_autograd=False: {name} has no manual backward and is differentiated "
        f"by autograd ({reason}) See yetanotherspdnet.functions.dispatch.AUTOGRAD_ONLY.",
        AutogradOnlyWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class DualPath:
    """
    One operation with its autograd and manual-backward implementations.

    Attributes
    ----------
    autograd : Callable or None
        Implementation differentiated by ``torch.autograd``; None when the
        autograd path is not supported (``unsupported_reason`` says why).
    manual : Callable or None
        Implementation with a hand-written backward; None for an operation of
        :data:`AUTOGRAD_ONLY` (named by ``autograd_only``).
    unsupported_reason : str
        Why the autograd path is not available (empty if it is).
    autograd_only : str
        Key of :data:`AUTOGRAD_ONLY` when ``manual`` is None.
    """

    autograd: Callable | None
    manual: Callable | None
    unsupported_reason: str = ""
    autograd_only: str = ""

    def __post_init__(self) -> None:
        """Check that a missing manual path is listed in :data:`AUTOGRAD_ONLY`."""
        if self.manual is None and self.autograd_only not in AUTOGRAD_ONLY:
            raise ValueError(
                "an operation without manual backward must be listed in "
                f"AUTOGRAD_ONLY, got autograd_only={self.autograd_only!r}"
            )

    def select(self, use_autograd: bool) -> Callable:
        """
        Return the autograd or the manual-backward implementation.

        Parameters
        ----------
        use_autograd : bool
            Whether to return the autograd implementation.

        Returns
        -------
        Callable
            The selected implementation.

        Raises
        ------
        ValueError
            If ``use_autograd`` is True and the autograd path is not supported.

        Warns
        -----
        AutogradOnlyWarning
            If ``use_autograd`` is False and the operation has no manual
            backward (once per operation and process).
        """
        if not use_autograd:
            if self.manual is None:
                warn_autograd_only(self.autograd_only)
                return self.autograd
            return self.manual
        if self.autograd is None:
            raise ValueError(self.unsupported_reason)
        return self.autograd


def select_path(
    autograd: Callable | None, manual: Callable | None, use_autograd: bool
) -> Callable:
    """
    Pick the autograd or the manual-backward implementation of an operation.

    Shorthand for ``DualPath(autograd, manual).select(use_autograd)``, used by
    the layers instead of ``use_autograd`` ladders.

    Parameters
    ----------
    autograd : Callable or None
        Implementation differentiated by autograd
    manual : Callable or None
        Implementation with a hand-written backward
    use_autograd : bool
        Whether to return the autograd implementation

    Returns
    -------
    Callable
        The selected implementation
    """
    return DualPath(autograd, manual).select(use_autograd)
