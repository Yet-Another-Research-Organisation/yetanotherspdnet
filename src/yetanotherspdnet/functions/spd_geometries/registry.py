"""Per-geometry registry of the operations used by the SPD batch normalizations.

Each ``mean_type`` of the batch normalization layers maps to a
:class:`Geometry`: its mean, geodesic, scalar dispersion (standard deviation)
and parallel transport, each as a :class:`DualPath` pairing the autograd
implementation (``snake_case``) with the manual-backward one (``CamelCase``).
The layers look their operations up here instead of branching on
``mean_type``, so a missing operation is reported when the layer is built.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ..dispatch import DualPath
from . import (
    affine_invariant,
    affine_invariant_gevd,
    bures_wasserstein,
    euclidean,
    kullback_leibler,
    kullback_leibler_symmetrized,
    log_euclidean,
)


@dataclass(frozen=True)
class Geometry:
    """
    Operations of one SPD geometry, as used by the batch normalizations.

    Attributes
    ----------
    mean : DualPath
        Batch mean, ``mean(data, **options)`` (``mean(data, t)`` when
        ``learnable_t``).
    geodesic : DualPath
        ``geodesic(point1, point2, t)``, used for the running mean (its
        autograd path, under ``torch.no_grad``) and for the minibatch
        regularization of the mean.
    std : DualPath
        Scalar standard deviation ``std(data, reference_point)``.
    transport : DualPath or None
        Parallel transport ``transport(data, current_reference, new_reference)``
        (``norm_and_bias_mode="parallel_transport"``); None if not available.
    mean_options : Mapping[str, object]
        Keyword options forwarded from ``mean_options`` to the mean, with their
        default value (None: only forwarded when given).
    learnable_t : bool
        Whether the mean takes a learnable position ``t`` (adaptive GAH).
    """

    mean: DualPath
    geodesic: DualPath
    std: DualPath
    transport: DualPath | None
    mean_options: Mapping[str, object] = field(default_factory=dict)
    learnable_t: bool = False

    def mean_kwargs(self, mean_options: Mapping | None) -> dict:
        """
        Keyword arguments of the mean, from the layer's ``mean_options``.

        Parameters
        ----------
        mean_options : Mapping or None
            Options given to the layer (unknown keys are ignored).

        Returns
        -------
        dict
            Options of :attr:`mean_options` that are given or have a default.
        """
        given = mean_options or {}
        kwargs = {}
        for key, default in self.mean_options.items():
            if key in given:
                kwargs[key] = given[key]
            elif default is not None:
                kwargs[key] = default
        return kwargs


def _affine_invariant_geodesic_gevd(
    point1: torch.Tensor, point2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """Affine-invariant geodesic (GEVD backend), without the decomposition."""
    return affine_invariant_gevd.affine_invariant_geodesic_gevd(point1, point2, t)[0]


_TRANSPORT_AUTOGRAD_NAN = (
    "norm_and_bias_mode='parallel_transport' with use_autograd=True is not "
    "supported for mean_type={mean_type!r}: the transport takes square roots or "
    "logarithms of the bias, which starts at the identity, where autograd "
    "through torch.linalg.eigh gives NaN gradients (repeated eigenvalues). Use "
    "use_autograd=False (hand-written backward)."
)

_AI = affine_invariant
_GEVD = affine_invariant_gevd
_LE = log_euclidean
_KL = kullback_leibler
_SKL = kullback_leibler_symmetrized
_BW = bures_wasserstein

_AI_STD = DualPath(_AI.affine_invariant_std_scalar, _AI.AffineInvariantStdScalar.apply)
_SKL_STD = DualPath(
    _SKL.symmetrized_kullback_leibler_std_scalar,
    _SKL.SymmetrizedKullbackLeiblerStdScalar.apply,
)
_AI_GEODESIC = DualPath(
    _AI.affine_invariant_geodesic, _AI.AffineInvariantGeodesic.apply
)
# the arithmetic, harmonic and GAH means have no transport of their own: they
# use the affine-invariant one (GEVD backend)
_GEVD_TRANSPORT = DualPath(
    _GEVD.affine_invariant_parallel_transport_gevd,
    _GEVD.AffineInvariantParallelTransportGEVD,
)

GEOMETRIES: dict[str, Geometry] = {
    "affine_invariant": Geometry(
        mean=DualPath(_AI.affine_invariant_mean, _AI.AffineInvariantMean),
        geodesic=_AI_GEODESIC,
        std=_AI_STD,
        transport=DualPath(
            None,
            _AI.AffineInvariantParallelTransport,
            _TRANSPORT_AUTOGRAD_NAN.format(mean_type="affine_invariant"),
        ),
        mean_options={"n_iterations": None},
    ),
    "affine_invariant_gevd": Geometry(
        mean=DualPath(_GEVD.affine_invariant_mean_gevd, _GEVD.AffineInvariantMeanGEVD),
        geodesic=DualPath(
            _affine_invariant_geodesic_gevd, _GEVD.AffineInvariantGeodesicGEVD.apply
        ),
        # the GEVD backend computes the same affine-invariant mean
        std=_AI_STD,
        transport=_GEVD_TRANSPORT,
        mean_options={"n_iterations": None},
    ),
    "log_euclidean": Geometry(
        mean=DualPath(_LE.log_euclidean_mean, _LE.LogEuclideanMean),
        geodesic=DualPath(_LE.log_euclidean_geodesic, _LE.LogEuclideanGeodesic),
        std=DualPath(_LE.log_euclidean_std_scalar, _LE.LogEuclideanStdScalar.apply),
        transport=DualPath(
            None,
            _LE.LogEuclideanParallelTransportSPD,
            _TRANSPORT_AUTOGRAD_NAN.format(mean_type="log_euclidean"),
        ),
    ),
    "arithmetic": Geometry(
        mean=DualPath(euclidean.arithmetic_mean, euclidean.ArithmeticMean.apply),
        geodesic=DualPath(
            euclidean.euclidean_geodesic, euclidean.EuclideanGeodesic.apply
        ),
        std=DualPath(
            _KL.left_kullback_leibler_std_scalar,
            _KL.LeftKullbackLeiblerStdScalar.apply,
        ),
        transport=_GEVD_TRANSPORT,
    ),
    "harmonic": Geometry(
        mean=DualPath(_KL.harmonic_mean, _KL.HarmonicMean.apply),
        geodesic=DualPath(_KL.harmonic_curve, _KL.HarmonicCurve.apply),
        std=DualPath(
            _KL.right_kullback_leibler_std_scalar,
            _KL.RightKullbackLeiblerStdScalar.apply,
        ),
        transport=_GEVD_TRANSPORT,
    ),
    "geometric_arithmetic_harmonic": Geometry(
        mean=DualPath(
            _SKL.geometric_arithmetic_harmonic_mean,
            _SKL.GeometricArithmeticHarmonicMean,
        ),
        geodesic=DualPath(
            _SKL.geometric_euclidean_harmonic_curve,
            _SKL.GeometricEuclideanHarmonicCurve,
        ),
        std=_SKL_STD,
        transport=_GEVD_TRANSPORT,
    ),
    "adaptive_geometric_arithmetic_harmonic": Geometry(
        mean=DualPath(
            _SKL.adaptive_geometric_arithmetic_harmonic_mean,
            _SKL.AdaptiveGeometricArithmeticHarmonicMean,
        ),
        # running mean and minibatch regularization on the AI geodesic
        geodesic=_AI_GEODESIC,
        # same dispersion as the (fixed t = 1/2) GAH mean
        std=_SKL_STD,
        transport=_GEVD_TRANSPORT,
        learnable_t=True,
    ),
    "bures_wasserstein": Geometry(
        mean=DualPath(_BW.bures_wasserstein_mean, _BW.BuresWassersteinMean),
        geodesic=DualPath(_BW.bures_wasserstein_geodesic, _BW.BuresWassersteinGeodesic),
        std=DualPath(
            _BW.bures_wasserstein_std_scalar, _BW.BuresWassersteinStdScalar.apply
        ),
        transport=None,
        mean_options={"n_iterations": 1},
    ),
}


# GBWBN normalization steps (BatchNormSPDMeanScalarVariance with
# mean_type="bures_wasserstein"): center at the identity, scale, bias
BURES_WASSERSTEIN_STEPS: dict[str, DualPath] = {
    "center": DualPath(_BW.bures_wasserstein_center, _BW.BuresWassersteinCenter),
    "scale": DualPath(_BW.bures_wasserstein_scale, _BW.BuresWassersteinScale),
    "bias": DualPath(_BW.bures_wasserstein_bias, _BW.BuresWassersteinBias),
}


def get_geometry(mean_type: str) -> Geometry:
    """
    Return the operations of the geometry named ``mean_type``.

    Parameters
    ----------
    mean_type : str
        One of the keys of :data:`GEOMETRIES`.

    Returns
    -------
    Geometry
        Mean, geodesic, standard deviation and transport of that geometry.

    Raises
    ------
    ValueError
        If ``mean_type`` is not a registered geometry.
    """
    try:
        return GEOMETRIES[mean_type]
    except KeyError:
        raise ValueError(
            f"mean_type must be one of {list(GEOMETRIES)}, got {mean_type!r}"
        ) from None
