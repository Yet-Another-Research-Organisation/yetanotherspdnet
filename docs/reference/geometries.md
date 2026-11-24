# Geometries

`yetanotherspdnet.functions.spd_geometries` — one module per Riemannian
geometry of the SPD manifold, each with its geodesic, mean, scalar
dispersion and, where the batch normalization needs them, exponential and
logarithmic maps. The {doc}`../user_guide/geometries` guide compares them and
gives their formulas.

| Module | Geometry | Mean used by the batch normalization |
|---|---|---|
| `affine_invariant` | affine-invariant (Fisher–Rao) | Karcher mean (iterative) |
| `log_euclidean` | log-Euclidean | $\exp(\overline{\log X})$ |
| `euclidean` | Euclidean (left Kullback–Leibler mean) | arithmetic mean |
| `kullback_leibler` | left / right Kullback–Leibler | arithmetic / harmonic mean |
| `kullback_leibler_symmetrized` | symmetrized Kullback–Leibler | GAH: AI midpoint of harmonic and arithmetic (adaptive: learned point $t$) |
| `bures_wasserstein` | Bures–Wasserstein | fixed-point barycenter |
| `affine_invariant_gevd` | affine-invariant, through generalized eigendecompositions | Karcher mean; used by the parallel-transport batch normalization |

## Affine-invariant

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.affine_invariant
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.affine_invariant
   :members:
   :member-order: bysource
```

## Affine-invariant (generalized eigendecomposition)

Geodesic, exponential, logarithm, Karcher mean and parallel transport of the
affine-invariant geometry computed from the generalized eigendecomposition
of a pair $(A, B)$ instead of square roots of $B$. Used by
`norm_and_bias_mode="parallel_transport"` of the batch normalization.

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.affine_invariant_gevd
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.affine_invariant_gevd
   :members:
   :member-order: bysource
```

## Log-Euclidean

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.log_euclidean
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.log_euclidean
   :members:
   :member-order: bysource
```

## Euclidean

Euclidean geodesic and arithmetic mean, the building blocks of the
Kullback–Leibler and log-Euclidean geometries.

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.euclidean
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.euclidean
   :members:
   :member-order: bysource
```

## Kullback–Leibler (arithmetic and harmonic)

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.kullback_leibler
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.kullback_leibler
   :members:
   :member-order: bysource
```

## Symmetrized Kullback–Leibler (GAH)

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized
   :members:
   :member-order: bysource
```

## Bures–Wasserstein

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.bures_wasserstein
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.bures_wasserstein
   :members:
   :member-order: bysource
```

## Registry used by the batch normalization

Each `mean_type` of the batch normalization layers maps to a `Geometry`
holding its mean, geodesic, scalar standard deviation and parallel transport,
each with its autograd and manual-backward implementations. The layers look
their operations up here, so an unsupported combination (for example the
parallel transport of the Bures–Wasserstein mean) is rejected when the layer
is built.

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.registry
   :members: DualPath, Geometry, get_geometry
   :member-order: bysource
```
