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
| `kullback_leibler` | left / right Kullback–Leibler | arithmetic / harmonic mean |
| `kullback_leibler_symmetrized` | symmetrized Kullback–Leibler | GAH: AI midpoint of harmonic and arithmetic (adaptive: learned point $t$) |
| `bures_wasserstein` | Bures–Wasserstein | fixed-point barycenter |

## Affine-invariant

```{dualpath-table} yetanotherspdnet.functions.spd_geometries.affine_invariant
```

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_geometries.affine_invariant
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
