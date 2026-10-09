# Geometries on the SPD manifold

The set $\mathcal{S}_n^{++}$ of $n \times n$ SPD matrices can carry several
Riemannian (or information) geometries. They lead to different distances,
geodesics and, above all, different **means**, which is what batch
normalization relies on. Each geometry has its own module under
`yetanotherspdnet.functions.spd_geometries`, with the same naming pattern:
`<geometry>_geodesic`, `<geometry>_mean`, `<geometry>_std_scalar`, and
exp/log maps where they are needed.

All functions take batches `(..., n, n)`; means reduce over every leading
dimension and return one `(n, n)` matrix.

## Overview

| Geometry | Module | Mean | Cost | `mean_type` in batch norm |
|---|---|---|---|---|
| Affine-invariant (AI) | `affine_invariant` | Karcher mean, iterative | highest | `"affine_invariant"` |
| Log-Euclidean (LE) | `log_euclidean` | $\exp(\frac1N\sum \log P_i)$ | one eigendecomposition per matrix | `"log_euclidean"` |
| Euclidean / left KL | `kullback_leibler` | arithmetic $\frac1N\sum P_i$ | lowest | `"arithmetic"` |
| Right KL | `kullback_leibler` | harmonic $(\frac1N\sum P_i^{-1})^{-1}$ | low | `"harmonic"` |
| Symmetrized KL | `kullback_leibler_symmetrized` | GAH: AI midpoint of arithmetic and harmonic | low | `"geometric_arithmetic_harmonic"` |
| Symmetrized KL, adaptive | `kullback_leibler_symmetrized` | point $t$ on the AI geodesic harmonic → arithmetic, $t$ learned | low | `"adaptive_geometric_arithmetic_harmonic"` |
| Bures–Wasserstein (BW) | `bures_wasserstein` | fixed-point barycenter | medium | `"bures_wasserstein"` |

## Affine-invariant

The reference geometry: invariant under congruence $P \mapsto A P A^\top$,
so it is insensitive to the units and mixing of the underlying signals.

$$
d_{AI}(P_1, P_2) = \big\lVert \log(P_1^{-1/2} P_2 P_1^{-1/2}) \big\rVert_F,
\qquad
\gamma(t) = P_1^{1/2}\big(P_1^{-1/2} P_2 P_1^{-1/2}\big)^{t} P_1^{1/2}.
$$

The mean (Fréchet/Karcher mean) has no closed form and is computed by a
fixed-point iteration (`affine_invariant_mean(data, n_iterations=5)`). The
number of iterations trades accuracy for cost; in batch normalization a few
iterations are usually enough because the running mean smooths the estimate.

## Log-Euclidean

Maps the manifold to the flat space of symmetric matrices with the matrix
logarithm, averages there, and maps back:

$$
\bar P = \exp\Big(\frac1N \sum_i \log P_i\Big),
\qquad
\gamma(t) = \exp\big((1-t)\log P_1 + t \log P_2\big).
$$

A closed form, invariant to similarity (orthogonal changes of basis and
scaling) but not to general congruence. A good default when speed matters.

## Kullback–Leibler family

Seeing $P$ as the covariance of a zero-mean Gaussian, the Kullback–Leibler
divergence gives two dual means: the **arithmetic** mean (left KL) and the
**harmonic** mean (right KL). The **symmetrized** KL (Jeffreys) divergence has
as mean the affine-invariant midpoint of the two, the GAH mean:

$$
\bar P_{GAH} = \gamma_{AI}\big(\bar P_{harm}, \bar P_{arith}, \tfrac12\big).
$$

It is almost as cheap as the arithmetic mean and much closer to the AI mean.
The **adaptive** variant replaces $\tfrac12$ by a parameter $t$ learned
during training.

## Bures–Wasserstein

The 2-Wasserstein distance between the Gaussians $\mathcal{N}(0, P_i)$:

$$
d_{BW}(P_1, P_2)^2 = \operatorname{tr} P_1 + \operatorname{tr} P_2
- 2 \operatorname{tr}\big(P_1^{1/2} P_2 P_1^{1/2}\big)^{1/2}.
$$

Its geometry has non-negative curvature and closed-form exp/log maps:

$$
\mathrm{Log}_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2B,
\qquad
\mathrm{Exp}_B(V) = B + V + Z B Z \ \text{ with } BZ + ZB = V .
$$

The barycenter is computed by a fixed-point iteration
(`bures_wasserstein_mean(data, n_iterations=...)`). BW is the geometry of the
GBWBN batch normalization (see {doc}`batchnorm`).

```{note}
`Exp_I` is only injective on tangent vectors $V$ such that $I + V/2$ is
positive definite. Batch normalization transports tangent vectors to the
identity before applying `Exp_I`; when the data are very spread out, some of
them leave that domain and the centred batch no longer has exactly the
identity as mean. For concentrated data, centring is exact.
```

## Choosing a geometry

- **Default:** `"affine_invariant"` for fidelity, `"log_euclidean"` or
  `"geometric_arithmetic_harmonic"` when batches are large or matrices big.
- The arithmetic and harmonic means are the cheapest. They are useful as
  baselines, but they are biased towards large (respectively small)
  eigenvalues.
- `"bures_wasserstein"` together with `BatchNormSPDMeanScalarVariance` gives
  GBWBN, which also learns a pre/post power transform.
