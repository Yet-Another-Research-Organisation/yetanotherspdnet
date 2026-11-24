# Geometries

The set $\mathcal{S}^{++}_n$ of SPD matrices carries several Riemannian (or
information) geometries. They lead to different distances, geodesics and, above
all, different **means**, which is what the batch normalizations rely on. Each
geometry has its module under `yetanotherspdnet.functions.spd_geometries`, and
the batch normalizations find their operations through one registry. API:
{doc}`../reference/geometries`.

## Mathematics

Notation: $X, Y \in \mathcal{S}^{++}_n$, $V \in \mathrm{Sym}_n$ a tangent
vector, $X_1, \dots, X_N$ a batch, $G$ a reference point (mean), $\#_t$ the
affine-invariant geodesic, $A \#_t B = A^{1/2}(A^{-1/2} B A^{-1/2})^t A^{1/2}$.

### Affine-invariant

The Fisher–Rao metric of zero-mean Gaussians,
$\langle U, V \rangle_X = \operatorname{tr}(X^{-1} U X^{-1} V)$
{ref}`[Pennec et al. 2006] <ref-pennec2006>`, {ref}`[Moakher 2005] <ref-moakher2005>`.
It is invariant under every congruence $X \mapsto A X A^\top$, $A$
invertible, hence insensitive to the units and to linear mixing of the
underlying signals; its curvature is non-positive, so means are unique.

$$
d(X, Y) = \big\lVert \log(X^{-1/2} Y X^{-1/2}) \big\rVert_F,
$$

$$
\gamma(t) = X \#_t Y,
$$

$$
\mathrm{Exp}_X(V) = X^{1/2} \exp(X^{-1/2} V X^{-1/2}) X^{1/2},
$$

$$
\mathrm{Log}_X(Y) = X^{1/2} \log(X^{-1/2} Y X^{-1/2}) X^{1/2},
$$

$$
\Gamma_{X \to Y}(V) = E V E^\top,\ E = (Y X^{-1})^{1/2}.
$$

The **Karcher mean** $\bar X = \arg\min_G \sum_i d(G, X_i)^2$ has no closed
form; it is the fixed point of
$G \leftarrow G^{1/2} \exp\big(\frac1N \sum_i \log(G^{-1/2} X_i G^{-1/2})\big) G^{1/2}$.
Dispersion: $\sigma^2 = \frac1N \sum_i d(G, X_i)^2$.

### Log-Euclidean

The matrix logarithm is a diffeomorphism onto the vector space
$\mathrm{Sym}_n$; the log-Euclidean geometry is the Euclidean one in this chart
{ref}`[Arsigny et al. 2007] <ref-arsigny2007>`:

$$
d(X, Y) = \lVert \log X - \log Y \rVert_F,
$$

$$
\bar X = \exp\Big(\frac1N \sum_i \log X_i\Big),
$$

$$
\gamma(t) = \exp\big((1-t)\log X + t \log Y\big).
$$

Its exponential map is $\mathrm{Exp}_X(V) = \exp(\log X + D\log_X[V])$, its
parallel transport is the identity in log coordinates. Flat (zero curvature),
closed-form, invariant under orthogonal congruences and scaling, not under
general congruences.

### Kullback–Leibler family

Seeing $X$ as the covariance of $\mathcal N(0, X)$, the Kullback–Leibler
divergence
$\mathrm{KL}(X \Vert Y) = \tfrac12\big[\operatorname{tr}(Y^{-1}X) - n - \log\det(Y^{-1}X)\big]$
is not symmetric, and its two barycenters are closed forms:

| mean (`mean_type`) | barycenter | curve used for the running mean | dispersion $\sigma^2$ |
|---|---|---|---|
| `"arithmetic"` (left KL) | $\frac1N \sum_i X_i$ | $(1-t)X + tY$ | $\frac2N \sum_i \mathrm{KL}(X_i \Vert G)$ |
| `"harmonic"` (right KL) | $\big(\frac1N \sum_i X_i^{-1}\big)^{-1}$ | $\big((1-t)X^{-1} + tY^{-1}\big)^{-1}$ | $\frac2N \sum_i \mathrm{KL}(G \Vert X_i)$ |
| `"geometric_arithmetic_harmonic"` (symmetrized KL) | $\bar X_H \#_{1/2} \bar X_A$ | $H(t) \#_{1/2} A(t)$ (both curves above) | $\frac1N \sum_i J(X_i, G)$ |
| `"adaptive_geometric_arithmetic_harmonic"` | $\bar X_H \#_{t} \bar X_A$, $t$ learned | affine-invariant geodesic | as GAH |

The **GAH mean** is the barycenter of the symmetrized (Jeffreys) divergence
$J = \mathrm{KL}(X \Vert Y) + \mathrm{KL}(Y \Vert X)$
{ref}`[Wang & Vemuri 2004] <ref-wang2004>`: the affine-invariant midpoint of
the harmonic and arithmetic means. It costs two cheap means and one geodesic
and is close to the Karcher mean: it equals the geometric mean for two
commuting matrices, and agrees with it to second order in the spread of a
concentrated batch. The adaptive variant learns the point $t = \operatorname{sigmoid}(\tau)$
on the geodesic from the harmonic to the arithmetic mean ($t = 1/2$ at
initialization). The arithmetic mean is biased towards large eigenvalues, the
harmonic one towards small eigenvalues.

### Bures–Wasserstein

The 2-Wasserstein distance between $\mathcal N(0, X)$ and $\mathcal N(0, Y)$
{ref}`[Bhatia et al. 2019] <ref-bhatia2019>`,
{ref}`[Malagò et al. 2018] <ref-malago2018>`:

$$
d(X, Y)^2 = \operatorname{tr} X + \operatorname{tr} Y
- 2 \operatorname{tr}\big(X^{1/2} Y X^{1/2}\big)^{1/2},
$$

$$
\mathrm{Log}_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2B,
$$

$$
\mathrm{Exp}_B(V) = B + V + Z B Z \ \text{ with } BZ + ZB = V,
$$

$$
\Gamma_{B \to I}(V) = U\Big[\sqrt{\tfrac{2}{\lambda_i + \lambda_j}}\,(U^\top V U)_{ij}\Big] U^\top,
$$

$$
\Gamma_{I \to G}(V) = U'\Big[\sqrt{\tfrac{\delta_i + \delta_j}{2}}\,(U'^\top V U')_{ij}\Big] U'^\top,
$$

for $B = U \operatorname{diag}(\lambda) U^\top$, $G = U' \operatorname{diag}(\delta) U'^\top$:
the exact Levi-Civita transports between $I$ and a point (checked against
an integration of the transport equation in the tests). At the identity,
$\mathrm{Exp}_I(V) = (I + V/2)^2$, injective only on $I + V/2 \succ 0$ (see
{doc}`batchnorm`). The barycenter is the fixed point
$G \leftarrow G^{-1/2}\big(\frac1N \sum_i (G^{1/2} X_i G^{1/2})^{1/2}\big)^2 G^{-1/2}$.
The curvature is non-negative; the metric is invariant under orthogonal
congruences, not under scaling ($d(cX, cY) = \sqrt c\, d(X, Y)$), which is the
origin of the folding of GBWBN.

## Implementation

### One registry

Each `mean_type` maps to a `Geometry` of
`functions.spd_geometries.registry`, holding its mean, geodesic, scalar
dispersion and parallel transport, each as a `DualPath` (autograd and manual
backward). The batch normalizations look their operations up there, so an
unsupported combination (for example the parallel transport of the
Bures–Wasserstein mean) is rejected when the layer is built.

| `mean_type` | mean | geodesic | dispersion | transport (`norm_and_bias_mode="parallel_transport"`) |
|---|---|---|---|---|
| `"affine_invariant"` | Karcher iteration (`n_iterations`) | $\#_t$ | AI | AI, manual only |
| `"affine_invariant_gevd"` | Karcher iteration, generalized eigendecompositions | $\#_t$ | AI | AI (GEVD) |
| `"log_euclidean"` | closed form | LE | LE | LE, manual only |
| `"arithmetic"` | closed form | segment | left KL | AI (GEVD) |
| `"harmonic"` | closed form | harmonic curve | right KL | AI (GEVD) |
| `"geometric_arithmetic_harmonic"` | closed form | GAH curve | Jeffreys | AI (GEVD) |
| `"adaptive_geometric_arithmetic_harmonic"` | closed form, $t$ learned | $\#_t$ | Jeffreys | AI (GEVD) |
| `"bures_wasserstein"` | fixed point (`n_iterations=1`) | BW | BW | none (GBWBN has its own steps) |

### Cost and backward

| geometry | eigendecompositions per matrix (mean) | notes |
|---|---|---|
| affine-invariant | 1 per iteration (log of the whitened matrix), plus 2 per iteration for $G^{\pm 1/2}$ and the exponential | Karcher iterations unrolled, each step with Daleckii–Krein backwards; step size $\eta_k = 0.95^k$, starting at $I$ |
| affine-invariant (GEVD) | 1 generalized eigendecomposition per pair $(X_i, G)$ | avoids the square roots of $G$; used by the parallel-transport batch norm |
| log-Euclidean | 1 ($\log X_i$) and 1 for the mean | closed form |
| arithmetic, harmonic | 0 (Cholesky inverses for the harmonic mean) | cheapest |
| GAH | 0 per matrix, 2 for the geodesic between the two means | almost the price of the arithmetic mean |
| Bures–Wasserstein | 1 ($X_i^{1/2}$) and one SVD per iteration | cross terms through the polar factor |

The Bures–Wasserstein cross terms $(B^{1/2} X B^{1/2})^{1/2}$ are computed
from the SVD of $X^{1/2} B^{1/2}$ (condition number at most
$\sqrt{\kappa(X)\kappa(B)}$) instead of an eigendecomposition of
$B^{1/2} X B^{1/2}$ (condition number $\kappa(X)\kappa(B)$); with the polar
factor $W$, every formula is written without an inverse square root, and the
manual backward of the polar factor is finite at repeated singular values
({doc}`../numerics/manual_backward`).

Every operation exists twice (`snake_case` autograd, `CamelCase` manual): the
tables of {doc}`../reference/geometries` pair them. A few metric functions
that no layer differentiates (`bures_wasserstein_distance_squared`,
`affine_invariant_log`, `log_euclidean_exp`, `log_euclidean_log`) have only the
autograd path; they are listed in {doc}`../numerics/manual_backward`.

## Options and strategy

- **Default:** `"affine_invariant"` for fidelity; `"log_euclidean"` or
  `"geometric_arithmetic_harmonic"` when batches are large or matrices big
  (closed forms, the GAH mean is the default of `SPDnet`).
- **Iterations.** `mean_options={"n_iterations": 5}` for the affine-invariant
  and Bures–Wasserstein means. A few iterations are enough in a batch
  normalization: the running mean smooths the estimate, and the backward cost
  grows with the number of iterations.
- **Baselines.** The arithmetic and harmonic means are the cheapest, and
  biased towards large (respectively small) eigenvalues.
- **Learned compromise.** `"adaptive_geometric_arithmetic_harmonic"` lets
  each layer learn where to sit between the harmonic and the arithmetic mean
  (attribute `t_gah`).
- **Bures–Wasserstein** is the geometry of GBWBN
  (`BatchNormSPDMeanScalarVariance` with `mean_type="bures_wasserstein"`). It
  is not scale invariant: read the folding section of {doc}`batchnorm` before
  using it on data whose eigenvalues are far from 1.
- **Precision.** The affine-invariant means lose accuracy fastest in
  float32 (unsafe from $\kappa = 10^6$ for the mean-only layer); the
  arithmetic mean stays safe everywhere ({doc}`../numerics/precision`).

```python
import torch

from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
    geometric_arithmetic_harmonic_mean,
)
from yetanotherspdnet.functions.spd_geometries.registry import get_geometry
from yetanotherspdnet.random.spd import random_SPD

X = random_SPD(n_features=6, n_matrices=16, cond=50, dtype=torch.float64)
G = geometric_arithmetic_harmonic_mean(X)
geometry = get_geometry("geometric_arithmetic_harmonic")
manual_mean = geometry.mean.select(use_autograd=False)  # the manual-backward path
print(torch.allclose(manual_mean(X), G))  # True
```
