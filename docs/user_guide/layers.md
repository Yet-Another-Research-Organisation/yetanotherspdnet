# Layers

The SPDNet layers of `yetanotherspdnet.nn` and the layers that build SPD
matrices from raw samples. API: {doc}`../reference/layers`. The constraints on
their parameters (orthonormal weights) are in {doc}`parametrizations`.

## Mathematics

### BiMap: the linear layer

`BiMap` maps SPD matrices to smaller SPD matrices by a congruence
{ref}`[Huang & Van Gool 2017] <ref-huang2017>`,

$$
X \in \mathcal{S}^{++}_{n_{in}} \;\longmapsto\; W^\top X W \in \mathcal{S}^{++}_{n_{out}},
\qquad W \in \mathbb{R}^{n_{in} \times n_{out}},\ n_{out} \le n_{in}.
$$

The output is SPD as long as $W$ has full column rank:
$y^\top W^\top X W y = (Wy)^\top X (Wy) > 0$ for $y \ne 0$. SPDNet constrains
$W$ to the Stiefel manifold
$\mathrm{St}(n_{in}, n_{out}) = \{W : W^\top W = I\}$, which rules out
degenerate weights and bounds the spectrum of the output by that of the input
(Cauchy interlacing: $\lambda_{\min}(X) \le \lambda_k(W^\top X W) \le
\lambda_{\max}(X)$).

### ReEig and ReEigBias: the non-linearities

`ReEig` is the SPD analogue of ReLU: with $X = U \operatorname{diag}(\lambda)
U^\top$,

$$
\mathrm{ReEig}(X) = U \operatorname{diag}\big(\max(\lambda_i, \epsilon)\big) U^\top .
$$

Besides the non-linearity, it keeps the conditioning of the output bounded
after a dimension reduction. `ReEigBias` adds a learned shift $b_i$ per
eigenvalue (in ascending order) and a two-sided clamp,
$U \operatorname{diag}\big(\operatorname{clamp}(\lambda_i + b_i, \epsilon,
1/\epsilon)\big) U^\top$: it starts as a two-sided ReEig ($b = 0$) and caps
the condition number at $\epsilon^{-2}$.

```{diagram} reeig
:alt: ReEig and ReEigBias as maps of the eigenvalues
:width: 60%

ReEig clamps the eigenvalues below $\epsilon$; ReEigBias shifts them by a
learned $b$ and clamps them to $[\epsilon, 1/\epsilon]$.
```

### LogEig, Vec, Vech: back to vectors

`LogEig` computes $\log X = U \operatorname{diag}(\log \lambda_i) U^\top$. The
logarithm is a diffeomorphism from $\mathcal{S}^{++}_n$ onto the vector space
$\mathrm{Sym}_n$, where a Euclidean classifier makes sense (it is the chart of
the log-Euclidean geometry, {ref}`[Arsigny et al. 2007] <ref-arsigny2007>`).
`Vec` flattens all $n^2$ entries; `Vech` keeps the upper triangle,
$n(n+1)/2$ entries, which carry all the information of a symmetric matrix.

### From samples to SPD matrices

`SampleCovariance` maps samples `(..., n_samples, p)` to
$\frac1n \sum_i (x_i - \bar x)(x_i - \bar x)^\top$. `MEstimation` solves
the fixed point of an M-estimator of scatter,

$$
\Sigma = \frac1n \sum_{i=1}^n u\!\left(x_i^\top \Sigma^{-1} x_i\right) x_i x_i^\top,
$$

whose weight $u$ down-weights outlying samples: Tyler
{ref}`[Tyler 1987] <ref-tyler1987>`, $u(q) = p/q$ (scale invariant, so the
trace or the determinant is normalized at each iteration); Student-t,
$u(q) = (p + \nu)/(\nu + q)$; Huber. They are the first layer of a network fed
with raw signals or with CNN feature maps (covariance pooling).

## Implementation

| layer | eigendecompositions (forward) | backward |
|---|---|---|
| `BiMap` | 0 | manual (`CongruenceRectangular`); the static Stiefel parametrization itself is differentiated by autograd |
| `ReEig`, `ReEigBias` | 1 | Daleckii–Krein, $f'(\lambda) = \mathbb 1[\lambda > \epsilon]$ |
| `LogEig` | 1 | Daleckii–Krein with the closed-form divided differences of $\log$ |
| `Vec`, `Vech` | 0 | manual (index maps) |
| `SampleCovariance` | 0 | autograd (products) |
| `MEstimation` | 0 (Cholesky solves) | implicit differentiation at the fixed point (`use_autograd=False`) or unrolled iterations |

The Daleckii–Krein backward of ReEig gives no gradient to the clamped
directions, and the divided differences between a clamped and an unclamped
eigenvalue interpolate between 0 and 1: it stays finite when several
eigenvalues are clamped to the same $\epsilon$, where autograd through `eigh`
divides by zero ({doc}`../numerics/manual_backward`).

The implicit backward of `MEstimation` solves the adjoint equation
$w = g + J_\Sigma^\top w$ by iteration with vector-Jacobian products only, so
its memory does not grow with the number of iterations
({doc}`../numerics/manual_backward`).

`ReEig`, `ReEigBias` and `LogEig` take `eigh_dtype`: with
`eigh_dtype=torch.float64` in a float32 model the eigendecomposition runs in
float64 and its result is cast back ({doc}`../numerics/precision`).

## Options and strategy

`ReEig(eps)`
: The threshold must match the **scale of the data**: eigenvalues below it are
  erased. The layer default is $10^{-4}$; the models use `reeig_eps=1e-3`.
  Published configurations rescale the data instead: HDM05 covariances are
  multiplied by 190 before an $\epsilon = 0.8$ ReEig, HyperLeaf uses
  $\epsilon = 0.01$ ({doc}`models`). Look at the spectrum of a few input
  matrices after the first BiMap before choosing.

`ReEigBias`
: Use it after estimators that can produce very large eigenvalues, or when
  the condition number must be capped ($1/\epsilon$ on both sides).

`Vec` or `Vech`
: `Vech` (`vec_type="vech"` in the models) halves the size of the linear
  layer with the same information; `Vec` matches older SPDNet code.

`MEstimation`
: Pass the weight as a partial,
  `functools.partial(student_function, n_features=p, nu=3.0)`; Tyler's weight
  needs `normalize="trace"` or `"determinant"`; `shrinkage` regularizes small
  sample sizes ($n < p$). Keep `use_autograd=False` (implicit backward) unless
  the number of iterations is tiny.

```python
import functools

import torch

from yetanotherspdnet.functions.m_estimators import student_function
from yetanotherspdnet.nn import BiMap, LogEig, MEstimation, ReEig, Vech

p = 16
pooling = torch.nn.Sequential(
    MEstimation(functools.partial(student_function, n_features=p, nu=3.0)),
    BiMap(p, 8),
    ReEig(eps=1e-4),
    LogEig(),
    Vech(),
)
samples = torch.randn(4, 200, p, dtype=torch.float64)  # 4 sets of 200 samples
print(pooling(samples).shape)  # torch.Size([4, 36])
```
