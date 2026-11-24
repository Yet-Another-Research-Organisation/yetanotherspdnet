# Concepts

This page is the map: what an SPD matrix network computes, the geometries the
package offers, its three levels, what flows between them and the conventions
every function and layer follows.

## Mathematics

### The SPD manifold

A symmetric matrix $X \in \mathrm{Sym}_n$ is *positive definite* (SPD) when
$y^\top X y > 0$ for every $y \ne 0$, that is when all its eigenvalues are
positive. The set $\mathcal{S}^{++}_n$ of SPD matrices is an open convex cone
of dimension $n(n+1)/2$, not a vector space: the difference of two
covariances is generally not a covariance, and the straight segment between two
SPD matrices, although it stays SPD, distorts their spectra (the arithmetic
mean inflates the determinant, the *swelling effect*). Its tangent space at
any point is $\mathrm{Sym}_n$ itself.

Covariance matrices are the natural descriptor of EEG/MEG channels, radar and
SAR pixels, hyperspectral patches, skeleton joints over time and CNN feature
maps. SPDNet {ref}`Huang & Van Gool 2017 <ref-huang2017>` transforms them with
layers that map SPD matrices to SPD matrices (a congruence $W^\top X W$, an
eigenvalue rectification) and only takes a matrix logarithm right before a
Euclidean classifier.

A 2 × 2 SPD matrix is drawn as the ellipse $\{x : x^\top X^{-1} x = 1\}$,
whose semi-axes are the square roots of the eigenvalues along the
eigenvectors. The diagrams of this documentation are computed exactly on such
matrices; a symmetric (tangent) matrix is drawn as a cross, one segment per
eigenvector, of length the absolute eigenvalue (blue: positive, red:
negative).

### Geometries at a glance

To average, interpolate or normalize SPD matrices one needs a geometry: a
distance with its geodesics, means and transports. The package has seven,
detailed in {doc}`geometries`:

| geometry | distance or divergence | mean | invariance |
|---|---|---|---|
| affine-invariant | $\lVert \log(X^{-1/2} Y X^{-1/2}) \rVert_F$ | Karcher mean (iterative) | congruence $X \mapsto A X A^\top$ |
| log-Euclidean | $\lVert \log X - \log Y \rVert_F$ | $\exp(\overline{\log X})$ | orthogonal congruence, scaling |
| arithmetic (left KL) | Kullback–Leibler | $\frac1N \sum X_i$ | congruence (of the divergence) |
| harmonic (right KL) | Kullback–Leibler | $(\frac1N \sum X_i^{-1})^{-1}$ | congruence (of the divergence) |
| GAH (symmetrized KL) | Jeffreys | AI midpoint of harmonic and arithmetic | congruence |
| adaptive GAH | — | AI point $t$ (learned) from harmonic to arithmetic | congruence |
| Bures–Wasserstein | $W_2$ between $\mathcal N(0, X)$ and $\mathcal N(0, Y)$ | fixed-point barycenter | orthogonal congruence |

## Implementation

### Three levels

```{diagram} package_levels
:alt: The levels of the package

Dependencies only go downwards: `model` uses `nn`, which uses `functions`.
`random` generates SPD and orthonormal matrices for tests and synthetic data.
```

You can use any level on its own: compute a Karcher mean with
`functions.spd_geometries.affine_invariant`, put a `BatchNormSPDMean` in your
own network, or train an `SPDnet` directly.

### What flows through a network

A batch of SPD matrices has shape `(B, n, n)`; a batch of sequences
`(B, L, n, n)` (every function works on the last two dimensions). Following
one `SPDnet` on HDM05 (93 × 93 skeleton covariances, 117 classes):

| step | operation | shape |
|---|---|---|
| input | covariance of the joint coordinates | `(B, 93, 93)` |
| `BiMap` | $X \mapsto W^\top X W$, $W \in \mathrm{St}(93, 84)$ | `(B, 84, 84)` |
| `ReEig` | $U \max(\Lambda, \epsilon) U^\top$ | `(B, 84, 84)` |
| `BatchNormSPDMean` (optional) | centre the batch at a learned SPD bias | `(B, 84, 84)` |
| `BiMap`, `ReEig`, (BN) | same, $84 \to 63$ | `(B, 63, 63)` |
| `LogEig` | $U \log(\Lambda) U^\top$: symmetric, flat space | `(B, 63, 63)` |
| `Vec` / `Vech` | flatten ($n^2$ or $n(n+1)/2$ entries) | `(B, 3969)` |
| `Linear` | Euclidean classifier | `(B, 117)` |

Everything before `LogEig` stays on the SPD manifold: each layer's output is
SPD whenever its input is.

### Conventions

**Two gradient paths.** Every operation built on a matrix decomposition
exists as a `snake_case` function differentiated by autograd and a
`CamelCase` `torch.autograd.Function` (or a function chaining such classes)
with a hand-written backward. Layers choose with `use_autograd`, `False` by
default: manual backward is the norm, and the few operations without one are
listed ({doc}`../numerics/manual_backward`).

**Matrix functions return tuples.** `logm_SPD`, `sqrtm_SPD`, `powm_SPD`, …
return `(result, eigvals, eigvecs)` so that callers can reuse the
decomposition; take `[0]` for the matrix. The `Function` classes return the
matrix only.

**float64 by default.** Every layer and function takes `dtype` and `device`.
float32 works where the measured table of {doc}`../numerics/precision` says
so, and the layers built on eigendecompositions take
`eigh_dtype=torch.float64` (mixed precision).

**Parameters live on manifolds.** `BiMap` weights stay orthonormal and the
batch normalization biases SPD through `torch.nn.utils.parametrize`
parametrizations: the optimizer updates an unconstrained tensor, a map sends
it to the manifold ({doc}`parametrizations`).

## Options and strategy

| To decide… | Read |
|---|---|
| which layers, and the threshold of ReEig | {doc}`layers` |
| static or dynamic parametrizations | {doc}`parametrizations` |
| which geometry for the batch normalization | {doc}`geometries`, {doc}`batchnorm` |
| whether to use residual blocks, and which step | {doc}`residual` |
| the constructor arguments of the models | {doc}`models` |
| `use_autograd`, float32, `eigh_dtype` | {doc}`../numerics/index` |
