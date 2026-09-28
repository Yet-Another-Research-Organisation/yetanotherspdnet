# How the library fits together

This page is the map: what the three levels of the package do, what flows
between them, and the conventions every function and layer follows. The
other guides go into each part.

## Three levels

```text
model        SPDnet, RResNet, GBWBNRResNet                 classifiers, one constructor
  │            builds a torch.nn.Sequential of ↓
nn           BiMap, ReEig, LogEig, BatchNorm…, ResidualBlock   modules with parameters
  │            call ↓ and keep their parameters on a manifold with parametrizations
functions    spd_linalg, spd_geometries/*, m_estimators, stiefel   pure tensor functions
```

Dependencies only go downwards. You can use any level on its own: compute a
Karcher mean with `functions.spd_geometries.affine_invariant`, put a
`BatchNormSPDMean` in your own network, or train an `SPDnet` directly.

## What flows through a network

A batch of SPD matrices has shape `(B, n, n)`; a batch of sequences of SPD
matrices `(B, L, n, n)` (every function works on the last two dimensions).
Following one `SPDnet` on HDM05 (93×93 skeleton covariances, 117 classes):

| Step | Operation | Shape |
|---|---|---|
| input | covariance of the joint coordinates | `(B, 93, 93)` |
| `BiMap` | $X \mapsto W^\top X W$, $W \in \mathrm{St}(93, 84)$ | `(B, 84, 84)` |
| `ReEig` | $U \max(\Lambda, \epsilon) U^\top$ | `(B, 84, 84)` |
| `BatchNormSPDMean` (optional) | centre the batch at a learned SPD bias | `(B, 84, 84)` |
| `BiMap`, `ReEig`, (BN) | same, $84 \to 63$ | `(B, 63, 63)` |
| `LogEig` | $U \log(\Lambda) U^\top$: symmetric, flat space | `(B, 63, 63)` |
| `Vec` / `Vech` | flatten ($n^2$ or $n(n+1)/2$ entries) | `(B, 3969)` |
| `Linear` | Euclidean classifier | `(B, 117)` |

Everything before `LogEig` stays on the SPD manifold; the layers are designed
so that their output is SPD whenever their input is.

## Conventions

**float64 by default.** Eigendecompositions of ill-conditioned matrices lose
accuracy quickly in float32, and the gradients of spectral functions divide
by eigenvalue gaps. Every layer and function takes `dtype` and `device`
arguments; float32 is possible but not the tested default.

**Two gradient paths.** Every operation built on an eigendecomposition
exists as a `snake_case` function differentiated by autograd and a
`CamelCase` `torch.autograd.Function` with a hand-written backward. Layers
choose with `use_autograd`, which defaults to `False` (hand-written). The
hand-written backwards stay finite and exact when eigenvalues are close or
equal, which autograd through `torch.linalg.eigh` does not
({doc}`numerics`).

**Matrix functions return tuples.** `logm_SPD`, `sqrtm_SPD`, `powm_SPD`, …
return `(result, eigvals, eigvecs)` so that callers can reuse the
decomposition; take `[0]` for the matrix. The `Function` classes return the
matrix only.

**Parameters live on manifolds.** `BiMap` weights must stay orthonormal and
batch normalization biases SPD. Both are `torch.nn.utils.parametrize`
parametrizations: the optimizer updates an unconstrained tensor, and a map
sends it to the manifold. The map can be fixed (*static*) or re-centred
during training (*dynamic*) ({doc}`layers`).

## Where to go next

| To understand… | Read |
|---|---|
| BiMap, ReEig, LogEig and the static / dynamic parametrizations | {doc}`layers` |
| which geometry to use and what its mean is | {doc}`geometries` |
| what the batch normalization does, and its options | {doc}`batchnorm` |
| the residual block of RResNet | {doc}`residual` |
| the constructor arguments of the models, grouped by role | {doc}`models` |
| `use_autograd`, float64 and the hand-written backwards | {doc}`numerics` |
