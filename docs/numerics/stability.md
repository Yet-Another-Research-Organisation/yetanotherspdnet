# Stability

Real covariances are ill-conditioned (condition numbers of $10^{6}$ on HDM05,
$10^{7}$ on HyperLeaf), batches can hold identical or scalar matrices, and
parameters start at the identity. This page lists what breaks in those cases
and what the package does about it.

| situation | what breaks | what the package does | where |
|---|---|---|---|
| equal eigenvalues (identity, after ReEig, scalar batches) | autograd through `eigh` divides by $\lambda_i - \lambda_j = 0$ | Daleckii–Krein backwards, finite at ties | {doc}`manual_backward` |
| nearly equal eigenvalues | the quotient $\frac{f(\lambda_i) - f(\lambda_j)}{\lambda_i - \lambda_j}$ cancels | closed-form divided differences, relative tie threshold | {doc}`manual_backward` |
| eigenvalues near 0 | $\log$, $x^{-1/2}$ overflow; float32 has no digits left | floors (`spd_floor`, ReEig), mixed precision | below |
| products of ill-conditioned matrices | $B^{1/2} X B^{1/2}$ squares the conditioning | SVD / polar factor of $X^{1/2} B^{1/2}$ | {doc}`manual_backward` |
| Bures–Wasserstein exponential outside its domain | folding, nearly singular outputs | `bw_init_metric="trace_scale"`, $\theta < 1$, float64 | {doc}`../user_guide/batchnorm` |
| unnormalized residual steps | $\lVert V \rVert_X \sim 1/\lambda_{\min}$, the exponential overflows | unit step, floored input | below |
| zero dispersion (identical matrices) | $s/\sigma$ and $\sqrt{\sigma^2}$ backward divide by 0 | floors on $\sigma$, subgradient 0 at 0 | below |
| a batch of one matrix | its statistics are trivial | running statistics, `UserWarning` | {doc}`../user_guide/batchnorm` |

## Floors and clamps

- **ReEig.** $f(\lambda) = \max(\lambda, \epsilon)$ with
  $f'(\lambda) = \mathbb 1[\lambda > \epsilon]$ in the Daleckii–Krein
  backward: no gradient flows through the clamped directions, and the divided
  differences between a clamped and an unclamped eigenvalue interpolate between
  0 and 1. $\epsilon$ must match the scale of the data: eigenvalues below it
  are erased ({doc}`../user_guide/layers`).
- **Residual blocks.** The input eigenvalues are floored at
  `spd_linalg.spd_floor(λ) = max(1e-8, ε max λ)` per matrix, inside the
  single decomposition of the block: $10^{-8}$ in float64 unless
  $\max\lambda > 4.5 \cdot 10^{7}$; in float32, the rounding level of the matrix
  itself (condition number capped at $8 \cdot 10^{6}$). The floored matrix is
  never formed and decomposed again, so its eigenvalues near the floor are exact
  (re-forming it left an error $\varepsilon \lVert X \rVert$, relative
  $2 \cdot 10^{-8}$ at the $10^{-8}$ floor in float64). On well-conditioned
  inputs the floor is exactly the identity, values and gradients.
- **Dispersions.** The scalar standard deviations are $\sqrt{\max(v, 0)}$
  with a zero subgradient at 0 (`std_from_variance`); the exponent $s/\sigma$
  of `BatchNormSPDMeanScalarVariance` uses $\max(\sigma, 3.2 \cdot 10^{-3})$,
  as bounded as the GBWBN factor $s/\sqrt{\sigma^2 + 10^{-5}}$. A batch of
  identical matrices gives finite outputs and gradients.
- **Output clamp.** The reference RResNet clamps the output eigenvalues to
  $[10^{-8}, 10^{8}]$ (`projx`). The unit step keeps them in
  $[e^{-1}\lambda_{\min}(X), e\,\lambda_{\max}(X)]$, so the clamp is only
  applied with `output_clamp=True` or for the unbounded steps (`"none"`,
  `"learned_scale"`), with `EighClamp` (Daleckii–Krein backward).

## The residual step

The affine-invariant norm of the field, $\lVert V \rVert_X = \lVert X^{-1/2}
V X^{-1/2} \rVert_F$, grows like $1/\lambda_{\min}(X)$: an unnormalized step
overflows on real covariances. The unit step (the reference code's choice)
moves every matrix by exactly one unit of affine-invariant distance
({doc}`../user_guide/residual`). The block needs two decompositions:
$X = U \Lambda U^\top$ gives the floored spectrum, $X^{\pm 1/2}$ and $\log X$
with one backward (`FlooredEighFunctions`): since all outputs share $U$, the
chain rule through the floor and through each matrix function $f_k$ collapses
to

$$
\bar X = U\Big(L^{c} \circ \big[\operatorname{diag}(\bar{\tilde\lambda}) +
\textstyle\sum_k L^{f_k} \circ U^\top \bar F_k U\big]\Big)U^\top,
$$

with $L^{c}$ the Loewner matrix of the floor and $L^{f_k}$ those of the
$f_k$ (closed-form divided differences); the exponential of
$\hat W = X^{-1/2} V X^{-1/2} / \nu$ is the second decomposition.

Two options are numerically delicate:

- `residual="branch_pt"` whitens the branch output by the input,
  $N = X^{-1/2} Z X^{-1/2}$, whose condition number reaches
  $\kappa(X)\kappa(Z)$: float32 is unsafe from $\kappa = 10^{4}$, even with
  `eigh_dtype=torch.float64` ($N$ itself is formed in float32), and float64
  decompositions fail on some batches at $\kappa = 10^{8}$;
- `basis="eigenbasis"` puts a position-dependent map of the sorted spectrum on
  the eigenvectors of $X$: discontinuous at repeated eigenvalues and
  ill-conditioned near them (float32 gradient errors of order 1); float64 only.

## Bures–Wasserstein batch normalization

Two separate problems, two remedies:

- **Cross terms.** $(B^{1/2} X B^{1/2})^{1/2}$ computed from the SVD of
  $X^{1/2} B^{1/2}$ (conditioning $\sqrt{\kappa(X)\kappa(B)}$ instead of
  $\kappa(X)\kappa(B)$), with a manual backward of the polar factor; float64
  relative error at $\kappa = 10^{9}$ from $1.3 \cdot 10^{-9}$ (75 % failures)
  to $3 \cdot 10^{-13}$ ({doc}`manual_backward`).
- **Folding.** $\mathrm{Exp}_I(V) = (I + V/2)^2$ is injective only on
  $I + V/2 \succ 0$; outside, the matrix is folded back and can come out
  numerically singular. Folding depends on how far the eigenvalues are from 1,
  not on the condition number as such; it is the cause of the remaining float32
  failures of GBWBN. Remedies: `bw_init_metric="trace_scale"` (absorbs the
  common scale), $\theta < 1$ (compresses the spread), float64
  ({doc}`../user_guide/batchnorm`).

## CPU and GPU decompositions

cuSOLVER's `eigh` can fail to converge on nearly singular batches that LAPACK
handles on CPU, and LAPACK's float32 `eigh` fails to converge on some batches
at $\kappa \ge 10^{4}$. `eigh_dtype=torch.float64` removes those convergence
failures ({doc}`precision`), not the failures caused by folding. For matrices
of a few hundred rows, a CPU run with a few threads is often as fast as a small
GPU.
