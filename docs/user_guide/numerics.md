# Numerical and optimization techniques

Training on the SPD manifold means differentiating eigendecompositions,
solving matrix equations inside backward passes, keeping parameters on
manifolds, and surviving very ill-conditioned data. This page collects the
techniques the library uses for each of these, with their equations and where
they live in the code.

## 1. Spectral functions and the Daleckii–Krein backward

Almost every operation is a *spectral function*: with $X = U
\operatorname{diag}(\lambda) U^\top$,

$$
f(X) = U \operatorname{diag}\big(f(\lambda_1), \dots, f(\lambda_n)\big) U^\top
$$

($\log$, $\exp$, $\sqrt{\cdot}$, $x^p$, ReEig's $\max(x, \epsilon)$, the
scaled softplus, …; `functions.spd_linalg.eigh_operation`). Its gradient is
given by the **Daleckii–Krein formula**: for an upstream gradient $\bar Y$,

$$
\bar X = U \big( L \circ (U^\top \bar Y U) \big) U^\top,
\qquad
L_{ij} =
\begin{cases}
\dfrac{f(\lambda_i) - f(\lambda_j)}{\lambda_i - \lambda_j} & \lambda_i \neq \lambda_j,\\[2mm]
f'(\lambda_i) & \lambda_i = \lambda_j,
\end{cases}
$$

where $\circ$ is the entrywise product and $L$ the *Loewner matrix* of divided
differences (`eigh_operation_grad`).

**Computing the divided differences.** The plain quotient cancels when
$\lambda_i \approx \lambda_j$: its rounding error is about
$\varepsilon |\lambda| / |\lambda_i - \lambda_j|$. For the functions of the
library it is replaced by closed forms that never cancel ($l = \min$, $d =
|\lambda_i - \lambda_j|$):

| $f$ | divided difference |
|---|---|
| $\sqrt x$ | $1 / (\sqrt{\lambda_i} + \sqrt{\lambda_j})$ |
| $x^{-1/2}$ | $-1 / \big(\sqrt{\lambda_i}\sqrt{\lambda_j}(\sqrt{\lambda_i} + \sqrt{\lambda_j})\big)$ |
| $x^{-1}$ | $-1 / (\lambda_i \lambda_j)$ |
| $\log x$ | $\operatorname{log1p}(d / l) / d$ |
| $\exp x$ | $e^{l} \operatorname{expm1}(d) / d$ |
| $x^t$ | $l^t \operatorname{expm1}\big(t \operatorname{log1p}(d / l)\big) / d$ |

and the derivative at the midpoint $f'((\lambda_i + \lambda_j)/2)$ is only used
at rounding-level ties, $d \le c\, n\, \varepsilon \max_k |\lambda_k|$ with
$c = 1$ (`EIGVAL_TIE_RTOL`) and $\varepsilon$ the machine epsilon of the dtype.
Other functions (softplus, ReEig's clamp) use the quotient down to
$d = (24 \varepsilon)^{1/3} \max(|\lambda_i|, |\lambda_j|)$, where its
rounding error equals the $O(d^2)$ error of the midpoint derivative, and the
midpoint derivative below. The midpoint keeps $L$ exactly symmetric. The former
absolute threshold $|\lambda_i - \lambda_j| < 10^{-6}$ made the gradient of
$\sqrt{\cdot}$ 10.7 % wrong for eigenvalues $10^{-7}$ and $5 \cdot 10^{-7}$;
the kernels are now within $10^{-10}$ relative of a 50-digit reference.

Autograd through `torch.linalg.eigh` instead differentiates the eigenvectors,
which involves $1/(\lambda_i - \lambda_j)$: infinite for equal eigenvalues.
Equal eigenvalues are not a corner case here: ReEig clamps every small
eigenvalue to the same $\epsilon$, and parameters initialized at the identity
(batch normalization bias, GBWBN's $M$ and $G$) have all their eigenvalues
equal. The divided difference $L_{ij}$ stays bounded in both cases.

```{figure} ../_static/diagrams/daleckii_krein.svg
:width: 95%

After ReEig, three eigenvalues equal $\epsilon$. Left: the factors
$1/|\lambda_i - \lambda_j|$ that autograd multiplies by are infinite on the
tied block. Right: the Loewner matrix of the Daleckii–Krein backward is
bounded (0 inside the clamped block, where $f' = 0$).
```

**The two gradient paths.** Each such operation exists as a `snake_case`
function differentiated by autograd (`logm_SPD`, `sqrtm_SPD`, …) and as a
`CamelCase` `torch.autograd.Function` with the backward above (`LogmSPD`,
`SqrtmSPD`, …). Layers choose with `use_autograd`, `False` (manual) by
default; the models accept a dict per layer type:

```python
SPDnet(
    input_dim=8,
    hidden_layers_size=[4],
    output_dim=3,
    use_autograd={"bimap": True, "batchnorm": True},
)  # others stay manual
```

Valid keys: `"bimap"`, `"reeig"`, `"logeig"`, `"batchnorm"`, `"vec"`, and
`"residual"` for the residual models. The {doc}`../reference/spd_linalg` page
lists the pairs.

**Composite operations are built by chaining these primitives** rather than
by writing one large backward: the whitening, the means and the batch
normalizations are compositions of manually differentiated steps, so each
piece stays exact.

**Manual backward is the norm.** With `use_autograd=False` (the default of
every layer, model and parametrization) no matrix decomposition (`eigh`,
`eigvalsh`, Cholesky, generalized eigendecomposition) is differentiated by
autograd: each one sits inside a `torch.autograd.Function` with a hand-written
backward. Autograd still differentiates the elementary operations between
them (products, sums, polynomials such as the Bures–Wasserstein
$\mathrm{Exp}_I(S) = (I + S/2)^2$), which is exact. The layers pick their
implementations through one helper, `functions.dispatch.select_path` /
`DualPath`. The test suite walks the autograd graph of every layer and model
built with the defaults and fails if it finds a decomposition node outside the
explicit list below (`functions.dispatch.AUTOGRAD_ONLY`).

| operation without manual backward | why, and the manual alternative |
|---|---|
| `BiMap(parametrization_mode="static")` | torch's `parametrizations.orthogonal` (Householder product); its backward has no eigenvalue gap. `parametrization_mode="dynamic"` uses the manual QR / polar retractions. |
| `spd_linalg.inv_SPD` | Cholesky inverse, differentiated by autograd (stable); only used inside manual backwards. |
| `bures_wasserstein_distance_squared`, `affine_invariant_log`, `log_euclidean_exp`, `log_euclidean_log` | metrics, not differentiated by any layer. |

Reaching one of them with `use_autograd=False` emits a single
`AutogradOnlyWarning` per process (raising would break the default
`BiMap`), never a silent fallback.

## 2. Matrix equations inside backward passes

Some backwards require solving a linear matrix equation. With
$A = V \operatorname{diag}(\alpha) V^\top \succ 0$, the Sylvester (Lyapunov)
equation $AZ + ZA = C$ has the closed form

$$
Z = V \left[ \frac{(V^\top C V)_{ij}}{\alpha_i + \alpha_j} \right]_{ij} V^\top
$$

(`solve_sylvester_SPD`): the denominators are positive even when eigenvalues
are equal. It appears in:

- **whitening** $X \mapsto G^{-1/2} X G^{-1/2}$ (the centring of the batch
  normalization): $\bar X = G^{-1/2} \bar Y G^{-1/2}$, and the gradient with
  respect to $G$ solves $G^{1/2} Z + Z G^{1/2} = -2\,\mathrm{sym}\big(\sum_b
  \bar X_b X_b G^{-1/2}\big)$, summed over the batch (`Whitening`);
- **the Bures–Wasserstein exponential** $\mathrm{Exp}_B(V) = B + V + ZBZ$ with
  $BZ + ZB = V$: differentiated implicitly (`LyapunovSolveSPD`),
  $\bar V = W$ and $\bar B = -(WZ + ZW)$ where $BW + WB = \bar Z$;
- **the Bures–Wasserstein parallel transport** from $I$ to $G$, the square root
  of the operator $S \mapsto (GS + SG)/2$, whose derivative solves a Sylvester
  equation between operators with denominators
  $b_{ij} + b_{kj}$, $b_{ij} = \sqrt{(\delta_i + \delta_j)/2} > 0$
  (`ParallelTransportFromIdentityBW`), finite even at $G = I$;
- **the polar factor** $W = UV^\top$ of $A = U\Sigma V^\top$
  (`spd_linalg.PolarFactor`), through which every Bures–Wasserstein cross
  term is computed ([section 3](#3-means-as-iterations)): differentiating
  $A = WH$ gives $dW = U\hat\Omega V^\top$ with
  $\hat\Omega_{ij} = (M_{ij} - M_{ji})/(\sigma_i + \sigma_j)$,
  $M = U^\top dA\,V$, a Sylvester equation with positive denominators, finite
  at repeated singular values (the SVD's own backward has
  $1/(\sigma_i^2 - \sigma_j^2)$).

## 3. Means as iterations

| Mean | Computation | Notes |
|---|---|---|
| affine-invariant (Karcher) | $M_{k+1} = M_k^{1/2} \exp\!\big(\eta_k \tfrac1N \sum_i \log(M_k^{-1/2} X_i M_k^{-1/2})\big) M_k^{1/2}$, $\eta_k = 0.95^k$ | starts at $I$; `mean_options={"n_iterations": 5}` |
| log-Euclidean | $\exp\big(\tfrac1N\sum_i \log X_i\big)$ | closed form |
| arithmetic / harmonic | $\tfrac1N\sum_i X_i$ / $\big(\tfrac1N\sum_i X_i^{-1}\big)^{-1}$ | inverses through Cholesky |
| GAH | $H \#_{1/2} A = H^{1/2}\big(H^{-1/2} A H^{-1/2}\big)^{1/2} H^{1/2}$ | closed form, close to the Karcher mean at the cost of two cheap means |
| adaptive GAH | $H \#_t A$, $t = \operatorname{sigmoid}(\tau)$ learned | $t = 1/2$ at initialization |
| Bures–Wasserstein | $G_{k+1} = G_k^{-1/2}\big(\tfrac1N\sum_i (G_k^{1/2} X_i G_k^{1/2})^{1/2}\big)^2 G_k^{-1/2} = C_k C_k^\top$, $C_k = \tfrac1N \sum_i X_i^{1/2} W_i$ | from the arithmetic mean, one iteration by default; $W_i$ polar factor of $X_i^{1/2} G_k^{1/2}$ (below) |

**Bures–Wasserstein cross terms.** Every BW operation needs
$(B^{1/2} X B^{1/2})^{1/2}$. An eigendecomposition of $B^{1/2} X B^{1/2}$ works
at the product of the condition numbers of $X$ and $B$: its small eigenvalues
lose their digits from $\kappa \approx 10^{8}$ in float64 and $10^{4}$ in
float32. The package takes instead the SVD of $A = X^{1/2} B^{1/2} = U\Sigma
V^\top$ (condition number at most $\sqrt{\kappa(X)\kappa(B)}$):
$(B^{1/2} X B^{1/2})^{1/2} = V\Sigma V^\top$, and with the polar factor
$W = UV^\top$ every formula is written without inverse square root:
$(BX)^{1/2} = B^{1/2} W^\top X^{1/2}$ (log map), $d_{BW}^2 = \lVert X^{1/2} W
- B^{1/2}\rVert_F^2$ (distance and std, no cancellation near $X = B$; $W$
minimizes this over orthogonal matrices, so it does not enter the gradient),
the geodesic $Y_t Y_t^\top$ with $Y_t = (1-t)X_1^{1/2} + tX_2^{1/2}W$, and the
mean step above. Against a 60-digit reference ($n = 6$, $\kappa(X), \kappa(B)
\le 10^9$), the float64 relative error at $\kappa = 10^9$ goes from
$1.3 \cdot 10^{-9}$ with 75 % failures (log map) and 1.0 (std) to
$3 \cdot 10^{-13}$ and $10^{-14}$; in float32 the operations no longer fail
up to $\kappa = 10^{7}$ (error $\le 3.4 \cdot 10^{-5}$), where the
eigendecomposition failed from $10^{4}$ to $10^{6}$.

The iterations are unrolled, and the gradient flows through each of them with
the manual spectral backwards. A small, fixed number of iterations is
deliberate: the batch statistics only need to be good enough to normalize,
and the backward cost grows with the number of iterations.

## 4. Implicit differentiation at a fixed point

An M-estimator of scatter is the fixed point
$\Sigma^\star = F(\Sigma^\star, x)$, with
$F(\Sigma, x) = \frac1n\sum_i u(x_i^\top \Sigma^{-1} x_i)\, x_i x_i^\top$.
Differentiating the $K$ unrolled iterations costs memory proportional to $K$.
At the fixed point, the implicit function theorem gives instead

$$
w = g + J_\Sigma^\top w, \qquad \bar x = J_x^\top w,
$$

with $g$ the upstream gradient and $J_\Sigma$, $J_x$ the Jacobians of $F$. The
library solves this adjoint equation by iteration, using only
vector-Jacobian products (`functions.m_estimators.MEstimator`), so the memory
cost does not depend on $K$. Tyler's estimator is scale invariant
($F(c\Sigma) = cF(\Sigma)$, so $J_\Sigma$ has the eigenvalue 1); normalizing
the trace or the determinant at each iteration removes that direction.

```{figure} ../_static/diagrams/implicit_diff.svg
:width: 90%

Unrolled vs implicit differentiation of a fixed point.
```

## 5. Parameters on manifolds without a Riemannian optimizer

Constraints are `torch.nn.utils.parametrize` parametrizations, so standard
optimizers work unchanged ({doc}`layers` has the equations of the static and
dynamic modes):

- orthonormal BiMap weights: `orthogonal` (static) or a QR / polar retraction
  of a tangent vector at a moving reference point (dynamic);
- SPD biases: $S \mapsto f(S)$ with $f = \exp$ or the scaled softplus
  $f(x) = \log_2(1 + 2^x)$, which satisfies $f(0) = 1$: a zero parameter is the
  identity matrix;
- positive scalars (the batch normalization scale $s$): scaled softplus;
  scalars in $(0, 1)$ (the adaptive GAH $t$): sigmoid.

```{figure} ../_static/diagrams/parametrization.svg
:width: 85%

A static chart distorts the geometry far from its base point; the dynamic mode
moves the base point to the current weight every `n_steps_ref_update` steps.
```

The reference point and the last value of an adaptive parametrization are
separate buffers (`clone()`d): a buffer sharing the storage of the reference
point would silently move it at every forward pass.

## 6. Batch normalization mechanics

- **Centring** by whitening with the batch mean $\bar X$ (a congruence, so
  affine-invariant distances within the batch are preserved), then
  **re-biasing** by the congruence with the learned $G$
  ({doc}`batchnorm`).
- **Scaling** (`BatchNormSPDMeanScalarVariance`): the centred matrices are
  raised to the power $s/\sigma$, which moves them along the geodesics from
  $I$ by the factor $s/\sigma$; $\sigma^2$ is the mean squared distance to the
  mean in the chosen geometry.
- **Running statistics** follow the geometry: $M_r \leftarrow M_r \#_\mu
  \bar X$ (geodesic step of size `momentum` $\mu$) and
  $\sigma_r^2 \leftarrow (1-\mu)\sigma_r^2 + \mu\,\sigma^2$.
- **Minibatch smoothing** (`norm_strategy="minibatch"`): normalize with
  $m_k = m_{k-1} \#_{\mu_k} \bar X_k$ instead of $\bar X_k$, to reduce the noise
  of small batches.
- **A batch of one matrix** has no statistics (its mean is itself, its
  dispersion zero): it is normalized with the running statistics, which are not
  updated, and a warning is emitted.
- **GBWBN** works in Bures–Wasserstein geometry, whose exponential at the
  identity $\mathrm{Exp}_I(V) = (I + V/2)^2$ is only injective when
  $I + V/2 \succ 0$. Outside that domain, a matrix is folded back and can come
  out numerically singular. The BW metric is not scale invariant, so folding
  is driven by how far the eigenvalues are from 1 (their common scale and
  their spread), not by the condition number as such: $\kappa = 10^2$ data
  with eigenvalues in $[10^{-2}, 1]$ never fold, the same data times $10^2$
  fold at the centring step for 99.8 % of the matrices ($\theta = 1$).
  `bw_init_metric="trace_scale"` absorbs the common scale into $M$ and $G$
  (0 %), and the power $\theta < 1$ compresses the spread; see {doc}`batchnorm`.

```{figure} ../_static/diagrams/bw_fold.svg
:width: 60%

The scalar picture of the Bures–Wasserstein exponential at the identity:
beyond $v = -2$ two tangent vectors give the same point, and $v = -2$ gives a
singular matrix.
```

## 7. The residual step

- **Unit step.** The affine-invariant norm of the spectral vector field,
  $\lVert V\rVert_X = \lVert X^{-1/2} V X^{-1/2} \rVert_F$, grows like
  $1/\lambda_{\min}(X)$; an unnormalized step overflows on real covariances.
  The step is normalized to unit length, as in the reference implementation
  (`step="clip"` and `step="learned_scale"` bound or learn the length, see
  {doc}`residual`).
- **Two eigendecompositions per block.** One decomposition
  $X = U\Lambda U^\top$ gives the floored spectrum of the field,
  $X^{\pm 1/2}$ and $\log X$ (`FlooredEighFunctions`, one Daleckii–Krein
  backward for all outputs: since they share $U$, the chain rule through the
  floor and through each matrix function collapses to
  $\bar X = U\big(L^{c} \circ [\operatorname{diag}(\bar{\tilde\lambda}) +
  \sum_k L^{f_k} \circ U^\top \bar F_k U]\big)U^\top$, with the closed-form
  divided differences of section 1). The norm is
  $\lVert W \rVert_F$, $W = X^{-1/2} V X^{-1/2}$ (no Cholesky
  factorization), and the exponential $X^{1/2} e^{W/\nu} X^{1/2}$ is the
  second decomposition. Before: ReEig, `eigvalsh`, Cholesky, $X^{\pm1/2}$,
  $\exp$ and the output clamp (5 decompositions and a Cholesky); the floored
  matrix was formed and decomposed again, so its eigenvalues near the floor
  carried an error $\varepsilon\lVert X\rVert$ (relative $2\cdot 10^{-8}$ at
  the $10^{-8}$ floor in float64). Forward + backward on CPU (3 threads, batch 64,
  $n = 64$, float64): affine-invariant block 64 → 30 ms, log-Euclidean block
  40 → 25 ms, RResNet with two blocks 145 → 76 ms (1.5–2.5× faster over
  $n \in \{16, 64\}$, float32 and float64).
- **Floor, no output clamp.** The input eigenvalues are floored at
  `spd_floor(λ) = max(1e-8, ε max λ)` per matrix: $10^{-8}$ in float64 unless
  $\max\lambda > 4.5 \cdot 10^7$, the rounding level of the matrix itself in
  float32 (condition number capped at $8 \cdot 10^6$). The reference clamps
  the output eigenvalues to $[10^{-8}, 10^8]$ (`projx`); the unit step keeps
  them in $[e^{-1}\lambda_{\min}(X), e\,\lambda_{\max}(X)]$ (the
  eigenvalues of $e^{W/\nu}$ are in $[e^{-1}, e]$), so the clamp is only
  applied with `output_clamp=True` or for the unbounded steps (`"none"`,
  `"learned_scale"`), with `EighClamp` (Daleckii–Krein backward;
  `affine_invariant_projx` is a deprecated alias).
- **Branch with transport.** `residual="branch_pt"` whitens the branch output
  by the input, $N = X^{-1/2} Z X^{-1/2}$, whose condition number reaches
  $\kappa(X)\kappa(Z)$: float32 is unsafe from $\kappa = 10^4$, even
  with `eigh_dtype=torch.float64` (the matrix $N$ itself is formed in
  float32), and float64 decompositions fail on some batches at
  $\kappa = 10^8$.
- **Eigenbasis field.** `basis="eigenbasis"` puts a position-dependent map of
  the sorted spectrum on the eigenvectors of $X$: discontinuous at repeated
  eigenvalues, ill-conditioned near them (float32 gradient errors of order
  1); float64 only.

## 8. ReEig and its variants

```{figure} ../_static/diagrams/reeig.svg
:width: 55%

ReEig clamps the eigenvalues below $\epsilon$; ReEigBias shifts them by a
learned $b$ and clamps them to $[\epsilon, 1/\epsilon]$.
```

The derivative used in the Daleckii–Krein backward is $f'(\lambda) =
\mathbb{1}[\lambda > \epsilon]$: no gradient flows through the clamped
directions, and the divided differences between a clamped and an unclamped
eigenvalue interpolate between 0 and 1. $\epsilon$ must match the scale of
the data: eigenvalues below it are erased (for instance, HDM05 covariances
are multiplied by 190 before an $\epsilon = 0.8$ ReEig).

## 9. Precision and devices

- **float64 by default.** The conditioning of real covariances reaches
  $10^{6}$ (HDM05) to $10^{7}$ (HyperLeaf); eigendecompositions and the
  divided differences lose too many digits in float32. Every function and layer
  takes `dtype` and `device`.
- **Mixed precision: `eigh_dtype`.** `ReEig`, `ReEigBias`, `LogEig`, the batch
  normalizations, `SpectralVectorField`, `ResidualBlock` and the models take
  `eigh_dtype=torch.float64`: in a float32 model the eigendecompositions are
  computed on the float32 matrices cast to float64 and cast back
  (`spd_linalg.eigh_precision`, `symmetric_eigh`), everything else stays
  float32. float32 `eigh` resolves the eigenvalues only to about
  $\varepsilon_{32} \lVert X \rVert$; in float64 the small eigenvalues of the
  (rounded) float32 matrix are exact to $\varepsilon_{64}\lVert X\rVert$. It
  cannot remove the rounding of the input itself, so it helps most where the
  decomposition, not the data, is the limit (log-Euclidean, ReEig, the
  residual blocks, SPDnet). Against float64 on the original (unrounded) data
  it cannot go below the rounding of the input, about $\kappa\,
  \varepsilon_{32}$ (log-Euclidean batch norm at $\kappa = 10^{6}$: output
  error $7 \cdot 10^{-3}$ in float32, $7 \cdot 10^{-4}$ mixed). On CPU (3
  threads, batch 32, $n = 64$) a mixed forward + backward costs 77–93 %
  (median 86 %) of a float64 one, plain float32 59–81 % (median 72 %). GPU
  timings were not measured.
- **Eigenvalue floors follow the dtype.** `spd_linalg.spd_floor(λ) =
  max(1e-8, ε max λ)` (per matrix) is the floor of the residual blocks; ReEig
  keeps its explicit `eps=1e-4`. The scalar standard deviations are
  $\sqrt{\max(v, 0)}$ with a zero subgradient at 0 (`std_from_variance`), and
  the exponent $s/\sigma$ of `BatchNormSPDMeanScalarVariance` uses
  $\max(\sigma, 3.2 \cdot 10^{-3})$, as bounded as the GBWBN scaling
  $s/\sqrt{\sigma^2 + 10^{-5}}$: a batch of identical matrices gives finite
  outputs and gradients.
- **CPU vs GPU eigendecompositions.** cuSOLVER's `eigh` can fail to converge on
  nearly singular batches that LAPACK handles on CPU. For matrices of a few
  hundred rows, a CPU run with a few threads is often as fast as a small GPU.
  LAPACK's float32 `eigh` also fails to converge on some batches at
  $\kappa \ge 10^{4}$; `eigh_dtype=torch.float64` removes those convergence
  failures, not the failures of GBWBN caused by folding (a folded, singular
  centred matrix, see below).
- **Tests.** The manual backwards are checked with `torch.autograd.gradcheck`
  in float64 (repeated, nearly repeated eigenvalues, the identity,
  $\kappa = 10^{6}$) and compared with the autograd path; every layer is run in
  float32 against a float64 reference for $\kappa \le 10^{4}$.

### When is float32 enough?

Measured with the precision benchmark (random SPD batches of condition number
$\kappa$, $n \in \{16, 64\}$, batch 32, 8 seeds, CPU; forward + backward in
float32 against float64 on the same float32-rounded input, default options,
training mode). The output error is the larger of the errors against float64
on the rounded input and on the original input. **safe**: no failure, relative
error $\le 10^{-4}$ on the output and $\le 10^{-2}$ on the input gradient; **degraded**: no failure,
output error $\le 10^{-2}$, gradient error $\le 1$; **unsafe**: failures
(exceptions, non-finite values) or larger errors. "mixed" is float32 with
`eigh_dtype=torch.float64`.

| layer | float32: κ ≤ 1e2 | 1e4 | 1e6 | 1e8 | mixed: κ ≤ 1e2 | 1e4 | 1e6 | 1e8 |
|---|---|---|---|---|---|---|---|---|
| ReEig(1e-4) | safe | safe | safe | degraded | safe | safe | safe | safe |
| BN scalar var. [adaptive GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [AI] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [AI (GEVD)] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [arithmetic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [BW] | safe | unsafe | unsafe | unsafe | safe | unsafe | unsafe | unsafe |
| BN scalar var. [GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [harmonic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN scalar var. [LE] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [adaptive GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [AI] | safe | safe | unsafe | unsafe | safe | safe | degraded | unsafe |
| BN mean [AI (GEVD)] | safe | safe | unsafe | unsafe | safe | safe | degraded | unsafe |
| BN mean [arithmetic] | safe | safe | safe | safe | safe | safe | safe | safe |
| BN mean [BW] | safe | safe | degraded | unsafe | safe | safe | degraded | degraded |
| BN mean [GAH] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [harmonic] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| BN mean [LE] | safe | safe | degraded | unsafe | safe | safe | degraded | unsafe |
| ResidualBlock [AI] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [LE] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [LE, log output] | safe | degraded | unsafe | unsafe | safe | safe | degraded | unsafe |
| ResidualBlock [congruence] | safe | safe | degraded | degraded | safe | safe | degraded | degraded |
| ResidualBlock [branch_pt] | safe | unsafe | unsafe | unsafe | safe | unsafe | unsafe | unsafe |
| ResidualBlock [eigenbasis] | degraded | degraded | degraded | degraded | degraded | degraded | degraded | degraded |
| GBWBNRResNet | safe | safe | safe | unsafe | safe | safe | unsafe | unsafe |
| RResNet | safe | safe | degraded | degraded | safe | safe | safe | degraded |
| SPDnet | safe | safe | safe | degraded | safe | safe | safe | degraded |

The Bures–Wasserstein rows were measured after the SVD cross term
([section 3](#3-means-as-iterations)), which removed the failures of the
cross term itself: the BW operations (mean, geodesic, centring, std) no
longer fail in float32 up to $\kappa = 10^{6}$ (the geodesic through the
eigendecomposition of $B^{1/2}XB^{1/2}$ failed at $10^{6}$ and was 50 times
less accurate at $10^{4}$), and their forward errors are equal or smaller
everywhere.
The scalar-variance normalization (GBWBN) stays unsafe in float32 and mixed
precision from $\kappa = 10^{4}$ at $n = 64$ (12–62 % of the batches): the
eigenvalues of these batches spread over $[10^{-2}, 10^{2}]$, the centring
folds (see {doc}`batchnorm`) and returns matrices that are singular to
rounding, whose square root in the next step has no float32 digits left.
Use float64 for it, or keep the spectrum close to 1 ($\theta < 1$,
`bw_init_metric="trace_scale"` for the common scale). `GBWBNRResNet` (a BiMap
to $n/2$ first) is safe in float32 up to $\kappa = 10^{6}$; in mixed precision
it still fails on 1 batch in 8 at $\kappa = 10^{6}$ and $n = 64$ (the rows
of the models were measured on the code with both the SVD cross term and the
Cholesky-free residual blocks).

The residual blocks no longer fail in float32 at $\kappa = 10^{8}$ (failure
rate 0.12 before: the Cholesky factorization of the norm and the
decomposition of the re-formed floored matrix). Their log output
(`le_output="log"`) is compared in log coordinates, where the small
eigenvalues weigh as much as the large ones: it carries the same information
as the log-Euclidean block followed by LogEig.
