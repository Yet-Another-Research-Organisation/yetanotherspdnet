# Batch normalization

Euclidean batch normalization subtracts the batch mean, divides by the batch
standard deviation, then applies a learned scale and shift. The SPD layers in
`yetanotherspdnet.nn.batchnorm` do the same on the manifold, in the geometry
chosen with `mean_type` (see {doc}`geometries`).

```{figure} ../_static/diagrams/batchnorm_steps.svg
:width: 100%

2×2 SPD matrices drawn as ellipses: the batch is centred (its mean becomes
$I$), rescaled along geodesics from $I$, then moved to the learned bias $G$.
```

## The two layers

`BatchNormSPDMean`
: **Centring and bias.** Each matrix is transported so that the batch mean
  $\bar X$ moves to the identity, then moved to a learned SPD bias
  (`Covbias`, $G$). With the default `norm_and_bias_mode="whitening"` this is
  $X \mapsto G\,\bar X^{-1/2} X \bar X^{-1/2}\,G$, so the batch mean lands
  on $G^2$; with `norm_and_bias_mode="parallel_transport"` the batch is
  parallel-transported from $\bar X$ to $G$ and the mean lands on $G$ (see
  [Centring and bias](#centring-and-bias)).
  With `mean_type="adaptive_geometric_arithmetic_harmonic"`, the batch mean
  is the point $t$ of the affine-invariant geodesic from the harmonic to the
  arithmetic mean, with $t = \operatorname{sigmoid}(\cdot)$ learned per
  layer (attribute `t_gah`, $t = 1/2$ gives the GAH mean).

`BatchNormSPDMeanScalarVariance`
: **Centring, scalar dispersion and bias.** Between centring and bias, the
  centred matrices are rescaled along geodesics from the identity by
  $s / \sigma$: $\sigma$ is the batch dispersion (a scalar, from
  `<geometry>_std_scalar`) and $s$ a learned positive scalar (`stdScalarbias`).
  With `mean_type="bures_wasserstein"` this is **GBWBN** (Wang et al., 2025),
  which additionally learns a pre-transform $X \mapsto M^{-1/2} X^\theta M^{-1/2}$
  and its inverse, and learns its bias $G$ on the input manifold, used as
  $\hat G = M^{-1/2} G^\theta M^{-1/2}$ in the transformed space.

GBWBN options (`bw_theta`, `bw_batch_stats_grad`, `bw_init_metric`):

- `bw_theta` (default 0.5): power deformation $X \mapsto X^\theta$. The
  variance is that of the deformed metric, $d_{BW}^2(X^\theta, Y^\theta) /
  \theta^2$ (the std is divided by $|\theta|$). It must be finite and nonzero
  (`ValueError` otherwise; $X^0 = I$ erases the data). The best value depends
  on the dataset: 0.5 on HDM05 and 0.75 on NTU in the paper's ablation (its
  Table 10); 0.5 is also the reference code's default.
- `bw_batch_stats_grad` (default `True`): whether gradients flow through the
  batch mean and variance. `True` is a choice of this package, not a
  prescription of the paper; `False` treats them as constants, as the
  reference implementation does.
- `bw_init_metric` (default `"identity"`): `"trace_scale"` sets, on the first
  training batch, $M = cI$ and $G = c^{1/\theta} I$ with
  $c = \operatorname{tr}(\bar B)/n$, $\bar B$ the BW mean of the $X_i^\theta$
  (see [Folding](#folding-and-data-scale)).

In the models, pass them as `batchnorm_bw_options={"bw_theta": 0.25}`.

`parametrization` and `parametrization_mode` apply to $M$ and $G$ as to the
bias of the other layers: with `"dynamic"`, the reference points of both move
to their current values every `n_steps_ref_update` optimizer steps (call
`layer.register_optimizer_hook(optimizer)`), which keeps the softplus / exp
chart centred where the parameters are.

```{note}
The GBWBN parameters $M$ and $G$ start at the identity, where all
eigenvalues are equal. Gradients through `torch.linalg.eigh` are undefined
there, so keep `use_autograd=False` (the default) for this layer: the manual
backwards (Daleckii–Krein for $M^{\pm 1/2}$ and $M^\theta$, the polar factor
of the cross terms, implicit differentiation for the Lyapunov solve and the
transport to $\hat G$) stay finite and exact at the identity.
```

### Folding and data scale

The BW exponential at the identity, $\mathrm{Exp}_I(V) = (I + V/2)^2$, is only
injective on $I + V/2 \succ 0$; outside it the matrix is folded back (two
tangent vectors give the same point, see {doc}`numerics`), and a folded
matrix close to singular loses its small eigenvalues to rounding (the cause of
the remaining float32 failures of this layer). The BW metric is not scale
invariant: for data of scale $c$, the centring step transports a tangent
vector of size $\sim c$ from a mean of size $\sim c$ to $I$, which gives a
vector of size $\sim\sqrt c$. **Folding is driven by how far the eigenvalues
are from 1, not by the condition number as such**: matrices with
$\kappa = 10^2$ and eigenvalues in $[10^{-2}, 1]$ never fold, the same matrices
times $10^2$ almost always do. Measured on random batches (20 batches of 32),
fraction of folded matrices:

| data ($\kappa$, eigenvalues) | $n$ | $\theta$ | `"identity"` | `"trace_scale"` |
|---|---|---|---|---|
| $10^2$, $[10^{-2}, 1]$ (and $\times 10^{-2}$) | 8 | 1 and 0.5 | 0 | 0 |
| $10^2$, $[1, 10^{2}]$ | 8 | 1 | 99.8 % | 0 |
| $10^2$, $[1, 10^{2}]$ | 8 | 0.5 | 1.1 % | 0 |
| $10^2$, $[10^{2}, 10^{4}]$ | 8 | 0.5 | 98.3 % | 0 |
| $10^2$, $[10^{-1}, 10]$ | 64 | 1 | 22 % | 0 |
| $10^4$, $[10^{-2}, 10^{2}]$ | 64 | 0.5 | 22 % | 0 |
| $10^4$, $[10^{-2}, 10^{2}]$ | 64 | 1 | 100 % | 100 % |
| $10^6$, $[10^{-3}, 10^{3}]$ | 64 | 0.5 | 100 % | 57 % |

`bw_init_metric="trace_scale"` removes the common scale: the transformed data
$X_i^\theta / c$ have unit scale, and $\hat G = G^\theta / c = I$
(initializing $M$ alone would move the folding to the bias step: 97 % at
scale $10^2$, $\theta = 1$). The spread of the eigenvalues on both sides of 1
remains, and $\theta < 1$ is what compresses it ($X^\theta$ maps $[10^{-2},
10^{2}]$ to $[10^{-1}, 10]$ at $\theta = 1/2$). With `"trace_scale"`, the
layer is scale-equivariant at initialization ($f(sX) = s f(X)$) and its
outputs keep the scale of the data instead of being centred at $I$.

### Running statistics

The running mean is stored **in the input space**,
$B_r = (M^{1/2} \hat B_r M^{1/2})^{1/\theta}$, and mapped with the current $M$
when it is used (each running update, and evaluation). It is thus not left in
the coordinates of an old $M$ when $M$ is trained. This is exact when $M$
changes by a factor ($M \to cM$: the BW mean is 1-homogeneous); for a general
change of $M$ it is the natural convention but not an invariance (the BW mean
does not commute with congruences). The running standard deviation is a
distance of the deformed metric of each update and is kept as is. With
`norm_strategy="minibatch"`, the smoothed mean of the previous batch stays in
the transformed space (it changes every step anyway).

Checkpoints saved before this change (state dict version 1) are converted
when loaded: the running mean is mapped to the input space with the
checkpoint's own $M$, and the missing reference points of dynamic layers are
set to $I$ (the chart those versions used), so old checkpoints load with
`strict=True` and evaluate as before. Keep the `_metadata` of the state dict
(`torch.save(model.state_dict())` does) so that a new checkpoint is not
converted again.

### Package, paper and reference code

| | paper (Algorithm 1) | reference code (jjscc/GBWBN) | this package |
|---|---|---|---|
| variance | to the batch mean | to the identity (`cal_var`) | to the batch mean |
| $M$, $G$ | Riemannian SGD | Riemannian SGD, mixing affine-invariant and BW updates | softplus / exp parametrization, any PyTorch optimizer, static or dynamic chart |
| gradient through the batch statistics | not specified | none | yes (`bw_batch_stats_grad=True`), optional |
| $\theta$ | hyper-parameter (ablation in Table 10) | stored in a class attribute shared by all layers (`Power_op._power`), which corrupts its gradient | per layer, validated |
| running-mean momentum | — | 0.1 | 0.01 for the layer; 0.1 in `GBWBNRResNet` |
| running mean | — | transformed space | input space |
| cross term $(B^{1/2}XB^{1/2})^{1/2}$ | — | eigendecomposition | SVD (polar factor) |

The conformity test of the package compares the layer with the paper's
formulas (`TestGBWBNConformity`), not with the reference code.

In the models (`SPDnet`, `RResNet`, `GBWBNRResNet`) these correspond to
`batchnorm_type="mean_only"` and `batchnorm_type="mean_var_scalar"`, and every
layer option is exposed with a `batchnorm_` prefix.

## Training and evaluation

| Mode | Statistics used | Running statistics |
|---|---|---|
| `model.train()`, batch of ≥ 2 matrices | batch mean (and dispersion) | updated with `momentum` |
| `model.train()`, a single matrix | running mean (and dispersion) | **not** updated, a `UserWarning` is emitted |
| `model.eval()` | running mean (and dispersion) | not updated |

A batch of one matrix has no batch statistics: its mean is the matrix itself
and its dispersion is zero, so normalizing with them would map every input to
the identity. Such batches (for instance a last incomplete batch of size 1)
are therefore normalized like in evaluation mode. Use `drop_last=True` in
your `DataLoader` to avoid them entirely.

## Centring and bias

`norm_and_bias_mode` chooses how `BatchNormSPDMean` moves the batch from its
mean $\bar X$ to the bias:

`"whitening"` (default)
: Whitening by the mean, then congruence by `Covbias`:
  $X \mapsto G\,\bar X^{-1/2} X \bar X^{-1/2}\,G$. The mean is sent to $G^2$.
  Only $\bar X^{-1/2}$ needs an eigendecomposition.

`"parallel_transport"`
: Parallel transport from $\bar X$ to $G$ in the geometry of the mean: the
  mean is sent to $G$ itself, and distances to the mean are preserved in that
  geometry. Affine-invariant: $X \mapsto E X E^\top$ with
  $E = G^{1/2}(G^{-1/2}\bar X G^{-1/2})^{-1/2}G^{-1/2}$; log-Euclidean:
  $X \mapsto \exp(\log X - \log \bar X + \log G)$. The arithmetic,
  harmonic and (adaptive) GAH means use the affine-invariant transport,
  computed through a generalized eigendecomposition
  (`spd_geometries.affine_invariant_gevd`). Not available for
  `"bures_wasserstein"`, nor for `BatchNormSPDMeanScalarVariance`.
  With `use_autograd=True` the affine-invariant and log-Euclidean transports
  raise a `ValueError`: they take square roots or logarithms of the bias,
  which starts at the identity, where autograd through `torch.linalg.eigh`
  gives NaN gradients. The default hand-written backward has no such issue.

The two modes parametrize the bias differently ($G^2$ versus $G$), so a
trained `Covbias` is not interchangeable between them.

## Main options

`momentum`
: Update rate of the running statistics:
  `running = geodesic(running, batch, momentum)`.

`norm_strategy`
: `"classical"` normalizes each batch with its own mean.
  `"minibatch"` normalizes with a smoothed mean
  `m_k = geodesic(m_{k-1}, batch_mean, minibatch_momentum_k)`, where $m_{k-1}$
  is the mean used for the previous batch. This reduces the noise of small
  batches. The step `minibatch_momentum_k` follows `minibatch_mode`
  (`"constant"`, `"decay"` towards `minibatch_momentum`, or `"growth"` from it)
  over `minibatch_maxstep` steps.

`parametrization`, `parametrization_mode`
: How the SPD bias stays SPD: `"softplus"` or `"exp"` maps on the
  eigenvalues. The `"dynamic"` mode re-centres the parametrization on the
  current value every `n_steps_ref_update` optimizer steps. It requires
  `layer.register_optimizer_hook(optimizer)` after creating the optimizer.

`mean_options`
: Extra arguments of the mean function, for instance
  `{"n_iterations": 5}` for the affine-invariant and Bures–Wasserstein means.
