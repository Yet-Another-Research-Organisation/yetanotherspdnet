# Models

`SPDnet`, `RResNet` and `GBWBNRResNet` take a batch of SPD matrices
`(B, n, n)` and return logits `(B, n_classes)`. They take 30 to 39 constructor
arguments because they expose every option of the layers they contain; most
can be left at their default. API: {doc}`../reference/models`.

## Mathematics

`SPDnet`
: The SPDNet of {ref}`[Huang & Van Gool 2017] <ref-huang2017>`,
  $[\mathrm{BiMap} \to \mathrm{ReEig} \to (\mathrm{BatchNorm})]^{L} \to
  \mathrm{LogEig} \to \mathrm{Vec} \to \mathrm{Linear}$, with
  $L$ = `len(hidden_layers_size)`; the batch normalization of
  {ref}`[Brooks et al. 2019] <ref-brooks2019>` or its variants. The default
  choice.

`RResNet`
: A Riemannian residual network {ref}`[Katsman et al. 2023] <ref-katsman2023>`
  with several stages,
  $[\mathrm{BiMap} \to (\mathrm{ReEig}) \to (\mathrm{BatchNorm}) \to
  \mathrm{ResidualBlock}^{k}]^{L} \to \mathrm{LogEig} \to \dots$: the residual
  blocks move each matrix along a learned tangent vector ({doc}`residual`). No
  ReEig by default.

`GBWBNRResNet`
: The single-stage residual network of the GBWBN paper
  {ref}`[Wang et al. 2025] <ref-wang2025>`,
  $\mathrm{BiMap} \to \mathrm{GBWBN} \to \mathrm{ResidualBlock} \to
  \mathrm{LogEig} \to \dots$. Kept to reproduce that paper; `RResNet` covers
  the same architecture with more options.

## Implementation

Every model builds a `torch.nn.Sequential` of the layers of {doc}`layers`,
{doc}`batchnorm` and {doc}`residual`, and

- forwards `use_autograd` (a bool, or a dict per layer type: `"bimap"`,
  `"reeig"`, `"logeig"`, `"batchnorm"`, `"vec"`, and `"residual"` for the
  residual models), `dtype`, `device`, `eigh_dtype` and `generator` to them;
- forwards `register_optimizer_hook(optimizer)` to every layer with a dynamic
  parametrization;
- has a stable `repr` and a hash of its configuration,
  `create_model_name_hash()` (CRC32 of the `repr`, useful to name runs).

The defaults are float64 and manual backwards; the measured float32 behaviour
of the three models is in {doc}`../numerics/precision` (`SPDnet` is the most
robust).

## Options and strategy

### Arguments by role

Shared by the three models unless noted.

| role | arguments | what to know |
|---|---|---|
| **Sizes** | `input_dim`, `hidden_layers_size` (`hidden_dim` for `GBWBNRResNet`), `output_dim`, `n_residual_blocks` (RResNet) | `hidden_layers_size=[84, 63]` means two BiMaps $n \to 84 \to 63$. |
| **Non-linearity** | `reeig_eps`, `reeig` (RResNet) | Eigenvalues below `reeig_eps` are clamped: set it to the scale of the data ({doc}`layers`). |
| **BiMap weights** | `bimap_parametrized`, `bimap_parametrization_mode`, `bimap_parametrization_options`, `bimap_n_steps_ref_update` | Stiefel constraint, static or dynamic ({doc}`parametrizations`). |
| **Batch normalization: which** | `batchnorm`, `batchnorm_type`, `batchnorm_mean_type`, `batchnorm_mean_options` | `batchnorm_type="mean_only"` centres; `"mean_var_scalar"` also rescales (and is GBWBN with `"bures_wasserstein"`). Default mean: GAH for `SPDnet`, affine-invariant for `RResNet`, Bures–Wasserstein for `GBWBNRResNet` ({doc}`batchnorm`). |
| **Batch normalization: statistics** | `batchnorm_momentum`, `batchnorm_norm_strategy`, `batchnorm_minibatch_mode`, `batchnorm_minibatch_momentum`, `batchnorm_minibatch_maxstep` | Running statistics and smoothing of the batch mean. |
| **Batch normalization: bias** | `batchnorm_parametrization`, `batchnorm_parametrization_mode`, `batchnorm_n_steps_ref_update`, `batchnorm_norm_and_bias_mode` (SPDnet) | How the SPD bias stays SPD, and how the batch is moved to it. |
| **GBWBN** | `batchnorm_bw_options` | `{"bw_theta": 0.5, "bw_batch_stats_grad": True, "bw_init_metric": "identity"}` by default ({doc}`batchnorm`). |
| **Residual blocks** (RResNet, GBWBNRResNet) | `spectrum_type`, `spectrum_hidden_dim`, `spectrum_n_layers`, `spectrum_kernel_size`, `stiefel_parametrization_mode`, `stiefel_n_steps_ref_update`, `residual_metric`, `residual_options` | The spectral field and the exponential map; `residual_options` passes the other options of `ResidualBlock` (step length, eigenbasis field, branch with transport, log output) ({doc}`residual`). |
| **Head** | `use_logeig`, `vec_type`, `softmax` | `vec_type="vech"` halves the size of the linear layer. |
| **Numerics** | `use_autograd`, `dtype`, `device`, `eigh_dtype`, `generator` | {doc}`../numerics/index`. |

```{tip}
If any part uses a dynamic parametrization (`*_parametrization_mode="dynamic"`),
register the optimizer once after building it:
`model.register_optimizer_hook(optimizer)`.
```

### Which model

- Start with `SPDnet` (with `batchnorm=True` when the batches are large
  enough, 16 or more); it is the most robust in float32 and the cheapest.
- Try `RResNet` when depth helps (several BiMap stages with one or two blocks
  each); keep the residual defaults for comparability, or `step="clip"` /
  `"learned_scale"` for gentler blocks.
- Use `GBWBNRResNet` to reproduce the GBWBN paper; it needs float64 on
  ill-conditioned data and benefits from `bw_init_metric="trace_scale"` when
  the eigenvalues are far from 1.

### Configurations of published experiments

These configurations reproduce published results with this library (the
training loop is in the `spdnet-benchmark-demo` repository).

SPDNet batch normalization experiments, HyperLeaf (204 × 204 hyperspectral
covariances, 4 cultivars); SGD with Nesterov momentum, 5 warm-up epochs,
learning rate halved on plateau, early stopping:

```python
from yetanotherspdnet import GBWBNRResNet, SPDnet

SPDnet(
    input_dim=204,
    hidden_layers_size=[184, 158],
    output_dim=4,
    reeig_eps=0.01,
    batchnorm=True,
    batchnorm_mean_type="arithmetic",  # batch 48, lr 0.05
)
```

Same experiments, HDM05 (93 × 93 skeleton covariances scaled by 190, 117
classes):

```python
SPDnet(
    input_dim=93,
    hidden_layers_size=[84, 63],
    output_dim=117,
    reeig_eps=0.8,
    batchnorm=True,
    batchnorm_mean_type="geometric_arithmetic_harmonic",  # batch 16, lr 0.25
)
```

GBWBN paper, HDM05 (raw matrices; Adam with AMSGrad, lr 2.5e-3, batch 30, 200
epochs):

```python
GBWBNRResNet(  # GBWBN (momentum 0.1) is on by default
    input_dim=93,
    hidden_dim=30,
    output_dim=117,
    batchnorm_bw_options={"bw_theta": 0.5},
)
```
