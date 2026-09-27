# Models and their options

`SPDnet`, `RResNet` and `GBWBNRResNet` take 28 to 37 constructor arguments,
because they expose every option of the layers they contain. Most can be
left at their default. This page groups them by role; the
{doc}`../reference/models` page lists them one by one.

## Which model

`SPDnet`
: The SPDNet of Huang & Van Gool (AAAI 2017):
  `[BiMap → ReEig → (BatchNorm)] × L → LogEig → Vec → Linear`, with
  `L = len(hidden_layers_size)`. The default choice.

`RResNet`
: A Riemannian residual network (Katsman et al., NeurIPS 2023) with several
  stages: `[BiMap → (ReEig) → (BatchNorm) → ResidualBlock × k] × L`. The
  residual blocks move each matrix along a learned tangent vector
  ({doc}`residual`). No ReEig by default.

`GBWBNRResNet`
: The single-stage residual network of the GBWBN paper (Wang et al., 2025):
  `BiMap → GBWBN → ResidualBlock → LogEig`. Kept to reproduce that paper;
  `RResNet` covers the same architecture with more options.

## Arguments by role

Shared by the three models unless noted.

| Role | Arguments | What to know |
|---|---|---|
| **Sizes** | `input_dim`, `hidden_layers_size` (`hidden_dim` for `GBWBNRResNet`), `output_dim`, `n_residual_blocks` (RResNet) | `hidden_layers_size=[84, 63]` means two BiMaps $n \to 84 \to 63$. |
| **Non-linearity** | `reeig_eps`, `reeig` (RResNet) | Eigenvalues below `reeig_eps` are clamped: set it to the scale of the data ({doc}`layers`). |
| **BiMap weights** | `bimap_parametrized`, `bimap_parametrization_mode`, `bimap_parametrization_options`, `bimap_n_steps_ref_update` | Stiefel constraint, static or dynamic ({doc}`layers`). |
| **Batch normalization: which** | `batchnorm`, `batchnorm_type`, `batchnorm_mean_type`, `batchnorm_mean_options` | `batchnorm_type="mean_only"` centres; `"mean_var_scalar"` also rescales (and is GBWBN with `"bures_wasserstein"`). Default mean: GAH for `SPDnet`, affine-invariant for `RResNet`, Bures–Wasserstein for `GBWBNRResNet` ({doc}`batchnorm`). |
| **Batch normalization: statistics** | `batchnorm_momentum`, `batchnorm_norm_strategy`, `batchnorm_minibatch_mode`, `batchnorm_minibatch_momentum`, `batchnorm_minibatch_maxstep` | Running statistics and smoothing of the batch mean. |
| **Batch normalization: bias** | `batchnorm_parametrization`, `batchnorm_parametrization_mode`, `batchnorm_n_steps_ref_update` | How the SPD bias stays SPD. |
| **GBWBN** | `batchnorm_bw_options` | `{"bw_theta": 0.5, "bw_batch_stats_grad": True}` by default ({doc}`batchnorm`). |
| **Residual blocks** (RResNet, GBWBNRResNet) | `spectrum_type`, `spectrum_hidden_dim`, `spectrum_n_layers`, `spectrum_kernel_size`, `stiefel_parametrization_mode`, `stiefel_n_steps_ref_update`, `residual_metric` | The spectral vector field and the exponential map ({doc}`residual`). |
| **Head** | `use_logeig`, `vec_type`, `softmax` | `vec_type="vech"` halves the size of the linear layer. |
| **Numerics** | `use_autograd`, `dtype`, `device`, `generator` | `use_autograd` takes a bool or a dict per layer type (`"bimap"`, `"reeig"`, `"logeig"`, `"batchnorm"`, `"vec"`, `"residual"`) ({doc}`numerics`). |

## After building the optimizer

If any part uses a dynamic parametrization (`*_parametrization_mode="dynamic"`),
register the optimizer once:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)
model.register_optimizer_hook(optimizer)
```

## Configurations of the published experiments

These configurations reproduce published results with this library (see the
`spdnet-benchmark-demo` repository for the training loop).

SPDNet batch normalization paper, HyperLeaf (204×204 hyperspectral
covariances, 4 cultivars); SGD with Nesterov momentum, 5 warm-up epochs,
learning rate halved on plateau, early stopping:

```python
SPDnet(
    input_dim=204, hidden_layers_size=[184, 158], output_dim=4,
    reeig_eps=0.01,
    batchnorm=True, batchnorm_mean_type="arithmetic",   # batch 48, lr 0.05
)
```

Same paper, HDM05 (93×93 skeleton covariances scaled by 190, 117 classes):

```python
SPDnet(
    input_dim=93, hidden_layers_size=[84, 63], output_dim=117,
    reeig_eps=0.8,
    batchnorm=True, batchnorm_mean_type="geometric_arithmetic_harmonic",  # batch 16, lr 0.25
)
```

GBWBN paper, HDM05 (raw matrices; Adam with AMSGrad, lr 2.5e-3, batch 30,
200 epochs):

```python
GBWBNRResNet(   # GBWBN (momentum 0.1) is on by default
    input_dim=93, hidden_dim=30, output_dim=117,
    batchnorm_bw_options={"bw_theta": 0.5},
)
```
