# API reference

One page per part of the library, in dependency order: models are built from
layers, layers from functions. Each page starts with what the objects are
for and a summary table; the detailed entries follow. For the reasoning
behind them (geometries, gradient paths, parametrizations), see the
{doc}`../user_guide/index` and {doc}`../numerics/index`.

| Page | Contents |
|---|---|
| {doc}`models` | `SPDnet`, `RResNet`, `GBWBNRResNet`: complete classifiers |
| {doc}`layers` | `BiMap`, `ReEig`, `ReEigBias`, `LogEig`, `Vec`, `Vech`; covariance estimation layers |
| {doc}`batchnorm` | Riemannian batch normalization (mean only, mean + scalar variance, GBWBN) |
| {doc}`residual` | Spectral vector field and residual block of the residual networks |
| {doc}`parametrizations` | SPD, Stiefel and positive-scalar parametrizations of the parameters |
| {doc}`spd_linalg` | Matrix functions through eigendecomposition, congruences, vectorizations |
| {doc}`geometries` | Geodesics, means, dispersions, exp/log maps and transports of the geometries, and their registry |
| {doc}`m_estimators` | Sample covariance and robust M-estimators of scatter |
| {doc}`utilities` | Gradient-path selection, scalar maps, Stiefel projections, random SPD/Stiefel matrices |
| {doc}`deprecated` | Old names and their replacements |

```{toctree}
:hidden:

models
layers
batchnorm
residual
parametrizations
spd_linalg
geometries
m_estimators
utilities
deprecated
```
