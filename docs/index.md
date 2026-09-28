# Yet Another SPDNet

**Deep learning on Symmetric Positive Definite matrices, in PyTorch.**
Covariance matrices are SPD: they live on a curved manifold, not in a vector
space. This library keeps them there through the network, with SPD matrix
functions whose gradients stay exact on ill-conditioned data, six Riemannian
geometries, Riemannian batch normalization, residual networks and robust
covariance estimators.

```{figure} _static/diagrams/spdnet_pipeline.svg
:width: 100%

An SPDNet on HDM05 skeleton covariances: the matrices stay SPD through BiMap,
ReEig and the batch normalization, and only the last layers map them to a
vector space for a Euclidean classifier.
```

## Why SPD matrices?

Covariance matrices are the natural descriptor of EEG/MEG channels, radar and
SAR pixels, hyperspectral patches, skeleton joints over time and CNN feature
maps. A difference of two covariances is generally not a covariance, and
flattening them into vectors ignores the geometry that makes them comparable.
SPDNet (Huang & Van Gool, AAAI 2017) instead transforms them with layers that
map SPD matrices to SPD matrices, and only takes a matrix logarithm right
before the classifier.

## Start here

1. {doc}`installation`: one dependency, PyTorch.
2. {doc}`quickstart`: matrix functions, means, layers, a model trained in ten
   lines.
3. {doc}`user_guide/concepts`: the three levels of the package, the shapes
   flowing through a network, the conventions (float64, two gradient paths,
   parametrizations).
4. The guide of the part you need, below.

## Find your way

| I want to… | Read |
|---|---|
| train a classifier on covariance matrices | {doc}`user_guide/models`, {doc}`reference/models` |
| understand BiMap, ReEig, LogEig and their parametrizations | {doc}`user_guide/layers` |
| choose a geometry (affine-invariant, log-Euclidean, GAH, Bures–Wasserstein, …) | {doc}`user_guide/geometries` |
| add a batch normalization, or reproduce GBWBN | {doc}`user_guide/batchnorm` |
| use residual blocks (RResNet) | {doc}`user_guide/residual` |
| estimate robust covariances inside a network (M-estimators) | {doc}`reference/m_estimators`, {doc}`reference/layers` |
| understand why training is stable (Daleckii–Krein, implicit differentiation, …) | {doc}`user_guide/numerics` |
| look up a function or a class | {doc}`reference/index` |

## What makes it different

**Exact gradients where autograd fails.** Every eigendecomposition-based
operation has a hand-written backward (Daleckii–Krein), finite and exact when
eigenvalues are equal, as after a ReEig or at an identity initialization.
The autograd version is kept alongside, selectable per layer
({doc}`user_guide/numerics`).

**Six geometries, one interface.** Affine-invariant, log-Euclidean,
arithmetic and harmonic (Kullback–Leibler), GAH and its learned variant, and
Bures–Wasserstein, each with its mean, dispersion, geodesic and, where
needed, exponential and logarithmic maps ({doc}`user_guide/geometries`).

**Riemannian batch normalization** in any of these geometries, with running
statistics that follow the geometry, and GBWBN implemented as in its paper
({doc}`user_guide/batchnorm`).

**Residual networks** whose step stays finite on real, ill-conditioned
covariances ({doc}`user_guide/residual`).

**Robust covariance pooling**: Tyler and Student-t M-estimators
differentiated at their fixed point, as network layers
({doc}`reference/m_estimators`).

## The package

```{figure} _static/diagrams/package_levels.svg
:width: 90%

Dependencies only go downwards: `model` uses `nn`, which uses `functions`.
The neighbouring repositories provide the data and the training loop.
```

| Level | Modules | Reference |
|---|---|---|
| `model` | `SPDnet`, `RResNet`, `GBWBNRResNet` | {doc}`reference/models` |
| `nn` | `BiMap`, `ReEig`, `ReEigBias`, `LogEig`, `Vec`, `Vech`, `SampleCovariance`, `MEstimation` | {doc}`reference/layers` |
| | `BatchNormSPDMean`, `BatchNormSPDMeanScalarVariance` | {doc}`reference/batchnorm` |
| | `ResidualBlock`, `SpectralVectorField` | {doc}`reference/residual` |
| | SPD, Stiefel and scalar parametrizations | {doc}`reference/parametrizations` |
| `functions` | `spd_linalg`: matrix functions, congruences, vectorizations | {doc}`reference/spd_linalg` |
| | `spd_geometries.*`: the six geometries | {doc}`reference/geometries` |
| | `m_estimators`: sample covariance, Tyler, Student-t | {doc}`reference/m_estimators` |
| | `scalar_functions`, `stiefel`, `random` | {doc}`reference/utilities` |

## Related repositories

- [spdnet-datasets](https://github.com/Yet-Another-Research-Organisation/spdnet-datasets):
  loaders for HyperLeaf, HDM05, FUSAR-Ship, radar and hyperspectral datasets,
  synthetic SPD data.
- [spdnet-training](https://github.com/Yet-Another-Research-Organisation/spdnet-training):
  PyTorch Lightning module, Hydra command line, Optuna search, CNN backbones
  with covariance pooling.
- [spdnet-benchmark-demo](https://github.com/Yet-Another-Research-Organisation/spdnet-benchmark-demo):
  a rerunnable benchmark of the models and batch normalizations.

## Authors and license

Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet. Released
under the MIT License.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

user_guide/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Development

contributing
```
