# Yet Another SPDNet

**Yet Another SPDNet** is a PyTorch library for deep learning on
**Symmetric Positive Definite (SPD) matrices**. It provides SPD matrix
functions with stable hand-written gradients, several Riemannian geometries on
the SPD manifold, the SPDNet layers, Riemannian batch normalization, and
complete models (SPDNet, Riemannian residual networks).

## Why SPD matrices?

Covariance matrices are SPD: symmetric, with strictly positive eigenvalues.
They are the natural descriptor of many signals: EEG/MEG channels, radar and
SAR pixels, hyperspectral patches, skeleton joints over time, CNN feature maps.
SPD matrices do not form a vector space (a difference of two covariances is
generally not a covariance), but a curved **Riemannian manifold**.
Flattening them into vectors and feeding a standard network ignores that
geometry.

SPDNet (Huang & Van Gool, AAAI 2017) instead keeps the data on the manifold
through the network:

$$
X \;\xrightarrow{\text{BiMap}}\; W^\top X W
\;\xrightarrow{\text{ReEig}}\; U \max(\Lambda, \epsilon) U^\top
\;\xrightarrow{\ \cdots\ }\;
\xrightarrow{\text{LogEig}}\; \log X
\;\xrightarrow{\text{Vec + Linear}}\; \text{logits}
$$

BiMap reduces the dimension with an orthonormal $W$, ReEig is the SPD
analogue of ReLU, and LogEig maps to the flat space of symmetric matrices
right before a Euclidean classifier.

## What the library provides

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} SPD linear algebra
`functions.spd_linalg`: matrix log/exp/sqrt/powers, congruences, whitening,
vectorization. Each operation has an autograd path and a manual
(Daleckii–Krein) backward.
:::

:::{grid-item-card} Riemannian geometries
`functions.spd_geometries`: geodesics, means, dispersions and exp/log maps for
the affine-invariant, log-Euclidean, Kullback–Leibler, symmetrized KL (GAH) and
Bures–Wasserstein geometries.
:::

:::{grid-item-card} Layers
`nn`: BiMap, ReEig, LogEig, Vec/Vech, SPD batch normalization in any of
the geometries above, spectral residual blocks. Orthogonality and positivity
are enforced by parametrizations.
:::

:::{grid-item-card} Models
`SPDnet`, `RResNet` and `GBWBNRResNet`, ready to train, with every option
(geometry, batch normalization, gradient path) available as a constructor
argument.
:::
::::

## How the package is organised

```text
yetanotherspdnet/
├── functions/                 pure tensor functions (no parameters)
│   ├── spd_linalg.py          eigen-based matrix functions, congruences, vec/vech
│   ├── m_estimators.py        sample covariance, robust M-estimators (Tyler, Student-t)
│   ├── scalar_functions.py    scalar maps applied to eigenvalues
│   ├── stiefel.py             projections/retractions on the Stiefel manifold
│   └── spd_geometries/        one module per geometry
│       ├── affine_invariant.py
│       ├── log_euclidean.py
│       ├── kullback_leibler.py
│       ├── kullback_leibler_symmetrized.py
│       └── bures_wasserstein.py
├── nn/                        torch.nn.Module layers built on functions/
│   ├── base.py                BiMap, ReEig, ReEigBias, LogEig, Vec, Vech
│   ├── estimation.py          SampleCovariance, MEstimation (samples -> SPD)
│   ├── batchnorm.py           Riemannian batch normalization
│   ├── rresnet_layers.py      spectral vector field, residual block
│   └── parametrizations.py    SPD / Stiefel / positive-scalar parametrizations
├── model.py                   SPDnet, RResNet, GBWBNRResNet
└── random/                    random SPD and Stiefel generators
```

Dependencies only go downwards: `model` uses `nn`, which uses `functions`.

## Where to go next

- New to the library? Start with {doc}`installation`, then the {doc}`quickstart`.
- To understand the design choices (geometries, batch normalization, gradient
  paths, dtype), read the {doc}`user_guide/index`.
- Looking for a function or a class? See the {doc}`reference`.

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

reference
```

```{toctree}
:hidden:
:caption: Development

contributing
```

## Authors and license

Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet. Released
under the MIT License.
