# API Reference

This page provides detailed documentation for all modules, classes, and functions in Yet Another SPDNet.

The API documentation is automatically generated from the source code docstrings using sphinx-autoapi.

## Package Structure

```
yetanotherspdnet/
├── spd.py          # SPD matrix operations
├── nn.py           # Neural network layers
├── means.py        # SPD mean computations
├── stiefel.py      # Stiefel manifold utilities
└── models.py       # High-level model classes
```

## Core Modules

### spd - SPD Matrix Operations

The `spd` module provides fundamental operations on Symmetric Positive Definite matrices:

- Matrix logarithm and exponential
- Square root and inverse square root
- Whitening and congruence transformations
- Eigenvalue operations
- Vectorization utilities

### nn - Neural Network Layers

The `nn` module contains PyTorch layers designed for SPD matrices:

- **BiMap**: Bilinear mapping layer
- **ReEig**: Eigenvalue rectification layer
- **LogEig**: Matrix logarithm layer
- **Vec**: Vectorization layer
- **BatchNormSPDMean**: Batch normalization using SPD means

### means - SPD Mean Computation

The `means` module implements various mean computation methods:

- Arithmetic mean (Euclidean)
- Geometric mean (Riemannian/Karcher mean)
- Harmonic mean
- Log-Euclidean mean
- Power mean
- Composite methods

### models - High-Level Models

The `models` module provides complete model architectures:

- **SPDNet**: Complete SPDNet architecture with configurable layers

## Detailed API Documentation

The complete API documentation with all classes, methods, and parameters is available below:

```{toctree}
:maxdepth: 2

autoapi/index
```

## Type Hints

All functions and classes include type hints for better IDE support and type checking with mypy:

```python
def sqrtm_SPD(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix square root of SPD matrices.

    Parameters
    ----------
    X : torch.Tensor
        Input SPD matrices of shape (..., n, n)

    Returns
    -------
    torch.Tensor
        Square root of X with same shape
    """
    ...
```

## Automatic Differentiation

All operations support automatic differentiation through PyTorch's autograd system:

```python
X = spd.random_SPD(50, 10)
X.requires_grad = True

Y = spd.logm_SPD(X)
loss = Y.norm()
loss.backward()

print(X.grad)  # Gradients computed automatically
```
