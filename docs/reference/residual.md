# Residual blocks

`yetanotherspdnet.nn.rresnet_layers` — the building blocks of `RResNet` and
`GBWBNRResNet`. `SpectralVectorField` computes a tangent vector from the
spectrum of each matrix; `ResidualBlock` moves the matrix along it with an
affine-invariant (unit step) or log-Euclidean exponential map. See
{doc}`../user_guide/residual`.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.nn.rresnet_layers

.. autosummary::
   :nosignatures:

   ResidualBlock
   SpectralVectorField
   affine_invariant_norm
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.rresnet_layers.ResidualBlock
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.rresnet_layers.SpectralVectorField
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autofunction:: yetanotherspdnet.nn.rresnet_layers.affine_invariant_norm
```
