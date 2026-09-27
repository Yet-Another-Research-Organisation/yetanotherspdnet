# Parametrizations

`yetanotherspdnet.nn.parametrizations` — modules registered with
`torch.nn.utils.parametrize` to keep a parameter on its manifold while a
standard optimizer updates an unconstrained tensor. The *adaptive* variants
work around a reference point that moves during training (the `"dynamic"`
mode of `BiMap` and of the batch normalization bias); see
{doc}`../user_guide/layers`.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.nn.parametrizations

.. autosummary::
   :nosignatures:

   SPDParametrization
   SPDAdaptiveParametrization
   StiefelAdaptiveParametrization
   ScalarSoftPlusParametrization
   ScalarSigmoidParametrization
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.parametrizations.SPDParametrization
   :members: forward, right_inverse
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.parametrizations.SPDAdaptiveParametrization
   :members: forward, right_inverse, update_reference_point
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.parametrizations.StiefelAdaptiveParametrization
   :members: forward, right_inverse, update_reference_point
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.parametrizations.ScalarSoftPlusParametrization
   :members: forward, right_inverse
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.nn.parametrizations.ScalarSigmoidParametrization
   :members: forward, right_inverse
```
