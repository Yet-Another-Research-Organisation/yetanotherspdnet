# Models

`yetanotherspdnet.model` — complete classifiers taking a batch of SPD matrices
`(B, n, n)` and returning logits `(B, n_classes)`. They assemble the
{doc}`layers`, optional {doc}`batchnorm` and, for the residual networks, the
{doc}`residual` blocks. Their constructors expose every option of those
layers; the {doc}`../user_guide/models` guide groups these arguments by role.

```{eval-rst}
.. currentmodule:: yetanotherspdnet.model

.. autosummary::
   :nosignatures:

   SPDnet
   RResNet
   GBWBNRResNet
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.model.SPDnet
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.model.RResNet
   :members: forward, register_optimizer_hook
```

```{eval-rst}
.. autoclass:: yetanotherspdnet.model.GBWBNRResNet
   :members: forward, register_optimizer_hook
```
