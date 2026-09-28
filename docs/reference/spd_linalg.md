# SPD linear algebra

`yetanotherspdnet.functions.spd_linalg` — matrix functions computed through
the eigendecomposition $X = U \operatorname{diag}(\lambda) U^\top$,
$f(X) = U \operatorname{diag}(f(\lambda)) U^\top$, plus congruences, whitening
and vectorizations. Every operation exists twice: a function differentiated
by autograd, and a `torch.autograd.Function` whose backward is written by
hand (Daleckii–Krein formula), which stays exact and finite when eigenvalues
are close or repeated. Layers pick one with `use_autograd` (manual by
default); see {doc}`../user_guide/numerics`.

Functions returning a matrix built from an eigendecomposition
(`sqrtm_SPD`, `logm_SPD`, …) return the tuple `(result, eigvals, eigvecs)`:
index `[0]` for the matrix. The `Function` classes return the matrix only.

```{dualpath-table} yetanotherspdnet.functions.spd_linalg
```

## Details

```{eval-rst}
.. automodule:: yetanotherspdnet.functions.spd_linalg
   :members:
   :member-order: bysource
```
