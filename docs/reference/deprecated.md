# Deprecated names

Renamed or moved functions keep their old name as a **deprecated alias**: the
alias behaves like its replacement and emits a `DeprecationWarning` the first
time it is used. The aliases are left out of the other reference pages. The
table below is generated from the package (every object carrying the
`__deprecated__` marker set by `yetanotherspdnet._deprecation`).

```{deprecated-table}
```

To find the uses in your code, turn the warnings into errors:

```bash
python -W error::DeprecationWarning -m pytest
```
