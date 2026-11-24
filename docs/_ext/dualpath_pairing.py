"""Pairing of the autograd / manual-backward twins, for ``dualpath-table``.

Kept free of Sphinx imports so that it can be unit-tested.

The autograd path is the ``snake_case`` name and the manual path the
``CamelCase`` name, whatever their kind: a manual path is either a
``torch.autograd.Function`` class (``SqrtmSPD``) or a function chaining such
classes (``AffineInvariantMean``). Pairing on the name's first letter rather
than on functions vs classes keeps ``affine_invariant_mean`` and
``AffineInvariantMean``, both functions, in the same row. Objects marked
``__deprecated__`` (deprecated aliases) are left out.
"""

from __future__ import annotations

import inspect


def key(name: str) -> str:
    """Name with underscores removed and lowercased (``sqrtm_SPD`` -> ``sqrtmspd``)."""
    return name.replace("_", "").lower()


def public_members(module) -> list[tuple[str, object]]:
    """Public functions and classes defined in *module*, in source order."""
    return [
        (name, obj)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == module.__name__
        and not hasattr(obj, "__deprecated__")
    ]


def pair_members(
    members: list[tuple[str, object]],
) -> list[tuple[str | None, str | None]]:
    """
    One ``(autograd_name, manual_name)`` row per operation, in source order.

    Either entry is None when the operation has a single path.
    """
    autograd = {key(n): n for n, _ in members if n[0].islower()}
    manual = {key(n): n for n, _ in members if not n[0].islower()}
    rows, seen = [], set()
    for name, _obj in members:
        k = key(name)
        if k in seen:
            continue
        seen.add(k)
        rows.append((autograd.get(k), manual.get(k)))
    return rows


def deprecated_members(module) -> list[tuple[str, str]]:
    """
    ``(alias, message)`` for the deprecated aliases bound in *module*.

    The aliases are found by their ``__deprecated__`` marker in ``vars(module)``,
    whatever their ``__module__`` (a function alias built with
    ``functools.wraps`` reports the module of its target).
    """
    return [
        (name, str(obj.__deprecated__))
        for name, obj in vars(module).items()
        if not name.startswith("_") and hasattr(obj, "__deprecated__")
    ]


def replacement(message: str) -> str:
    """Name or expression recommended by a deprecation message (``"use X instead"``)."""
    start = message.find("use ")
    end = message.find(" instead", start)
    if start < 0 or end < 0:
        return ""
    return message[start + 4 : end].strip()
