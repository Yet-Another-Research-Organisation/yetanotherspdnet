"""Deprecated aliases of renamed or moved public objects."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, cast


# aliases that already warned in this process (each alias warns once)
_WARNED: set[str] = set()


def _warn_once(alias: str, message: str) -> None:
    """Emit the DeprecationWarning of *alias* the first time it is used."""
    if alias in _WARNED:
        return
    _WARNED.add(alias)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _mark_deprecated(obj: Any, message: str) -> None:
    """Set the ``__deprecated__`` marker read by the documentation on *obj*."""
    obj.__deprecated__ = message


def _deprecated(
    alias: str,
    target: Any,
    note: str = "",
    replacement: str | None = None,
    call: Callable | None = None,
) -> Any:
    """
    Return a deprecated alias of *target* named *alias*.

    The alias behaves like *target* and emits a ``DeprecationWarning`` the first
    time it is used (called, or ``.apply``-ed for a ``torch.autograd.Function``).
    It carries a ``__deprecated__`` attribute, which the documentation uses to
    leave it out of the reference tables.

    Parameters
    ----------
    alias : str
        Old (deprecated) name.
    target : Callable or type
        Object to use instead: a function, or a ``torch.autograd.Function``
        subclass.
    note : str, optional
        Extra sentence appended to the warning message.
    replacement : str, optional
        Name to recommend in the warning, when it is not *target* itself
        (the alias keeps the old behaviour but a different object replaces it).
    call : Callable, optional
        What the alias forwards its arguments to, when the old signature
        differs from the one of *target* (default: *target*, or ``target.apply``
        for a ``torch.autograd.Function``).

    Returns
    -------
    Callable or type
        Function (or ``torch.autograd.Function`` subclass) forwarding to
        *target*.
    """
    if replacement is None:
        replacement = f"{target.__module__}.{target.__qualname__}"
    message = f"{alias} is deprecated, use {replacement} instead."
    if note:
        message += f" {note}"
    doc = f"Deprecated: use :obj:`{replacement}` instead."

    if isinstance(target, type):
        forward_to = call or cast(Any, target).apply

        def apply(cls, *args, **kwargs):  # noqa: ARG001
            _warn_once(alias, message)
            return forward_to(*args, **kwargs)

        return type(
            alias,
            (target,),
            {
                "__doc__": doc,
                "__module__": target.__module__,
                "__deprecated__": message,
                "apply": classmethod(apply),
            },
        )

    forward_to = call or target

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        _warn_once(alias, message)
        return forward_to(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = alias
    wrapper.__doc__ = doc
    _mark_deprecated(wrapper, message)
    del wrapper.__wrapped__  # keep autodoc / inspect on the alias, not the target
    return wrapper
