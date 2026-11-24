"""Sphinx directives rendering tables from the code of the package.

``dualpath-table``: one row per operation of a module.

Most operations of the library exist twice (see the numerics guide): a
``snake_case`` function differentiated by autograd and a ``CamelCase``
``torch.autograd.Function`` (or a ``CamelCase`` function chaining such classes)
with a hand-written backward. This directive pairs them by name
(``sqrtm_SPD`` / ``SqrtmSPD``, ``vec_batch`` / ``VecBatch``; see
``dualpath_pairing.py``) and renders a table with the first docstring line, so
the reference shows each operation once instead of two unrelated entries::

    ```{dualpath-table} yetanotherspdnet.functions.spd_linalg
    ```

``autograd-only-table``: the operations without manual backward, read from
``functions.dispatch.AUTOGRAD_ONLY`` (name and reason).

``deprecated-table``: every deprecated alias of the package (objects marked
``__deprecated__``, see ``yetanotherspdnet._deprecation``), with the module
that defines it and its replacement.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from docutils import nodes
from docutils.statemachine import StringList
from dualpath_pairing import (
    deprecated_members,
    pair_members,
    public_members,
    replacement,
)
from sphinx.util.docutils import SphinxDirective


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    line = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return line.replace("|", "\\|")


def _role(obj) -> str:
    return "py:class" if inspect.isclass(obj) else "py:func"


class DualPathTable(SphinxDirective):
    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        module = importlib.import_module(self.arguments[0])
        objs = dict(public_members(module))
        rows = []
        for func, manual in pair_members(list(objs.items())):
            main = objs[func] if func else objs[manual]
            rows.append(
                (
                    f"{{{_role(objs[func])}}}`~{module.__name__}.{func}`"
                    if func
                    else "--",
                    f"{{{_role(objs[manual])}}}`~{module.__name__}.{manual}`"
                    if manual
                    else "--",
                    _summary(main),
                )
            )
        lines = [
            "| autograd path | manual backward | |",
            "|---|---|---|",
            *(f"| {a} | {b} | {c} |" for a, b, c in rows),
        ]
        container = nodes.container(classes=["dualpath-table"])
        self.state.nested_parse(
            StringList(lines, source="dualpath-table"), self.content_offset, container
        )
        return [container]


def _table(directive: SphinxDirective, lines: list[str], cls: str) -> list[nodes.Node]:
    container = nodes.container(classes=[cls])
    directive.state.nested_parse(
        StringList(lines, source=cls), directive.content_offset, container
    )
    return [container]


def _short(name: str) -> str:
    return name.removeprefix("yetanotherspdnet.")


def _replacement_cell(new: str) -> str:
    if not new:
        return "--"
    if re.fullmatch(r"[\w.]+", new):  # a name: link it
        return f"{{py:obj}}`{new} <yetanotherspdnet.{new}>`"
    return f"`{new}`"  # an expression, e.g. a call with fixed arguments


class AutogradOnlyTable(SphinxDirective):
    """Table of ``functions.dispatch.AUTOGRAD_ONLY``."""

    required_arguments = 0
    has_content = False

    def run(self) -> list[nodes.Node]:
        dispatch = importlib.import_module("yetanotherspdnet.functions.dispatch")
        lines = [
            "| operation without manual backward | why, and the manual alternative |",
            "|---|---|",
        ]
        for name, reason in dispatch.AUTOGRAD_ONLY.items():
            if re.fullmatch(r"[\w.]+", name):
                cell = f"{{py:obj}}`~{name}`"
            else:
                cell = f"`{_short(name)}`"
            reason = reason.replace("|", "\\|")
            lines.append(f"| {cell} | {reason} |")
        return _table(self, lines, "autograd-only-table")


class DeprecatedTable(SphinxDirective):
    """Table of the deprecated aliases of the whole package."""

    required_arguments = 0
    has_content = False

    def run(self) -> list[nodes.Node]:
        package = importlib.import_module("yetanotherspdnet")
        rows = []
        seen = set()
        for info in pkgutil.walk_packages(package.__path__, "yetanotherspdnet."):
            module = importlib.import_module(info.name)
            for alias, message in deprecated_members(module):
                obj = getattr(module, alias)
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                new = replacement(message)
                if re.fullmatch(r"\w+", new):  # bare name: same module
                    new = f"{info.name}.{new}"
                note = message.split(" instead", 1)[-1].lstrip(". ") if new else message
                rows.append((_short(info.name), alias, _short(new), note))
        lines = []
        for module in sorted({row[0] for row in rows}):
            lines += [
                f"**`{module}`**",
                "",
                "| deprecated name | use instead |",
                "|---|---|",
            ]
            for _, alias, new, note in sorted(r for r in rows if r[0] == module):
                note = note.replace("|", "/")
                lines.append(f"| `{alias}` | {_replacement_cell(new)} {note} |")
            lines.append("")
        return _table(self, lines, "deprecated-table")


def setup(app):
    app.add_directive("dualpath-table", DualPathTable)
    app.add_directive("autograd-only-table", AutogradOnlyTable)
    app.add_directive("deprecated-table", DeprecatedTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
