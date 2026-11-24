"""Sphinx directive ``dualpath-table``: one row per operation of a module.

Most operations of the library exist twice (see the numerics guide): a
``snake_case`` function differentiated by autograd and a ``CamelCase``
``torch.autograd.Function`` (or a ``CamelCase`` function chaining such classes)
with a hand-written backward. This directive pairs them by name
(``sqrtm_SPD`` / ``SqrtmSPD``, ``vec_batch`` / ``VecBatch``; see
``dualpath_pairing.py``) and renders a table with the first docstring line, so
the reference shows each operation once instead of two unrelated entries::

    ```{dualpath-table} yetanotherspdnet.functions.spd_linalg
    ```
"""

from __future__ import annotations

import importlib
import inspect

from docutils import nodes
from docutils.statemachine import StringList
from dualpath_pairing import pair_members, public_members
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


def setup(app):
    app.add_directive("dualpath-table", DualPathTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
