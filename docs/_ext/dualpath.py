"""Sphinx directive ``dualpath-table``: one row per operation of a module.

Most operations of the library exist twice (see the numerics guide): a
``snake_case`` function differentiated by autograd and a ``CamelCase``
``torch.autograd.Function`` with a hand-written backward. This directive pairs
them by name (``sqrtm_SPD`` / ``SqrtmSPD``, ``vec_batch`` / ``VecBatch``) and
renders a table with the first docstring line, so the reference shows each
operation once instead of two unrelated entries::

    ```{dualpath-table} yetanotherspdnet.functions.spd_linalg
    ```
"""

from __future__ import annotations

import importlib
import inspect

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


def _key(name: str) -> str:
    return name.replace("_", "").lower()


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    line = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return line.replace("|", "\\|")


def _public(module) -> list[tuple[str, object]]:
    return [
        (name, obj)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == module.__name__
    ]


class DualPathTable(SphinxDirective):
    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        module = importlib.import_module(self.arguments[0])
        members = _public(module)
        functions = {_key(n): n for n, o in members if inspect.isfunction(o)}
        classes = {_key(n): n for n, o in members if inspect.isclass(o)}
        objs = dict(members)
        rows, seen = [], set()
        for name, _obj in members:  # source order
            key = _key(name)
            if key in seen:
                continue
            seen.add(key)
            func, cls = functions.get(key), classes.get(key)
            main = objs[func] if func else objs[cls]
            rows.append(
                (
                    f"{{py:func}}`~{module.__name__}.{func}`" if func else "--",
                    f"{{py:class}}`~{module.__name__}.{cls}`" if cls else "--",
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
