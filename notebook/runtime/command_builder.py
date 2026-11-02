"""Abstract base class for template-driven CAS command builders.

Concrete implementations live in maxima_command_builder.py and
julia_command_builder.py.  This module also owns the BuildResult type shared
by both builders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

from notebook.models.ir import Expr, CasOperation
from notebook.runtime.expr_generator import ExprGenerator


@dataclass
class BuildResult:
    ok: bool
    command: str
    error: str | None = None


class CommandBuilder:
    """Template-driven CAS command builder.

    Loads an operation spec JSON (``backends/<name>.full.json``) and renders
    :class:`CasOperation` nodes into ready-to-execute CAS strings.

    Subclasses set :attr:`_backend_label` and supply the default spec path via
    their ``__init__``.
    """

    _backend_label: str = "unknown"

    def __init__(self, spec_path: Path | str) -> None:
        with open(spec_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._templates: dict[str, str] = {}
        self._defaults: dict[str, dict] = {}
        for name, entry in data["operations"].items():
            self._templates[name] = entry["template"]
            if "defaults" in entry:
                self._defaults[name] = entry["defaults"]
        self._compiler = ExprGenerator()

    #: 研究者が書いた**判別子**が、同じ操作の別テンプレートを選ぶ。
    #: 値そのものが接尾辞になるので（`side="left"` → `multiply_both_sides_left"`）、
    #: 表に無い値は何も選ばない。
    _VARIANT_FIELDS = ("side", "direction")

    def build(self, cas_op: CasOperation) -> BuildResult:
        operation = cas_op.operation
        # A SIDE picks a different template of the same operation.  It is only
        # ever written for a product that does not commute — `\left(1-G^0\Sigma
        # \right)^{-1}` applied to a matrix equation — and the two sides are
        # genuinely different equations, so this is the researcher saying which
        # one they mean, not a guess.  Without it both-sides multiplication put
        # the factor on the LEFT of the label and on the RIGHT of the value, and
        # the Dyson equation could only be TYPED: `(1-G^0Σ)G = G^0` multiplied by
        # `(1-G^0Σ)^{-1}` came back with the left side unreduced (measured
        # 2026-08-21).  Maxima reduces `A^^-1 . A` to 1 on its own, so what was
        # missing was only the side and the non-commutative product.
        # `direction` is the same shape and was measured to need it: a
        # ONE-SIDED LIMIT is a different question from the two-sided one
        # (`limit(abs(x)/x, x, 0)` is `ind`; the sides are 1 and -1), and the
        # template had no place for it, so the side was rendered in the DISPLAY
        # (`\lim_{x\to 0^{+}}`) and dropped on the way to the CAS — the label
        # said one thing and the value answered another (LIMIT-ONE-SIDED,
        # 2026-09-11).  Widening is safe by construction: the variant template
        # must EXIST, so a field named `direction` on an operation with no
        # variant cannot steer anything.
        for _field in self._VARIANT_FIELDS:
            _value = cas_op.args.get(_field)
            if isinstance(_value, str) and f"{operation}_{_value}" in self._templates:
                operation = f"{operation}_{_value}"
                break
        template = self._templates.get(operation)
        if template is None:
            return BuildResult(
                ok=False,
                command="",
                error=f"No {self._backend_label} template for operation '{operation}'",
            )

        resolved = self._resolve_args(cas_op.args)
        for k, v in self._defaults.get(operation, {}).items():
            resolved.setdefault(k, str(v))

        try:
            command = template.format_map(resolved)
        except KeyError as missing:
            return BuildResult(
                ok=False,
                command="",
                error=f"Template placeholder {missing} not found in args for '{operation}'",
            )

        return BuildResult(ok=True, command=command)

    @property
    def supported_operations(self) -> list[str]:
        return list(self._templates.keys())

    def _resolve_args(self, args: dict[str, Union[Expr, List[Expr], str, int, float]]) -> dict:
        resolved: dict = {}
        for key, value in args.items():
            if isinstance(value, list):
                resolved[key] = ", ".join(self._resolve_value(v) for v in value)
            else:
                resolved[key] = self._resolve_value(value)
        return resolved

    def _resolve_value(self, value: object) -> str:
        if isinstance(value, Expr):
            return self._compiler.generate(value)
        return str(value)
