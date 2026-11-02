"""NotebookEngine mixin: Wick contraction (QA and SymPy chain paths).

Extracted verbatim from engine.py (refactor Stage 3, 2026-07-04),
following the EngineHistoryMixin / EnginePersistenceMixin pattern:
state is reached via self; NotebookEngine appears only in annotations
(PEP 563), so there is no import from engine.py and no cycle.
"""
from __future__ import annotations

import re
from notebook.runtime.i18n import t
from notebook.models.expression import (  # noqa: F401
    Expression,
)


class EngineWickMixin:
    """Wick contraction (QA and SymPy chain paths).  (see module docstring)."""

    def _with_wick_warnings(self, terms: "list[Expression]") -> "list[Expression]":
        """Record whether the driver's rules reached any operator pair at all.

        An empty answer has TWO meanings and the researcher cannot tell them
        apart: the contractions may have vanished (δ(-k,k)=0 — physically
        correct, and asserted by `test_wick_contract_sympy_delta_vanishes`), or
        no rule may have reached a pair.  The driver knows which, because it
        counts matches BEFORE any coefficient is evaluated.

        Kept on the ENGINE rather than on an expression: the case worth speaking
        about is precisely the one with no expression to hang a warning on.
        """
        self.last_wick_warnings = [
            t(_k, self.lang)
            for _k in self._handler.result_warnings("wick_contract", self._adapter)
        ]
        return terms

    def wick_contract(
        self,
        expr_id: str,
        contraction_rules: "list[dict]",
        include_constant: bool = False,
    ) -> "list[Expression]":
        """Perform Wick contractions and register each result as a separate Expression.

        Each contraction rule specifies a pair of operator types and a c-number
        value to replace the contraction.  All possible pair contractions matching
        the rules are generated and returned as individual Expressions.

        Args:
            expr_id:           Source expression ID (e.g. "E3").
            contraction_rules: List of dicts with keys:
                               - op1:   operator spec ("a'" or "a")
                               - op2:   operator spec
                               - value: contraction value string.
                                        Use INDEX1/INDEX2 as placeholders for
                                        the indices of op1/op2 respectively.
                                        Plain function name → auto-expanded to
                                        "name(INDEX1,INDEX2)".
            include_constant:  Include doubly-contracted scalar (constant) terms.

        Returns:
            List of new Expressions, one per contraction term.
            Empty list if no contractions matched.

        Raises:
            ValueError: if the CAS returns an error.
            KeyError:   if expr_id is not found.
        """
        if expr_id not in self.expressions:
            raise KeyError(f"wick_contract: expression {expr_id!r} not found")

        # SymPy path: expression has algebraic indices (sympy_canonical set or
        # canonical contains -k / k+q patterns that QA cannot handle), OR
        # a contraction rule uses delta(INDEX1,INDEX2) (KroneckerDelta template).
        _src_expr = self.expressions[expr_id]
        _sympy_str = getattr(_src_expr, "sympy_canonical", None)
        if _sympy_str is None:
            # Route to SymPy when the expression has algebraic indices (-k, k+q).
            # _nb_parse_op_str_multi handles both single-index and multi-index
            # (k,σ spin+momentum) operators, so no exclusion needed.
            _check_str = _src_expr.input_canonical or _src_expr.canonical or ""
            if re.search(r"\(\s*-[A-Za-z]|\+[A-Za-z]|[A-Za-z]\s*\+", _check_str):
                _sympy_str = _check_str

        # Also route to SymPy when any contraction rule uses delta() template
        # (QA/Julia cannot evaluate delta(INDEX1,INDEX2) — it's a parse error).
        _has_delta_rule = any(
            "delta(" in str(r.get("value", "")) or "KroneckerDelta(" in str(r.get("value", ""))
            for r in contraction_rules
        )
        if _sympy_str is None and _has_delta_rule:
            _check_str = _src_expr.input_canonical or _src_expr.canonical or ""
            if _check_str:
                _sympy_str = _check_str

        if _sympy_str is not None:
            return self._with_wick_warnings(self._wick_contract_sympy(
                expr_id, _sympy_str, contraction_rules, include_constant
            ))

        step, raw_value = self.run_operation("wick_contract", {
            "expr": expr_id,
            "contraction_rules": contraction_rules,
            "include_constant": include_constant,
        })

        # raw_value is the Julia repr of Vector{QuExpr}, e.g.:
        # "QuExpr[-V_HF a†(2) a(4), V_HF a†(2) a(3), ...]"
        # Parse it by executing each element individually in the Julia session.
        if not raw_value or raw_value.strip() in ("QuExpr[]", "[]"):
            return []

        # Ask Julia to return the length and individual elements.
        nb_result = f"_nb_wc_result_{step.step_id.replace('-','_')}"
        store_cmd = f"{nb_result} = _nb_wick_contract"
        # The result is already stored from run_operation; re-execute to get list.
        # Re-execute directly using the stored nb var from the step.
        src_nb = f"_nb_{expr_id}"
        import json as _json
        rules_json    = _json.dumps([
            {
                "op1_name":   r.get("op1", "").rstrip("'†"),
                "op1_create": r.get("op1", "").endswith(("'", "†")),
                "op2_name":   r.get("op2", "").rstrip("'†"),
                "op2_create": r.get("op2", "").endswith(("'", "†")),
                "value": r.get("value", ""),
            }
            for r in contraction_rules
        ])
        # Build symmetry_map from SymbolContext.
        _sym_map: dict[str, str] = {}
        for r in contraction_rules:
            _vt = r.get("value", "")
            _fn = _vt.split("(")[0].strip() if "(" in _vt else _vt.strip()
            if _fn:
                _prop = self.symbol_context.get(_fn)
                if _prop and _prop.quantity.symmetry:
                    _sym_map[_fn] = _prop.quantity.symmetry

        rules_escaped    = rules_json.replace("\\", "\\\\").replace('"', '\\"')
        symmetry_escaped = _json.dumps(_sym_map).replace("\\", "\\\\").replace('"', '\\"')
        incl             = "true" if include_constant else "false"

        # Store results in a session variable for indexed access.
        store_r = self._adapter.execute(
            f'_nb_wc_tmp = _nb_wick_contract({src_nb}, "{rules_escaped}", {incl}, "{symmetry_escaped}")'
        )
        if not store_r.ok:
            raise ValueError(f"wick_contract storage failed: {store_r.error}")

        len_r = self._adapter.execute("length(_nb_wc_tmp)")
        if not len_r.ok:
            return []
        n_terms = int(len_r.value.strip())

        expressions: list[Expression] = []
        for idx in range(1, n_terms + 1):
            item_r = self._adapter.execute(f"_nb_wc_tmp[{idx}]")
            if not item_r.ok:
                continue
            display_str = item_r.value.strip()

            # register_expression allocates IDs, creates a Step in self.steps,
            # and stores the expression in self.expressions.  This is required so
            # that _active_expr_ids() can find these expressions later.
            # canonical = "_nb_{eid}" after we store the Julia variable.
            # We first register with the display string, then fix up the canonical.
            latex_str = self._handler.postprocess_result_latex(
                display_str, display_str, self.symbol_context
            )
            expr = self.register_expression(
                canonical=display_str,   # temporary; will be overwritten below
                latex=latex_str,
            )
            # Now store proper Julia session variable and update canonical.
            nb_var = f"_nb_{expr.display_id}"
            self._adapter.execute(f"{nb_var} = _nb_wc_tmp[{idx}]")
            # Update canonical in the expressions dict to the _nb_ reference.
            self.expressions[expr.display_id] = self.expressions[expr.display_id].model_copy(
                update={"canonical": nb_var}
            )
            expressions.append(self.expressions[expr.display_id])

        return expressions
    def _wick_contract_sympy(
        self,
        expr_id: str,
        sympy_str: str,
        contraction_rules: "list[dict]",
        include_constant: bool,
    ) -> "list[Expression]":
        """SymPy path for wick_contract: handles algebraic indices (-k, k+q, …).

        Calls _nb_sympy_wick in Julia and registers each result term as a
        separate Expression with sympy_canonical set for chaining.
        """
        import json as _json

        # Build rules JSON (same structure as QA rules)
        julia_rules = [
            {
                "op1_name":   r.get("op1", "").rstrip("'†"),
                "op1_create": r.get("op1", "").endswith(("'", "†")),
                "op2_name":   r.get("op2", "").rstrip("'†"),
                "op2_create": r.get("op2", "").endswith(("'", "†")),
                "value":      r.get("value", ""),
            }
            for r in contraction_rules
        ]
        rules_json = _json.dumps(julia_rules)
        rules_escaped = rules_json.replace("\\", "\\\\").replace('"', '\\"')
        incl = "true" if include_constant else "false"

        # Detect per-operator statistics and convert from prime notation to cd/c.
        # The backend handler encapsulates the SymPy-Wick-specific preparation
        # (Julia collects statistics + converts operator notation; base no-op).
        stats_json, sympy_str = self._handler.prepare_sympy_wick_inputs(
            sympy_str, self.symbol_context
        )
        # Collect momentum variable names from operator index expressions.
        # For multi-index operators like ad(-k,u), collect first index only (-k → k).
        # Single-letter identifiers with algebraic arithmetic (-, +) are momentum vars;
        # other single letters (spin labels u,d,↑,↓) are kept as SymPy Symbols.
        _vars: set[str] = set()
        for _m in re.finditer(r"[a-z_][a-z_0-9']*\(([^)]+)\)", sympy_str):
            _args = _m.group(1).split(",")
            for _arg in _args:
                # Extract alphabetic tokens from each index argument
                for _tok in re.findall(r"[a-z][a-z0-9_]*", _arg):
                    if len(_tok) == 1:
                        _vars.add(_tok)
        vars_str = ",".join(sorted(_vars)) if _vars else "k"

        # Escape the sympy_str for Julia string literal
        sympy_escaped = sympy_str.replace("\\", "\\\\").replace('"', '\\"')

        stats_esc = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        cmd = (f'_nb_sympy_wick("{sympy_escaped}", "{rules_escaped}", '
               f'{incl}, "{stats_esc}", "{vars_str}")')

        result = self._adapter.execute(cmd)
        if not result.ok:
            raise ValueError(f"wick_contract (SymPy): {result.error}")

        raw_json = result.value.strip()
        if not raw_json or raw_json in ("[]", "null"):
            return []

        try:
            terms_raw = _json.loads(raw_json)
        except _json.JSONDecodeError:
            return []

        expressions: list[Expression] = []
        for term in terms_raw:
            if not isinstance(term, list) or len(term) < 2:
                continue
            val_str, lat_str = str(term[0]), str(term[1])
            if not val_str or val_str == "0":
                continue

            latex_str = self._handler.postprocess_result_latex(
                val_str, lat_str, self.symbol_context
            )
            expr = self.register_expression(
                canonical=val_str,
                latex=latex_str,
            )
            # Mark as SymPy result for chaining
            self.expressions[expr.display_id] = self.expressions[expr.display_id].model_copy(
                update={"sympy_canonical": val_str}
            )
            expressions.append(self.expressions[expr.display_id])

        return expressions
