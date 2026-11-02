"""Julia backend mixin: SymPy-chain routing and statistics helpers.

Extracted from JuliaBackendHandler (AUD-4 stage 2b) as a mixin to shrink the
handler class.  These methods decide when an operation must route through the
SymPy chain (algebraic k-space indices) instead of QuantumAlgebra, and build
the corresponding _nb_sympy_* commands.  They are mixed into
JuliaBackendHandler and resolve sibling methods via self / MRO; they reference
no module-level state beyond BuildResult, this module's logger and local
imports.

The logger is the sentence above going stale: the extraction left a
``_logger.debug`` call behind whose name lives in ``julia_backend_handler``,
so the "best-effort, keep the original" fallback in
``prepare_sympy_wick_inputs`` raised ``NameError`` instead of falling back
(measured 2026-08-13).  A module that logs owns its logger.
``tests/test_static_name_resolution.py`` is the mechanism that now says so.
"""
from __future__ import annotations

import json
import logging
import re

from notebook.runtime.backend_handler import BuildResult

_logger = logging.getLogger(__name__)


class JuliaSympyMixin:
    """SymPy-chain routing helpers mixed into JuliaBackendHandler."""

    # ------------------------------------------------------------------
    # SymPy routing helpers for normal_order
    # ------------------------------------------------------------------

    def _get_sympy_canonical(self, expr_ref: str, expr_lookup: dict | None = None) -> "str | None":
        """Return sympy_canonical if the expression was computed by SymPy, else None.

        Looks up the '__sympy__<expr_ref>' key in expr_lookup (injected by engine.py).
        """
        if expr_lookup is not None:
            return expr_lookup.get(f"__sympy__{expr_ref}") or None
        return None

    def _maybe_sympy_normal_order(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Try to build a SymPy normal_order command.

        Returns BuildResult if SymPy routing is appropriate, None to fall
        through to the default QA template.

        SymPy routing is triggered when:
          (A) The input expression has sympy_canonical set (it came from a
              previous SymPy operation — chain path).
          (B) The input expression's canonical contains algebraic index
              expressions (-k, k+q, …) that QA cannot handle.
        """
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        expr_ref = str(args.get("expr", ""))
        if not expr_ref:
            return None

        # --- Path A: SymPy chain ---
        sympy_str = self._get_sympy_canonical(expr_ref, expr_lookup)
        if sympy_str is not None:
            stats_json, vars_str = self._sympy_stats_and_vars(sympy_str, ctx)
            stats_esc = stats_json.replace("\\", "\\\\").replace('"', '\\"')
            cmd = f'_nb_sympy_normal_order("{sympy_str}", "{stats_esc}", "{vars_str}")'
            return BuildResult(ok=True, command=cmd)

        # --- Path B: algebraic index detection ---
        canonical = expr_lookup.get(expr_ref, "")
        # Prefer input_canonical for algebraic index detection (keeps -k etc.)
        ic = expr_lookup.get(f"__input__{expr_ref}", canonical)
        if not ic:
            return None

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        surf = parser.parse(ic)
        if surf is None:
            return None
        ir = resolver.resolve(surf, ctx)
        from notebook.models.ir import Equation as _EqIR
        if isinstance(ir, _EqIR):
            ir = ir.rhs

        jcg = JuliaCodeGenerator(context=ctx)
        if not jcg._has_algebraic_indices(ir):
            return None  # No algebraic indices → use QA

        # Has algebraic indices → route to SymPy.
        # Serialize from the IR (which this method already built above), NOT by
        # regex over the researcher's canonical.  Measured 2026-08-07, the old
        # string route was wrong in two independent ways and silent in both:
        #   * it only knew the paren spelling `c'(k+q,u)`, so the braced
        #     spelling `c'_{k+q,u}` — the one CLAUDE.md documents for multi-index
        #     operators, and the one the QA index error itself tells the
        #     researcher to use — went to SymPy unconverted and came back as the
        #     literal `1`;
        #   * it kept this project's space-as-multiplication convention, which
        #     SymPy cannot parse, so `V(q) c'_{k+q,u} c_{k,u}` lost `V(q)`
        #     entirely — the electron-electron interaction strength, gone with
        #     no error anywhere.
        # The IR has neither problem: both spellings resolve to the same nodes,
        # and the serializer emits `*`.
        _extra_vars: set[str] = set()
        sympy_expr_str = jcg.serialize_to_sympy(ir, _extra_vars)
        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        vars_str   = self._collect_vars_from_sympy_str(sympy_expr_str, _extra_vars)
        cmd = f'_nb_sympy_normal_order("{sympy_expr_str}", "{stats_esc}", "{vars_str}")'
        return BuildResult(ok=True, command=cmd)

    def _maybe_sympy_combine(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route combine_expressions (+/-) to SymPy when both operands have sympy_canonical.

        Concatenates the two sympy_canonical strings with the operator, then
        calls _nb_sympy_normal_order to produce properly formatted output.
        Supports + and - only (operator products require normal-ordering logic).
        """
        op = str(args.get("op", "+"))
        if op not in ("+", "-"):
            return None

        a_ref = str(args.get("expr_a", "") or args.get("a", ""))
        b_ref = str(args.get("expr_b", "") or args.get("b", ""))
        a_sympy = self._get_sympy_canonical(a_ref, expr_lookup)
        b_sympy = self._get_sympy_canonical(b_ref, expr_lookup)

        if a_sympy is None or b_sympy is None:
            return None  # only route when BOTH are SymPy expressions

        b = b_sympy.strip()
        if op == "+":
            # Avoid "A + -B" (produces spurious sign token); use "A -B" directly
            combined = f"{a_sympy} {b}" if b.startswith("-") else f"{a_sympy} + {b}"
        else:  # "-"
            # Negate b: flip sign if single-term negative, else wrap in parens
            if b.startswith("-"):
                combined = f"{a_sympy} + {b[1:].lstrip()}"  # A - (-B) = A + B
            else:
                combined = f"{a_sympy} - ({b})"

        # Use _nb_sympy_normal_order to format (already normal-ordered terms
        # won't be reordered; the function just formats and joins them).
        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        vars_str   = self._collect_vars_from_canonical(combined)
        comb_esc   = combined.replace("\\", "\\\\").replace('"', '\\"')
        cmd = f'_nb_sympy_normal_order("{comb_esc}", "{stats_esc}", "{vars_str}")'
        return BuildResult(ok=True, command=cmd)

    def _maybe_sympy_collect_by_operators(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route collect_by_operators to SymPy when expression has sympy_canonical.

        Groups additive terms by operator sequence, sums scalar coefficients
        (with SymPy factor()), and returns the collected expression.
        """
        expr_ref = str(args.get("expr", ""))
        sympy_str = self._get_sympy_canonical(expr_ref, expr_lookup)
        if sympy_str is None:
            return None

        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        vars_str   = self._collect_vars_from_canonical(sympy_str)
        sym_esc    = sympy_str.replace("\\", "\\\\").replace('"', '\\"')
        cmd = (f'_nb_sympy_collect_by_operators("{sym_esc}",'
               f' "{stats_esc}", "{vars_str}")')
        return BuildResult(ok=True, command=cmd)

    def _maybe_sympy_substitute(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route substitute to SymPy when expression has sympy_canonical.

        Uses SymPy subs(lhs, rhs) which propagates into function arguments:
          cd(k)*c(-k)  with k→p  →  cd(p)*c(-p)
          cd(k)*c(-k)  with k→-k →  cd(-k)*c(k)

        lhs/rhs are index expressions (single symbols or algebraic expressions).
        Returns None to fall through to QA substitute template.
        """
        expr_ref = str(args.get("expr", ""))
        lhs_str  = str(args.get("lhs", "") or args.get("old", "")).strip()
        rhs_str  = str(args.get("rhs", "") or args.get("new", "")).strip()

        if not lhs_str or not rhs_str:
            return None

        sympy_str = self._get_sympy_canonical(expr_ref, expr_lookup)
        if sympy_str is None:
            return None

        # Build vars_str from the expression plus lhs/rhs tokens
        import re
        combined = sympy_str + " " + lhs_str + " " + rhs_str
        vars_found = sorted(set(re.findall(r"\b([a-z])\b", combined)))
        vars_found = [v for v in vars_found if v not in ("c", "a", "e", "i", "j")]
        vars_str = ",".join(vars_found) if vars_found else "k"

        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        lhs_esc    = lhs_str.replace("\\", "\\\\").replace('"', '\\"')
        rhs_esc    = rhs_str.replace("\\", "\\\\").replace('"', '\\"')
        sym_esc    = sympy_str.replace("\\", "\\\\").replace('"', '\\"')
        cmd = (f'_nb_sympy_substitute("{sym_esc}", "{lhs_esc}",'
               f' "{rhs_esc}", "{stats_esc}", "{vars_str}")')
        return BuildResult(ok=True, command=cmd)

    def _maybe_collect_spin_pm(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route collect_spin_pm to _nb_collect_spin_pm (Julia QA).

        Extracts sp_x, sp_y (and sm_x, sm_y) from the Sp and Sm definitions
        stored in lhs_alias_map / expr_lookup, then calls the Julia function
        that groups QuExpr terms by non-spin operator key and applies the
        inverse linear transform Sx/Sy → c_Sp·Sp + c_Sm·Sm.

        args:
          expr     -- expression ID to collect
          sp       -- E-ref or canonical of Sp definition (e.g. "Sx(1) + im*Sy(1)")
          sm       -- E-ref or canonical of Sm definition (e.g. "Sx(1) - im*Sy(1)")
          sp_label -- display label for Sp (default "Sp")
          sm_label -- display label for Sm (default "Sm")
        """
        import re as _re

        expr_ref  = str(args.get("expr", ""))
        sp_ref    = str(args.get("sp", ""))
        sm_ref    = str(args.get("sm", ""))
        sp_label  = str(args.get("sp_label", "Sp"))
        sm_label  = str(args.get("sm_label", "Sm"))

        if not (expr_ref and sp_ref and sm_ref):
            return None

        # Resolve Sp/Sm definitions through expr_lookup
        sp_canonical = expr_lookup.get(sp_ref, sp_ref)
        sm_canonical = expr_lookup.get(sm_ref, sm_ref)

        # Extract sp_x, sp_y from sp_canonical.
        # The canonical looks like "Sx(1) + im*Sy(1)" or "Sx(n) + im*Sy(n)".
        # We need the coefficients of Sx and Sy (ignoring site index).
        def _extract_xy_coeff(canonical: str) -> "tuple[str, str]":
            """Return (coeff_Sx_str, coeff_Sy_str) from a spin linear combination."""
            # Normalise: replace Sx_n / Sx(n) / Sˣ(n) with a marker
            c = canonical.strip()
            # Find coefficient of Sx: look for pattern (coeff*)?Sx[(_n)|(n)]
            sx_m = _re.search(
                r'([+\-]?\s*(?:[^+\-*/]*\*\s*)?)'   # optional coeff
                r'S(?:x|ˣ)\s*(?:\(.*?\)|_\w+)?',    # Sx or Sˣ with optional index
                c)
            sy_m = _re.search(
                r'([+\-]?\s*(?:[^+\-*/]*\*\s*)?)'
                r'S(?:y|ʸ)\s*(?:\(.*?\)|_\w+)?',
                c)

            def _coeff_from_match(m) -> str:
                if m is None:
                    return "0"
                raw = m.group(1).strip().rstrip('*').strip()
                if raw in ('', '+'):
                    return "1"
                if raw == '-':
                    return "-1"
                return raw or "1"

            return _coeff_from_match(sx_m), _coeff_from_match(sy_m)

        sp_x, sp_y = _extract_xy_coeff(sp_canonical)
        sm_x, sm_y = _extract_xy_coeff(sm_canonical)

        # Resolve Julia source for expression
        canonical = expr_lookup.get(expr_ref, "")
        if canonical.startswith("_nb_"):
            nb_src = canonical
        else:
            ic = expr_lookup.get(f"__input__{expr_ref}", canonical)
            if not ic:
                return None
            try:
                from notebook.runtime.shared_parser import SharedParser as _SP
                from notebook.runtime.julia_semantic_resolver import (
                    JuliaSemanticResolver as _JSR,
                )
                from notebook.runtime.julia_code_generator import (
                    JuliaCodeGenerator as _JCG,
                )
                from notebook.models.ir import Equation as _EqIR
                _surf = _SP().parse(ic)
                if _surf is None:
                    return None
                _ir = _JSR().resolve(_surf, ctx)
                if isinstance(_ir, _EqIR):
                    _ir = _ir.rhs
                _jcg = _JCG(context=ctx)
                _jcg.static_index_mode = True
                nb_src = f"normal_form({_jcg.generate(_ir)})"
            except Exception:
                return None

        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')

        cmd = (f'_nb_collect_spin_pm({nb_src}, '
               f'"{_esc(sp_x)}", "{_esc(sp_y)}", '
               f'"{_esc(sm_x)}", "{_esc(sm_y)}", '
               f'"{_esc(sp_label)}", "{_esc(sm_label)}")')
        return BuildResult(ok=True, command=cmd)

    def _maybe_operator_identity(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route substitute to _nb_operator_identity when lhs is a QA operator.

        Triggered when:
          - lhs symbol is registered in SymbolContext as a quantum operator
            (statistics: fermion, boson, or noncommutative)
          - expression is a QA QuExpr (not a SymPy chain result)

        This handles operator-valued linear identities such as:
          Sx → (Sp+Sm)/2,  Sy → -im*(Sp-Sm)/2
        inside a QuExpr, preserving non-commutativity via Cartesian product
        expansion followed by normal_form.

        Works for any backend domain (spin, fermionic, bosonic).
        Maxima and SymPy paths do not use this — Maxima subst() already
        handles expression-pattern replacement, and the SymPy path is
        guarded by _maybe_sympy_substitute above.
        """
        import re as _re

        expr_ref = str(args.get("expr", ""))
        lhs_str  = str(args.get("lhs", "") or args.get("old", "")).strip()
        rhs_str  = str(args.get("rhs", "") or args.get("new", "")).strip()

        if not (expr_ref and lhs_str and rhs_str):
            return None

        # Only trigger for QA expressions (no sympy_canonical)
        if self._get_sympy_canonical(expr_ref, expr_lookup) is not None:
            return None

        # Extract the base operator name: "Sx" from "Sx" or "Sx(1)".
        # Suffix-style spin components carry the site as an underscore segment
        # (e.g. "Sz_1"), and a bare ^\w* match would greedily swallow it ("Sz_1")
        # and miss the registered operator "Sz".  Resolve to the LONGEST prefix
        # that is actually registered, stripping trailing "_<seg>" segments.
        lhs_tok = _re.match(r"^([A-Za-z_]\w*)", lhs_str)
        if not lhs_tok:
            return None
        lhs_base = lhs_tok.group(1)
        if ctx.registry.get(lhs_base) is None and "_" in lhs_base:
            _parts = lhs_base.split("_")
            while len(_parts) > 1:
                _parts = _parts[:-1]
                _cand = "_".join(_parts)
                if ctx.registry.get(_cand) is not None:
                    lhs_base = _cand
                    break

        # Check if lhs is registered as a quantum operator.
        # ctx.registry holds SymbolProperty (with .quantity and .statistics).
        # NOTE: a scalar parameter may carry a default statistics.kind of "bose"
        # (see SymbolProperty defaults), so statistics alone is not sufficient —
        # we must also confirm the Quantity is NOT a scalar.  Otherwise scalar
        # substitutions (e.g. k → sqrt(2*m*mu)/hbar) get misrouted here and the
        # operator-identity expansion silently no-ops.
        prop  = ctx.registry.get(lhs_base)
        qty_kind = getattr(getattr(prop, "quantity", None), "kind", None)
        is_op = (prop is not None and
                 qty_kind not in ("scalar", "anonymous") and
                 getattr(getattr(prop, "statistics", None), "kind", None)
                 in ("fermi", "bose", "noncommutative"))
        if not is_op:
            return None

        # Build rhs_template: replace the index in rhs_str with "IDX" placeholder.
        # e.g. rhs = "(Sp(1)+Sm(1))/2" with lhs = "Sx(1)" → replace "1" with "IDX"
        # e.g. rhs = "(Sp(i)+Sm(i))/2" with lhs = "Sx(i)" → replace "i" with "IDX"
        lhs_idx_m = _re.search(r"\(([^)]+)\)", lhs_str)
        lhs_idx = None
        if lhs_idx_m:
            lhs_idx = lhs_idx_m.group(1).strip()
        elif lhs_str.startswith(lhs_base + "_"):
            # Suffix style "Sz_1": the site index is the segment after the base.
            lhs_idx = lhs_str[len(lhs_base) + 1:].strip()
        if lhs_idx:
            # Replace the index in rhs with IDX placeholder (word-boundary safe)
            rhs_template = _re.sub(
                r"\b" + _re.escape(lhs_idx) + r"\b",
                "IDX", rhs_str
            )
        else:
            # No explicit index in lhs (bare name like "Sx") — keep rhs as-is
            rhs_template = rhs_str

        # Resolve rhs through lhs_alias_map / expr_lookup (E-ref → canonical)
        if rhs_template in expr_lookup:
            rhs_template = expr_lookup[rhs_template]

        # If rhs is a bare operator name (no parentheses/operators), append (IDX)
        # so Julia can call it as a constructor with the matched index.
        # e.g. "alpha" → "alpha(IDX)" → "alpha(1)" at substitution time.
        if _re.match(r'^[A-Za-z_]\w*$', rhs_template):
            rhs_prop = ctx.registry.get(rhs_template)
            rhs_is_op = (rhs_prop is not None and
                         getattr(getattr(rhs_prop, "statistics", None), "kind", None)
                         in ("fermi", "bose", "noncommutative"))
            if rhs_is_op:
                rhs_template = rhs_template + "(IDX)"

        # Resolve the Julia source expression.
        # CAS-derived expressions have canonical starting with "_nb_" → use directly.
        # Registered expressions need inline evaluation.
        canonical = expr_lookup.get(expr_ref, "")
        if canonical.startswith("_nb_"):
            nb_src = canonical
        else:
            # Build inline Julia code from input_canonical via IR pipeline.
            ic = expr_lookup.get(f"__input__{expr_ref}", canonical)
            if not ic:
                return None
            try:
                from notebook.runtime.shared_parser import SharedParser as _SP
                from notebook.runtime.julia_semantic_resolver import (
                    JuliaSemanticResolver as _JSR,
                )
                from notebook.runtime.julia_code_generator import (
                    JuliaCodeGenerator as _JCG,
                )
                from notebook.models.ir import Equation as _EqIR

                _surf = _SP().parse(ic)
                if _surf is None:
                    return None
                _ir = _JSR().resolve(_surf, ctx)
                if isinstance(_ir, _EqIR):
                    _ir = _ir.rhs
                _jcg = _JCG(context=ctx)
                _jcg.static_index_mode = True
                nb_src = f"normal_form({_jcg.generate(_ir)})"
            except Exception:
                return None

        lhs_esc = lhs_base.replace('"', '\\"')
        rhs_esc = rhs_template.replace("\\", "\\\\").replace('"', '\\"')
        cmd = f'_nb_operator_identity({nb_src}, "{lhs_esc}", "{rhs_esc}")'
        return BuildResult(ok=True, command=cmd)

    def _maybe_sympy_vacuum_expectation(
        self,
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route vacuum_expectation to SymPy when expression has algebraic indices.

        Mirrors _maybe_sympy_normal_order: two routing paths.
          Path A: expression already has sympy_canonical → use it directly.
          Path B: input_canonical has algebraic indices → convert and route.

        Returns None to fall through to QA template.
        """
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        expr_ref = str(args.get("expr", ""))

        # --- Path A: SymPy chain ---
        sympy_str = self._get_sympy_canonical(expr_ref, expr_lookup)
        if sympy_str is not None:
            stats_json, vars_str = self._sympy_stats_and_vars(sympy_str, ctx)
            stats_esc = stats_json.replace("\\", "\\\\").replace('"', '\\"')
            cmd = f'_nb_sympy_vacuum_expectation("{sympy_str}", "{stats_esc}", "{vars_str}")'
            return BuildResult(ok=True, command=cmd)

        # --- Path B: algebraic index detection ---
        ic = (
            expr_lookup.get(f"__input__{expr_ref}")
            or expr_lookup.get(expr_ref, "")
        )
        if not ic:
            return None

        parser   = SharedParser()
        resolver = JuliaSemanticResolver()
        surf = parser.parse(ic)
        if surf is None:
            return None
        ir = resolver.resolve(surf, ctx)
        from notebook.models.ir import Equation as _EqIR
        if isinstance(ir, _EqIR):
            ir = ir.rhs
        jcg = JuliaCodeGenerator(context=ctx)
        if not jcg._has_algebraic_indices(ir):
            return None

        # Serialize from the IR, for the reasons written out in
        # _maybe_sympy_normal_order: the regex route understands only the paren
        # spelling and keeps space-as-multiplication, and the driver used to
        # turn the resulting parse failure into the literal 1.
        _extra_vars: set[str] = set()
        sympy_expr_str = jcg.serialize_to_sympy(ir, _extra_vars)
        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        vars_str   = self._collect_vars_from_sympy_str(sympy_expr_str, _extra_vars)
        cmd = f'_nb_sympy_vacuum_expectation("{sympy_expr_str}", "{stats_esc}", "{vars_str}")'
        return BuildResult(ok=True, command=cmd)

    def _prime_to_cd_notation(self, canonical: str) -> str:
        """Convert prime/dagger operator notation to SymPy cd/ad format.

        c'(k)  → cd(k)     a'(k)  → ad(k)
        c†(k)  → cd(k)     a†(k)  → ad(k)
        Scalar coefficients (-xi, Delta, etc.) are preserved unchanged.
        """
        import re
        # c'(k) / c†(k) → cd(k)  for any operator base name
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)['†]\(", lambda m: m.group(1) + "d(", canonical)
        return s

    def _build_mixed_commutator(
        self,
        a_ref: str,
        b_ref: str,
        a_sympy: "str | None",
        b_sympy: "str | None",
        args: dict,
        expr_lookup: dict,
        ctx: "SymbolContext",
    ) -> "BuildResult | None":
        """Route commutator to SymPy when one or both operands have sympy_canonical.

        Strategy:
          - a_str: from a's sympy_canonical, or converted from input_canonical
          - b_str: from b's sympy_canonical, or literal operator string from args

        Calls _nb_sympy_general_comm(a_str, b_str, stats_json, vars_str) which
        uses the derivation property [A, B1B2] = [A,B1]B2 + B1[A,B2].
        """
        import re, json

        # ── Resolve a_str ───────────────────────────────────────────────
        if a_sympy:
            a_str = a_sympy
        else:
            # Convert QA expression to SymPy cd/c notation
            a_ic = (
                expr_lookup.get(f"__input__{a_ref}")
                or expr_lookup.get(a_ref, "")
            )
            # a_ref may itself be a literal canonical like "c(1)"
            if not a_ic and not a_ref.startswith("_nb_"):
                a_ic = a_ref
            if not a_ic:
                return None
            a_str = self._prime_to_cd_notation(a_ic)

        # ── Resolve b_str ───────────────────────────────────────────────
        if b_sympy:
            b_str = b_sympy
        else:
            b_ic = (
                expr_lookup.get(f"__input__{b_ref}")
                or expr_lookup.get(b_ref, "")
            )
            if not b_ic and not b_ref.startswith("_nb_"):
                b_ic = b_ref
            if not b_ic:
                return None
            b_str = self._prime_to_cd_notation(b_ic)

        if not a_str or not b_str:
            return None

        stats_json = self._collect_stats_json(ctx)
        stats_esc  = stats_json.replace("\\", "\\\\").replace('"', '\\"')
        vars_str   = self._collect_vars_from_canonical(a_str + " " + b_str)
        a_esc = a_str.replace("\\", "\\\\").replace('"', '\\"')
        b_esc = b_str.replace("\\", "\\\\").replace('"', '\\"')
        cmd = (
            f'_nb_sympy_general_comm("{a_esc}", "{b_esc}",'
            f' "{stats_esc}", "{vars_str}")'
        )
        return BuildResult(ok=True, command=cmd)

    def _collect_stats_json(self, ctx: "SymbolContext") -> str:
        """Build per-operator statistics JSON from the symbol context.

        Returns a JSON string like '{"c":"fermion","a":"boson"}' for use with
        _nb_sympy_normal_order and _nb_sympy_wick.  Operators declared with
        define_fermion_operator get "fermion"; define_boson_operator gets "boson".
        """
        import json
        stats_map: dict[str, str] = {}
        if ctx:
            for name, entry in ctx.registry.items():
                base = name.rstrip("'†")
                qty = entry.quantity
                kind = getattr(qty, "kind", None)
                if kind in ("fermion", "boson"):
                    stats_map[base] = kind
                elif kind == "operator":
                    # Heuristic fallback for operators declared as generic "operator"
                    if base.lower() in ("c", "d", "f"):
                        stats_map[base] = "fermion"
                    elif base.lower() in ("a", "b"):
                        stats_map[base] = "boson"
        if not stats_map:
            stats_map["__default__"] = "fermion"
        return json.dumps(stats_map)

    def prepare_sympy_wick_inputs(
        self, sympy_str: str, ctx: "SymbolContext"
    ) -> "tuple[str, str]":
        """Collect per-operator statistics and convert operator notation to
        ``cd``/``c`` form for the ``_nb_sympy_wick`` call.

        Overrides the base no-op so that shared engine code (`_wick_contract_sympy`)
        need not know it is running on Julia.  Returns
        ``(statistics_json, converted_sympy_str)``.
        """
        stats_json = self._collect_stats_json(ctx)
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.models.ir import Equation as _EqIR
        _surf = SharedParser().parse(sympy_str)
        if _surf is not None:
            _ir = JuliaSemanticResolver().resolve(_surf, ctx)
            if isinstance(_ir, _EqIR):
                _ir = _ir.rhs
            _jcg = JuliaCodeGenerator(context=ctx)
            try:
                _converted = _jcg._serialize_lhs_to_sympy(_ir, set())
                if _converted:
                    sympy_str = _converted
            except Exception:
                # Best-effort cd/c conversion; keep the original on failure.
                _logger.debug("SymPy-Wick operator conversion failed; keeping original",
                              exc_info=True)
        return stats_json, sympy_str

    def _sympy_stats_and_vars(
        self, sympy_str: str, ctx: "SymbolContext"
    ) -> "tuple[str, str]":
        """Infer statistics JSON and variable list from a SymPy canonical string."""
        import re
        stats_json = self._collect_stats_json(ctx)
        # Collect all single-letter identifiers used as indices
        vars_found = sorted(set(re.findall(r"\b([a-z])\b", sympy_str)))
        vars_found = [v for v in vars_found if v not in ("c", "a", "e", "i", "j")]
        vars_str = ",".join(vars_found) if vars_found else "k"
        return stats_json, vars_str

    def _collect_vars_from_canonical(self, canonical: str) -> str:
        """Extract momentum variable names from a canonical expression string.

        String-level fallback, kept for the paths whose input really is a SymPy
        string already (the chain path, and the combine/commutator builders).
        Callers that have just SERIALIZED an expression should scan that output via
        `_collect_vars_from_sympy_str` instead —
        this regex only sees indices written inside PARENTHESES, so for the
        braced spelling `c'_{k+q,u}` it found none and silently fell back to
        the single default "k", leaving q and p undeclared (measured
        2026-08-07).
        """
        import re
        # Index variables appear inside operator parentheses: c(k), c(-k), c(k+q)
        idx_raw = re.findall(r"[a-z_][a-z_0-9']*\(([^)]+)\)", canonical)
        vars_found: set[str] = set()
        for idx_expr in idx_raw:
            for tok in re.findall(r"[a-z][a-z0-9_]*", idx_expr):
                if len(tok) <= 2:  # short names are likely momentum vars
                    vars_found.add(tok)
        return ",".join(sorted(vars_found)) if vars_found else "k"

    def _collect_vars_from_sympy_str(
        self, sympy_str: str, extra_vars: "set[str] | None" = None
    ) -> str:
        """Momentum names for a string this module GENERATED, plus IndexSum dummies.

        Deliberately the same scan as `_collect_vars_from_canonical` and
        deliberately NOT a second implementation: the regex was never the
        defect — it was being handed the researcher's own canonical, where the
        braced spelling hides the indices from a paren-based scan.  Run over the
        serializer's output, both spellings have already become one string, so
        one scan answers for both.

        These names are declared `nonzero` on the SymPy side so that δ(k, -k)
        evaluates to 0; a momentum missing here is one SymPy treats as an
        ordinary free symbol.
        """
        found = {v for v in (extra_vars or ()) if v}
        scanned = self._collect_vars_from_canonical(sympy_str)
        if scanned:
            found.update(t for t in scanned.split(",") if t)
        return ",".join(sorted(found)) if found else "k"

