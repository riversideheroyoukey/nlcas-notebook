"""Inequality-constraint registration (LHS <op> RHS, op in <=, >=, <, >).

A sibling to H-1's equation-form registration (``ui/interpret_pipeline.py``),
kept in its own module per the R1b/R1c/R1d precedent already established in
this codebase (``ui/interpret_equation_intercepts.py``, ``ui/interpret_early.py``,
``ui/interpret_rules.py``, ``ui/interpret_l1.py``).

Unlike H-1, this does NOT define a named symbol — there is no LHS "value" to
bind, so H-1's dedup/ID-expansion/lhs_alias_map machinery is intentionally not
reused.  The canonical stored is the FULL relation string (e.g.
"2*x_A+x_B<=100"), not RHS-only.
"""
from __future__ import annotations

import re
from functools import partial

from notebook.runtime.backend_handler import beautify_bare_canonical_latex
from notebook.runtime.latex_converter import normalize_relational_ops
from notebook.runtime.shared_parser import find_top_level_relation
from ui.interpret_common import _HAS_CJK_RE, _find_expr_by_canonical

_OP_LATEX: dict[str, str] = {
    "<=": r"\leq", ">=": r"\geq", "<": "<", ">": ">",
}

# A trailing plain-ASCII instruction word leaking into the RHS, e.g.
# "x <= 100 differentiate" — mirrors H-1's own _RHS_TRAILING_WORD_RE guard.
_TRAILING_WORD_RE = re.compile(r'\s+[A-Za-z]{3,}\s*$')


async def _try_inequality_intercept(
    raw: str,
    engine: "NotebookEngine",
    loop,
    registered: list[dict],
    canonical_to_id: dict[str, str],
) -> dict | None:
    """Detect a whole-token inequality relation and register it.

    Returns ``{"latex": ..., "canonical": ...}`` on success (the caller wraps
    this into an ``InputToken(type="math", ...)``), or ``None`` to let the
    token fall through to H-1 / the normal pipeline unchanged.
    """
    if _HAS_CJK_RE.search(raw) or _TRAILING_WORD_RE.search(raw):
        return None

    normalized = normalize_relational_ops(raw)
    rel = find_top_level_relation(normalized)
    if rel is None or rel[1] in ("=", "≈"):
        # A plain equation, an approximation, or no relation at all — not our
        # concern.  `≈` is a relation but not a CONSTRAINT: there is no
        # `assume(a ≈ b)` for this path to build, and claiming it here cost the
        # researcher their own notation, because this route REBUILDS the display
        # from the canonical — `\langle \hat c^\dagger \hat c\rangle` came back
        # as `expectation(...)` on screen (measured 2026-08-21).
        return None

    try:
        canonical = engine._handler.latex_to_canonical(raw) or raw
        rel2 = find_top_level_relation(canonical)
        if rel2 is None:
            return None
        pos, op = rel2
        lhs_latex = beautify_bare_canonical_latex(canonical[:pos]).strip()
        rhs_latex = beautify_bare_canonical_latex(canonical[pos + len(op):]).strip()
        display_latex = f"{lhs_latex} {_OP_LATEX.get(op, op)} {rhs_latex}"

        existing_id = _find_expr_by_canonical(engine, canonical)
        if existing_id:
            canonical_to_id[canonical] = existing_id
            expr = engine.expressions[existing_id]
            registered.append({
                "expr_id":           expr.display_id,
                "canonical":         expr.canonical,
                "latex":             expr.latex,
                "step_id":           expr.created_by_step,
                "undefined_symbols": expr.undefined_symbols,
            })
            return {"latex": expr.latex, "canonical": expr.canonical}

        expr = await loop.run_in_executor(
            None,
            partial(engine.register_expression, canonical,
                    latex=display_latex, lhs_label=""),
        )
        canonical_to_id[canonical] = expr.display_id
        registered.append({
            "expr_id":           expr.display_id,
            "canonical":         expr.canonical,
            "latex":             expr.latex,
            "step_id":           expr.created_by_step,
            "undefined_symbols": expr.undefined_symbols,
        })
        return {"latex": expr.latex, "canonical": expr.canonical}
    except Exception:
        return None
