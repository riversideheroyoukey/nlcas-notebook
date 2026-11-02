"""N-1: natural-language trigger for the deterministic numeric-fork path.

Deliberately narrow (open_items.md: "数値フォークの NL トリガー... 唯一の未着手の
機能残"): a single sweep variable over an expression that has EXACTLY that one free
symbol.  Anything needing multiple variables, fixed-value substitution, or LLM
codegen falls through (returns ``None``) to the existing numeric-fork modal —
this is the deterministic-template ("G1" whitelist) path only, so the confirm
click IS the approval, with no separate review gate.
"""
from __future__ import annotations

from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from ui.interpret_plan import _ZENKAKU_DIGITS, _find_target

_DEFAULT_POINTS = 200


def _to_float(raw: "str | None") -> "float | None":
    if raw is None:
        return None
    try:
        return float(raw.translate(_ZENKAKU_DIGITS))
    except ValueError:
        return None


async def _try_numeric_fork_nl(engine, loop, lang, text, registered) -> "dict | None":
    """Explicit single-variable sweep ask → deterministic-template numeric
    fork, confirmed by click (mirrors the needs_confirmation shape used by
    _try_plan_candidates/_try_plan_freeform, ui/interpret_plan.py)."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    m = nlp("numeric_fork_sweep_nl", lang).match(stripped)
    if m is None:
        return None

    target = m.group("target") or _find_target(engine, stripped)
    if not target or target not in engine.expressions:
        return None
    expr = engine.expressions[target]
    if (expr.canonical or "").lstrip().startswith("["):
        return None  # list/matrix canonical — not a scalar sweep target

    var = m.group("var")
    vmin = _to_float(m.group("min"))
    vmax = _to_float(m.group("max"))
    if vmin is None or vmax is None:
        return None
    points_raw = m.group("points")
    points = int(points_raw.translate(_ZENKAKU_DIGITS)) if points_raw else _DEFAULT_POINTS

    free = await loop.run_in_executor(
        None, engine._numeric_free_symbols, expr.canonical)
    if free is None or var not in free or len(free) != 1:
        return None  # multi-variable / unlowerable — defer to the modal

    args = {
        "expr_id": target,
        "problem_type": "parameter_sweep",
        "target_quantity": "",
        "sweep_parameters": {var: {"min": vmin, "max": vmax, "points": points}},
        "fixed_values": {},
    }
    return {
        "status": "needs_confirmation",
        "operation": "numeric_fork_nl",
        "args": args,
        "confirmation_prompt": t("numeric.nl_fork_confirm_prompt", lang,
                                 var=var, min=vmin, max=vmax, points=points,
                                 expr_id=target),
        "registered_expressions": registered,
    }
