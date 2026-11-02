r"""EXTREMUM-BINDER: `\max_i x_i` as a display-only object.

An extremum over a bound index has no backend command — Maxima's `lmax` needs a
concrete list, so a symbolic range has none at all (measured 2026-08-16).  Read
as an ordinary subscript it lowered to ``max(i) * x(i)``, and Maxima's ``max``
of one argument IS that argument, so ``\max_i x_i`` computed as ``i·x(i)``:
valid CAS text, a different quantity, silent.

Refusing it outright was tried the same day and was worse in a different way —
the notation stopped being WRITABLE, and `\max_i` / `\sup_x` are everyday
notation in optimisation, analysis, statistics and machine learning.  So this
interceptor claims the raw text before the general pipeline and registers a
display-only object, exactly as `_try_integral_input` does for a range-bearing
definite integral: **written down and shown now, refused at the point of USE**
(`operand_guards`).

The claim is deliberately narrow — a binder needs a SUBSCRIPT.  `\max(a, b)` is
an ordinary two-argument call that already works, and `\max z = c_1x + c_2y`
(an LP written in prose) has no subscript either; neither is claimed.
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.extremum_form import build_extremum_latex, parse_latex_extremum
from notebook.runtime.nl_patterns import nlp


async def _try_extremum_input(engine, loop, lang, text, registered) -> "dict | None":
    r"""Register `\max_i x_i` / `\sup_{x \in X} f(x)` as display-only."""
    desc = parse_latex_extremum(text)
    if desc is None:
        return None
    latex = build_extremum_latex(desc)
    expr = await loop.run_in_executor(
        None, partial(engine.register_extremum_form, latex, desc))
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "registered", "operation": "register_extremum",
            "expr_id": expr.display_id, "latex": expr.latex,
            "registered_expressions": registered}


def _latest_extremum_form(engine) -> "str | None":
    """The most recent extremum object on the active branch, or None."""
    active = engine._active_expr_ids()
    for eid in reversed(list(engine.expressions)):
        if eid in active and getattr(engine.expressions[eid],
                                     "extremum_descriptor", None):
            return eid
    return None


async def _try_evaluate_extremum(engine, loop, lang, text, registered) -> "dict | None":
    r"""Evaluate an extremum object (NL pattern ``evaluate_extremum``).

    The spellings live in `i18n/nl_patterns.json` and are NOT repeated here —
    a second copy in a docstring is a second place to keep in step.

    A CONCRETE integer range has a command (`lmax(makelist(…))`); a symbolic
    one does not, measured — so this evaluates the first and fails with a
    reason for the second, rather than sending something the CAS rejects.

    **Declining is narrow on purpose.**  The phrase also names integrals and
    ordinary compute-asks, so this fires ONLY when the target really is an
    extremum object; everything else falls through untouched.
    """
    m = nlp("evaluate_extremum", lang).match(text)
    if m is None:
        return None
    tgt = m.group("target") or ""
    if tgt.startswith("E"):
        eid = tgt if (tgt in engine.expressions
                      and tgt in engine._active_expr_ids()) else None
    else:
        eid = _latest_extremum_form(engine)
    if eid is None or not getattr(engine.expressions.get(eid),
                                  "extremum_descriptor", None):
        return None                      # not ours — an integral, or a plain compute
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.evaluate_extremum_form, eid))
    except Exception as _e:
        # Broad on purpose: a symbolic range and a CAS refusal both have to
        # surface as a failed step carrying their reason.
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "completed", "operation": "evaluate_extremum",
            "expr_id": expr.display_id, "latex": expr.latex,
            "canonical": expr.canonical,
            "registered_expressions": registered}
