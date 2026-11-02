"""FREE-PARAM: general ODE solutions, Laplace definitions, specialization.

Pure rule-based parsers extracted from interpret_pipeline.py (R-10b).
run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re
from functools import partial

from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp

from ui.interpret_common import (_EXPR_ID_RE, _HAS_CJK_RE, _expand_expr_ids,
                                 assumption_display_latex)


# FREE-PARAM value substitution stated explicitly — used by
# _try_substitute_free_params: "q(0)=0, \dot{q}(0)=0 とする" / "Set q(0)=0,
# q'(0)=0 [in E5]".  Externalized to i18n/nl_patterns.json["fp_subst"] (named
# groups eid, body — the conditions in body are parsed by _parse_conditions
# and mapped to the IC free parameters).
# FREE-PARAM specialization: determine a free parameter (e.g. the integration
# constant C) from a condition F(var=point)=value, e.g. 「E2 で x=0 のとき 0
# という条件で C を決めて」 / "In E2, when x=0, determine C such that this
# equals 0" → particular antiderivative.  Externalized to
# i18n/nl_patterns.json["specialize_condition"] (named groups eid, var, point,
# value; the constant's own name (C) is not captured in either language — the
# underlying engine.specialize_by_condition call operates on "the" free
# parameter of the expression, inferred from *eid*).
def _latest_free_param_expr(engine) -> "str | None":
    active = engine._active_expr_ids()
    for eid in reversed(list(engine.expressions)):
        if eid in active and getattr(engine.expressions[eid], "free_parameters", None):
            return eid
    return None
_COND_TOKEN_RE = re.compile(
    r"(?:\\(dot|ddot|dddot)\{(\w+)\}|(\w+)('*))"
    r"\s*\(\s*([^)]*?)\s*\)\s*=\s*([^,，、]+)")
_DOT_ORDER = {"dot": 1, "ddot": 2, "dddot": 3}
# "E1 を q(0)=0, q'(0)=v0 で特殊化して" (imperative) / "E2 の初期条件を x(0)=1,
# x'(0)=0 とする" (declaration — the initial/boundary distinction is for the
# researcher's own clarity only, the solve step doesn't need to know which
# kind of condition it is) / English: "determine the constants in E5 from
# q(0)=0, q'(0)=v0" / "specialize E5 with ..." / "set the initial/boundary
# conditions of E5 to ...".  Externalized to
# i18n/nl_patterns.json["specialize_multi"] — per-branch group names
# (eid1/cond1, eid2/cond2, ...) since Python's re forbids reusing a group
# name across alternation branches; extract via .groupdict().get(...).
def _parse_conditions(text: str) -> "list[dict]":
    """Parse derivative/value conditions like ``q(0)=0, q'(0)=v0`` or
    ``\\dot{q}(0)=v0`` into ``[{order, point, value, depvar}]`` (order 0 = value)."""
    conds: list[dict] = []
    for m in _COND_TOKEN_RE.finditer(text):
        dot, dot_name, prime_name, primes, point, value = m.groups()
        if dot:
            order, name = _DOT_ORDER[dot], dot_name
        else:
            order, name = len(primes or ""), prime_name
        conds.append({"order": order, "point": point.strip(),
                      "value": value.strip(), "depvar": name})
    return conds
async def _try_specialize_conditions(engine, loop, lang, text, registered) -> "dict | None":
    """Specialize an equation-form general solution by SEVERAL conditions on the
    dependent variable and its derivatives (FREE-PARAM) — e.g.
    ``E5 を q(0)=0, q'(0)=v0 で特殊化して`` fixes k₁,k₂.  Generalizes the single-
    parameter specialize to N conditions / equation form; reused for ODE and (later)
    Laplace results.  Rule-based, before LLM."""
    m = nlp("specialize_multi", lang).match(text)
    if m is None:
        return None
    _sm_gd = m.groupdict()
    eid = _sm_gd.get('eid1') or _sm_gd.get('eid2') or _sm_gd.get('eid3')
    cond_text = _sm_gd.get('cond1') or _sm_gd.get('cond2') or _sm_gd.get('cond3')
    if not (eid in engine.expressions and eid in engine._active_expr_ids()
            and getattr(engine.expressions[eid], "free_parameters", None)):
        return None
    conds = _parse_conditions(cond_text)
    if not conds:
        return None
    depvar = conds[0].get("depvar")
    try:
        ex = await loop.run_in_executor(None, partial(
            engine.specialize_by_conditions, eid, conds, depvar, None))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id,
                       "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "specialize_by_conditions",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
async def _try_substitute_free_params(engine, loop, lang, text, registered) -> "dict | None":
    r"""Fix a free parameter by VALUE (FREE-PARAM instance 4).

    The spelling is `fp_subst` in i18n/nl_patterns.json; it is not repeated
    here, because a second copy in a docstring is a second place to keep in
    step (and the NL ratchet reads the first line of a raw docstring as a
    pattern literal).

    The intercept only fires when the named parameters really are free
    parameters of the target expression (else it falls through to e.g. H-1
    registration).

    **The initial-condition form used to be handled here and is not any more.**
    It built the names ``q_ic0`` / ``q_ic1`` from ``q(0)=0, \dot q(0)=0 とする``,
    and the only thing that ever PRODUCED free parameters with those names was
    ``laplace_by_definition``'s integration-by-parts branch, deleted with the
    macro on 2026-08-15 (SCENARIO-MACRO case 7).  So the branch could no longer
    match anything, and a spelling that had worked the day before answered
    "Could not interpret" (open_items IC-NL-ORPHANED).

    Measured 2026-08-16, what survives is the same CONDITIONS written with a
    TARGET — ``E5 の初期条件を q(0)=0, \dot q(0)=0 とする`` and ``E5 を
    q(0)=q0, q'(0)=v0 で特殊化して`` both run ``specialize_by_conditions``.  So
    the conditions syntax is not the orphan; the BARE form is, and it is bare
    that is the problem — there is no target to apply it to, and guessing one
    is exactly the kind of invention this system refuses elsewhere.  Rather
    than fall through to a message that explains nothing, it is refused here
    with the two spellings that work.
    """
    m = nlp("fp_subst", lang).match(text)
    if m is None:
        return None
    eid_m, body = m.group("eid"), m.group("body")
    conds = _parse_conditions(body)
    if conds and all(c["point"] in ("0", "０") for c in conds) and not eid_m:
        # The BARE form — `q(0)=0, \dot q(0)=0 とする`, no target named.  The
        # researcher writes it (reported 2026-08-16), so it is ROUTED, not
        # refused: the target is the latest expression carrying free
        # parameters, which is the rule this very function already uses for the
        # bare `C=0 とする`.  Answering the two forms differently would be a
        # second answer to one question.
        #
        # ONLY the bare form is handled here.  Falling through is load-bearing
        # for the rest: `E5 の初期条件を q(0)=0 … とする` also matches
        # `fp_subst`, and the interceptor that owns it (`specialize_multi`)
        # runs LATER — claiming everything with conditions swallowed the
        # spelling that works (measured while writing this).
        _ic_eid = _latest_free_param_expr(engine)
        if not _ic_eid:
            # Nothing to apply them to: say so, with the spellings that do work.
            return {"status": "failed",
                    "error": t("free_params.bare_initial_conditions", lang,
                               conditions=body.strip()),
                    "registered_expressions": registered}
        try:
            ex = await loop.run_in_executor(None, partial(
                engine.specialize_by_conditions, _ic_eid, conds,
                conds[0].get("depvar"), None))
        except Exception as _e:
            return {"status": "failed", "error": str(_e),
                    "registered_expressions": registered}
        registered.append({"expr_id": ex.display_id,
                           "canonical": ex.canonical, "latex": ex.latex})
        return {"status": "completed", "operation": "specialize_by_conditions",
                "expr_id": ex.display_id, "latex": ex.latex,
                "canonical": ex.canonical,
                "registered_expressions": registered}
    # plain "name=value[, name=value]" — free variables (no derivative/point)
    plain = re.findall(r'([A-Za-z]\w*)\s*=\s*([^,，、]+)', body)
    if not plain:
        return None
    pairs = [(nm, val.strip()) for nm, val in plain]
    eid = eid_m or _latest_free_param_expr(engine)
    if not (eid and eid in engine.expressions and eid in engine._active_expr_ids()):
        return None
    free = set(engine.expressions[eid].free_parameters or [])
    if not all(p in free for p, _ in pairs):
        return None
    try:
        ex = await loop.run_in_executor(None, partial(
            engine.substitute_free_parameters, eid, pairs))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id,
                       "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "substitute_free_parameters",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
async def _try_specialize_condition(engine, loop, lang, text, registered) -> "dict | None":
    """Specialize a general solution by a condition F(var=point)=value, fixing a
    free parameter (FREE-PARAM).  Operates on a referenced E-ID or the latest
    expression that carries free parameters."""
    m = nlp("specialize_condition", lang).match(text)
    if m is None:
        return None
    eid = m.group('eid')
    var, point, value = m.group('var'), m.group('point'), m.group('value')
    eid = eid or _latest_free_param_expr(engine)
    if not (eid and eid in engine.expressions and eid in engine._active_expr_ids()
            and getattr(engine.expressions[eid], "free_parameters", None)):
        return None
    try:
        ex = await loop.run_in_executor(None, partial(
            engine.specialize_by_condition, eid, var, point, value))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id,
                       "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "specialize_by_condition",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
# "E3 を一般に解いて" / "solve E3 in general" / "find the general solution of
# E3 [for q]" / "solve the differential equation E3 [for q]".  Externalized to
# i18n/nl_patterns.json["ode_general"] — per-branch group names (eid1/var1,
# eid2/var2, ...) since Python's re forbids reusing a group name across
# alternation branches; extract via .groupdict().get(...).
async def _try_solve_ode_general(engine, loop, lang, text, registered) -> "dict | None":
    """General solution of a linear ODE (FREE-PARAM): the result carries its
    arbitrary constants (``k_1``, ``k_2`` / ``C``) as named free parameters that a
    later condition specializes.  Rule-based, before LLM and before the algebraic
    solve handler (so "一般に解いて" is not mis-routed to solve()).  The dependent /
    independent variables are inferred from the session's time-dependent quantities,
    or taken from an explicit "… を q について …" / "… for q …"."""
    m = nlp("ode_general", lang).match(text)
    if m is None:
        return None
    _og_gd = m.groupdict()
    eid = _og_gd.get('eid1') or _og_gd.get('eid2') or _og_gd.get('eid3') or _og_gd.get('eid4')
    depvar = _og_gd.get('var1') or _og_gd.get('var2') or _og_gd.get('var3') or _og_gd.get('var4')
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    try:
        ex = await loop.run_in_executor(None, partial(
            engine.solve_ode_general, eid, depvar, None))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id,
                       "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "solve_ode_general",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
# General "assume <condition>": adds a CAS assumption (assume(condition)) that
# the researcher states explicitly and visibly as its own step — e.g. before
# solving a 2nd-order ODE whose characteristic-root sign Maxima would
# otherwise ask about interactively (hanging the persistent adapter; see
# solve_ode_general's docstring in maxima_backend_handler.py). Deliberately
# general and un-opinionated: WHICH condition to assume (e.g. underdamped vs.
# overdamped) is the researcher's modeling choice — this rule only wires an
# existing operation (add_assumption) to natural language, it never chooses
# or defaults a condition itself.
# "<condition> と仮定する" / "assume <condition>" → add_assumption.
# Externalized to i18n/nl_patterns.json["assume_condition"] (named group
# condition).
async def _try_add_assumption(engine, loop, lang, text, registered) -> "dict | None":
    """Rule-based: "<condition> と仮定する" / "assume <condition>" → add_assumption.

    <condition> may be a bare E-id (its own canonical/latex is used) or a raw
    math expression (converted via latex_to_canonical, same as other rule-based
    handlers in this module)."""
    m = nlp("assume_condition", lang).match(text)
    if m is None:
        return None
    condition_raw = m.group("condition").strip()
    if not condition_raw:
        return None
    if _EXPR_ID_RE.match(condition_raw):
        # Bare E-id alone ("E1 と仮定する"): use its own canonical/latex.
        _cond_expr = engine.expressions.get(condition_raw)
        if _cond_expr is None or condition_raw not in engine._active_expr_ids():
            return None
        condition_canonical = _cond_expr.canonical
        condition_display = _cond_expr.latex or _cond_expr.canonical
    else:
        # General condition, possibly with an E-id EMBEDDED in a larger
        # expression ("E1 > 0 と仮定する") — expand it to E1's actual value
        # (H-2 style) before converting to canonical, so the assumption sent
        # to the CAS is about the real expression, not the literal token "E1"
        # (which Maxima would otherwise treat as an undefined bare symbol).
        # The DISPLAY keeps the user's original E-id reference (more legible
        # than the fully expanded form), matching H-2 convention elsewhere.
        _expanded = _expand_expr_ids(condition_raw, engine)
        condition_canonical = engine._handler.latex_to_canonical(_expanded) or _expanded
        # With an E-id in it the raw text is kept deliberately (above); with
        # none there is no reference to preserve, and the raw text is not
        # display latex at all.
        condition_display = (
            condition_raw if _expanded != condition_raw
            else assumption_display_latex(condition_raw))
    try:
        _, asm = await loop.run_in_executor(None, partial(
            engine.add_assumption, condition_canonical, condition_display))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    return {"status": "assumption_added", "operation": "add_assumption",
            "assumption_id": asm.assumption_id, "condition": condition_display,
            "registered_expressions": registered}
# "q のラプラス変換を Q とおく" / "Let Q be the Laplace transform of q" —
# English word order names the CAPITAL (s-domain) symbol first and the base
# (time-domain) symbol last, the reverse of the Japanese pattern; externalized
# to i18n/nl_patterns.json["laplace_define"] with named groups base/cap so the
# language-specific word order no longer needs to be handled in Python.
async def _try_define_laplace(engine, loop, lang, text, registered) -> "dict | None":
    """Lay out a Laplace-transform definition card: "q のラプラス変換を Q とおく" →
    Q(s) = ∫₀^∞ q(t) e^{-st} dt.  Anchors the s-domain symbol to its definition so
    the subsequent transform does not use a magic symbol.  Rule-based, before LLM."""
    m = nlp("laplace_define", lang).match(text)
    if m is None:
        return None
    base, cap = m.group("base"), m.group("cap")
    try:
        ex = await loop.run_in_executor(None, partial(
            engine.define_laplace_transform, base, cap))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id,
                       "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "define_laplace_transform",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
