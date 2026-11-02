"""S-o: definite-integral objects, change of variables, evaluation.

Pure rule-based parsers extracted from interpret_pipeline.py (R-10b).
run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re
from functools import partial

from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.shared_parser import split_relation

from ui.interpret_common import (
    _EXPR_ID_RE, _HAS_CJK_RE, FULLWIDTH_DIGITS, math_operand, number_word,
    pick_group, taylor_center_note,
)



# Definite integral with explicit bounds, e.g.
#   「E3 を r で 0 から 1 まで定積分して」 / 「E4 を theta で 0 から 2π まで定積分して」
# Rule-based (no LLM) — definite integration over given limits is routine and must
# not depend on a model to parse the variable/bounds.  Distinct from the
# change-of-variables grammar (which needs へ … で変数変換); this needs まで定積分.
def _parse_new_var_spec(parts: "list[str]", engine) -> "tuple[list[str], list[list[str]] | None]":
    """Parse a new-variable spec list.  Each item is either a bare name (`r`) or a
    name with an explicit range (`r:0..1`, `theta:0..2π`).  Returns (names, bounds)
    where bounds is OUTER → INNER as listed, or None when no item carries a range.
    Ranges are normalized for the backend (π → %pi)."""
    names: list[str] = []
    bounds: list[list[str]] = []
    all_ranged = True
    norm = engine._handler.normalize_bound
    for p in parts:
        if ":" in p and ".." in p:
            name, rng = p.split(":", 1)
            lo, hi = rng.split("..", 1)
            names.append(name.strip())
            bounds.append([name.strip(), norm(lo.strip()), norm(hi.strip())])
        else:
            names.append(p.strip())
            all_ranged = False
    return names, (bounds if (all_ranged and bounds) else None)
def _llm_derive_bounds(llm, desc: dict, old_vars: list, new_vars: list,
                       transform: list) -> "list[list[str]] | None":
    """Ask the LLM for the new integration bounds of a change of variables (the
    fallback between the auto-table and the user override).  Returns bounds
    (OUTER → INNER, Maxima constants) or None on any failure."""
    import json as _json
    region = desc.get("region")
    old_bounds = desc.get("bounds")
    domain = (f"region {region}" if region else f"bounds {old_bounds}")
    system = (
        "You are a calculus assistant. Given a multiple definite integral and a "
        "change of variables, return ONLY the NEW integration bounds as a JSON list "
        "of [variable, lower, upper], OUTERMOST integral first. Use Maxima syntax for "
        "constants (pi is %pi). Output JSON only, no prose.")
    user = (
        f"Old integration domain: {domain}\n"
        f"Old variables: {old_vars}\nSubstitution: "
        f"{[f'{o}={t}' for o, t in zip(old_vars, transform)]}\n"
        f"New variables (outer..inner): {new_vars}\nJSON:")
    try:
        resp = llm.chat(system, user)
        if not getattr(resp, "ok", False) or not resp.text:
            return None
        txt = resp.text.strip()
        m = re.search(r"\[.*\]", txt, re.DOTALL)
        data = _json.loads(m.group(0) if m else txt)
        bounds = [[str(v).strip(), str(lo).strip(), str(hi).strip()]
                  for v, lo, hi in data]
        return bounds if len(bounds) == len(new_vars) else None
    except Exception:
        return None
def _latest_vector_expr(engine) -> "str | None":
    r"""Most recent live expression that HAS something to open into components.

    Targeted rather than "the latest expression": `成分に展開して` with no E-id
    should reach for the `\int d^3k …` the researcher is looking at, not for
    whatever happened to be registered last.
    """
    from notebook.runtime.vector_components import has_openable
    active = engine._active_expr_ids()
    for eid in reversed(list(engine.expressions)):
        if eid in active and has_openable(engine.expressions[eid].ir_ast,
                                          engine.symbol_context):
            return eid
    return None


def _latest_integral_form(engine) -> "str | None":
    """Most recent live expression carrying an integral_descriptor."""
    active = engine._active_expr_ids()
    for eid in reversed(list(engine.expressions)):
        if eid in active and getattr(engine.expressions[eid], "integral_descriptor", None):
            return eid
    return None
def _split_top_commas(s: str) -> "list[str]":
    """Split on top-level commas, ignoring those inside (), [], {} — protects
    function-argument commas such as atan2(v, u)."""
    parts, depth, buf = [], 0, []
    for ch in str(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return [p for p in parts if p]
async def _try_change_vars(engine, loop, lang, text, registered, supervisor=None) -> "dict | None":
    """Multivariable change of integration variables (S-o).  Parses the old/new
    variable lists and the substitution rules from NL and dispatches
    `change_integration_variables` (Jacobian determinant auto-inserted).  The
    transform may be written as equations (`x=r*cos(theta), y=r*sin(theta)`) or as
    a bare RHS list ordered to match old_vars (`r*cos(theta), r*sin(theta)`)."""
    m = nlp("change_vars", lang).match(text)
    if m is None:
        return None
    target  = m.group("expr").strip()
    old_raw = m.group("oldv").strip()
    new_raw = m.group("newv").strip()
    trans_raw = m.group("trans").strip()
    # E-ID target must be a live expression; a literal integrand (`1`, `x^2+y^2`)
    # is used verbatim (pure volume element / un-registered integrand).
    if _EXPR_ID_RE.match(target):
        if target not in engine.expressions or target not in engine._active_expr_ids():
            return None
    elif _HAS_CJK_RE.search(target):
        # A literal integrand must be a math expression, not stray natural language.
        return None
    # `\theta` and `theta` are the same variable to the researcher — and the
    # DISPLAY of this very workflow writes `d\theta`, so writing the TeX form
    # back is the natural thing to do.  Left unconverted, the new-variable names
    # miss the standard-coordinate table (polar × disk), so the automatic bounds
    # never fire and the researcher is asked to supply what the table already
    # knows (measured 2026-08-01: plain `r,theta` → the full ∫₀^{2π}∫₀¹ r³, but
    # `r,\theta` → "needs_bounds").  The conversion is idempotent.
    _canon = lambda _s: math_operand(engine, _s)
    old_vars = [_canon(v) for v in _split_top_commas(old_raw)]
    new_raw_parts = _split_top_commas(new_raw)
    trans_parts = [_canon(p) for p in _split_top_commas(trans_raw)]
    if not (old_vars and new_raw_parts and trans_parts):
        return None
    # New-var spec may carry explicit bounds: `r:0..1, theta:0..2π` → names + bounds.
    new_vars, new_bounds = _parse_new_var_spec(new_raw_parts, engine)
    new_vars = [_canon(v) for v in new_vars]
    # If the transform is given as equations (old = rhs), reorder the RHS to match
    # old_vars; otherwise the parts are already a positional RHS list.
    # "is every part an equation" decides between the named and the positional
    # transform spec.  No realistic input separates the two readings — a bracketed
    # `=` would have to sit in the OLD variable's name — so this is the one
    # definition rather than a measured fix.
    _rels = [split_relation(p) for p in trans_parts]
    if all(r is not None and "=" in r[1] for r in _rels):
        rhs_map: dict[str, str] = {}
        for r in _rels:
            rhs_map[r[0]] = r[2]
        if set(rhs_map) != set(old_vars):
            return None
        transform = [rhs_map[o] for o in old_vars]
    else:
        transform = trans_parts
    if len(transform) != len(old_vars):
        return None

    # Integral-form target: rewrite the whole definite integral (integrand·J + new
    # bounds), staying unevaluated.  New bounds come from the user spec, else a
    # standard-coordinate table, else we ask the user to supply them.
    if _EXPR_ID_RE.match(target) and getattr(
            engine.expressions.get(target), "integral_descriptor", None):
        from notebook.runtime.integral_form import classify_transform, derive_new_bounds
        desc = engine.expressions[target].integral_descriptor or {}
        # New-bound resolution order:
        #   1. explicit user override (`r:0..1, theta:0..2π`) — always wins,
        #   2. automatic table (standard coordinate change × standard region),
        #   3. LLM proposal (when an llm_client is configured),
        #   4. otherwise ask the user to supply the bounds.
        bounds = new_bounds
        if bounds is None:
            kind = classify_transform(old_vars, transform)
            bounds = derive_new_bounds(kind, desc.get("region"), new_vars) if kind else None
        if bounds is None and supervisor is not None and getattr(supervisor, "_llm", None):
            bounds = await loop.run_in_executor(None, partial(
                _llm_derive_bounds, supervisor._llm, desc, old_vars, new_vars, transform))
        if bounds is None:
            return {"status": "needs_bounds", "operation": "change_integration_variables",
                    "message": t("integral.need_new_bounds", lang),
                    "new_vars": new_vars, "registered_expressions": registered}
        try:
            expr = await loop.run_in_executor(None, partial(
                engine.change_variables_integral_form,
                target, old_vars, new_vars, transform, bounds))
        except Exception as _e:
            return {"status": "failed", "error": str(_e),
                    "registered_expressions": registered}
        registered.append({"expr_id": expr.display_id,
                           "canonical": expr.canonical, "latex": expr.latex})
        return {"status": "registered", "operation": "change_integration_variables",
                "expr_id": expr.display_id, "latex": expr.latex,
                "registered_expressions": registered}

    args = {"expr": target, "old_vars": ", ".join(old_vars),
            "new_vars": ", ".join(new_vars), "transform": ", ".join(transform)}
    try:
        step, val = await loop.run_in_executor(
            None, partial(engine.run_operation, "change_integration_variables", args))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    oid = step.output_refs[0] if step.output_refs else None
    oe = engine.expressions.get(oid) if oid else None
    return {"status": "completed", "operation": "change_integration_variables",
            "expr_id": oid, "step_id": step.step_id,
            "latex": oe.latex if oe else "",
            "canonical": oe.canonical if oe else val,
            "operation_label": step.visible_action,
            "registered_expressions": registered}
async def _try_integral_in_coordinates(engine, loop, lang, text, registered) -> "dict | None":
    r"""`integrate X in polar coordinates` / `X を極座標で積分して` —— 断って道を示す。

    `in <名前> coordinates` は**副詞句であって「前置詞＋変数」ではない**。共有の
    前置詞表が `in` を持つ（`integrate f in x` は読めるべき綴り）ので、`polar` が
    変数の欄に入り、研究者が変数として書いていない記号が答えに立っていた——実測
    2026-09-18、極・球・円筒の3系で `polar*x^2+C` などを `completed`（HEAD でも同じ）。
    日本語は同じ綴りが**声を上げて**失敗しており、英語だけが黙っていた。

    **判別は語彙ではなく性質**——`coordinates` / `座標` が後ろに付く語は座標系である。
    語彙の表にすると、研究者が `define_coordinate_system` で定義した名前が漏れる。

    **案内する道は実測したもの**（日英とも `%pi/2` に落ちる、`scenario_coordinate_jacobian`
    の綴り）。起票のとき私が書いた道（`convert E1 to polar coordinates`）は**測ると
    別のものだった**——`to_polar` は複素数の極形式で、`polarform(...)` を返す。
    """
    m = nlp("integral_in_coordinates", lang).match(text)
    if m is None:
        return None
    return {"status": "failed",
            "error": t("integral.coordinate_change_spelling", lang,
                       system=m.group("system").strip(),
                       integrand=m.group("integrand").strip()),
            "registered_expressions": registered}


async def _try_definite_integral(engine, loop, lang, text, registered) -> "dict | None":
    """Definite integral over explicit bounds — `<expr> を <var> で <a> から <b>
    まで定積分して`.  Rule-based so the bounds are parsed deterministically (no LLM);
    closes the change-of-variables loop (e.g. ∫₀¹ r³ dr).  Target is an E-ID or a
    literal integrand; unicode/TeX π in bounds is normalized for the backend."""
    m = nlp("def_integral", lang).match(text)
    if m is None:
        return None
    target = m.group("integrand").strip()
    var = m.group("var").strip()
    lower = m.group("lower").strip()
    upper = m.group("upper").strip()
    if _EXPR_ID_RE.match(target):
        if target not in engine.expressions or target not in engine._active_expr_ids():
            return None
    elif _HAS_CJK_RE.search(target):
        return None
    else:
        # The researcher writes back what the notebook DISPLAYED, which is TeX.
        # Passing it on raw made `1/\sqrt{\xi^2+\Delta^2} を xi で … 定積分して`
        # fail with "IndexedSymbol index must resolve to a Symbol or IndexSymbol"
        # — an internal parser message, for an expression the plain-ascii
        # spelling of which works fine (measured 2026-08-01).  The conversion is
        # idempotent, so an already-canonical integrand is unchanged.
        target = math_operand(engine, target)
    # Same for the variable: `\xi` and `xi` are the same variable to the
    # researcher, and only one of them used to reach the CAS.
    var = math_operand(engine, var)
    # An integration variable must be an ATOM.  An undeclared subscripted name
    # is read as an INDEX in an operation (CLAUDE.md), so `x_1` lowers to `x(1)`
    # and Maxima rejects `integrate(x(1)^2, x(1), 0, 1)` — measured 2026-08-02,
    # reported to the researcher as the bare "Maxima evaluation error", which
    # says neither what is wrong nor that DECLARING the name fixes it (measured:
    # declared, `x_1` stays `x_1` and the integral returns 1/3).  The check is
    # narrow on purpose — this operation's requirement is unambiguous and known
    # right here, unlike a blanket pre-CAS guard (JULIA-INDEXED-SCALAR).
    _lowered = getattr(engine._handler.to_backend_code(var), "code", var)
    if "(" in (_lowered or ""):
        # The BASE is named too, because declaring it as a vector is the other
        # way out and the message used to offer only one.  Measured 2026-08-17,
        # both work and they differ: a vector base keeps `k_x` a COMPONENT
        # (`k[x]`, which Maxima integrates happily — so "the CAS cannot
        # integrate an indexed quantity" was never the real reason), while
        # declaring the whole name makes it an ordinary atom.  For a researcher
        # who arrived here from `\int d^3k`, the vector is the natural one.
        return {"status": "failed",
                "error": t("integral.var_is_indexed", lang, var=var,
                           base=var.rsplit("_", 1)[0] or var,
                           lowered=_lowered),
                "registered_expressions": registered}
    norm = engine._handler.normalize_bound
    args = {"expr": target, "var": var,
            "lower": norm(lower), "upper": norm(upper)}
    try:
        step, val = await loop.run_in_executor(
            None, partial(engine.run_operation, "integrate_definite", args))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    oid = step.output_refs[0] if step.output_refs else None
    oe = engine.expressions.get(oid) if oid else None
    return {"status": "completed", "operation": "integrate_definite",
            "expr_id": oid, "step_id": step.step_id,
            "latex": oe.latex if oe else "",
            "canonical": oe.canonical if oe else val,
            "operation_label": step.visible_action,
            "registered_expressions": registered}
async def _try_integral_input(engine, loop, lang, text, registered) -> "dict | None":
    """Register a display-only unevaluated definite integral (S-o step 1) from
    either LaTeX integral notation (``\\int_a^b\\int_c^d f \\, dx \\, dy``) or a
    region predicate (``x^2+y^2<=1 の領域で f を定積分``).  The CAS is not invoked."""
    from notebook.runtime.integral_form import (
        parse_latex_definite_integral, build_integral_latex, latex_var,
    )
    desc = None
    region_latex = None
    if "\\int" in text or "\\iint" in text:
        desc = parse_latex_definite_integral(text)
        if desc is not None:
            # A descriptor's bounds are BACKEND text: `derive_new_bounds` writes
            # `2*%pi`, `_try_definite_integral` above normalizes, and the display
            # converts back (`format_constant_for_latex`).  This site — the one
            # that reads bounds the RESEARCHER typed — was the only one not
            # walking through that door, so `\int_0^{2\pi} r \, d\theta` reached
            # Maxima as `integrate((r), theta, 0, 2\pi)` and came back as `_res_`,
            # THE ADAPTER'S OWN SCRATCH NAME, registered as a success.  Measured
            # 2026-08-16: `\int_{-\infty}^{\infty} e^{-x^2} dx` answered
            # `sqrt(%pi)*erf(infty)` — the Gaussian integral, wrong, as a success
            # — and the published `\int_0^{2\pi}\int_0^1 r\,dr\,d\theta` gave
            # `_res_` when typed back in, though the change-of-variables route
            # that PRINTS it evaluates fine (its bounds come from the table).
            norm = engine._handler.normalize_bound
            # Keep what was WRITTEN for the display before normalizing for the
            # CAS — `\frac{\pi}{2}` has no way back from `((%pi)/(2))`.
            desc["bounds_latex"] = [[lo, hi] for _v, lo, hi in desc["bounds"]]
            desc["bounds"] = [[v, norm(lo), norm(hi)] for v, lo, hi in desc["bounds"]]
            desc["region"] = {"kind": "rectangle"}
    if desc is None:
        m = nlp("region_integral", lang).match(text)
        if m is None:
            return None
        lhs       = m.group("lhs").strip()
        _op       = m.group("op").strip()
        rhs       = m.group("rhs").strip()
        integrand = m.group("integrand").strip()
        # Disk/ball: x^2+y^2(+z^2) <= C → radius from C (1→1, a^2→a, else sqrt(C)).
        norm = re.sub(r"\s|\*", "", lhs)
        if norm in ("x^2+y^2", "y^2+x^2"):
            kind, rkind, isign, diff = "disk", "x^2+y^2", r"\iint", r"\, dx \, dy"
        elif norm in ("x^2+y^2+z^2",):
            kind, rkind, isign, diff = ("ball", "x^2+y^2+z^2", r"\iiint",
                                        r"\, dx \, dy \, dz")
        else:
            return None
        rstr = rhs.strip()
        mm = re.fullmatch(r"(.+?)\^2", rstr)
        radius = "1" if rstr == "1" else (mm.group(1) if mm else f"sqrt({rstr})")
        desc = {"integrand_latex": integrand,
                "integrand_canonical": engine._handler.latex_to_canonical(integrand),
                "bounds": [], "region": {"kind": kind, "radius": radius}}
        region_latex = rf"{isign}_{{{rkind} \le {rhs}}} ({integrand}) {diff}"
    if desc.get("integrand_canonical") is None:
        desc["integrand_canonical"] = engine._handler.latex_to_canonical(
            desc["integrand_latex"])
    # `format_const` is not optional here.  The bounds are BACKEND text by the
    # time they reach the descriptor (above), so without it the display showed
    # `\int_{0}^{2*%\pi}` — CAS syntax, and mangled, because the display
    # normalizer then read the bare `pi` and added a backslash of its own.  The
    # change-of-variables path passes it; this one did not, so the same integral
    # was written down two ways depending on how it got here (measured
    # 2026-08-16).
    latex = region_latex or build_integral_latex(
        desc, format_const=engine._handler.format_constant_for_latex)
    expr = await loop.run_in_executor(
        None, partial(engine.register_integral_form, latex, desc))
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "registered", "operation": "register_integral",
            "expr_id": expr.display_id, "latex": expr.latex,
            "registered_expressions": registered}
async def _try_evaluate_integral(engine, loop, lang, text, registered) -> "dict | None":
    """Evaluate an unevaluated definite integral (S-o step 4): `E_n を評価して` /
    `この定積分を評価して`.  Only fires for an integral-form target.

    **Declining is not free.**  Measured 2026-08-03 (EVAL-INTEGRAL-EN-FALLTHROUGH):
    with no integral form present, `evaluate this integral` fell through to the
    alias scan, where the bare noun `integral` is an alias of
    ``integrate_indefinite`` — and its positional field mapper took the word
    **`evaluate` as the integration variable**, producing
    ``\\int x^2+1 \\, devaluate = evaluate(x^2+1) + C`` with ``status=completed``.
    A sentence naming a thing that does not exist came back as a registered
    expression.  Japanese did not do this only because its counterpart
    (`積分を計算して`) landed on ``integrate_definite``, whose four required
    fields could not be filled — it produced an EMPTY confirmation dialog
    instead, which is not much better.

    So a phrase that names an integral is answered here, with a reason, when
    there is no integral to answer about.  The one case that still declines is
    an **explicit E-id**: `E5 を計算して` / `evaluate E5` is ordinary "compute
    this", not an integral-specific request, and the supervisor should still get
    its turn at it.
    """
    m = nlp("evaluate_integral", lang).match(text)
    if m is None:
        return None
    tgt = m.group("target") or ""
    named_expr = tgt.startswith("E")
    if named_expr:
        eid = tgt if (tgt in engine.expressions and tgt in engine._active_expr_ids()) else None
    else:
        eid = _latest_integral_form(engine)
    if eid is not None and "'integrate(" in (engine.expressions[eid].canonical or ""):
        # A HELD integral (the component expansion's output).  "評価して" is the
        # separate request that design is built around, so it is answered here
        # rather than falling through to generic compute — which would `ratsimp`
        # the noun and hand back the same held form, looking like a no-op.
        try:
            expr = await loop.run_in_executor(
                None, partial(engine.evaluate_held_integrals, eid))
        except Exception as _e:
            # Broad on purpose (see above): "evaluate this" over all space is
            # exactly where Maxima asks an unanswerable convergence question,
            # and that has to surface as a failed step with its reason.
            return {"status": "failed", "error": str(_e),
                    "registered_expressions": registered}
        registered.append({"expr_id": expr.display_id,
                           "canonical": expr.canonical, "latex": expr.latex})
        return {"status": "completed", "operation": "evaluate_integral",
                "expr_id": expr.display_id, "latex": expr.latex,
                "canonical": expr.canonical,
                "registered_expressions": registered}
    if eid is None or not getattr(engine.expressions.get(eid), "integral_descriptor", None):
        if named_expr:
            return None          # "compute E5" — not ours to refuse
        return {"status": "failed",
                "error": t("integral.no_form_to_evaluate", lang),
                "registered_expressions": registered}
    if not (engine.expressions[eid].integral_descriptor or {}).get("bounds"):
        # A region form with no explicit bounds must be transformed first.
        return {"status": "needs_bounds",
                "message": t("integral.eval_needs_transform", lang),
                "registered_expressions": registered}
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.evaluate_integral_form, eid))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "completed", "operation": "evaluate_integral",
            "expr_id": expr.display_id, "latex": expr.latex,
            "canonical": expr.canonical,
            "registered_expressions": registered}

async def _try_taylor_expand(engine, loop, lang, text, registered) -> "dict | None":
    """Taylor / Maclaurin expansion, e.g.
    `taylor expand E1 in lam around 0 to order 2`
    (JA: `E1 を lam について 0 のまわりで 2 次までテイラー展開して`).

    Rule-based because `taylor_expand` has FOUR required fields
    (expr/var/point/order) and the positional field mapper cannot fill them.
    Measured 2026-08-05, preparing S-j (two-level perturbation theory):

        E1 を lam について 0 のまわりで 2 次までテイラー展開して
            → point=2, order=0          (swapped)
        taylor expand E1 in lam around 0 to order 2
            → var="expand", point="in"  (words taken by position)

    Both came back as `needs_confirmation` with those values already filled in,
    so clicking through gives an expansion about the wrong point at the wrong
    order — the BP-6 shape ("a mis-read field yields a plausible wrong value
    with no error anywhere"), and answered the same way F-9 answered it: read
    each field from its own linguistic marker rather than from where it sits.

    The point is optional and defaults to 0 — that is the Maclaurin case, which
    is what "テイラー展開" without a stated centre means.
    """
    m = nlp("taylor_expand_nl", lang).match(text)
    if m is None:
        return None
    target = (pick_group(m, "expr") or "").strip()
    if _EXPR_ID_RE.match(target):
        if target not in engine.expressions or target not in engine._active_expr_ids():
            return None
    elif _HAS_CJK_RE.search(target):
        return None
    else:
        target = math_operand(engine, target)
    var = math_operand(engine, (pick_group(m, "var") or "").strip())
    # 展開中心の読みと「研究者に何を伝えるか」は **1つの答え**
    # (`interpret_common.taylor_center_note`)。ここは `or "0"` と黙って既定を
    # 入れていて、同じ要求が supervisor 経路では確認を返していた。
    point, _center_note = taylor_center_note(pick_group(m, "point"), lang)
    args = {"expr": target, "var": var,
            "point": engine._handler.normalize_bound(point),
            "order": str(number_word(pick_group(m, "order")))}
    if _center_note:
        return {"status": "needs_confirmation", "operation": "taylor_expand",
                "args": args, "candidates": [], "pending_dependencies": [],
                "confirmation_prompt": _center_note + t("interpreter.confirm_run", lang),
                "registered_expressions": registered}
    try:
        step, val = await loop.run_in_executor(
            None, partial(engine.run_operation, "taylor_expand", args))
    except (ValueError, RuntimeError) as _e:
        # Narrow on purpose: run_operation raises ValueError for a rejected
        # operand or a CAS error and RuntimeError for a dead adapter.  Anything
        # else here is a bug in this function and must not be dressed up as
        # "the researcher asked for something impossible".
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    oid = step.output_refs[0] if step.output_refs else None
    oe = engine.expressions.get(oid) if oid else None
    if oe:
        registered.append({"expr_id": oe.display_id,
                           "canonical": oe.canonical, "latex": oe.latex})
    return {"status": "completed", "operation": "taylor_expand",
            "expr_id": oid, "latex": oe.latex if oe else "",
            "canonical": oe.canonical if oe else val,
            "operation_label": step.visible_action,
            "registered_expressions": registered}


async def _try_series_expand_no_var(engine, loop, lang, text,
                                    registered) -> "dict | None":
    r"""次数を書いた展開なのに**変数が書かれていない**綴り。

    `E1 を 3次まで展開して` / `expand E1 to 3rd order` は、黙って代数 `expand` に
    落ちていた——`exp(x)` が `%e^x` のまま `completed` で返る（実測 2026-09-06）。
    次数を書いた時点で代数展開ではないので、`taylor_expand` へ回す。**変数は
    推測しない**: `taylor_expand` は `var` を要求するので、検証器が「var が要る」と
    言う。書かれていないものを埋めるより、書かれていないと言うほうがよい。

    `taylor_expand_nl` の**後ろ**に置く——変数が書かれていればそちらが答える。
    """
    m = nlp("series_expand_no_var", lang).match(text)
    if m is None:
        return None
    eid = (m.group("eid") or "").strip()
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    # 展開中心は書かれていないので既定（マクローリン）。その読みと「研究者に何を
    # 伝えるか」は `taylor_center_note` に1つだけある——`or "0"` と黙って埋めない。
    point, _center_note = taylor_center_note(None, lang)
    args = {"expr": eid, "order": str(number_word(m.group("order"))),
            "point": engine._handler.normalize_bound(point)}
    # **変数は推測しない。** 不足フィールドの訊き方はこの系に1つある
    # （`pending_op` ＋ `interpreter.missing_field.<欄>`）ので、そこへ載せる
    # ——自前で `str(_e)` を返すと、監督層の経路が日本語で返している同じ問いに
    # 英語の生の文言で答えることになる（`Missing required fields: [...]`）。
    engine.pending_op = {"operation": "taylor_expand", "args": args,
                         "missing": ["var"]}
    return {"status": "needs_confirmation", "operation": "taylor_expand",
            "args": args, "candidates": [], "pending_dependencies": [],
            "confirmation_prompt": (_center_note.rstrip() + "\n"
                                    + t("interpreter.missing_field.var", lang)),
            "registered_expressions": registered}


async def _try_expand_components(engine, loop, lang, text, registered) -> "dict | None":
    r"""Open the vectors of an expression into components (``expand_components``).

    The way a `\int \frac{d^3k}{(2\pi)^3}` becomes something a one-variable CAS
    can hold, and the reason that measure is allowed to be WRITTEN at all: the
    dimension is opened explicitly, on the researcher's word, rather than being
    guessed at input or refused outright.  Rule-based, before the LLM — the
    request names no quantities to extract, only a target.
    """
    from notebook.runtime.vector_components import UnopenedVector
    m = nlp("expand_components", lang).match(text)
    if m is None:
        return None
    target = m.group("target") or ""
    eid = target if (target in engine.expressions
                     and target in engine._active_expr_ids()) else None
    if eid is None:
        eid = _latest_vector_expr(engine)
    if eid is None:
        return {"status": "failed", "error": t("integral.no_form_to_evaluate", lang),
                "registered_expressions": registered}
    try:
        expr, names = await loop.run_in_executor(
            None, partial(engine.expand_vector_components, eid))
    except UnopenedVector as _uv:
        return {"status": "failed",
                "error": t("engine.vector_not_opened", lang, name=_uv.name),
                "registered_expressions": registered}
    except Exception as _e:
        # Broad on purpose, same as every other interceptor here: the CAS, the
        # rewrite and the registration can each fail in their own way, and a
        # request the researcher made must come back as a failed step with the
        # reason rather than as a traceback through the pipeline.
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "completed", "operation": "expand_components",
            "expr_id": expr.display_id, "latex": expr.latex,
            "canonical": expr.canonical,
            "message": t("engine.components_opened", lang, names=", ".join(names)),
            "registered_expressions": registered}
