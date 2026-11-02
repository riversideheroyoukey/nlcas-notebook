"""Rule-based short circuits: operator/transform substitutions and other
direct operation dispatches that bypass the LLM.

Extracted verbatim from interpret_pipeline.py (refactor Stage 2,
2026-07-04).  run_interpret_pipeline imports this back; behaviour is
unchanged.
"""
from __future__ import annotations

import logging
import re

from notebook.engine import OperationRefused
from functools import partial
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.operand_boundary import may_fall_back_to_the_recent_expression
from ui.interpret_common import (math_operand, number_word,
                                 operands_as_written, pick_group)
from notebook.runtime.shared_parser import split_relation
from ui.interpret_l1 import (  # noqa: F401
    _is_operator_symbol,
    _op_name_index,
    _parse_transform_subst,
    _subst_lhs_rhs,
)
from ui.interpret_registry import (
    Interceptor, InterceptorContext, run_interceptors,
)

_logger = logging.getLogger(__name__)


async def _try_rule_short_circuits(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive: bool = True, alias_free_text: str = "") -> "dict | None":
    """Rule-based short-circuits (2e2–2h): wrt_qdot, differentiate, transform
    substitution, substitute, set_notation, set_latex_name, combine.  Returns a
    response dict to short-circuit, or None to continue.  R1d (docs/27): the
    former giant if/return chain is now an ordered registry — the firing order is
    the data in _RULE_INTERCEPTORS, not code position.  `_rule_order` is the order
    pre-extracted at step 2e (kept in the caller for the later LLM path)."""
    return await run_interceptors(
        InterceptorContext(engine=engine, loop=loop, lang=lang, text=supervisor_text,
                           registered=registered, req=req, rule_order=_rule_order,
                           interactive=interactive, alias_free_text=alias_free_text),
        _RULE_INTERCEPTORS)


# ---------------------------------------------------------------------------
# R1d head batch (docs/27): the diff / transform-subst head blocks, extracted
# verbatim.  They fire FIRST, so they head the list.
# ---------------------------------------------------------------------------
async def _try_rule_wrt_qdot(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2e2. Rule-based wrt_qdot (∂/∂q̇) short-circuit.
    # Detects "q̇で偏微分" / "速度で偏微分" / "∂/∂qdot" patterns and maps to
    # {operation: differentiate, wrt_qdot: <qty>}.  Requires a registered
    # time-dependent quantity to be identifiable from text or selected expression.
    # Every alternative must include a differentiation verb (微分) adjacent to the
    # velocity, so a bare \dot{...} token that merely appears elsewhere (e.g. in a
    # substitution RHS like "… を -2\dot{r}\dot{θ}/r で置き換えて") does NOT trigger
    # the wrt_qdot short-circuit.
    # wrt-q̇ differentiation keyword detector (JA and EN forms) externalized to
    # nl_patterns.json["qdot_diff_kw"].  Every alternative pairs a differentiation
    # verb/noun with the velocity token so a bare \dot{...} elsewhere (e.g. in a
    # substitution RHS) does NOT trigger the wrt_qdot short-circuit.
    if nlp("qdot_diff_kw", lang).search(supervisor_text):
        # Resolve which quantity's velocity to use.
        _qdot_qty_name: str | None = None
        for _qn in engine.symbol_context.registry:
            if re.search(
                r'(?<![A-Za-z0-9_])' + re.escape(_qn) + r'(?![A-Za-z0-9_])',
                supervisor_text
            ):
                _qdot_qty_name = _qn
                break
        if _qdot_qty_name is None and req.selected_expr_id:
            _sel = engine.expressions.get(req.selected_expr_id)
            if _sel and _sel.quantity_name and _sel.quantity_name in engine.symbol_context.registry:
                _qdot_qty_name = _sel.quantity_name
        if _qdot_qty_name is None and len(engine.symbol_context.registry) == 1:
            _qdot_qty_name = next(iter(engine.symbol_context.registry))
        # Resolve target expression: an explicit "En を" (or English "of En" /
        # "En with respect to") reference wins (so "E1 を \dot{y} で偏微分" /
        # "the derivative of E1 with respect to \dot{y}" targets the
        # Lagrangian E1, not the most recent expression), then the selected
        # expr, then the most recent.
        _qdot_expr_id: str | None = None
        _qdot_eid_m = nlp("qdot_eid", lang).search(supervisor_text)
        if _qdot_eid_m:
            _qdot_gd = _qdot_eid_m.groupdict()
            _qdot_eid = _qdot_gd.get("eid1") or _qdot_gd.get("eid2")
            if _qdot_eid and _qdot_eid in engine.expressions:
                _qdot_expr_id = _qdot_eid
        # ここの E-id は**任意**（`\dot{q} で偏微分して` だけで要求になる）ので、
        # 選択中／直前の式へ落ちること自体は能力である。落ちてよいかだけを、この族
        # の全員と同じ helper に訊く（理由と実測はその docstring）。
        if not _qdot_expr_id and may_fall_back_to_the_recent_expression(
                engine.symbol_context, supervisor_text):
            _qdot_expr_id = (req.selected_expr_id or None
                             or engine.most_recent_expr_id())
        if _qdot_qty_name and _qdot_expr_id:
            _qdot_args = {"expr": _qdot_expr_id, "wrt_qdot": _qdot_qty_name}
            try:
                step, value = await loop.run_in_executor(
                    None, partial(engine.run_operation, "differentiate", _qdot_args)
                )
            except Exception as e:
                return {"status": "failed", "error": str(e),
                        "registered_expressions": registered}
            _qdot_out_id = step.output_refs[0] if step.output_refs else None
            _qdot_expr   = engine.expressions.get(_qdot_out_id) if _qdot_out_id else None
            return {
                "status":    "completed",
                "operation": "differentiate",
                "expr_id":   _qdot_out_id,
                "step_id":   step.step_id,
                "latex":     _qdot_expr.latex   if _qdot_expr else "",
                "canonical": _qdot_expr.canonical if _qdot_expr else value,
                "operation_label": step.visible_action,
                "registered_expressions": registered,
            }
    return None


async def _try_rule_differentiate(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2e3. Rule-based differentiate short-circuit.
    # Handles the unambiguous pattern "En を X で (偏)微分して" directly,
    # bypassing the LLM entirely.  Variable X may be a plain identifier (q, t, x)
    # or a simple LaTeX command (\omega, \xi_k, etc.).
    #
    # Supported surface forms (all bypass LLM):
    #   "E1 を q で偏微分して"
    #   "E1 を t で微分して"
    #   "E1 の q に関する偏微分"
    #   "E1 を q について微分して"
    #   "E2 を t で2階微分して"  ← order already extracted as _rule_order by 2e
    #
    # Does NOT fire when:
    #   - wrt_qdot pattern was detected (already handled above by 2e2)
    #   - No recognizable expr ref AND no selected/recent expression
    #   - Variable is not extractable from text
    # Externalized to i18n/nl_patterns.json["diff_rule"] — JA's two word-orders
    # ("を...で微分" / "の...に関する微分") and the English "differentiate E1
    # with respect to k" form are one alternation per language, using
    # per-branch group names (eid1/var1, eid2/var2) since Python's re forbids
    # reusing a group name across alternation branches; extract via
    # .groupdict().get(...) so it works regardless of which branch matched.
    _diff_rule_m = nlp("diff_rule", lang).search(supervisor_text)
    if _diff_rule_m:
        _dr_gd = _diff_rule_m.groupdict()
        _dr_expr_token = _dr_gd.get("eid1") or _dr_gd.get("eid2")   # e.g. "E15" or "E_{total}"
        _dr_var_raw    = _dr_gd.get("var1") or _dr_gd.get("var2")   # e.g. "q" or "\omega"
        # Resolve expr token to an actual expression ID in the current branch.
        # Also handle subscripted forms (E_{total}) via lhs_alias_map lookup.
        _dr_expr_id: str | None = None
        if _dr_expr_token in engine.expressions and _dr_expr_token in engine._active_expr_ids():
            _dr_expr_id = _dr_expr_token
        # Try lhs_alias_map for subscripted E-ID forms (e.g. E_{total} → lhs key)
        if _dr_expr_id is None:
            _dr_lhs_key = re.sub(r'^E_\{([^}]+)\}$', r'E_\1', _dr_expr_token)  # E_{total} → E_total
            _dr_alias_eid = engine.lhs_alias_map.get(_dr_expr_token) or engine.lhs_alias_map.get(_dr_lhs_key)
            if _dr_alias_eid and _dr_alias_eid in engine._active_expr_ids():
                _dr_expr_id = _dr_alias_eid
        # **この規則は式番号を必ず要求する**（`diff_rule` の両言語・両分岐）ので、
        # ここで未解決なら「書いた番号が無い」ということ——選択中の式や直前の式で
        # 代用せず降り、後段の別名走査に声を出して断らせる。落ちる枝を2つとも
        # 消した理由と実測は `may_fall_back_to_the_recent_expression` の docstring。
        # **微分の変数は研究者が書いた名前**なので、式そのものと同じ扉を通す。
        # ここはずっと `.strip()` だけで（註だけが「convert LaTeX」と言っていた）、
        # 式は `latex_to_canonical` を通るのに変数は通らなかった——**同じ名前が
        # 2つの CAS 対象になる**。実測 2026-09-19、`z^2 + \bar{y}` に対して:
        #
        #     diff(z^2 + y_bar, bar(y), 1)  →  0   completed
        #
        # 式の側は飾りのある名前 `y_bar`、変数の側は**関数呼び出し** `bar(y)`。
        # `\hat{H}` / `\tilde{x}` も同じで、日英ともこうなる。`solve` は 2026-08-21
        # にこの扉を通すよう直されており（`solve(z^2 + y_bar, y_bar)` は正しい）、
        # `math_operand` の docstring は「**変数**を含め、捕まえた被演算子は渡す前に
        # 変換する。各所で言うから、当たる所と当たらない所ができる」と書いている
        # ——同じ問いの4件目である。
        # **共有の扉を呼ぶ**——素の `latex_to_canonical` に差し替えても、
        # ここでは何も変わらない（`{{symbol}}` の捕獲は空白も変換不能な
        # テキストも生まないので、strip も失敗時の退避も届かない。変異試験で
        # 実測）。それでも扉を呼ぶのは、この問いの答えを1箇所に置くためである。
        _dr_var = math_operand(engine, _dr_var_raw)
        if _dr_expr_id and _dr_var:
            if _rule_order:
                _dr_order = _rule_order
            else:
                # _rule_order (upstream, interpret_pipeline.py) already covers
                # "twice"/"N times" phrasing; this local fallback additionally
                # covers ordinal phrasing ("3rd order"/"3階") that upstream
                # doesn't scope to, since it's differentiate-specific here.
                _dr_order_m = nlp("diff_order_ordinal", lang).search(supervisor_text)
                _dr_order = (number_word(_dr_order_m.group(1))
                             if _dr_order_m else 1) or 1
            _dr_args: dict = {"expr": _dr_expr_id, "var": _dr_var}
            if _dr_order > 1:
                _dr_args["order"] = _dr_order
            # Derivative kind.  "全微分" → total (d/dt), "偏微分" → partial (∂/∂t).
            # When the user said plain "微分" and the choice is ambiguous (a time
            # derivative of a time-dependent expression: ∂/∂t ≠ d/dt), do NOT guess
            # — ask interactively, or in non-interactive use require an explicit
            # 偏微分/全微分.
            if nlp("total_derivative_marker", lang).search(supervisor_text):
                _dr_args["_is_total_derivative"] = True
            elif nlp("partial_derivative_marker", lang).search(supervisor_text):
                _dr_args["_is_partial_derivative"] = True
            elif engine.differentiate_is_ambiguous(_dr_expr_id, _dr_var):
                if not interactive:
                    return {
                        "status": "failed",
                        "error": t("interpreter.diff_specify_kind", lang, var=_dr_var),
                        "registered_expressions": registered,
                    }
                return {
                    "status": "needs_confirmation",
                    "operation": "differentiate",
                    "args": _dr_args,
                    "confirmation_prompt": t("interpreter.diff_partial_or_total", lang,
                                             expr_id=_dr_expr_id, var=_dr_var),
                    "candidates": [
                        {"operation": "differentiate",
                         "args": {**_dr_args, "_is_total_derivative": True},
                         "display_ja": t("interpreter.diff_total_choice", "ja", var=_dr_var),
                         "display_en": t("interpreter.diff_total_choice", "en", var=_dr_var)},
                        {"operation": "differentiate",
                         "args": {**_dr_args, "_is_partial_derivative": True},
                         "display_ja": t("interpreter.diff_partial_choice", "ja", var=_dr_var),
                         "display_en": t("interpreter.diff_partial_choice", "en", var=_dr_var)},
                    ],
                    "registered_expressions": registered,
                }
            try:
                _dr_step, _dr_value = await loop.run_in_executor(
                    None, partial(engine.run_operation, "differentiate", _dr_args)
                )
            except Exception as _dr_e:
                return {"status": "failed", "error": str(_dr_e),
                        "registered_expressions": registered}
            _dr_out_id   = _dr_step.output_refs[0] if _dr_step.output_refs else None
            _dr_out_expr = engine.expressions.get(_dr_out_id) if _dr_out_id else None
            return {
                "status":    "completed",
                "operation": "differentiate",
                "expr_id":   _dr_out_id,
                "step_id":   _dr_step.step_id,
                "latex":     _dr_out_expr.latex    if _dr_out_expr else "",
                "canonical": _dr_out_expr.canonical if _dr_out_expr else _dr_value,
                "operation_label": _dr_step.visible_action,
                "registered_expressions": registered,
            }
    return None


async def _try_rule_transform_subst(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2e3.4. L-1 step 4: apply a multi-equation transform to a target, routing by
    # type.  "E5, E6 を E10 に代入して" substitutes each transform equation into the
    # target — operators via operator_substitute (non-commutative), scalars via
    # substitute (commutative).  Single-source commutative falls through to 2e3.5.
    _ts = _parse_transform_subst(supervisor_text, lang)
    if _ts:
        _ts_srcs, _ts_tgt = _ts
        if engine.expressions.get(_ts_tgt) is not None:
            _ts_pairs = [(_s, _subst_lhs_rhs(engine, _s)) for _s in _ts_srcs]
            _ts_pairs = [(s, lr) for s, lr in _ts_pairs if lr]
            _ts_is_op = any(_is_operator_symbol(engine, lr[0]) for _, lr in _ts_pairs)
            # Single-source commutative case is handled by the existing 2e3.5 block.
            if _ts_pairs and (_ts_is_op or len(_ts_pairs) > 1):
                try:
                    if _ts_is_op:
                        _rules = []
                        for _, (_lhs, _rhs) in _ts_pairs:
                            _nm, _idx = _op_name_index(_lhs)
                            _rule = {"op_name": _nm, "rhs": _rhs}
                            if _idx:
                                _rule["formal_index"] = _idx
                            _rules.append(_rule)
                        _ts_step, _ts_val = await loop.run_in_executor(
                            None, partial(engine.run_operation, "operator_substitute",
                                          {"expr": _ts_tgt, "op_rules": _rules}))
                    else:
                        # A source whose left symbol is ALREADY GONE from the
                        # running target has nothing left to do — and that is the
                        # normal case here, not an error.  H-1's P2 binds every
                        # definition CAS-side, so the FIRST `subst` expands all of
                        # them at once: measured on `p = u a + v b` / `g = w a` /
                        # `H = p + g`, step 1 returns the complete `a*w+b*v+a*u`
                        # and step 2 then refuses because `g` is not in it.  The
                        # refusal is correct about the text and wrong about the
                        # work — the whole chain reported FAILED while the right
                        # answer sat registered in front of the researcher
                        # (open_items TRANSFORM-SUBST-CHAIN).
                        #
                        # TWO conditions, and the second was learned the hard way.
                        #
                        # The symbol must have been in the target the researcher
                        # NAMED: a source that never applied to this target at all
                        # is a real mistake and still fails.  That is the
                        # difference between "already done" and "never there", and
                        # only the ORIGINAL target can tell them apart.
                        #
                        # And the left side must be a BARE SYMBOL, because that is
                        # the only shape P2 can bind — H-1 registers a symbol only
                        # for a simple left side (a compound one goes into
                        # `lhs_alias_map` alone), so "an earlier step already
                        # expanded it" cannot happen otherwise.  Without this,
                        # measured on `\sum_{i=1}^{3} c_i = u a + v b` substituted
                        # into a target holding that sum, suppression turned a LOUD
                        # failure into a SILENT WRONG ANSWER: step 1 does not match
                        # the sum at all (Maxima evaluates `sum(c(i),i,1,3)` rather
                        # than pattern-matching it) and returns the target with its
                        # sums merely expanded — `c(3)+d(2)+c(2)+d(1)+c(1)`, no
                        # substitution performed, reported as success.  That step-1
                        # defect is open_items TRANSFORM-SUBST-CHAIN cause 2 and is
                        # NOT fixed here; the chain's failure is currently the only
                        # thing telling the researcher, so it must keep telling.
                        _orig = engine.expressions.get(_ts_tgt)
                        _orig_text = f"{_orig.canonical or ''} {_orig.latex or ''}" if _orig else ""
                        _cur_tgt = _ts_tgt
                        _ts_step = None
                        for _, (_lhs, _rhs) in _ts_pairs:
                            try:
                                _ts_step, _ts_val = await loop.run_in_executor(
                                    None, partial(engine.run_operation, "substitute",
                                                  {"expr": _cur_tgt, "lhs": _lhs, "rhs": _rhs}))
                            except OperationRefused as _exc:
                                if (_exc.reason_code == "lhs_not_found"
                                        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",
                                                         (_lhs or "").strip())
                                        and _mentions_symbol(_orig_text, _lhs)):
                                    continue          # an earlier step covered it
                                raise
                            _cur_tgt = (_ts_step.output_refs or [_cur_tgt])[0]
                except Exception as _ts_exc:
                    return {"status": "failed", "error": str(_ts_exc),
                            "registered_expressions": registered}
                _ts_out = (_ts_step.output_refs or [None])[0] if _ts_step else None
                _ts_oe = engine.expressions.get(_ts_out) if _ts_out else None
                return {
                    "status": "completed",
                    "operation": "operator_substitute" if _ts_is_op else "substitute",
                    "expr_id": _ts_out,
                    "latex": _ts_oe.latex if _ts_oe else "",
                    "canonical": _ts_oe.canonical if _ts_oe else "",
                    "registered_expressions": registered,
                }
    return None


def _mentions_symbol(text: str, name: str) -> bool:
    """Whether *name* appears in *text* as a whole identifier.

    Deliberately a plain word-boundary scan and NOT the substitution guard's own
    `lhs_may_match`: that predicate answers "can the CAS match this?", weighing
    spelling variants and CAS-side bindings, and is the Maxima handler's to own.
    The question here is different and much smaller — "did the researcher's
    target mention this symbol at all?" — and it is asked of the ORIGINAL text,
    before any substitution has run, purely to tell "already applied" from
    "never applied".  Answering it with the CAS predicate would put a matching
    decision in two layers, which is how they drift.
    """
    if not name or not text:
        return False
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name.strip())
                     + r"(?![A-Za-z0-9_])", text) is not None


async def _run_substitute_and_report(engine, loop, expr_id, lhs, rhs, registered):
    """Run one substitution and build its response — for BOTH rules that read one.

    Substitution is claimed by two rules whose split does not line up between
    the languages (`E2 の A を B に置換して` takes `subst_replace`, its English
    sibling takes `substitute_into`, and `E6 の q に p を代入して` is the
    mirror).  The corpus recorded that as a drift detector because both answer
    identically; measured 2026-08-27 they did not, and the difference was on
    the SCREEN — one returned no ``operation_label`` (the timeline card's own
    label), and the other appended its result to ``registered_expressions``,
    which the UI renders FIRST and then skips ``data`` as already shown, so the
    card the researcher got was the unlabelled one.  Neither is visible to a
    comparison of operation and operands, which is what the corpus compares.
    """
    lhs = engine._handler.latex_to_canonical(lhs) or lhs
    rhs = engine._handler.latex_to_canonical(rhs) or rhs
    try:
        step, value = await loop.run_in_executor(None, partial(
            engine.run_operation, "substitute",
            {"expr": expr_id, "lhs": lhs, "rhs": rhs}))
    except Exception as exc:                                    # noqa: BLE001
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    out_id = step.output_refs[0] if step.output_refs else None
    out_expr = engine.expressions.get(out_id) if out_id else None
    return {"status": "completed", "operation": "substitute",
            "expr_id": out_id, "step_id": step.step_id,
            "latex": out_expr.latex if out_expr else "",
            "canonical": out_expr.canonical if out_expr else value,
            "operation_label": step.visible_action,
            "registered_expressions": registered}


async def _try_rule_substitute_into(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive, alias_free_text="") -> "dict | None":
    # 2e3.4. Rule-based symbol→value substitution: "En の X に Y を代入して".
    # Replace X (a symbol/expr) with Y (a value/expr) inside En.  The supervisor's
    # positional token mapping fails to fill the substitute `rhs` when Y contains a
    # "/" (e.g. λ/μ), so parse it deterministically here.  X must NOT be a bare E-ID
    # (那 form — "Em に En を代入" — is the equation substitution handled by
    # _DAINYU_RE below).
    # NL pattern externalized to i18n/nl_patterns.json (named groups eid/lhs/rhs;
    # the JP and EN regexes differ in word order but share the group names).
    # Only the symbol→VALUE form ("X に Y を代入" with Y a literal value).  When Y is
    # an E-ID ("X に En を代入"), the value is an equation reference → let _DAINYU_RE
    # resolve its lhs/rhs.  Likewise skip an E-ID lhs.
    # Also skip when rhs itself contains a "、"-separated second "に" clause: the
    # lazy rhs group then swallows a whole SECOND substitution clause (e.g. the
    # Bogoliubov double-substitution "En の a(k) に X を、b(k) に Y を代入する"), since
    # this pattern's single trailing "を(代入|…)" anchor is the only one in the
    # string and the lazy quantifier is forced to expand rhs across the "、" clause
    # boundary.  The second clause's own trailing "を" (right before the shared
    # "代入" anchor) is NOT part of the captured rhs — it's consumed by the fixed
    # suffix — so the guard can only look for "、" followed by "に", not a full
    # "に…を" pair.  Such multi-rule substitutions are handled by the backend's own
    # (Julia) operator double/multi-substitution patterns further down the pipeline.
    # English mirrors the Japanese "、-then-に" case, but swallows the second
    # clause into LHS instead of RHS (its "for" pattern anchors on end-of-
    # string via the lhs group, not on a shared trailing keyword) — e.g.
    # "substitute E8 for c(k,u) and E10 for c(-k,d)" ends up with
    # lhs="c(k,u) and E10 for c(-k,d)".  A second " and ... for " signals
    # this chaining and defers to the backend's own multi-substitution
    # pattern, same rationale as the Japanese guard above.
    # Also skip when lhs names a registered quantum-operator symbol (e.g.
    # "Sz_2", "S^z_2") rather than a scalar: this generic path dispatches to
    # plain "substitute" (whole-operator identity, no per-site index), the
    # same SUBST-SITE-INDEX issue as the 2e4 short-circuit below — a
    # mean-field decoupling "Sz_2 → mz" must go through the backend's
    # index-aware operator_substitute instead.
    _si_m = nlp("substitute_into", lang).match(supervisor_text)
    # The operands AS WRITTEN, read before the guards below — because every one
    # of those guards asks about an OPERAND ("is it a bare E-id?", "does it name
    # a registered operator?"), and 2a-sym has already replaced the researcher's
    # symbol with an E-id in `supervisor_text`.  Measured 2026-08-27:
    # `E1 の x に 2 を代入して` after `solve E1 for x` arrived as
    # `E1 の E2 に 2 を代入して`, the bare-E-id guard fired on a symbol the
    # researcher never wrote, and the rule declined — the request failed.
    _si_lhs, _si_rhs = operands_as_written(
        nlp("substitute_into", lang).match(alias_free_text) if alias_free_text else None,
        lambda m, n: m.group(n), ("lhs", "rhs"),
        ((_si_m.group("lhs").strip(), _si_m.group("rhs").strip()) if _si_m else ("", "")))
    # chained_subst_guard: a second clause chained onto the match signals a
    # multi-substitution request the backend's own multi-subst path should
    # handle instead.  Which field carries that signal is language-specific
    # (JA's "、...に" trails in rhs; EN's "and...for" trails in lhs, per
    # _si_m's own group layout for that language) — see comments above.
    _si_guard_field = (_si_lhs if lang == "en" else _si_rhs) if _si_m else ""
    if (_si_m and not re.fullmatch(r'E\d+', _si_lhs)
            and not re.fullmatch(r'E\d+', _si_rhs)
            and not nlp("chained_subst_guard", lang).search(_si_guard_field)
            and not any(
                _entry.quantity.kind == "operator"
                and _si_lhs.startswith(_name)
                for _name, _entry in engine.symbol_context.registry.items())):
        _si_eid = _si_m.group("eid")
        if _si_eid in engine.expressions and _si_eid in engine._active_expr_ids():
            return await _run_substitute_and_report(
                engine, loop, _si_eid, _si_lhs, _si_rhs, registered)
    return None


async def _try_expr_with_assignment(engine, loop, lang, text,
                                    registered) -> "dict | None":
    r"""`E1 with x = 2` / `evaluate E1 at x = 2` —— どちらも代入である。

    直す前は `expr_id_formula` が拾って canonical `(a*x + b)*with*x = 2` を
    **登録**し、`registered` を返していた（実測 2026-09-06）。`with` を
    `instruction_marker` に足して直すのは誤り——式として登録される行に6件出る。
    **語ではなく形**で読む（それらの行は `=` を含まない）。
    """
    m = (nlp("expr_with_assignment", lang).match(text)
         or nlp("evaluate_at_point", lang).match(text))
    if m is None:
        return None
    eid = m.group("eid")
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    return await _run_substitute_and_report(
        engine, loop, eid, m.group("lhs").strip(), m.group("rhs").strip(),
        registered)


# ---------------------------------------------------------------------------
# R1d (docs/27): rule short-circuit tail interceptors, extracted verbatim from
# _try_rule_short_circuits (each original if-block wrapped in an async def that
# returns None on fall-through; indentation unchanged).  All keep the full
# (req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive)
# signature even when they use only some args, so the bodies stay verbatim.
# ---------------------------------------------------------------------------
async def _try_rule_dainyu(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2e3.5. Rule-based formula substitution: "En を Em に代入して"
    # When En is a solved equation (has lhs_label, e.g. g = lω²) and Em is the
    # target expression, extract lhs/rhs from En and substitute into Em.
    # Two accepted orderings:
    #   "En を Em に代入して" → src=group(1), tgt=group(2), of_clause=group(3)
    #   "Em に En を代入して" → tgt=group(4), of_clause=group(5), src=group(6)
    # The optional "の…" clause is normally decorative filler ("Em の値に"), but
    # when it names a specific operator call with an index — "Em の c(k,u) に"
    # — that names a TARGETED per-index substitution, which this whole-operator
    # identity substitution cannot express (it would silently drop the index and
    # rewrite every c(...) occurrence with the same literal RHS, corrupting
    # index-dependent results like a Bogoliubov transform).  Bail out and let
    # the backend's operator_substitute interceptor (which does honor the
    # index) handle it instead.
    # English: "Substitute E4 into E1" (src, tgt) — group(7)/group(8).  No "の…"
    # index-clause form is offered in English (that case is expected to go
    # through operator_substitute's own English trigger instead), so there is
    # no group(9) equivalent to bail out on.
    # Grammar externalized to nl_patterns.json["dainyu"] (session-language).
    # 語順の枝は `pick_group` で読む——接尾辞をここに書き並べると**枝を足すたびに
    # 2箇所を揃える**形になる（二十八問目で切り出した道具）。実際そうなっていて、
    # 目的語が先に来る英語の語順（`In E2, substitute E1`）を足すのに、パターンと
    # ここの両方を触る必要があった。
    _dainyu_m = nlp("dainyu", lang).search(supervisor_text)
    if _dainyu_m:
        if re.match(r"^[A-Za-z][A-Za-z0-9_']*\(",
                    pick_group(_dainyu_m, "idx") or ""):
            _dainyu_m = None
    if _dainyu_m:
        _dy_src_id = pick_group(_dainyu_m, "src")
        _dy_tgt_id = pick_group(_dainyu_m, "tgt")
        _dy_active  = engine._active_expr_ids()
        _dy_src_expr = engine.expressions.get(_dy_src_id)
        _dy_tgt_expr = engine.expressions.get(_dy_tgt_id)
        if (_dy_src_expr and _dy_tgt_expr
                and _dy_tgt_id in _dy_active):
            # Determine the substitution lhs/rhs from the source expression.
            # Strategy: strip Maxima solve brackets everywhere (raw solve results
            # like E8 store canonical="[g = l*omega^2]", which must become lhs="g",
            # rhs="l*omega^2").  _update_lhs_alias adds "[g" → "E8" as a side effect;
            # skip those bracket-prefixed keys.
            #
            # Priority:
            # 1. lhs_alias_map: skip keys that start with "[" (raw solve artifacts)
            # 2. Sibling expression: raw solve result (E8) → clean sibling (E9)
            # 3. Bracket-stripped canonical split
            # 4. Bracket-stripped latex split
            def _strip_solve_brackets(s: str) -> str:
                return re.sub(r'^\[+|\]+$', '', s).strip()

            def _dy_source_text(eid: str) -> str:
                """An expression's text AS THE CAS MUST SEE IT.

                A sum written WITHOUT a range keeps no range in its canonical —
                that is the researcher's own writing — and its placeholder pair
                lives in the IR.  Read raw and handed to `subst`, it reached
                Maxima as `sum(body, n)`, a two-argument sum it cannot evaluate,
                so substituting the mode expansion `\\psi = \\sum_n \\varphi_n c_n`
                into anything failed with a bare "Maxima evaluation error"
                (measured 2026-08-21).  The operation path and H-2's reference
                splice both ask this question through `placeholder_cas_view`;
                this is the third place that has to ask it, not answer it
                (EXPR-REF-PROVENANCE).
                """
                _e = engine.expressions.get(eid)
                return (engine.placeholder_cas_view(eid)
                        or (_e.canonical if _e else "") or "")

            _dy_lhs: str | None = None
            _dy_rhs: str | None = None

            # 1. lhs_alias_map (skip keys with "[" — raw solve artifacts like
            #    "[g" or "\\left[ g" that come from bracket-wrapped CAS output)
            # Prefer canonical (non-LaTeX) keys so that subst receives a valid
            # Maxima expression.  LaTeX keys like \ddot{q} are parsed by
            # SharedParser as ddot(q) (a function call) which Maxima doesn't
            # recognise as a derivative.  Keys not starting with '\' are
            # canonical (e.g. 'diff(q(t),t,2), g, omega); LaTeX keys start with '\'.
            _dy_lhs_candidates: list[str] = []
            for _al_key, _al_eid in engine.lhs_alias_map.items():
                if _al_eid == _dy_src_id and "[" not in _al_key:
                    _dy_lhs_candidates.append(_al_key)
            # Sort: non-backslash (canonical) first
            _dy_lhs_candidates.sort(key=lambda k: (k.startswith("\\"), k))
            if _dy_lhs_candidates:
                _dy_lhs = _dy_lhs_candidates[0]
                # If the best candidate is a LaTeX form (starts with \),
                # convert it to backend canonical so that build_command
                # receives a valid CAS expression (e.g. \ddot{q} →
                # 'diff(q(t),t,2)) rather than ddot(q) from SharedParser.
                # A key that merely CONTAINS a backslash is LaTeX too — a compound
                # derivative LHS such as "\frac{d^2}{d t d z} A_{z}+\frac{…} A_{y}"
                # does not start with one.
                if "\\" in _dy_lhs:
                    _dy_lhs = engine._handler.lhs_latex_to_canonical(_dy_lhs)
                if "\\" in _dy_lhs:
                    # Conversion did not fully land: the backend has no canonical
                    # form for this display string.  Handing it to the CAS would
                    # push raw LaTeX (\frac{…}) into the resolver, which reads it
                    # as a subscript and dies (H-10).  Drop the candidate and let
                    # the canonical split below read the same equation off
                    # _dy_src_expr.canonical instead.
                    _dy_lhs = None
                if _dy_lhs is not None:
                    # Strip brackets from canonical (may be "[g = lω²]" for raw)
                    _raw = _strip_solve_brackets(_dy_source_text(_dy_src_id))
                    # `"=" in x` was the naive spelling of "is this an equation":
                    # it reads `\sum_{i=1}^3*c_i = u*a + v*b` as splitting at the
                    # SUBSCRIPT, so `substitute` was handed the replacement
                    # `1}^3*c_i = u*a + v*b` (measured 2026-08-13).  Which
                    # relations count is kept as written — the guard also fired
                    # on `<=`/`>=` and an inequality reaches here — so only the
                    # SCAN becomes bracket-aware.  `:=` needs no guard of its
                    # own; find_top_level_relation excludes it.
                    _rel_raw = split_relation(_raw)
                    _dy_rhs = (_rel_raw[2] if (_rel_raw is not None
                                               and "=" in _rel_raw[1]) else _raw)

            # 2. Sibling check: raw solve result → find clean sibling
            if _dy_lhs is None:
                _dy_creator_step = next(
                    (s for s in engine.steps.values()
                     if _dy_src_id in (s.output_refs or [])),
                    None
                )
                if _dy_creator_step:
                    for _sib_id in (_dy_creator_step.output_refs or []):
                        if _sib_id == _dy_src_id:
                            continue
                        for _al_key2, _al_eid2 in engine.lhs_alias_map.items():
                            if _al_eid2 == _sib_id and "[" not in _al_key2:
                                _sib_expr = engine.expressions.get(_sib_id)
                                _dy_lhs = _al_key2
                                _dy_rhs = (_dy_source_text(_sib_id) if _sib_expr else None)
                                break
                        if _dy_lhs:
                            break

            # 3. Canonical split (bracket-stripped)
            # This LHS is BOTH the pattern `substitute` searches for and the
            # label written into the timeline (and from there into the exported
            # .tex).  Measured 2026-08-13: `y \le x^2 + a` produced the pattern
            # `y <`, and the step COMPLETED — the value was right, because the
            # builder lowers `y <` back to `y`, while the published label read
            # `E2 を[E2] y < → x^2 + a`.  Wrong in what is written down rather
            # than in what is computed, the same shape as `x + x = 2 x=y`.
            if _dy_lhs is None:
                _src_canon = _strip_solve_brackets(_dy_source_text(_dy_src_id))
                _rel_can = split_relation(_src_canon)
                if _rel_can is not None and "=" in _rel_can[1]:
                    _dy_lhs, _dy_rhs = _rel_can[0], _rel_can[2]

            # 4. Latex split (bracket-stripped)
            if _dy_lhs is None:
                _src_latex = _strip_solve_brackets(_dy_src_expr.latex or "")
                # `==` needs no guard of its own; find_top_level_relation excludes it.
                _rel_lat = split_relation(_src_latex)
                if _rel_lat is not None and "=" in _rel_lat[1]:
                    _dy_lhs, _dy_rhs = _rel_lat[0], _rel_lat[2]
            if _dy_lhs and _dy_rhs:
                _dy_args: dict = {"expr": _dy_tgt_id, "lhs": _dy_lhs, "rhs": _dy_rhs}
                try:
                    _dy_step, _dy_value = await loop.run_in_executor(
                        None, partial(engine.run_operation, "substitute", _dy_args)
                    )
                except Exception as _dy_e:
                    return {"status": "failed", "error": str(_dy_e),
                            "registered_expressions": registered}
                _dy_out_id   = _dy_step.output_refs[0] if _dy_step.output_refs else None
                _dy_out_expr = engine.expressions.get(_dy_out_id) if _dy_out_id else None
                return {
                    "status":    "completed",
                    "operation": "substitute",
                    "expr_id":   _dy_out_id,
                    "step_id":   _dy_step.step_id,
                    "latex":     _dy_out_expr.latex    if _dy_out_expr else "",
                    "canonical": _dy_out_expr.canonical if _dy_out_expr else _dy_value,
                    "operation_label": _dy_step.visible_action,
                    "registered_expressions": registered,
                }
    return None


async def _try_rule_substitute(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive, alias_free_text="") -> "dict | None":
    # 2e4. Rule-based substitute short-circuit.
    # Handles "En の A を B で置き換えて" directly, bypassing the LLM.
    # Also handles "En の A → B" and "En で A = B と置換して" forms.
    # The RHS (group 3) is a full expression that may contain spaces
    # ("-2 \dot{r} \dot{theta} / r"), so it is captured lazily as `.+?` up to the
    # trailing 置換/置き換え keyword — NOT `[^\s]+?`, which would stop at the first
    # space and drop multi-token replacements.
    # Grammar externalized to nl_patterns.json.  The primary form ("En の A を B で
    # 置き換え" / "replace A with B in En" / "substitute B for A in En") is
    # "subst_replace" (session-language); the arrow/= fallback ("En A → B", "En A =
    # B と置換") is "subst_replace_alt", kept ja-only and matched with lang="ja"
    # because arrows are language-neutral — this preserves the original
    # RE2-then-RE two-phase search order exactly.  Shared groups eid/lhs/rhs (JA)
    # and one lettered EN branch per word order — eid{A-F}/lhs{A-F}/rhs{A-F}, since
    # `re` forbids reusing a group name across an alternation.  A-D are
    # replace…with / substitute…for (each in both word orders); E-F are the
    # value-setting form ("set/let/put X = V in En", 2026-09-05).  The letters are
    # NOT enumerated on the reading side — see `_sr_pick`.
    def _sr_pick(m, stem):
        """枝の接尾辞を**列挙しない**——枝を足すたびに2箇所を揃える形だった。

        英語側は語順が枝ごとに違うので `eidA`…`eidD` と綴り分けてある（Python の
        `re` は alternation をまたいだ同名グループを禁じる）。ここが `A`〜`D` を
        書き並べていたので、2026-09-05 に「値を置く」形（`set X = V in En`）の
        枝 E/F を足したとき、**パターンだけ直して読めない**状態になりうる。
        値が入っている枝はどのみち1つなので、接尾辞は問わない。
        """
        return pick_group(m, stem)

    _subst_rule_m = (nlp("subst_replace", lang).search(supervisor_text)
                     or nlp("subst_replace_alt", "ja").search(supervisor_text))
    if _subst_rule_m and any(
            _entry.quantity.kind == "operator"
            and (_sr_pick(_subst_rule_m, "lhs") or "").strip().startswith(_name)
            for _name, _entry in engine.symbol_context.registry.items()):
        # The LHS names a registered quantum-operator symbol (e.g. "Sz_2",
        # "S^z_2") rather than a scalar — this generic short-circuit sends
        # such cases through the plain "substitute" op (whole-operator
        # identity replacement, no per-site index), silently dropping the
        # site index and corrupting index-dependent results (e.g. a
        # mean-field decoupling "Sz_2 → mz" replacing every site's Sz, not
        # just site 2).  Bail out and let the backend's index-aware
        # operator_substitute interceptor handle it instead.
        _subst_rule_m = None
    if _subst_rule_m:
        _sr_expr_token = _sr_pick(_subst_rule_m, "eid")
        _sr_lhs_raw    = _sr_pick(_subst_rule_m, "lhs").strip()
        _sr_rhs_raw    = _sr_pick(_subst_rule_m, "rhs").strip()
        _sr_active = engine._active_expr_ids()
        _sr_expr_id: str | None = None
        if _sr_expr_token in engine.expressions and _sr_expr_token in _sr_active:
            _sr_expr_id = _sr_expr_token
        if _sr_expr_id is None:
            _sr_expr_id = req.selected_expr_id
        if _sr_expr_id and _sr_lhs_raw and _sr_rhs_raw:
            # Convert both sides the way `substitute_into` — the OTHER rule that
            # answers this same request — has always done.  Substitution is
            # claimed by two rules whose split does not line up between the
            # languages (measured 2026-08-23 over the instruction corpus:
            # `E6 の q に p を代入して` takes `substitute_into` while its English
            # sibling `substitute p for q in E6` takes THIS one, and
            # `central_force_conservation` is the mirror), so the same request
            # was recorded two ways — `\ddot{theta}` here, `theta_ddot` there.
            #
            # Measured before changing it, on the six real spellings the corpus
            # holds: raw and converted produce the IDENTICAL result every time,
            # `\rho`/`\ddot{theta}` included, because the builder converts what
            # it is given.  So this is about what is RECORDED — the step's args
            # are the history — and three of the four substitution routes
            # already record the converted form.  This was the odd one out.
            _sr_lhs_raw, _sr_rhs_raw = operands_as_written(
                ((nlp("subst_replace", lang).search(alias_free_text)
                  or nlp("subst_replace_alt", "ja").search(alias_free_text))
                 if alias_free_text else None),
                _sr_pick, ("lhs", "rhs"), (_sr_lhs_raw, _sr_rhs_raw))
            return await _run_substitute_and_report(
                engine, loop, _sr_expr_id, _sr_lhs_raw, _sr_rhs_raw, registered)
    return None


async def _try_rule_set_notation(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2f. Rule-based set_notation short-circuit.
    # Detects "Leibniz/ドット記法にして" patterns and resolves them directly,
    # bypassing the LLM.  Only fires when a registered quantity name is found in
    # the text so ambiguous uses of "Leibniz" (e.g. "Leibnizの積分則") are passed
    # through to the LLM unchanged.
    # Notation-format keywords externalized to nl_patterns.json (session-language).
    _notation_fmt_rule: str | None = None
    if nlp("notation_leibniz_kw", lang).search(supervisor_text):
        _notation_fmt_rule = "leibniz"
    elif nlp("notation_dot_kw", lang).search(supervisor_text):
        _notation_fmt_rule = "dot"

    if _notation_fmt_rule is not None:
        _notation_name_rule: str | None = None
        for _qn in engine.symbol_context.registry:
            if re.search(
                r'(?<![A-Za-z0-9_])' + re.escape(_qn) + r'(?![A-Za-z0-9_])',
                supervisor_text
            ):
                _notation_name_rule = _qn
                break
        # Fallback 1: no quantity name in text, but a derivative expression is selected
        # → use its quantity_name (e.g. user selects E2=q̇ and says "leibniz記法にして")
        if _notation_name_rule is None and req.selected_expr_id:
            _sel = engine.expressions.get(req.selected_expr_id)
            if _sel and _sel.quantity_name and _sel.quantity_name in engine.symbol_context.registry:
                _notation_name_rule = _sel.quantity_name
        # Fallback 2: exactly one quantity registered — unambiguous even without selection
        if _notation_name_rule is None and len(engine.symbol_context.registry) == 1:
            _notation_name_rule = next(iter(engine.symbol_context.registry))
        if _notation_name_rule is not None:
            try:
                await loop.run_in_executor(
                    None, partial(engine.set_notation, _notation_name_rule, _notation_fmt_rule)
                )
            except Exception as e:
                return {"status": "failed", "error": str(e), "registered_expressions": registered}
            _updated = await loop.run_in_executor(
                None, partial(engine.rerender_notation_for_quantity, _notation_name_rule)
            )
            _nlabel = t("interpreter.notation_dot" if _notation_fmt_rule == "dot"
                         else "interpreter.notation_leibniz", lang)
            _msg = t("interpreter.notation_changed", lang, name=_notation_name_rule, label=_nlabel)
            engine.record_quantity_update(t("interpreter.notation_switch", lang), _msg)
            return {
                "status":    "confirmed",
                "operation": "set_notation",
                "message":   _msg,
                "updated_exprs": [{"expr_id": eid, "latex": lat} for eid, lat in _updated],
                "registered_expressions": registered,
            }
    return None


async def _try_rule_set_latex_name(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2g. Rule-based set_latex_name short-circuit.
    # Detects "LaTeX名を\phi にして" patterns and resolves them directly,
    # bypassing the LLM.  Requires (1) a LaTeX-name keyword, (2) a registered
    # quantity name, and (3) a LaTeX string (\cmd or \cmd{...}) in the text.
    _LATEX_STRING_RE  = re.compile(r'(\\[A-Za-z]+(?:\{[^{}]*\})*)')
    if nlp("latex_name_kw", lang).search(supervisor_text):
        _latex_qty_name: str | None = None
        for _qn in engine.symbol_context.registry:
            if re.search(
                r'(?<![A-Za-z0-9_])' + re.escape(_qn) + r'(?![A-Za-z0-9_])',
                supervisor_text
            ):
                _latex_qty_name = _qn
                break
        _m_latex_str = _LATEX_STRING_RE.search(supervisor_text)
        _latex_str_rule = _m_latex_str.group(1) if _m_latex_str else None
        if _latex_qty_name is not None and _latex_str_rule is not None:
            try:
                await loop.run_in_executor(
                    None, partial(engine.set_latex_name, _latex_qty_name, _latex_str_rule)
                )
            except Exception as e:
                return {"status": "failed", "error": str(e), "registered_expressions": registered}
            _msg = t("interpreter.latex_name_set", lang, name=_latex_qty_name, latex=_latex_str_rule)
            engine.record_quantity_update(t("interpreter.latex_name_label", lang), _msg)
            return {
                "status":    "confirmed",
                "operation": "set_latex_name",
                "message":   _msg,
                "registered_expressions": registered,
            }
    return None


async def _try_rule_combine(req, engine, loop, lang, supervisor_text, registered, _rule_order, interactive) -> "dict | None":
    # 2h. Rule-based combine_expressions short-circuit.
    # Detects "E1 と E2 を足す/引く/掛ける/割る" patterns and maps to combine_expressions.
    # Grammar externalized to nl_patterns.json["combine"] (session-language).
    # JA has consistent operand order (a=first, b=second) with the operator as a
    # keyword group `kw`; EN reverses for "subtract B from A" / "divide A by B",
    # so each EN verb has its own a_<op>/b_<op> groups placing the operands right.
    def _combine_kw_to_op(kw: str) -> str:
        k = kw.lower()
        if any(k.startswith(p) for p in ('足', 'たす', 'プラス', '+')):
            return '+'
        if any(k.startswith(p) for p in ('引', 'ひく', 'マイナス', '-')):
            return '-'
        if any(k.startswith(p) for p in ('掛', 'かけ', '×')):
            return '*'
        if any(k.startswith(p) for p in ('割', 'わ', '÷')):
            return '/'
        return '+'

    _m_combine = nlp("combine", lang).search(supervisor_text)
    if _m_combine:
        _cgd = _m_combine.groupdict()
        if _cgd.get("kw"):                       # JA: a=first, b=second, op from kw
            _comb_a, _comb_b = _cgd["a"], _cgd["b"]
            _comb_op = _combine_kw_to_op(_cgd["kw"])
        else:                                    # EN: operator-specific operand slots
            for _en_op, _en_sym in (("add", "+"), ("mul", "*"),
                                    ("sub", "-"), ("div", "/")):
                if _cgd.get("a_" + _en_op):
                    _comb_a = _cgd["a_" + _en_op]
                    _comb_b = _cgd["b_" + _en_op]
                    _comb_op = _en_sym
                    break
        active_ids = engine._active_expr_ids()
        if _comb_a in active_ids and _comb_b in active_ids:
            _comb_args = {"expr_a": _comb_a, "expr_b": _comb_b, "op": _comb_op}
            try:
                step, value = await loop.run_in_executor(
                    None, partial(engine.run_operation, "combine_expressions", _comb_args)
                )
            except Exception as e:
                return {"status": "failed", "error": str(e),
                        "registered_expressions": registered}
            _comb_out_id = step.output_refs[0] if step.output_refs else None
            _comb_expr   = engine.expressions.get(_comb_out_id) if _comb_out_id else None
            return {
                "status":    "completed",
                "operation": "combine_expressions",
                "expr_id":   _comb_out_id,
                "step_id":   step.step_id,
                "latex":     _comb_expr.latex    if _comb_expr else "",
                "canonical": _comb_expr.canonical if _comb_expr else value,
                "operation_label": step.visible_action,
                "registered_expressions": registered,
            }
    return None


_RULE_INTERCEPTORS: "list[Interceptor]" = [
    # R1d head batch: diff / transform-subst (fire first).
    Interceptor(
        "wrt_qdot",
        lambda c: _try_rule_wrt_qdot(c.req, c.engine, c.loop, c.lang, c.text,
                                     c.registered, c.rule_order, c.interactive),
        note="2e2: ∂/∂q̇ (differentiate wrt a generalized velocity); before the "
             "plain differentiate short-circuit so a q̇-derivative is not taken as "
             "an ordinary variable derivative."),
    Interceptor(
        "differentiate",
        lambda c: _try_rule_differentiate(c.req, c.engine, c.loop, c.lang, c.text,
                                          c.registered, c.rule_order, c.interactive),
        note="2e3: 「En を X で (偏/全)微分」 — direct differentiate (may prompt "
             "partial-vs-total when ambiguous)."),
    Interceptor(
        "transform_subst",
        lambda c: _try_rule_transform_subst(c.req, c.engine, c.loop, c.lang, c.text,
                                            c.registered, c.rule_order, c.interactive),
        note="2e3.4: multi-equation transform apply (operator vs scalar routing); "
             "before the single-source substitute paths."),
    Interceptor(
        "substitute_into",
        lambda c: _try_rule_substitute_into(c.req, c.engine, c.loop, c.lang, c.text,
                                            c.registered, c.rule_order, c.interactive, c.alias_free_text),
        note="2e3.4: 「En の X に Y を代入」 symbol→value (deterministic rhs parse "
             "for '/' operands; bails on operator symbols and E-id values)."),
    Interceptor(
        "expr_with_assignment",
        lambda c: _try_expr_with_assignment(c.engine, c.loop, c.lang, c.text,
                                            c.registered),
        note="`E1 with x = 2`。語ではなく**形**で読む——`with` を "
             "instruction_marker に足すと式の登録が止まる（式の行に6件出る）。"),
    # R1d tail batch: substitution / notation / combine.
    Interceptor(
        "dainyu",
        lambda c: _try_rule_dainyu(c.req, c.engine, c.loop, c.lang, c.text,
                                   c.registered, c.rule_order, c.interactive),
        note="2e3.5: 「En を Em に代入」 — substitute a solved equation En into Em."),
    Interceptor(
        "substitute",
        lambda c: _try_rule_substitute(c.req, c.engine, c.loop, c.lang, c.text,
                                       c.registered, c.rule_order, c.interactive, c.alias_free_text),
        note="2e4: 「En の A を B で置き換え」 — direct substitute (bails on operator "
             "symbols so index-aware operator_substitute handles those)."),
    Interceptor(
        "set_notation",
        lambda c: _try_rule_set_notation(c.req, c.engine, c.loop, c.lang, c.text,
                                         c.registered, c.rule_order, c.interactive),
        note="2f: Leibniz / dot notation switch for a registered quantity."),
    Interceptor(
        "set_latex_name",
        lambda c: _try_rule_set_latex_name(c.req, c.engine, c.loop, c.lang, c.text,
                                           c.registered, c.rule_order, c.interactive),
        note="2g: 「LaTeX名を \\phi にして」 — set a quantity's LaTeX name."),
    Interceptor(
        "combine",
        lambda c: _try_rule_combine(c.req, c.engine, c.loop, c.lang, c.text,
                                    c.registered, c.rule_order, c.interactive),
        note="2h: 「E1 と E2 を 足す/引く/掛ける/割る」 → combine_expressions."),
]
