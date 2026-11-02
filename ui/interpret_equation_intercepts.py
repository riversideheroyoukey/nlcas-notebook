"""H-5〜H-9 equation-operation intercepts (RHS ops, both-sides ops, solve,
L-1 inverse transforms).

Extracted verbatim from interpret_pipeline.py (refactor Stage 2,
2026-07-04).  run_interpret_pipeline imports this back; behaviour is
unchanged.
"""
from __future__ import annotations

import logging

from notebook.runtime.shared_parser import split_relation, strip_lhs
from notebook.runtime.lhs_formatting import _rhs_of_latex
import re
from functools import partial
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from ui.interpret_common import (  # noqa: F401
    _HAS_CJK_RE,
    _MULTI_WORD_RE,
    _expand_expr_ids,
    _expand_symbol_refs,
    _find_expr_by_canonical,
    FULLWIDTH_DIGITS,
    number_literal,
    number_word,
    pick_group,
)
from ui.interpret_equation_ops import (  # noqa: F401
    rhs_operation_for,
    _both_sides_new_lhs,
    _match_rhs_op,
    side_of,
    _parse_both_sides,
    _parse_rhs_to_operation,
    _parse_solve,
)
from ui.interpret_l1 import (  # noqa: F401
    _forward_has_operators,
    _gram_rows,
    _infer_old_basis,
    _is_verify_request,
    _new_basis_symbols,
    _operator_statistics,
    _parse_apply_constraints,
    _parse_canonical_constraints,
    _parse_inverse_transform,
    _parse_maxima_matrix,
    _parse_metric_constraints,
)
from ui.interpret_registry import (
    Interceptor, InterceptorContext, run_interceptors,
)

#: 左辺が「名前」でなく式か —— 演算子か数字か空白を含めば式である。
#: 名前だけの左辺（H-1 の `E = …` の `E`）を簡約するのは意味が無いので断る。
_LHS_IS_COMPOUND_RE = re.compile(r"[+\-*/^()\s]|\d")

_logger = logging.getLogger(__name__)


async def _try_equation_intercepts(engine, loop, lang, supervisor_text, registered,
                                   alias_free_text="") -> "dict | None":
    """Equation / L-1 rule-based intercepts (H-5z/H-5/H-7 RHS ops, L-1
    metric/canonical/inverse transforms, H-9 solve, H-8 both-sides, pending-op
    follow-up).  Returns a response dict to short-circuit, or None to continue.
    R1d (docs/27): the former giant if/return chain is now an ordered registry —
    the firing order is the data in _EQUATION_INTERCEPTORS, not code position."""
    return await run_interceptors(
        InterceptorContext(engine=engine, loop=loop, lang=lang,
                           text=supervisor_text, registered=registered,
                           alias_free_text=alias_free_text),
        _EQUATION_INTERCEPTORS)


# ---------------------------------------------------------------------------
# R1d head batch (docs/27): the RHS-op + metric/apply constraint head blocks,
# extracted verbatim.  They fire FIRST, so they head the list.
# ---------------------------------------------------------------------------
async def _try_eq_rhs_set(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-h5z. "En の右辺を X とする" — register a new equation "RHS = X".
    # X can be any expression: 0, 1, omega^2, E7, etc.
    # Examples:
    #   "E12 の右辺を０とする"  → l²m(q̈+ω²q) = 0
    #   "E5 の右辺を１とする"   → normalize: expr = 1
    # Externalized to i18n/nl_patterns.json["rhs_set"] (named groups eid, value).
    _rhs_set_m = nlp("rhs_set", lang).search(supervisor_text)
    if _rhs_set_m:
        # 英語は語順が枝ごとに違う（`set X to V` / `let X be V`）ので群を綴り分けて
        # あり、値の入っている枝を `pick_group` が取る（`re` は alternation を
        # またいだ同名グループを禁じる）。
        _rzeid    = pick_group(_rhs_set_m, "eid")
        _rz_val_raw = (pick_group(_rhs_set_m, "value") or "").strip()  # "０", "0", "omega^2"
        # H-5z は右辺を置く操作。左辺を置けと言われたら**断る**——「左辺を 0 とする」
        # は等式そのものを書き換える別の要求で、いま在る機構は右辺しか置けない
        # （黙って右辺を置くと、研究者が書いたのと違う式が登録される）。
        if side_of(_rhs_set_m) == "left":
            return {"status": "failed",
                    "error": t("interpreter.set_left_side_unsupported", lang,
                               expr_id=_rzeid),
                    "registered_expressions": registered}
        _rz_src   = engine.expressions.get(_rzeid)
        if _rz_src:
            # Extract the RHS of En (the expression whose RHS we're constraining)
            _rz_canonical = _rz_src.canonical or ""
            _rz_latex     = _rz_src.latex     or ""
            # H-5z registers a NEW EXPRESSION from these halves, so a mis-split
            # is not a command that fails and is gone — it is PERSISTED.
            # Measured 2026-08-13 on `\sum_{i=1}^{3} c_i = u a + v b`, 「E1 の
            # 右辺を 0 とする」 completed and stored
            #     canonical  1}^3*c_i = u*a + v*b = 0
            #     latex      1}^{3} c_i = u a + v b = 0
            # — a fragment head and two `=`, written into the session file, the
            # timeline and the exported .tex, and read by every later operation
            # that references it.
            _rel_can = split_relation(_rz_canonical)
            if _rel_can is not None and "=" in _rel_can[1]:
                _rz_rhs_can = _rel_can[2]
            elif _rz_canonical:
                # Canonical (no '=') is already the RHS; latex carries the full
                # "LHS = RHS" display form.  Use canonical directly to avoid
                # re-parsing LaTeX (which can produce incorrect Maxima syntax,
                # e.g. m(...) misread as function call).
                _rz_rhs_can = _rz_canonical
            else:
                # Fall back: no canonical, derive from latex
                # Unreachable in practice — measured over the 19 saved sessions,
                # none of their 75 expressions has an empty canonical — but it is
                # the same question, so it gets the same answer.
                _rz_rhs_lat = strip_lhs(_rz_latex)
                _rz_rhs_can = engine._handler.latex_to_canonical(_rz_rhs_lat) or _rz_rhs_lat

            # Convert the value X to canonical (handle 全角０, E-IDs, LaTeX)
            # 全角→半角と、語で書かれた数（`number_words`——ここには全角の写しが
            # 2本在った）。**綴りごと**畳むのは、この下で新しい式の LaTeX を
            # `_rz_val_raw` から組むから: 値の側だけ直すと canonical は `0`、
            # 画面と .tex は `= zero` になり、表示と計算が割れる。
            _rz_val_raw = number_literal(_rz_val_raw.translate(FULLWIDTH_DIGITS)) \
                or _rz_val_raw.translate(FULLWIDTH_DIGITS)
            _rz_val_expanded = _expand_expr_ids(_rz_val_raw, engine)
            _rz_val_can = engine._handler.latex_to_canonical(_rz_val_expanded) or _rz_val_expanded

            if _rz_rhs_can and _rz_rhs_can != _rz_val_can:
                # Build canonical for new equation: "RHS_can = val_can"
                # Per H-1 convention, canonical stores the LHS expression (the constrained one)
                # and latex shows "LHS_expr = val".
                # This one is the DISPLAY half of the same registration.
                _rz_rhs_lat_disp = strip_lhs(_rz_latex)
                _rz_new_latex  = f"{_rz_rhs_lat_disp} = {_rz_val_raw}"
                _rz_new_can    = f"{_rz_rhs_can} = {_rz_val_can}"
                # Only de-duplicate on the FULL equation form "RHS = value".
                # Do NOT fall back to checking the plain RHS expression — that
                # would incorrectly treat E12 (the RHS alone) as a duplicate of
                # the new constraint equation "E12_RHS = 0".
                _rz_existing = _find_expr_by_canonical(engine, _rz_new_can)
                if _rz_existing:
                    # Equation already exists — return it as the result.
                    registered.append({
                        "expr_id":           _rz_existing,
                        "canonical":         engine.expressions[_rz_existing].canonical,
                        "latex":             engine.expressions[_rz_existing].latex,
                        "step_id":           engine.expressions[_rz_existing].created_by_step,
                        "undefined_symbols": engine.expressions[_rz_existing].undefined_symbols,
                    })
                    _op_label = t("interpreter.rhs_set_value", lang).format(
                        eid=_rzeid, val=_rz_val_raw)
                    _rz_ex_expr = engine.expressions[_rz_existing]
                    return {
                        "status": "completed",
                        "operation": "rhs_set",
                        "expr_id":   _rz_existing,
                        "step_id":   _rz_ex_expr.created_by_step,
                        "latex":     _rz_ex_expr.latex,
                        "registered_expressions": registered,
                        "supervision": _op_label,
                    }
                else:
                    try:
                        _rz_expr = await loop.run_in_executor(
                            None,
                            partial(engine.register_expression,
                                    _rz_new_can, latex=_rz_new_latex,
                                    skip_cas_bind=True),  # constraint eq, not a definition
                        )
                        registered.append({
                            "expr_id":           _rz_expr.display_id,
                            "canonical":         _rz_expr.canonical,
                            "latex":             _rz_expr.latex,
                            "step_id":           _rz_expr.created_by_step,
                            "undefined_symbols": _rz_expr.undefined_symbols,
                        })
                        _op_label = t("interpreter.rhs_set_value", lang).format(
                            eid=_rzeid, val=_rz_val_raw)
                        return {
                            "status": "completed",
                            "operation": "rhs_set",
                            "expr_id":   _rz_expr.display_id,
                            "step_id":   _rz_expr.created_by_step,
                            "latex":     _rz_expr.latex,
                            "registered_expressions": registered,
                            "supervision": _op_label,
                        }
                    except Exception:
                        # Rule-based H-5z handling failed; fall through to LLM.
                        # Logged so a routing/build bug is not silently masked.
                        _logger.debug("H-5z rule path failed; falling through to LLM",
                                      exc_info=True)
    return None


async def _try_eq_evaluate_rhs(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-h5. H-5: "En の右辺を計算して" / "Simplify the right-hand side of En" —
    # evaluate the RHS of a registered equation.
    # Detected before LLM to avoid ambiguity; dispatches through run_operation.
    _h5_pair = _match_rhs_op(supervisor_text, lang)
    if _h5_pair:
        _h5_eid, _h5_keyword, _h5_side = _h5_pair  # ("E5", "整理", "right")
        _h5_src = engine.expressions.get(_h5_eid)
        # H-1 expressions store only the RHS in canonical (no "=").
        # Detect them via the latex field which carries "LHS = RHS".
        _h5_canonical = _h5_src.canonical or "" if _h5_src else ""
        _h5_latex     = _h5_src.latex     or "" if _h5_src else ""
        # Bracket-aware throughout: `"=" in t` reads a bounded sum's own
        # subscript as the equation, so `\sum_{i=1}^{3} i^2 = 14` was split into
        # `\sum_{i` and `1}^3*i^2 = 14` and Maxima was handed the tail — it
        # cannot read that, and its scratch name `_res_` came back as the
        # result (measured 2026-08-11; the same defect engine.py and
        # lhs_formatting had already been repaired for).
        _h5_can_rel = split_relation(_h5_canonical)
        _h5_lat_rel = split_relation(_h5_latex)
        _h5_is_eq = ((_h5_can_rel is not None and _h5_can_rel[1] == "=")
                     or (_h5_lat_rel is not None and _h5_lat_rel[1] == "="))
        if _h5_src and _h5_is_eq:
            if _h5_can_rel is not None and _h5_can_rel[1] == "=":
                _h5_lhs_raw, _, _h5_rhs = _h5_can_rel
            else:
                # H-1 expression: canonical IS the RHS; LHS is in latex
                _h5_rhs = _h5_canonical
                _h5_lhs_raw = _h5_lat_rel[0] if _h5_lat_rel else _h5_latex
            # **どちらの辺か。** 2026-09-05 まで右辺しか読めず、`left-hand side` と
            # 書いた文は別名走査に落ちて**式全体**に操作が適用されていた（成功つき）。
            # 左辺が**名前**のときだけは断る——H-1 の等式は canonical に右辺しか
            # 持たず（CLAUDE.md）、左辺は `E` のような名前なので簡約する意味が無い。
            if _h5_side == "left":
                _h5_lhs_side = (_h5_lhs_raw or "").strip()
                if not _LHS_IS_COMPOUND_RE.search(_h5_lhs_side):
                    return {"status": "failed",
                            "error": t("interpreter.lhs_is_a_name", lang,
                                       expr_id=_h5_eid, lhs=_h5_lhs_side),
                            "registered_expressions": registered}
                _h5_rhs = _h5_lhs_side
                _h5_lat_rel = None      # 表示は正準形から（左辺の latex は別に無い）
            # Expand any E-ID references still present in the chosen side
            _h5_rhs = _expand_expr_ids(_h5_rhs, engine)
            _h5_rhs = _expand_symbol_refs(_h5_rhs, engine)
            # The scaffolding below is registered with a DISPLAY latex, not the
            # canonical.  `internal=True` keeps it off the timeline, but its latex
            # still reaches the screen as the RESULT's left-hand label, and the
            # canonical is not LaTeX: measured on the entropy of a fair die, the
            # published card read `-\sum_{x=1}^6*((1)/(6))*\log ((1)/(6)) = \log 6`
            # — CAS `*` and an unbraced range, in the researcher's own paper.  The
            # source's own RHS latex is exactly what they wrote.
            _h5_rhs_disp = _rhs_of_latex(_h5_latex) if _h5_lat_rel else _h5_rhs

            # H-7: keyword → explicit operation (expand / factor / simplify / …)
            _h5_explicit_op: str | None = rhs_operation_for(_h5_keyword)

            try:
                if _h5_explicit_op is not None:
                    # Register RHS as expression, then apply the named operation
                    _existing_rhs = _find_expr_by_canonical(engine, _h5_rhs)
                    if _existing_rhs:
                        _h5_rhs_id = _existing_rhs
                    else:
                        _h5_rhs_expr = await loop.run_in_executor(
                            None,
                            # Same scaffolding as the H-8 site below (which already
                            # marks it): the researcher asked to OPERATE on the right
                            # side, not to see the bare right side.  Published, it was
                            # worse than noise — `latex=<canonical>` puts CAS text in a
                            # LaTeX slot, and Maxima's `%e` opens a LaTeX COMMENT, so
                            # the rest of that line vanished from the exported .tex
                            # (measured in scenario_damped_oscillator_numeric_jp E6).
                            partial(engine.register_expression, _h5_rhs, latex=_h5_rhs_disp,
                                    internal=True),
                        )
                        _h5_rhs_id = _h5_rhs_expr.display_id
                    _h5_step, _ = await loop.run_in_executor(
                        None,
                        # The operation really does run on the scaffold, so the
                        # history is right — but the LABEL built from it names
                        # `_T1`, a key the researcher has never seen.  Only this
                        # handler knows the scaffold is E-n's right-hand side.
                        partial(engine.run_operation, _h5_explicit_op,
                                {"expr": _h5_rhs_id, "label_rhs_of": _h5_eid}),
                    )
                    _h5_out_id = _h5_step.output_refs[0] if _h5_step.output_refs else None
                    _h5_out = engine.expressions.get(_h5_out_id) if _h5_out_id else None
                else:
                    # 計算/評価: try to detect operation from RHS canonical pattern
                    op_info = _parse_rhs_to_operation(_h5_rhs)
                    if op_info is not None:
                        _h5_op, _h5_args = op_info
                        for _fk in list(_h5_args.keys()):
                            _fv = _h5_args[_fk]
                            if not (isinstance(_fv, str) and _fv in engine.expressions):
                                _fv_stripped = _fv.strip()
                                _existing = _find_expr_by_canonical(engine, _fv_stripped)
                                if _existing:
                                    _h5_args[_fk] = _existing
                                else:
                                    _tmp_expr = await loop.run_in_executor(
                                        None,
                                        partial(engine.register_expression, _fv_stripped,
                                                latex=_fv_stripped),
                                    )
                                    _h5_args[_fk] = _tmp_expr.display_id
                        _h5_step, _ = await loop.run_in_executor(
                            None, partial(engine.run_operation, _h5_op, _h5_args)
                        )
                        _h5_out_id = _h5_step.output_refs[0] if _h5_step.output_refs else None
                        _h5_out = engine.expressions.get(_h5_out_id) if _h5_out_id else None
                    else:
                        # Fallback: register RHS then simplify
                        _existing_rhs = _find_expr_by_canonical(engine, _h5_rhs)
                        if _existing_rhs:
                            _h5_rhs_id = _existing_rhs
                        else:
                            _h5_rhs_expr = await loop.run_in_executor(
                                None,
                                # Scaffolding — see the sibling branch above.
                                partial(engine.register_expression, _h5_rhs,
                                        latex=_h5_rhs_disp, internal=True),
                            )
                            _h5_rhs_id = _h5_rhs_expr.display_id
                        _h5_step, _ = await loop.run_in_executor(
                            None, partial(engine.run_operation, "simplify",
                                          # Same scaffold, same hint as the
                                          # named-operation branch above.
                                          {"expr": _h5_rhs_id,
                                           "label_rhs_of": _h5_eid})
                        )
                        _h5_out_id = _h5_step.output_refs[0] if _h5_step.output_refs else None
                        _h5_out = engine.expressions.get(_h5_out_id) if _h5_out_id else None
            except Exception as _h5_exc:
                return {"status": "failed", "error": str(_h5_exc),
                        "registered_expressions": registered}

            if _h5_out is None:
                return {"status": "failed",
                        "error": t("interpreter.rhs_eval_failed", lang),
                        "registered_expressions": registered}

            # Attach the source equation's LHS to the result so the timeline
            # displays  "−i dSx₁/dt = −2BγiSy₁"  rather than bare "−2BγiSy₁".
            _h5_final_latex = _h5_out.latex
            if _h5_lhs_raw and "=" not in _h5_out.latex:
                _h5_final_latex = f"{_h5_lhs_raw} = {_h5_out.latex}"
                engine.expressions[_h5_out_id] = _h5_out.model_copy(
                    update={"latex": _h5_final_latex}
                )

            registered.append({
                "expr_id":   _h5_out.display_id,
                "canonical": _h5_out.canonical,
                "latex":     _h5_final_latex,
            })
            return {
                "status":    "completed",
                "operation": "evaluate_rhs",
                "expr_id":   _h5_out.display_id,
                "canonical": _h5_out.canonical,
                "latex":     _h5_final_latex,
                "registered_expressions": registered,
            }
    return None


async def _try_eq_metric_constraints(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-L1g2. L-1 (non-operator metric): derive the metric-preservation constraints
    # M G Mᵀ = G from the forward transform and the declared inner products.
    _mc_eids = _parse_metric_constraints(supervisor_text, lang)
    if _mc_eids:
        _gram = dict(getattr(engine, "gram_inner_products", {}) or {})
        if not _gram:
            return {"status": "failed",
                    "error": t("interpreter.no_inner_products", lang),
                    "registered_expressions": registered}
        # old basis = first-seen symbols across the inner-product declarations
        _basis: list[str] = []
        for (_a, _b) in _gram:
            for _s in (_a, _b):
                if _s not in _basis:
                    _basis.append(_s)
        _mc_args = {
            "forward": ", ".join(_mc_eids),
            "basis": ", ".join(_basis),
            "gram": _gram_rows(_gram, _basis),
        }
        try:
            _mc_step, _mc_val = await loop.run_in_executor(
                None, partial(engine.run_operation,
                              "derive_metric_constraints", _mc_args)
            )
        except Exception as _mc_exc:
            return {"status": "failed", "error": str(_mc_exc),
                    "registered_expressions": registered}
        # The result is a symmetric matrix M G Mᵀ − G; its non-zero upper-triangular
        # entries are the constraints.  Register each distinct one as "entry = 0".
        _mc_ids = _mc_step.output_refs or []
        _mc_src = engine.expressions.get(_mc_ids[0]) if _mc_ids else None
        # Prefer the raw CAS return value (Julia's stored canonical is opaque "_nb_E1";
        # only the returned value carries the actual matrix string).
        _mat = _parse_maxima_matrix(
            (_mc_val or "") or (_mc_src.canonical if _mc_src else ""))
        if _mc_ids and _mat:
            engine.expressions.pop(_mc_ids[0], None)   # drop the raw matrix
        _seen_mc: set[str] = set()
        _mc_any = False
        for _r in range(len(_mat or [])):
            for _c in range(_r, len(_mat[_r])):
                _entry = (_mat[_r][_c] or "").strip()
                _norm = _entry.replace(" ", "")
                if _norm in ("0", "-0", "") or _norm in _seen_mc:
                    continue
                _seen_mc.add(_norm)
                _cexpr = await loop.run_in_executor(
                    None, partial(engine.register_dispersion,
                                  f"{_entry} = 0", {"kind": "metric_constraint"}))
                registered.append({"expr_id": _cexpr.display_id,
                                   "canonical": _cexpr.canonical, "latex": _cexpr.latex})
                if not hasattr(engine, "canonical_constraint_ids"):
                    engine.canonical_constraint_ids = []
                engine.canonical_constraint_ids.append(_cexpr.display_id)
                _mc_any = True
        if not _mc_any:
            return {"status": "completed", "operation": "derive_metric_constraints",
                    "message": t("interpreter.metric_already_preserved", lang),
                    "registered_expressions": registered}
        return {"status": "completed", "operation": "derive_metric_constraints",
                "expr_id": registered[-1]["expr_id"] if registered else None,
                "registered_expressions": registered}
    return None


async def _try_eq_apply_constraints(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-L1a. L-1 step 4 (apply): reduce a target expression under derived canonical
    # constraints (ratsubst / Symbolics.substitute), folding e.g. the Bogoliubov
    # inverse denominator u²+v² → 1.
    _ac = _parse_apply_constraints(supervisor_text, lang)
    if _ac:
        _ac_rels = _ac["relations"]
        if _ac_rels is None:
            _ac_rels = list(getattr(engine, "canonical_constraint_ids", []) or [])
        if not _ac_rels:
            return {"status": "failed",
                    "error": t("interpreter.no_active_constraints", lang),
                    "registered_expressions": registered}
        _ac_args = {"expr": _ac["target"], "relations": ", ".join(_ac_rels)}
        try:
            _ac_step, _ = await loop.run_in_executor(
                None, partial(engine.run_operation,
                              "apply_canonical_constraints", _ac_args)
            )
        except Exception as _ac_exc:
            return {"status": "failed", "error": str(_ac_exc),
                    "registered_expressions": registered}
        _ac_ids = _ac_step.output_refs or []
        for e in (engine.expressions[i] for i in _ac_ids if i in engine.expressions):
            registered.append({"expr_id": e.display_id,
                               "canonical": e.canonical, "latex": e.latex})
        return {"status": "completed", "operation": "apply_canonical_constraints",
                "expr_id": _ac_ids[0] if _ac_ids else None,
                "registered_expressions": registered}
    return None


# ---------------------------------------------------------------------------
# R1d (docs/27): equation / L-1 tail interceptors, extracted verbatim from
# _try_equation_intercepts (each original if-block wrapped in an async def that
# returns None on fall-through; indentation unchanged).  The ordered list below
# IS the firing order.
# ---------------------------------------------------------------------------
async def _try_eq_canonical_constraints(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-L1c. L-1 step 4: canonical constraints of an operator transform.  For each
    # forward def new_i = RHS_i(old), compute {new_i, new_i†}_± (anticommutator for
    # fermions, commutator for bosons) and register the constraint = 1 (Bogoliubov:
    # u²+v²=1 fermions, u²−v²=1 bosons).  The metric η is implicit in the choice of
    # (anti)commutator, which follows from the operators' statistics.
    _cc_eids = _parse_canonical_constraints(supervisor_text, lang)
    if _cc_eids:
        _cc_verify = _is_verify_request(supervisor_text, lang)
        _handler = getattr(engine, "_handler", None)
        if _handler is None or not getattr(_handler, "supports_operator_adjoint", False):
            return {"status": "failed",
                    "error": t("interpreter.canonical_constraint_unsupported", lang),
                    "registered_expressions": registered}
        # Collect (rhs, adjoint) of each new operator.  The full canonical condition
        # is the pairwise matrix {new_i, new_j†}_± = δ_ij plus the anomalous
        # {new_i, new_j}_± = 0 — i.e. M†ηM = η in component form.
        _news: list[tuple[str, str]] = []
        for _eid in _cc_eids:
            _expr = engine.expressions.get(_eid)
            if _expr is None:
                continue
            _latex = _expr.latex or ""
            # The latex arm is the empty-canonical fallback again (unreachable as
            # measured); `strip_lhs` returns the text unchanged when it is not an
            # equation, which is what the old `else _latex` did.
            _rhs = _expr.canonical or strip_lhs(_latex)
            _adj = _handler.adjoint_canonical(_rhs)
            if _adj:
                _news.append((_rhs, _adj))
        if not _news:
            return {"status": "failed",
                    "error": t("interpreter.canonical_constraint_none", lang),
                    "registered_expressions": registered}
        _op = ("anticommutator" if _operator_statistics(engine, _news[0][0]) == "fermi"
               else "commutator")
        _seen: set[str] = set()

        async def _bracket(_a: str, _b: str, _target: str) -> "str | None":
            """Run [a,b]±, drop it if the c-number already equals *target* or is a
            duplicate, else register it as the constraint 'c = target'."""
            try:
                _st, _ = await loop.run_in_executor(
                    None, partial(engine.run_operation, _op, {"a": _a, "b": _b})
                )
            except Exception:
                return None
            _ids = _st.output_refs or []
            if not _ids:
                return None
            _oid = _ids[0]
            _r = engine.expressions.get(_oid)
            if _r is None:
                return None
            try:
                _clean = _handler.latex_to_canonical(_r.latex) or _r.canonical
            except Exception:
                _clean = _r.canonical
            _norm = (_clean or "").replace(" ", "")
            _is_trivial = (_target == "0" and _norm in ("0", "-0", "")) or _norm == _target
            _key = f"{_norm}={_target}"
            if _is_trivial or _key in _seen:
                engine.expressions.pop(_oid, None)
                return None
            _seen.add(_key)
            engine.expressions[_oid] = _r.model_copy(update={
                "latex": f"{_r.latex} = {_target}",
                "canonical": f"{_clean} = {_target}",
            })
            _r2 = engine.expressions[_oid]
            registered.append({"expr_id": _r2.display_id,
                               "canonical": _r2.canonical, "latex": _r2.latex})
            if not hasattr(engine, "canonical_constraint_ids"):
                engine.canonical_constraint_ids = []
            if _oid not in engine.canonical_constraint_ids:
                engine.canonical_constraint_ids.append(_oid)
            return _oid

        _cc_any = False
        _n = len(_news)
        for _i in range(_n):
            for _j in range(_i, _n):
                # {new_i, new_j†} = δ_ij
                if await _bracket(_news[_i][0], _news[_j][1],
                                  "1" if _i == _j else "0") is not None:
                    _cc_any = True
                # anomalous {new_i, new_j} = 0 — only off-diagonal; {new_i, new_i}
                # vanishes identically (fermion/boson) so it is never a constraint.
                if _i < _j and await _bracket(
                        _news[_i][0], _news[_j][0], "0") is not None:
                    _cc_any = True
        if not _cc_any:
            # Every (anti)commutator already equals its canonical target → the
            # transform is canonical with no extra condition.
            return {"status": "completed", "operation": "verify_canonical",
                    "message": t("interpreter.canonical_is_canonical", lang),
                    "registered_expressions": registered}
        return {
            "status": "completed",
            "operation": "verify_canonical" if _cc_verify else "derive_canonical_constraints",
            "message": (t("interpreter.canonical_conditional", lang)
                        if _cc_verify else None),
            "expr_id": registered[-1]["expr_id"] if registered else None,
            "registered_expressions": registered,
        }
    return None


async def _try_eq_inverse_transform(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-L1. L-1: inverse linear basis transform.  old = M^{-1} new is the solution
    # of the forward system {new_i = Σ M_ij old_j} for the old basis — so we identify
    # the old basis and delegate to solve_system (reusing its result registration).
    _l1_parsed = _parse_inverse_transform(supervisor_text, lang)
    if _l1_parsed:
        _l1_eids = _l1_parsed["eids"]
        _l1_vars = _l1_parsed["vars"]
        if len(_l1_eids) < 2:
            return {"status": "failed",
                    "error": t("interpreter.inverse_transform_need_eqs", lang),
                    "registered_expressions": registered}
        if not _l1_vars:
            _l1_vars = _infer_old_basis(engine, _l1_eids)
        if not _l1_vars:
            engine.pending_op = {
                "operation": "build_inverse_transform",
                "args": {
                    "equations": ", ".join(_l1_eids),
                    "vars": "",
                    "new_basis": ", ".join(_new_basis_symbols(engine, _l1_eids)),
                },
                "missing": ["vars"],
            }
            _l1_prompt = ("interpreter.inverse_transform_operator_prompt"
                          if _forward_has_operators(engine, _l1_eids)
                          else "interpreter.inverse_transform_basis_prompt")
            return {
                "status": "needs_confirmation",
                "operation": "build_inverse_transform",
                "confirmation_prompt": t(_l1_prompt, lang),
                "registered_expressions": registered,
            }
        _l1_args = {
            "equations": ", ".join(_l1_eids),
            "vars": ", ".join(_l1_vars),
            "new_basis": ", ".join(_new_basis_symbols(engine, _l1_eids)),
        }
        try:
            _l1_step, _ = await loop.run_in_executor(
                None, partial(engine.run_operation, "build_inverse_transform", _l1_args)
            )
        except Exception as _l1_exc:
            return {"status": "failed", "error": str(_l1_exc),
                    "registered_expressions": registered}
        _l1_ids = _l1_step.output_refs or []
        for e in (engine.expressions[i] for i in _l1_ids if i in engine.expressions):
            registered.append({"expr_id": e.display_id,
                               "canonical": e.canonical, "latex": e.latex})
        return {"status": "completed", "operation": "build_inverse_transform",
                "expr_id": _l1_ids[0] if _l1_ids else None,
                "registered_expressions": registered}
    return None


async def _try_eq_solve(engine, loop, lang, supervisor_text, registered,
                        alias_free_text="") -> "dict | None":
    # 2b-h9. H-9: solve equations.  The SOLVE-FOR variables are re-read from
    # the researcher's own text — see `_parse_solve`.
    _h9_parsed = _parse_solve(supervisor_text, lang, alias_free_text)
    if _h9_parsed:
        _h9_op, _h9_args = _h9_parsed
        # No solve-for variable named → ask, rather than run without one.
        # This branch used to say `_h9_op == "solve_system"`, so the SINGLE
        # equation fell through to `run_operation` with an empty `var` and came
        # back as a failure — the same question ("which variable?") answered two
        # ways inside one family (measured 2026-09-05).  The field's name is the
        # only thing that differs between them.
        _h9_missing = "vars" if _h9_op == "solve_system" else "var"
        if not _h9_args.get(_h9_missing):
            engine.pending_op = {
                "operation": _h9_op,
                "args": _h9_args,
                "missing": [_h9_missing],
            }
            return {
                "status": "needs_confirmation",
                "operation": _h9_op,
                "confirmation_prompt": t("interpreter.solve_vars_prompt", lang),
                "registered_expressions": registered,
            }
        try:
            _h9_step, _h9_raw = await loop.run_in_executor(
                None, partial(engine.run_operation, _h9_op, _h9_args)
            )
        except Exception as _h9_exc:
            return {"status": "failed", "error": str(_h9_exc),
                    "registered_expressions": registered}

        _h9_out_ids = _h9_step.output_refs or []
        _h9_exprs = [engine.expressions[i] for i in _h9_out_ids if i in engine.expressions]
        for _i, e in enumerate(_h9_exprs):
            registered.append({
                "expr_id":   e.display_id,
                "canonical": e.canonical,
                "latex":     e.latex,
                # Solve returns SEVERAL roots, so its result cannot be carried by
                # the response's single `expr_id` — every root is appended here,
                # and the UI renders those first and then skips `data` as already
                # shown (`notebook_main.js`).  So the operation's label has to
                # travel with the FIRST root or it never reaches the screen:
                # measured 2026-08-27, every one of the corpus's 30 solve lines
                # drew an UNLABELLED card, in both languages.  The later roots
                # are further answers to the same operation, not further
                # operations, so only the first carries it.
                **({"operation_label": _h9_step.visible_action} if _i == 0 else {}),
            })
        return {
            "status":    "completed",
            "operation": _h9_op,
            "expr_id":   _h9_out_ids[0] if _h9_out_ids else None,
            "operation_label": _h9_step.visible_action,
            "registered_expressions": registered,
        }
    return None


async def _try_eq_both_sides(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-h8s. A SIDE ("左から掛ける" / "from the left") means the product does not
    # commute, so this is not H-8's text rewrite — it is the real operation, with
    # the side handed to the backend (which has a template per side).  H-8 builds
    # a new RHS as text and a LABEL for the left side; for a matrix equation the
    # left side is where the work happens: `(1-G^0Σ)^{-1}` on the left of
    # `(1-G^0Σ)G = G^0` reduces to `G`, and Maxima does that reduction itself.
    _h8s = nlp("both_sides_multiply_side", lang).search(supervisor_text)
    if _h8s and _h8s.group("eid") in engine.expressions:
        _side = "left" if _h8s.group("side") in ("左", "left") else "right"
        _factor = (_h8s.group("operand") or "").strip()
        _factor = _expand_expr_ids(_factor, engine)
        _factor = _expand_symbol_refs(_factor, engine)
        if "\\" in _factor:
            # The researcher's own spelling arrives as LaTeX; the equation is
            # already canonical.  Same two doors, same order, as the solve-for
            # variable (`_build_solve_equation`).
            _factor = engine._handler.latex_to_canonical(_factor) or _factor
        try:
            _s_step, _s_val = await loop.run_in_executor(
                None, partial(engine.run_operation, "multiply_both_sides",
                              {"expr": _h8s.group("eid"), "factor": _factor,
                               "side": _side}))
        except Exception as _s_exc:
            return {"status": "failed", "error": str(_s_exc),
                    "registered_expressions": registered}
        _s_ids = _s_step.output_refs or []
        _s_expr = engine.expressions.get(_s_ids[0]) if _s_ids else None
        return {
            "status": "completed", "operation": "multiply_both_sides",
            "expr_id": _s_ids[0] if _s_ids else None,
            "step_id": _s_step.step_id,
            "latex": _s_expr.latex if _s_expr else "",
            "canonical": _s_expr.canonical if _s_expr else _s_val,
            "operation_label": _s_step.visible_action,
            "registered_expressions": registered,
        }
    # 2b-h8. H-8: "En の両辺を X で割る/にかける/に足す/から引く"
    _h8_parsed = _parse_both_sides(supervisor_text, lang)
    if _h8_parsed:
        _h8_eid, _h8_op, _h8_operand = _h8_parsed
        _h8_src = engine.expressions.get(_h8_eid)
        _h8_canonical = _h8_src.canonical or "" if _h8_src else ""
        _h8_latex     = _h8_src.latex     or "" if _h8_src else ""
        # Bracket-aware, for the same reason as H-5 above.
        _h8_can_rel = split_relation(_h8_canonical)
        _h8_lat_rel = split_relation(_h8_latex)
        _h8_is_eq = ((_h8_can_rel is not None and _h8_can_rel[1] == "=")
                     or (_h8_lat_rel is not None and _h8_lat_rel[1] == "="))
        if _h8_src and _h8_is_eq:
            # Extract RHS (same logic as H-5)
            if _h8_can_rel is not None and _h8_can_rel[1] == "=":
                _, _, _h8_rhs = _h8_can_rel
            else:
                _h8_rhs = _h8_canonical
            # The LHS here is used for ONE thing — the label the result is
            # displayed under — so it is taken from the LATEX side whenever
            # there is one.  Read off the canonical it showed the researcher
            # `\sum_{i=1}^3*i^2` (CAS spelling, literal `*`) where they had
            # written `\sum_{i=1}^{3} i^2`.
            _h8_lhs = (_h8_lat_rel[0] if _h8_lat_rel is not None
                       else (_h8_can_rel[0] if _h8_can_rel is not None
                             else _h8_latex))
            _h8_rhs = _expand_expr_ids(_h8_rhs, engine)
            _h8_rhs = _expand_symbol_refs(_h8_rhs, engine)
            _h8_operand_display = _h8_operand  # original latex form for the LHS label
            if _h8_operand:
                _h8_operand = _expand_expr_ids(_h8_operand.strip(), engine)
                _h8_operand = _expand_symbol_refs(_h8_operand, engine)
                _h8_operand_display = _h8_operand
                # Convert a LaTeX operand (e.g. \lambda — Greek input is normalized to
                # \lambda upstream) to a CAS-valid canonical token (lambda); otherwise
                # the new RHS becomes "(...) / (\lambda)" which Maxima cannot parse.
                if "\\" in _h8_operand:
                    _h8_operand = engine._handler.latex_to_canonical(_h8_operand) or _h8_operand

            # Build new RHS canonical
            if _h8_op == "divide" and _h8_operand:
                _h8_new_rhs = f"({_h8_rhs}) / ({_h8_operand})"
            elif _h8_op == "multiply" and _h8_operand:
                _h8_new_rhs = f"({_h8_rhs}) * ({_h8_operand})"
            elif _h8_op == "add" and _h8_operand:
                _h8_new_rhs = f"({_h8_rhs}) + ({_h8_operand})"
            elif _h8_op == "subtract" and _h8_operand:
                _h8_new_rhs = f"({_h8_rhs}) - ({_h8_operand})"
            else:
                _h8_new_rhs = _h8_rhs  # simplify: use rhs as-is

            # New display LHS (uses the original latex operand, e.g. \lambda, so the
            # label renders \frac{L}{\lambda} — not the CAS canonical "lambda").
            _h8_new_lhs = _both_sides_new_lhs(_h8_lhs, _h8_op, _h8_operand_display)

            try:
                # Register the new RHS expression
                _h8_rhs_existing = _find_expr_by_canonical(engine, _h8_new_rhs)
                if _h8_rhs_existing:
                    _h8_rhs_id = _h8_rhs_existing
                else:
                    _h8_rhs_expr = await loop.run_in_executor(
                        None,
                        partial(engine.register_expression, _h8_new_rhs,
                                latex=_h8_new_rhs,
                                # Scaffolding: the researcher asked to multiply
                                # both sides, not to see the bare RHS.  Keyed
                                # `_T1` so the RESULT gets the next E number and
                                # the timeline reads E1 → E2, not E1 → E3
                                # (INTERNAL-INTERMEDIATE-EID-GAP).
                                internal=True),
                    )
                    _h8_rhs_id = _h8_rhs_expr.display_id

                # Simplify the new RHS.  The operation really is a simplify of
                # the scaffold, and the history says so — but a LABEL built from
                # that names `_T1`, which the researcher has never seen, and
                # calls the step a simplification when they asked to divide both
                # sides (open_items SCAFFOLD-LABEL).  Only this handler knows
                # both; the wording stays in `_visible_action`, which already
                # has the both-sides vocabulary and the particle rule.
                _H8_LABEL_OP = {"divide": "divide_both_sides",
                                "multiply": "multiply_both_sides",
                                "add": "add_to_both_sides",
                                "subtract": "subtract_from_both_sides"}
                _h8_step, _ = await loop.run_in_executor(
                    None,
                    partial(engine.run_operation, "simplify",
                            {"expr": _h8_rhs_id,
                             "label_expr_ref": _h8_eid,
                             "label_operation": _H8_LABEL_OP.get(_h8_op, "simplify")}),
                )
                # Mark the temp intermediate as hidden AFTER simplify so that
                # _active_expr_ids() can find it during the simplify call above.
                if _h8_rhs_existing is None and _h8_rhs_id in engine.expressions:
                    engine.expressions[_h8_rhs_id] = engine.expressions[_h8_rhs_id].model_copy(
                        update={"hidden": True}
                    )
                _h8_out_id = _h8_step.output_refs[0] if _h8_step.output_refs else None
                _h8_out    = engine.expressions.get(_h8_out_id) if _h8_out_id else None
            except Exception as _h8_exc:
                return {"status": "failed", "error": str(_h8_exc),
                        "registered_expressions": registered}

            if _h8_out is None:
                return {"status": "failed",
                        "error": t("interpreter.rhs_eval_failed", lang),
                        "registered_expressions": registered}

            # Attach new LHS label to the result expression.  The simplify
            # result may ALREADY carry a label of its own ("<source> = <value>",
            # which _SINGLE_EXPR_OPS attaches), and prefixing a second one built
            # an expression with two `=`: measured 2026-08-11,
            # `\frac{S}{2} = (\sum_{i=1}^3*i^2) / (2) = 7`, which is not an
            # equation at all and shows the researcher the un-evaluated middle.
            # strip_lhs leaves a label-free result untouched.
            _h8_final_latex = f"{_h8_new_lhs} = {strip_lhs(_h8_out.latex)}"
            engine.expressions[_h8_out_id] = _h8_out.model_copy(
                update={"latex": _h8_final_latex}
            )
            registered.append({
                "expr_id":   _h8_out.display_id,
                "canonical": _h8_out.canonical,
                "latex":     _h8_final_latex,
            })
            return {
                "status":    "completed",
                "operation": "both_sides_op",
                "expr_id":   _h8_out.display_id,
                "canonical": _h8_out.canonical,
                "latex":     _h8_final_latex,
                "registered_expressions": registered,
            }
    return None


async def _try_eq_pending_followup(engine, loop, lang, supervisor_text, registered) -> "dict | None":
    # 2b-followup. If there is a pending operation waiting for a single missing
    # argument, try to resolve it with the current input before normal processing.
    if engine.pending_op is not None:
        _pend = engine.pending_op
        _pend_missing = _pend.get("missing", [])
        _resolved = None
        if len(_pend_missing) == 1:
            _pf_field = _pend_missing[0]
            _pf_text  = supervisor_text.strip()
            if _pf_field == 'order':
                _m_pf = re.match(r'^\s*([0-9]+|[一二三四五六七八九])\s*[階次]?\s*$', _pf_text)
                if _m_pf:
                    _d = _m_pf.group(1)
                    _resolved = {**_pend["args"], "order": str(number_word(_d))}
            elif _pf_field == 'var':
                # The answer is a variable NAME, read the way every other
                # variable is read (`pending_var_answer` → `{{symbol}}`).  A
                # single Latin letter was the whole rule until 2026-09-05, and
                # anything else did not merely fail: the branch fell out of this
                # chain, the pending question was dropped, and the answer was
                # REGISTERED AS A NEW EXPRESSION (measured: `\theta` came back
                # `status=registered`).
                _m_pf = nlp("pending_var_answer", lang).match(_pf_text)
                if _m_pf:
                    _resolved = {**_pend["args"], "var": _m_pf.group("var")}
            elif _pf_text and not _HAS_CJK_RE.search(_pf_text) and not _MULTI_WORD_RE.search(_pf_text):
                # Expression IDs, formulas, bounds (lower/upper), lhs/rhs, etc.
                _resolved = {**_pend["args"], _pf_field: _pf_text}
        if _resolved is not None:
            engine.pending_op = None
            # Run the same post-resolve checks the normal pipeline applies.
            # check_diff_context: detect whether other variables depend on diff var.
            if _pend["operation"] == 'differentiate':
                _pf_diff_var = _resolved.get('var', '')
                _pf_src = engine.expressions.get(_resolved.get('expr', ''))
                if _pf_src and _pf_diff_var:
                    _pf_conflicts = await loop.run_in_executor(
                        None, partial(engine.check_diff_context,
                                      _pf_src.canonical, _pf_diff_var)
                    )
                    if _pf_conflicts:
                        dep_vars = list({v for v, _ in _pf_conflicts})
                        return {
                            "status": "needs_dep_confirmation",
                            "dep_vars": dep_vars,
                            "operation": _pend["operation"],
                            "args": _resolved,
                            "diff_var": _pf_diff_var,
                            "registered_expressions": registered,
                        }
            # re_evaluate_expr: special routing — calls engine method directly
            # (not run_operation) so auto-detect substitutions from lhs_alias_map works.
            if _pend["operation"] == "re_evaluate_expr":
                _re_expr_id   = _resolved.get("expr", "")
                _re_subs_raw  = _resolved.get("substitutions", {})
                # substitutions may arrive as empty dict → auto-detect
                _re_subs: dict[str, str] | None = _re_subs_raw if _re_subs_raw else None
                try:
                    new_expr = await loop.run_in_executor(
                        None,
                        partial(engine.re_evaluate_expr, _re_expr_id, _re_subs)
                    )
                except Exception as exc:
                    return {"status": "failed", "error": str(exc),
                            "registered_expressions": registered}
                if new_expr is None:
                    return {"status": "failed",
                            "error": "re_evaluate_expr: backend returned None",
                            "registered_expressions": registered}
                registered.append({
                    "expr_id":   new_expr.display_id,
                    "canonical": new_expr.canonical,
                    "latex":     new_expr.latex,
                })
                return {
                    "status":    "completed",
                    "operation": "re_evaluate_expr",
                    "out_expr_id":   new_expr.display_id,
                    "out_canonical": new_expr.canonical,
                    "out_latex":     new_expr.latex,
                    "registered_expressions": registered,
                }

            try:
                step, out_exprs = await loop.run_in_executor(
                    None, partial(engine.run_operation, _pend["operation"], _resolved)
                )
            except Exception as exc:
                return {"status": "failed", "error": str(exc),
                        "registered_expressions": registered}
            out_id = step.output_refs[0] if step.output_refs else None
            out_expr = engine.expressions.get(out_id) if out_id else None
            return {
                "status":    "completed",
                "operation": _pend["operation"],
                "expr_id":   out_id,
                "step_id":   step.step_id,
                "latex":     out_expr.latex     if out_expr else "",
                "canonical": out_expr.canonical if out_expr else "",
                "operation_label": step.visible_action,
                "registered_expressions": registered,
            }
        else:
            # Unrelated input — discard the pending op silently.
            engine.pending_op = None
    return None


_EQUATION_INTERCEPTORS: "list[Interceptor]" = [
    # R1d head batch: RHS ops + metric/apply constraints (fire first).
    Interceptor(
        "rhs_set",
        lambda c: _try_eq_rhs_set(c.engine, c.loop, c.lang, c.text, c.registered),
        note="H-5z: 「En の右辺を X とする」 → register a new equation 'RHS = X'."),
    Interceptor(
        "evaluate_rhs",
        lambda c: _try_eq_evaluate_rhs(c.engine, c.loop, c.lang, c.text, c.registered),
        note="H-5/H-7: 「En の右辺を 計算/展開/整理/因数分解」 → evaluate/named-op the RHS."),
    Interceptor(
        "metric_constraints",
        lambda c: _try_eq_metric_constraints(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1: derive the metric-preservation constraints M G Mᵀ = G."),
    Interceptor(
        "apply_constraints",
        lambda c: _try_eq_apply_constraints(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1 step 4: reduce a target under derived canonical constraints "
             "(u²+v²→1)."),
    # R1d tail batch: L-1 canonical/inverse + solve + both-sides + follow-up.
    Interceptor(
        "canonical_constraints",
        lambda c: _try_eq_canonical_constraints(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1 step 4: canonical constraints {new_i,new_j†}_± of an operator "
             "transform (Bogoliubov u²±v²=1)."),
    Interceptor(
        "inverse_transform",
        lambda c: _try_eq_inverse_transform(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1 inverse linear basis transform old = M^{-1} new (delegates to "
             "solve_system)."),
    Interceptor(
        "solve",
        lambda c: _try_eq_solve(c.engine, c.loop, c.lang, c.text, c.registered,
                                c.alias_free_text),
        note="H-9: solve an equation / system (may prompt for solve-for vars)."),
    Interceptor(
        "both_sides",
        lambda c: _try_eq_both_sides(c.engine, c.loop, c.lang, c.text, c.registered),
        note="H-8: 「En の両辺を X で 割る/かける/足す/引く」."),
    Interceptor(
        "pending_followup",
        lambda c: _try_eq_pending_followup(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Resolve a pending_op waiting on one missing arg with the current "
             "input; LAST so every other intercept gets first crack."),
]
