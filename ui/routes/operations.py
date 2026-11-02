"""Operation HTTP endpoints — extracted from notebook_bridge (R2).

Owns the notebook's *compute* surface: /register, /operation (with its
side-effect and dependency-context helpers), /re_evaluate, /wick_contract.
Paths compose under the bridge's ``/api/notebook`` prefix, so they are
byte-for-byte the same as before the split; only their home moved.

The engine singleton is imported lazily inside handlers (cycle + patch safety —
see ui/routes/history.py).  The two helpers receive the live ``engine`` from the
handler, so they read the display language via ``engine.lang`` directly — the
former ``_get_lang()`` returns exactly that when the engine is live, which it
always is on these code paths.  That drops the last bridge coupling for this
family (only ``get_engine`` remains, and only in the handlers).
"""
from __future__ import annotations

import asyncio
import re
from functools import partial
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from notebook.runtime.i18n import t as _t

router = APIRouter()


class RegisterRequest(BaseModel):
    latex: str
    canonical: str = ""     # optional override; if empty, auto-converted from latex


class OperationRequest(BaseModel):
    operation: str
    args: dict[str, Any]
    # Optional: populated when the call is a user-confirmed ambiguity resolution
    user_text: str = ""
    candidates: list[dict] = []
    # Dependency conflict resolution
    dep_confirmed: bool = False      # True = user already decided; skip conflict check
    remove_deps_for: list[str] = []  # vars whose deps to remove before executing
    # LLM-inferred depends() declarations, deferred until user confirms
    pending_dependencies: list[dict] = []


class ReEvaluateRequest(BaseModel):
    expr_id: str
    substitutions: dict[str, str] = {}


class WickContractRequest(BaseModel):
    expr_id: str
    contraction_rules: list[dict]
    include_constant: bool = False


@router.post("/register")
async def register_expression(req: RegisterRequest):
    """Register a new expression from LaTeX (or an explicit canonical form)."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    display_latex = engine._handler.normalize_display_latex(req.latex)
    canonical = req.canonical.strip() or engine._handler.latex_to_canonical(display_latex)
    if not canonical:
        raise HTTPException(status_code=400, detail="Empty expression")

    loop = asyncio.get_event_loop()
    try:
        expr = await loop.run_in_executor(
            None,
            partial(engine.register_expression, canonical, latex=display_latex or canonical),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "expr_id": expr.display_id,
        "canonical": expr.canonical,
        "latex": expr.latex,
        "undefined_symbols": expr.undefined_symbols,
    }


async def _handle_side_effect_operation(
    req: OperationRequest, engine, loop
) -> dict | None:
    """Handle operations that update engine state without producing an expression.

    Returns a response dict when handled, or None to continue to normal execution.
    """
    if req.operation == 'set_notation':
        name = req.args.get('name', '').strip()
        fmt  = req.args.get('format', '').strip().lower()
        if fmt in ('leibniz', 'leibniz_notation', 'd/dt', 'ライプニッツ'):
            fmt = 'leibniz'
        elif fmt in ('dot', 'dot_notation', 'ドット'):
            fmt = 'dot'
        if name not in engine.symbol_context.registry:
            _expr = engine.expressions.get(name)
            if _expr and _expr.quantity_name and _expr.quantity_name in engine.symbol_context.registry:
                name = _expr.quantity_name
            elif len(engine.symbol_context.registry) == 1:
                name = next(iter(engine.symbol_context.registry))
        try:
            await loop.run_in_executor(None, partial(engine.set_notation, name, fmt))
        except ValueError as e:
            return {'ok': False, 'detail': str(e)}
        _updated = await loop.run_in_executor(
            None, partial(engine.rerender_notation_for_quantity, name)
        )
        _sn_lang = engine.lang
        label = _t('bridge.notation_label_dot' if fmt == 'dot'
                   else 'bridge.notation_label_leibniz', _sn_lang)
        _msg = _t('bridge.notation_switched', _sn_lang, name=name, label=label)
        engine.record_quantity_update(_t('bridge.notation_switch_action', _sn_lang), _msg)
        return {
            'ok': True, 'status': 'confirmed', 'operation': 'set_notation',
            'message': _msg,
            'updated_exprs': [{'expr_id': eid, 'latex': lat} for eid, lat in _updated],
        }

    if req.operation == 'declare_time_dependent':
        name = req.args.get('name', '').strip()
        time_parameter = (req.args.get('time_parameter') or 't').strip() or 't'
        try:
            await loop.run_in_executor(None, partial(engine.declare_time_dependent, name, time_parameter))
        except Exception as e:
            return {'ok': False, 'detail': str(e)}
        _td_lang = engine.lang
        _msg = _t('bridge.time_dependent_declared', _td_lang,
                  name=name, time_parameter=time_parameter)
        engine.record_quantity_update(_t('bridge.time_dependent_action', _td_lang), _msg)
        return {'ok': True, 'status': 'confirmed', 'operation': 'declare_time_dependent', 'message': _msg}

    if req.operation == 'set_latex_name':
        name  = req.args.get('name', '').strip()
        latex = req.args.get('latex', '').strip()
        try:
            await loop.run_in_executor(None, partial(engine.set_latex_name, name, latex))
        except Exception as e:
            return {'ok': False, 'detail': str(e)}
        _ln_lang = engine.lang
        _msg = _t('bridge.latex_name_set', _ln_lang, name=name, latex=latex)
        engine.record_quantity_update(_t('bridge.latex_name_action', _ln_lang), _msg)
        return {'ok': True, 'status': 'confirmed', 'operation': 'set_latex_name', 'message': _msg}

    if req.operation == 'resolve_notation':
        # R-8: the user picked one interpretation of an ambiguous notation
        # token (|X| → abs/determinant, Δ → laplacian/symbol).  Record the
        # choice in notation_usage (persisted with the session, so the same
        # token is never re-asked), apply any declaration side effect, then
        # register the disambiguated expression.
        lang = engine.lang
        token = str(req.args.get('token', '') or '')
        role = str(req.args.get('role', '') or '')
        rewritten = str(req.args.get('rewritten', '') or '')
        latex = str(req.args.get('latex', '') or '')
        declare = str(req.args.get('declare', '') or '')
        if not (token and role and rewritten):
            return {'ok': False, 'detail': 'resolve_notation: token/role/rewritten required'}
        engine.symbol_context.notation_usage[token] = role
        if declare:
            await loop.run_in_executor(
                None, partial(engine.register_quantity, declare, kind='scalar'))
        try:
            canonical = await loop.run_in_executor(
                None, partial(engine._handler.latex_to_canonical, rewritten))
            expr = await loop.run_in_executor(
                None, partial(engine.register_expression,
                              canonical or rewritten, latex))
        except ValueError as e:
            return {'ok': False, 'detail': str(e)}
        if req.user_text:
            await loop.run_in_executor(
                None, partial(engine.record_ambiguity_resolution, req.user_text,
                              'resolve_notation', dict(req.args),
                              req.candidates or []))
        return {
            'ok': True, 'status': 'confirmed', 'operation': 'resolve_notation',
            'expr_id': expr.display_id, 'latex': expr.latex,
            'canonical': expr.canonical,
            'message': _t('notation.resolved', lang, token=token, role=role),
        }

    if req.operation == 'execute_plan_candidate':
        # C-Ext-5 priority 3: the researcher picked one Adviser candidate.
        # Build a source="llm" plan from it (validated AGAIN inside
        # create_candidate_plan — defense in depth), execute stop-safe, and
        # record the choice as an ambiguity-resolution step.
        goal = str(req.args.get('goal', '') or '')
        title = str(req.args.get('title', '') or '')
        target = str(req.args.get('target_expr_id', '') or '')
        steps = req.args.get('steps') or []
        try:
            plan = await loop.run_in_executor(None, partial(
                engine.create_candidate_plan, goal, target, steps, title))
            plan = await loop.run_in_executor(None, partial(
                engine.execute_plan, plan.plan_id))
        except ValueError as e:
            return {'ok': False, 'detail': str(e)}
        if req.user_text:
            await loop.run_in_executor(None, partial(
                engine.record_ambiguity_resolution, req.user_text,
                'execute_plan_candidate',
                {'goal': goal, 'title': title, 'plan_id': plan.plan_id},
                req.candidates or []))
        summary_step = [s for s in engine.steps.values()
                        if s.kind.value == 'plan_execution'][-1]
        return {
            'ok': plan.status == 'completed',
            'status': 'confirmed' if plan.status == 'completed' else 'failed',
            'operation': 'execute_plan_candidate',
            'plan_id': plan.plan_id,
            'message': summary_step.visible_action,
            'plan_step_results': [r.model_dump(mode='json')
                                  for r in plan.step_results],
        }

    if req.operation == 'numeric_fork_nl':
        # N-1 NL trigger: the confirm click IS the approval for this
        # deterministic-template-only path (ui/interpret_numeric.py) — fork,
        # generate, and run in one shot, no separate review gate.
        expr_id = str(req.args.get('expr_id', '') or '')
        try:
            task = await loop.run_in_executor(None, partial(
                engine.fork_numeric_task, expr_id,
                problem_type=req.args.get('problem_type', 'parameter_sweep'),
                target_quantity=req.args.get('target_quantity', ''),
                sweep_parameters=req.args.get('sweep_parameters') or {},
                fixed_values=req.args.get('fixed_values') or {}))
            await loop.run_in_executor(None, partial(
                engine.generate_numeric_code, task.task_id))
            import ui.session_io as session_io
            if session_io.get_current_filename() is None:
                await session_io.do_save(engine)
            execution = await loop.run_in_executor(None, partial(
                engine.run_numeric_task, task.task_id))
        except ValueError as e:
            return {'ok': False, 'detail': str(e), 'operation': 'numeric_fork_nl'}
        from ui.routes.numeric import _shape_numeric_run_response
        shaped = _shape_numeric_run_response(engine, execution)
        return {
            'ok': shaped['ok'],
            'status': 'confirmed' if shaped['ok'] else 'failed',
            'operation': 'numeric_fork_nl',
            'step_id': shaped['step_id'],
            'text': shaped['result_text'],
            'figure_artifact_id': shaped['figure_artifact_id'],
        }

    if req.operation == 'add_assumption':
        condition = req.args.get('condition', '').strip()
        try:
            _, asm = await loop.run_in_executor(None, partial(engine.add_assumption, condition, condition))
        except Exception as e:
            return {'ok': False, 'detail': str(e)}
        return {
            'ok': True, 'status': 'assumption_added',
            'assumption_id': asm.assumption_id, 'condition': condition,
            'message': _t('bridge.assumption_added', engine.lang, condition=condition),
        }

    if req.operation == 'remove_assumption':
        condition = req.args.get('condition', '').strip()
        try:
            await loop.run_in_executor(None, partial(engine.remove_assumption_by_canonical, condition))
        except Exception as e:
            return {'ok': False, 'detail': str(e)}
        return {'ok': True, 'status': 'assumption_removed', 'condition': condition}

    return None


async def _prepare_operation_context(
    req: OperationRequest, args: dict, engine, loop
) -> dict | None:
    """Set up dependency declarations before a CAS operation.

    For differentiate: checks for dependency conflicts (returns a
    needs_dep_confirmation dict if the user must choose), then applies
    any pending depends() declarations and implicit-diff setup.
    For all operations: applies remove_deps_for and pending_dependencies.

    Returns a conflict-response dict that the caller should return directly,
    or None when the operation may proceed.
    """
    if req.operation == 'differentiate' and not req.dep_confirmed:
        diff_var = req.args.get('var', '')
        src_expr = engine.expressions.get(req.args.get('expr', ''))
        if src_expr and diff_var:
            conflicts = await loop.run_in_executor(
                None, partial(engine.check_diff_context, src_expr.canonical, diff_var)
            )
            if conflicts:
                return {
                    "ok": False,
                    "needs_dep_confirmation": True,
                    "dep_vars": list({v for v, _ in conflicts}),
                    "operation": req.operation,
                    "args": req.args,
                    "diff_var": diff_var,
                }

    for var in req.remove_deps_for:
        await loop.run_in_executor(None, partial(engine.remove_variable_dependency, var))

    for dep in req.pending_dependencies:
        dependent = dep.get("dependent")
        independent = dep.get("independent", [])
        if dependent and independent:
            await loop.run_in_executor(None, partial(engine.declare_depends, dependent, independent))

    if req.operation == 'differentiate':
        if req.dep_confirmed and not req.remove_deps_for:
            args['_is_total_derivative'] = True
        implicit_lhs = args.pop('_implicit_lhs', None)
        if implicit_lhs:
            var = args.get('var', '')
            if var:
                await loop.run_in_executor(None, partial(engine.declare_depends, implicit_lhs, [var]))
        elif args.get('var'):
            src_expr = engine.expressions.get(args.get('expr', ''))
            if src_expr:
                _m = re.match(r"^'diff\(\s*([A-Za-z_]\w*)\s*,", src_expr.canonical)
                if _m:
                    await loop.run_in_executor(
                        None, partial(engine.declare_depends, _m.group(1), [args['var']])
                    )
    else:
        args.pop('_implicit_lhs', None)

    # F: warn when define_vector_field / define_scalar_field is used without a
    # coordinate system set on the session (cartesian fallback will be used).
    if req.operation in ("define_vector_field", "define_scalar_field"):
        if engine.session.coordinate_system is None:
            warn_msg = _t("bridge.coord_fallback_warning", engine.lang)
            # Return early with warning — execution continues via normal path after this
            # by returning None, but we attach the warning via a special key that the
            # client can display as a banner.
            # Implementation: proceed (return None) but inject warning into args so
            # the bridge can surface it in the response below.
            req.args["_coord_fallback_warning"] = warn_msg

    return None


@router.post("/operation")
async def run_operation(req: OperationRequest):
    """Execute a notebook operation (e.g. simplify, diff) and return the new expression."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()

    side_effect_resp = await _handle_side_effect_operation(req, engine, loop)
    if side_effect_resp is not None:
        return side_effect_resp

    args = dict(req.args)
    conflict_resp = await _prepare_operation_context(req, args, engine, loop)
    if conflict_resp is not None:
        return conflict_resp

    if req.user_text and req.candidates:
        await loop.run_in_executor(
            None,
            partial(engine.record_ambiguity_resolution, req.user_text, req.operation, args, req.candidates),
        )

    try:
        step, value = await loop.run_in_executor(
            None, partial(engine.run_operation, req.operation, args)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    expr_id = step.output_refs[0] if step.output_refs else None
    expr = engine.expressions.get(expr_id) if expr_id else None

    resp: dict = {
        "ok": True,
        "expr_id": expr_id,
        "step_id": step.step_id,
        "canonical": expr.canonical if expr else value,
        "latex": expr.latex if expr else "",
        "value": value,
        "operation_label": step.visible_action,
    }
    # F: surface coordinate-system fallback warning if set
    _coord_warn = args.pop("_coord_fallback_warning", None)
    if _coord_warn:
        resp["warning"] = _coord_warn
    return resp


@router.post("/re_evaluate")
async def re_evaluate_expression(req: ReEvaluateRequest):
    """Re-evaluate a composite expression with substituted symbol values.

    Works identically on Maxima and Julia — backend difference is hidden.
    If substitutions is empty, lhs_alias_map is used for auto-detection (see engine).
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        new_expr = await loop.run_in_executor(
            None,
            partial(engine.re_evaluate_expr, req.expr_id,
                    req.substitutions if req.substitutions else None)
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Expression not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if new_expr is None:
        raise HTTPException(status_code=400,
                            detail="re_evaluate_expr returned None (backend unsupported?)")
    return {
        "ok": True,
        "expr_id": new_expr.display_id,
        "canonical": new_expr.canonical,
        "latex": new_expr.latex,
    }


@router.post("/wick_contract")
async def wick_contract_expression(req: WickContractRequest):
    """Wick contraction: extract operator pairs, replace with c-numbers.

    Returns list of output expressions (one per contraction term).
    Julia (second_quantization) session only.
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        terms = await loop.run_in_executor(
            None,
            partial(engine.wick_contract, req.expr_id,
                    req.contraction_rules, req.include_constant)
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "terms": [
            {"expr_id": t.display_id, "canonical": t.canonical, "latex": t.latex}
            for t in terms
        ],
        # An empty `terms` has two meanings — the contractions vanished, or no
        # rule reached a pair at all — and only the driver can tell them apart.
        # Carried separately because there is no expression to hang it on when
        # the list is empty, which is exactly the case worth saying something
        # about (open_items NOOP-REWRITE-FAMILY).
        "warnings": list(getattr(engine, "last_wick_warnings", []) or []),
    }
