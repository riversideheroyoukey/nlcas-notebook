"""LLM-facing HTTP endpoints — extracted from notebook_bridge (R2).

Owns the interpretation and LLM-assist surface: POST /interpret (delegates to
interpret_pipeline), GET /supervisor-list, GET /adviser-list, GET
/adviser-status, and POST /advise (with its session-narrative helper).  Paths
compose under the bridge's ``/api/notebook`` prefix, so they are byte-for-byte
the same as before the split; only their home moved.

interpret_pipeline and the config helpers are leaf imports at module top.  The
bridge-owned surface is imported LAZILY inside handlers (cycle + patch safety):
the singleton/roster accessors get_engine / get_supervisor / get_adviser /
get_advisers.  ``_build_session_narrative`` originated with /advise, so it moved
here with the family (design rule 3, docs/27 R2); ui/interpret_plan.py also
reuses it (lazily imported, C-Ext-5 §11.1 context-supply increment #2) rather
than building a second session-history compressor.
"""
from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from notebook.runtime.i18n import t as _t
from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
from config.config_utils import get_advise_narrative_max_items, get_default_supervisor_target

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AdviseRequest(BaseModel):
    expr_id: str
    question: str = ""
    adviser_name: str | None = None


@router.post("/interpret")
async def interpret_command(req: InterpretRequest):
    """Interpret a mixed text+math token sequence as a notebook operation."""
    from ui.notebook_bridge import get_engine, get_supervisor
    import traceback, logging
    try:
        return await run_interpret_pipeline(req, get_engine(), get_supervisor())
    except Exception as e:
        logging.error("interpret_command unhandled: %s\n%s", e, traceback.format_exc())
        return {"status": "failed", "error": repr(e), "registered_expressions": []}


@router.get("/supervisor-list")
async def list_supervisors():
    """Return all discovered supervisor endpoints with the current default marked."""
    from notebook.runtime.llm_client import LLMClient
    preferred = get_default_supervisor_target() or ""
    entries = []
    try:
        clients = LLMClient.find_all_endpoints(role="supervisor", models_root=_PROJECT_ROOT / "models")
        clients.sort(key=lambda c: (0 if c.endpoint_name == preferred else 1, c.endpoint_name))
        for llm in clients:
            entries.append({
                "name":    llm.endpoint_name,
                "model":   getattr(llm, "_model", ""),
                "scope":   getattr(llm, "scope", "local"),
                "default": llm.endpoint_name == preferred,
            })
    except Exception:
        pass
    return {"supervisors": entries, "default": preferred}


@router.get("/adviser-list")
async def list_advisers():
    """Return all discovered adviser endpoints with their reachability status."""
    from ui.notebook_bridge import get_advisers
    loop = asyncio.get_event_loop()

    entries = []
    for name, adv in get_advisers().items():
        if not adv.available:
            entries.append({"name": name, "model": "", "available": False})
            continue
        llm = adv._llm  # LLMClient
        reachable = await loop.run_in_executor(None, partial(adv.ping, 5))
        entries.append({
            "name": name,
            "model": getattr(llm, "_model", ""),
            "scope": getattr(llm, "scope", "local"),
            "available": reachable,
        })
    return {"advisers": entries}


@router.get("/adviser-status")
async def get_adviser_status():
    """Check if any adviser is reachable (backward-compat endpoint)."""
    from ui.notebook_bridge import get_adviser, get_advisers
    if not get_advisers():
        return {"available": False, "reason": "not_configured"}
    adviser = get_adviser()
    if not adviser.available:
        return {"available": False, "reason": "not_configured"}
    loop = asyncio.get_event_loop()
    reachable = await loop.run_in_executor(None, partial(adviser.ping, 5))
    return {"available": reachable, "reason": None if reachable else "unreachable"}


def _build_session_narrative(engine, up_to_step_id: str | None, max_items: int = 15) -> list[dict]:
    """Build a compressed chronological narrative of the session up to (and including) up_to_step_id."""
    from notebook.runtime.replay import compute_lineage
    from notebook.models.step import StepKind as SK

    lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)

    # Find cutoff index
    cutoff = len(lineage)
    if up_to_step_id:
        for i, step in enumerate(lineage):
            if step.step_id == up_to_step_id:
                cutoff = i + 1
                break

    items: list[dict] = []
    for step in lineage[:cutoff]:
        if step.kind == SK.user_input and step.user_text:
            items.append({"kind": "user_input", "text": step.user_text})
        elif step.kind == SK.expression_registration:
            for eid in step.output_refs:
                expr = engine.expressions.get(eid)
                if expr:
                    items.append({"kind": "registration", "expr_id": expr.display_id,
                                  "latex": expr.latex})
        elif step.kind == SK.cas_operation:
            for eid in step.output_refs:
                expr = engine.expressions.get(eid)
                if expr:
                    src_id = step.input_refs[0] if step.input_refs else None
                    src_expr = engine.expressions.get(src_id) if src_id else None
                    items.append({"kind": "cas_op",
                                  "expr_id": expr.display_id,
                                  "latex": expr.latex,
                                  "cas_repr": expr.cas_repr,
                                  "operation": step.operation or "",
                                  # session-localized action label (visible_action
                                  # is already rendered in the session language)
                                  "operation_label": step.visible_action,
                                  "source_id": src_expr.display_id if src_expr else None})
        elif step.kind == SK.manual_note and step.user_text:
            items.append({"kind": "note", "text": step.user_text[:200]})
        elif step.kind == SK.assumption_add and step.visible_action:
            items.append({"kind": "assumption", "text": step.visible_action})

    # Return only the most recent max_items
    return items[-max_items:]


@router.post("/advise")
async def advise_expression(req: AdviseRequest):
    """Generate mathematical/physical commentary on a CAS result expression."""
    from ui.notebook_bridge import get_engine, get_adviser
    from notebook.runtime.llm_adviser import AdviseContext
    from notebook.runtime.operation_validator import OperationValidator

    engine  = get_engine()
    adviser = get_adviser(req.adviser_name)

    if not adviser.available:
        return {"ok": False, "error": _t("adviser.not_configured", engine.lang)}

    if req.expr_id not in engine.expressions:
        raise HTTPException(status_code=404, detail=f"Expression '{req.expr_id}' not found")

    expr = engine.expressions[req.expr_id]

    # --- Derivation info from the creating step ---
    src_step = engine.steps.get(expr.created_by_step) if expr.created_by_step else None
    _lang = engine.lang
    operation          = None
    operation_display  = None
    operation_args     = {}
    source_expr_id     = None
    source_expr_latex  = None

    if src_step and src_step.operation:
        operation      = src_step.operation
        operation_args = src_step.operation_args
        try:
            spec = OperationValidator().get_spec(operation)
            if spec:
                operation_display = spec["display"].get(_lang) or spec["display"].get("en")
        except Exception:
            pass
        if src_step.input_refs:
            source_expr_id = src_step.input_refs[0]
            src_expr = engine.expressions.get(source_expr_id)
            if src_expr:
                source_expr_latex = src_expr.latex

    # --- Recent expressions (newest first, excluding target, up to 5) ---
    sorted_exprs = sorted(
        engine.expressions.values(),
        key=lambda e: e.created_by_step or "",
        reverse=True,
    )
    recent_exprs = [
        {"id": e.display_id, "latex": e.latex, "canonical": e.canonical}
        for e in sorted_exprs
        if e.display_id != req.expr_id
    ][:5]

    # --- Assumptions ---
    assumption_strs = [a.display or a.canonical for a in engine.assumptions.values()]

    session_context = _build_session_narrative(
        engine,
        up_to_step_id=expr.created_by_step,
        max_items=get_advise_narrative_max_items(),
    )

    ctx = AdviseContext(
        expr_id=req.expr_id,
        expr_latex=expr.latex,
        expr_canonical=expr.canonical,
        expr_cas_repr=expr.cas_repr,
        operation=operation,
        operation_display=operation_display,
        operation_args=operation_args,
        source_expr_id=source_expr_id,
        source_expr_latex=source_expr_latex,
        question=req.question,
        session_title=engine.session.title,
        assumptions=assumption_strs,
        recent_exprs=recent_exprs,
        session_context=session_context,
        lang=_lang,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(adviser.advise, ctx))
    return {"ok": result.ok, "text": result.text, "error": result.error}
