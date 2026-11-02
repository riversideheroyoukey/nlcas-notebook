"""History / annotation HTTP endpoints — extracted from notebook_bridge (R2).

This family owns the notebook's *content editing* surface: research notes,
deletion of unreferenced expressions / notes / user-input / define steps, the
branch summary, the timeline view, and CAS assumption management.  Every path
composes under the bridge's ``/api/notebook`` prefix, so the routes are
byte-for-byte the same as before the split; only their home moved.

Handlers reach the engine singleton by lazily importing ``get_engine`` from
``ui.notebook_bridge`` INSIDE each handler — a top-level import would both cycle
(this module is mounted from the bottom of notebook_bridge) and break
``patch("ui.notebook_bridge.get_engine")`` in tests, which rebinds the name on
the bridge module after import.
"""
from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class NoteRequest(BaseModel):
    text: str


class AssumptionRequest(BaseModel):
    canonical: str
    display: str = ""


@router.post("/note")
async def add_note(req: NoteRequest):
    """Save a user-written research note into the session history."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Note text is empty")
    loop = asyncio.get_event_loop()
    step = await loop.run_in_executor(None, partial(engine.record_manual_note, text))
    return {"ok": True, "step_id": step.step_id}


@router.delete("/expression/{expr_id}")
async def delete_expression(expr_id: str):
    """Delete an unreferenced expression entry and its registration step."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if expr_id not in engine.expressions:
        raise HTTPException(status_code=404, detail=f"Expression '{expr_id}' not found")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, partial(engine.delete_expression, expr_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "expr_id": expr_id}


@router.delete("/note/{step_id}")
async def delete_note(step_id: str):
    """Delete a manual_note step and re-wire the parent chain."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if step_id not in engine.steps:
        raise HTTPException(status_code=404, detail=f"Step '{step_id}' not found")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, partial(engine.delete_note_step, step_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "step_id": step_id}


@router.delete("/step/{step_id}")
async def delete_user_step(step_id: str):
    """Delete a user_input step (non-state-affecting) and re-wire the parent chain."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if step_id not in engine.steps:
        raise HTTPException(status_code=404, detail=f"Step '{step_id}' not found")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, partial(engine.delete_user_input_step, step_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "step_id": step_id}


@router.delete("/define/{step_id}")
async def delete_define(step_id: str):
    """Delete a define operation step, unregister its symbols, and rewire the parent chain."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if step_id not in engine.steps:
        raise HTTPException(status_code=404, detail=f"Step '{step_id}' not found")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, partial(engine.delete_define_step, step_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "step_id": step_id}


def _generate_summary_text(engine) -> str:
    """Build a human-readable summary of the active branch's history."""
    from notebook.runtime.replay import compute_lineage
    from notebook.runtime.i18n import t
    from notebook.models.step import StepKind as SK

    lang    = engine.lang
    branch  = engine.branches.get(engine.active_branch_id)
    label   = (branch.label or engine.active_branch_id) if branch else engine.active_branch_id
    lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)

    _RANGE_KINDS = (SK.expression_registration, SK.cas_operation,
                    SK.assumption_add, SK.assumption_remove,
                    SK.manual_note)
    state_steps = [s for s in lineage if s.kind in _RANGE_KINDS]
    step_range  = (t("summary.step_range", lang, first=state_steps[0].step_id,
                     last=state_steps[-1].step_id)
                   if state_steps else t("summary.no_steps", lang))

    expr_lines: list[str] = []
    note_lines: list[str] = []
    for step in lineage:
        if step.kind in (SK.expression_registration, SK.cas_operation):
            for ref in step.output_refs:
                expr = engine.expressions.get(ref)
                if not expr:
                    continue
                latex  = expr.latex or expr.canonical
                action = f"  ← {step.visible_action}" if step.kind != SK.expression_registration else ""
                expr_lines.append(f"{ref}: ${latex}${action}")
        elif step.kind == SK.manual_note and step.user_text:
            note_lines.append(f"{step.visible_action}: {step.user_text}")

    asm_lines = [t("summary.assumption", lang, text=a.display or a.canonical)
                 for a in engine.assumptions.values()]

    header = t("summary.branch_header", lang, label=label, step_range=step_range,
               n_expr=len(expr_lines), n_asm=len(engine.assumptions))
    parts = [header] + expr_lines
    if note_lines:
        parts += [""] + note_lines
    if asm_lines:
        parts += [""] + asm_lines
    return "\n".join(parts)


@router.post("/summary")
async def generate_summary():
    """Generate a branch summary, store it as a manual_note, and return it."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop   = asyncio.get_event_loop()
    text   = await loop.run_in_executor(None, lambda: _generate_summary_text(engine))
    step   = await loop.run_in_executor(None, partial(engine.record_manual_note, text))
    return {"ok": True, "step_id": step.step_id, "text": text}


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

@router.get("/timeline")
async def get_timeline():
    """Return timeline entries for the active branch in chronological order.

    Entry-building lives in ui/timeline_serializer.py (R2); this endpoint is
    a thin delegation over the engine singleton."""
    from ui.notebook_bridge import get_engine
    from ui.timeline_serializer import build_timeline_entries
    engine = get_engine()
    return {
        "title":  engine.session.title,
        "branch": engine.active_branch_id,
        "entries": build_timeline_entries(engine),
    }


# ---------------------------------------------------------------------------
# Assumption management
# ---------------------------------------------------------------------------

@router.get("/assumptions")
async def list_assumptions():
    """List all active assumptions in the current session."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    return {
        "assumptions": [
            {
                "assumption_id": aid,
                "canonical":     asm.canonical,
                "display":       asm.display,
                "branch_id":     asm.branch_id,
            }
            for aid, asm in engine.assumptions.items()
        ]
    }


@router.post("/assumption")
async def add_assumption(req: AssumptionRequest):
    """Add a CAS assumption (e.g. x > 0) to the session."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    canonical = req.canonical.strip()
    if not canonical:
        raise HTTPException(status_code=400, detail="Assumption condition is empty")
    loop = asyncio.get_event_loop()
    try:
        _, asm = await loop.run_in_executor(
            None, partial(engine.add_assumption, canonical, req.display or canonical)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok":            True,
        "assumption_id": asm.assumption_id,
        "canonical":     asm.canonical,
        "display":       asm.display,
    }


@router.delete("/assumption/{assumption_id}")
async def remove_assumption(assumption_id: str):
    """Remove an active assumption by its ID."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if assumption_id not in engine.assumptions:
        raise HTTPException(status_code=404, detail=f"Assumption '{assumption_id}' not found")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, partial(engine.remove_assumption, assumption_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "assumption_id": assumption_id}
