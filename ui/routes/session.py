"""Session lifecycle / status / workspace HTTP endpoints — extracted from
notebook_bridge (R2).

Owns session state (GET /session, GET /status), UI config (GET /ui-lang, GET
/domains), lifecycle (DELETE /session, POST /session/new), file-level branching
(POST /fork), and plan-status polling (GET /plans, GET /plan/status/{id}).  Paths
compose under the bridge's ``/api/notebook`` prefix, so they are byte-for-byte
the same as before the split; only their home moved.

Leaf utilities (session_io, DOMAINS_ROOT, config helpers, i18n) are imported at
module top.  The bridge-owned surface is imported LAZILY inside handlers (cycle +
patch safety): the singleton accessors (get_engine), lifecycle mutators
(start_engine / stop_engine), the language accessor (_get_lang — kept on the
bridge because it reads the _engine global and is shared across families), and
the adviser roster accessor (get_advisers).
"""
from __future__ import annotations

import asyncio
import json as _json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from notebook.domain_paths import DOMAINS_ROOT
from notebook.runtime.i18n import t as _t
import ui.session_io as session_io
from config.config_utils import (get_default_language, get_default_supervisor_target,
                                  get_default_adviser_target)

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class NewSessionRequest(BaseModel):
    title: str = "Notebook"
    mode: str = "standard"
    domains: list[str] = []
    language: str = Field(default_factory=get_default_language)
    supervisor_name: str | None = None
    coordinate_system: str | None = None  # "cartesian"|"spherical"|"cylindrical"|None


class ForkRequest(BaseModel):
    step_id: str
    snapshot_filename: str = ""


class BranchForkRequest(BaseModel):
    step_id: str
    label: str = ""


class BranchSwitchRequest(BaseModel):
    branch_id: str


@router.get("/session")
async def get_session():
    """Return the current session state: expressions, step count, active branch."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    expressions = {
        eid: {
            "canonical": expr.canonical,
            "latex": expr.latex,
            "branch_id": expr.branch_id,
        }
        for eid, expr in engine.expressions.items()
    }
    return {
        "title": engine.session.title,
        "backend": engine.session.primary_backend,
        "active_branch": engine.active_branch_id,
        "branches": {
            bid: {"label": b.label, "forked_from_step_id": b.forked_from_step_id}
            for bid, b in engine.branches.items()
        },
        "expressions": expressions,
        "step_count": len(engine.steps),
        "domains": engine.session.domains,
    }


@router.delete("/session")
async def clear_session():
    """Restart the engine with a fresh empty session (destroys current state)."""
    from ui.notebook_bridge import start_engine, stop_engine
    stop_engine()
    start_engine()
    return {"ok": True, "message": "Session cleared"}


@router.get("/status")
async def get_status():
    """Return a compact session + LLM status summary for the Status dialog."""
    from ui.notebook_bridge import get_engine, get_advisers, _get_lang
    from config.config_utils import get_default_vlm_target, get_default_refiner_target
    engine = get_engine()
    lang   = _get_lang()

    # Resolve domain labels from meta.json files
    domain_labels: list[str] = []
    for d in engine.session.domains:
        meta_path = DOMAINS_ROOT / d / "meta.json"
        if meta_path.exists():
            meta  = _json.loads(meta_path.read_text(encoding="utf-8"))
            label = meta.get("label_ja" if lang == "ja" else "label_en", d)
        else:
            label = d
        domain_labels.append(label)

    def _llm_list(names: list[str], selected: str) -> list[dict]:
        return [{"name": n, "selected": n == selected} for n in names]

    # Supervisor
    sup_selected = get_default_supervisor_target() or ""
    try:
        from notebook.runtime.llm_client import LLMClient
        sup_all = [c.endpoint_name for c in
                   LLMClient.find_all_endpoints(role="supervisor", models_root=_PROJECT_ROOT / "models")]
    except Exception:
        sup_all = [sup_selected] if sup_selected else []

    # Adviser
    adv_selected = get_default_adviser_target() or ""
    _advisers = get_advisers()
    adv_all = list(_advisers.keys()) if _advisers else []

    # OCR Engine (models/engines/*/config.json, type == "ocr_engine")
    ocr_selected = get_default_vlm_target() or ""
    ocr_all: list[str] = []
    for p in sorted((_PROJECT_ROOT / "models" / "engines").glob("*/config.json")):
        try:
            cfg = _json.loads(p.read_text(encoding="utf-8"))
            if cfg.get("type") == "ocr_engine":
                ocr_all.append(p.parent.name)
        except Exception:
            pass

    # OCR Refiner (models/refiners/*/config.json)
    ref_selected = get_default_refiner_target() or ""
    ref_all: list[str] = []
    for p in sorted((_PROJECT_ROOT / "models" / "refiners").glob("*/config.json")):
        try:
            _json.loads(p.read_text(encoding="utf-8"))
            ref_all.append(p.parent.name)
        except Exception:
            pass

    return {
        "title":        engine.session.title,
        "lang":         lang,
        "backend":      engine.session.primary_backend,
        "domains":      domain_labels,
        "step_count":   len(engine.steps),
        "supervisors":  _llm_list(sup_all, sup_selected),
        "advisers":     _llm_list(adv_all, adv_selected),
        "ocr_engines":  _llm_list(ocr_all, ocr_selected),
        "ocr_refiners": _llm_list(ref_all, ref_selected),
        "current_filename": session_io.get_current_filename(),
        "coordinate_system": engine.session.coordinate_system,
    }


@router.get("/ui-lang")
async def get_ui_lang():
    """Return the active UI language and the front-end message table.

    Reads ui_messages.json once per process (cached).  The active language
    is the running session's language when available, otherwise the default
    from default_settings.json.  The front end calls this on page load to
    initialise its _t() helper and apply data-i18n attributes.
    """
    from ui.notebook_bridge import _get_lang
    import functools

    @functools.lru_cache(maxsize=1)
    def _load_ui_messages() -> dict:
        path = _PROJECT_ROOT / "i18n" / "ui_messages.json"
        return _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    return {"lang": _get_lang(), "messages": _load_ui_messages()}


@router.post("/session/new")
async def new_session(req: NewSessionRequest):
    """Create a new session with optional domain hints."""
    from ui.notebook_bridge import start_engine, stop_engine
    stop_engine()
    start_engine(title=req.title, mode=req.mode, domains=req.domains, language=req.language,
                 supervisor_name=req.supervisor_name or None,
                 coordinate_system=req.coordinate_system or None)
    return {"ok": True, "title": req.title, "domains": req.domains, "language": req.language,
            "coordinate_system": req.coordinate_system}


@router.get("/domains")
async def list_domains():
    """Return domains grouped by category, ordered by categories.json."""
    from ui.notebook_bridge import _get_lang
    domains_root = DOMAINS_ROOT
    if not domains_root.exists():
        return {"categories": []}

    # Load category order from categories.json
    cat_path = domains_root / "categories.json"
    if cat_path.exists():
        cat_defs = _json.loads(cat_path.read_text(encoding="utf-8"))
    else:
        cat_defs = []
    lang = _get_lang()
    cat_order   = {c["key"]: c["order"] for c in cat_defs}
    cat_backend = {c["key"]: c.get("default_backend") for c in cat_defs}
    # Resolve category labels for the active language; fall back to label_ja then key
    cat_label_ja = {c["key"]: c.get("label_ja", c.get("label", c["key"])) for c in cat_defs}
    cat_label_en = {c["key"]: c.get("label_en", cat_label_ja.get(c["key"], c["key"])) for c in cat_defs}

    def _cat_label(key: str) -> str:
        return (cat_label_ja if lang == "ja" else cat_label_en).get(key, key)

    # Read all domain meta.json files
    domain_entries: list[dict] = []
    for p in sorted(domains_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        meta_path = p / "meta.json"
        meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        label_ja = meta.get("label_ja", meta.get("label", p.name))
        label_en = meta.get("label_en", label_ja)
        domain_entries.append({
            "name":     p.name,
            "label":    label_ja if lang == "ja" else label_en,
            "label_ja": label_ja,
            "label_en": label_en,
            "category": meta.get("category", "other"),
            "status":   meta.get("status", "active"),  # "active" | "future" | "placeholder"
        })

    # Group by category, sorted by category order then resolved label
    groups: dict[str, list[dict]] = {}
    for d in domain_entries:
        groups.setdefault(d["category"], []).append(d)

    result = []
    for cat_key in sorted(groups, key=lambda k: cat_order.get(k, 999)):
        result.append({
            "key":             cat_key,
            "label":           _cat_label(cat_key),
            "label_ja":        cat_label_ja.get(cat_key, cat_key),
            "label_en":        cat_label_en.get(cat_key, cat_key),
            "default_backend": cat_backend.get(cat_key),
            "domains":         sorted(groups[cat_key], key=lambda d: d["label"]),
        })
    return {"categories": result}


# ---------------------------------------------------------------------------
# Fork (file-level branch)
# ---------------------------------------------------------------------------

@router.post("/fork")
async def fork_branch(req: ForkRequest):
    """Save a trimmed session file containing the lineage up to step_id."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if req.step_id not in engine.steps:
        raise HTTPException(status_code=404, detail=f"Step '{req.step_id}' not found")
    snapshot_file = await _save_fork_snapshot(engine, req.step_id, req.snapshot_filename)
    return {"ok": True, "snapshot_file": snapshot_file}


# ---------------------------------------------------------------------------
# Live branch (in-session fork) — a SECOND branch that coexists with others in
# the same loaded engine, distinct from /fork above (which saves a trimmed
# session as a new FILE). Required for numeric_branch_comparison() to ever
# have two branches to compare.
# ---------------------------------------------------------------------------

@router.post("/branch/fork")
async def live_branch_fork(req: BranchForkRequest):
    """Create a live branch headed at step_id and switch to it."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if req.step_id not in engine.steps:
        raise HTTPException(status_code=404, detail=f"Step '{req.step_id}' not found")
    branch = engine.fork_branch(req.step_id, req.label)
    return {"ok": True, "branch_id": branch.branch_id,
            "active_branch": engine.active_branch_id}


@router.post("/branch/switch")
async def live_branch_switch(req: BranchSwitchRequest):
    """Switch which branch subsequent operations are recorded against."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    if req.branch_id not in engine.branches:
        raise HTTPException(status_code=404, detail=f"Branch '{req.branch_id}' not found")
    engine.switch_branch(req.branch_id)
    return {"ok": True, "active_branch": engine.active_branch_id}


async def _save_fork_snapshot(engine, up_to_step_id: str, filename: str) -> str:
    """Save a trimmed session containing only the lineage up to up_to_step_id."""
    from notebook.runtime.replay import compute_lineage
    from notebook.models.branch import Branch

    full_lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)
    trimmed_steps: dict = {}
    for step in full_lineage:
        trimmed_steps[step.step_id] = step
        if step.step_id == up_to_step_id:
            break

    step_ids = set(trimmed_steps.keys())
    trimmed_exprs = {
        eid: expr for eid, expr in engine.expressions.items()
        if expr.created_by_step in step_ids
    }

    # Reconstruct active assumptions at up_to_step_id by replaying assumption steps.
    # engine.assumptions reflects the current end-of-session state, so assumptions
    # removed AFTER up_to_step_id would be missing — rebuild from step history instead.
    from notebook.models.step import StepKind as SK
    active_assumptions: dict = {}
    for step in trimmed_steps.values():
        if step.kind == SK.assumption_add:
            aid = step.operation_args.get("assumption_id", "")
            canonical = step.operation_args.get("canonical", "")
            if aid in engine.assumptions:
                active_assumptions[aid] = engine.assumptions[aid].model_dump(mode="json")
            else:
                # Removed later in the session (after up_to_step_id) — reconstruct.
                active_assumptions[aid] = {
                    "assumption_id": aid,
                    "canonical": canonical,
                    "display": canonical,
                    "scope": "branch",
                    "branch_id": "main",
                }
        elif step.kind == SK.assumption_remove:
            rc = (step.rendered_command or "").strip()
            m = re.match(r"forget\((.+)\)$", rc)
            if m:
                canonical_removed = m.group(1)
                to_del = [a for a, v in active_assumptions.items()
                          if v.get("canonical") == canonical_removed]
                for a in to_del:
                    del active_assumptions[a]

    snap_branch = Branch(
        branch_id="main",
        head_step_id=up_to_step_id,
        forked_from_step_id=None,
        label="main",
    )

    snap_title = engine.session.title
    if engine.session.title:
        suffix = _t("session.snapshot_suffix", engine.lang)
        if not snap_title.endswith(suffix):
            snap_title = f"{snap_title}{suffix}"

    snap_session = engine.session.model_copy(
        update={"active_branch_id": "main", "title": snap_title}
    )

    # Trim lhs_alias_map to only include expressions in the fork lineage
    trimmed_lhs_alias = {
        lhs: eid for lhs, eid in engine.lhs_alias_map.items()
        if eid in trimmed_exprs
    }

    data = {
        "version": "1",
        "session":     snap_session.model_dump(mode="json"),
        "expressions": {k: v.model_dump(mode="json") for k, v in trimmed_exprs.items()},
        "assumptions": active_assumptions,
        "steps":       {k: v.model_dump(mode="json") for k, v in trimmed_steps.items()},
        "branches":    {"main": snap_branch.model_dump(mode="json")},
        "lhs_alias_map": trimmed_lhs_alias,
        "expr_counter": engine._expr_counter,
        "internal_counter": engine._internal_counter,
        "step_counter": engine._step_counter,
    }

    stem = re.sub(r'[^\w\-]', '_', filename.strip())[:40] if filename.strip() else ""
    if not stem:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"snapshot_{ts}"
    fname = stem if stem.endswith(".json") else f"{stem}.json"
    path = session_io.sessions_dir() / fname
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: path.write_text(_json.dumps(data, indent=2, ensure_ascii=False))
    )
    return fname


# ---------------------------------------------------------------------------
# Plan layer (C-Ext-5) — status polling for step-by-step progress (B-1).
# execute_plan updates the stored plan incrementally, so polling this while a
# plan runs shows live per-step results; a future UI panel reads the same shape.
# ---------------------------------------------------------------------------

@router.get("/plan/status/{plan_id}")
async def plan_status(plan_id: str):
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    plan = engine.plans.get(plan_id)
    if plan is None:
        return {"ok": False, "detail": f"plan {plan_id} not found"}
    return {"ok": True, "plan": plan.model_dump(mode="json")}


@router.post("/plan/retry/{plan_id}")
async def plan_retry(plan_id: str):
    """Retry a stopped/failed plan (D-5, open_items.md) — B-3's retry_plan was
    engine-only until now; this is its first HTTP surface.  Creates a NEW plan
    (immutable history, retried_from=plan_id), so the original stays in
    engine.plans unchanged."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        plan = await loop.run_in_executor(None, lambda: engine.retry_plan(plan_id))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    summary_step = [s for s in engine.steps.values()
                    if s.kind.value == 'plan_execution'][-1]
    return {
        "ok": plan.status == "completed",
        "status": "completed" if plan.status == "completed" else "failed",
        "plan_id": plan.plan_id,
        "step_id": summary_step.step_id,
        "message": summary_step.visible_action,
        "plan_step_results": [r.model_dump(mode="json") for r in plan.step_results],
    }


@router.post("/plan/replan/{plan_id}")
async def plan_replan(plan_id: str):
    """Failure→replan loop (open_items C-Ext-5, 2026-07-18): the "🔄 別の方法で
    再計画" button on a stopped plan card.  Unlike /plan/retry (same plan,
    re-executed once the researcher fixed the root cause), this feeds the
    failure into the Adviser and asks for a DIFFERENT approach — returns the
    same needs_confirmation shape as a fresh 「解法の候補」 ask; the chosen
    candidate still requires confirmation before it runs."""
    from ui.notebook_bridge import get_engine, _get_lang
    from ui.interpret_plan import propose_replan
    engine = get_engine()
    loop = asyncio.get_event_loop()
    return await propose_replan(engine, loop, _get_lang(), plan_id)


@router.get("/plans")
async def plans_list():
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    return {"ok": True,
            "plans": {pid: p.model_dump(mode="json")
                      for pid, p in engine.plans.items()}}


@router.get("/plan/branch_comparison")
async def plan_branch_comparison():
    """C-Ext-5 B-4 (read-only): group already-recorded plan_execution steps
    by branch for comparison — mirrors GET-side of numeric_branch_comparison."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    return {"ok": True, "comparison": engine.plan_branch_comparison()}
