"""Numeric-layer HTTP endpoints (N-1) — extracted from notebook_bridge (R2).

Every path here composes under the bridge's ``/api/notebook`` prefix, so the
routes are byte-for-byte the same as before the split; only their home moved.
The engine singleton and the shared ``_build_session_narrative`` helper are
imported LAZILY inside handlers (this module is mounted from the bottom of
notebook_bridge, so a top-level import would be a cycle) — the same lazy-import
idiom the bridge already uses pervasively.
"""
from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.config_utils import get_default_coder_target, get_default_numeric_timeout_s
from ui.numeric_products import numeric_downloads, product_path
from notebook.runtime.i18n import t as _t
import ui.session_io as session_io

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class NumericForkRequest(BaseModel):
    expr_id: str
    problem_type: str = "parameter_sweep"
    target_quantity: str = ""
    sweep_parameters: dict = {}
    fixed_values: dict = {}
    approximations: list[str] = []
    spec: "dict | None" = None   # a NumericSpec dump (docs/24/25) — approved via the ChatBot


@router.post("/numeric/fork")
async def numeric_fork(req: NumericForkRequest):
    """Fork a theory expression into a NumericTask (provenance snapshot).

    When ``spec`` (a faceted NumericSpec, docs/24) is given, it takes
    precedence: fork_numeric_task validates and projects it onto the flat
    fields itself (re-validating regardless of any earlier
    /numeric/propose_spec check — defense in depth), so the flat fields on
    this request are ignored in that case.  The response's ``codegen_route``
    tells the caller which generate endpoint to call next (deterministic
    template vs LLM) without re-deriving that dispatch client-side — routing
    stays a single server-side decision (intent_registry.codegen_route)."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    spec = None
    if req.spec:
        from notebook.models.numeric import NumericSpec
        spec = NumericSpec.model_validate(req.spec)
    try:
        task = await loop.run_in_executor(None, partial(
            engine.fork_numeric_task, req.expr_id,
            problem_type=req.problem_type,
            target_quantity=req.target_quantity,
            sweep_parameters=req.sweep_parameters,
            fixed_values=req.fixed_values,
            approximations=req.approximations,
            spec=spec,
        ))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    from notebook.numeric.intent_registry import codegen_route_for_spec
    route = codegen_route_for_spec(task.spec) if task.spec else "deterministic"
    return {"ok": True, "task": task.model_dump(mode="json"), "codegen_route": route}


class NumericProposeSpecRequest(BaseModel):
    expr_id: str
    goal: str = ""
    prior_spec: "dict | None" = None    # a NumericSpec dump from a previous turn
    refinement: str = ""                # researcher's correction (refine turn only)


@router.post("/numeric/propose_spec")
async def numeric_propose_spec(req: NumericProposeSpecRequest):
    """One B-γ dialogue turn (docs/25 §3): the Adviser LLM proposes (or, given
    ``prior_spec``+``refinement``, revises) a NumericSpec for ``expr_id``,
    grounded in the notebook's context up to that expression.  The spec is
    NOT forked into a NumericTask here — that happens on the researcher's
    explicit spec approval, via the existing POST /numeric/fork with a
    ``spec`` body (fork_numeric_task re-validates regardless, defense in
    depth)."""
    from ui.notebook_bridge import get_engine, _build_session_narrative
    engine = get_engine()
    expr = engine.expressions.get(req.expr_id)
    if expr is None:
        return {"ok": False,
                "detail": _t("numeric.task_source_missing", engine.lang,
                             expr_id=req.expr_id)}

    from ui.interpret_plan import _resolve_adviser_chat
    chat = _resolve_adviser_chat()
    if chat is None:
        return {"ok": False, "detail": _t("plan.adviser_unavailable", engine.lang)}

    chat.max_tokens = 4096
    from notebook.models.numeric import NumericSpec
    from notebook.numeric.spec_adviser import propose_spec
    prior = NumericSpec.model_validate(req.prior_spec) if req.prior_spec else None
    free_symbols = engine._numeric_free_symbols(expr.canonical)
    narrative = _build_session_narrative(engine, up_to_step_id=engine.active_branch.head_step_id)
    catalog = [{"expr_id":eid, "canonical":e.canonical, "latex":e.latex,
                "branch_id":e.branch_id} for eid, e in engine.expressions.items()]
    target_summary = expr.latex or expr.canonical

    loop = asyncio.get_event_loop()
    spec, issues = await loop.run_in_executor(None, partial(
        propose_spec, chat, req.goal, target_summary, free_symbols,
        narrative, prior, req.refinement, catalog))
    if spec is None:
        error = getattr(chat, "last_error", "")
        return {"ok": False, "stage": "adviser" if error else "parse",
                "response_excerpt": getattr(chat, "last_reply", "")[:2000],
                "detail": (_t("numeric.adviser_failed", engine.lang, detail=error) if error
                           else _t("numeric.spec_unparseable", engine.lang))}
    # Translate issue codes server-side (same policy as every other endpoint
    # in this file) — the frontend never re-derives i18n from a raw code.
    issue_msgs = [_t(i["code"], engine.lang,
                     **{k: v for k, v in i.items() if k != "code"})
                 for i in issues]
    return {"ok": True, "spec": spec.model_dump(mode="json"), "issues": issue_msgs}


@router.post("/numeric/generate/{task_id}")
async def numeric_generate(task_id: str):
    """Deterministically generate the code artifact for a task."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        art = await loop.run_in_executor(
            None, partial(engine.generate_numeric_code, task_id))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return {"ok": True, "artifact_id": art.artifact_id,
            "code": art.content, "meta": art.metadata}


def _shape_numeric_run_response(engine, execution) -> dict:
    """Shape a completed NumericExecution into the response both
    ``POST /numeric/run/{task_id}`` and the ``numeric_fork_nl`` NL-trigger
    confirm-dispatch (ui/routes/operations.py) return — kept in one place so
    the two call sites can never drift apart."""
    from notebook.models.step import StepKind
    artifacts = [
        engine.numeric_artifacts[a].model_dump(mode="json")
        for a in execution.result_artifact_ids
    ]
    # The reflux step (numeric_result) carries the human-readable summary and
    # the figure id; hand them back so the UI can append the SAME entry shape
    # that GET /timeline emits on reload — no full-timeline refetch needed.
    _fig_id = next((a for a in execution.result_artifact_ids
                    if engine.numeric_artifacts[a].artifact_type == "figure"), None)
    _step = next((s for s in engine.steps.values()
                  if s.kind == StepKind.numeric_result
                  and (s.operation_args or {}).get("execution_id")
                  == execution.execution_id), None)
    # The step status is the SOURCE OF TRUTH for the UI: it is failed both on a
    # sandbox error AND on an all-NaN result (the run exited 0 but left the
    # expression's valid regime).  Reflect it as `ok` and suppress the (blank)
    # figure so the UI flags the NaN case instead of showing an empty plot.
    _result_ok = (_step is None) or getattr(_step.status, "value", "") != "failed"
    return {"ok": _result_ok,
            "execution": execution.model_dump(mode="json"),
            "artifacts": artifacts,
            "downloads": numeric_downloads(engine, execution.result_artifact_ids),
            "result_text": _step.visible_action if _step else "",
            "figure_artifact_id": _fig_id if _result_ok else None,
            "step_id": _step.step_id if _step else None}


class NumericRunRequest(BaseModel):
    timeout_s: float | None = None    # None = use the configured default


@router.post("/numeric/run/{task_id}")
async def numeric_run(task_id: str, req: "NumericRunRequest | None" = None):
    """Run the task's code in the sandbox and reflux the result step.

    Numeric outputs are anchored to the saved session's folder/filename
    (<stem>_numeric/); an unsaved session is auto-saved first."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    if session_io.get_current_filename() is None:
        await session_io.do_save(engine)
    timeout_s = (req.timeout_s if req and req.timeout_s else None) or get_default_numeric_timeout_s()
    try:
        execution = await loop.run_in_executor(
            None, partial(engine.run_numeric_task, task_id, timeout_s=timeout_s))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return _shape_numeric_run_response(engine, execution)


@router.get("/numeric/tasks")
async def numeric_tasks():
    """List numeric tasks with their executions (for a future panel).

    Each task carries its stale report (docs/26 §7) — report-only, the UI
    may badge it but nothing is blocked."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    tasks = {}
    for tid, task in engine.numeric_tasks.items():
        d = task.model_dump(mode="json")
        d["stale"] = engine.numeric_task_stale_info(tid)
        tasks[tid] = d
    return {
        "ok": True,
        "tasks": tasks,
        "executions": {xid: x.model_dump(mode="json")
                       for xid, x in engine.numeric_executions.items()},
    }


@router.get("/numeric/graph")
async def numeric_graph():
    """The derived numeric-computation DAG (docs/26 §6) for the graph
    panel — plus the branch comparison (docs/26 §10-c), which shares the
    modal.  Both are read-only derivations; no CAS state is touched."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    return {"ok": True,
            "graph": engine.numeric_graph_view(),
            "branch_comparison": engine.numeric_branch_comparison()}


@router.get("/numeric/params/{expr_id}")
async def numeric_params(expr_id: str):
    """Free symbols of an expression that a numeric fork must bind.

    The task-editor modal asks the researcher to pick ONE of these as the
    sweep variable and supply a fixed value for the rest.  This reuses the
    EXACT lowering the deterministic template uses (canonical_to_numpy), so
    the modal asks for precisely the symbols codegen will demand — no drift
    between what the UI collects and what generate_numeric_code needs.
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    expr = engine.expressions.get(expr_id)
    if expr is None:
        return {"ok": False,
                "detail": _t("numeric.task_source_missing", engine.lang,
                             expr_id=expr_id)}
    # Same non-scalar guard as fork_numeric_task: a list/matrix canonical
    # ([[…]] from eigenvalues/solve) is not a scalar sweep target.
    if (expr.canonical or "").lstrip().startswith("["):
        return {"ok": False,
                "detail": _t("numeric.list_not_supported", engine.lang,
                             expr_id=expr_id)}
    from notebook.numeric.expr_to_numpy import canonical_to_numpy
    from notebook.runtime.shared_parser import UnknownNotationError
    try:
        _expr_np, free = canonical_to_numpy(expr.canonical, engine.symbol_context)
    except UnknownNotationError as exc:
        # Not lowerable by the deterministic template (whitelist-external
        # function / non-scalar construct).  Slice 1 is deterministic-only;
        # the LLM path (a later slice) can still handle such expressions.
        return {"ok": False,
                "detail": str(engine._translate_unknown_notation(exc))}
    params = sorted(free)
    # LaTeX display form per parameter (e.g. "omega_0" -> "\omega_{0}"), so
    # the modal can render the SAME Greek/subscript notation the rest of the
    # UI uses instead of showing the bare canonical identifier.
    display = {p: engine._handler._canonical_name_to_latex(p) for p in params}
    return {"ok": True, "expr_id": expr_id, "latex": expr.latex,
            "parameters": params, "display": display}


@router.get("/numeric/file/{artifact_id}")
async def numeric_file(artifact_id: str):
    """Download a known product; resolved paths must remain in this session."""
    from ui.notebook_bridge import get_engine
    try:
        path = product_path(get_engine(), artifact_id)
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="numeric product not found")
    return FileResponse(str(path), filename=path.name)


@router.get("/numeric/figure/{artifact_id}")
async def numeric_figure(artifact_id: str):
    """Serve a numeric figure artifact (PNG) by id, so the timeline can
    display the plot inline.

    Defense in depth: the artifact must be a known 'figure' artifact of THIS
    session, and its resolved path must sit inside the session's own numeric
    output root (<stem>_numeric/).  This blocks a crafted artifact_id or a
    tampered path field from reading arbitrary files off disk.
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    art = engine.numeric_artifacts.get(artifact_id)
    if art is None or art.artifact_type != "figure" or not art.path:
        raise HTTPException(status_code=404, detail="figure not found")
    try:
        # _numeric_output_root() raises ValueError on an unsaved session
        # (no figures can exist yet); relative_to() raises ValueError if the
        # resolved path escapes the session's numeric root.
        root = engine._numeric_output_root().resolve()
        fpath = Path(art.path).resolve()
        fpath.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="figure not found")
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="figure not found")
    return FileResponse(str(fpath), media_type="image/png")


# ---------------------------------------------------------------------------
# Numeric layer, vibe-coding path (N-1 phase 2): LLM writes the script from
# the TeX/canonical form; the researcher reviews and approves; the sandbox
# runs it.  The review gate is enforced engine-side.
# ---------------------------------------------------------------------------

def _resolve_coder_chat(name: "str | None" = None, models_root=None):
    """Resolve the numeric-layer coding LLM as a chat(system, user) callable.

    Resolution order: explicit *name* > config coder_target > first endpoint
    with a 'coder' role > the advisor chain (fallback).  The role's own
    temperature/max_tokens from endpoint.json apply (coder default 4096 —
    scripts are longer than plan-candidate JSON)."""
    from notebook.runtime.llm_client import LLMClient
    preferred = name or get_default_coder_target() or None
    llm = LLMClient.find_endpoint(role="coder",
                                  models_root=models_root or (_PROJECT_ROOT / "models"),
                                  preferred_name=preferred)
    if llm is None:
        from ui.interpret_plan import _resolve_adviser_chat
        return _resolve_adviser_chat()

    def _chat(system: str, user: str) -> str:
        resp = llm.chat(system=system, user=user,
                        temperature=getattr(llm, "default_temperature", 0.2),
                        max_tokens=getattr(llm, "default_max_tokens", None) or 4096)
        return resp.text if getattr(resp, "ok", False) else ""
    _chat.endpoint_name = getattr(llm, "endpoint_name", "")   # introspection
    return _chat


class NumericGenerateLLMRequest(BaseModel):
    coder: str = ""     # optional endpoint name; empty = coder_target default


@router.get("/numeric/coders")
async def numeric_coders():
    """List endpoints offering the 'coder' role (for a selection dropdown
    when entering the numeric layer) and which one is the default."""
    from notebook.runtime.llm_client import LLMClient
    clients = LLMClient.find_all_endpoints(role="coder",
                                           models_root=_PROJECT_ROOT / "models")
    return {"ok": True,
            "default": get_default_coder_target(),
            "coders": [{"name": c.endpoint_name, "model": c.model,
                        "max_tokens": getattr(c, "default_max_tokens", None)}
                       for c in clients]}


@router.post("/numeric/generate_llm/{task_id}")
async def numeric_generate_llm(task_id: str,
                               req: "NumericGenerateLLMRequest | None" = None):
    """Ask the coding LLM for the script (returned for review, NOT runnable
    until /numeric/approve).  Optional body {"coder": name} overrides the
    configured default endpoint."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    chat = _resolve_coder_chat((req.coder or None) if req else None)
    try:
        art = await loop.run_in_executor(None, partial(
            engine.generate_numeric_code_llm, task_id, chat))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return {"ok": True, "artifact_id": art.artifact_id, "code": art.content,
            "reviewed": False,
            "coder": getattr(chat, "endpoint_name", ""),
            "had_template_scaffold": art.metadata.get("had_template_scaffold")}


class NumericRefluxRequest(BaseModel):
    name: str    # symbol name for the theory-layer equation `name = value`


@router.post("/numeric/reflux/{artifact_id}")
async def numeric_reflux(artifact_id: str, req: NumericRefluxRequest):
    """Register a typed scalar artifact as a theory expression (docs/26 §5:
    the layer-crossing counterpart of value_from binding).  The name becomes
    a substitutable symbol via the H-1 equation machinery."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        expr = await loop.run_in_executor(None, partial(
            engine.reflux_numeric_value, artifact_id, req.name))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return {"ok": True, "expr_id": expr.display_id, "latex": expr.latex,
            "canonical": expr.canonical}


class NumericApproveRequest(BaseModel):
    edited_source: "str | None" = None   # None = approve as-generated


@router.post("/numeric/approve/{artifact_id}")
async def numeric_approve(artifact_id: str,
                          req: "NumericApproveRequest | None" = None):
    """The researcher's explicit review approval of LLM-generated code.

    docs/25 §5: the code panel is editable; an optional body
    {"edited_source": "..."} carries the researcher's edit, which becomes the
    approved (and hence runnable) artifact — omitting it approves the code
    as-generated."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    edited = req.edited_source if req else None
    try:
        art = await loop.run_in_executor(None, partial(
            engine.approve_numeric_code, artifact_id, edited))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return {"ok": True, "artifact_id": art.artifact_id, "reviewed": True,
            "edited": art.metadata.get("edited", False)}


@router.post("/numeric/diffcheck/{task_id}")
async def numeric_diffcheck(task_id: str):
    """Differential check: run LLM code and the deterministic template side
    by side and compare their NB_RESULT summaries (whitelist-covered tasks
    only — where a ground truth exists)."""
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    loop = asyncio.get_event_loop()
    if session_io.get_current_filename() is None:
        await session_io.do_save(engine)
    try:
        verdict = await loop.run_in_executor(None, partial(
            engine.differential_check_llm_code, task_id))
    except ValueError as e:
        return {"ok": False, "detail": str(e)}
    return {"ok": True, "verdict": verdict}
