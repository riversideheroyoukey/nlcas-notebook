"""Notebook engine bridge for the FastAPI UI server.

Owns the NotebookEngine singleton lifecycle and exposes REST endpoints
under /api/notebook/.  Import this module from ui/app.py and call
start_engine() / stop_engine() in the lifespan context manager.

All Maxima operations are blocking (subprocess I/O).  Endpoints that
call the engine run the work in a thread-pool executor so the uvicorn
event loop is never blocked.

Interpretation pipeline logic lives in ui/interpret_pipeline.py.
Session persistence logic lives in ui/session_io.py.
"""
from __future__ import annotations

import asyncio
import json as _json
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from notebook.domain_paths import DOMAINS_ROOT
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.i18n import t as _t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

# InputToken / InterpretRequest are re-exported for tests that import them via
# ui.notebook_bridge; /interpret itself now lives in ui/routes/advisory.py (R2).
from ui.interpret_pipeline import InputToken, InterpretRequest  # noqa: F401
import ui.session_io as session_io
from config.config_utils import (get_default_language, get_default_supervisor_target,
                                  get_default_adviser_target, get_default_coder_target,
                                  get_supervisor_scope, get_adviser_scope)

router = APIRouter(prefix="/api/notebook", tags=["notebook"])


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine:     NotebookEngine | None = None
_supervisor: LLMSupervisor | None = None
_advisers:   dict = {}   # name -> LLMAdviser



def _get_lang() -> str:
    """Return the active display language.

    Uses the running session's language when the engine is live;
    falls back to default_settings.json so pre-session endpoints
    (e.g. /domains) also return correctly localised labels.
    """
    if _engine is not None:
        return _engine.lang
    return get_default_language()


def _make_supervisor(domains: list[str] | None = None, *, primary_backend: str,
                     preferred_name: str | None = None) -> LLMSupervisor:
    """primary_backend is REQUIRED: defaulting it to a CAS name meant a caller
    that forgot it silently restricted the supervisor to Maxima's operation set
    (open_items AUDIT-SHARED-CAS-LEAK)."""
    try:
        from notebook.runtime.llm_client import LLMClient
        name = preferred_name or get_default_supervisor_target() or None
        llm = LLMClient.find_endpoint(
            role="supervisor",
            scope=get_supervisor_scope() or None,
            models_root=_PROJECT_ROOT / "models",
            preferred_name=name,
        )
    except Exception:
        llm = None
    return LLMSupervisor(validator=OperationValidator(), llm_client=llm, domains=domains or [],
                         primary_backend=primary_backend)


def _make_all_advisers() -> dict:
    """Discover all endpoint.json files with advisor role and return {name: LLMAdviser}."""
    from notebook.runtime.llm_adviser import LLMAdviser
    from notebook.runtime.llm_client import LLMClient
    result: dict = {}
    try:
        preferred = get_default_adviser_target() or None
        clients = LLMClient.find_all_endpoints(role="advisor",
                                               models_root=_PROJECT_ROOT / "models")
        if preferred:
            clients.sort(key=lambda c: (0 if c.endpoint_name == preferred else 1,
                                        c.endpoint_name))
        for llm in clients:
            result[llm.endpoint_name] = LLMAdviser(llm_client=llm)
    except Exception:
        pass
    return result


def _infer_session_mode(domains: list[str]) -> SessionMode:
    """Session mode implied by the selected domains' categories.

    A category in ``domains/categories.json`` declares a ``default_backend``;
    the mode it corresponds to is resolved by ``backend_factory`` rather than by
    comparing against a backend name here, so the mode→backend table has a
    single owner and a third CAS needs no edit in this file (open_items
    AUDIT-SHARED-CAS-LEAK).  Falls back to ``standard_symbolic`` when no domain
    asks for anything else."""
    from notebook.runtime.backend_factory import session_mode_for_backend
    _domains_root = DOMAINS_ROOT
    _cats_path = _domains_root / "categories.json"
    try:
        cats = {c["key"]: c for c in _json.loads(_cats_path.read_text(encoding="utf-8"))}
    except Exception:
        return SessionMode.standard_symbolic
    for domain in domains:
        meta_path = _domains_root / domain / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            cat = cats.get(meta.get("category", ""), {})
            _mode = session_mode_for_backend(cat.get("default_backend") or "")
            if _mode and _mode != SessionMode.standard_symbolic.value:
                return SessionMode(_mode)
        except Exception:
            continue
    return SessionMode.standard_symbolic


def start_engine(title: str = "Notebook", mode: str = "standard", domains: list[str] | None = None,
                 language: str | None = None, supervisor_name: str | None = None,
                 coordinate_system: str | None = None) -> None:
    global _engine, _supervisor, _advisers
    if _engine is not None:
        # Defensive close-before-reopen: without this, calling start_engine()
        # twice without an intervening stop_engine() silently overwrites the
        # module-level _engine reference, orphaning the previous engine's CAS
        # subprocess (Maxima or Julia) with no caller left to close it. Julia
        # has a weakref-finalizer safety net for exactly this; Maxima has
        # none (see MaximaAdapter), so this guard is the only thing that
        # reliably prevents a leaked Maxima process in that case.
        stop_engine()
    if language is None:
        language = get_default_language()
    if domains:
        session_mode = _infer_session_mode(domains)
    else:
        session_mode = (
            SessionMode.standard_symbolic if mode == "standard"
            else SessionMode.second_quantization
        )
    _engine     = NotebookEngine(title=title, mode=session_mode,
                                 coordinate_system=coordinate_system).__enter__()
    if domains:
        _engine.session.domains = domains
        # start() (called by __enter__ above) already copied the then-empty domains
        # into symbol_context; re-sync so domain-gated behaviour (e.g. Lagrangian
        # mode) sees the selected domains.
        _engine.symbol_context.domains = list(domains)
    _engine.session.language = [language, "en"] if language != "en" else ["en", "ja"]
    _supervisor = _make_supervisor(domains=domains or [], primary_backend=_engine.session.primary_backend,
                                   preferred_name=supervisor_name)
    _advisers   = _make_all_advisers()
    session_io.reset_save_state()


def stop_engine() -> None:
    global _engine, _supervisor, _advisers
    if _engine is not None:
        _engine.__exit__(None, None, None)
        _engine     = None
        _supervisor = None
        _advisers   = {}


def install_loaded_engine(new_engine: "NotebookEngine") -> "NotebookEngine":
    """Adopt a freshly loaded engine as the singleton and rebuild the supervisor
    and advisers for its session.  The /load route (ui/routes/persistence.py, R2)
    calls this so the module-global reassignment stays here, where the globals
    live, rather than being reached into from a route module."""
    global _engine, _supervisor, _advisers
    _engine     = new_engine
    _supervisor = _make_supervisor(domains=new_engine.session.domains,
                                   primary_backend=new_engine.session.primary_backend)
    _advisers   = _make_all_advisers()
    return _engine


def get_engine() -> NotebookEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="NotebookEngine not started")
    return _engine


def get_supervisor() -> LLMSupervisor:
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not started")
    return _supervisor


def get_adviser(name: str | None = None):
    """Return the named adviser, or the first available one if name is None."""
    if name and name in _advisers:
        return _advisers[name]
    # Default: first entry (dict preserves insertion = alphabetical order)
    for adv in _advisers.values():
        return adv
    raise HTTPException(status_code=503, detail="Adviser not configured")


def get_advisers() -> dict:
    """Return the {name: LLMAdviser} roster for the current session (an empty dict
    when none are configured).  Symmetric with get_engine/get_supervisor: route
    modules read the roster through this instead of the _advisers global, so the
    module state stays owned here (R2, docs/27)."""
    return _advisers


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compute endpoints (/register, /operation, /re_evaluate, /wick_contract) and
# their side-effect / dependency-context helpers live in ui/routes/operations.py
# (R2, docs/27) — mounted at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session state / status / lifecycle / domains / fork routes live in
# ui/routes/session.py (R2, docs/27) — mounted at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LaTeX export (/export/latex) and its TeX-rendering helpers live in
# ui/routes/export.py (R2, docs/27) — mounted at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Notes / deletions / summary / timeline / assumptions live in
# ui/routes/history.py (R2, docs/27) — mounted at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM-facing routes (/interpret, /supervisor-list, /adviser-list,
# /adviser-status, /advise) + the session-narrative helper live in
# ui/routes/advisory.py (R2, docs/27) — mounted at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session persistence & file-browser endpoints live in ui/routes/persistence.py
# (R2, docs/27) — mounted at the bottom of this file.  /load reassigns the
# engine singleton via install_loaded_engine() above.
# ---------------------------------------------------------------------------


# Plan-status routes (/plans, /plan/status) live in ui/routes/session.py (R2).


# ---------------------------------------------------------------------------
# Numeric-layer routes live in ui/routes/numeric.py (R2, docs/27).  Mount them
# under this router's /api/notebook prefix and re-export the public names so
# existing imports (tests, notebook_main paths) keep resolving via
# ui.notebook_bridge.  Imported at the BOTTOM so get_engine and the shared
# helpers above are already defined when numeric.py lazily reaches for them.
# ---------------------------------------------------------------------------
from ui.routes.numeric import router as _numeric_router  # noqa: E402
router.include_router(_numeric_router)
from ui.routes.numeric import (  # noqa: E402,F401
    NumericForkRequest, NumericProposeSpecRequest, NumericGenerateLLMRequest,
    NumericRefluxRequest, NumericApproveRequest, NumericRunRequest,
    _resolve_coder_chat,
    numeric_fork, numeric_propose_spec, numeric_generate, numeric_run,
    numeric_tasks, numeric_graph, numeric_params, numeric_figure,
    numeric_coders, numeric_generate_llm, numeric_reflux, numeric_approve,
    numeric_diffcheck,
)

# History / annotation routes (notes, deletions, summary, timeline, assumptions)
# live in ui/routes/history.py (R2, docs/27).  Same mount + re-export idiom as
# numeric above, so existing imports via ui.notebook_bridge keep resolving.
from ui.routes.history import router as _history_router  # noqa: E402
router.include_router(_history_router)
from ui.routes.history import (  # noqa: E402,F401
    NoteRequest, AssumptionRequest, _generate_summary_text,
    add_note, delete_expression, delete_note, delete_user_step, delete_define,
    generate_summary, get_timeline, list_assumptions, add_assumption,
    remove_assumption,
)

# Session-persistence & file-browser routes (/save, /load, /browse-dirs, …) live
# in ui/routes/persistence.py (R2, docs/27).  /load calls install_loaded_engine
# (defined above) to reassign the engine singleton.
from ui.routes.persistence import router as _persistence_router  # noqa: E402
router.include_router(_persistence_router)
from ui.routes.persistence import (  # noqa: E402,F401
    MkdirRequest, SaveRequest, LoadRequest,
    browse_dirs, make_dir, get_save_status, save_session, autosave_session,
    list_sessions, load_session,
)

# LaTeX export route (/export/latex) + its TeX-rendering helpers live in
# ui/routes/export.py (R2, docs/27).  Pure helpers + get_engine, no bridge state.
from ui.routes.export import router as _export_router  # noqa: E402
router.include_router(_export_router)
from ui.routes.export import (  # noqa: E402,F401
    export_latex, _tex_escape, _you_card_latex, _note_to_latex,
    _text_unicode_to_math, _note_inline, _has_nl_words,
)

# Compute routes (/register, /operation, /re_evaluate, /wick_contract) + their
# side-effect / dependency-context helpers live in ui/routes/operations.py (R2,
# docs/27).  Only get_engine coupling (lazy); the helpers use engine.lang.
from ui.routes.operations import router as _operations_router  # noqa: E402
router.include_router(_operations_router)
from ui.routes.operations import (  # noqa: E402,F401
    RegisterRequest, OperationRequest, ReEvaluateRequest, WickContractRequest,
    _handle_side_effect_operation, _prepare_operation_context,
    register_expression, run_operation, re_evaluate_expression,
    wick_contract_expression,
)

# Session state / status / lifecycle / domains / fork / plans routes live in
# ui/routes/session.py (R2, docs/27).  Reaches the bridge lazily for get_engine,
# start_engine/stop_engine, _get_lang, and get_advisers (the roster accessor).
from ui.routes.session import router as _session_router  # noqa: E402
router.include_router(_session_router)
from ui.routes.session import (  # noqa: E402,F401
    NewSessionRequest, ForkRequest, _save_fork_snapshot,
    get_session, clear_session, get_status, get_ui_lang, new_session,
    list_domains, fork_branch, plan_status, plans_list,
)

# LLM-facing routes (/interpret, /supervisor-list, /adviser-list, /adviser-status,
# /advise) + the session-narrative helper live in ui/routes/advisory.py (R2,
# docs/27).  Reaches the bridge lazily for get_engine/get_supervisor/get_adviser/
# get_advisers.
from ui.routes.handoff import router as _handoff_router  # noqa: E402
router.include_router(_handoff_router)

from ui.routes.advisory import router as _advisory_router  # noqa: E402
router.include_router(_advisory_router)
from ui.routes.advisory import (  # noqa: E402,F401
    AdviseRequest, _build_session_narrative,
    interpret_command, list_supervisors, list_advisers, get_adviser_status,
    advise_expression,
)
