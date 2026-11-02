"""Session-persistence & file-browser HTTP endpoints — extracted from
notebook_bridge (R2).

This family owns saving/loading sessions and browsing the ``sessions/``
directory: /browse-dirs, /mkdir, /save-status, /save, /autosave, /sessions,
/load.  Every path composes under the bridge's ``/api/notebook`` prefix, so the
routes are byte-for-byte the same as before the split; only their home moved.

``session_io`` is a leaf module (it does not import the bridge), so it is
imported at module top.  Anything that lives on the bridge — the engine
singleton, ``stop_engine``, and the ``install_loaded_engine`` installer that owns
the module-global reassignment for /load — is imported LAZILY inside the
handler, both to avoid the import cycle (this module is mounted from the bottom
of notebook_bridge) and to keep ``patch("ui.notebook_bridge.get_engine")``
working in tests.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import ui.session_io as session_io

router = APIRouter()


class MkdirRequest(BaseModel):
    path: str


class SaveRequest(BaseModel):
    filename: str = ""       # explicit name; empty → auto-generated
    overwrite: bool = False  # True → overwrite last saved file
    directory: str = ""      # relative to sessions/; empty = sessions/ root


class LoadRequest(BaseModel):
    filename: str


@router.get("/browse-dirs")
async def browse_dirs(path: str = ""):
    """Return immediate subdirectories and .json session files within sessions/."""
    base = session_io.sessions_dir().resolve()
    target = (base / path).resolve() if path else base
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Path outside sessions directory")
    target.mkdir(parents=True, exist_ok=True)
    dirs = sorted(
        d.name for d in target.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )
    files = sorted(
        f.name for f in target.iterdir()
        if f.is_file() and f.suffix == ".json" and not f.name.startswith('.')
    )
    rel = str(target.relative_to(base)) if target != base else ""
    if rel == ".":
        rel = ""
    can_go_up = target != base
    parent_rel = str(target.parent.relative_to(base)) if can_go_up else None
    if parent_rel == ".":
        parent_rel = ""
    return {"path": rel, "dirs": dirs, "files": files, "can_go_up": can_go_up, "parent": parent_rel}


@router.post("/mkdir")
async def make_dir(req: MkdirRequest):
    """Create a new subdirectory within sessions/."""
    base = session_io.sessions_dir().resolve()
    target = (base / req.path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Path outside sessions directory")
    target.mkdir(parents=True, exist_ok=True)
    rel = str(target.relative_to(base))
    return {"ok": True, "path": rel}


@router.get("/save-status")
async def get_save_status():
    """Return the filename of the last explicit save (None if never saved)."""
    return {"current_filename": session_io.get_current_filename()}


@router.post("/save")
async def save_session(req: SaveRequest):
    """Save the current session.

    overwrite=True  → overwrite the last saved file (requires prior save).
    overwrite=False → create a new file (auto-named if filename is empty).
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    try:
        filename = await session_io.do_save(engine, req.filename, req.overwrite, req.directory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "filename": filename}


@router.post("/autosave")
async def autosave_session():
    """Rotate-autosave to sessions/autosave/slot_N.json (N=1-5).

    Skipped if no new steps have been added since the last autosave.
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    try:
        result = await session_io.do_autosave(engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/sessions")
async def list_sessions(path: str = ""):
    """List .json files directly in sessions/<path>/."""
    sessions = await session_io.do_list_sessions(path=path)
    return {"sessions": sessions}


@router.post("/load")
async def load_session(req: LoadRequest):
    """Load a session from sessions/<filename>.json, replaying its active branch."""
    from ui.notebook_bridge import stop_engine, install_loaded_engine
    # Check file existence before stopping the current engine
    if not (session_io.sessions_dir() / req.filename).exists():
        raise HTTPException(status_code=404, detail=f"Session file not found: {req.filename}")

    stop_engine()
    try:
        new_engine = await session_io.do_load(req.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")

    engine = install_loaded_engine(new_engine)

    expressions = {
        eid: {"canonical": expr.canonical, "latex": expr.latex, "branch_id": expr.branch_id}
        for eid, expr in engine.expressions.items()
    }
    return {
        "ok":            True,
        "title":         engine.session.title,
        "active_branch": engine.active_branch_id,
        "expressions":   expressions,
        "step_count":    len(engine.steps),
    }
