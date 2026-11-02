"""Session persistence: save, autosave, load, and list operations.

Module-level state tracks the current filename and autosave slot so that
successive save/autosave calls in the same session can overwrite or rotate
correctly.  Call reset_save_state() when starting a new session.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from notebook.engine import NotebookEngine


# ---------------------------------------------------------------------------
# Module-level persistence state
# ---------------------------------------------------------------------------

_current_filename: str | None = None
_autosave_slot: int = 0
_autosave_last_step_count: int = -1
_AUTOSAVE_SLOTS = 5

_SESSIONS_DIR = _PROJECT_ROOT / "sessions"


def reset_save_state() -> None:
    """Reset save state — call when starting a fresh session."""
    global _current_filename, _autosave_slot, _autosave_last_step_count
    _current_filename = None
    _autosave_slot = 0
    _autosave_last_step_count = -1


def get_current_filename() -> str | None:
    return _current_filename


def sessions_dir() -> Path:
    _SESSIONS_DIR.mkdir(exist_ok=True)
    return _SESSIONS_DIR


def _safe_filename(title: str) -> str:
    stem = re.sub(r'[^\w\-]', '_', title.strip())[:40] or "session"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}.json"


# ---------------------------------------------------------------------------
# Save / autosave
# ---------------------------------------------------------------------------

def _snap_path(json_path: Path, backend: str = "maxima") -> Path:
    """Return the snapshot path (.jl or .mac) alongside a .json session file."""
    ext = ".jl" if backend.startswith("julia") else ".mac"
    return json_path.with_suffix(ext)


async def do_save(engine: NotebookEngine, filename: str = "", overwrite: bool = False,
                  directory: str = "") -> str:
    """Save session to sessions/[directory/]<filename>.json (+ .mac); return relative path used."""
    global _current_filename
    base = sessions_dir()
    if overwrite and _current_filename:
        rel = _current_filename
    else:
        stem = filename.strip() or _safe_filename(engine.session.title)
        if not stem.endswith(".json"):
            stem += ".json"
        rel = f"{directory}/{stem}" if directory else stem
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(engine.save, path))
    await loop.run_in_executor(None, partial(engine.save_cas_snapshot, _snap_path(path, engine.session.primary_backend)))
    _current_filename = rel
    return rel


async def do_autosave(engine: NotebookEngine) -> dict:
    """Rotate-autosave to sessions/autosave/slot_N.json (+ .mac); skip if unchanged."""
    global _autosave_slot, _autosave_last_step_count
    current_count = engine._step_counter
    if current_count == _autosave_last_step_count:
        return {"ok": True, "skipped": True, "reason": "no changes"}
    _autosave_slot = (_autosave_slot % _AUTOSAVE_SLOTS) + 1
    autosave_dir = sessions_dir() / "autosave"
    autosave_dir.mkdir(exist_ok=True)
    path = autosave_dir / f"slot_{_autosave_slot}.json"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(engine.save, path, is_autosave=True))
    await loop.run_in_executor(None, partial(engine.save_cas_snapshot, _snap_path(path, engine.session.primary_backend)))
    _autosave_last_step_count = current_count
    return {"ok": True, "skipped": False, "slot": _autosave_slot}


# ---------------------------------------------------------------------------
# List / load
# ---------------------------------------------------------------------------

async def do_list_sessions(path: str = "") -> list[dict]:
    """Return metadata for .json files directly in sessions/<path>/, newest first."""
    base = sessions_dir()
    if path:
        target = (base / path).resolve()
        if not str(target).startswith(str(base.resolve())):
            return []
    else:
        target = base
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(target.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        try:
            import json as _json
            data = _json.loads(f.read_text(encoding="utf-8"))
            sess = data.get("session", {})
            rel = str(f.relative_to(base))
            result.append({
                "filename":    rel,
                "title":       sess.get("title", f.stem),
                "modified_at": datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
                "expr_count":  len(data.get("expressions", {})),
                "step_count":  len(data.get("steps", {})),
            })
        except Exception:
            continue
    return result


async def do_load(filename: str) -> NotebookEngine:
    """Load a session from sessions/<filename>.json.

    CAS state is restored from the .mac snapshot if present; otherwise falls
    back to full replay of the active branch.  The notebook JSON is always
    the authoritative source for expression/step data.
    """
    path = sessions_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {filename}")
    loop = asyncio.get_event_loop()
    new_engine = await loop.run_in_executor(None, partial(NotebookEngine.load, path))
    # Use the filename stem as the session title when the stored title is the
    # generic default ("Notebook" / "Untitled"), so the status dialog shows
    # the file name rather than the uninformative placeholder.
    stored_title = new_engine.session.title or ""
    if stored_title in ("", "Notebook", "Untitled"):
        file_stem = path.stem  # e.g. "BCS" from "BCS.json"
        # Strip autosave slot suffix: "slot_3" → keep as is, but readable names preferred
        new_engine.session = new_engine.session.model_copy(update={"title": file_stem})
    await loop.run_in_executor(None, new_engine.start)

    snap = _snap_path(path, new_engine.session.primary_backend)
    if snap.exists():
        ok = await loop.run_in_executor(None, partial(new_engine.load_cas_snapshot, snap))
        if not ok:
            # Snapshot load failed — fall back to full replay
            await loop.run_in_executor(None, new_engine.replay_current_branch)
    else:
        await loop.run_in_executor(None, new_engine.replay_current_branch)

    global _current_filename, _autosave_last_step_count
    _current_filename = filename
    _autosave_last_step_count = new_engine._step_counter
    return new_engine
