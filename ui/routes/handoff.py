"""SCALAR-HANDOFF HTTP endpoints (docs/34 §5).

Three steps, because a button cannot be read the way a typed sentence can:
PREVIEW shows what would cross, CREATE does it, and opening the result is a
separate, explicit act.  The NL route (`ui/interpret_handoff.py`) skips the
preview — there the researcher has already written what they want in words.

Mounted under the bridge's ``/api/notebook`` prefix; the engine singleton is
imported lazily inside the handlers, the idiom every route module here uses.
"""
from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from notebook.runtime.i18n import t as _t
from notebook.runtime.scalar_handoff import HandoffRefused, perform_handoff

router = APIRouter()


class HandoffRequest(BaseModel):
    expr_id: str
    target_backend: str = "maxima"


def _lang(engine) -> str:
    return engine.lang


@router.post("/handoff/preview")
async def handoff_preview(req: HandoffRequest):
    """What WOULD cross, without recording anything.

    The declarations are the interesting part: a handoff carries the source
    session's c-numbers with their signs, and seeing that list is what tells the
    researcher whether the target session will be able to continue.
    """
    from ui.notebook_bridge import get_engine
    engine = get_engine()
    try:
        handoff = await asyncio.get_event_loop().run_in_executor(
            None, partial(engine.build_scalar_handoff, req.expr_id,
                          req.target_backend))
    except HandoffRefused as exc:
        return {"ok": False,
                "error": _t(exc.i18n_key, _lang(engine), **exc.fmt_kwargs)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "ok": True,
        "expr_id": req.expr_id,
        "target_backend": handoff.target_backend,
        "export_latex": handoff.export_latex,
        "declarations": [d.model_dump() for d in handoff.declarations],
        "undeclared": handoff.undeclared,
    }


@router.post("/handoff/create")
async def handoff_create(req: HandoffRequest):
    """Record the crossing and write the target session file."""
    from ui.notebook_bridge import get_engine
    from ui.session_io import sessions_dir
    engine = get_engine()
    loop = asyncio.get_event_loop()
    try:
        handoff = await loop.run_in_executor(
            None, partial(engine.fork_scalar_handoff, req.expr_id,
                          req.target_backend))
    except HandoffRefused as exc:
        return {"ok": False,
                "error": _t(exc.i18n_key, _lang(engine), **exc.fmt_kwargs)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        path = await loop.run_in_executor(
            None, partial(perform_handoff, handoff, sessions_dir()))
    except (ValueError, OSError, RuntimeError) as exc:
        # The crossing is already on the source timeline; say the target failed
        # rather than pretend nothing happened.
        return {"ok": False,
                "error": _t("handoff.target_failed", _lang(engine),
                            handoff_id=handoff.handoff_id,
                            reason=str(exc)[:200])}

    import pathlib
    name = pathlib.Path(path).name
    _v = handoff.verification or {}
    if _v.get("status") == "agreed":
        note = _t("handoff.verified", _lang(engine), points=_v.get("points", 0))
    elif _v.get("status") == "differed":
        note = _t("handoff.verify_differed", _lang(engine),
                  reason=_v.get("reason", ""), detail=_v.get("detail", ""))
    else:
        note = _t("handoff.verify_skipped", _lang(engine),
                  reason=_v.get("reason", ""))
    return {"ok": True, "handoff_id": handoff.handoff_id, "filename": name,
            "verification": _v, "note": note,
            "message": _t("handoff.created", _lang(engine),
                          expr_id=req.expr_id, backend=handoff.target_backend,
                          filename=name, count=len(handoff.declarations))}
