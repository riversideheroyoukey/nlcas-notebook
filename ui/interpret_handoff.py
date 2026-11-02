"""SCALAR-HANDOFF: the natural-language route to hand an expression across.

Rule-based, before the LLM.  The command names an expression and a target
backend and nothing else is inferred — which expression crosses is exactly the
researcher's choice (that is why the operator check can be a check rather than
BP-4's automatic c-number detection), so guessing it would defeat the design.

The new session is written as a FILE and not opened, matching `/fork`: the
current session stays loaded and untouched, and the researcher decides when to
switch.  Opening it automatically would end the source session's context
mid-calculation, which is exactly the moment they are least likely to want it.

Reference: docs/developer_manuals/34 §5-5.1.
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.scalar_handoff import HandoffRefused, perform_handoff

_BACKEND_ALIAS = {"maxima": "maxima", "julia": "julia_full"}


async def _try_scalar_handoff(engine, loop, lang, text, registered) -> "dict | None":
    """`E5 を Maxima セッションに渡して` / `hand E5 off to a Maxima session`."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    m = nlp("scalar_handoff_nl", lang).match(stripped)
    if m is None:
        return None

    target = (m.group("target") or "").strip()
    if target not in engine.expressions or target not in engine._active_expr_ids():
        return None

    raw_backend = (m.groupdict().get("backend") or "").strip().lower()
    backend = _BACKEND_ALIAS.get(raw_backend, "maxima")

    try:
        handoff = await loop.run_in_executor(
            None, partial(engine.fork_scalar_handoff, target, backend))
    except HandoffRefused as _exc:
        return {"status": "failed",
                "error": t(_exc.i18n_key, lang, **_exc.fmt_kwargs),
                "registered_expressions": registered}
    except ValueError as _exc:
        return {"status": "failed", "error": str(_exc),
                "registered_expressions": registered}

    from ui.session_io import sessions_dir
    try:
        path = await loop.run_in_executor(
            None, partial(perform_handoff, handoff, sessions_dir()))
    except (ValueError, OSError, RuntimeError) as _exc:
        # The descriptor and its step already exist; only the target session
        # failed to build.  Say so rather than pretending the handoff did not
        # happen — the source timeline now shows a crossing that has no file.
        # Narrow deliberately: a CAS refusal is a ValueError, writing the file
        # an OSError, a dead adapter a RuntimeError.  Anything else is a bug
        # here and must surface as one rather than be reported as the
        # researcher's handoff failing.
        return {"status": "failed",
                "error": t("handoff.target_failed", lang,
                           handoff_id=handoff.handoff_id, reason=str(_exc)[:200]),
                "registered_expressions": registered}

    import pathlib
    name = pathlib.Path(path).name
    # Say what was verified — and, just as plainly, what was not.  An
    # unverified crossing reported as a verified one is the failure this whole
    # feature is trying not to become.
    _v = handoff.verification or {}
    if _v.get("status") == "agreed":
        _note = t("handoff.verified", lang, points=_v.get("points", 0))
    elif _v.get("status") == "differed":
        _note = t("handoff.verify_differed", lang, reason=_v.get("reason", ""),
                  detail=_v.get("detail", ""))
    else:
        _note = t("handoff.verify_skipped", lang, reason=_v.get("reason", ""))
    return {"status": "completed",
            "operation": "fork_scalar_handoff",
            "verification": _v,
            "message": t("handoff.created", lang, expr_id=target,
                         backend=backend, filename=name,
                         count=len(handoff.declarations)) + " " + _note,
            "handoff_id": handoff.handoff_id,
            "handoff_file": name,
            "undeclared": handoff.undeclared,
            "registered_expressions": registered}
