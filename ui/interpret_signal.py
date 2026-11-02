"""F-8 (signal_processing / control_theory): rule-based NL triggers for the
definition-based Fourier/inverse-Fourier/Z transforms.

Mirrors ``_try_laplace_definition`` (ui/interpret_free_params.py) exactly —
these are engine methods (notebook/engine_fourier_mixin.py), not
schema/run_operation-driven ops, so each needs its own interceptor.
``convolve`` is an ordinary generic alias-driven operation (schema + backend
template only) and needs no dedicated interceptor here.
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.nl_patterns import nlp
from ui.interpret_common import _expand_expr_ids


async def _try_z_transform_by_definition(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("z_transform_by_definition", lang).match(text)
    if m is None:
        return None
    target = _expand_expr_ids(m.group("target").strip(), engine)
    n = m.group("nvar") or "n"
    z = m.group("zvar") or "z"
    try:
        ex = await loop.run_in_executor(
            None, partial(engine.z_transform_by_definition, target, n, z))
    except Exception as _e:
        return {"status": "failed", "error": str(_e), "registered_expressions": registered}
    registered.append({"expr_id": ex.display_id, "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": "z_transform_by_definition",
            "expr_id": ex.display_id, "latex": ex.latex, "canonical": ex.canonical,
            "registered_expressions": registered}
