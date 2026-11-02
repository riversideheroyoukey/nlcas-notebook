"""Typed-numeric-graph helpers (docs/26): source hash + binding compatibility.

The graph itself is never stored — it is DERIVED from refs (expression→task
via ``source_expression_ids``, artifact→task via ``input_artifact_refs``,
task→artifact via ``artifact.task_id``; docs/26 §6).  This module holds the
pure pieces G1 needs: the fork-time content hash for stale detection and the
small closed table of which artifact value_kinds may bind into which input
roles.  Engine-state-dependent logic (binding resolution, the stale check
itself) lives in ``EngineNumericMixin``.
"""
from __future__ import annotations

import hashlib

# Which artifact value_kinds may bind into which InputVar roles (docs/26 §5).
# Deliberately tiny for G1: only a scalar value can become a fixed value or
# an iteration's initial value.  Extending this table IS how a new binding
# becomes legal (data, not schema) — e.g. G2's loop may allow "trace" nowhere
# but let a vector bind into a parameter_set-style input.
BINDABLE_VALUE_KINDS: dict[str, frozenset[str]] = {
    "fixed": frozenset({"scalar"}),
    "free": frozenset({"scalar"}),
}


def can_bind(role: str, value_kind: str) -> bool:
    """True if an artifact of *value_kind* may bind into an input of *role*."""
    return value_kind in BINDABLE_VALUE_KINDS.get(role, frozenset())


def source_hash(canonical: str, assumption_canonicals: "list[str]") -> str:
    """Content hash of a fork's frozen source (docs/26 §7).

    Whitespace-normalized canonical + the SORTED assumption canonicals, so the
    hash is insensitive to Maxima's pretty-print line wrapping and to
    assumption insertion order, but changes when the expression or any
    assumption actually changes.  Prefixed with the algorithm so a future
    migration can tell hash generations apart (same reasoning as R-1's
    versioned loads).
    """
    parts = [" ".join((canonical or "").split())]
    parts += sorted(" ".join((a or "").split()) for a in assumption_canonicals)
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
