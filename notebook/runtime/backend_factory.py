"""Factory for creating CAS backend components from a backend name string.

Centralises all backend-specific imports so that ``engine.py`` depends only
on the ``BackendHandler`` ABC and this factory — no concrete adapter or
handler class leaks into the engine.
"""
from __future__ import annotations

from notebook.runtime.backend_handler import BackendHandler
from notebook.runtime.command_builder import CommandBuilder

#: Registry of known backend names. Extend here when adding a new backend.
_KNOWN_BACKENDS = {"maxima", "julia_full"}

#: Which backend a session mode runs on when the caller does not say.
#: Lives HERE, with the name registry, and not in ``engine.py``: this module is
#: the single owner of the backend-name namespace, so adding a CAS must not
#: require editing the engine (open_items AUDIT-SHARED-CAS-LEAK).
_DEFAULT_BACKEND_FOR_MODE: dict[str, str] = {
    "standard_symbolic": "maxima",
    "second_quantization": "julia_full",
}


def session_mode_for_backend(backend: str) -> "str | None":
    """Inverse of :func:`default_backend_for_mode`: the session mode whose
    default backend is *backend* (as a ``SessionMode`` value string), or ``None``.

    Used when the mode has to be inferred from configuration that names a
    backend — e.g. a domain category's ``default_backend`` in
    ``domains/categories.json``.  Callers ask here instead of comparing against
    a backend name themselves, so the two directions cannot drift apart and a
    third CAS needs no edit outside this module.
    """
    for mode, name in _DEFAULT_BACKEND_FOR_MODE.items():
        if name == backend:
            return mode
    return None


def default_backend_for_mode(mode) -> str:
    """Backend name for *mode* — a ``SessionMode`` or its ``.value`` string.

    Raises ``ValueError`` for a mode with no entry, rather than falling back to
    a backend.  A silent fallback was the real hazard here: a NEW SessionMode
    whose entry someone forgot would quietly start a Maxima session, and since
    the backend cannot be switched mid-session (Core Principle #6) the
    researcher would have to start over — with no error anywhere.  Adding a mode
    must state which CAS it runs on.
    """
    key = getattr(mode, "value", mode)
    try:
        return _DEFAULT_BACKEND_FOR_MODE[key]
    except KeyError:
        raise ValueError(
            f"No default backend for session mode {key!r}. Add an entry to "
            f"backend_factory._DEFAULT_BACKEND_FOR_MODE (known modes: "
            f"{sorted(_DEFAULT_BACKEND_FOR_MODE)})."
        ) from None


def make_backend(primary_backend: str) -> tuple[object, CommandBuilder, BackendHandler]:
    """Return *(adapter, builder, handler)* for the given backend name.

    Raises ``ValueError`` for unknown backend strings so callers fail fast
    rather than silently falling through to a wrong backend.

    The adapter type is untyped here to avoid importing both adapter classes
    at the module level; callers assign it to ``self._adapter`` which is
    typed as the common protocol (``start / stop / run``).
    """
    if primary_backend == "julia_full":
        from notebook.runtime.julia_adapter import JuliaAdapter
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.julia_command_builder import JuliaCommandBuilder
        builder = JuliaCommandBuilder()
        return JuliaAdapter(), builder, JuliaBackendHandler(builder)
    if primary_backend == "maxima":
        from notebook.runtime.maxima_adapter import MaximaAdapter
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.maxima_command_builder import MaximaCommandBuilder
        builder = MaximaCommandBuilder()
        return MaximaAdapter(), builder, MaximaBackendHandler(builder)
    raise ValueError(
        f"Unknown backend {primary_backend!r}. Known backends: {sorted(_KNOWN_BACKENDS)}"
    )
