"""Minimal i18n helper for the Python backend.

Usage:
    from notebook.runtime.i18n import t
    msg = t("engine.register", lang, display_id="E1")

Keys and translations live in i18n/messages.json.
Fallback chain: requested lang → "en" → raw key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MESSAGES_PATH = Path(__file__).resolve().parent.parent.parent / "i18n" / "messages.json"
_cache: dict | None = None

# The language to use when there is no reader to identify — no session, or a
# session that records none.  English, for the same reason this module already
# falls back to "en" for a missing translation: a default of Japanese would
# re-assert the very assumption the i18n layer exists to remove.
DEFAULT_LANG = "en"


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(_MESSAGES_PATH.read_text(encoding="utf-8"))
    return _cache


def t(key: str, lang: str = DEFAULT_LANG, **kwargs: Any) -> str:
    """Return the localised string for *key* in *lang*, formatted with **kwargs**.

    Falls back to "en" if *lang* is not present, then to the raw key string.
    """
    table = _load()
    entry = table.get(key, {})
    template = entry.get(lang) or entry.get("en") or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    return template


def reload() -> None:
    """Force reload of the messages file (useful in tests or dev)."""
    global _cache
    _cache = None
