# i18n Design Notes

**Date:** 2026-05-16

---

## Overview

The system supports multiple display languages (currently `en` and `ja`).
The active language is stored per-session and drives both backend messages and
the browser UI.

---

## Architecture

### Single source of truth

`config/default_settings.json` holds the application-wide default language:

```json
{ "language": "en" }
```

`config/config_utils.get_default_language()` reads this value at call time
(no cache), so changing the file is effective immediately on the next request.

### 言語コードは `ja`。`jp` は言語コードではない（2026-09-05）

言語コードは **`ja` / `en`**（ISO 639-1）で、`i18n/*.json` の全キー・
`session.language`・`lang` 引数・`t(key, lang)`・`nlp(key, lang)`・
`DEFAULT_LANG` はすべてこの綴りである。

`jp` はリポジトリ内に4か所あるが、**言語コードとして正しいものは1つも無い**:

| 綴り | 何を指すか | どこに |
|---|---|---|
| `jp` | **成果物のファイル名接尾辞**（`scenario_x_jp.tex` / `README_jp.md`） | `scripts/generate_scenario_examples.py` は `language`（`"ja"`）と `suffix`（`"jp"`）を**別の引数として**受ける |
| `jp` | 上の接尾辞が**そのままコーパス行の欄名**になったもの | `scripts/build_instruction_corpus.py` がシナリオ文書の `jp:` / `en:` ブロックを読むため。`tests/corpus/instruction_corpus.json` の行は `{"jp": …, "en": …}` |
| `jp` | **量子数 j′**（言語と無関係の同綴り） | `ui/interpret_angular.py` の `_pick(a, "jp")`（Wigner–Eckart） |
| ~~`jp`~~ | **正規表現の名前付きグループ**を言語名で綴った唯一の場所 | `f1_lattice_mention` — 2026-09-05 に `ja` へ揃えた |

**なぜ書き残すか:** コーパス行の欄が `jp` であることを知らずに `ja` で読むと、
**日本語の語がすべて「0行」と出て、英語しか候補に上がらない**。`nl_worklist.py`
を書き直したときに実際にそれで一度「日本語の割れは無い」と読みかけた
（`scripts/nl_worklist.py` の docstring に測定として残してある）。

### Backend (`notebook/runtime/i18n.py`)

```python
t(key, lang, **kwargs)  # fallback chain: lang → "en" → raw key
```

Message table: `i18n/messages.json`

`engine.lang` returns `session.language[0]`, which is set at session creation.
All backend strings (supervisor, adviser, engine visible actions) go through
`t()`.

### Default language (`i18n.DEFAULT_LANG`, 2026-07-28)

```python
DEFAULT_LANG = "en"   # notebook/runtime/i18n.py
```

**One constant answers "which language, when there is no reader to identify?"**
Two places ask it, and both must give the same answer:

| Caller | When it fires |
|---|---|
| `NotebookEngine.lang` | `session.language` is empty |
| `BackendHandler.lang` | the handler is UNBOUND — no engine behind it (a unit test, a script, tooling) |

It is **English**, matching what `t()` already did on its own (`lang → "en" →
raw key`, and `t`'s own default argument).  A default of Japanese would
re-assert, in miniature, the assumption this whole layer exists to remove: the
reader is unknown, not Japanese.

Two things this does **not** affect, both worth stating because both were
raised as objections and neither holds:

1. **Saved sessions missing the `language` key never reach the fallback.**
   `Session.language`'s `default_factory` restores `["ja", "en"]` first, so an
   old file still reports `ja`.  Backward compatibility is carried by the model
   default, not by this constant.
2. **Replay cannot be disturbed by it.** Replay re-executes saved
   `rendered_command`s and restores canonicals from the saved `Expression`s —
   natural language is never re-interpreted (Core Principle #8,
   `notebook/runtime/replay.py`).  No language-driven path (`nlp()` pattern
   selection, `latex_to_canonical`) runs during a replay at all, so the session
   language cannot change a replayed result.  Equivalently: a step that was
   never taken leaves no record, and replay only re-runs what is recorded.

### Reaching the session language from a backend handler (I18N-1, 2026-07-28)

A `BackendHandler` raises user-facing text of its own (a circular substitution,
an unreadable notation).  It has no session reference, and for a long time it
guessed with `getattr(ctx, "lang", "ja")` on the `SymbolContext` — which has no
`lang` field, so **every one of those messages came out in Japanese**, English
sessions included.

The fix is a bound accessor, not a field:

```python
handler.bind_language_source(lambda: engine.lang)   # engine.__init__ and from_dict
handler.lang                                        # live read, DEFAULT_LANG when unbound
```

* **Not a field on `SymbolContext`.**  Language is a property of the session;
  `SymbolContext` is persisted, so a copy there would be a second source of
  truth that a saved file could restore stale.
* **Live, not snapshotted.**  `start_engine` assigns `session.language` *after*
  the engine — and therefore its handler — already exists, so a value captured
  in the constructor is always stale.
* Bound at the two points that already wire the symbol context, so a loaded
  session (`from_dict`, which builds a fresh handler) is rebound too.

Tests: `tests/test_handler_session_language.py`.

### Session creation (`ui/notebook_bridge.py`)

```python
class NewSessionRequest(BaseModel):
    language: str = Field(default_factory=get_default_language)

def start_engine(..., language: str | None = None):
    if language is None:
        language = get_default_language()
    session.language = [language, "en"] if language != "en" else ["en", "ja"]
```

Input aliases (operation names) are always accepted in **all** languages
regardless of session language — `_build_alias_index()` in `OperationValidator`
merges all language aliases into one index.

### Frontend (`ui/notebook.html` + `ui/notebook_main.js`)

**API endpoint:** `GET /api/notebook/ui-lang`

Returns `{ lang, messages }` where `lang` comes from `engine.lang` when a
session is running, or `get_default_language()` before a session starts.
`messages` is the full `i18n/ui_messages.json` table (file-level cache).

**Initialisation (inline script in `notebook.html`):**

```javascript
fetch('/api/notebook/ui-lang')
    .then(data => {
        _msgs = data.messages;
        _lang = data.lang;
        window._lang = _lang;
        applyI18n();          // walks [data-i18n] and [data-i18n-ph] elements
    });

window._reloadI18n = _fetchAndApply;   // callable from notebook_main.js
```

`window._t(key, vars)` performs `{placeholder}` substitution using the live
`_msgs` and `_lang` closure variables.

**Language change on new session:**

After `POST /api/notebook/session/new` succeeds, `notebook_main.js` calls
`await window._reloadI18n()`.  This re-fetches `/api/notebook/ui-lang` (which
now reflects the new session's language) and re-applies i18n to the entire
page, so header buttons and all dynamic strings switch atomically.

**Language selector:**

The "+ New" modal contains a `<select id="new-session-language">`.
`openNewSessionModal()` pre-selects the current `window._lang`.
The chosen value is sent as `language` in the session creation request.

---

## Known Limitation: Pre-fetch Fallback

`notebook.html` contains two hardcoded values that are only in effect for the
fraction of a second before the `/api/notebook/ui-lang` response arrives:

```javascript
let _lang = 'ja';
window._lang = 'ja';
```

These cannot be made config-driven without switching from static HTML to
server-side rendering (e.g. Jinja2 templates served by FastAPI).  For a
local-server application the API response is effectively instantaneous, so
this transient state is invisible in practice.

**Decision:** keep `notebook.html` as a static file.  The pre-fetch fallback
will remain as-is.  Changing `default_settings.json` requires no edits to
any source file other than the config itself.

---

## Checklist for Adding a New Language

1. Add every key to `i18n/messages.json` with the new ISO 639-1 code.
2. Add every key to `i18n/ui_messages.json` with the new code.
3. Add a `<option value="xx">…</option>` to the `<select>` in `notebook.html`.
4. Update `start_engine()` in `notebook_bridge.py` if the new language should
   be paired with a different fallback (currently all non-`en` languages fall
   back to `en`).
5. Update `domain/*/meta.json` and `domains/categories.json` with
   `label_xx` fields.
