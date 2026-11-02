"""I18N-1: a backend handler's user-facing messages follow the session language.

`MaximaBackendHandler` reached for the session language as
`getattr(ctx, "lang", "ja")` on the SymbolContext — which carries no `lang`
field, so the lookup always hit the fallback and every message came out in
Japanese, English sessions included.

The language is now read through a live accessor the engine binds
(`BackendHandler.bind_language_source`).  It must be LIVE, not captured at
construction: `ui.notebook_bridge.start_engine` assigns `session.language`
only after the engine (and therefore its handler) already exists.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.i18n import DEFAULT_LANG, t

import ui.notebook_bridge as nb


class TestHandlerFollowsSessionLanguage:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_handler_lang_matches_engine(self, lang):
        nb.start_engine(title="test_i18n1", mode="standard", language=lang)
        eng = nb.get_engine()
        try:
            assert eng.lang == lang
            assert eng._handler.lang == lang
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_circular_substitution_error_is_translated(self, lang):
        """A user-facing BuildResult error, previously always Japanese."""
        nb.start_engine(title="test_i18n1_circ", mode="standard", language=lang)
        eng = nb.get_engine()
        try:
            src = eng.register_expression("q+1")
            with pytest.raises(Exception) as excinfo:
                eng.run_operation("substitute", {
                    "expr": src.display_id, "lhs": "q",
                    "rhs": "-'diff(q,t,2)/w^2"})
            expected = t("engine.circular_substitution", lang, lhs="q")
            assert str(excinfo.value) == expected
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_notation_warning_is_translated(self, caplog, lang):
        """The BP-9 derivative warning, which had to emit both languages while
        the handler could not see the session language."""
        import logging
        nb.start_engine(title="test_i18n1_warn", mode="standard", language=lang)
        eng = nb.get_engine()
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                eng._handler.latex_to_canonical(r"\frac{d^3}{d t d z} f")
            text = " ".join(r.getMessage() for r in caplog.records)
            assert t("notation.deriv_unreadable", lang).split("{")[0] in text
            other = "en" if lang == "ja" else "ja"
            assert t("notation.deriv_unreadable", other).split("{")[0] not in text
        finally:
            nb.stop_engine()


class TestLanguageIsReadLive:

    def test_language_assigned_after_construction_is_seen(self):
        """start_engine sets session.language AFTER building the engine, so a
        value snapshotted in the constructor would be stale."""
        eng = NotebookEngine(title="live")
        try:
            assert eng._handler.lang == "ja"
            eng.session.language = ["en", "ja"]
            assert eng._handler.lang == "en"
        finally:
            eng.close()

    def test_unbound_handler_falls_back_to_english(self):
        """A handler with no engine behind it is tooling, not a user session —
        English is the default for a reader we cannot identify."""
        from notebook.runtime.backend_factory import make_backend
        _adapter, _builder, handler = make_backend("maxima")
        try:
            assert handler.lang == DEFAULT_LANG == "en"
        finally:
            _adapter.close()

    def test_session_recording_no_language_also_defaults_to_english(self):
        """Same "nobody to identify" case, same answer — one constant covers
        both.  (A saved file MISSING the key never reaches this: the model's
        default_factory restores ["ja", "en"] first, which is asserted below.)"""
        eng = NotebookEngine(title="nolang")
        try:
            eng.session.language = []
            assert eng.lang == DEFAULT_LANG
            assert eng._handler.lang == DEFAULT_LANG
        finally:
            eng.close()

    def test_missing_language_key_restores_the_model_default(self):
        from notebook.models.session import Session, SessionMode
        import json
        s = Session(title="t", mode=SessionMode.standard_symbolic,
                    primary_backend="maxima")
        d = json.loads(s.model_dump_json())
        d.pop("language")
        assert Session(**d).language == ["ja", "en"]

    def test_binding_still_wins_over_the_fallback(self):
        """The fallback must never leak into a real session (a Japanese session
        must not start reporting in English)."""
        eng = NotebookEngine(title="bound")
        try:
            assert eng._handler.lang == "ja"
        finally:
            eng.close()


class TestLanguageSurvivesSaveLoad:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_loaded_session_rebinds(self, tmp_path, lang):
        """from_dict builds a fresh handler; it must be rebound too."""
        nb.start_engine(title="test_i18n1_io", mode="standard", language=lang)
        eng = nb.get_engine()
        try:
            eng.register_expression("x+1")
            path = tmp_path / "s.json"
            eng.save(path)
        finally:
            nb.stop_engine()
        loaded = NotebookEngine.load(path)
        try:
            assert loaded.lang == lang
            assert loaded._handler.lang == lang
        finally:
            loaded.close()
