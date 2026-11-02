r"""SCALAR-HANDOFF, the natural-language route (docs/34 §5.1, slice 2).

`E5 を Maxima セッションに渡して` / `hand E5 off to a Maxima session` — rule-based
and before the LLM, because which expression crosses is exactly the
researcher's choice: guessing it would defeat the design that lets the operator
check be a CHECK rather than BP-4's automatic c-number detection.

What these pin:

* the trigger fires only on an explicit ask (an ordinary compute-ask must never
  be swallowed into a session fork), in BOTH languages;
* a refusal reaches the researcher as the translated reason, not a traceback;
* the target session file is really written and really usable — the same
  end-to-end claim as the engine-level tests, made through the NL door;
* the current session is left alone (the file is written, not opened) — the
  `/fork` shape.
"""
import asyncio
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import Quantity
from ui.interpret_pipeline import (InterpretRequest, InputToken,
                                   run_interpret_pipeline)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, cmd):
    tok = InputToken(type="text", value=cmd)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _session(lang="ja"):
    nb.start_engine(title="handoff_nl", mode="standard", language=lang,
                    domains=[])
    eng = nb.get_engine()
    eng.symbol_context.register("Delta", Quantity(
        name="Delta", kind="scalar", domain="real", sign="positive"))
    eng.symbol_context.register("omega_D", Quantity(
        name="omega_D", kind="scalar", domain="real", sign="positive"))
    return eng, _sup()


class TestTheTriggerFires:
    def test_japanese(self):
        eng, sup = _session("ja")
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            r = _run(eng, sup, f"{e.display_id} を Maxima セッションに渡して")
            assert r["status"] == "completed", r
            assert r["operation"] == "fork_scalar_handoff"
            assert r["handoff_file"].endswith(".json")
        finally:
            nb.stop_engine()

    def test_english(self):
        eng, sup = _session("en")
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            r = _run(eng, sup, f"hand {e.display_id} off to a Maxima session")
            assert r["status"] == "completed", r
            assert r["handoff_file"].endswith(".json")
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("cmd", ["{e} を計算して", "{e} を展開して",
                                     "{e} を x で微分して"])
    def test_ordinary_asks_are_not_swallowed(self, cmd):
        """A session fork must never be what an ordinary compute-ask does."""
        eng, sup = _session("ja")
        try:
            e = eng.register_expression("x^2 + 1")
            r = _run(eng, sup, cmd.format(e=e.display_id))
            assert r.get("operation") != "fork_scalar_handoff", r
        finally:
            nb.stop_engine()

    def test_unknown_expression_falls_through(self):
        eng, sup = _session("ja")
        try:
            eng.register_expression("x + 1")
            r = _run(eng, sup, "E99 を Maxima セッションに渡して")
            assert r.get("operation") != "fork_scalar_handoff"
        finally:
            nb.stop_engine()


class TestRefusalReachesTheResearcher:
    @pytest.mark.parametrize("lang,phrase", [
        ("ja", "{e} を Maxima セッションに渡して"),
        ("en", "hand {e} off to a Maxima session"),
    ])
    def test_operator_bearing_expression_is_refused_in_words(self, lang, phrase):
        eng, sup = _session(lang)
        try:
            eng.symbol_context.register("c", Quantity(name="c", kind="operator"))
            e = eng.register_expression("c * Delta")
            r = _run(eng, sup, phrase.format(e=e.display_id))
            assert r["status"] == "failed", r
            err = r["error"]
            assert "c" in err
            assert "handoff." not in err          # translated, not a raw key
            assert "Traceback" not in err
        finally:
            nb.stop_engine()


class TestTheTargetIsUsableAndTheSourceUntouched:
    def test_target_file_can_continue_the_calculation(self):
        """The whole point, through the NL door: the integral whose sign
        question killed the session runs in the produced session."""
        from notebook.engine import NotebookEngine
        from ui.session_io import sessions_dir
        eng, sup = _session("ja")
        created = None
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            r = _run(eng, sup, f"{e.display_id} を Maxima セッションに渡して")
            created = sessions_dir() / r["handoff_file"]
        finally:
            nb.stop_engine()
        assert created and created.exists()
        try:
            with NotebookEngine.load(created) as eng2:
                eng2.replay_current_branch()
                _s, val = eng2.run_operation("integrate_definite", {
                    "expr": "1/sqrt(xi^2 + Delta^2)", "var": "xi",
                    "lower": "0", "upper": "omega_D"})
                assert val.replace(" ", "") == "asinh(omega_D/Delta)"
        finally:
            created.unlink(missing_ok=True)

    def test_source_session_is_left_loaded_and_recorded(self):
        """A file is written, not opened — and the crossing is on the source
        timeline, because which session a result came from is part of what it
        means."""
        from notebook.models.step import StepKind
        from ui.session_io import sessions_dir
        eng, sup = _session("ja")
        created = None
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            r = _run(eng, sup, f"{e.display_id} を Maxima セッションに渡して")
            created = sessions_dir() / r["handoff_file"]
            # still the same engine, same expression, one new step
            assert nb.get_engine() is eng
            assert e.display_id in eng.expressions
            kinds = [s.kind for s in eng.steps.values()]
            assert StepKind.scalar_handoff in kinds
        finally:
            nb.stop_engine()
            if created:
                created.unlink(missing_ok=True)
