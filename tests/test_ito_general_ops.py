"""Itô's lemma written with GENERAL OPERATIONS only (SCENARIO-MACRO, case 1).

Replaces the value assertions of the deleted `tests/test_ito_lemma.py`.  The
operation `apply_ito_lemma` is gone (open_items SCENARIO-MACRO): it called
`differentiate` three times and combined the results in a fixed linear
combination, which the researcher can type — and the formula is finance, not
something the system should own.

What is pinned here is that the REPLACEMENT route still produces the textbook
answers, in both languages:

    f = ln S  →  df = (μ − σ²/2) dt + σ dW
    g = S²    →  dg = (S²σ² + 2S²μ) dt + 2S²σ dW

Measured against the macro's own output before it was deleted: same values, the
arrangement of the display differs only because the macro ran `ratsimp`.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

DOMAIN = "financial_mathematics"


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[DOMAIN], primary_backend="maxima")


def _run(eng, sup, value, kind="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if kind == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(result):
    """The id this step actually registered — never an E-number literal, which
    would assert the author's assumption instead of the run's behaviour."""
    eid = result.get("expr_id")
    if not eid:
        regs = result.get("registered_expressions") or []
        eid = regs[-1]["expr_id"] if regs else None
    assert eid, f"step registered no expression: {result}"
    return eid


# The scenario's own commands, per language (docs/scenarios/
# scenario_ito_black_scholes_{jp,en}.md), so the two cannot drift apart.
_D_T = {"ja": "{eid} を t で偏微分して",
        "en": "take the partial derivative of {eid} with respect to t"}
_D_S = {"ja": "{eid} を S で偏微分して",
        "en": "take the partial derivative of {eid} with respect to S"}
_D_SS = {"ja": "{eid} を S で2階偏微分して",
         "en": "take the 2nd order partial derivative of {eid} with respect to S"}


def _ito(eng, sup, lang, name, body):
    """Run the scenario's five steps for one function; return the result latex."""
    r0 = _run(eng, sup, rf"{name} = {body}", "math")
    ft = _eid(_run(eng, sup, _D_T[lang].format(eid=_eid(r0))))
    fs = _eid(_run(eng, sup, _D_S[lang].format(eid=_eid(r0))))
    fss = _eid(_run(eng, sup, _D_SS[lang].format(eid=_eid(r0))))
    r = _run(eng, sup, rf"d{name} = ({ft} + \mu S {fs} + "
                       rf"\frac{{1}}{{2}}\sigma^2 S^2 {fss}) dt + \sigma S {fs} dW",
             "math")
    assert r["status"] == "registered", r
    return eng.expressions[_eid(r)].latex


class TestItoByGeneralOperations:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_log_s_gives_the_textbook_result(self, lang):
        nb.start_engine(title=f"ito_log_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            latex = _ito(eng, sup, lang, "f", r"\ln S")
            # df = (μ − σ²/2) dt + σ dW
            flat = latex.replace(" ", "")
            assert r"\mu-\frac{\sigma^2}{2}" in flat, latex
            assert "dW\\sigma" in flat or "\\sigmadW" in flat, latex
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_s_squared(self, lang):
        nb.start_engine(title=f"ito_sq_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            latex = _ito(eng, sup, lang, "g", r"S^2")
            # dg = (S²σ² + 2S²μ) dt + 2S²σ dW
            flat = latex.replace(" ", "")
            assert "S^2\\sigma^2+2S^2\\mu" in flat, latex
            assert "2S^2dW\\sigma" in flat or "2S^2\\sigmadW" in flat, latex
        finally:
            nb.stop_engine()

    def test_partial_vs_total_is_asked_not_guessed(self):
        """`differentiate E1 by t` on f(S,t) is genuinely ambiguous — S depends
        on t — so the system refuses rather than picking one.  This friction is
        why the scenario says 「偏微分して」 explicitly."""
        nb.start_engine(title="ito_ambig", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            r0 = _run(eng, sup, r"f = \ln S", "math")
            r = _run(eng, sup, f"{_eid(r0)} を t で微分して")
            assert r["status"] == "failed", r
            assert "偏微分" in r["error"] and "全微分" in r["error"], r["error"]
        finally:
            nb.stop_engine()

    def test_apply_ito_lemma_is_gone(self):
        """The macro must not come back without a decision: it is neither an
        engine method nor a schema operation (open_items SCENARIO-MACRO)."""
        from notebook.engine import NotebookEngine
        assert not hasattr(NotebookEngine, "apply_ito_lemma")
        v = OperationValidator()
        assert "apply_ito_lemma" not in v.known_operations
