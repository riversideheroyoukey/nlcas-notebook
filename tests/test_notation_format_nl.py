"""NL short-circuit for the derivative notation format (Leibniz vs dot).

"q を leibniz 記法にして" / "use leibniz notation for q" is intercepted
rule-based (ui/interpret_rules._try_rule_set_notation) before the LLM.  The
keyword detectors (notation_leibniz_kw / notation_dot_kw) are externalized to
i18n/nl_patterns.json and looked up by session language (docs/28 §3.1), so this
covers both JA and EN inputs per the bilingual-tests rule.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, text):
    tok = InputToken(type="text", value=text)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _setup_dot_derivative(eng):
    """Register q̇ (dot notation) as the single quantity in the session."""
    eng.declare_time_dependent("q")
    e1 = eng.register_expression("q(t) = q(t)")
    eng.run_operation("differentiate", {"expr": e1.display_id, "var": "t"})


def _switch_to_leibniz(lang, trigger):
    nb.start_engine(title="notfmt", mode="standard", language=lang, domains=[])
    eng = nb.get_engine()
    sup = _sup()
    try:
        _setup_dot_derivative(eng)
        r = _run(eng, sup, trigger)
        assert r["status"] == "confirmed", r
        assert r["operation"] == "set_notation"
        # every updated derivative expression now renders in Leibniz form
        assert any(r"\frac" in u["latex"] for u in r["updated_exprs"])
    finally:
        nb.stop_engine()


def test_leibniz_ja():
    _switch_to_leibniz("ja", "q を leibniz 記法にして")


def test_leibniz_en():
    _switch_to_leibniz("en", "use leibniz notation for q")


class TestQdotDiffKeyword:
    """The wrt-q̇ differentiation detector (nl_patterns.json["qdot_diff_kw"]) must
    fire on JA and EN differentiation phrasing that names a velocity token, but
    NOT on a bare \\dot{...} appearing only inside a substitution RHS."""

    def test_ja_fires(self):
        from notebook.runtime.nl_patterns import nlp
        assert nlp("qdot_diff_kw", "ja").search("q̇ で偏微分して")
        assert nlp("qdot_diff_kw", "ja").search("速度で微分して")

    def test_en_fires(self):
        from notebook.runtime.nl_patterns import nlp
        assert nlp("qdot_diff_kw", "en").search(r"differentiate E1 with respect to \dot{q}")
        assert nlp("qdot_diff_kw", "en").search(r"the derivative of E1 w.r.t. \dot{y}")

    def test_bare_dot_in_substitution_rhs_does_not_fire(self):
        from notebook.runtime.nl_patterns import nlp
        # a \dot{...} in a replacement RHS is not a differentiation request
        assert not nlp("qdot_diff_kw", "ja").search(r"E1 の a を -2\dot{r}\dot{θ}/r で置き換えて")
        assert not nlp("qdot_diff_kw", "en").search(r"in E1, replace a with -2\dot{r}\dot{theta}/r")
