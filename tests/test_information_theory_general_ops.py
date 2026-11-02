"""Information-theoretic quantities written as ORDINARY SUMS (SCENARIO-MACRO, case 2).

Replaces `tests/test_information_theory.py`, deleted with the three macros it
tested (`shannon_entropy` / `kl_divergence` / `mutual_information`, open_items
SCENARIO-MACRO).  Each was one `sum(...)` template, and none of them had a
natural-language trigger — the scenario reached them through the generator's
`@op:` backdoor, so a researcher could not.

The value assertions move here, stated against the route a researcher actually
takes.  One of them matters more than the rest: the old
`test_kl_divergence_precedence_regression` recorded that an unparenthesised
`{p}/{q}` in the template made D(1/3‖1/2) evaluate to `-log(6)` instead of
`log(2/3)`.  **That defect was fixed inside the template and nowhere else**, so
the researcher's own spelling `\\log\\frac{p}{q}` still carried it until
2026-08-14 — the macro had met the bug, fixed it privately, and left the general
path broken.  The converter fix is pinned in
`tests/test_latex_converter.py::TestFractionIsOneOperand`; what is pinned here is
the VALUE, through the researcher's spelling.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

DOMAIN = "information_theory"


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[DOMAIN], primary_backend="maxima")


def _run(eng, sup, value, kind="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if kind == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(result):
    """The id this step actually registered — never an E-number literal."""
    eid = result.get("expr_id")
    if not eid:
        regs = result.get("registered_expressions") or []
        eid = regs[-1]["expr_id"] if regs else None
    assert eid, f"step registered no expression: {result}"
    return eid


_EVAL_RHS = {"ja": "{eid} の右辺を計算して",
             "en": "compute the right-hand side of {eid}"}


def _value_of(eng, sup, lang, latex):
    """Register `name = <sum>` and evaluate its right-hand side."""
    r = _run(eng, sup, latex, "math")
    assert r["status"] == "registered", r
    out = _run(eng, sup, _EVAL_RHS[lang].format(eid=_eid(r)))
    assert out["status"] == "completed", out
    return eng.expressions[_eid(out)].canonical.strip()


class TestSumsEvaluateToTheTextbookValues:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_entropy_of_a_fair_die(self, lang):
        """Written as its own definition, this returned **0** until the
        `\\frac`-as-one-operand fix — `\\log \\frac{1}{6}` meant `log(1)/6`."""
        nb.start_engine(title=f"it_die_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            got = _value_of(eng, sup, lang,
                            r"H = -\sum_{x=1}^{6} \frac{1}{6}\log \frac{1}{6}")
            assert got == "log(6)", got
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_entropy_of_a_fair_die_in_bits(self, lang):
        """The unit information theory actually uses.

        `\\log_2 6` is ≈ 2.585 bits.  Until 2026-08-16 the based spelling
        became `log(2) * 6` — the logarithm OF 2, times 6, ≈ 4.16 — which is
        valid Maxima and therefore raised nothing at all.  The scenario was
        written in nats *because of that*, not by choice.
        """
        nb.start_engine(title=f"it_die_bits_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            got = _value_of(eng, sup, lang,
                            r"H = -\sum_{x=1}^{6} \frac{1}{6}\log_2 \frac{1}{6}")
            assert got.replace(" ", "") == "log(6)/log(2)", got
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_kl_divergence_keeps_its_precedence(self, lang):
        """The old template-level regression, restated as a value through the
        researcher's own spelling: D(1/3‖1/2) over three outcomes is log(2/3),
        and the unparenthesised reading gave -log(6)."""
        nb.start_engine(title=f"it_kl_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            got = _value_of(eng, sup, lang,
                            r"D = \sum_{x=1}^{3}\frac{1}{3}\log\frac{1/3}{1/2}")
            assert got == "log(2/3)", got
            assert got != "-log(6)", "the unparenthesised reading is back"
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_independence_gives_zero_mutual_information(self, lang):
        nb.start_engine(title=f"it_mi_{lang}", mode="standard",
                        language=lang, domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            got = _value_of(eng, sup, lang,
                            r"I_0 = \sum_{x=1}^{2}\sum_{y=1}^{2} p_x(x) p_y(y)"
                            r"\log \frac{p_x(x) p_y(y)}{p_x(x) p_y(y)}")
            assert got == "0", got
        finally:
            nb.stop_engine()


class TestGeneralOperationsCompose:
    """What the macros could not do: the quantities meet `differentiate` and
    `substitute` as ordinary expressions."""

    def test_bernoulli_entropy_derivative(self):
        nb.start_engine(title="it_bern", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup, r"H_b = -p\log p - (1-p)\log(1-p)", "math")
            d = _run(eng, sup, f"{_eid(r)} を p で微分して")
            assert d["status"] == "completed", d
            canon = eng.expressions[_eid(d)].canonical.replace(" ", "")
            assert canon == "log(1-p)-log(p)", canon
        finally:
            nb.stop_engine()

    def test_kl_of_a_distribution_with_itself_is_zero(self):
        nb.start_engine(title="it_kl0", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup,
                     r"D = p\log \frac{p}{q} + (1-p)\log \frac{1-p}{1-q}", "math")
            s = _run(eng, sup, f"{_eid(r)} の q に p を代入して")
            assert s["status"] == "completed", s
            assert eng.expressions[_eid(s)].canonical.strip() == "0"
        finally:
            nb.stop_engine()


def test_the_three_macros_are_gone():
    """They must not come back without a decision (open_items SCENARIO-MACRO)."""
    v = OperationValidator()
    for op in ("shannon_entropy", "kl_divergence", "mutual_information"):
        assert op not in v.known_operations, op
