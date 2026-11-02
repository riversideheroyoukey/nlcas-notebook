"""BP-8: plain-text registration of an equation containing 2+ `diff(...)` calls.

The pure-formula path in `ui/interpret_pipeline.py` (§2c) defers to the LLM
supervisor when the input looks like a natural-language instruction, measured
as "2 or more English words of 3+ letters that are not known math functions".
A function CALL head is not an English word, but the count did not know that:
`diff(V,t) + diff(V,S,2) = 0` scored 2 on its two `diff` calls, lost the
pure-formula path, and failed with "Could not interpret" — while the same
equation wrapped as a `$…$` math chip registered fine (the BP-8 workaround).

These tests pin both directions: any callable is excluded from the NL-word
count, and a genuine NL instruction is still handed to the supervisor.
"""
import asyncio

import pytest

from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.llm_supervisor import LLMSupervisor

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


BLACK_SCHOLES = ("diff(V,t) + (1/2)*sigma^2*S^2*diff(V,S,2) "
                 "+ r*S*diff(V,S) - r*V = 0")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _session(lang):
    nb.start_engine(title="test_bp8", mode="standard", language=lang,
                    domains=["financial_mathematics"])
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["financial_mathematics"],
                        primary_backend="maxima")
    return nb.get_engine(), sup


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("text", [
    "diff(V,t) + diff(V,S,2) = 0",       # minimal repro: two bare diff calls
    "diff(V,t) + diff(V,S) = 0",
    BLACK_SCHOLES,                        # the equation BP-8 was found on
])
def test_multi_diff_equation_registers_as_plain_text(lang, text):
    eng, sup = _session(lang)
    try:
        res = _run(eng, sup, text)
        assert res.get("status") == "registered", res
        regs = res["registered_expressions"]
        assert len(regs) == 1
        assert regs[0]["canonical"] == text
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_plain_text_matches_math_chip_workaround(lang):
    """The plain-text path must produce what the `$…$` math chip produced —
    same canonical, same latex, same LHS alias.

    The alias is compared by what it RESOLVES: since 2026-08-09 an LHS is
    registered under both the spelling as written and its canonical, because
    the map has consumers on both sides of that line (`_expand_symbol_refs`
    scans LaTeX, `_auto_detect_substitutions` scans canonicals).  The two
    routes genuinely receive different strings — the math chip sends
    `\\sigma`, the text box sends `sigma` — so the extra spelling the chip
    route also answers to is not a difference in behaviour.  What must match
    is that both routes make the equation reachable under the canonical
    spelling, and that they agree on the expression itself.
    """
    out = {}
    for ttype in ("math", "text"):
        eng, sup = _session(lang)
        try:
            res = _run(eng, sup, BLACK_SCHOLES, ttype)
            assert res.get("status") == "registered", (ttype, res)
            reg = res["registered_expressions"][0]
            aliases = dict(eng.lhs_alias_map)
            canonical_lhs = reg["canonical"].split(" = ")[0].strip()
            out[ttype] = (reg["canonical"], reg["latex"],
                          aliases.get(canonical_lhs))
        finally:
            nb.stop_engine()
    assert out["text"] == out["math"]
    assert out["text"][2] is not None, "the canonical LHS must be a usable key"


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_single_diff_equation_still_registers(lang):
    """The one-`diff` case always worked; guard against regressing it."""
    eng, sup = _session(lang)
    try:
        res = _run(eng, sup, "diff(V,t) + V = 0")
        assert res.get("status") == "registered", res
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_english_instruction_still_defers_to_supervisor(lang):
    """The NL guard itself must survive: an English instruction is NOT a
    formula, even though it contains a call-shaped `a(1)`.  With no LLM client
    configured the supervisor fails — which is exactly the evidence that the
    text was routed to it rather than registered."""
    eng, sup = _session(lang)
    try:
        res = _run(eng, sup, "define a(1) as a bose operator")
        assert res.get("status") != "registered", res
    finally:
        nb.stop_engine()
