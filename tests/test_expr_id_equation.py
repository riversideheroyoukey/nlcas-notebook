"""`E_n = X` forms the explicit equation <value of E_n> = X.

When the LHS of "=" is an existing expression-ID, the system expands E_n to its
value and registers the equation so the value is shown and can be solved / used
as a constraint.  Surfaced in S-n: the user wanted to set the reactance
Im Z = ωL−1/(ωC) to 0 (resonance) and have "... = 0" displayed explicitly,
rather than registering a literal "E3 = 0" with E3 kept symbolic.
"""
import asyncio

from ui import notebook_bridge as nb
from ui.interpret_pipeline import run_interpret_pipeline
from ui.notebook_bridge import InterpretRequest, InputToken
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(domains, setup_canonicals, command):
    nb.start_engine(title="eqid", mode="standard", language="ja", domains=domains)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=domains, primary_backend="maxima")
        for c in setup_canonicals:
            eng.register_expression(c)
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=command)]),
                eng, sup))
        out = {eid: ex.canonical for eid, ex in eng.expressions.items()}
        return r, out
    finally:
        nb.stop_engine()


def test_expr_id_equals_zero_forms_explicit_equation():
    # E1 is the reactance; "E1 = 0" must give the explicit "value = 0" equation,
    # not a literal "E1 = 0".
    r, out = _run([], ["L*w - 1/(C*w)"], "E1 = 0")
    assert r["status"] == "registered"
    eqs = [c for c in out.values() if c.replace(" ", "") == "L*w-1/(C*w)=0"]
    assert eqs, f"expected 'L*w-1/(C*w) = 0' among {list(out.values())}"
    # and it must NOT have stored the literal "E1 = 0"
    assert "E1 = 0" not in out.values()


def test_expr_id_equals_nonzero_value():
    r, out = _run([], ["a*x^2 + b"], "E1 = c")
    assert r["status"] == "registered"
    assert any(c.replace(" ", "") == "a*x^2+b=c" for c in out.values())


def test_bare_eid_equation_not_intercepted_when_eid_absent():
    # No E5 exists → the intercept must not fire; falls through to normal handling.
    r, out = _run([], ["x + 1"], "E5 = 0")
    # E5 is absent, so this is NOT our explicit-equation case; it should register
    # the literal equation rather than raising.
    assert any(c.replace(" ", "") == "E5=0" for c in out.values())


def test_a_bounded_sum_on_the_left_keeps_its_head():
    r"""The value of E_n is the RHS of the EQUATION, not of the first `=`.

    ONE-QUESTION-MANY-ANSWERS ratchet (`ui/interpret_pipeline.py` 3→0).  This
    site registers what it splits, so a mis-split is persisted rather than used
    once in a command.  Measured 2026-08-13: with E1 stored as
    `\sum_{i=1}^{3} c_i = u a + v b`, typing `E1 = 0` registered

        1}^3*c_i = u*a + v*b = 0

    — the subscript read as the equation, and two `=` in a canonical that goes
    to disk.  Nothing failed.
    """
    source = r"\sum_{i=1}^{3} c_i = u a + v b"
    r, out = _run([], [source], "E1 = 0")
    assert r["status"] == "registered"
    # Only the NEW expression is asserted: the source legitimately contains
    # `1}^`, so a check over every canonical fires on the right answer.  That is
    # the fourth sentinel in this ratchet's tests to have been wrong that way.
    # `_run` registers the setup string directly, so the canonical keeps the
    # researcher's spacing (`u a + v b`) rather than the pipeline's `u*a + v*b`.
    registered_now = [c for eid, c in out.items() if c != source]
    assert registered_now == ["u a + v b = 0"], out
