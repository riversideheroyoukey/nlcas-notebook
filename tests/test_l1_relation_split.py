r"""L-1's `=` splits are code generation, not display.

ONE-QUESTION-MANY-ANSWERS ratchet, second lowering (`ui/interpret_l1.py` 9→0).
Counting where the nine values went — the step docs/handoffs §4 requires before
any unification — put **eight of nine into a CAS argument**:

* `_infer_old_basis`  → `build_inverse_transform(vars=…)`
* `_new_basis_symbols`→ `build_inverse_transform(new_basis=…)`, which the command
  unbinds block-locally inside Maxima
* `_subst_lhs_rhs`    → `substitute(lhs, rhs)` / `operator_substitute`'s op_rules,
  whose `op_name` / `formal_index` are read off that LHS

Only `_forward_has_operators` (which of two prompts to show) is a display.

Unlike engine.py's seven — where no reachable input separated the naive split
from the bracket-aware one — this file had a **live** one, measured 2026-08-13
through the real `E1, E2 を E3 に代入して` route.  With a forward equation whose
LEFT side carries a bounded sum, Maxima was sent

    ratsimp(subst(sum(c(i), i, 1, 3)=1}^3*c_i = u*a + v*b, c(1) + d^2))

— the replacement is a fragment with the head gone and an `=` still in it.

**Why these tests sit at the helper level.** The route cannot be driven to a
green end state for this shape, for two reasons that have nothing to do with the
split (both measured the same day, both pre-existing):

1. With ordinary named sources, H-1's P2 binding has already expanded the second
   source inside Maxima, so the chain's second `substitute` reports "not in the
   target" — a failure for work that was already done.
2. With compound sources (no P2 binding), the first substitution leaves the CAS's
   own spelling of the second sum, and the guard compares it against the
   researcher's spelling, so it again reports "not in the target".

Asserting on a green end state would therefore mean asserting on those defects.
The helpers are the unit that changed; the rendered command below is the closest
researcher-facing artefact that the defect actually reached.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_l1 import (_forward_has_operators, _new_basis_symbols,
                             _subst_lhs_rhs)
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline

SUM_TRANSFORM = r"\sum_{i=1}^{3} c_i = u a + v b"


def _run(engine, supervisor, value, ttype="text"):
    token = (InputToken(type="math", value=value, latex=value, canonical="")
             if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[token]), engine, supervisor))


@pytest.fixture
def session():
    nb.start_engine(title="test_l1_split", mode="standard", language="ja",
                    domains=[])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=[], primary_backend="maxima")
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


class TestABoundedSumOnTheLeftKeepsItsHead:
    def test_substitution_rule_is_the_whole_equation(self, session):
        engine, supervisor = session
        assert _run(engine, supervisor, SUM_TRANSFORM,
                    "math").get("status") == "registered"
        lhs, rhs = _subst_lhs_rhs(engine, "E1")
        assert lhs == r"\sum_{i=1}^3*c_i", lhs
        assert rhs == "u*a + v*b", rhs
        assert "=" not in rhs          # the head-loss signature: `1}^3*c_i = …`

    def test_new_basis_is_the_whole_left_side(self, session):
        """This string is unbound inside a Maxima `block([...])`."""
        engine, supervisor = session
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        assert _new_basis_symbols(engine, ["E1"]) == [r"\sum_{i=1}^3*c_i"]

    def test_the_command_maxima_receives_carries_no_fragment(self, session):
        """The closest researcher-facing artefact the defect reached.

        Only the FIRST rule's command is asserted: the chain's later steps stop
        on the two pre-existing defects in this module's docstring.
        """
        engine, supervisor = session
        for source in (SUM_TRANSFORM, "d = w a", r"H = \sum_{i=1}^{3} c_i + d"):
            _run(engine, supervisor, source, "math")
        _run(engine, supervisor, "E1, E2 を E3 に代入して")
        commands = [s.rendered_command for s in engine.steps.values()
                    if s.rendered_command and "subst(" in s.rendered_command]
        assert commands, "no substitution reached the CAS at all"
        first = commands[0]
        assert "1}^" not in first, first
        assert "sum(c(i), i, 1, 3)=u * a + v * b" in first, first


class TestTheDeliberatelyKeptBehaviours:
    def test_an_inequality_still_yields_its_right_hand_side(self, session):
        r"""`"=" in canon` also fired on `<=`, and an inequality reaches here.

        Which relations count is kept as written — only the scan changed.  What
        the scan fixes is the LHS: `y <= x^2 + a` used to hand `_op_name_index`
        the operator name `y <` (measured).
        """
        engine, supervisor = session
        _run(engine, supervisor, r"y \le x^2 + a", "math")
        assert _subst_lhs_rhs(engine, "E1") == ("y", "x^2 + a")

    def test_a_plain_transform_is_untouched(self, session):
        """The control: the everyday shape must not move."""
        engine, supervisor = session
        _run(engine, supervisor, "c_k = u a_k + v b_k", "math")
        assert _subst_lhs_rhs(engine, "E1") == ("c_k", "u*a_k + v*b_k")
        assert _new_basis_symbols(engine, ["E1"]) == ["c_k"]
        assert _forward_has_operators(engine, ["E1"]) is False
