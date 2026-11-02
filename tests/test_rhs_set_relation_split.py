r"""H-5z registers what it splits, so a mis-split is PERSISTED.

ONE-QUESTION-MANY-ANSWERS ratchet, sixth lowering
(`ui/interpret_equation_intercepts.py` 4→0).  Counting where the four values
went put two of them into `register_expression` — the canonical AND the latex of
a NEW expression — and the other two into an empty-canonical fallback that no
saved session reaches (measured: 0 of 75 expressions in `sessions/*.json` has an
empty canonical).

That first pair is the worst destination this ratchet has met.  The five files
before it fed a command or a label: the string was used once and gone.  Here it
is written into the session file, the timeline and the exported .tex, and every
later operation that references the expression reads it back.

Measured 2026-08-13 — `\sum_{i=1}^{3} c_i = u a + v b`, then
「E1 の右辺を 0 とする」:

    status     completed          ← no error anywhere
    canonical  1}^3*c_i = u*a + v*b = 0
    latex      1}^{3} c_i = u a + v b = 0

A fragment head and two `=`, stored.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.shared_parser import find_top_level_relation
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline


def _run(engine, supervisor, value, ttype="text"):
    token = (InputToken(type="math", value=value, latex=value, canonical="")
             if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[token]), engine, supervisor))


@pytest.fixture
def session():
    nb.start_engine(title="test_rhs_set_split", mode="standard", language="ja",
                    domains=[])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=[], primary_backend="maxima")
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


def _relation_count(text: str) -> int:
    n, rest = 0, text
    while True:
        found = find_top_level_relation(rest)
        if found is None:
            return n
        n += 1
        rest = rest[found[0] + len(found[1]):]


@pytest.mark.parametrize("source", [
    r"\sum_{i=1}^{3} c_i = u a + v b",   # bracketed `=` before the equation's
    "F = u a + v b",                     # control: the everyday shape
])
def test_the_registered_constraint_is_the_right_hand_side_set_to_zero(
        session, source):
    engine, supervisor = session
    assert _run(engine, supervisor, source, "math").get("status") == "registered"
    assert _run(engine, supervisor, "E1 の右辺を 0 とする").get("status") == "completed"

    stored = engine.expressions["E2"]
    assert stored.canonical == "u*a + v*b = 0", stored.canonical
    assert stored.latex == "u a + v b = 0", stored.latex
    # Said twice on purpose: the defect was a SECOND relation appearing in a
    # string that is saved to disk, so the shape is asserted as well as the text.
    assert _relation_count(stored.canonical) == 1
    assert _relation_count(stored.latex) == 1


def test_an_h1_expression_whose_canonical_is_the_rhs_already(session):
    """The middle branch — canonical carries no `=` — must stay untouched.

    H-1 stores the RHS only when the left side is a plain symbol, so this is the
    common path and it never had a split to get wrong.
    """
    engine, supervisor = session
    _run(engine, supervisor, "y = x^2")            # the TEXT box, not a math chip
    # Measured, not assumed, and it took two tries: which branch runs depends on
    # the INPUT ROUTE.  The same `y = x^2` stores `x^2` from the text box and
    # `y = x^2` from a `$…$` math chip — the two routes genuinely hand the
    # pipeline different strings (CLAUDE.md, "A LOOKUP KEY is a fourth thing").
    assert engine.expressions["E1"].canonical == "x^2"
    assert _run(engine, supervisor, "E1 の右辺を 0 とする").get("status") == "completed"
    assert engine.expressions["E2"].canonical == "x^2 = 0"
