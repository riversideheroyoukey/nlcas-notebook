r"""A result that is already a statement gets no second left-hand side.

ONE-QUESTION-MANY-ANSWERS 残作業 1 (docs/open_items.md): the 2026-08-11 source
audit measured 18 of the raw `=` split sites and said nothing about the rest.
Re-run 2026-08-13 with the taint following DERIVED strings and with the audit
comparing MEANING rather than spacing, it named one disagreement — and behind it
sat this, which has nothing to do with sums:

    x + x = y   →  「E1 を簡約して」  →  latex  `x + x = 2 x=y`
                                        canonical `2*x = y`   (correct)

`_format_single_expr_latex` re-attached the source's left-hand side to a result
that already carried one.  Its guard asked whether the result STARTS WITH the
source's LHS *text* — a different question, which answers wrongly the instant
the CAS simplifies that side (`x + x` comes back as `2 x`).  The guard could not
have worked anyway: it looked for `=` in a four-character window while Maxima's
`tex1` writes `2 x=y` unspaced and our own display writes ` = `, so the two
spellings of "this is an equation" never met.

The fix asks `find_top_level_relation`, which this same file already uses 500
lines above for the mirror question ("is the SOURCE an equation") — one
definition, spacing-blind and bracket-aware.

These tests drive the real request pipeline, not the formatter: the 2026-08-09
lesson (docs/handoffs §2.2) is that probing a layer directly reports defects on
a path nobody takes, and its reverse — a fix that never reaches the researcher.

`TestTheDummyIndexIsNotAVariableOfTheEquation` below covers the second half of
the same migration: engine.py's remaining seven raw `=` splits now ask
`shared_parser` too (ratchet 7→0), and one of them decided something the
researcher sees.
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
    nb.start_engine(title="test_result_label", mode="standard",
                    language="ja", domains=[])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=[], primary_backend="maxima")
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


def _count_top_level_equals(latex: str) -> int:
    """How many `=` a reader sees at the top level of *latex*."""
    n, rest = 0, latex
    while True:
        rel = find_top_level_relation(rest)
        if rel is None:
            return n
        if rel[1] == "=":
            n += 1
        rest = rest[rel[0] + len(rel[1]):]


@pytest.mark.parametrize("source, instruction", [
    ("x + x = y", "E1 を簡約して"),                       # compound LHS, simplify
    ("a + a = b + b", "E1 を展開して"),                    # compound both sides, expand
    (r"\sum_{i=1}^{3} i^2 = S", "E1 を簡約して"),          # the nested-`=` shape too
])
def test_result_of_an_equation_carries_exactly_one_equals(session, source,
                                                          instruction):
    engine, supervisor = session
    assert _run(engine, supervisor, source, "math").get("status") == "registered"
    assert _run(engine, supervisor, instruction).get("status") == "completed"

    result = engine.expressions["E2"]
    assert _count_top_level_equals(result.latex) == 1, result.latex
    # The canonical was right all along — this was never a computation defect,
    # which is why nothing failed and the wrong string went to the paper.
    assert _count_top_level_equals(result.canonical) == 1, result.canonical


def test_a_named_equation_still_gets_its_label(session):
    """The label must not be dropped where it IS the point.

    `H = x + x` simplifies to the bare value `2*x`; without the source's LHS the
    researcher loses which quantity the line is about.
    """
    engine, supervisor = session
    _run(engine, supervisor, "H = x + x", "math")
    assert _run(engine, supervisor, "E1 を簡約して").get("status") == "completed"
    result = engine.expressions["E2"]
    assert result.latex.startswith("H ="), result.latex
    assert "=" not in result.canonical, result.canonical


def test_a_plain_expression_result_gains_no_relation(session):
    """No source equation, no invented one."""
    engine, supervisor = session
    _run(engine, supervisor, "x + x", "math")
    assert _run(engine, supervisor, "E1 を簡約して").get("status") == "completed"
    assert _count_top_level_equals(engine.expressions["E2"].latex) == 0


class TestTheDummyIndexIsNotAVariableOfTheEquation:
    r"""`differentiate_is_ambiguous` reads the RHS — of the EQUATION, not of the
    first `=` in the text.

    The naive split cut `\sum_{i=1}^3*i^2 = S` at its SUBSCRIPT, so the "right
    hand side" it measured was `1}^3*i^2 = S`, and the dummy index `i` counted
    as a free variable of the expression.  Differentiating by `S` — the only
    variable there actually is — then demanded "partial or total?", a question
    with no answer, about a symbol the researcher had not written twice.

    Measured 2026-08-13 (naive → True, bracket-aware → False).  The partial-vs-
    total decision is the sharpest of the seven engine.py sites because it is
    the one the researcher is asked about out loud; the other six feed the CAS
    or the display.
    """

    def test_differentiating_by_the_only_symbol_asks_nothing(self, session):
        engine, supervisor = session
        assert _run(engine, supervisor, r"\sum_{i=1}^{3} i^2 = S",
                    "math").get("status") == "registered"
        assert engine.differentiate_is_ambiguous("E1", "S") is False

    def test_a_free_symbol_beside_the_sum_still_asks(self):
        """The guard must still fire where it should — this is not "answer no".

        `\sum_{i=1}^{3} i^2 = S + b` differentiated by S leaves `b`, whose
        dependence on S nobody has stated.
        """
        nb.start_engine(title="test_result_label2", mode="standard",
                        language="ja", domains=[])
        engine = nb.get_engine()
        supervisor = LLMSupervisor(validator=OperationValidator(),
                                   llm_client=None, domains=[],
                                   primary_backend="maxima")
        try:
            _run(engine, supervisor, r"\sum_{i=1}^{3} i^2 = S + b", "math")
            assert engine.differentiate_is_ambiguous("E1", "S") is True
        finally:
            nb.stop_engine()
