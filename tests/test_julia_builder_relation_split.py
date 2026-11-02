r"""The Julia builders' `=` splits are command text — and one is a string literal.

ONE-QUESTION-MANY-ANSWERS ratchet, fifth lowering
(`notebook/runtime/julia_backend_handler.py` 5→0).  Counting first, as
docs/handoffs §4 requires: **all five land in a Julia command string**, four were
measurably broken, and `_build_apply_constraints`'s target LHS is also written
back into the RESULT text (`"{tgt_lhs} = " * string(_r)`) — the same two roles
the 代入 label has.

Measured 2026-08-13 on `\sum_{i=1}^{3} p_i = u*x + v*y`:

    Symbolics.substitute(1}^{3} p_i = u*x + v*y, …)
    Symbolics.coeff(1}^{3} p_i = u*x + v*y, x)
    _nb_solve_equation("\sum_{i", "1}^{3} p_i = u*x + v*y", "u")

Unlike Maxima — which accepted some of these — Julia refuses all four with a
ParseError.  Loud, but pointing nowhere: the solve one interpolates the fragment
into a STRING LITERAL, so what the researcher is shown is a complaint that `\s`
is an invalid escape.

`@variables` is scanned off the same text, so a mis-split warped the declaration
and the body independently.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.julia_backend_handler import JuliaBackendHandler

SUM_EQUATION = r"\sum_{i=1}^{3} p_i = u*x + v*y"
LEFT_SIDE = r"\sum_{i=1}^{3} p_i"
#: What head-loss leaves: the subscript's remainder standing on its own.
MANGLED = "1}^{3} p_i"


def head_lost(command: str) -> bool:
    """Does *command* contain the remainder WITHOUT its `\sum_{i=` head?

    Written this way because the obvious `MANGLED not in command` fires on the
    CORRECT answer — `1}^{3} p_i` is a substring of `\sum_{i=1}^{3} p_i`.  That
    is the third sentinel in this ratchet's tests to have been wrong the same
    way (see tests/test_maxima_builder_relation_split.py), so it is a named
    function here rather than an inline `not in`.  **A sentinel that fires on
    the right answer measures nothing.**
    """
    return MANGLED in command.replace(LEFT_SIDE, "")


class TestThePureFunctions:
    """No CAS needed — these two are static."""

    def test_eq_str_for_solve_keeps_the_whole_left_side(self):
        assert JuliaBackendHandler._eq_str_for_solve(SUM_EQUATION, {}) == (
            r"\sum_{i=1}^{3} p_i", "u*x + v*y")

    def test_a_definition_is_still_not_an_equation(self):
        assert JuliaBackendHandler._eq_str_for_solve("f(x) := x^2", {}) is None
        assert JuliaBackendHandler._eq_str_for_solve("x + y", {}) is None

    def test_solve_output_parsing_is_unchanged(self):
        parse = JuliaBackendHandler.parse_solve_result
        assert parse("x = 2, y = 1") == [("x", "2"), ("y", "1")]
        assert parse("") == []


@pytest.fixture(scope="module")
def julia_session():
    """ONE Julia process for the whole module (T-2: start costs ~0.9 s each)."""
    with NotebookEngine(title="jl_split", mode=SessionMode.second_quantization) as engine:
        engine.start()
        # `p` must be a declared scalar: since 2026-08-22 a WRITTEN range is
        # expanded (`\sum_{i=1}^{3} p_i` → `p_1 + p_2 + p_3`), and an indexed
        # scalar whose base is undeclared is refused — exactly as the written-out
        # spelling always was.  The two spellings now agree, which is the point;
        # before, the bounded sum was lowered by DROPPING its range.
        engine.run_operation("define_scalar_parameter", {"name": "p"})
        engine.register_expression(SUM_EQUATION)          # E1
        engine.register_expression("F = x^2 + y^2")       # E2
        yield engine


def _command(engine, operation, args):
    """The command built for *operation*, whether or not Julia accepted it."""
    before = set(engine.steps)
    try:
        engine.run_operation(operation, args)
    except Exception:
        pass                      # the command is the artefact, not the outcome
    new = [s for sid, s in engine.steps.items()
           if sid not in before and s.rendered_command]
    assert new, f"{operation} built no command"
    return new[-1].rendered_command


class TestTheCommandsCarryTheWholeEquation:
    def test_constraint_relations(self, julia_session):
        command = _command(julia_session, "apply_canonical_constraints",
                           {"expr": "E2", "relations": "E1"})
        assert not head_lost(command), command
        assert r"\sum_{i=1}^{3} p_i => u*x + v*y" in command, command

    def test_constraint_target(self, julia_session):
        command = _command(julia_session, "apply_canonical_constraints",
                           {"expr": "E1", "relations": "E2"})
        assert not head_lost(command), command
        assert "Symbolics.substitute(u*x + v*y, " in command, command

    def test_metric_coefficients(self, julia_session):
        command = _command(julia_session, "derive_metric_constraints",
                           {"forward": "E1", "basis": "x, y", "gram": "1,0;0,1"})
        assert not head_lost(command), command
        assert "Symbolics.coeff(u*x + v*y, x)" in command, command

    def test_solve_operands(self, julia_session):
        command = _command(julia_session, "solve_equation",
                           {"equation": "E1", "var": "u"})
        assert not head_lost(command), command
        # Both operands arrive WHOLE — this ratchet's subject — and each arrives
        # as Julia CODE, which is BUILDER-OPERAND-NOT-LOWERED's half (closed
        # 2026-08-15; the shape is pinned in TestTheOperandIsLowered below).
        assert '_nb_solve_equation(' in command, command
        assert '"u * x + v * y"' in command or '"u*x + v*y"' in command, command


class TestTheOperandIsLowered:
    r"""Closed 2026-08-15 (open_items BUILDER-OPERAND-NOT-LOWERED).

    The solve command interpolates its operands into a Julia STRING LITERAL, so
    the researcher's `\sum` used to make `\s` an invalid escape and the call
    failed to LEX — a character-level ParseError with no sign of the cause.

    Lowering here does NOT go through `to_backend_code`: that door is Maxima's.
    On this side `parse_ir` returns None, so it answers `passthrough` and hands
    the raw text straight on — measured.  The Julia route is
    parser → JuliaSemanticResolver → JuliaCodeGenerator, the same one
    `_make_build_ctx` uses.
    """

    def test_the_operand_reaches_julia_as_code(self, julia_session):
        command = _command(julia_session, "solve_equation",
                           {"equation": "E1", "var": "u"})
        assert "\\sum" not in command, command
        # …and the Σ is GONE from the command because it was lowered — either
        # into QuantumAlgebra's own `∑` (no range written) or, since 2026-08-22,
        # into its terms (a concrete range, which QA's formal `∑` cannot hold).
        assert ("_QA.∑" in command or "sum" in command.lower()
                or "param(:p, 1)" in command), command
