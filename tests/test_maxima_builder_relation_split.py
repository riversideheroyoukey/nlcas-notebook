r"""The Maxima builders' `=` splits are command text, so a mis-split is CAS code.

ONE-QUESTION-MANY-ANSWERS ratchet, third lowering
(`notebook/runtime/maxima_backend_handler.py` 6→0).  Counting where the six
values went — the step docs/handoffs §4 requires before any unification — put
**all six into a CAS command string**, and four of them were measurably wrong.
With a forward equation carrying a bounded sum on its LEFT, the commands built
on 2026-08-13 were:

    ratsimp(diff((1}^3*c_i = u*a + v*b),q)*diff((q*p),p) - …)
    block([F], F = ratsubst(1}^3*c_i = u*a + v*b, \sum_{i, a^2 + b^2))
    _M: matrix([coeff(1}^3*c_i = u*a + v*b, a), coeff(1}^3*c_i = u*a + v*b, b)])

The `ratsubst` line is wrong on BOTH sides at once: a replacement that still
carries an `=`, and the raw LaTeX fragment `\sum_{i` as the pattern.

**The finding worth keeping is why the sixth site was safe.**
`_build_substitute` splits the same spelling and was fine — because the lowering
pass runs BEFORE it, so `\sum_{i=1}^3*c_i` has already become
`sum(c(i), i, 1, 3)` and the bracketed `=` is gone.  The other three read
`expr_lookup` raw.  One file, one spelling, two outcomes, decided by the ORDER of
two passes that nothing states — the same shape as open_items
SPLIT-REACH-ASSUMPTION, this time inside a single file.

These drive `engine.run_operation`, the API the interceptors call, and assert on
`Step.rendered_command` — the artefact the defect actually reached.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline

SUM_TRANSFORM = r"\sum_{i=1}^{3} c_i = u a + v b"
#: What head-loss leaves behind, whatever the builder: the remainder of the
#: subscript followed by the equation's own `=`, where a value was expected.
#: Not the bare `1}^` — the CORRECT pattern `\sum_{i=1}^3*c_i` contains that,
#: and a sentinel that fires on the right answer measures nothing.
MANGLED = "1}^3*c_i ="


def _run(engine, supervisor, value, ttype="text"):
    token = (InputToken(type="math", value=value, latex=value, canonical="")
             if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[token]), engine, supervisor))


def _session(domains=()):
    nb.start_engine(title="test_mx_split", mode="standard", language="ja",
                    domains=list(domains))
    return nb.get_engine(), LLMSupervisor(
        validator=OperationValidator(), llm_client=None,
        domains=list(domains), primary_backend="maxima")


def _last_command(engine):
    commands = [s.rendered_command for s in engine.steps.values()
                if s.rendered_command]
    assert commands, "no command reached the CAS"
    return commands[-1]


@pytest.fixture
def plain():
    engine, supervisor = _session()
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


@pytest.fixture
def mechanics():
    engine, supervisor = _session(["classical_mechanics"])
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


class TestTheCommandsCarryTheWholeEquation:
    def test_poisson_bracket_operand(self, mechanics):
        engine, supervisor = mechanics
        _run(engine, supervisor, "q と p を正準対とする")
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        _run(engine, supervisor, "G = q*p", "math")
        engine.run_operation("poisson_bracket", {"a": "E1", "b": "E2"})
        command = _last_command(engine)
        assert MANGLED not in command, command
        assert "diff((u*a + v*b),q)" in command, command

    def test_canonical_constraints_pattern_and_replacement(self, plain):
        engine, supervisor = plain
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        _run(engine, supervisor, "F = a^2 + b^2", "math")
        engine.run_operation("apply_canonical_constraints",
                             {"expr": "E2", "relations": "E1"})
        command = _last_command(engine)
        assert MANGLED not in command, command
        assert "ratsubst(u * a + v * b, " in command.replace("u*a + v*b", "u * a + v * b"), command
        # The block-local symbol is the target's LHS, not a fragment of it.
        assert command.startswith("block([F], F = "), command

    def test_canonical_constraints_target_side(self, plain):
        r"""The TARGET's own split is a DIFFERENT site from the relation's.

        Written after the first version of this class passed under a mutation of
        it: with a simple target (`F = a^2 + b^2`) the naive split is right, so
        the case measured nothing.  Naive, with the sum as the target:
            block([\sum_{i], \sum_{i = ratsubst(2*w, u, 1}^3*c_i = u*a + v*b))
        — the block-local symbol is a fragment of the researcher's left side.
        """
        engine, supervisor = plain
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        _run(engine, supervisor, "u = 2 w", "math")
        engine.run_operation("apply_canonical_constraints",
                             {"expr": "E1", "relations": "E2"})
        # Asserted WHOLE, not by the MANGLED sentinel: here the correct command
        # contains `…1}^3*c_i = …` as well, because the target's own left side
        # is the sum.  A sentinel that fires on the right answer measures
        # nothing — the second time this file learned that in one sitting.
        # Asserted WHOLE, not by the MANGLED sentinel: here the correct command
        # contains `…1}^3*c_i = …` as well, because the target's own left side
        # is the sum.  A sentinel that fires on the right answer measures
        # nothing.
        #
        # The shape changed on 2026-08-15 (BUILDER-OPERAND-NOT-LOWERED).  What
        # this used to assert was measured to FAIL in Maxima: a compound left
        # side was put inside `block([…])`, which can only shadow a SYMBOL, and
        # the command came back as `_res_` — the adapter's scratch name — with
        # no error anywhere.  Nothing is P2-bound for a compound left side, so
        # nothing needs shadowing; it is lowered like the rest of the maths.
        cmd = _last_command(engine)
        assert cmd == ("sum(c(i), i, 1, 3) = "
                       "ratsubst(2 * w, u, u * a + v * b)"), cmd

    def test_metric_constraints_coefficients(self, plain):
        engine, supervisor = plain
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        engine.run_operation("derive_metric_constraints",
                             {"forward": "E1", "basis": "a, b",
                              "gram": "1,0;0,1"})
        command = _last_command(engine)
        assert MANGLED not in command, command
        assert "coeff(u*a + v*b, a)" in command, command


class TestWhatWasAlreadyRight:
    def test_substitute_no_longer_splits_at_all(self, plain):
        """The sixth site is GONE (2026-08-14), which is better than being
        "safe by pass ORDER" — the property this file's docstring called an
        unwritten dependency.

        The builder used to cut the source down to its right-hand side, and that
        cut applied to EVERY equation: measured on
        `L(s^2 Q - s q_0 - u_0) + … = V` the substitution's source became the
        bare `V`, so no target could ever be found.  It now passes the whole
        equation and shadows the left symbol instead (`block([H], …)`), which
        needs no split and therefore cannot mangle one.
        """
        engine, supervisor = plain
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        engine.run_operation("substitute", {"expr": "E1", "lhs": "u", "rhs": "2"})
        command = _last_command(engine)
        assert MANGLED not in command, command
        # the equation survives whole — both sides reach subst()
        assert "sum(c(i), i, 1, 3)=u * a + v * b" in command, command

    def test_solve_output_parsing_is_unchanged(self):
        """CAS output: `var = val`, the `=` always leftmost.  Pure function."""
        parse = MaximaBackendHandler.parse_solve_result
        assert parse("[[x = 2, y = 1]]") == [("x", "2"), ("y", "1")]
        assert parse("[x = -1, x = 1]") == [("x", "-1"), ("x", "1")]
        assert parse("[]") == []


class TestTheOperandsAreLowered:
    r"""Closed 2026-08-15 (open_items BUILDER-OPERAND-NOT-LOWERED).

    `ratsubst`'s pattern used to be the researcher's own `\sum_{i=1}^3*c_i`,
    raw LaTeX Maxima cannot read.  Everything that goes into the command as
    MATHEMATICS is lowered now; the block-local symbol is not, because it goes
    in as a NAME — and that split is exactly why `to_backend_code` could not
    simply be widened to every operation, which CLAUDE.md records as measured
    wrong (a Poisson bracket returned 0, `convolve` hung).
    """

    def test_the_pattern_reaches_maxima_as_cas_code(self, plain):
        engine, supervisor = plain
        _run(engine, supervisor, SUM_TRANSFORM, "math")
        _run(engine, supervisor, "F = a^2 + b^2", "math")
        engine.run_operation("apply_canonical_constraints",
                             {"expr": "E2", "relations": "E1"})
        cmd = _last_command(engine)
        assert "sum(c(i), i, 1, 3)" in cmd, cmd
        assert "\\sum" not in cmd, cmd
