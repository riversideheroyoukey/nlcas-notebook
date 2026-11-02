"""`^` is right-associative, so a power in a DENOMINATOR needs parentheses.

Two silent-wrong bugs, found 2026-07-30 while widening the user⇄CAS boundary
(docs/32 Phase 3); both predate that work and neither raised anything:

1. Division is emitted as ``base^-1``.  With an unparenthesised Pow base,
   ``k/r^2`` became ``k * r^2^-1`` — which the CAS reads right-associatively as
   ``k * r^(2^-1)`` = ``k*sqrt(r)``.  ``expand(k/r^2)`` returned ``k*sqrt(r)``.
   Surfaced as the central-force scenario's energy conservation ceasing to be 0
   (``d(-k/r)/dt`` no longer cancelled); see tests/test_central_force_scenario.py.

2. The parser read ``a^b^c`` LEFT-associatively, so the IR said ``(a^b)^c`` while
   every CAS says ``a^(b^c)``.  That stayed hidden because the generator re-emitted
   ``a^b^c`` unparenthesised and the CAS re-read it right-associatively — a right
   answer from a wrong AST.  Fixing (1) removed the cover, so (2) is fixed too.

The differential check against live Maxima is the real assertion here: the point
is that the emitted code MEANS what the user wrote, not that it looks a
particular way.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.ir import Superscript
from notebook.runtime.shared_parser import SharedParser


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="pow codegen") as e:
        yield e


class TestMeaningIsPreserved:
    """For each input, `input - lowered(input)` must be 0 in Maxima."""

    @pytest.mark.parametrize("expr", [
        "k/r^2",          # the reported shape: 1/r² potentials are everywhere
        "a/b^2",
        "1/x^3",
        "x^2/y^2",
        "a/b^c",          # symbolic exponent in the denominator
        "2^3^2",          # chained powers: 2^9 = 512, not (2^3)^2 = 64
        "x^y^z",
        "x^(y^2)",
        "(x^2)^3",
        "a/(b*c)",        # already correct — guard against over-parenthesising
        "a/b",
        "(a/b)^2",
        "e^(-a*t)",
    ])
    def test_differential_against_live_maxima(self, engine, expr):
        lowered = engine._handler.to_backend_code(expr).code
        r = engine._adapter.execute(f"ratsimp(({expr}) - ({lowered}))")
        assert r.ok
        assert r.value.strip() == "0", (
            f"{expr!r} lowered to {lowered!r}, which is a DIFFERENT expression")

    def test_power_in_denominator_is_parenthesised(self, engine):
        assert engine._handler.to_backend_code("k/r^2").code == "k * (r^2)^-1"

    def test_operation_on_such_an_expression_is_not_mangled(self, engine):
        """The user-visible form of the bug."""
        ex = engine.register_expression("k/r^2")
        st, val = engine.run_operation("expand", {"expr": ex.display_id})
        assert st.status.value == "success"
        assert val.replace(" ", "") == "k/r^2"
        assert "sqrt" not in val


class TestAssociativity:

    def test_power_chain_is_right_associative_in_the_ir(self):
        """a^b^c must be Superscript(a, Superscript(b, c))."""
        ast = SharedParser().parse("a^b^c")
        assert isinstance(ast, Superscript)
        assert isinstance(ast.script, Superscript), "read as (a^b)^c"
        assert getattr(ast.base, "name", None) == "a"

    def test_subscripts_stay_left_associative(self):
        """H_0_hat is Decorated(Subscript(H, 0)) — the `^` fix must not touch
        the subscript chain."""
        from notebook.models.ir import Decorated, Subscript
        ast = SharedParser().parse("H_0_hat")
        assert isinstance(ast, Decorated)
        assert isinstance(ast.base, Subscript)

    def test_mixed_superscript_subscript_reading_is_unchanged(self):
        """``a^b_c`` is ``(a^b)_c``: the exponent recursion covers ``^`` only, so
        a following subscript still attaches to the whole script chain."""
        from notebook.models.ir import Subscript
        ast = SharedParser().parse("a^b_c")
        assert isinstance(ast, Subscript)
        assert isinstance(ast.base, Superscript)
