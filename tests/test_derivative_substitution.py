"""H-10: substituting an equation whose LHS is a derivative (or a SUM of them).

Found via scenario_coulomb_gauge: substituting the gauge condition
∂(∇·A)/∂t = 0 into the expanded ∇·E failed with "IndexedSymbol index must
resolve to a Symbol or IndexSymbol, got Pow", so the scenario had to stop at
"see it by eye" instead of deriving the Poisson equation in the CAS.

Three separate defects stacked up, each pinned below:

1. Maxima's own tex1 display form of an unevaluated derivative
   (``\\frac{d^2}{d t d z} A_{z}``) had no inverse, so the LaTeX
   lhs_alias_map key could not be turned back into ``'diff(A_z,t,1,z,1)``.
2. The substitution rule shipped that unconverted LaTeX straight to the CAS
   instead of falling back to the source expression's canonical — that is
   where the ``\\frac{…}`` reached the resolver and raised.
3. Even with a correct canonical LHS, Maxima's ``subst`` is purely syntactic
   and never matches a COMPOUND (sum) LHS — it returned the target unchanged.
   ``ratsubst`` does the algebraic match.
"""
import asyncio

import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.llm_supervisor import LLMSupervisor

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _handler():
    return NotebookEngine()._handler


class TestDerivativeDisplayInverse:
    """Defect 1: Maxima's derivative DISPLAY form must round-trip to canonical."""

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{d z} \varphi",              "'diff(phi,z,1)"),
        (r"\frac{d^2}{d z^2} \varphi",          "'diff(phi,z,2)"),
        (r"\frac{d^2}{d t d z} A_{z}",          "'diff(A_z,t,1,z,1)"),
        (r"\frac{d^3}{d t d z^2} f",            "'diff(f,t,1,z,2)"),
        (r"\frac{d^3}{d x d y d z} f",          "'diff(f,x,1,y,1,z,1)"),
        (r"\frac{d^2}{d t d \theta} A_{z}",     "'diff(A_z,t,1,theta,1)"),
    ])
    def test_display_form_inverts(self, latex, canonical):
        assert _handler().lhs_latex_to_canonical(latex) == canonical

    def test_sum_of_derivatives_inverts_termwise(self):
        """The Coulomb-gauge shape: a compound LHS of three mixed partials."""
        src = (r"\frac{d^2}{d t d z} A_{z}+\frac{d^2}{d t d y} A_{y}"
               r"+\frac{d^2}{d t d x} A_{x}")
        assert _handler().lhs_latex_to_canonical(src) == (
            "'diff(A_z,t,1,z,1)+'diff(A_y,t,1,y,1)+'diff(A_x,t,1,x,1)")

    @pytest.mark.parametrize("latex", [
        r"\frac{d^3}{d t d z} f",        # numerator order != sum of denominator orders
        r"\frac{a}{b} c",                # an ordinary fraction, not a derivative
        r"\frac{d^2}{d t x d z} f",      # unparsed text between denominator factors
    ])
    def test_non_derivative_is_left_verbatim(self, latex):
        """Never guess: anything that is not exactly the generated shape is
        returned untouched, so the caller can reject it (see defect 2)."""
        assert _handler().lhs_latex_to_canonical(latex) == latex

    def test_dot_notation_path_still_wins(self):
        assert _handler().lhs_latex_to_canonical(r"\ddot{q}") == "'diff(q,t,2)"


class TestCompoundSubstLhs:
    """Defect 3: which LHS shapes Maxima's syntactic subst cannot match."""

    @pytest.mark.parametrize("lhs", [
        "a+b",
        "a-b",
        "'diff(u,t,1,z,1)+'diff(v,t,1,y,1)",
        "-a-b-c",
    ])
    def test_compound(self, lhs):
        assert MaximaBackendHandler._is_compound_subst_lhs(lhs) is True

    @pytest.mark.parametrize("lhs", [
        "a",
        "-a",                       # leading unary sign is not an addition
        "'diff(A_z,t,1,z,1)",
        "f(a+b)",                   # the sum is bracketed, not top level
        "%e^-x",                    # sign after an operator
        "sin(a-b)*c",
    ])
    def test_atomic(self, lhs):
        assert MaximaBackendHandler._is_compound_subst_lhs(lhs) is False


class TestSubstituteCommandShape:
    """Defect 3, end to end through a live Maxima session."""

    TARGET = "-'diff(phi,z,2)-'diff(u,t,1,z,1)-'diff(v,t,1,y,1)"

    def test_compound_lhs_uses_ratsubst_and_cancels(self):
        nb.start_engine(title="test_h10_ratsubst", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        try:
            src = eng.register_expression(self.TARGET)
            step, value = eng.run_operation("substitute", {
                "expr": src.display_id,
                "lhs":  "'diff(u,t,1,z,1)+'diff(v,t,1,y,1)",
                "rhs":  "0",
            })
            assert "ratsubst(" in step.rendered_command
            assert "subst(" not in step.rendered_command.replace("ratsubst(", "")
            # The whole gauge sum cancels; only the scalar-potential term is left.
            assert value.replace(" ", "") == "-'diff(phi,z,2)"
        finally:
            nb.stop_engine()

    def test_atomic_lhs_keeps_subst(self):
        """Unchanged behaviour for the ordinary case: plain subst, and only the
        named term is replaced."""
        nb.start_engine(title="test_h10_subst", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        try:
            src = eng.register_expression(self.TARGET)
            step, value = eng.run_operation("substitute", {
                "expr": src.display_id, "lhs": "'diff(v,t,1,y,1)", "rhs": "0"})
            assert "ratsubst(" not in step.rendered_command
            assert "subst(" in step.rendered_command
            flat = value.replace(" ", "")
            assert "'diff(v,t,1,y,1)" not in flat
            assert "'diff(u,t,1,z,1)" in flat and "'diff(phi,z,2)" in flat
        finally:
            nb.stop_engine()


class TestUnconvertibleLatexAliasFallsBack:
    """Defect 2: an lhs_alias_map key the backend cannot canonicalise must be
    DROPPED (falling back to the source expression's canonical), never handed
    to the CAS as raw LaTeX."""

    @pytest.mark.parametrize("lang,command", [
        ("ja", "E2 に E1 を代入して"),
        ("en", "substitute E1 into E2"),
    ])
    def test_substitution_still_completes(self, lang, command):
        nb.start_engine(title="test_h10_alias", mode="standard", language=lang,
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["electromagnetism"], primary_backend="maxima")
        try:
            src = eng.register_expression(
                "'diff(u,t,1,z,1)+'diff(v,t,1,y,1) = 0")
            eng.register_expression(
                "-'diff(phi,z,2)-'diff(u,t,1,z,1)-'diff(v,t,1,y,1)")
            # Half-convertible on purpose: the second \frac has an order that does
            # not add up, so restoration leaves LaTeX behind.
            eng.lhs_alias_map[r"\frac{d^2}{d t d z} u+\frac{d^2}{d q} v"] = \
                src.display_id
            res = asyncio.run(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=command)]),
                eng, sup))
            assert res.get("status") == "completed", res
            assert res["canonical"].replace(" ", "") == "-'diff(phi,z,2)"
        finally:
            nb.stop_engine()
