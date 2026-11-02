"""Tests for F-8 signal-processing operations: fourier_transform_by_definition
/ inverse_fourier_by_definition / z_transform_by_definition (engine methods,
notebook/engine_fourier_mixin.py — mirror EngineLaplaceMixin's
explicit-function branch, no schema entry) and convolve (ordinary generic
alias-driven operation, notebook/runtime/maxima_signal_mixin.py).
"""
import asyncio

import pytest
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.builder import MaximaCommandBuilder

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["signal_processing"], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


@pytest.fixture(scope="module")
def b():
    return MaximaCommandBuilder()


class TestSchemaCompleteness:
    def test_transform_methods_not_in_schema(self, v):
        """Engine methods (mirror laplace_by_definition), not
        schema/run_operation-driven ops."""
        assert "fourier_transform_by_definition" not in v.known_operations
        assert "inverse_fourier_by_definition" not in v.known_operations
        assert "z_transform_by_definition" not in v.known_operations


class TestFourierByGeneralOperations:
    """Fourier transform / inverse / convolution written as ORDINARY DEFINITE
    INTEGRALS (SCENARIO-MACRO case 5).

    Replaces the value assertions of `fourier_transform_by_definition`,
    `inverse_fourier_by_definition` and `convolve`, deleted 2026-08-14: all
    three assembled an `integrate(...)` call the researcher can write.

    The macro also called `_positive_params_assume` — assume EVERY symbolic
    parameter positive, integrate, forget — which is a hidden modelling choice
    (CLAUDE.md, no-hardcoded-modeling-choices).  Here `a > 0` is an explicit
    step, which is why `test_parametrised_gaussian_needs_its_own_assumption`
    asserts BOTH halves: with the assumption it evaluates, without it the CAS
    asks and the operation fails with a reason.
    """

    @staticmethod
    def _value(eng, latex):
        """Register the LaTeX the researcher types, then evaluate it."""
        ex = eng.register_expression(
            eng._handler.latex_to_canonical(latex), latex=latex)
        step, _ = eng.run_operation("simplify", {"expr": ex.display_id})
        return eng.expressions[step.output_refs[0]].canonical.strip()

    def test_gaussian_round_trip(self):
        """e^{-t^2} → e^{-w^2/4}sqrt(pi) → e^{-t^2}, exactly."""
        nb.start_engine(title="test_f8_general_rt", mode="standard", language="en",
                        domains=["signal_processing"])
        eng = nb.get_engine()
        try:
            fwd = self._value(
                eng, r"\int_{-\infty}^{\infty} e^{-t^2} e^{-i \omega t} dt")
            assert fwd.replace(" ", "") == "%e^-(omega^2/4)*sqrt(%pi)", fwd
            back = self._value(
                eng, r"\frac{1}{2\pi}\int_{-\infty}^{\infty} \sqrt{\pi}"
                     r" e^{-\frac{\omega^2}{4}} e^{i \omega t} d\omega")
            assert back.replace(" ", "") == "%e^-t^2", back
        finally:
            nb.stop_engine()

    def test_parametrised_gaussian_needs_its_own_assumption(self):
        nb.start_engine(title="test_f8_general_a", mode="standard", language="en",
                        domains=["signal_processing"])
        eng = nb.get_engine()
        try:
            latex = r"\int_{-\infty}^{\infty} e^{-a t^2} e^{-i \omega t} dt"
            with pytest.raises(Exception):        # the CAS asks about sign(a)
                self._value(eng, latex)
            eng.add_assumption("a > 0", "a > 0")
            got = self._value(eng, latex)
            assert got.replace(" ", "") == "(%e^-(omega^2/(4*a))*sqrt(%pi))/sqrt(a)", got
        finally:
            nb.stop_engine()

    def test_gaussian_self_convolution(self):
        """Twice the variance.  Also the input that exposed the `d\tau` measure
        defect (open_items MEASURE-READ-TWICE) — before the fix this raised a
        raw pydantic ValidationError."""
        nb.start_engine(title="test_f8_general_conv", mode="standard", language="en",
                        domains=["signal_processing"])
        eng = nb.get_engine()
        try:
            got = self._value(
                eng, r"\int_{-\infty}^{\infty} e^{-\tau^2} e^{-(t-\tau)^2} d\tau")
            assert got.replace(" ", "") == "(%e^-(t^2/2)*sqrt(%pi))/sqrt(2)", got
        finally:
            nb.stop_engine()


class TestZTransform:
    def test_geometric_sequence(self):
        """Z{(1/2)^n}(z) = 2z/(2z-1) (+ a vanishing boundary artifact, per
        z_transform_definition_command's documented limitation)."""
        nb.start_engine(title="test_f8_z", mode="standard", language="en",
                         domains=["signal_processing"])
        eng = nb.get_engine()
        try:
            ex = eng.z_transform_by_definition("(1/2)^n")
            assert "(2*z)/(2*z-1)" in ex.canonical.replace(" ", "")
        finally:
            nb.stop_engine()

    def test_nl_trigger(self):
        nb.start_engine(title="test_f8_z_nl", mode="standard", language="en",
                         domains=["signal_processing"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "z transform (1/2)^n by definition")
            assert r["status"] == "completed"
            assert r["operation"] == "z_transform_by_definition"
        finally:
            nb.stop_engine()


class TestDefinitionCardIsNotAnEquation:
    """The trap `_transform_source_body` exists for, pinned where it lives.

    A "definition card" result carries `=` in its LATEX for display
    (`𝓕{f}(ω) = ∫… = value`) while its canonical is a bare value.  Feeding such
    a result back as the next call's target must resolve the CANONICAL — an
    earlier version reconstructed an equation from the latex instead, built
    `%int_… = …`, and HUNG the persistent adapter.

    Moved here 2026-08-14 from tests/test_ito_lemma.py, which was deleted with
    `apply_ito_lemma` (open_items SCENARIO-MACRO).  That file held the ONLY test
    naming this trap, and the trap is in this mixin, not in the deleted one —
    deleting the macro's tests wholesale would have dropped the guard while
    leaving the code it guards in place.
    """

    def test_feeding_a_transform_result_back_resolves_the_canonical(self):
        """Driven through the Z transform: it is the transform that REMAINS
        (SCENARIO-MACRO case 5 removed the two Fourier ones, and this test
        moved here from tests/test_ito_lemma.py in case 1 — it has now
        outlived two of its subjects, which is the point of moving it to where
        the TRAP is rather than leaving it with whatever called the code)."""
        nb.start_engine(title="test_f8_chain", mode="standard", language="en",
                        domains=["signal_processing"])
        eng = nb.get_engine()
        try:
            ex = eng.z_transform_by_definition("(1/2)^n")
            assert "=" in ex.latex          # the display IS an equation chain
            body = eng._transform_source_body(ex.display_id)
            assert "=" not in body, f"latex mistaken for an H-1 equation: {body!r}"
            assert body.strip() == (ex.canonical or "").strip()
        finally:
            nb.stop_engine()
