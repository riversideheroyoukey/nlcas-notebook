r"""INTEGRAL-CLAIM-ASYMMETRY: a definite integral is the same OBJECT however it is written.

Every spelling below already produced the right ANSWER — that was measured when
the ticket was filed.  What split was the object KIND: `\int_0^{2\pi} r\,d\theta`
became a first-class integral form (transformable, evaluable, `この定積分を評価
して` finds it) while `\int_0^{\frac{\pi}{2}} \sin x\,dx` became an ordinary
expression, purely because the claiming regex could not nest braces in a bound.

Measuring the repair moved the design twice, and both measurements are pinned
here because either would be easy to undo:

* **The IR route was the ticket's own suggestion and is wrong for this.**  ②'s
  IR carries every bound (verified in `test_integral_reader_corpus`), but its
  upper limit for `\frac{\pi}{2}` is `%pi * 2^-1` — rendering the researcher's
  own spelling back would need the CAS, and route ① is *defined* by not calling
  the CAS at input.  So ① keeps its own reading; only the LIMITS were repaired,
  with balanced braces.
* **Widening the claim without routing `簡約` would have been a regression.**
  A claimed integral refuses generic operations (`integral.display_only_form`),
  so the spellings newly claimed would have STOPPED answering `簡約して`, which
  they answered before.  The two halves belong together.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.integral_form import parse_latex_definite_integral
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOMAIN = "electromagnetism"

#: (spelling, written display, value).  One INTEGRAL per row; the point is that
#: the kind, the display and the value agree across the spellings of a bound.
CASES = [
    (r"\int_0^1 x^2 \, dx",                 r"\int_{0}^{1} x^2 \, dx",              "1/3"),
    (r"\int_0^{2\pi} r \, d\theta",         r"\int_{0}^{2\pi} r \, d\theta",        "2*%pi*r"),
    (r"\int_0^{\frac{\pi}{2}} \sin x \, dx", r"\int_{0}^{\frac{\pi}{2}} \sin x \, dx", "1"),
    (r"\int_0^{\sqrt{2}} x \, dx",          r"\int_{0}^{\sqrt{2}} x \, dx",         "1"),
    (r"\int_{-\infty}^{\infty} e^{-x^2} \, dx",
     r"\int_{-\infty}^{\infty} e^{-x^2} \, dx",                                     "sqrt(%pi)"),
    (r"\int_{a}^{b} f(x) \, dx",            r"\int_{a}^{b} f(x) \, dx",             None),
]


def _session(lang="ja"):
    nb.start_engine(title="claim", mode="standard", language=lang, domains=[DOMAIN])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=[DOMAIN], primary_backend="maxima")

    def run(text):
        return asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))
    return eng, run


class TestTheClaimIsDecidedByNotationNotSpelling:
    @pytest.mark.parametrize("latex,display,_value", CASES,
                             ids=[c[0][:34] for c in CASES])
    def test_it_is_an_integral_form(self, latex, display, _value):
        eng, run = _session()
        try:
            r = run(latex)
            eid = r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]
            assert eng.expressions[eid].integral_descriptor is not None, latex
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("latex,display,_value", CASES,
                             ids=[c[0][:34] for c in CASES])
    def test_the_written_bound_survives_into_the_display(self, latex, display, _value):
        r"""The descriptor's bounds are BACKEND text by then (`2*%pi`), so the
        display is a reconstruction — and a reconstruction of `\frac{\pi}{2}`
        from `((%pi)/(2))` cannot get back what was written.  The written form
        is kept beside the backend one for exactly this."""
        eng, run = _session()
        try:
            r = run(latex)
            eid = r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]
            assert eng.expressions[eid].latex == display
        finally:
            nb.stop_engine()


class TestBothVerbsAnswer:
    r"""`簡約` must keep working for the spellings the claim newly covers."""

    @pytest.mark.parametrize("latex,_display,value",
                             [c for c in CASES if c[2] is not None],
                             ids=[c[0][:34] for c in CASES if c[2] is not None])
    @pytest.mark.parametrize("verb", ["評価", "簡約"])
    def test_value(self, latex, _display, value, verb):
        eng, run = _session()
        try:
            r = run(latex)
            eid = r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]
            out = run(f"{eid} を{verb}して")
            assert out["status"] == "completed", out
            assert out["canonical"].replace(" ", "") == value
        finally:
            nb.stop_engine()

    def test_english_simplify_too(self):
        eng, run = _session("en")
        try:
            r = run(r"\int_0^{\frac{\pi}{2}} \sin x \, dx")
            eid = r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]
            out = run(f"simplify {eid}")
            assert out["status"] == "completed", out
            assert out["canonical"].strip() == "1"
        finally:
            nb.stop_engine()

    def test_an_ordinary_expression_is_untouched_by_the_widened_verb(self):
        """`簡約` still means simplify for everything that is not an integral
        form — the interceptor declines and the generic path runs."""
        eng, run = _session()
        try:
            r = run("x^2 + 2*x + 1")
            eid = r["registered_expressions"][-1]["expr_id"]
            out = run(f"{eid} を簡約して")
            assert out["status"] == "completed", out
            assert "x^2" in out["canonical"]
        finally:
            nb.stop_engine()


class TestWhatIsStillDeclined:
    r"""Deliberate, and recorded so a later reader does not take it for an
    oversight: these are about the MEASURE, not the limits, and reading them
    needs ②'s IR walk.  Their ANSWERS are already right (the reader corpus
    checks that); what they lack is object status."""

    @pytest.mark.parametrize("latex", [
        r"\int x^2 \, dx",                                        # no bounds at all
        r"\int_0^1 dx \, x^2",                                    # leading measure
        r"\int_{-\infty}^{\infty} \frac{d\omega}{2\pi} G(\omega)",  # fraction measure
    ])
    def test_not_claimed(self, latex):
        assert parse_latex_definite_integral(latex) is None
