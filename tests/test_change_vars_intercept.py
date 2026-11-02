"""S-o: NL → rule intercept for `change_integration_variables`.

The substitution must be parsed deterministically (no LLM) — change of variables
is routine in definite integration.  Guards ui/interpret_pipeline._try_change_vars.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
    _split_top_commas,
)
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["electromagnetism"], primary_backend="maxima")


def _run(eng, sup, cmd):
    req = InterpretRequest(tokens=[InputToken(type="text", value=cmd)])
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(req, eng, sup))


def _register(eng, sup, expr):
    """Register a math expression and return its (globally-unique) display id."""
    req = InterpretRequest(tokens=[InputToken(type="math", value=expr, canonical=expr)])
    r = asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(req, eng, sup))
    return r["registered_expressions"][-1]["expr_id"]


class TestChangeVarsRegex:
    def test_parses_equation_form(self):
        m = nlp("change_vars", "ja").match(
            "E1 を x,y から r,theta へ x=r*cos(theta), y=r*sin(theta) で変数変換して")
        assert m is not None
        assert m.group("expr") == "E1"
        assert _split_top_commas(m.group("oldv")) == ["x", "y"]
        assert _split_top_commas(m.group("newv")) == ["r", "theta"]

    def test_protects_function_arg_commas(self):
        # atan2(v, u) must stay one piece
        parts = _split_top_commas("sqrt(u^2+v^2), atan2(v, u)")
        assert parts == ["sqrt(u^2+v^2)", "atan2(v, u)"]

    def test_no_match_without_keyword(self):
        assert nlp("change_vars", "ja").match("E1 を x,y から r,theta へ 何か") is None


class TestChangeVarsPipeline:
    def _engine(self):
        nb.start_engine(title="s_o intercept", mode="standard", language="ja",
                        domains=["electromagnetism"])
        return nb.get_engine()

    def test_polar_via_nl(self):
        """∬(x²+y²)dxdy → polar: raw Jacobian product, then simplify → r³."""
        eng = self._engine()
        sup = _sup()
        try:
            eid = _register(eng, sup, "x^2+y^2")
            r = _run(eng, sup,
                     f"{eid} を x,y から r,theta へ "
                     "x=r*cos(theta), y=r*sin(theta) で変数変換して")
            assert r["status"] == "completed", r
            assert r["operation"] == "change_integration_variables"
            cid = r["expr_id"]
            _, polar = eng.run_operation("simplify", {"expr": cid})
            assert polar.replace(" ", "") == "r^3"
        finally:
            nb.stop_engine()

    def test_spherical_volume_element_literal_integrand(self):
        """Pure volume element: integrand `1` is a LITERAL — no need to register it
        (validator only rejects unknown E-ID-shaped refs).  → r²sin(theta)."""
        eng = self._engine()
        sup = _sup()
        try:
            r = _run(eng, sup,
                     "1 を x,y,z から r,theta,phi へ "
                     "x=r*sin(theta)*cos(phi), y=r*sin(theta)*sin(phi), "
                     "z=r*cos(theta) で変数変換して")
            assert r["status"] == "completed", r
            cid = r["expr_id"]
            _, sph = eng.run_operation("simplify", {"expr": cid})
            assert sph.replace(" ", "") == "r^2*sin(theta)"
        finally:
            nb.stop_engine()

    def test_general_linear_jacobian(self):
        """General (non-coordinate) Jacobian with a *positive-orientation* new-var
        order: x=2u+v, y=u+3v has det J = 5 (the area-scaling factor).  Integrand
        x → 2u+v, so the new integrand is 5(2u+v) = 10u+5v."""
        eng = self._engine()
        sup = _sup()
        try:
            eid = _register(eng, sup, "x")
            r = _run(eng, sup,
                     f"{eid} を x,y から u,v へ "
                     "x=2*u+v, y=u+3*v で変数変換して")
            assert r["status"] == "completed", r
            _, val = eng.run_operation("simplify", {"expr": r["expr_id"]})
            assert val.replace(" ", "") == "5*v+10*u"
        finally:
            nb.stop_engine()

    def test_signed_determinant_is_documented(self):
        """The op inserts the SIGNED det J (not |det J|): an orientation-reversing
        new-var order yields a negative factor.  Guards the documented convention
        (order new vars so det J > 0 for a positive integration measure)."""
        eng = self._engine()
        sup = _sup()
        try:
            eid = _register(eng, sup, "x+y")
            r = _run(eng, sup,
                     f"{eid} を x,y から u,v へ "
                     "x=(u+v)/2, y=(u-v)/2 で変数変換して")  # det J = -1/2
            _, val = eng.run_operation("simplify", {"expr": r["expr_id"]})
            assert val.replace(" ", "") == "-(u/2)"
        finally:
            nb.stop_engine()

    def test_change_then_definite_integral_full_nl(self):
        """The transformed integrand IS a genuine integration-variable change, and
        the WHOLE loop runs through NL (no LLM): change vars → simplify →
        ∫₀¹ r³ dr → ∫₀^{2π} dθ = π/2.  Bounds (incl. 2π) parsed deterministically."""
        eng = self._engine()
        sup = _sup()
        try:
            eid = _register(eng, sup, "x^2+y^2")
            r1 = _run(eng, sup,
                      f"{eid} を x,y から r,theta へ "
                      "x=r*cos(theta), y=r*sin(theta) で変数変換して")
            r2 = _run(eng, sup, f"{r1['expr_id']} を簡約して")
            assert r2["canonical"].replace(" ", "") == "r^3"
            # ∫₀¹ r³ dr = 1/4
            r3 = _run(eng, sup, f"{r2['expr_id']} を r で 0 から 1 まで定積分して")
            assert r3["status"] == "completed", r3
            assert r3["operation"] == "integrate_definite"
            assert r3["canonical"].replace(" ", "") == "1/4"
            # ∫₀^{2π} (1/4) dθ = π/2 — unicode 2π bound normalized to %pi
            r4 = _run(eng, sup, f"{r3['expr_id']} を theta で 0 から 2π まで定積分して")
            assert r4["status"] == "completed", r4
            assert r4["canonical"].replace(" ", "") in ("%pi/2", "(%pi)/2")
        finally:
            nb.stop_engine()

    def test_iterated_integral_display_is_clean(self):
        """Display of an iterated definite integral must not nest the inner
        integral's equation, must render `d\\theta` (not `dtheta`), and the 2π
        bound as `2\\pi` (not the Maxima `%pi`)."""
        eng = self._engine()
        sup = _sup()
        try:
            eid = _register(eng, sup, "x^2+y^2")
            r1 = _run(eng, sup,
                      f"{eid} を x,y から r,theta へ "
                      "x=r*cos(theta), y=r*sin(theta) で変数変換して")
            r2 = _run(eng, sup, f"{r1['expr_id']} を簡約して")
            r3 = _run(eng, sup, f"{r2['expr_id']} を r で 0 から 1 まで定積分して")
            r4 = _run(eng, sup, f"{r3['expr_id']} を theta で 0 から 2π まで定積分して")
            tex = eng.expressions[r4["expr_id"]].latex
            assert tex == r"\int_{0}^{2\pi} \frac{1}{4} \, d\theta = \frac{\pi}{2}", tex
        finally:
            nb.stop_engine()

    def test_definite_integral_literal_and_tex_pi(self):
        """Definite integral on a literal integrand, with a TeX-π bound:
        ∫₀^{π} sin(theta) dθ = 2."""
        eng = self._engine()
        sup = _sup()
        try:
            r = _run(eng, sup, r"sin(theta) を theta で 0 から \pi まで定積分して")
            assert r["status"] == "completed", r
            assert r["canonical"].replace(" ", "") == "2"
        finally:
            nb.stop_engine()

    def test_definite_integral_english(self):
        """Same definite integral driven in English (def_integral EN pattern,
        session-language lookup): integrate sin(theta) over theta from 0 to π."""
        nb.start_engine(title="defint en", mode="standard", language="en",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, r"integrate sin(theta) over theta from 0 to \pi")
            assert r["status"] == "completed", r
            assert r["operation"] == "integrate_definite"
            assert r["canonical"].replace(" ", "") == "2"
        finally:
            nb.stop_engine()
