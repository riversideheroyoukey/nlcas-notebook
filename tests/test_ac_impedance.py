"""S-n scenario: series RLC impedance Z = R + jωL + 1/(jωC), resonance ω=1/√(LC).

Guards docs/scenario_ac_impedance.md and the electrical_engineering convention
that `j` is the imaginary unit (C-4): in an EE Maxima session the real/imaginary
parts split correctly and Im Z = 0 solves to ω = ±1/√(LC).
"""
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


def _ee_session():
    e = NotebookEngine(title="s_n ac impedance", mode=SessionMode.standard_symbolic)
    e.start()
    e.session.domains = ["electrical_engineering"]
    e.symbol_context.domains = ["electrical_engineering"]
    return e


def test_real_and_imaginary_parts():
    with _ee_session() as e:
        Z = e.register_expression("Z = R + j*w*L + 1/(j*w*C)")
        _, re_z = e.run_operation("real_part", {"expr": Z.display_id})
        _, im_z = e.run_operation("imaginary_part", {"expr": Z.display_id})
        # real_part / imaginary_part act on the RHS value, not the equation
        # object (regression: realpart(Z = expr) previously mangled it to
        # "L j w + 1/(C j w) + R = R").
        assert re_z.replace(" ", "") == "R"                       # Re Z = R
        assert im_z.replace(" ", "") == "L*w-1/(C*w)"             # Im Z = ωL − 1/(ωC)


def test_resonance_solves_to_one_over_sqrt_lc():
    with _ee_session() as e:
        Z = e.register_expression("Z = R + j*w*L + 1/(j*w*C)")
        step, im_z = e.run_operation("imaginary_part", {"expr": Z.display_id})
        assert im_z.replace(" ", "") == "L*w-1/(C*w)"
        # Resonance is Im Z = 0.  solve_equation treats a bare expression as
        # "expr = 0", so the reactance can be solved directly — no need to first
        # form an explicit "... = 0" equation.
        _, sol = e.run_operation("solve_equation",
                                 {"equation": step.output_refs[0], "var": "w"})
        s = sol.replace(" ", "")
        # ω = ±1/√(LC)  (Maxima: sqrt(1/(C*L)))
        assert "w=sqrt(1/(C*L))" in s and "w=-sqrt(1/(C*L))" in s


# --- NL is rule-based (no LLM) for 実部/虚部 extraction, and the imaginary unit
# round-trips j → %i (compute) → j (display) ----------------------------------
import asyncio
import re as _re

from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _ee_pipeline():
    import ui.notebook_bridge as nb
    nb.start_engine(title="ac", mode="standard", language="ja",
                    domains=["electrical_engineering"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["electrical_engineering"], primary_backend="maxima")
    return nb, eng, sup


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


def test_real_imag_extraction_is_rule_based():
    """"実部を取り出して"/"虚部を求めて" (て-form) resolve WITHOUT the LLM."""
    nb, eng, sup = _ee_pipeline()
    try:
        e = _eid(_run(eng, sup, "Z = R + j w L + 1/(j w C)", "math"))
        for phrase, op, want in [
            (f"{e} の実部を取り出して", "real_part", "R"),
            (f"{e} の実部を求めて", "real_part", "R"),
            (f"{e} の虚部を取り出して", "imaginary_part", "L*w-1/(C*w)"),
            (f"{e} の虚部を求めて", "imaginary_part", "L*w-1/(C*w)"),
        ]:
            r = _run(eng, sup, phrase)
            assert r["operation"] == op, (phrase, r.get("operation"), r.get("error"))
            assert eng.expressions[_eid(r)].canonical.replace(" ", "") == want
    finally:
        nb.stop_engine()


def test_imaginary_unit_displays_as_j():
    """Round-trip: input j → %i (compute) → j (display).  A result that retains
    the imaginary unit shows j, not Maxima's bare i."""
    with _ee_session() as e:
        Z = e.register_expression("Z = R + j*w*L")
        step, _ = e.run_operation("complex_conjugate", {"expr": Z.display_id})
        out = e.expressions[step.output_refs[0]]
        assert "j" in out.latex
        # no bare imaginary "i" leaked into the display
        assert _re.search(r"(?<![A-Za-z\\])i(?![A-Za-z])", out.latex) is None


def test_imaginary_j_does_not_disturb_index_i():
    """In EE the imaginary unit is j; `i` is current/index.  An index/exponent i
    (x^i, V_i) must survive — only the imaginary unit becomes j."""
    with _ee_session() as e:
        U = e.register_expression("U = x^i + j*w")
        step, _ = e.run_operation("complex_conjugate", {"expr": U.display_id})
        out = e.expressions[step.output_refs[0]]
        c = out.canonical.replace(" ", "")
        assert "x^i" in c                 # i-th power preserved (i is NOT imaginary)
        assert "x^j" not in c
        assert "j*w" in c                 # imaginary unit is j
        assert "%i" not in c


def test_phasor_exponent_uses_j():
    """A phasor e^{jωt}: the imaginary unit in the exponent displays as j."""
    with _ee_session() as e:
        P = e.register_expression("P = e^{j*w*t}")
        step, _ = e.run_operation("complex_conjugate", {"expr": P.display_id})
        out = e.expressions[step.output_refs[0]]
        assert "j" in out.canonical and "%i" not in out.canonical
