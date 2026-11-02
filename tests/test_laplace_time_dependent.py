"""Laplace transform of a time-dependent variable (control theory / S-e).

Regression: Maxima's `laplace` ignores `depends(q,t)` on a bare atom and treats q
as a constant (laplace(q,t,s) = q/s).  The engine now rewrites time-dependent
symbols to function form q(t) before the laplace build, so derivatives transform
correctly (s²Q(s) − s q(0) − q'(0)).
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["control_theory"], primary_backend="maxima")


def _run(eng, sup, value, ttype="text", canonical=None):
    tok = (InputToken(type="math", value=value, canonical=canonical)
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestLaplaceTimeDependent:
    def test_laplace_of_time_dependent_atom_is_a_function(self):
        nb.start_engine(title="lap1", mode="standard", language="ja",
                        domains=["control_theory"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            r1 = _run(eng, sup, "q", "math", "q")               # E1
            r = _run(eng, sup, f"{r1['registered_expressions'][-1]['expr_id']} "
                               "を t から s へラプラス変換して")
            # q(t) → laplace(q(t),t,s), NOT the constant q/s
            assert "laplace(q(t),t,s)" in r["canonical"]
            assert r["canonical"].replace(" ", "") != "q/s"
        finally:
            nb.stop_engine()

    def test_rlc_ode_laplace(self):
        """Differentiate q(t) twice, assemble L q'' + R q' + q/C, and Laplace it →
        the s-domain operator with genuine s²·laplace(q(t),t,s) terms."""
        nb.start_engine(title="lap2", mode="standard", language="ja",
                        domains=["control_theory"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _run(eng, sup, "q", "math", "q")["registered_expressions"][-1]["expr_id"]
            e2 = _run(eng, sup, f"{e1} を t で全微分して")["expr_id"]
            e3 = _run(eng, sup, f"{e2} を t で全微分して")["expr_id"]
            # second derivative is a genuine time derivative
            assert eng.expressions[e3].canonical.replace(" ", "") == "'diff(q,t,2)"
            ode = _run(eng, sup, f"L*{e3} + R*{e2} + {e1}/C", "math",
                       f"L*{e3} + R*{e2} + {e1}/C")["registered_expressions"][-1]["expr_id"]
            r = _run(eng, sup, f"{ode} を t から s へラプラス変換して")
            canon = r["canonical"].replace(" ", "")
            assert "s^2*'laplace(q(t),t,s)" in canon
            assert "'laplace(q(t),t,s)/C" in canon
        finally:
            nb.stop_engine()
