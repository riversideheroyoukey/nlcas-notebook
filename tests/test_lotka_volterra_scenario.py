"""S-q end-to-end: Lotka–Volterra stability (no LLM).

Fixed points from the nullclines (two solution sets), then the Jacobian at the
interior equilibrium (c/d, a/b) has eigenvalues ±i√(ac) — a neutral center.
Guards the multi-solution-set solve fix and the rule-based path.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, cmd):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=cmd)]), eng, sup))


class TestLotkaVolterra:
    def test_fixed_points_and_neutral_center(self):
        nb.start_engine(title="s_q", mode="standard", language="ja",
                        domains=["mathematical_biology"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["mathematical_biology"], primary_backend="maxima")
        try:
            _run(eng, sup, "a*x - b*x*y = 0")
            _run(eng, sup, "d*x*y - c*y = 0")
            r = _run(eng, sup, "E1 と E2 の連立方程式を x と y について解いて")
            assert r["status"] == "completed", r
            # both fixed points registered: (0,0) and (c/d, a/b)
            vals = {e["canonical"] for e in r["registered_expressions"]}
            assert {"0", "c/d", "a/b"} <= vals

            for cmd in ("a*x - b*x*y を x で偏微分して",
                        "a*x - b*x*y を y で偏微分して",
                        "d*x*y - c*y を x で偏微分して",
                        "d*x*y - c*y を y で偏微分して"):
                _run(eng, sup, cmd)

            rm = _run(eng, sup, "[0, -b*c/d; a*d/b, 0]")
            mid = rm["registered_expressions"][-1]["expr_id"]
            re_ = _run(eng, sup, f"{mid} の固有値を求めて")
            assert re_["status"] == "completed", re_
            # eigenvalues ±√(-ac) = ±i√(ac): a neutral center
            canon = re_["canonical"].replace(" ", "")
            assert "sqrt(-(a*c))" in canon and "-sqrt(-(a*c))" in canon
        finally:
            nb.stop_engine()
