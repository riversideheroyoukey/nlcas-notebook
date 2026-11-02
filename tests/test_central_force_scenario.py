"""End-to-end regression for scenario S-h (central-force conserved quantities)
and the multi-token substitution-RHS parsing it relies on.

Runs the exact natural-language command sequence of the saved example
(domains/classical_mechanics/examples/scenario_central_force_conservation)
through the rule-based pipeline (no LLM) and asserts that differentiating each
conserved quantity and substituting the equations of motion cancels to 0:

  - angular momentum  L_z = m r² θ̇        → dL_z/dt = 0
  - energy            E   = ½m(ṙ²+r²θ̇²)-k/r → dE/dt   = 0

This guards both the classical-mechanics dot-notation form unification and the
substitute short-circuit's multi-token RHS capture ("-2 \\dot{r} \\dot{theta} / r"),
which previously stopped at the first space and dropped the replacement.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, eng, sup))


def test_central_force_conserved_quantities():
    nb.start_engine(title="test_central_force", mode="standard",
                    language="ja", domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        setup = [
            "r を時間依存関数として定義する",
            "theta を時間依存関数として定義する",
            "m と k をスカラーとする",
        ]
        for c in setup:
            assert _run(eng, sup, c)["status"] in ("confirmed", "declared")

        # Angular momentum conservation.
        assert _run(eng, sup, r"L_z = m r^2 \dot{theta}")["status"] == "registered"
        assert _run(eng, sup, "E1 を t で全微分して")["status"] == "completed"
        r3 = _run(eng, sup, r"E2 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して")
        assert r3["status"] == "completed"
        assert r3["canonical"] == "0"          # dL_z/dt = 0

        # Energy conservation (Kepler potential -k/r).
        assert _run(eng, sup,
                    r"E_tot = (1/2) m (\dot{r}^2 + r^2 \dot{theta}^2) - k / r"
                    )["status"] == "registered"
        assert _run(eng, sup, "E4 を t で全微分して")["status"] == "completed"
        assert _run(eng, sup,
                    r"E5 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して"
                    )["status"] == "completed"
        r7 = _run(eng, sup,
                  r"E6 の \ddot{r} を r \dot{theta}^2 - k / (m r^2) に置換して")
        assert r7["status"] == "completed"
        assert r7["canonical"] == "0"          # dE/dt = 0
    finally:
        nb.stop_engine()
