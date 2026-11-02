"""H-1 registration must render its stored display LaTeX with real backslashes
for bare Greek names / sqrt() calls (J-9, scenario-narrative-notes review,
2026-07-08, item 11) — not the raw canonical-syntax text as typed.

Found via scenario_risk_pooling_safety_stock: "SS = z * sigma * sqrt(L)"
registered E1's latex as the literal, unconverted "z sigma sqrt(L)" (no
backslash for \\sigma or \\sqrt{L}), because the H-1 pipeline used the raw RHS
text directly whenever no E-id substitution occurred (ui/interpret_pipeline.py).
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, value):
    tok = InputToken(type="text", value=value)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def test_h1_rhs_with_bare_greek_and_sqrt_renders_with_backslashes():
    nb.start_engine(title="h1_display", mode="standard", language="en",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        r = _run(eng, sup, "SS = z * sigma * sqrt(L)")
        assert r["status"] == "registered"
        expr_id = r["registered_expressions"][0]["expr_id"]
        expr = eng.expressions[expr_id]
        assert expr.latex == r"SS = z \sigma \sqrt{L}"
    finally:
        nb.stop_engine()
