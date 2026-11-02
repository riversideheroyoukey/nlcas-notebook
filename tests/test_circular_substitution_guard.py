"""The circular-substitution guard in maxima_backend_handler.py's
_build_substitute must only block a DERIVATIVE self-reference (the true
danger: q -> -'diff(q,t,2)/w^2, where the multi-variant substitution loop
chains subst(bare_q=RHS, subst('diff(q,t,2)=RHS, expr)) and each nested call
re-inserts a copy of the derivative for the other to match again, diverging).

Found via the S-w practice-loop scenario (risk-pooling safety stock,
2026-07-05): a plain algebraic self-reference like L -> n*L (no derivative
involved) was wrongly blocked too — Maxima's subst() is single-pass and
handles this correctly in one step (confirmed directly against a live
Maxima session). The guard was narrowed to only fire when the LHS appears
inside a 'diff(...) call in the RHS.
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


class TestPlainAlgebraicSelfReferenceAllowed:
    def test_linear_scaling_substitution_succeeds(self):
        """L -> n*L is a legitimate one-shot scaling substitution, not circular."""
        nb.start_engine(title="circ_ok", mode="standard", language="ja",
                        domains=["supply_chain_logistics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["supply_chain_logistics"],
                                primary_backend="maxima")
            _run(eng, sup, "SS = z * sigma * sqrt(L)")
            r = _run(eng, sup, "E1 の L に n * L を代入して")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical == "sqrt(L*n)*sigma*z"
        finally:
            nb.stop_engine()

    def test_english_linear_scaling_substitution_succeeds(self):
        nb.start_engine(title="circ_ok_en", mode="standard", language="en",
                        domains=["supply_chain_logistics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["supply_chain_logistics"],
                                primary_backend="maxima")
            _run(eng, sup, "SS = z * sigma * sqrt(L)")
            r = _run(eng, sup, "In E1, substitute n * L for L")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical == "sqrt(L*n)*sigma*z"
        finally:
            nb.stop_engine()


class TestDerivativeSelfReferenceStillBlocked:
    def test_q_solved_in_terms_of_its_own_second_derivative_is_blocked(self):
        """The true danger case: q = -q̈/ω² substituted back must still fail."""
        nb.start_engine(title="circ_danger", mode="standard", language="ja",
                        domains=["classical_mechanics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["classical_mechanics"],
                                primary_backend="maxima")
            _run(eng, sup, "q を時間依存関数として定義する")
            _run(eng, sup, r"\ddot{q} + w^2 * q = 0")
            r = _run(eng, sup, "E1 を q について解いて")
            assert r["status"] == "completed", r
            r2 = _run(eng, sup, r"E1 の q に E2 を代入して")
            assert r2["status"] == "failed"
            assert "循環代入" in r2["error"]
        finally:
            nb.stop_engine()
