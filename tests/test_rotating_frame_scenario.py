"""S-m end-to-end: single-spin rotating frame (NMR/ESR), rebuilt on GENERAL
single-spin operations (no canned per-scenario derivation).

Register a custom spin Hamiltonian, transform to the rotating frame, take the
effective field, and take the precession-EOM commutator — each a general
spin_algebra primitive (parse_spin_expression / commutator / coeff_vector_norm
/ commutator).
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


def _engine():
    nb.start_engine(title="s_m", mode="standard", language="ja",
                    domains=["magnetic_resonance"])
    return nb.get_engine(), LLMSupervisor(
        validator=OperationValidator(), llm_client=None,
        domains=["magnetic_resonance"], primary_backend="maxima")


class TestRotatingFrameGeneralOps:
    """The worked NMR/ESR session as a sequence of general single-spin operations,
    each its own command / YOU card producing one equation."""

    def test_workflow(self):
        eng, sup = _engine()
        try:
            r1 = _run(eng, sup, r"H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1")
            assert eng.expressions[r1["expr_id"]].latex == \
                r"H_{lab} = -\omega_0 S^{z}_{1} - \omega_1 S^{x}_{1}"
            r2 = _run(eng, sup, r"H_{rot} = E1 + \omega S^z_1")   # add the frame term
            assert eng.expressions[r2["expr_id"]].latex == \
                r"H_{rot} = \left( \omega - \omega_0 \right) S^{z}_{1} - \omega_1 S^{x}_{1}"
            r3 = _run(eng, sup, r"E2 の有効磁場を求めて")
            assert eng.expressions[r3["expr_id"]].latex == \
                r"\omega_{\mathrm{eff}} = \sqrt{ \left(\omega - \omega_0\right)^2 + \omega_1^2 }"
            r4 = _run(eng, sup, r"E2 と S^z_1 の交換子を求めて")
            assert eng.expressions[r4["expr_id"]].latex == \
                r"[H_{rot}, S^z_1] = i \omega_1 S^{y}_{1}"
            assert len({r1["expr_id"], r2["expr_id"],
                        r3["expr_id"], r4["expr_id"]}) == 4
        finally:
            nb.stop_engine()
