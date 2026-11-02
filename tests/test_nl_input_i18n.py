"""English (input-NL i18n) parsing of operations, mirroring the Japanese forms.

The NL grammar patterns live in i18n/nl_patterns.json (named-group regex per
language).  These tests drive the SAME pipeline with English commands and check
that each operation is interpreted rule-based (no LLM), exactly as the generator's
English scenario runs do.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _engine(domain):
    nb.start_engine(title="i18n", mode="standard", language="en", domains=[domain])
    return nb.get_engine(), LLMSupervisor(
        validator=OperationValidator(), llm_client=None, domains=[domain],
        primary_backend="maxima")


def _run(eng, sup, value):
    tok = InputToken(type="text", value=value)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _last(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestMatrixOpsEnglish:
    """scenario_matrix_diagonalization driven in English."""

    def test_diagonalization_workflow(self):
        eng, sup = self._eng()
        try:
            _run(eng, sup, "A = [2, 1; 1, 2]")                       # E1
            r = _run(eng, sup, "Find the eigenvalues of E1")
            assert r["operation"] == "eigenvalues"
            r = _run(eng, sup, "Find the eigenvectors of E1")
            assert r["operation"] == "eigenvectors"
            r = _run(eng, sup, "Diagonalize E1")                     # E4 = P
            assert r["operation"] == "diagonalize"
            r = _run(eng, sup, "Find the diagonal matrix of E1")     # E5 = diag(3,1)
            assert r["operation"] == "diagonal_matrix"
            r = _run(eng, sup, "Find the inverse of E4")             # E6 = P^-1
            assert r["operation"] == "inverse_matrix"
            r = _run(eng, sup, "Find the matrix product of E6 and E1")   # E7 = P^-1 A
            assert r["operation"] == "matrix_multiply"
            r = _run(eng, sup, "Find the matrix product of E7 and E4")   # E8 = P^-1 A P
            assert r["operation"] == "matrix_multiply"
            # P^-1 A P = diag(3, 1)
            c = eng.expressions[_last(r)].canonical.replace(" ", "")
            assert "3" in c and "0" in c
        finally:
            nb.stop_engine()

    def _eng(self):
        return _engine("linear_algebra")


class TestIntegralWorkflowEnglish:
    """scenario_coordinate_jacobian (change-of-variables) driven in English."""

    def test_polar_disk(self):
        eng, sup = _engine("electromagnetism")
        try:
            r = _run(eng, sup, "Definite-integrate x^2+y^2 over the region x^2+y^2<=1")
            assert r["operation"] == "register_integral"
            r = _run(eng, sup, "Change variables in E1 from x,y to r,theta "
                               "via x=r*cos(theta), y=r*sin(theta)")
            assert r["operation"] == "change_integration_variables"
            r = _run(eng, sup, "Evaluate this definite integral")
            assert r["operation"] == "evaluate_integral"
            # ∬_disk (x²+y²) dA = π/2
            assert "pi" in eng.expressions[_last(r)].latex.replace("\\", "")
        finally:
            nb.stop_engine()


class TestRotatingFrameEnglish:
    """scenario_rotating_frame (NMR/ESR) driven in English — general spin ops."""

    def test_rotating_frame_workflow(self):
        eng, sup = _engine("magnetic_resonance")
        try:
            _run(eng, sup, r"H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1")    # E1
            r = _run(eng, sup, r"H_{rot} = E1 + \omega S^z_1")               # E2 (add frame term)
            assert r["operation"] == "spin"
            assert r"\left( \omega - \omega_0 \right) S^{z}_{1}" in r["latex"]
            r = _run(eng, sup, "Find the effective field of E2")              # E3
            assert r["operation"] == "spin" and r"\sqrt" in r["latex"]
            r = _run(eng, sup, "Find the commutator of E2 and S^z_1")         # E4
            assert r["operation"] == "spin"
            assert r["latex"] == r"[H_{rot}, S^z_1] = i \omega_1 S^{y}_{1}"
        finally:
            nb.stop_engine()
