"""AM-1 (残④): characteristic polynomial of an INLINE H-1 matrix.

The item was filed when `A = matrix([…])` + charpoly generated a broken Maxima
command and `define_matrix` was the only working route.  The matrix-literal work
(`normalize_matrix_literal`: `[a, b; c, d]` → `matrix([a,b],[c,d])`) fixed it;
these tests pin that the inline route stays working, since nothing else covered
charpoly-on-an-H-1-matrix end to end.

Maxima's `charpoly(M, x)` is det(M − xI), so a 2×2 comes back as
x²−tr·x+det and an odd-sized matrix carries the overall minus sign.
"""
import asyncio

import pytest

from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.llm_supervisor import LLMSupervisor

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _session():
    nb.start_engine(title="test_am1_charpoly", mode="standard", language="ja",
                    domains=["analytical_mechanics"])
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["analytical_mechanics"], primary_backend="maxima")
    return nb.get_engine(), sup


@pytest.mark.parametrize("register,expected", [
    # 2x2 numeric: eigenvalues 1 and 3
    ("A = [2, 1; 1, 2]",                 "x^2-4*x+3"),
    # explicit matrix(...) call form
    ("B = matrix([2,1],[1,2])",          "x^2-4*x+3"),
    # 3x3 tridiagonal: eigenvalues 2, 2±sqrt(2)  → -(x-2)(x^2-4x+2)
    ("C = [2, 1, 0; 1, 2, 1; 0, 1, 2]",  "-x^3+6*x^2-10*x+4"),
])
def test_inline_matrix_charpoly(register, expected):
    eng, sup = _session()
    try:
        assert _run(eng, sup, register)["status"] == "registered"
        res = _run(eng, sup, "E1 の特性多項式を x について求めて")
        assert res["status"] == "completed", res
        assert res["canonical"].replace(" ", "") == expected
    finally:
        nb.stop_engine()


def test_symbolic_matrix_charpoly():
    """[a, b; c, d] → x²−(a+d)x+(ad−bc)."""
    eng, sup = _session()
    try:
        _run(eng, sup, "D = [a, b; c, d]")
        res = _run(eng, sup, "E1 の特性多項式を x について求めて")
        assert res["status"] == "completed", res
        flat = res["canonical"].replace(" ", "")
        assert flat == "x^2-d*x-a*x+a*d-b*c"
    finally:
        nb.stop_engine()


def test_command_is_well_formed_maxima():
    """The AM-1 symptom was a BROKEN generated command, so pin its shape."""
    eng, sup = _session()
    try:
        _run(eng, sup, "A = [2, 1; 1, 2]")
        res = _run(eng, sup, "E1 の特性多項式を x について求めて")
        step = eng.steps[res["step_id"]]
        assert step.rendered_command == "expand(charpoly(matrix([2, 1], [1, 2]),x))"
    finally:
        nb.stop_engine()


def test_reference_by_name_works_too():
    """`A の特性多項式を…` (H-1 alias) must reach the same command as `E1 の…`."""
    eng, sup = _session()
    try:
        _run(eng, sup, "A = [2, 1; 1, 2]")
        res = _run(eng, sup, "A の特性多項式を x について求めて")
        assert res["status"] == "completed", res
        assert res["canonical"].replace(" ", "") == "x^2-4*x+3"
    finally:
        nb.stop_engine()


def test_charpoly_then_solve_gives_normal_mode_frequencies():
    """The workflow AM-1 was blocking: coupled-oscillation stiffness matrix
    [[2k,-k],[-k,2k]] → charpoly → solve → eigenvalues k and 3k."""
    eng, sup = _session()
    try:
        _run(eng, sup, "K = [2*k, -k; -k, 2*k]")
        res = _run(eng, sup, "E1 の特性多項式を lam について求めて")
        assert res["canonical"].replace(" ", "") == "lam^2-4*k*lam+3*k^2"
        _run(eng, sup, "E2 を lam について解いて")
        roots = {e.canonical.replace(" ", "") for e in eng.expressions.values()}
        assert "k" in roots and "3*k" in roots
    finally:
        nb.stop_engine()
