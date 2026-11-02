"""Tests for H-9: solve_equation / solve_system.

Covers:
- _parse_solve: single equation, system with/without explicit vars
- parse_solve_result: Maxima output parsing
- pipeline: solve dispatches run_operation, registers individual solutions
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


# ---------------------------------------------------------------------------
# Helpers shared with test_h5_rhs_eval
# ---------------------------------------------------------------------------

def _make_expression(eid: str, canonical: str, latex: str = "") -> MagicMock:
    expr = MagicMock()
    expr.display_id = eid
    expr.canonical = canonical
    expr.latex = latex or canonical
    expr.created_by_step = "step_1"
    expr.undefined_symbols = []
    return expr


def _make_step(out_ids: "list[str]") -> MagicMock:
    step = MagicMock()
    step.output_refs = out_ids
    step.step_id = "step_out"
    step.visible_action = "solve"
    return step


def _make_engine(expressions: dict) -> MagicMock:
    engine = MagicMock()
    engine.active_branch_id = "branch_1"
    engine.steps = {}
    engine.branches = {}
    engine.expressions = expressions
    engine.symbol_context = SymbolContext()
    engine.quantity_registry = {}
    engine.lhs_alias_map = {}
    engine.pending_op = None
    engine.lang = "ja"
    engine._active_expr_ids.return_value = set(expressions.keys())
    engine._handler = MagicMock()
    engine._handler.operands_provided.return_value = None
    engine._handler.latex_to_canonical.side_effect = lambda x: x
    engine._handler.normalize_display_latex.side_effect = lambda x: x
    return engine


def _make_supervisor() -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup._v = OperationValidator()
    sup.needs_expression_injection.return_value = False
    return sup


def _run(engine, supervisor, text: str) -> dict:
    from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


# ---------------------------------------------------------------------------
# Unit tests for _parse_solve
# ---------------------------------------------------------------------------

class TestParseSolve:
    def test_single_equation(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("E1 を m について解いて")
        assert r is not None
        op, args = r
        assert op == "solve_equation"
        assert args["equation"] == "E1"
        assert args["var"] == "m"

    def test_system_with_vars(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("E1 と E2 の連立方程式を x と y について解いて")
        assert r is not None
        op, args = r
        assert op == "solve_system"
        assert "E1" in args["equations"] and "E2" in args["equations"]
        assert "x" in args["vars"] and "y" in args["vars"]

    def test_system_without_vars(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("E1 と E2 の連立方程式を解いて")
        assert r is not None
        op, args = r
        assert op == "solve_system"
        assert args["vars"] == ""

    def test_three_equations(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("E1 と E2 と E3 の連立方程式を x と y と z について解いて")
        assert r is not None
        op, args = r
        assert op == "solve_system"
        assert all(e in args["equations"] for e in ("E1", "E2", "E3"))

    def test_no_match(self):
        from ui.interpret_pipeline import _parse_solve
        assert _parse_solve("E1 を展開して") is None
        assert _parse_solve("E1 の右辺を計算して") is None

    # Externalized to i18n/nl_patterns.json (所見G/(b) migration); EN needs
    # lang='en' explicitly now (session-lang-only lookup, not JA+EN try-both).
    def test_single_equation_en(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("solve E1 for m", lang="en")
        assert r is not None
        op, args = r
        assert op == "solve_equation"
        assert args["equation"] == "E1"
        assert args["var"] == "m"

    def test_system_with_vars_en(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("solve E1 and E2 for x and y", lang="en")
        assert r is not None
        op, args = r
        assert op == "solve_system"
        assert "E1" in args["equations"] and "E2" in args["equations"]
        assert "x" in args["vars"] and "y" in args["vars"]

    def test_system_without_vars_en(self):
        from ui.interpret_pipeline import _parse_solve
        r = _parse_solve("solve E1 and E2", lang="en")
        assert r is not None
        op, args = r
        assert op == "solve_system"
        assert args["vars"] == ""

    def test_no_match_en(self):
        from ui.interpret_pipeline import _parse_solve
        assert _parse_solve("expand E1", lang="en") is None


# ---------------------------------------------------------------------------
# Unit tests for parse_solve_result (Maxima output parsing)
# ---------------------------------------------------------------------------

class TestJuliaParseSolveResult:
    """Julia solve result format: 'x = val1, y = val2'"""

    def _parse(self, s: str):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        return JuliaBackendHandler.parse_solve_result(s)

    def test_single_var(self):
        pairs = self._parse("m = F/a")
        assert pairs == [("m", "F/a")]

    def test_two_vars(self):
        pairs = self._parse("c_k = u_k*alpha_k - v_k*alpha_mk, c_dag = v_k*alpha_k + u_k*alpha_mk")
        assert len(pairs) == 2
        assert pairs[0][0] == "c_k"
        assert pairs[1][0] == "c_dag"

    def test_with_nested_parens(self):
        pairs = self._parse("x = (a + b) / (a - b), y = 2*a / (a - b)")
        assert len(pairs) == 2
        assert pairs[0][0] == "x"
        assert "a + b" in pairs[0][1]

    def test_empty(self):
        assert self._parse("") == []

    def test_quoted_string(self):
        # Julia may return quoted strings
        pairs = self._parse('"m = F/a"')
        assert pairs == [("m", "F/a")]


class TestJuliaBuildSolve:
    """Julia build_command for solve_equation / solve_system."""

    def _make_handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from unittest.mock import MagicMock
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        h._builder = MagicMock()
        return h

    def test_solve_equation_command(self):
        h = self._make_handler()
        expr_lookup = {"E1": "F = m * a"}
        result = h._build_solve("solve_equation",
                                {"equation": "E1", "var": "m"},
                                expr_lookup)
        assert result.ok
        assert "_nb_solve_equation" in result.command
        assert '"F"' in result.command
        assert '"m * a"' in result.command
        assert '"m"' in result.command

    def test_solve_system_command(self):
        h = self._make_handler()
        expr_lookup = {
            "E1": "alpha_k = u_k * c_k + v_k * c_dag",
            "E2": "alpha_mk = - v_k * c_k + u_k * c_dag",
        }
        result = h._build_solve("solve_system",
                                {"equations": "E1, E2", "vars": "c_k, c_dag"},
                                expr_lookup)
        assert result.ok
        assert "solve_for" in result.command
        assert "~" in result.command
        assert "c_k" in result.command and "c_dag" in result.command

    def test_missing_equation_fails(self):
        h = self._make_handler()
        result = h._build_solve("solve_equation", {"equation": "E1", "var": "m"}, {})
        assert not result.ok  # E1 not in expr_lookup, can't extract LHS/RHS


class TestParseSolveResult:
    def _parse(self, s: str):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        return MaximaBackendHandler.parse_solve_result(s)

    def test_single_var(self):
        pairs = self._parse("[m = F/a]")
        assert pairs == [("m", "F/a")]

    def test_double_bracket(self):
        pairs = self._parse("[[x = 2, y = 1]]")
        assert ("x", "2") in pairs
        assert ("y", "1") in pairs

    def test_empty(self):
        assert self._parse("[]") == []
        assert self._parse("") == []

    def test_complex_rhs(self):
        pairs = self._parse("[c_k = u_k*alpha_k - v_k*alpha_mk]")
        assert pairs[0][0] == "c_k"
        assert "alpha_k" in pairs[0][1]

    def test_multiple_solution_sets(self):
        """A system with several roots returns a list of solution SETS; each set's
        (var, val) pairs must be parsed (regression: was over-stripping the outer
        brackets and splitting across sets — e.g. Lotka–Volterra fixed points)."""
        pairs = self._parse("[[x = 0, y = 0], [x = c/d, y = a/b]]")
        assert pairs == [("x", "0"), ("y", "0"), ("x", "c/d"), ("y", "a/b")]

    def test_three_solution_sets(self):
        pairs = self._parse("[[x = 1], [x = 2], [x = 3]]")
        assert pairs == [("x", "1"), ("x", "2"), ("x", "3")]


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestH9Pipeline:
    def test_single_equation_dispatches(self):
        e1 = _make_expression("E1", "F = m * a", "F = m a")
        e_raw = _make_expression("E_raw", "[m = F/a]")
        e_sol = _make_expression("E2", "F/a")

        expressions = {"E1": e1, "E_raw": e_raw, "E2": e_sol}
        engine = _make_engine(expressions)
        out_step = _make_step(["E_raw", "E2"])
        engine.run_operation.return_value = (out_step, "[m = F/a]")

        sup = _make_supervisor()
        response = _run(engine, sup, "E1 を m について解いて")

        assert response["status"] == "completed"
        assert response["operation"] == "solve_equation"
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "solve_equation"
        assert call_args[0][1]["equation"] == "E1"
        assert call_args[0][1]["var"] == "m"

    def test_system_with_vars_dispatches(self):
        e1 = _make_expression("E1", "alpha_k = u_k * c_k + v_k * c_dag")
        e2 = _make_expression("E2", "alpha_mk = - v_k * c_k + u_k * c_dag")
        e_out = _make_expression("E3", "[[c_k = ..., c_dag = ...]]")

        expressions = {"E1": e1, "E2": e2, "E3": e_out}
        engine = _make_engine(expressions)
        out_step = _make_step(["E3"])
        engine.run_operation.return_value = (out_step, "[[c_k=..., c_dag=...]]")

        sup = _make_supervisor()
        response = _run(engine, sup,
                        "E1 と E2 の連立方程式を c_k と c_dag について解いて")

        assert response["status"] == "completed"
        assert response["operation"] == "solve_system"
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "solve_system"
        assert "c_k" in call_args[0][1]["vars"]

    def test_system_without_vars_prompts(self):
        e1 = _make_expression("E1", "2*x + y = 5")
        e2 = _make_expression("E2", "x + y = 3")
        expressions = {"E1": e1, "E2": e2}
        engine = _make_engine(expressions)

        sup = _make_supervisor()
        response = _run(engine, sup, "E1 と E2 の連立方程式を解いて")

        # Should prompt for variables (pending_op)
        assert response["status"] == "needs_confirmation"
        assert engine.pending_op is not None
        assert engine.pending_op["operation"] == "solve_system"

    def test_supervisor_not_called(self):
        e1 = _make_expression("E1", "F = m * a", "F = m a")
        e_out = _make_expression("E2", "F/a")
        expressions = {"E1": e1, "E2": e_out}
        engine = _make_engine(expressions)
        out_step = _make_step(["E2"])
        engine.run_operation.return_value = (out_step, "[m=F/a]")

        sup = _make_supervisor()
        _run(engine, sup, "E1 を m について解いて")
        sup.interpret.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests for _sympy_canonical_to_latex
# ---------------------------------------------------------------------------

class TestSympyCanonicalToLatex:
    def _convert(self, s: str) -> str:
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        return JuliaBackendHandler._sympy_canonical_to_latex(s)

    def test_sqrt_simple(self):
        assert self._convert("sqrt(x)") == "\\sqrt{x}"

    def test_sqrt_with_mul(self):
        assert self._convert("sqrt(2*m*mu)") == "\\sqrt{2*m*\\mu}"

    def test_sqrt_over_hbar(self):
        r = self._convert("sqrt(2*m*mu)/hbar")
        assert r == "\\frac{\\sqrt{2*m*\\mu}}{\\hbar}"

    def test_no_sqrt_passthrough(self):
        assert self._convert("a + b") == "a + b"
        assert self._convert("F/m") == "F/m"

    def test_sqrt_nested(self):
        r = self._convert("sqrt(sqrt(x))")
        assert r == "\\sqrt{\\sqrt{x}}"

    def test_hbar_converted(self):
        r = self._convert("sqrt(x)/hbar")
        assert "\\hbar" in r
        assert "hbar" not in r.replace("\\hbar", "")


class TestNoSolutionIsAFailureNotAnExpression:
    r"""An empty solution list is not a result (S-C-MEASURE-RESIDUALS).

    Maxima answers `[]` when a system has no solution (or it cannot find one).
    Nothing parses out of that, so the pre-allocated slot kept the raw value and
    the timeline showed `E3 = \left[ \right]` — reported as SUCCESS, consuming
    an expression number, and referenceable by later steps as if it were a
    value.  Measured 2026-08-01 on the S-c Bogoliubov condition × normalisation,
    and reproduced on a plainly inconsistent `x+y=1, x+y=2`.
    """
    import ui.notebook_bridge as _nb

    def _session(self, lang="ja"):
        import ui.notebook_bridge as nb
        nb.start_engine(title="nosol", mode="standard", language=lang, domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        return eng, sup

    def _run(self, eng, sup, cmd):
        from ui.interpret_pipeline import (InterpretRequest, InputToken,
                                           run_interpret_pipeline)
        tok = InputToken(type="text", value=cmd)
        return asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))

    def test_inconsistent_system_fails_with_a_reason_jp(self):
        import ui.notebook_bridge as nb
        eng, sup = self._session("ja")
        try:
            self._run(eng, sup, "x + y = 1")
            self._run(eng, sup, "x + y = 2")
            r = self._run(eng, sup, "E1 と E2 の連立方程式を x, y について解いて")
            assert r["status"] == "failed", r
            assert "解" in str(r.get("error", ""))
            # and nothing was written down
            assert set(eng.expressions) == {"E1", "E2"}
            assert not any("[]" in (e.canonical or "")
                           for e in eng.expressions.values())
        finally:
            nb.stop_engine()

    def test_inconsistent_system_fails_with_a_reason_en(self):
        import ui.notebook_bridge as nb
        eng, sup = self._session("en")
        try:
            self._run(eng, sup, "x + y = 1")
            self._run(eng, sup, "x + y = 2")
            r = self._run(eng, sup, "Solve E1 and E2 for x and y")
            assert r["status"] == "failed", r
            assert "solution" in str(r.get("error", "")).lower()
            assert set(eng.expressions) == {"E1", "E2"}
        finally:
            nb.stop_engine()

    def test_solvable_system_is_unaffected(self):
        import ui.notebook_bridge as nb
        eng, sup = self._session("ja")
        try:
            self._run(eng, sup, "x + y = 3")
            self._run(eng, sup, "x - y = 1")
            r = self._run(eng, sup, "E1 と E2 の連立方程式を x, y について解いて")
            assert r["status"] == "completed", r
            vals = {e.canonical for e in eng.expressions.values()}
            assert "2" in vals and "1" in vals
        finally:
            nb.stop_engine()
