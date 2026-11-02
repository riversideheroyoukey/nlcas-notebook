"""Tests for H-5: 「En の右辺を計算して」 (evaluate RHS of registered equation).

Covers:
- _parse_rhs_to_operation: commutator / diff / fallback
- run_interpret_pipeline: H-5 detection short-circuit before LLM
- H-5 dispatch for commutator RHS
- H-5 dispatch for unknown RHS (simplify fallback)
- No match when source expression has no "=" (not an equation)
"""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InterpretRequest,
    InputToken,
    run_interpret_pipeline,
    _parse_rhs_to_operation,
)


# ---------------------------------------------------------------------------
# Unit tests for _parse_rhs_to_operation
# ---------------------------------------------------------------------------

class TestParseRhsToOperation:
    def test_commutator(self):
        op, args = _parse_rhs_to_operation("commutator(E3, Sx_1)")
        assert op == "commutator"
        assert args["a"] == "E3"
        assert args["b"] == "Sx_1"

    def test_commutator_with_spaces(self):
        op, args = _parse_rhs_to_operation("commutator(E3,  Sx_1)")
        assert op == "commutator"
        assert args["b"] == "Sx_1"

    def test_diff_simple(self):
        op, args = _parse_rhs_to_operation("diff(m * l^2 * q_dot, t)")
        assert op == "differentiate"
        assert args["var"] == "t"
        assert "m" in args["expr"]

    def test_diff_with_order(self):
        op, args = _parse_rhs_to_operation("diff(q, t, 2)")
        assert op == "differentiate"
        assert args["order"] == "2"

    def test_fallback_returns_none(self):
        result = _parse_rhs_to_operation("m * l^2 * q_ddot + m * g * l * sin(q)")
        assert result is None

    def test_fallback_subtraction(self):
        result = _parse_rhs_to_operation("E3 - E4")
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests via run_interpret_pipeline
# ---------------------------------------------------------------------------

def _make_expression(eid: str, canonical: str, latex: str = "") -> MagicMock:
    expr = MagicMock()
    expr.display_id = eid
    expr.canonical = canonical
    expr.latex = latex or canonical
    expr.created_by_step = "step_1"
    expr.undefined_symbols = []
    return expr


def _make_step(out_id: str) -> MagicMock:
    step = MagicMock()
    step.output_refs = [out_id]
    step.step_id = "step_out"
    step.visible_action = "交換子"
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
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


class TestH5PipelineCommutator:
    """H-5 detects commutator RHS and dispatches run_operation("commutator", ...)."""

    def test_h1_expression_rhs_dispatched(self):
        """H-1 expression: canonical = RHS only, latex has LHS = RHS.

        Heisenberg EOM: -i dSx/dt = [H, Sx] registered via H-1.
        Canonical stores only 'commutator(E3, Sx_1)' (no '=').
        H-5 must detect the equation via the latex field.
        """
        e3 = _make_expression("E3", "- gamma * B * Sz_1", "H = -\\gamma B S_{z_1}")
        # H-1 style: canonical = RHS only, latex = full equation
        e5 = _make_expression(
            "E5",
            "commutator(E3, Sx_1)",           # canonical = RHS only (no "=")
            "-i \\frac{dS_x}{dt} = [H, S_x]", # latex = full equation
        )
        e6 = _make_expression("E6", "- 2 * B * gamma * i * Sy_1")
        expressions = {"E3": e3, "E5": e5, "E6": e6}
        engine = _make_engine(expressions)
        out_step = _make_step("E6")
        engine.run_operation.return_value = (out_step, "")
        supervisor = _make_supervisor()
        response = _run(engine, supervisor, "E5 の右辺を計算して")
        assert response["status"] == "completed"
        assert response["operation"] == "evaluate_rhs"
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "commutator"

    def test_commutator_rhs_dispatched(self):
        # E5 is the equation "-i * diff(Sx_1, t) = commutator(E3, Sx_1)"
        # E3 is H, E6 will be the commutator result
        e3 = _make_expression("E3", "- gamma * B * Sz_1", "H = -\\gamma B S_{z_1}")
        e5 = _make_expression(
            "E5",
            "- i * diff(Sx_1, t) = commutator(E3, Sx_1)",
        )
        e6 = _make_expression("E6", "- 2 * B * gamma * i * Sy_1")

        expressions = {"E3": e3, "E5": e5, "E6": e6}
        engine = _make_engine(expressions)

        # run_operation("commutator", ...) returns (step, "")
        out_step = _make_step("E6")
        engine.run_operation.return_value = (out_step, "")

        supervisor = _make_supervisor()

        response = _run(engine, supervisor, "E5 の右辺を計算して")

        assert response["status"] == "completed"
        assert response["operation"] == "evaluate_rhs"
        assert response["expr_id"] == "E6"
        # run_operation should have been called with commutator
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "commutator"

    def test_lhs_supervisor_not_called(self):
        """LLM supervisor must NOT be called for H-5 inputs."""
        e5 = _make_expression(
            "E5",
            "- i * diff(Sx_1, t) = commutator(E3, Sx_1)",
        )
        e3 = _make_expression("E3", "- gamma * B * Sz_1")
        e6 = _make_expression("E6", "0")
        expressions = {"E3": e3, "E5": e5, "E6": e6}
        engine = _make_engine(expressions)

        out_step = _make_step("E6")
        engine.run_operation.return_value = (out_step, "")

        supervisor = _make_supervisor()
        _run(engine, supervisor, "E5 の右辺を計算して")

        supervisor.interpret.assert_not_called()


class TestH5PipelineSimplifyFallback:
    """When RHS is a plain expression, simplify fallback is used."""

    def test_simplify_fallback_called(self):
        # EOM = E3 - E4 → RHS is an arithmetic expression after E-ID expansion
        rhs_canonical = "m * l^2 * q_ddot + m * g * l * sin(q)"
        e5 = _make_expression("E5", f"EOM = {rhs_canonical}")
        e_rhs = _make_expression("E_rhs", rhs_canonical)
        e_simplified = _make_expression("E6", "m * l * (l * q_ddot + g * sin(q))")

        expressions = {"E5": e5}
        engine = _make_engine(expressions)

        # register_expression returns the RHS expression
        engine.register_expression.return_value = e_rhs
        # run_operation("simplify", ...) returns the simplified result
        out_step = _make_step("E6")
        engine.expressions["E6"] = e_simplified
        engine.run_operation.return_value = (out_step, "")

        supervisor = _make_supervisor()
        response = _run(engine, supervisor, "E5 の右辺を計算して")

        assert response["status"] == "completed"
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "simplify"


class TestH7KeywordOperations:
    """H-7: 右辺を展開/整理/因数分解/有理式簡約 → explicit operation dispatch."""

    def _make_eq_engine(self, rhs_canonical: str) -> tuple:
        e1 = _make_expression("E1", f"F = {rhs_canonical}")
        e_out = _make_expression("E2", "result")
        expressions = {"E1": e1, "E2": e_out}
        engine = _make_engine(expressions)
        out_step = _make_step("E2")
        engine.run_operation.return_value = (out_step, "")
        engine.register_expression.return_value = _make_expression("E_rhs", rhs_canonical)
        engine.expressions["E_rhs"] = _make_expression("E_rhs", rhs_canonical)
        return engine, _make_supervisor()

    def _dispatched_op(self, engine) -> str:
        return engine.run_operation.call_args[0][0]

    def test_expand(self):
        engine, sup = self._make_eq_engine("(x + 1)^2")
        response = _run(engine, sup, "E1 の右辺を展開して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "expand"

    def test_factor(self):
        engine, sup = self._make_eq_engine("x^2 - 1")
        response = _run(engine, sup, "E1 の右辺を因数分解して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "factor"

    def test_simplify(self):
        engine, sup = self._make_eq_engine("x^2 / x")
        response = _run(engine, sup, "E1 の右辺を整理して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "simplify"

    def test_rational_simplify(self):
        engine, sup = self._make_eq_engine("(x^2 - 1) / (x - 1)")
        response = _run(engine, sup, "E1 の右辺を有理式簡約して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "rational_simplify"

    def test_yurika_alias(self):
        engine, sup = self._make_eq_engine("(x^2 - 1) / (x - 1)")
        response = _run(engine, sup, "E1 の右辺を有理化して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "rational_simplify"

    def test_keisan_still_uses_parse_rhs_logic(self):
        """「計算して」は _parse_rhs_to_operation ロジック（commutator 検出等）を使う。"""
        e3 = _make_expression("E3", "- gamma * B * Sz_1")
        e5 = _make_expression("E5", "commutator(E3, Sx_1)",
                              "-i \\frac{dS_x}{dt} = [H, S_x]")
        e6 = _make_expression("E6", "result")
        expressions = {"E3": e3, "E5": e5, "E6": e6}
        engine = _make_engine(expressions)
        out_step = _make_step("E6")
        engine.run_operation.return_value = (out_step, "")
        sup = _make_supervisor()
        response = _run(engine, sup, "E5 の右辺を計算して")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "commutator"

    def test_supervisor_not_called_for_h7(self):
        engine, sup = self._make_eq_engine("(x + 1)^2")
        _run(engine, sup, "E1 の右辺を展開して")
        sup.interpret.assert_not_called()

    # rhs_op externalized to i18n/nl_patterns.json (所見G/(b) migration); the
    # English pattern is only tried when the session is English (engine.lang).
    def test_expand_en(self):
        engine, sup = self._make_eq_engine("(x + 1)^2")
        engine.lang = "en"
        response = _run(engine, sup, "Expand the right-hand side of E1")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "expand"

    def test_factor_en(self):
        engine, sup = self._make_eq_engine("x^2 - 1")
        engine.lang = "en"
        response = _run(engine, sup, "Factor the right-hand side of E1")
        assert response["status"] == "completed"
        assert self._dispatched_op(engine) == "factor"


class TestH8BothSidesParse:
    """Unit tests for _parse_both_sides."""

    def test_divide(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺を m で割って")
        assert r is not None
        eid, op, operand = r
        assert eid == "E1" and op == "divide" and operand == "m"

    def test_multiply_wo(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺に 2 をかけて")
        assert r is not None
        _, op, operand = r
        assert op == "multiply" and operand == "2"

    def test_multiply_bai(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺を 2 倍して")
        assert r is not None
        _, op, operand = r
        assert op == "multiply" and operand == "2"

    def test_add(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺に V を足して")
        assert r is not None
        _, op, operand = r
        assert op == "add" and operand == "V"

    def test_subtract(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺から m * g を引いて")
        assert r is not None
        _, op, operand = r
        assert op == "subtract"
        assert "m" in operand and "g" in operand

    def test_simplify(self):
        from ui.interpret_pipeline import _parse_both_sides
        r = _parse_both_sides("E1 の両辺を整理して")
        assert r is not None
        _, op, operand = r
        assert op == "simplify" and operand is None

    def test_no_match(self):
        from ui.interpret_pipeline import _parse_both_sides
        assert _parse_both_sides("E1 を展開して") is None


class TestH8BothSidesPipeline:
    """H-8: both-sides operation dispatched through pipeline."""

    def test_divide_both_sides(self):
        e1 = _make_expression("E1", "F = m * a", "F = m a")
        e_rhs = _make_expression("E_rhs", "(m * a) / (m)")
        e_out = _make_expression("E2", "a")

        expressions = {"E1": e1}
        engine = _make_engine(expressions)
        engine.register_expression.return_value = e_rhs
        engine.expressions["E_rhs"] = e_rhs
        engine.expressions["E2"] = e_out

        out_step = _make_step("E2")
        engine.run_operation.return_value = (out_step, "")

        supervisor = _make_supervisor()
        response = _run(engine, supervisor, "E1 の両辺を m で割って")

        assert response["status"] == "completed"
        assert response["operation"] == "both_sides_op"
        # LHS should show F/m
        assert "F" in response["latex"]
        # Simplify must have been called
        call_args = engine.run_operation.call_args
        assert call_args[0][0] == "simplify"

    def test_supervisor_not_called(self):
        e1 = _make_expression("E1", "F = m * a", "F = m a")
        expressions = {"E1": e1}
        engine = _make_engine(expressions)
        e_rhs = _make_expression("E_rhs", "(m * a) / (m)")
        e_out = _make_expression("E2", "a")
        engine.register_expression.return_value = e_rhs
        engine.expressions["E_rhs"] = e_rhs
        engine.expressions["E2"] = e_out
        engine.run_operation.return_value = (_make_step("E2"), "")

        supervisor = _make_supervisor()
        _run(engine, supervisor, "E1 の両辺を m で割って")
        supervisor.interpret.assert_not_called()


class TestH5NotEquation:
    """When En is not an equation (no = in canonical), H-5 does not intercept."""

    def test_no_equation_falls_through_to_llm(self):
        e1 = _make_expression("E1", "m * l^2 * q")  # no "=" → not an equation
        expressions = {"E1": e1}
        engine = _make_engine(expressions)
        engine.lhs_alias_map = {}

        from notebook.runtime.llm_supervisor import InterpretResult
        sup = _make_supervisor()
        sup.interpret.return_value = InterpretResult(
            ok=True,
            operation="simplify",
            args={"expr": "E1"},
            requires_confirmation=True,
            confidence=0.9,
            source="llm",
            confirmation_prompt="simplify?",
            symbol_updates=[],
            dependencies=[],
        )
        response = _run(engine, sup, "E1 の右辺を計算して")
        # LLM was called because E1 is not an equation
        sup.interpret.assert_called_once()
