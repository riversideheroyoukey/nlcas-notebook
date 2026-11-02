"""Tests for the Context-Aware Supervisor pipeline behavior.

Covers the pending_dependencies flow (Bug 1 fix):
- symbol_updates applied pre-flight to SymbolContext
- depends() injections deferred until user confirms (not sent to Maxima pre-flight)
- needs_confirmation response includes pending_dependencies
- OperationRequest.pending_dependencies triggers declare_depends on confirm
- invalid Quantity.kind in symbol_updates is skipped silently
"""
import asyncio
from unittest.mock import MagicMock, patch, call

import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor, InterpretResult
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from ui.notebook_bridge import OperationRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Text that matches no rule-based early-return in the pipeline
_UNRESOLVABLE = "Can you expand E1 in a different way"


def _make_pipeline_engine() -> MagicMock:
    """Minimal mock engine that satisfies run_interpret_pipeline."""
    engine = MagicMock()
    engine.active_branch_id = "branch_1"
    engine.steps = {}
    engine.branches = {}
    engine.expressions = {}
    engine.symbol_context = SymbolContext()
    engine.quantity_registry = {}
    engine._active_expr_ids.return_value = set()
    # **本物の validator を持たせる。** 2026-09-10 以降、パイプラインは
    # 「この文は操作を名指しているか」をスキーマの別名で訊く
    # （`operand_boundary.text_names_an_operation`）。MagicMock の属性は
    # MagicMock なので `max_alias_words` が数にならず、走査が
    # `'<' not supported between 'int' and 'MagicMock'` で落ちる——
    # docstring が言う「run_interpret_pipeline を満たす」が満たせなくなる。
    engine._validator = OperationValidator()
    return engine


def _make_mock_supervisor(result: InterpretResult) -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup.interpret.return_value = result
    sup._v = OperationValidator()
    return sup


def _llm_result(**kwargs) -> InterpretResult:
    defaults = dict(
        ok=True,
        operation="expand",
        args={"expr": "E1"},
        requires_confirmation=True,
        confidence=0.9,
        source="llm",
        confirmation_prompt="expand E1?",
        symbol_updates=[],
        dependencies=[],
    )
    defaults.update(kwargs)
    return InterpretResult(**defaults)


def _run_pipeline(engine, supervisor, text=_UNRESOLVABLE) -> dict:
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


def _make_bridge_engine() -> MagicMock:
    """Minimal mock engine for bridge run_operation tests."""
    engine = MagicMock()
    engine.expressions = {}
    engine.quantity_registry = {}
    engine.symbol_context = SymbolContext()
    mock_step = MagicMock()
    mock_step.output_refs = []
    mock_step.step_id = "step_test"
    mock_step.visible_action = "展開する"
    engine.run_operation.return_value = (mock_step, "x+1")
    return engine


# ---------------------------------------------------------------------------
# A-6  symbol_updates applied pre-flight (before user confirms)
# ---------------------------------------------------------------------------

class TestSymbolUpdatesPreFlight:
    def test_symbol_updates_applied_before_confirmation(self):
        engine = _make_pipeline_engine()
        result = _llm_result(
            symbol_updates=[{"name": "L", "kind": "function"}],
            dependencies=[{"dependent": "L", "independent": ["q", "t"]}],
        )
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        engine.register_quantity.assert_called_once_with("L", kind="function")

    def test_symbol_updates_with_natural_vars_kwarg_mapping(self):
        engine = _make_pipeline_engine()
        result = _llm_result(
            symbol_updates=[{"name": "H", "kind": "function", "dependencies": ["p", "q"]}],
        )
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        engine.register_quantity.assert_called_once_with(
            "H", kind="function", natural_vars=["p", "q"]
        )

    def test_empty_symbol_updates_no_register_call(self):
        engine = _make_pipeline_engine()
        result = _llm_result(symbol_updates=[])
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        engine.register_quantity.assert_not_called()


# ---------------------------------------------------------------------------
# A-7  dependencies NOT sent to Maxima before confirmation
# ---------------------------------------------------------------------------

class TestDependenciesDeferred:
    def test_dependencies_NOT_sent_to_maxima_before_confirmation(self):
        engine = _make_pipeline_engine()
        result = _llm_result(
            dependencies=[{"dependent": "L", "independent": ["q", "t"]}],
        )
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        engine.declare_depends.assert_not_called()

    def test_multiple_dependencies_all_deferred(self):
        engine = _make_pipeline_engine()
        result = _llm_result(
            dependencies=[
                {"dependent": "L", "independent": ["q", "q_dot"]},
                {"dependent": "T", "independent": ["q_dot"]},
            ],
        )
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        engine.declare_depends.assert_not_called()


# ---------------------------------------------------------------------------
# A-8  needs_confirmation response includes pending_dependencies
# ---------------------------------------------------------------------------

class TestPendingDepsInResponse:
    def test_pending_dependencies_key_present(self):
        engine = _make_pipeline_engine()
        deps = [{"dependent": "L", "independent": ["q", "t"]}]
        result = _llm_result(dependencies=deps)
        sup = _make_mock_supervisor(result)
        response = _run_pipeline(engine, sup)
        assert response["status"] == "needs_confirmation"
        assert "pending_dependencies" in response

    def test_pending_dependencies_value_matches_llm_output(self):
        engine = _make_pipeline_engine()
        deps = [
            {"dependent": "L", "independent": ["q", "q_dot", "t"]},
            {"dependent": "T", "independent": ["q_dot"]},
        ]
        result = _llm_result(dependencies=deps)
        sup = _make_mock_supervisor(result)
        response = _run_pipeline(engine, sup)
        assert response["pending_dependencies"] == deps

    def test_empty_dependencies_gives_empty_pending(self):
        engine = _make_pipeline_engine()
        result = _llm_result(dependencies=[])
        sup = _make_mock_supervisor(result)
        response = _run_pipeline(engine, sup)
        assert response["pending_dependencies"] == []


# ---------------------------------------------------------------------------
# A-9  OperationRequest.pending_dependencies → declare_depends on confirm
# ---------------------------------------------------------------------------

class TestPendingDepsAppliedOnConfirm:
    def test_single_dep_applied(self):
        from ui.notebook_bridge import run_operation
        engine = _make_bridge_engine()
        req = OperationRequest(
            operation="expand",
            args={"expr": "E1"},
            pending_dependencies=[{"dependent": "L", "independent": ["q", "t"]}],
        )
        with patch("ui.notebook_bridge.get_engine", return_value=engine):
            asyncio.run(run_operation(req))
        engine.declare_depends.assert_called_once_with("L", ["q", "t"])

    def test_multiple_deps_all_applied(self):
        from ui.notebook_bridge import run_operation
        engine = _make_bridge_engine()
        req = OperationRequest(
            operation="expand",
            args={"expr": "E1"},
            pending_dependencies=[
                {"dependent": "L", "independent": ["q", "q_dot"]},
                {"dependent": "T", "independent": ["q_dot"]},
            ],
        )
        with patch("ui.notebook_bridge.get_engine", return_value=engine):
            asyncio.run(run_operation(req))
        assert engine.declare_depends.call_count == 2
        engine.declare_depends.assert_any_call("L", ["q", "q_dot"])
        engine.declare_depends.assert_any_call("T", ["q_dot"])

    def test_dep_missing_independent_skipped(self):
        from ui.notebook_bridge import run_operation
        engine = _make_bridge_engine()
        req = OperationRequest(
            operation="expand",
            args={"expr": "E1"},
            pending_dependencies=[{"dependent": "L", "independent": []}],
        )
        with patch("ui.notebook_bridge.get_engine", return_value=engine):
            asyncio.run(run_operation(req))
        engine.declare_depends.assert_not_called()


# ---------------------------------------------------------------------------
# A-10  empty pending_dependencies → declare_depends never called
# ---------------------------------------------------------------------------

class TestEmptyPendingDeps:
    def test_empty_list_no_declare_call(self):
        from ui.notebook_bridge import run_operation
        engine = _make_bridge_engine()
        req = OperationRequest(
            operation="expand",
            args={"expr": "E1"},
            pending_dependencies=[],
        )
        with patch("ui.notebook_bridge.get_engine", return_value=engine):
            asyncio.run(run_operation(req))
        engine.declare_depends.assert_not_called()

    def test_omitted_field_defaults_to_empty(self):
        req = OperationRequest(operation="expand", args={"expr": "E1"})
        assert req.pending_dependencies == []


# ---------------------------------------------------------------------------
# A-11  invalid Quantity.kind in symbol_updates → silently skipped
# ---------------------------------------------------------------------------

class TestInvalidKindSkipped:
    def test_invalid_kind_does_not_crash_pipeline(self):
        engine = _make_pipeline_engine()

        def _raise_on_invalid(name, **kwargs):
            kind = kwargs.get("kind", "anonymous")
            if kind not in ("anonymous", "scalar", "function", "conjugate"):
                raise ValueError(f"Invalid kind: {kind!r}")

        engine.register_quantity.side_effect = _raise_on_invalid
        result = _llm_result(
            symbol_updates=[{"name": "bad", "kind": "variable"}],
        )
        sup = _make_mock_supervisor(result)
        response = _run_pipeline(engine, sup)
        # Pipeline must not crash and must still return the confirmation response
        assert response["status"] == "needs_confirmation"

    def test_valid_entry_processed_after_invalid(self):
        engine = _make_pipeline_engine()

        def _raise_on_invalid(name, **kwargs):
            kind = kwargs.get("kind", "anonymous")
            if kind not in ("anonymous", "scalar", "function", "conjugate"):
                raise ValueError(f"Invalid kind: {kind!r}")

        engine.register_quantity.side_effect = _raise_on_invalid
        result = _llm_result(
            symbol_updates=[
                {"name": "bad", "kind": "variable"},
                {"name": "good", "kind": "function"},
            ],
        )
        sup = _make_mock_supervisor(result)
        _run_pipeline(engine, sup)
        registered_names = [c.args[0] for c in engine.register_quantity.call_args_list]
        assert "bad" in registered_names
        assert "good" in registered_names


class TestCandidateDedupHashableArgs:
    """Regression: confirmation candidates whose args carry a list value (e.g.
    a subscript-classification choice list) must not crash the candidate-dedup
    frozenset.  Declaring `J_z をスカラーとする` previously raised
    `TypeError: cannot use 'tuple' as a set element (unhashable type: 'list')`
    at the interpret-pipeline dedup."""

    def test_list_valued_candidate_args_do_not_crash(self):
        engine = _make_pipeline_engine()
        list_args = {"name": "J_z", "choices": ["component_index", "name_part"]}
        result = _llm_result(
            operation="declare_scalar",
            args=dict(list_args),
            candidates=[{"operation": "declare_scalar", "args": dict(list_args)}],
        )
        sup = _make_mock_supervisor(result)
        # No explicit E-id in the text → the dedup branch runs.
        out = _run_pipeline(engine, sup, text="J_z をスカラーとする")
        assert out is not None  # reached here ⇒ no unhashable-set TypeError
