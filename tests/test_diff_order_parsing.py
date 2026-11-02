"""Tests for differentiation-order parsing in the interpret pipeline.

Covers the rule-based extraction of "N階/N次/N回微分" → order=N, var, expr,
dispatched to run_operation("differentiate", ...) without invoking the LLM.

Regression: "N回微分" (回 counter) previously was not recognized as an order
clause, so "E1 を k で 2 回微分して" fell through to the LLM and failed with
"Missing required fields: ['var']".
"""
import asyncio
from unittest.mock import MagicMock

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


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
    step.visible_action = "differentiate"
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
    # These differentiate by k (not a time parameter) on non-time expressions, so
    # the ∂-vs-d choice is unambiguous; a real engine returns False here.
    engine.differentiate_is_ambiguous.return_value = False
    return engine


def _make_supervisor() -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup._v = OperationValidator()
    sup.needs_expression_injection.return_value = False
    return sup


def _run(engine, supervisor, text: str) -> dict:
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


class TestDiffOrderParsing:
    def _dispatch(self, text: str, lang: str = "ja") -> tuple[dict, MagicMock]:
        e1 = _make_expression("E1", "hbar^2*k^2/(2*m) - mu", "\\xi_k = ...")
        e_out = _make_expression("E2", "hbar^2/m")
        engine = _make_engine({"E1": e1, "E2": e_out})
        engine.lang = lang
        out_step = _make_step(["E2"])
        engine.run_operation.return_value = (out_step, "hbar^2/m")
        sup = _make_supervisor()
        resp = _run(engine, sup, text)
        return resp, engine

    def test_kai_counter_extracts_order_and_var(self):
        # "2 回微分" must be recognized as order=2 with var=k, dispatched rule-based.
        resp, engine = self._dispatch("E1 を k で 2 回微分して")
        assert resp["status"] == "completed"
        assert resp["operation"] == "differentiate"
        args = engine.run_operation.call_args[0][1]
        assert args["expr"] == "E1"
        assert args["var"] == "k"
        assert int(args["order"]) == 2

    def test_kai_counter_no_space(self):
        resp, engine = self._dispatch("E1 を k で 2回微分して")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        assert int(args["order"]) == 2

    def test_kanji_kai_counter(self):
        resp, engine = self._dispatch("E1 を k で 三回微分して")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        assert int(args["order"]) == 3

    def test_kai_counter_does_not_call_llm(self):
        e1 = _make_expression("E1", "hbar^2*k^2/(2*m) - mu")
        e_out = _make_expression("E2", "hbar^2/m")
        engine = _make_engine({"E1": e1, "E2": e_out})
        engine.run_operation.return_value = (_make_step(["E2"]), "hbar^2/m")
        sup = _make_supervisor()
        _run(engine, sup, "E1 を k で 2 回微分して")
        sup.interpret.assert_not_called()

    def test_single_diff_still_order_one(self):
        # "微分して" with no counter → order omitted (treated as first derivative).
        resp, engine = self._dispatch("E1 を k で微分して")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        # order is only injected when > 1
        assert "order" not in args or int(args["order"]) == 1

    def test_diff_var_not_alias_expanded(self):
        # Regression: once k is solved (E3: k = …), k is registered in
        # lhs_alias_map (k → E3).  Differentiating "by k" must keep var="k",
        # NOT expand it to E3 (which produced d/dE3 instead of d/dk).
        e1 = _make_expression("E1", "hbar^2*k^2/(2*m) - mu", "\\xi_k = ...")
        e3 = _make_expression("E3", "sqrt(2*m*mu)/hbar", "k = ...")
        e_out = _make_expression("E5", "hbar^2/m")
        engine = _make_engine({"E1": e1, "E3": e3, "E5": e_out})
        engine.lhs_alias_map = {"k": "E3"}   # k solved → alias to E3
        engine.run_operation.return_value = (_make_step(["E5"]), "hbar^2/m")
        sup = _make_supervisor()
        resp = _run(engine, sup, "E1 を k で 2 回微分して")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        assert args["var"] == "k"        # not "E3"
        assert int(args["order"]) == 2

    def test_diff_var_alias_single_order(self):
        e1 = _make_expression("E1", "hbar^2*k^2/(2*m) - mu", "\\xi_k = ...")
        e3 = _make_expression("E3", "sqrt(2*m*mu)/hbar", "k = ...")
        e_out = _make_expression("E5", "hbar^2*k/m")
        engine = _make_engine({"E1": e1, "E3": e3, "E5": e_out})
        engine.lhs_alias_map = {"k": "E3"}
        engine.run_operation.return_value = (_make_step(["E5"]), "hbar^2*k/m")
        sup = _make_supervisor()
        resp = _run(engine, sup, "E1 を k で微分して")
        assert resp["status"] == "completed"
        assert engine.run_operation.call_args[0][1]["var"] == "k"

    # Regression: EID/var matched via the English _DIFF_RULE_RE_EN pattern, but
    # the order was phrased as an ordinal ("3rd order") rather than "N times" /
    # "twice" (the only English forms interpret_pipeline.py's upstream
    # _ORDER_CLAUSE_RE_EN recognizes).  Previously this silently defaulted to
    # order=1 instead of falling through to the LLM or erroring.
    def test_en_ordinal_order_extracted(self):
        resp, engine = self._dispatch(
            "differentiate E1 with respect to k, 3rd order", lang="en")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        assert args["expr"] == "E1"
        assert args["var"] == "k"
        assert int(args["order"]) == 3

    def test_en_times_counter_extracts_order_and_var(self):
        # "N times" phrasing is handled upstream (interpret_pipeline.py
        # _ORDER_CLAUSE_RE_EN); confirm it still dispatches rule-based end-to-end.
        resp, engine = self._dispatch("differentiate E1 with respect to k 2 times", lang="en")
        assert resp["status"] == "completed"
        assert resp["operation"] == "differentiate"
        args = engine.run_operation.call_args[0][1]
        assert args["expr"] == "E1"
        assert args["var"] == "k"
        assert int(args["order"]) == 2

    def test_en_single_diff_still_order_one(self):
        resp, engine = self._dispatch("differentiate E1 with respect to k", lang="en")
        assert resp["status"] == "completed"
        args = engine.run_operation.call_args[0][1]
        assert "order" not in args or int(args["order"]) == 1
