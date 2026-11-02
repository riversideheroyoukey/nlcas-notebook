"""Inequality constraints as first-class notebook Expressions.

A constraint like "2*x_A + x_B <= 100" must register with the real
relational symbol (\\leq) in its display latex — not silently collapse into
an equation (the pre-existing shared_parser.py bug this feature also fixes:
_find_top_level_eq only guarded against := and ==, not <=/>=, so "<=" was
silently read as "=" and the "<" dropped).

Covers: LaTeX \\leq input, undefined-symbol scan on BOTH sides (the
regression test for the _scan_undefined fix — the old RHS-only scan would
have missed an undefined LHS symbol), and the end-to-end NL path in English
(the paper case-study's actual session language) plus a Japanese smoke test.
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


def test_latex_leq_input_registers_with_backslash_leq():
    nb.start_engine(title="ineq_latex", mode="standard", language="en",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        r = _run(eng, sup, r"x_A + 2 x_B \leq 80")
        assert r["status"] == "registered"
        expr_id = r["registered_expressions"][0]["expr_id"]
        expr = eng.expressions[expr_id]
        assert "<=" in expr.canonical
        assert r"\leq" in expr.latex
    finally:
        nb.stop_engine()


def test_undefined_symbol_scan_hits_both_sides():
    """Regression test for the _scan_undefined fix (maxima_backend_handler.py):
    an inequality with an undefined symbol only on the LHS must still be
    reported — the old "scan RHS only" behavior (correct for H-1's named
    definitions) would silently miss it here, since neither side of a
    relation is a definition."""
    nb.start_engine(title="ineq_undef", mode="standard", language="en",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        # register "abc" as a known symbol; leave "xyz" undefined
        _run(eng, sup, "abc = 50")
        r = _run(eng, sup, "xyz <= abc")
        assert r["status"] == "registered"
        expr_id = r["registered_expressions"][0]["expr_id"]
        assert "xyz" in eng.expressions[expr_id].undefined_symbols
    finally:
        nb.stop_engine()


def test_english_session_registers_or_case_study_constraints():
    """The paper's actual session mode: English, plain-text objective +
    two inequality constraints, all via the NL pipeline."""
    nb.start_engine(title="ineq_en_case_study", mode="standard", language="en",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        r1 = _run(eng, sup, "Z = 40*x_A + 30*x_B")
        assert r1["status"] == "registered"

        r2 = _run(eng, sup, "2*x_A + x_B <= 100")
        assert r2["status"] == "registered"
        e2 = eng.expressions[r2["registered_expressions"][0]["expr_id"]]
        assert e2.canonical == "2*x_A + x_B <= 100"
        assert r"\leq" in e2.latex

        r3 = _run(eng, sup, "x_A + 2*x_B <= 80")
        assert r3["status"] == "registered"
        e3 = eng.expressions[r3["registered_expressions"][0]["expr_id"]]
        assert r"\leq" in e3.latex
    finally:
        nb.stop_engine()


def test_japanese_session_still_registers_inequality():
    """Defense-in-depth smoke test: the feature is symbol-based (<=, \\leq
    are language-neutral), not phrase-based, so there is no separate ja
    regex to rot — but confirm a ja-language session still registers
    correctly and the intercept doesn't misfire."""
    nb.start_engine(title="ineq_ja", mode="standard", language="ja",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        r = _run(eng, sup, "2*x_A + x_B <= 100")
        assert r["status"] == "registered"
        expr_id = r["registered_expressions"][0]["expr_id"]
        assert r"\leq" in eng.expressions[expr_id].latex
    finally:
        nb.stop_engine()


def test_dedup_returns_existing_expression():
    nb.start_engine(title="ineq_dedup", mode="standard", language="en",
                    domains=["supply_chain_logistics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["supply_chain_logistics"],
                            primary_backend="maxima")
        r1 = _run(eng, sup, "2*x_A + x_B <= 100")
        r2 = _run(eng, sup, "2*x_A + x_B <= 100")
        assert (r1["registered_expressions"][0]["expr_id"]
                == r2["registered_expressions"][0]["expr_id"])
    finally:
        nb.stop_engine()
