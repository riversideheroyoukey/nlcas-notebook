"""Partial-vs-total derivative disambiguation (general: math + physics).

When a plain "微分" (no 偏微分/全微分) is given and ∂ vs d may differ, the system
must not guess: it asks interactively and errors in non-interactive use.  The
criterion is general — after removing the differentiation variable and any
declared-constant (scalar) symbols, if another variable (free or dependent)
remains, the choice is ambiguous.  Declaring constants removes the ambiguity.
Time is NOT special-cased; the time-dependent declaration is just the CAS-level
way to state a dependence.
"""
import asyncio
from unittest.mock import MagicMock

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, eng, sup))


def _engine(llm, decls=()):
    nb.start_engine(title="test_diff_amb", mode="standard", language="ja", domains=[])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(),
                        llm_client=(MagicMock() if llm else None),
                        domains=[], primary_backend="maxima")
    for d in decls:
        _run(eng, sup, d)
    return eng, sup


class TestDifferentiateIsAmbiguous:
    """engine.differentiate_is_ambiguous — the criterion itself (criterion B)."""

    def test_free_variable_is_ambiguous(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "z = x^2 + y")             # y is free
            assert eng.differentiate_is_ambiguous("E1", "x") is True
        finally:
            nb.stop_engine()

    def test_single_variable_is_unambiguous(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "w = x^2")                 # only the diff variable
            assert eng.differentiate_is_ambiguous("E1", "x") is False
        finally:
            nb.stop_engine()

    def test_declared_constants_are_excluded(self):
        eng, sup = _engine(False, decls=["a と b をスカラーとする"])
        try:
            _run(eng, sup, "p = a x + b")             # a,b declared scalar constants
            assert eng.differentiate_is_ambiguous("E1", "x") is False
        finally:
            nb.stop_engine()

    def test_free_constants_are_ambiguous(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "p = a x + b")             # a,b free (undeclared)
            assert eng.differentiate_is_ambiguous("E1", "x") is True
        finally:
            nb.stop_engine()


class TestDifferentiatePipeline:
    """The NL pipeline: ask interactively, error in batch, honour explicit kind."""

    def test_interactive_asks(self):
        eng, sup = _engine(True)
        try:
            _run(eng, sup, "z = x^2 + y")
            r = _run(eng, sup, "E1 を x で微分して")
            assert r["status"] == "needs_confirmation"
            assert len(r["candidates"]) == 2
        finally:
            nb.stop_engine()

    def test_batch_errors(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "z = x^2 + y")
            r = _run(eng, sup, "E1 を x で微分して")
            assert r["status"] == "failed"
            assert "全微分" in r["error"]
        finally:
            nb.stop_engine()

    def test_explicit_total_gives_d(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "z = x^2 + y")
            r = _run(eng, sup, "E1 を x で全微分して")
            assert r["status"] == "completed"
            assert r"\frac{dz}{dx}" in r["latex"]
        finally:
            nb.stop_engine()

    def test_explicit_partial_gives_partial(self):
        eng, sup = _engine(False)
        try:
            _run(eng, sup, "z = x^2 + y")
            r = _run(eng, sup, "E1 を x で偏微分して")
            assert r["status"] == "completed"
            assert r"\partial" in r["latex"]
        finally:
            nb.stop_engine()
