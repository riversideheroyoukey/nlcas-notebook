""""En を P とする" / "call En P" — bind a new named variable to an existing
expression's value (_try_name_assignment / _register_named_alias in
ui/interpret_pipeline.py).

Regression: _NAME_ASSIGN_RE was Japanese-only; there is no schema "rename"
operation, so English imperative phrasing ("call E3 P") had no LLM fallback
either — completely unreachable in English (equation-form "P = E3" already
worked via the separate H-1 path, so only imperative phrasing was broken).
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestNameAssignment:
    def test_ja_imperative(self):
        nb.start_engine(title="na1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "a x + b", "math"))
            r = _run(eng, sup, f"{e1} を P とする")
            assert r["status"] == "registered"
            new = eng.expressions[_eid(r)]
            assert new.latex.startswith("P =")
            assert new.canonical.replace(" ", "") == "a*x+b" or "a" in new.canonical
        finally:
            nb.stop_engine()

    def test_en_call(self):
        nb.start_engine(title="na2", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "a x + b", "math"))
            r = _run(eng, sup, f"call {e1} P")
            assert r["status"] == "registered"
            new = eng.expressions[_eid(r)]
            assert new.latex.startswith("P =")
        finally:
            nb.stop_engine()

    def test_en_name(self):
        nb.start_engine(title="na3", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "a x + b", "math"))
            r = _run(eng, sup, f"name {e1} P")
            assert r["status"] == "registered"
            new = eng.expressions[_eid(r)]
            assert new.latex.startswith("P =")
        finally:
            nb.stop_engine()

    def test_en_store_as(self):
        nb.start_engine(title="na4", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "a x + b", "math"))
            r = _run(eng, sup, f"store {e1} as P")
            assert r["status"] == "registered"
            new = eng.expressions[_eid(r)]
            assert new.latex.startswith("P =")
        finally:
            nb.stop_engine()
