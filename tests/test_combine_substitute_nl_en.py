"""English NL for combine_expressions and rule-based substitute (所見G stage 5).

These grammars were Japanese-only; their EN forms REVERSE operand order
("subtract B from A" = A−B, "divide A by B" = A/B, "substitute B for A" =
replace A with B).  The parser assigns operands from named groups; these tests
pin every reversal end-to-end (pipeline) and at the pattern level so a future
edit can't silently swap them — the worst-case failure the audit warned about.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, text, ttype="text", value=None):
    tok = (InputToken(type="math", value=value, canonical=value)
           if ttype == "math" else InputToken(type="text", value=text))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


def _norm(s):
    return (s or "").replace(" ", "")


# --------------------------------------------------------------------------
# Pattern-level word-order pins (fast, exhaustive)
# --------------------------------------------------------------------------

class TestCombinePatternOrder:
    def test_en_operand_order(self):
        def slots(t):
            gd = nlp("combine", "en").search(t).groupdict()
            return {k: v for k, v in gd.items() if v}
        assert slots("add E1 and E2") == {"a_add": "E1", "b_add": "E2"}
        assert slots("multiply E1 and E2") == {"a_mul": "E1", "b_mul": "E2"}
        # reversal: "subtract E2 from E1" is E1 − E2, so a=E1, b=E2
        assert slots("subtract E2 from E1") == {"a_sub": "E1", "b_sub": "E2"}
        assert slots("divide E1 by E2") == {"a_div": "E1", "b_div": "E2"}

    def test_ja_operand_order_unchanged(self):
        m = nlp("combine", "ja").search("E1 から E2 を引いて")
        assert (m.group("a"), m.group("b")) == ("E1", "E2")


class TestSubstReplacePatternOrder:
    def _fields(self, t):
        m = nlp("subst_replace", "en").search(t)
        gd = m.groupdict()
        pick = lambda s: next((gd[k] for k in gd if k.startswith(s) and gd[k]), None)
        return pick("eid"), pick("lhs"), pick("rhs")

    def test_replace_with(self):
        assert self._fields("replace a with b in E1") == ("E1", "a", "b")
        assert self._fields("in E1, replace a with -2*x") == ("E1", "a", "-2*x")

    def test_substitute_for_reverses(self):
        # "substitute B for A" means replace A with B — lhs=a, rhs=b
        assert self._fields("substitute b for a in E1") == ("E1", "a", "b")
        assert self._fields("in E1, substitute sin(x) for a") == ("E1", "a", "sin(x)")


# --------------------------------------------------------------------------
# End-to-end EN pipeline (parser → operation → result)
# --------------------------------------------------------------------------

class TestCombineEnglishPipeline:
    def _setup(self):
        nb.start_engine(title="comb en", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        a = _eid(_run(eng, sup, None, "math", "x"))
        b = _eid(_run(eng, sup, None, "math", "y"))
        return eng, sup, a, b

    def test_subtract_from_order(self):
        eng, sup, a, b = self._setup()
        try:
            # "subtract E2 from E1" must be E1 − E2 = x − y, NOT y − x
            r = _run(eng, sup, f"subtract {b} from {a}")
            assert r["operation"] == "combine_expressions", r
            assert _norm(r["canonical"]) == "x-y"
        finally:
            nb.stop_engine()

    def test_divide_by_order(self):
        eng, sup, a, b = self._setup()
        try:
            r = _run(eng, sup, f"divide {a} by {b}")
            assert r["operation"] == "combine_expressions", r
            assert _norm(r["canonical"]) in ("x/y", "x*y^-1")
        finally:
            nb.stop_engine()

    def test_add_order(self):
        eng, sup, a, b = self._setup()
        try:
            r = _run(eng, sup, f"add {a} and {b}")
            assert _norm(r["canonical"]) in ("x+y", "y+x")
        finally:
            nb.stop_engine()


class TestSubstituteEnglishPipeline:
    def _setup(self):
        nb.start_engine(title="subst en", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        e = _eid(_run(eng, sup, None, "math", "a*t"))
        return eng, sup, e

    def test_replace_with(self):
        eng, sup, e = self._setup()
        try:
            r = _run(eng, sup, f"replace a with b in {e}")
            assert r["operation"] == "substitute", r
            assert _norm(r["canonical"]) == "b*t"
        finally:
            nb.stop_engine()

    def test_substitute_for_reverses(self):
        eng, sup, e = self._setup()
        try:
            # "substitute b for a" replaces a with b → b*t (NOT a→a)
            r = _run(eng, sup, f"substitute b for a in {e}")
            assert r["operation"] == "substitute", r
            assert _norm(r["canonical"]) == "b*t"
        finally:
            nb.stop_engine()
