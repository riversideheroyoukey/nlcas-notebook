r"""Substituting SEVERAL equations into one target, in one instruction.

open_items TRANSFORM-SUBST-CHAIN.  「E1, E2 を E3 に代入して」 always failed from the
second source on, and the reason shown named the wrong thing:

    p = u a + v b   /   g = w a   /   H = p + g
    → 失敗「代入できません: g が対象の式に含まれていません」

Measured: the message is TRUE about the text and WRONG about the work.  H-1's P2
binds every definition CAS-side, so the FIRST `subst` expands all of them at
once — step 1 already returned the complete `a*w+b*v+a*u`.  Step 2 then looked
for `g`, correctly did not find it, and the chain reported FAILED **while the
right answer sat registered in front of the researcher**.

Two things had to exist before the chain could act on that:

* a way to tell THIS refusal from any other.  The only distinguishing thing was
  the researcher-facing message, which is translated — keying on it means
  matching Japanese prose.  `BuildResult.reason_code` / `OperationRefused` carry
  a stable identifier instead, set only where a caller has a reason to look.
* a way to tell "already applied" from "never applied".  The running target
  cannot say: the symbol is missing in both cases.  The ORIGINAL target can, and
  that is what the chain now asks.

The second test below is the one that keeps this honest — a source that never
applied to the target still fails, because suppressing that would turn a real
mistake into a silent no-op, which is the shape of defect this codebase spends
most of its time removing.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InputToken, InterpretRequest, run_interpret_pipeline)


@pytest.fixture
def session():
    nb.start_engine(title="chain", mode="standard", language="ja",
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    try:
        yield eng, sup
    finally:
        nb.stop_engine()


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or (r.get("registered_expressions") or [{}])[-1].get("expr_id")


CHAIN = {"ja": "{a}, {b} を {t} に代入して",
         "en": "substitute {a}, {b} into {t}"}


class TestTwoSourcesReachTheTarget:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_both_definitions_are_applied(self, lang):
        nb.start_engine(title=f"chain_{lang}", mode="standard", language=lang,
                        domains=["classical_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        try:
            a = _eid(_run(eng, sup, "p = u a + v b", "math"))
            b = _eid(_run(eng, sup, "g = w a", "math"))
            t = _eid(_run(eng, sup, "H = p + g", "math"))
            r = _run(eng, sup, CHAIN[lang].format(a=a, b=b, t=t))
            assert r["status"] == "completed", r
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            for gone in ("p+g", "+g", "p+"):
                assert gone not in canon, (gone, canon)
            for part in ("a*w", "b*v", "a*u"):
                assert part in canon, (part, canon)
        finally:
            nb.stop_engine()

    def test_the_label_still_names_the_target(self, session):
        """The chain's result is the target's, so it keeps the target's name."""
        eng, sup = session
        a = _eid(_run(eng, sup, "p = u a + v b", "math"))
        b = _eid(_run(eng, sup, "g = w a", "math"))
        t = _eid(_run(eng, sup, "H = p + g", "math"))
        r = _run(eng, sup, CHAIN["ja"].format(a=a, b=b, t=t))
        assert eng.expressions[_eid(r)].latex.startswith("H = "), r


class TestNeverAppliedIsStillAnError:
    """"Already done" and "never there" look identical in the running target.

    Only the ORIGINAL target separates them, and it must: a source the
    researcher named that has nothing to do with the target is a mistake worth
    reporting, and turning it into a silent no-op would be exactly the class of
    defect the substitution guard exists to catch.
    """

    def test_a_source_absent_from_the_original_target_fails(self, session):
        eng, sup = session
        a = _eid(_run(eng, sup, "p = u a + v b", "math"))
        b = _eid(_run(eng, sup, "z = q r", "math"))
        t = _eid(_run(eng, sup, "H = p + y", "math"))
        r = _run(eng, sup, CHAIN["ja"].format(a=a, b=b, t=t))
        assert r["status"] == "failed", r
        assert "z" in r["error"], r["error"]


class TestACompoundLeftSideIsSubstituted:
    r"""A summation on the left is matched by keeping it a NOUN.

    One root, two faces, both measured 2026-08-15 and both reported as success:
    Maxima EVALUATES `sum(c(i),i,1,3)` the moment it reads it, so

    * `subst(sum(...) = rhs, target)` never matches — the pattern is three terms
      by then — and the target came back with its sums merely EXPANDED,
      `c(3)+d(2)+c(2)+d(1)+c(1)`, no substitution done;
    * `subst(i=m, sum(c(i),i,1,3))` has no `i` left, so `relabel_index` returned
      the expanded sum whether or not the index it was given even existed.

    Quoting the aggregate (`'sum(...)`) is the form both need.  **Routing to
    `ratsubst` was tried first and MEASURED WRONG**: it matches algebraically, so
    a pattern that is only partly present gets SOLVED FOR ONE TERM —
    `H = c_1 + y` with `\sum_{i=1}^{3} c_i = u a + v b` became
    `H = y + b v + a u - c_3 - c_2`, terms the researcher never wrote, as a
    success.  Two existing tests were pinning that spelling and caught it.
    """

    def test_two_summations_are_replaced_by_their_values(self, session):
        eng, sup = session
        a = _eid(_run(eng, sup, r"\sum_{i=1}^{3} c_i = u a + v b", "math"))
        b = _eid(_run(eng, sup, r"\sum_{j=1}^{2} d_j = w a", "math"))
        t = _eid(_run(eng, sup,
                      r"H = \sum_{i=1}^{3} c_i + \sum_{j=1}^{2} d_j", "math"))
        r = _run(eng, sup, CHAIN["ja"].format(a=a, b=b, t=t))
        assert r["status"] == "completed", r
        canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
        for part in ("a*w", "b*v", "a*u"):
            assert part in canon, (part, canon)
        for term in ("c(1)", "c(3)", "d(2)", "sum("):
            assert term not in canon, (term, canon)

    def test_a_summation_absent_from_the_target_is_still_refused(self, session):
        r"""The case that killed the `ratsubst` route.

        Refusing is the right answer, and the wrong answer it protects against
        is named here so a future attempt recognises it.
        """
        eng, sup = session
        a = _eid(_run(eng, sup, r"\sum_{i=1}^{3} c_i = u a + v b", "math"))
        t = _eid(_run(eng, sup, "H = c_1 + y", "math"))
        r = _run(eng, sup, f"{a} を {t} に代入して")
        assert r["status"] == "failed", r
        canon = " ".join(e.canonical for e in eng.expressions.values())
        assert "c(3)" not in canon and "c(2)" not in canon, canon

    def test_the_partial_step_is_readable(self, session):
        """The intermediate keeps the sum that has not been substituted yet.

        Quoting is what makes a chain legible step by step: before, every sum in
        the target expanded on the first command whether or not it was the one
        being replaced.
        """
        eng, _sup = session
        e = eng.register_expression(r"H = \sum_{i=1}^3*c_i + \sum_{j=1}^2*d_j",
                                    latex=r"H = \sum_{i=1}^{3} c_i + \sum_{j=1}^{2} d_j")
        step, _ = eng.run_operation("substitute", {
            "expr": e.display_id, "lhs": r"\sum_{i=1}^3*c_i", "rhs": "u*a + v*b"})
        canon = eng.expressions[step.output_refs[0]].canonical.replace(" ", "")
        assert "'sum(d(j),j,1,2)" in canon, canon
        assert "d(1)" not in canon, canon


class TestRelabelIndex:
    """The other face of the same root, plus the guard it never had."""

    def test_a_bound_index_is_actually_renamed(self, session):
        eng, _sup = session
        e = eng.register_expression(r"\sum_{i=1}^3*c_i",
                                    latex=r"\sum_{i=1}^{3} c_i")
        step, _ = eng.run_operation("relabel_index", {
            "expr": e.display_id, "old_index": "i", "new_index": "m"})
        canon = eng.expressions[step.output_refs[0]].canonical.replace(" ", "")
        assert canon == "'sum(c(m),m,1,3)", canon

    def test_an_index_that_is_not_there_is_refused(self, session):
        """It used to answer exactly as if the rename had worked."""
        from notebook.engine import OperationRefused
        eng, _sup = session
        e = eng.register_expression(r"\sum_{i=1}^3*c_i",
                                    latex=r"\sum_{i=1}^{3} c_i")
        with pytest.raises(OperationRefused) as exc:
            eng.run_operation("relabel_index", {
                "expr": e.display_id, "old_index": "zzz", "new_index": "m"})
        assert exc.value.reason_code == "lhs_not_found"

    def test_the_operand_is_lowered(self, session):
        r"""The researcher's `\sum_{i=1}^{3}*c_i` reached Maxima verbatim, could
        not be read, and the adapter's scratch name `_res_` was registered as
        the RESULT (BUILDER-OPERAND-NOT-LOWERED)."""
        eng, _sup = session
        e = eng.register_expression(r"\sum_{i=1}^3*c_i",
                                    latex=r"\sum_{i=1}^{3} c_i")
        step, _ = eng.run_operation("relabel_index", {
            "expr": e.display_id, "old_index": "i", "new_index": "m"})
        assert "\\sum" not in (step.rendered_command or ""), step.rendered_command
        assert "_res_" not in eng.expressions[step.output_refs[0]].canonical


class TestTheReasonIsCarriedAsACode:
    """The chain must not read the message: it is translated.

    Pinned at the exception, so that a refusal growing a second reason cannot
    quietly reuse this code, and so that a translation change cannot break the
    chain's behaviour.
    """

    def test_the_refusal_carries_lhs_not_found(self):
        from notebook.engine import OperationRefused
        nb.start_engine(title="code", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        try:
            e = eng.register_expression("H = p + y", latex="H = p + y")
            with pytest.raises(OperationRefused) as exc:
                eng.run_operation("substitute",
                                  {"expr": e.display_id, "lhs": "z", "rhs": "q*r"})
            assert exc.value.reason_code == "lhs_not_found", exc.value.reason_code
        finally:
            nb.stop_engine()

    def test_an_ordinary_refusal_carries_no_code(self):
        """The default stays None — a code is added when a caller needs one."""
        from notebook.runtime.backend_handler import BuildResult
        assert BuildResult(ok=False, command="", error="x").reason_code is None
