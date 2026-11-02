r"""An expression reference brings the expression's own placeholder range.

EXPR-REF-PROVENANCE cause B.  A sum written without a range is given a
placeholder pair once, at registration, and the pair lives in the IR — the
canonical stays the researcher's own writing (`\sum_i*x_i`, no range at all).
Two paths then need a CAS view of that expression, and they answered
differently (measured 2026-08-17)::

    combine_expressions(E1, E2, "-")   →  0
    「E1 - E2」                          →  'sum(x(j),j,nbSumLo2,nbSumHi2)
                                            - 'sum(x(i),i,nbSumLo1,nbSumHi1)

Same two expressions, same subtraction.  E2 is E1's own simplification and
already carries E1's pair, because a CAS result's canonical IS what the CAS
returned.  What E1 could not do was carry its own: spliced as raw text by
`_expand_expr_ids`, it re-parsed inside the NEW expression and `BoundLedger`
handed it a fresh pair.  The identity of the expression the text came from was
dropped at the splice.

**The ticket called this a missing decision.  It was not.**  The decision was
already made and implemented — `placeholder_sum_bounds.cas_view`, on the
operation path — and the reference path simply did not ask it.  Both now go
through `cas_view_of`.

What is NOT asserted: that two separately written rangeless sums are equal.
The ledger is keyed per expression, so `\sum_i x_i` and `\sum_i y_i` keep
different pairs.  (Two IDENTICAL spellings cannot even coexist — registering
the same canonical twice returns the same expression.)
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or (r.get("registered_expressions") or [{}])[-1].get("expr_id")


class TestASourceCancelsAgainstItsOwnResult:
    def setup_method(self):
        nb.start_engine(title="provb", mode="standard", language="ja", domains=[])
        self.eng, self.sup = nb.get_engine(), _sup()

    def teardown_method(self):
        nb.stop_engine()

    def _simplify(self, eid):
        out = _run(self.eng, self.sup, f"{eid} を簡約して")
        return self.eng.expressions[_eid(out)].canonical.strip()

    @pytest.mark.parametrize("source", [
        r"\sum_i x_i",
        r"\sum_i \sum_j A_{i,j}",
        r"\sum_i (x_i + y_i)",
    ], ids=["simple", "nested", "compound_body"])
    def test_the_text_path_reaches_zero(self, source):
        e1 = _eid(_run(self.eng, self.sup, source, "math"))
        e2 = _eid(_run(self.eng, self.sup, f"{e1} を簡約して"))
        e3 = _eid(_run(self.eng, self.sup, f"{e1} - {e2}", "math"))
        assert self._simplify(e3) == "0"

    def test_the_two_paths_agree(self):
        r"""The property is not "the text path works" but "the two answers are
        the same" — that is what was broken."""
        e1 = _eid(_run(self.eng, self.sup, r"\sum_i x_i", "math"))
        e2 = _eid(_run(self.eng, self.sup, f"{e1} を簡約して"))
        step, _ = self.eng.run_operation(
            "combine_expressions", {"expr_a": e1, "expr_b": e2, "op": "-"})
        via_operation = self.eng.expressions[step.output_refs[0]].canonical.strip()
        e3 = _eid(_run(self.eng, self.sup, f"{e1} - {e2}", "math"))
        via_text = self._simplify(e3)
        assert via_operation == via_text == "0", (via_operation, via_text)

    def test_the_source_keeps_the_researchers_own_writing(self):
        r"""Only the SPLICE uses the CAS view.  E1's own canonical is still
        what was typed — the rule that a registered expression's canonical is
        the researcher's writing is untouched."""
        e1 = _eid(_run(self.eng, self.sup, r"\sum_i x_i", "math"))
        assert self.eng.expressions[e1].canonical == r"\sum_i*x_i"


class TestItStillAssertsNoEqualityNobodyStated:
    def setup_method(self):
        nb.start_engine(title="provb2", mode="standard", language="ja", domains=[])
        self.eng, self.sup = nb.get_engine(), _sup()

    def teardown_method(self):
        nb.stop_engine()

    def test_two_different_rangeless_sums_do_not_cancel(self):
        e1 = _eid(_run(self.eng, self.sup, r"\sum_i x_i", "math"))
        e2 = _eid(_run(self.eng, self.sup, r"\sum_i y_i", "math"))
        e3 = _eid(_run(self.eng, self.sup, f"{e1} - {e2}", "math"))
        out = _run(self.eng, self.sup, f"{e3} を簡約して")
        got = self.eng.expressions[_eid(out)].canonical
        assert got.strip() != "0"
        assert "x(i)" in got and "y(i)" in got, got

    def test_they_carry_different_placeholder_pairs(self):
        e1 = _eid(_run(self.eng, self.sup, r"\sum_i x_i", "math"))
        e2 = _eid(_run(self.eng, self.sup, r"\sum_i y_i", "math"))
        first = self.eng.placeholder_cas_view(e1)
        second = self.eng.placeholder_cas_view(e2)
        assert first and second
        assert "nbSumLo1" in first and "nbSumLo1" not in second, (first, second)

    def test_registering_the_same_sum_twice_is_one_expression(self):
        r"""Why the "two independently written IDENTICAL sums" case cannot be
        constructed: the canonical is the key, so the second registration
        returns the first expression."""
        _run(self.eng, self.sup, r"\sum_i x_i", "math")
        _run(self.eng, self.sup, r"\sum_i x_i", "math")
        assert len(self.eng.expressions) == 1, list(self.eng.expressions)


class TestTheDoorIsSharedWithTheOperationPath:
    def setup_method(self):
        nb.start_engine(title="provb3", mode="standard", language="ja", domains=[])
        self.eng, self.sup = nb.get_engine(), _sup()

    def teardown_method(self):
        nb.stop_engine()

    def test_an_expression_without_a_placeholder_gets_no_substitute(self):
        r"""`cas_view_of` returns None so the splice keeps the canonical — an
        ordinary expression must not start travelling as backend code."""
        e1 = _eid(_run(self.eng, self.sup, r"(x+1)^2", "math"))
        assert self.eng.placeholder_cas_view(e1) is None

    def test_an_unknown_id_is_not_an_error(self):
        assert self.eng.placeholder_cas_view("E999") is None

    def test_a_ranged_sum_needs_no_substitute(self):
        e1 = _eid(_run(self.eng, self.sup, r"\sum_{i=1}^{3} x_i", "math"))
        assert self.eng.placeholder_cas_view(e1) is None

    def test_both_callers_read_the_same_helper(self):
        r"""A ratchet: the two paths answered this differently once already."""
        import inspect
        from notebook.runtime import placeholder_sum_bounds as psb
        assert "cas_view_of" in inspect.getsource(psb.cas_view)
        import notebook.engine as eng_mod
        assert "cas_view_of" in inspect.getsource(
            eng_mod.NotebookEngine.placeholder_cas_view)
        import ui.interpret_common as ic
        assert "placeholder_cas_view" in inspect.getsource(ic._expand_expr_ids)
