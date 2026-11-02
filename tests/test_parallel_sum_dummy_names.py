r"""Two sums written side by side keep the index the researcher wrote.

EXPR-REF-PROVENANCE, cause A.  The ticket said expression references do not
carry provenance, so `E1 - E2` does not cancel.  Measured 2026-08-17, that is
only half of it, and the half it names is not the half a researcher meets
first::

    \sum_i x_i - \sum_i x_i     →  did not reduce to 0
    E1 - E1                     →  did not reduce to 0

Neither needs a reference to anything.  The RANGES matched — both terms carried
`nbSumLo1` — and what differed was the DUMMY NAME: the alphatiser renamed the
second sum's `i` to `j`, and **Maxima's noun forms do not alpha-convert**::

    ratsimp('sum(x(i),i,a,b) - 'sum(x(i),i,a,b))   →  0
    ratsimp('sum(x(i),i,a,b) - 'sum(x(j),j,a,b))   →  unchanged
    is('sum(x(i),i,a,b) = 'sum(x(j),j,a,b))        →  false

Renaming is required when scopes NEST — the inner binder really does shadow the
outer one — and prevents nothing when they are PARALLEL, because a parallel
binder is never in scope inside its sibling.  `_alphatize` threaded ONE mutable
`occupied` set through the whole walk, so a sibling looked exactly like a
nested one.

**The same question already had two answers.**  `julia_build_mixin` passes
`skip_alphatize=True` at all seven of its call sites, noting that
`Σ_k H_kin + Σ_k H_pair` must keep one `k` "so that normal_form() treats them
as the same physical momentum" — while `tests/test_index_contraction.py`
required the rename for that very Hamiltonian.  The default now agrees with the
consumer that cared.  (`skip_alphatize` still differs for nested sums and
stays.)

Cause B — a result and its own source (`E1 - E2`) getting different placeholder
RANGES — is untouched here and is the decision the ticket actually names.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.ir import Add, IndexSum, IndexSymbol, IndexedSymbol, Mul, Symbol
from notebook.runtime.index_contractor import alphatize_indices
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


class TestARangelessSumCancelsAgainstItself:
    r"""End to end, through the CAS — the property is the VALUE, not the text."""

    @pytest.mark.parametrize("setup,target", [
        (None, r"\sum_i x_i - \sum_i x_i"),
        (r"\sum_i x_i", "E1 - E1"),
        (r"\sum_i x_i", "2 * E1 - E1 - E1"),
        (None, r"\sum_i \sum_j A_{i,j} - \sum_i \sum_j A_{i,j}"),
    ], ids=["written_twice", "referenced_twice", "doubled", "nested_pair"])
    def test_it_reduces_to_zero(self, setup, target):
        nb.start_engine(title="par", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            if setup:
                _run(eng, sup, setup, "math")
            r = _run(eng, sup, target, "math")
            out = _run(eng, sup, f"{_eid(r)} を簡約して")
            got = eng.expressions[_eid(out)].canonical.strip()
            assert got == "0", got
        finally:
            nb.stop_engine()

    def test_two_different_bodies_do_not_collapse(self):
        r"""Keeping one name must not make `\sum_i x_i - \sum_i y_i` vanish —
        the reason the rename was introduced.  Measured, Maxima holds them as
        two terms, so the reason does not require the rename."""
        nb.start_engine(title="par2", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup, r"\sum_i x_i - \sum_i y_i", "math")
            out = _run(eng, sup, f"{_eid(r)} を簡約して")
            got = eng.expressions[_eid(out)].canonical
            assert got.strip() != "0"
            assert "x(i)" in got and "y(i)" in got, got
        finally:
            nb.stop_engine()


class TestNestingIsStillRenamed:
    r"""The half that must not change: an inner binder IS in the outer's scope,
    so leaving both as `i` would capture."""

    def _code(self, latex):
        eng = nb.get_engine()
        canonical = eng._handler.latex_to_canonical(latex) or latex
        ex = eng.register_expression(canonical, latex=latex)
        return eng._handler.generate_backend_code(ex.ir_ast)

    def setup_method(self):
        nb.start_engine(title="nest", mode="standard", language="ja", domains=[])

    def teardown_method(self):
        nb.stop_engine()

    def test_a_nested_sum_is_renamed(self):
        code = self._code(r"\sum_i \sum_i A_{i,i}")
        assert "A(j, j)" in code.replace(" ", " "), code

    def test_a_sum_nested_inside_a_body_is_renamed(self):
        r"""`\sum_i (x_i \sum_i y_i)` — the inner one shadows, and the outer's
        own `x_i` must keep `i`."""
        code = self._code(r"\sum_i (x_i \sum_i y_i)")
        assert "x(i)" in code and "y(j)" in code, code

    def test_parallel_sums_keep_the_written_name(self):
        code = self._code(r"\sum_i x_i + \sum_i y_i")
        assert "x(i)" in code and "y(i)" in code, code


class TestTheWalkerScopesNamesCorrectly:
    r"""Unit level, so a failure says WHICH rule broke."""

    def _sum(self, name, body_name):
        idx = IndexSymbol(name=name)
        return IndexSum(body=IndexedSymbol(base=Symbol(name=body_name),
                                           indices=[idx]),
                        dummy_indices=[IndexSymbol(name=name)])

    def test_siblings_are_independent(self):
        got = alphatize_indices(Add(args=[self._sum("i", "x"), self._sum("i", "y")]))
        assert [s.dummy_indices[0].name for s in got.args] == ["i", "i"]

    def test_three_siblings_are_all_independent(self):
        got = alphatize_indices(Add(args=[self._sum("i", "x"), self._sum("i", "y"),
                                          self._sum("i", "z")]))
        assert [s.dummy_indices[0].name for s in got.args] == ["i", "i", "i"]

    def test_nesting_still_renames(self):
        inner = self._sum("i", "x")
        outer = IndexSum(body=inner, dummy_indices=[IndexSymbol(name="i")])
        got = alphatize_indices(outer)
        assert got.dummy_indices[0].name == "i"
        assert got.body.dummy_indices[0].name == "j"

    def test_a_free_variable_of_the_same_name_still_renames(self):
        r"""A dummy colliding with a FREE name must still move — that name is in
        scope everywhere, unlike a sibling's binder."""
        got = alphatize_indices(Add(args=[Symbol(name="i"), self._sum("i", "x")]))
        assert got.args[1].dummy_indices[0].name != "i"

    def test_free_variables_are_added_even_when_a_set_is_supplied(self):
        r"""`occupied_names` is a UNION with the expression's free variables,
        not a replacement.  The test above takes the `is None` branch and so
        never reached the union — found by mutation, which is the point of
        writing the caller-supplied case separately."""
        got = alphatize_indices(Add(args=[Symbol(name="i"), self._sum("i", "x")]),
                                occupied_names={"a"})
        assert got.args[1].dummy_indices[0].name != "i"

    def test_a_renamed_dummy_does_not_collide_with_the_next_one(self):
        r"""Three dummies of one binder, all written `i`: the first keeps `i`
        and the other two must get DIFFERENT fresh names.  Forgetting to record
        the new name gives both of them `j` — two distinct indices collapsed
        into one, silently."""
        body = IndexedSymbol(base=Symbol(name="A"),
                             indices=[IndexSymbol(name="i")])
        got = alphatize_indices(IndexSum(
            body=body,
            dummy_indices=[IndexSymbol(name="i"), IndexSymbol(name="i"),
                           IndexSymbol(name="i")]))
        names = [d.name for d in got.dummy_indices]
        assert len(set(names)) == 3, names

    def test_the_caller_set_is_not_mutated(self):
        r"""The mechanism: one shared mutable `occupied` is what made a sibling
        look nested."""
        occupied = {"a"}
        alphatize_indices(Add(args=[self._sum("i", "x"), self._sum("i", "y")]),
                          occupied_names=occupied)
        assert occupied == {"a"}


class TestTheBcsHamiltonianKeepsOneMomentum:
    def test_parallel_terms_share_k(self):
        r"""The case `julia_build_mixin` had already opted out of alphatisation
        for: `Σ_k ξ_k c_k + Σ_k Δ c'_k c'_{-k}`."""
        k = IndexSymbol(name="k")
        kinetic = IndexSum(
            body=Mul(args=[IndexedSymbol(base=Symbol(name="xi"), indices=[k]),
                           IndexedSymbol(base=Symbol(name="c"), indices=[k])]),
            dummy_indices=[IndexSymbol(name="k")])
        pairing = IndexSum(
            body=Mul(args=[Symbol(name="Delta"),
                           IndexedSymbol(base=Symbol(name="c'"), indices=[k]),
                           IndexedSymbol(base=Symbol(name="c'"),
                                         indices=[IndexSymbol(name="-k")])]),
            dummy_indices=[IndexSymbol(name="k")])
        got = alphatize_indices(Add(args=[kinetic, pairing]))
        assert [s.dummy_indices[0].name for s in got.args] == ["k", "k"]
        names = sorted(i.name for a in got.args[1].body.args
                       if isinstance(a, IndexedSymbol) for i in a.indices)
        assert names == ["-k", "k"]
