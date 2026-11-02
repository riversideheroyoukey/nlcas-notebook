r"""A constraint between two integer indices is a pair of RANGES.

`\sum_{i \ne j}`, `\sum_{i<j}` and `\sum_{i \le j}` were refused wholesale, on
the ground that Maxima's `sum` has no place for a condition.  That premise is
true and the conclusion drawn from it was not: the CAS never needs a place for
the condition, because a summation index is an integer by construction, so
`i < j` IS `j` from `i+1` — the same warrant `_sum_dummy_indices` already cited
for rewriting `\sum_{j>i}^{M}`.  Measured 2026-08-17 against a live Maxima; see
`notebook/runtime/constrained_sum.py` for the table.

What stays refused is what genuinely has no rewrite, and each says so in its own
words: adjacency (`\langle i,j \rangle`) is a property of a lattice, not of the
index values, and `\prod_{i \ne j}` would be a division by the diagonal.
"""
import re

import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime import placeholder_sum_bounds as ph
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError


def _cas(latex: str) -> str:
    """Lower *latex*, WITH the placeholder-range pass — the production path.

    `tests/test_sum_family_ring2.py::_cas` deliberately omits that pass, which
    is why a relational constraint is refused there: it arrives with no range.
    A constrained sum is exactly the case where the two differ, so this file
    keeps its own helper rather than sharing one that measures a path nobody
    takes (CLAUDE.md: "Test through the interceptor, not `register_expression`").
    """
    counter = [0]

    def alloc(_name):
        counter[0] += 1
        return (f"nbSumLo{counter[0]}", f"nbSumHi{counter[0]}")

    canonical = MaximaBackendHandler(None).latex_to_canonical(latex)
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                          SymbolContext())
    ph.fill(ir, alloc)
    return MaximaCodeGenerator().generate(ir)


class TestTheRewrite:
    @pytest.mark.parametrize("latex,expect", [
        (r"\sum_{i<j}^{N} V_{i,j}",
         "sum(sum(V(i, j), j, i + 1, N), i, nbSumLo1, N)"),
        (r"\sum_{i \le j}^{N} V_{i,j}",
         "sum(sum(V(i, j), j, i, N), i, nbSumLo1, N)"),
        (r"\sum_{i \ne j}^{N} V_{i,j}",
         "sum(sum(V(i, j), j, nbSumLo1, N), i, nbSumLo1, N)"
         " - sum(V(i, i), i, nbSumLo1, N)"),
        # Π lowers identically for the ORDERING relations — no division.
        (r"\prod_{i<j}^{N} A_{i,j}",
         "product(product(A(i, j), j, i + 1, N), i, nbSumLo1, N)"),
    ])
    def test_lowers_to_nested_ranges(self, latex, expect):
        assert _cas(latex) == expect

    def test_an_unstated_end_is_a_placeholder_not_a_one(self):
        r"""`\sum_{i<j}^{N}` states the upper end only.

        The lower one becomes this sum's placeholder, exactly as a rangeless
        sum's does — a range is a physics choice and filling in a literal 1
        would be making it for the researcher.  It is substitutable, so nothing
        is a dead end.
        """
        out = _cas(r"\sum_{i<j}^{N} V_{i,j}")
        assert "nbSumLo1" in out and " 1," not in out.replace("i + 1,", "")

    def test_both_indices_share_ONE_range(self):
        r"""Two placeholder pairs would make `Σ_{i≠j}` a sum over two unrelated
        ranges, whose "diagonal" `i = j` means nothing.

        Counted at the ALLOCATOR, not in the generated string.  The generator
        reads `placeholder_bounds[0]` for both ends, so a second pair never
        reaches the output and an output-only check passes even when `fill`
        hands out two (found by mutation).  What `fill` is answerable for is not
        allocating a pair nobody uses: the allocator is the session-wide
        counter, its pairs are recorded in the `BoundLedger` and are rebuilt on
        load, so a phantom pair is a stored fact about an index that has no
        range of its own.
        """
        calls = []

        def alloc(name):
            calls.append(name)
            return (f"nbSumLo{len(calls)}", f"nbSumHi{len(calls)}")

        canonical = MaximaBackendHandler(None).latex_to_canonical(
            r"\sum_{i \ne j} V_{i,j}")
        ir = MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                              SymbolContext())
        ph.fill(ir, alloc)
        assert calls == ["i"]
        assert ir.placeholder_bounds == [["nbSumLo1", "nbSumHi1"]] * 2

    def test_an_unconstrained_pair_still_gets_a_pair_each(self):
        """The narrowing above must not leak into an ordinary multi-index sum,
        where two indices really are independent."""
        calls = []

        def alloc(name):
            calls.append(name)
            return (f"nbSumLo{len(calls)}", f"nbSumHi{len(calls)}")

        canonical = MaximaBackendHandler(None).latex_to_canonical(
            r"\sum_{m,n} h_{m,n}")
        ir = MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                              SymbolContext())
        ph.fill(ir, alloc)
        assert calls == ["m", "n"]

    def test_an_ordinary_multi_index_sum_still_gets_one_pair_each(self):
        out = _cas(r"\sum_{m,n} h_{m,n}")
        assert "nbSumLo1" in out and "nbSumLo2" in out


class TestStillRefusedAndWhy:
    @pytest.mark.parametrize("latex,key", [
        (r"\sum_{\langle i,j \rangle} V_{i,j}", "adjacency_sum"),
        (r"\sum_{\langle i j \rangle} V_{i,j}", "adjacency_sum"),
        (r"\prod_{i \ne j}^{N} A_{i,j}", "constrained_product_ne"),
    ])
    def test_refused_with_its_own_reason(self, latex, key):
        with pytest.raises(UnknownNotationError) as exc:
            _cas(latex)
        assert exc.value.args[0] == f"parser.unsupported_notation.{key}"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("key", ["adjacency_sum", "constrained_product_ne",
                                     "constrained_sum"])
    def test_every_reason_is_translated(self, lang, key):
        from notebook.runtime.i18n import t
        full = f"parser.unsupported_notation.{key}"
        msg = t(full, lang, binder="Σ", constraint=r"\ne", fn="sum")
        assert msg != full and len(msg) > 20


class TestTheCASAgrees:
    """The rewrite is checked by ASKING MAXIMA, not by comparing strings.

    Two silent-wrong bugs in this codebase survived string round-trip checks for
    months and only showed up under a differential check (CLAUDE.md, "What
    `Expression.canonical` Is").  So: generate the code, set the unstated end to
    1, and require `ratsimp(generated - reference) = 0`.
    """

    @staticmethod
    def _concrete(latex: str) -> str:
        return re.sub(r"nbSumLo\d+", "1", _cas(latex))

    @pytest.mark.parametrize("latex,reference", [
        (r"\sum_{i<j}^{3} V_{i,j}", "sum(sum(V(i,j),j,i+1,3),i,1,3)"),
        (r"\sum_{i \le j}^{3} V_{i,j}", "sum(sum(V(i,j),j,i,3),i,1,3)"),
        (r"\sum_{i \ne j}^{3} V_{i,j}",
         "sum(sum(V(i,j),j,1,3),i,1,3)-sum(V(i,i),i,1,3)"),
        (r"\prod_{i<j}^{3} A_{i,j}", "product(product(A(i,j),j,i+1,3),i,1,3)"),
    ])
    def test_differential_against_maxima_is_zero(self, latex, reference):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()
        got = self._concrete(latex)
        res = adapter.execute(f"ratsimp(({got}) - ({reference}))")
        assert str(getattr(res, "value", res)).strip() == "0"

    def test_i_lt_j_is_NOT_i_ne_j(self):
        r"""CLAUDE.md records that these two were once routed to a single
        `all_pairs` constraint, and `\sum_{i \ne j}` over three sites expanded
        to THREE terms instead of six, displayed back as `\sum_{i<j}` — half the
        sum, in notation the researcher had not written.  Pinned by COUNTING
        terms in the CAS's own answer, not by reading the generated string."""
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()

        def terms(latex):
            res = adapter.execute(self._concrete(latex))
            return str(getattr(res, "value", res)).count("V(")

        n_lt = terms(r"\sum_{i<j}^{3} V_{i,j}")        # N(N-1)/2
        n_ne = terms(r"\sum_{i \ne j}^{3} V_{i,j}")    # N(N-1)
        assert (n_lt, n_ne) == (3, 6)
