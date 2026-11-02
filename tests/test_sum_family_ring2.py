"""The ring outside B-10: Σ inside ∫, Π, and a constrained Σ.

B-10 (2026-08-08) fixed how a Σ or ∫ prefix is READ, and all eight of its cases
still hold (`tests/test_sum_bounds_and_dummy_names.py`).  Sweeping one ring
wider on 2026-08-09 found eight more forms, and every one of them returned a
DIFFERENT EXPRESSION as a success:

    \\int \\sum_i f_i(x) dx     → int * sum(f[i](x) * dx, i)   ∫ gone, dx a factor
    \\int dx \\sum_i f_i(x)     → integrate(1, x) * sum(…)     a well-formed `∫1dx`
    \\prod_i x_i               → prod(i) * x(i)               Π never taught
    \\prod_{i=1}^{N} x_i       → raw pydantic ValidationError
    \\sum_{i \\ne j} V_{i,j}     → sum(…, ne)                   a dummy named `ne`
    \\sum_{i<j} V_{i,j}        → sum(i)<j * V(i, j)           an inequality
    \\sum_{\\langle i,j \\rangle} → sum(…, langle)

Only `\\int \\left(\\sum_i …\\right) dx` came out right, and nobody brackets an
expansion written under an integral.
"""

import pytest

import ui.notebook_bridge as nb

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError


def _cas(latex: str) -> str:
    canonical = MaximaBackendHandler(None).latex_to_canonical(latex)
    resolved = MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                                SymbolContext())
    return MaximaCodeGenerator().generate(resolved)


class TestSumInsideIntegral:
    r"""The measure belongs to the ∫, whichever side of the Σ it is written on."""

    @pytest.mark.parametrize("latex", [
        r"\int \sum_i f_i(x) dx",
        r"\int dx \sum_i f_i(x)",
        r"\int \left(\sum_i f_i(x)\right) dx",
    ])
    def test_all_three_spellings_agree(self, latex):
        assert _cas(latex) == "integrate(sum(f[i](x), i), x)"

    def test_a_factor_between_them_stays_inside(self):
        assert _cas(r"\int f(x) \sum_i c_i dx") == "integrate(f(x) * sum(c(i), i), x)"

    def test_the_measure_may_be_a_fraction(self):
        assert _cas(r"\int \frac{d\omega}{2\pi} \sum_n G_n(\omega)") == (
            "integrate((2 * %pi)^-1 * sum(G[n](omega), n), omega)")

    def test_two_sums_under_one_integral(self):
        assert _cas(r"\int \sum_i \sum_j A_{i,j}(x) dx") == (
            "integrate(sum(sum(A[i, j](x), j), i), x)")

    def test_an_integral_under_a_sum_is_unaffected(self):
        # The other nesting order always worked; it must keep working.
        assert _cas(r"\sum_i \int f(x_i) dx") == "sum(integrate(f(x(i)), x), i)"


class TestProduct:
    r"""Π is read by the same machinery as Σ — only the operator differs."""

    @pytest.mark.parametrize("latex,expect", [
        (r"\prod_i x_i",           "product(x(i), i)"),
        (r"\prod_i (1+x_i)",       "product(1 + x(i), i)"),
        (r"\prod_{i=1}^{N} x_i",   "product(x(i), i, 1, N)"),
        (r"a \prod_i x_i",         "a * product(x(i), i)"),
    ])
    def test_forms(self, latex, expect):
        assert _cas(latex) == expect

    def test_partition_function(self):
        # Z = Π_i (1 + e^{-βε_i}) — ordinary statistical mechanics, and the
        # reason this is worth having at all.
        assert "product(1 + %e^(-beta * epsilon(i)), i)" in _cas(
            r"Z = \prod_i (1 + e^{-\beta \epsilon_i})")

    @pytest.mark.parametrize("latex,expect", [
        (r"\prod_i \sum_j A_{i,j}", "product(sum(A(i, j), j), i)"),
        (r"\sum_i \prod_j A_{i,j}", "sum(product(A(i, j), j), i)"),
        (r"\int \prod_i f_i(x) dx", "integrate(product(f[i](x), i), x)"),
    ])
    def test_mixed_with_the_others(self, latex, expect):
        assert _cas(latex) == expect

    def test_sum_is_not_disturbed(self):
        assert _cas(r"\sum_i x_i") == "sum(x(i), i)"
        assert _cas(r"\sum_i (x_i + y_i)") == "sum(x(i) + y(i), i)"


class TestConstrainedSum:
    r"""A condition written among the indices is not an index.

    It is READ (so the expression can be written and seen).  What happens next
    split on 2026-08-17: a condition comparing index VALUES is now COMPUTED, by
    rewriting it into ranges (`notebook/runtime/constrained_sum.py`), and only
    adjacency — a property of a lattice, not of the values — keeps the refusal.
    The computing half is pinned in `tests/test_constrained_sum_rewrite.py`,
    which runs through the placeholder pass; what stays here is the REFUSING
    half and the guarantee that nothing is silently mangled.

    Note this helper (`_cas`) does not fill placeholder ranges, so a relational
    constraint reaches the generator with no range at all and is refused for
    THAT reason.  The reason a notation is refused must not depend on which
    layer asked, which is why the classification runs before the range is
    looked at.

    **This is the general expression pipeline, which is the SAFETY NET, not the
    usual route.**  A constrained sum written as INPUT is claimed first by the
    F-1 interceptor (`_NS_BOND_SUB_RE` / `_NS_ALLPAIRS_SUB_RE` in
    `ui/interpret_f1.py`), which registers it as a `lattice_descriptor` —
    routing on the RAW TEXT, before any of this runs.  What reaches here is what
    that regex does not claim: multi-letter indices, an empty body, an
    expression built by a later operation.  Pinned separately in
    `TestTheLatticeRouteClaimsItFirst` below, because the two must not disagree
    about what the SAME notation means.
    """

    @pytest.mark.parametrize("latex,key", [
        # Adjacency has no rewrite: it is a property of the LATTICE, and the
        # notation does not name one.  Its own reason, pointing at the route
        # that does carry it.
        (r"\sum_{\langle i,j \rangle} V_{i,j}", "adjacency_sum"),
        (r"\sum_{\langle i j \rangle} V_{i,j}", "adjacency_sum"),
        # Π_{i≠j} would be a DIVISION by the diagonal, which is a rewrite only
        # when no diagonal factor vanishes — nothing says that.
        (r"\prod_{i \ne j} A_{i,j}", "constrained_product_ne"),
        # Relational: computable, but this layer supplies no range.
        (r"\sum_{i \ne j} V_{i,j}", "constrained_sum"),
        (r"\sum_{i \neq j} V_{i,j}", "constrained_sum"),
        (r"\sum_{i<j} V_{i,j}", "constrained_sum"),
        (r"\sum_{i \le j} V_{i,j}", "constrained_sum"),
    ])
    def test_refused_by_name_not_silently_mangled(self, latex, key):
        with pytest.raises(UnknownNotationError) as exc:
            _cas(latex)
        assert exc.value.args[0] == f"parser.unsupported_notation.{key}"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("key", ["constrained_sum", "adjacency_sum",
                                     "constrained_product_ne"])
    def test_the_reason_names_the_constraint_in_both_languages(self, lang, key):
        from notebook.runtime.i18n import t
        full = f"parser.unsupported_notation.{key}"
        msg = t(full, lang, binder="Σ", constraint=r"\ne", fn="sum")
        assert msg != full
        assert r"\ne" in msg

    def test_an_ordinary_multi_index_sum_still_works(self):
        assert _cas(r"\sum_{m,n} h_{m,n}") == "sum(sum(h(m, n), n), m)"

    @pytest.mark.parametrize("latex,expect", [
        (r"A_{i,j}", "A_{i,j}"),
        (r"H_{BCS}", "H_BCS"),
        # 2026-08-20: 波括弧は**保たれる**ようになった。この検査の目的
        # （`_subscript_constraint` が関係でない添字に触らないこと）は変わらず
        # 満たされる——触っていたら制約マーカーになっている。落ちていた頃の
        # `x_n+1` は、添字の中身が「x_n に 1 を足す」に化けていた当のもので、
        # `x_{n+1} = 2 x_n` を偽の等式として登録していた。
        (r"x_{n+1}", "x_{n+1}"),
        (r"c_{k,\uparrow}", r"c_{k,\uparrow}"),
    ])
    def test_an_ordinary_subscript_is_untouched(self, latex, expect):
        # `_subscript_constraint` runs over every `_{…}`, so the forms that do
        # NOT hold a relation have to come through the CONVERTER unchanged.
        # (What the resolver then does with them is `subscript_policy`'s
        # business and is pinned elsewhere.)
        assert MaximaBackendHandler(None).latex_to_canonical(latex) == expect

    def test_a_relation_outside_a_subscript_is_untouched(self):
        assert MaximaBackendHandler(None).latex_to_canonical(r"x \le y") == "x <= y"


class TestIdempotence:
    r"""`to_canonical` applies its rules until the text stops changing, so a new
    rule that fires on its own output loops.  It did: `\sum_{i \ne j}` stabilised
    as `{i,,,,,,\ne,,,,,,j}`."""

    @pytest.mark.parametrize("latex", [
        r"\sum_{i \ne j} V_{i,j}", r"\sum_{i<j} V", r"\sum_{i \leq j} V",
        r"\sum_{\langle i,j \rangle} V",
    ])
    def test_converting_twice_changes_nothing(self, latex):
        h = MaximaBackendHandler(None)
        once = h.latex_to_canonical(latex)
        assert h.latex_to_canonical(once) == once
        assert ",," not in once


class TestTheLatticeRouteClaimsItFirst:
    r"""Where a constrained sum actually goes when a researcher types it.

    Routing is decided by a regex over the RAW INPUT TEXT in the F-1
    interceptor, which runs before the general pipeline — the surface spelling
    is identical, so nothing downstream could tell them apart.  Measured
    2026-08-09: `\neq` was in that regex and `\ne` was not, so two spellings of
    ONE notation took two different routes and only one of them worked.
    """

    @staticmethod
    def _route(text: str) -> str:
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="route") as e:
            e.start()
            _run(_handle_interpret(text, e, _sup(), no_llm=True))
            exprs = list(e.expressions.values())
            desc = (exprs[-1].lattice_descriptor or {}) if exprs else {}
            return desc.get("pair_constraint") or "general"

    @pytest.mark.parametrize("latex,expect", [
        (r"\sum_{\langle i,j \rangle} V_{i,j}", "nearest_neighbor"),
        (r"\sum_{\langle ij \rangle} V_{i,j}",  "nearest_neighbor"),
        (r"\sum_{i<j} V_{i,j}",                 "all_pairs"),
        (r"\sum_{i \neq j} V_{i,j}",            "ordered_pairs"),
        (r"\sum_{i \ne j} V_{i,j}",             "ordered_pairs"),
        (r"\sum_{i \ne j} V_{ii,jj}",           "ordered_pairs"),
    ])
    def test_every_spelling_of_one_notation_takes_one_route(self, latex, expect):
        # Spellings of the SAME notation must agree (`\ne` vs `\neq`); DIFFERENT
        # notations must not be merged (`i<j` vs `i≠j` — see
        # TestOrderedPairsIsNotUnorderedPairs).
        assert self._route(latex) == expect

    def test_the_tight_binding_hamiltonian_is_unaffected(self):
        assert self._route(r"H = -t \sum_{\langle ij \rangle} c'_i c_j") == \
            "nearest_neighbor"


class TestOrderedPairsIsNotUnorderedPairs:
    r"""Σ_{i≠j} has twice the terms of Σ_{i<j}, and must say so.

    Both spellings were routed to the SAME `all_pairs` constraint, so
    `\sum_{i \ne j} V_{i,j}` over three sites expanded to THREE terms instead of
    six — and its display came back rewritten as `\sum_{i<j}`, notation the
    researcher had not written (measured 2026-08-10, while answering "does it
    give every term?").  Half a sum is the kind of wrong answer that looks
    right.
    """

    @staticmethod
    def _expand(text: str, sites: int = 3) -> "tuple[str, int]":
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="expand") as e:
            e.start()
            sup = _sup()
            _run(_handle_interpret(text, e, sup, no_llm=True))
            _run(_handle_interpret(f"E1 を {sites} サイトで展開して", e, sup,
                                   no_llm=True))
            exprs = list(e.expressions.values())
            return exprs[0].latex, exprs[-1].latex.count("V")

    @pytest.mark.parametrize("latex,terms", [
        (r"\sum_{i<j} V_{i,j}",                 3),   # N(N-1)/2
        (r"\sum_{i \ne j} V_{i,j}",             6),   # N(N-1)
        (r"\sum_{i \neq j} V_{i,j}",            6),
        (r"\sum_{\langle i,j \rangle} V_{i,j}", 2),   # a 3-site open chain
    ])
    def test_term_count(self, latex, terms):
        assert self._expand(latex)[1] == terms

    @pytest.mark.parametrize("latex,keep", [
        (r"\sum_{i<j} V_{i,j}",     "i<j"),
        (r"\sum_{i \ne j} V_{i,j}", r"\ne"),
    ])
    def test_the_display_keeps_what_was_written(self, latex, keep):
        assert keep in self._expand(latex)[0]

    def test_both_orders_are_present(self):
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="orders") as e:
            e.start()
            sup = _sup()
            _run(_handle_interpret(r"\sum_{i \ne j} V_{i,j}", e, sup, no_llm=True))
            _run(_handle_interpret("E1 を 3 サイトで展開して", e, sup, no_llm=True))
            out = list(e.expressions.values())[-1].latex.replace(" ", "")
        for pair in ("1,2", "2,1", "1,3", "3,1", "2,3", "3,2"):
            assert pair in out, pair


class TestChainedConstraint:
    r"""`\sum_{i<j<k<l}` — beyond a two-index bond, so the lattice route must
    not claim it.

    The bond-sum pattern ends with an OPTIONAL `}`, so a chained constraint
    matched its first pair and stopped: `\sum_{i<j<k<l} V_{i,j,k,l}` was
    registered as an ordinary Σ_{i<j} over (i,j) and displayed as
    `\sum_{i<j} <k<l} V_{i,j,k,l}` — broken LaTeX with a dangling brace, and a
    body that would have been glued into every expanded term (measured
    2026-08-10).

    Falling through to the general pipeline it is written, displayed as typed,
    and refused BY NAME when a CAS operation is asked for.  That is the honest
    answer for a notation whose meaning this system cannot yet carry.
    """

    _CHAINED = [
        r"\sum_{i<j<k} V_{i,j,k}",
        r"\sum_{i<j<k<l} V_{i,j,k,l}",
        r"\sum_{i \ne j \ne k} V_{i,j,k}",
    ]

    @pytest.mark.parametrize("latex", _CHAINED)
    def test_not_claimed_by_the_lattice_route(self, latex):
        assert TestTheLatticeRouteClaimsItFirst._route(latex) == "general"

    @pytest.mark.parametrize("latex", _CHAINED)
    def test_displayed_exactly_as_written(self, latex):
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="chained") as e:
            e.start()
            _run(_handle_interpret(latex, e, _sup(), no_llm=True))
            shown = list(e.expressions.values())[-1].latex
        assert shown == latex
        assert "} <" not in shown and "} \\ne" not in shown  # no dangling brace

    @pytest.mark.parametrize("latex", _CHAINED)
    def test_refused_by_name_when_computed(self, latex):
        with pytest.raises(UnknownNotationError) as exc:
            _cas(latex)
        assert exc.value.args[0] == "parser.unsupported_notation.constrained_sum"

    @pytest.mark.parametrize("latex,expect", [
        (r"\sum_{i<j} V_{i,j}",                 "all_pairs"),
        (r"\sum_{i \ne j} V_{i,j}",             "ordered_pairs"),
        (r"\sum_{\langle ij \rangle} V_{i,j}",  "nearest_neighbor"),
    ])
    def test_the_two_index_forms_still_route(self, latex, expect):
        # The guard rejects an UNCLOSED subscript, not a closed one.
        assert TestTheLatticeRouteClaimsItFirst._route(latex) == expect


class TestExplicitBounds:
    r"""What happens when a range IS written on a constrained sum."""

    @staticmethod
    def _shown(text: str) -> str:
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="bounds") as e:
            e.start()
            _run(_handle_interpret(text, e, _sup(), no_llm=True))
            exprs = list(e.expressions.values())
            return exprs[-1].latex if exprs else ""

    @pytest.mark.parametrize("latex", [
        r"\sum_{i<j}^{N} V_{i,j}",
        r"\sum_{i \ne j}^{N} V_{i,j}",
    ])
    def test_a_bound_takes_it_out_of_the_lattice_route(self, latex):
        # A bond descriptor expands by a SITE COUNT given later; it has nowhere
        # to put a symbolic range.  Claimed anyway, the superscript was left
        # dangling in the display (`\sum_{i<j} ^{N} V_{i,j}`) and the CAS then
        # failed on it (measured 2026-08-10).
        assert TestTheLatticeRouteClaimsItFirst._route(latex) == "general"
        assert self._shown(latex) == latex          # and nothing is dropped

    @pytest.mark.parametrize("latex", [
        r"\sum_{i<j}^{N} V_{i,j}",
        r"\sum_{i \ne j}^{N} V_{i,j}",
    ])
    def test_and_is_then_COMPUTED_by_rewriting_the_ranges(self, latex):
        r"""Was: "and is then refused by name".

        These two are exactly the forms the lattice route declines (a bond
        descriptor expands by a site count and has nowhere to put a symbolic
        range), so before 2026-08-17 a written range took the notation OUT of
        the one path that carried it and into a refusal — the worst of both.
        They are now rewritten into nested ranges; `tests/
        test_constrained_sum_rewrite.py` pins the forms and checks them against
        a live Maxima.

        `_cas` here supplies no placeholder range, so the assertion is only that
        the CONSTRAINT is no longer what stops it.
        """
        from notebook.runtime.constrained_sum import classify_constrained
        plan = classify_constrained(
            r"\ne" if r"\ne" in latex else "<", ["i", "j"], "1", "N", "sum")
        assert plan.kind in ("nested", "minus_diagonal")

    def test_the_nested_form_with_bounds_computes(self):
        # The way to WRITE a constrained sum this system can carry out: say the
        # constraint in the ranges.  Σ_{i<j} over 1..N is Σ_i Σ_{j=i+1}.
        assert _cas(r"\sum_{i=1}^{N} \sum_{j=i+1}^{N} V_{i,j}") == (
            "sum(sum(V(i, j), j, i + 1, N), i, 1, N)")


class TestABondSumIsDisplayOnly:
    r"""A generic CAS operation on a symbolic bond sum drops its constraint.

    The constraint lives in the descriptor, not in the expression text, so
    re-reading the text loses it — and the loss is silent.  Measured
    2026-08-10, `simplify` returned:

        \sum_{\langle ij\rangle} V_{i,j}  →  -(V(i,j)*nbSumLo1) + V(i,j)*nbSumHi1 + V(i,j)
        \sum_{i \ne j} V_{i,j}            →  'sum(V(i,j), i, nbSumLo1, nbSumHi1)

    The first is Maxima summing a CONSTANT over the placeholder range; the
    second keeps the SHAPE of the original with the constraint gone, which is
    what makes it dangerous.  Both were reported as successes.
    """

    _BOND_SUMS = [
        (r"\sum_{\langle ij \rangle} V_{i,j}", 2),
        (r"\sum_{i<j} V_{i,j}",                3),
        (r"\sum_{i \ne j} V_{i,j}",            6),
    ]

    @pytest.mark.parametrize("latex,_terms", _BOND_SUMS)
    def test_a_generic_operation_is_refused_with_a_reason(self, latex, _terms):
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="guard") as e:
            e.start()
            _run(_handle_interpret(latex, e, _sup(), no_llm=True))
            with pytest.raises(ValueError) as exc:
                e.run_operation("simplify", {"expr": "E1"})
        assert "nbSum" not in str(exc.value)
        assert str(exc.value) != "lattice.display_only_bond_sum"   # translated

    @pytest.mark.parametrize("latex,terms", _BOND_SUMS)
    def test_its_own_operation_still_works(self, latex, terms):
        # The guard must not close the door the researcher actually uses.
        assert TestOrderedPairsIsNotUnorderedPairs._expand(latex)[1] == terms

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_reason_is_translated_both_ways(self, lang):
        from notebook.runtime.i18n import t
        msg = t("lattice.display_only_bond_sum", lang,
                expr_id="E1", operation="simplify")
        assert msg != "lattice.display_only_bond_sum"
        assert "E1" in msg and "simplify" in msg


class TestTrailingSplitDifferential:
    r"""`\int f d\theta` — the differential of a Greek variable, TRAILING.

    The leading form (`\int d\omega f`) has read the split `d`·`\omega` pair
    since 2026-08-08; the trailing form did not, and **the trailing form is the
    one this system prints**: `\int_0^{2\pi} r\,dr\,d\theta` is in the published
    coordinate-jacobian examples.  Re-reading it produced a `Mul` — the integral
    silently became a product — and with an infinite bound the researcher got a
    raw pydantic ValidationError (measured 2026-08-14, SCENARIO-MACRO case 5,
    while writing the convolution over `\tau`).

    Every Greek integration variable is written this way: dθ, dφ, dτ, dω.
    """

    @pytest.mark.parametrize("latex,expect", [
        (r"\int_0^{2\pi} 1 d\theta", "integrate(1, theta, 0, 2 * %pi)"),
        (r"\int_0^{2\pi}\int_0^1 r dr d\theta",
         "integrate(integrate(r, r, 0, 1), theta, 0, 2 * %pi)"),
        (r"\int_0^1 x dx", "integrate(x, x, 0, 1)"),      # single token, unchanged
        (r"\int d x^2 dx", "integrate(d * x^2, x)"),      # bare `d` stays a constant
        (r"\int_{-\infty}^{\infty} f d\tau", "integrate(f, tau, -inf, inf)"),
    ])
    def test_the_measure_is_read(self, latex, expect):
        assert _cas(latex) == expect


class TestP2BindAndSolvingADefinition:
    r"""P2 binds an H-1 definition CAS-side; solving that definition must still
    work.  The two halves of open_items P2-BIND-NOT-LOWERED, together.

    Half 1 — the bind takes the RESEARCHER's writing, not CAS code.  Measured
    2026-08-14: `V = \int_0^{\infty} v e^{-s t} dt` bound `V` to
    `%e^-(s*t)*dt*int_0^inf*v` — the `%int_…` family, an integral that was never
    formed, carried silently into every later use of V.

    Half 2 — routing the bind through the lowering door makes it SUCCEED where
    it used to fail silently, and a successful bind turns "solve this definition
    for its variable" into a tautology: `solve(S = 14*x, x)` answers `all`.
    A local `block([S], …)` shadows the binding for that one call — no `kill`,
    no restore, nothing to leak if the call raises.

    Both are asserted here because fixing either alone is wrong: half 1 without
    half 2 removes a capability, half 2 without half 1 leaves the garbage bind.
    """

    def test_an_integral_definition_binds_nothing_rather_than_garbage(self):
        nb.start_engine(title="p2_integral", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            eng.register_expression(r"V = \int_0^(inf)*v*%e^(-s*t)*dt",
                                    latex=r"V = \int_0^{\infty} v e^{-s t} dt")
            got = eng._adapter.execute("V").value.strip()
            assert got == "V", got                     # opaque, not garbage
            assert "int_0^inf" not in got, got         # the measured nonsense
        finally:
            nb.stop_engine()

    def test_an_ordinary_definition_still_binds(self):
        """P2's value is untouched: a definition the CAS can hold still binds."""
        nb.start_engine(title="p2_ordinary", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            eng.register_expression("E = ((1)/(2))*m*v^2", latex=r"E = \frac{1}{2} m v^2")
            assert eng._adapter.execute("E").value.replace(" ", "") == "(m*v^2)/2"
        finally:
            nb.stop_engine()

    def test_solving_a_definition_for_a_variable_on_its_right(self):
        nb.start_engine(title="p2_solve_def", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            src = r"S = x \sum_{i=1}^{3} i^2"
            e = eng.register_expression(
                eng._handler.latex_to_canonical(src), latex=src)
            assert eng._adapter.execute("S").value.replace(" ", "") == "14*x"
            step, _ = eng.run_operation(
                "solve_equation", {"equation": e.display_id, "var": "x"})
            assert step.rendered_command.startswith("block([S],"), step.rendered_command
            got = eng.expressions[step.output_refs[0]].canonical.replace(" ", "")
            assert got == "S/14", got
            # the global binding survives the block — P2 keeps working after
            assert eng._adapter.execute("S").value.replace(" ", "") == "14*x"
        finally:
            nb.stop_engine()

    def test_an_equation_that_is_not_a_definition_is_not_shadowed(self):
        """The guard asks whether THIS expression defines the symbol, not what
        the left side looks like: a compound left side never binds, so it must
        expand as before."""
        nb.start_engine(title="p2_not_a_def", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            eng.register_expression("H = J*x*y", latex="H = J x y")
            eq = eng.register_expression("H + W = 5", latex="H + W = 5")
            r = eng._handler._build_solve_equation(
                {"equation": eq.display_id, "var": "x"},
                {eq.display_id: "H + W = 5"})
            assert not r.command.startswith("block(["), r.command
        finally:
            nb.stop_engine()

    def test_the_decision_never_asks_the_cas(self):
        """Shadowing a symbol whose bind never took is a no-op, so the guard does
        not need to know whether P2 succeeded — which is what makes the decision
        a pure function of notebook state, decidable line by line."""
        nb.start_engine(title="p2_no_cas_query", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            e = eng.register_expression(r"V = \int_0^(inf)*v*%e^(-s*t)*dt",
                                        latex=r"V = \int_0^{\infty} v e^{-s t} dt")
            assert eng._adapter.execute("V").value.strip() == "V"   # bind did NOT take
            r = eng._handler._build_solve_equation(
                {"equation": e.display_id, "var": "v"},
                {e.display_id: "V = integrate(v*%e^(-s*t),t,0,inf)"})
            assert r.command.startswith("block([V],"), r.command
        finally:
            nb.stop_engine()

    def test_redefining_the_symbol_moves_the_decision(self):
        """The one real order dependence, inherited from H-1/P2: after `S = y`
        the CAS's `S` IS `y`, so re-solving the OLD definition must NOT be
        shadowed — `S = 14*x` solved for x should give `y/14`, not `S/14`, which
        would name a symbol whose meaning has since changed."""
        nb.start_engine(title="p2_redefine", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            src = r"S = x \sum_{i=1}^{3} i^2"
            e1 = eng.register_expression(
                eng._handler.latex_to_canonical(src), latex=src)
            args = ({"equation": e1.display_id, "var": "x"},
                    {e1.display_id: "S = x*sum(i^2,i,1,3)"})
            assert eng._handler._build_solve_equation(*args).command.startswith(
                "block([S],")
            eng.register_expression("S = y", latex="S = y")
            assert not eng._handler._build_solve_equation(*args).command.startswith(
                "block([")
        finally:
            nb.stop_engine()


class TestSubstituteKeepsTheWholeEquation:
    r"""`substitute` must not throw away the left-hand side.

    The Maxima builder used to cut its source down to the RHS, because
    `subst` on a full `H = RHS` makes Maxima evaluate the bound `H` and answer
    the tautology `RHS' = RHS'`.  That cut applied to EVERY equation.  Measured
    2026-08-14 on `L(s^2 Q - s q_0 - u_0) + R(s Q - q_0) + Q/C = V`: the
    substitution's source became the bare `V`, so no target could ever be found
    — and had the target also appeared on the right, it would have substituted
    into HALF the equation and reported success.

    It now SHADOWS the left symbol (`block([H], …)`) and passes the whole
    equation.  Measured, all three shapes come out right, and the bound one
    comes out BETTER than cutting did — the label survives.
    """

    def test_content_on_the_left_survives(self):
        nb.start_engine(title="sub_left", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            src = r"L (s^2 Q - s q_0 - u_0) + R (s Q - q_0) + \frac{Q}{C} = V"
            e = eng.register_expression(
                eng._handler.latex_to_canonical(src), latex=src)
            step, _ = eng.run_operation(
                "substitute", {"expr": e.display_id, "lhs": "q_0", "rhs": "0"})
            out = eng.expressions[step.output_refs[0]].canonical.replace(" ", "")
            assert "=V" in out, out                 # the equation survived
            assert "q(0)" not in out, out           # the target was substituted
            assert out != "V", out                  # the measured failure
        finally:
            nb.stop_engine()

    def test_a_bound_definition_keeps_its_label_and_does_not_double(self):
        nb.start_engine(title="sub_def", mode="standard", language="en",
                        domains=["control_theory"])
        eng = nb.get_engine()
        try:
            e = eng.register_expression("H = J*x*y", latex="H = J x y")
            assert eng._adapter.execute("H").value.replace(" ", "") == "J*x*y"
            step, _ = eng.run_operation(
                "substitute", {"expr": e.display_id, "lhs": "x", "rhs": "2"})
            latex = eng.expressions[step.output_refs[0]].latex
            assert latex.count("=") == 1, latex     # `H = H=2 J y` was the bug
            assert latex.replace(" ", "").startswith("H="), latex
        finally:
            nb.stop_engine()


class TestALimitIsNotASwallowedBody:
    r"""A limit that is itself a one-argument call, read as a limit.

    `_give_back_bracket` exists for `\int (x+1) dx` — a bracketed body swallowed
    by the ∫ sign (B-10 ②-b).  It recognised that by shape: a script holding a
    one-argument FunctionCall.  A LIMIT has the same shape, so an ordinary
    `\sqrt{2}` upper bound was read as "the bound `sqrt` applied to the body
    `2`" — the name became the limit and its argument moved into the integrand,
    silently (measured 2026-08-16):

        \int_0^{\sqrt{2}} x dx   → integrate(2 * x, x, 0, sqrt)
        \int_{\sqrt{2}}^{2} x dx → integrate(2^2 * x, x)     ← bounds GONE
        \sum_{i=0}^{\sqrt{2}} i  → sum(2 * i, i, 0, sqrt)
        \int_0^{f(a)} x dx       → integrate(a * x, x, 0, f)

    Two arguments (`g(a,b)`) and a coefficient (`2 f(a)`) escaped, which is what
    kept it hidden.  The discriminator needs no guess: a SUPERSCRIPT is always a
    limit, and a SUBSCRIPT is an index only for Σ/Π — on an ∫ it is a limit too.

    These spellings are the ones route ② still reads (① claims the plain
    `\int_0^{\sqrt{2}} x \, dx` since INTEGRAL-CLAIM-ASYMMETRY), so the rule
    stays measurable from here.
    """

    @pytest.mark.parametrize("latex,expected", [
        (r"\sum_{i=0}^{\sqrt{4}} i",      "sum(i, i, 0, sqrt(4))"),
        (r"\prod_{i=1}^{\sqrt{4}} i",     "product(i, i, 1, sqrt(4))"),
        (r"\sum_{i=1}^{f(n)} i",          "sum(i, i, 1, f(n))"),
        (r"\int_0^{\sqrt{2}} dx \, x",    "integrate(x, x, 0, sqrt(2))"),
        (r"\int_{\sqrt{2}}^{2} dx \, x",  "integrate(x, x, sqrt(2), 2)"),
        (r"\int_0^{\sin(1)} dx \, x",     "integrate(x, x, 0, sin(1))"),
    ])
    def test_the_limit_stays_the_limit(self, latex, expected):
        assert _cas(latex) == expected

    @pytest.mark.parametrize("latex,expected", [
        # What the give-back is FOR must keep working (B-10 ②-b).
        (r"\int (x+1) dx",          "integrate(x + 1, x)"),
        (r"\int (x+1)^2 dx",        "integrate((x + 1)^2, x)"),
        (r"\sum_i (x_i+y_i)",       "sum(x(i) + y(i), i)"),
        (r"\sum_{i=1}^{3} (i+1)",   "sum(i + 1, i, 1, 3)"),
    ])
    def test_a_swallowed_body_is_still_given_back(self, latex, expected):
        assert _cas(latex) == expected
