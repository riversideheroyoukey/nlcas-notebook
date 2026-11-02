r"""Two things a sum has to keep: its RANGE, and its researcher's index names.

**B-10 — the range.**  Maxima's `sum` takes four arguments, and the generator
could only ever produce two, so `\sum_{i=1}^{N} x_i` was rejected outright even
though the researcher had written the range.  Three separate places lost it,
each silently:

1. the lexer had no `=` token at all, so a non-top-level `=` was skipped by
   `finditer` and `{i=1}` arrived as `Juxtaposition(i, 1)` — the bound gone
   before any resolver could see it;
2. `try_resolve_sum` read `lower`/`upper` into locals and never used them;
3. `_prepare_expr_lookup` asked `"=" in canonical`, which read
   `\sum_{i=1}^{3} i^2` as the equation `\sum_{i` = `1}^{3} i^2` and replaced
   the expression with that bogus right-hand side — measured, and present long
   before any of the sum work.

(3) is the shape this project keeps finding: a question that already has a
brace-aware answer (`find_top_level_relation`, used by `parse()`) being asked a
second time with a naive scan.

**SUM-DUMMY-NAMES — the names.**  QuantumAlgebra discards the written name and
numbers what is left by FIRST OCCURRENCE IN THE BODY — measured with marker
parameters, and preserved across `normal_form`.  Mapping `#ᵢ` to the i-th
BINDING (which is what `\sum_{m,n}` reads as, and what `_collect_dummy_names`
returns) therefore printed a sum that was not written:
`\sum_{m,n} c'_{n,u} c_{m,u}` came back as `c†_m c_n`.  The order is computed
from the generator's own output instead.

**B-10 ② — the range nobody wrote**, and everything else a sum stopped doing.
All closed 2026-08-08 (docs/handoffs/sum_bounds_handoff.md is the state two of them were
handed over in; the other three were found by measuring around those two):

* ②-a, the range is shared by the index NAME the researcher wrote, so the
  written name has to still be there when it is handed out;
* ②-b, a BRACKETED body — `\sum_i (x_i+y_i)`, `\int (x+1) dx` — was read as an
  application, so the binder stopped being one.  Wider than the ticket said, and
  it includes an ordinary indefinite integral that returned a SUCCESSFUL wrong
  answer;
* ②-c, the pair has to be REPRODUCIBLE, not merely unique — the IR that carried
  it does not survive a save;
* ②-d, only the FIRST factor of a product was examined, so `\sum_i x_i \sum_j
  y_j` and `a \sum_i x_i` were not read;
* ②-e, the CAS noun form `'sum(…)` could not be parsed back, so referencing a
  sum's own result into a formula silently stopped being a sum.
"""
import re

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser


def _cas(text):
    ctx = SymbolContext()
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(text), ctx)
    return MaximaCodeGenerator(context=ctx).generate(ir)


def _placeholder_ranges(code):
    """The placeholder range pairs a piece of CAS code uses, by number.

    Compared as a SET so the assertions say "one range" / "two ranges" without
    depending on the session counter, which is shared and keeps running.
    """
    return set(re.findall(r"nbSumLo(\d+)", code))


class TestTheRangeSurvivesTheParser:
    """(1) — the lexer now has an `=`, so a bracketed one is not skipped."""

    @pytest.mark.parametrize("text", ["{i=1}", "x_{i=1}", "f(i=1)", "[i=1]"])
    def test_a_bracketed_equals_is_not_dropped(self, text):
        from notebook.models.ir import Equation
        node = SharedParser().parse(text)
        found = []

        def walk(n):
            if isinstance(n, Equation):
                found.append(n)
            for attr in ("args", "indices"):
                for c in (getattr(n, attr, None) or ()):
                    walk(c)
            for attr in ("base", "sub", "script", "expr", "func", "left", "right"):
                c = getattr(n, attr, None)
                if c is not None and not isinstance(c, (str, int, float)):
                    walk(c)

        walk(node)
        assert found, f"{text} lost its `=`: {node!r}"

    def test_a_top_level_equation_is_unchanged(self):
        from notebook.models.ir import Equation
        assert isinstance(SharedParser().parse("i=1"), Equation)


class TestTheRangeReachesTheCAS:
    """(2)+(3) — and Maxima's four-argument form becomes available."""

    @pytest.mark.parametrize("text,expected", [
        (r"\sum_{i=1}^{N} x_i", "sum(x(i), i, 1, N)"),
        (r"\sum_{i=1}^{N} h_{i,j} y_i", "sum(h(i, j) * y(i), i, 1, N)"),
    ])
    def test_bounds_are_lowered(self, text, expected):
        assert _cas(text) == expected

    def test_a_brace_group_is_not_a_list(self):
        r"""`^{N}` is LaTeX grouping; left wrapped it rendered as `[N]`."""
        assert "[N]" not in _cas(r"\sum_{i=1}^{N} x_i")

    @pytest.mark.parametrize("text", [r"\sum_i x_i", r"\sum_{m,n} h_{m,n}"])
    def test_a_sum_without_a_range_stays_rangeless(self, text):
        """No range is invented — that is a physics choice, still undecided
        (open_items B-10 ②)."""
        assert _cas(text).count(",") <= 1 or "sum(sum(" in _cas(text)


class TestTheWholeSumEvaluates:
    """The point of all three fixes: a bounded sum is a CAS object."""

    def setup_method(self):
        nb.start_engine(title="sumb", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    @pytest.mark.parametrize("text,value", [
        (r"\sum_{i=1}^{3} i", "6"),
        (r"\sum_{i=1}^{3} i^2", "14"),
    ])
    def test_it_computes(self, text, value):
        ex = self.eng.register_expression(text)
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        assert self.eng.expressions[step.output_refs[0]].canonical.strip() == value

    def test_an_ordinary_equation_still_reduces_to_its_rhs(self):
        """The brace-aware test must not stop recognising real equations."""
        ex = self.eng.register_expression("y = x^2 + 1")
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        out = self.eng.expressions[step.output_refs[0]].canonical.replace(" ", "")
        assert out == "x^2+1"


class TestTheResearchersIndexNames:
    """SUM-DUMMY-NAMES: what is displayed is what was written."""

    def setup_method(self):
        nb.start_engine(title="dummies", mode="second_quantization",
                        language="ja", domains=["condensed_matter"])
        self.eng = nb.get_engine()
        self.eng.run_operation("define_fermion_operator", {"name": "c"})

    def teardown_method(self):
        nb.stop_engine()

    def _latex(self, text):
        ex = self.eng.register_expression(text)
        step, _ = self.eng.run_operation("normal_order", {"expr": ex.display_id})
        return self.eng.expressions[step.output_refs[0]].latex

    def test_the_written_order_is_the_displayed_order(self):
        """The case the binding-order map got wrong: the body names `n` first.

        Compared on the BODY, not the whole string — `\\sum` itself contains an
        `m`, which is how the first version of this assertion managed to fail on
        a correct result.
        """
        out = self._latex(r"\sum_{m,n} c'_{n,u} c_{m,u}")
        body = out.split("}", 1)[1] if "}" in out else out   # drop the Σ subscript
        assert body.index("n") < body.index("m"), out

    def test_three_indices_keep_their_own_names(self):
        out = self._latex(r"\sum_{m,n,p} c'_{p,u} c'_{n,u} c_{m,u}")
        for name in ("p", "n", "m"):
            assert name in out, (name, out)
        assert "k_{1}" not in out          # no neutral fallback when names are known

    def test_a_greek_name_is_not_manufactured_by_concatenation(self):
        r"""QA writes a term's indices with no separator, so index `m` next to
        spin `u` became the single word `mu` and then `\mu`."""
        out = self._latex(r"\sum_{m,n} c'_{m,u} c_{n,u}")
        assert r"\mu" not in out and r"\nu" not in out, out


class TestABracketedBodyIsStillTheBody:
    r"""②-b — the bracket after a Σ/∫ prefix belongs to the prefix.

    The parser is semantics-free, so `\sum_i (x_i+y_i)` and `f_i(x)` reach it as
    the same shape and it reads both as an application (CLAUDE.md, "A subscript
    that is APPLIED is part of the name").  For a binder that swallows the BODY,
    and each spelling loses it differently — every one of these was measured
    2026-08-08 and every one was silent:

    ``\sum_i (x_i+y_i)`` → the name ``sum_i``; ``\sum_{m,n} (h+g)`` → the Maxima
    subscripted function ``sum[m,n](…)``, which the CAS ACCEPTS; ``\int (x+1)
    dx`` → the product ``dx*int*x+dx*int``, returned as a success for an
    ordinary indefinite integral; ``\int (x+1)^2 dx`` → ``(int*(x+1))^2*dx``,
    the one where the exponent belongs to the body and not to the prefix.

    Asserted on VALUES: these have arithmetic answers, and a shape assertion is
    what let `sum[m,n](…)` look like progress.
    """

    def setup_method(self):
        nb.start_engine(title="bracketbody", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _register(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        return self.eng.register_expression(canonical, latex=latex)

    def _value(self, latex):
        ex = self._register(latex)
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        return self.eng.expressions[step.output_refs[0]].canonical.strip()

    @pytest.mark.parametrize("text,value", [
        (r"\sum_{i=1}^{3} (i + i^2)", "20"),          # 2 + 6 + 12
        (r"\sum_{i=1}^{3} (i+1)^2", "29"),            # 4 + 9 + 16
        (r"\int_0^1 (x+1) dx", "3/2"),
        (r"\int_0^1 (x+1)^2 dx", "7/3"),
    ])
    def test_it_computes(self, text, value):
        assert self._value(text) == value

    def test_an_indefinite_integral_of_a_bracketed_integrand(self):
        r"""`\int (x+1) dx` came back as `dx*int*x+dx*int` — a SUCCESS."""
        out = self._value(r"\int (x+1) dx").replace(" ", "")
        assert "int" not in out and "dx" not in out, out
        assert out == "(x^2+2*x)/2", out

    @pytest.mark.parametrize("text", [
        r"\sum_i (x_i + y_i)",
        r"\sum_{i}(x_i + y_i)",
        r"\sum_{m,n} (h_{m,n} + g_{m,n})",
    ])
    def test_a_rangeless_bracketed_sum_is_a_sum(self, text):
        """No arithmetic answer exists here, so the CAS form is the evidence:
        a four-argument `sum(`, not the name `sum_i` nor `sum[m,n]`."""
        code = self.eng._handler.generate_backend_code(self._register(text).ir_ast)
        assert "sum_i" not in code and "sum[" not in code, code
        assert code.startswith("sum(") and _placeholder_ranges(code), code

    @pytest.mark.parametrize("text,expected", [
        (r"f_i(x)", "f_i(x)"),        # the shape the sum forms are told apart from
        (r"g(x)", "g(x)"),
        (r"\sin (x)", "sin(x)"),
    ])
    def test_an_ordinary_application_is_left_alone(self, text, expected):
        """The discriminator is the HEAD name (`sum`/`int`), nothing else."""
        code = self.eng._handler.generate_backend_code(self._register(text).ir_ast)
        assert code == expected


class TestOneWrittenIndexIsOneRange:
    r"""②-a — `\sum_i x_i + \sum_i y_i` must not end up with two ranges.

    The researcher's own rule: two rangeless sums share a range exactly when the
    same letter was written, so the allocation is keyed on the written index
    name.  `alphatize_indices` renames a colliding dummy (`i`, then `j`) and ran
    FIRST, so by the time the ranges were handed out the two bindings no longer
    looked alike.  The rename is right — it is alpha-conversion — it just cannot
    be what the range is keyed on, so the allocation moved to before it.
    """

    def setup_method(self):
        nb.start_engine(title="onerange", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _code(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        return self.eng._handler.generate_backend_code(ex.ir_ast)

    def test_the_same_written_index_gets_one_range(self):
        code = self._code(r"\sum_i x_i + \sum_i y_i")
        assert len(_placeholder_ranges(code)) == 1, code

    def test_different_written_indices_get_different_ranges(self):
        code = self._code(r"\sum_i x_i + \sum_j y_j")
        assert len(_placeholder_ranges(code)) == 2, code

    def test_two_expressions_never_share_a_range(self):
        """The other half of the design, unchanged: a shared range asserts an
        equality nobody stated, and per-command allocation made a separately
        registered `\\sum_i x_i` minus another come back as 0."""
        first = _placeholder_ranges(self._code(r"\sum_i x_i"))
        second = _placeholder_ranges(self._code(r"\sum_i x_i"))
        assert first and second and first != second

    def test_the_rename_does_not_happen_for_parallel_sums(self):
        r"""Reversed 2026-08-17 (EXPR-REF-PROVENANCE): the dummies stay AS
        WRITTEN.

        This assertion used to require `y(j)`, justified as "the dummies stay
        distinct, which is what keeps `\sum_i x_i - \sum_i y_i` from being read
        as one sum".  **Measured, that does not happen**: Maxima keeps the two
        noun sums as two terms whether the dummies match or not, and evaluating
        gives the correct term-by-term answer either way::

            ratsimp('sum(x(i),i,a,b) - 'sum(y(i),i,a,b))
                → 'sum(x(i),i,a,b)-'sum(y(i),i,a,b)
            ev('sum(x(i),i,1,3) - 'sum(y(i),i,1,3), sum)
                → -y(3)+x(3)-y(2)+x(2)-y(1)+x(1)

        The rename was not free, though: it is what kept `\sum_i x_i -
        \sum_i x_i` from reducing to 0, because Maxima's noun forms do not
        alpha-convert.  Sharing a range is still not sharing a binding — that
        part of the design stands, and two SEPARATE EXPRESSIONS still never
        share a range (see the test above).  What changed is that parallel
        binders in ONE expression have disjoint scopes, so renaming them
        prevents nothing."""
        code = self._code(r"\sum_i x_i + \sum_i y_i")
        assert "x(i)" in code and "y(i)" in code, code


class TestARangelessSumSurvivesTheSave:
    r"""②-c — the pair has to be REPRODUCIBLE, not merely unique.

    `Expression.ir_ast` is `exclude=True`, so a reloaded session has no IR and
    the placeholder range could not be derived: `\sum_i x_i` went back to Maxima
    as a two-argument `sum` and was refused (measured 2026-08-08).  The counter
    was already persisted — that only stops a LATER sum from colliding, it
    cannot hand an existing one its own pair back.  So the allocation is
    remembered per (expression, written index) and the IR is rebuilt from the
    canonical on first use, which is deterministic.

    The property is not "it works after a reload" but "it is the SAME": a
    session that reloads must send the CAS what the session that saved it sent,
    or a result computed before the save stops cancelling against its source.
    """

    def setup_method(self):
        nb.start_engine(title="sumsave", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _register(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        return self.eng.register_expression(canonical, latex=latex)

    def _simplify(self, engine, expr_id):
        step, _ = engine.run_operation("simplify", {"expr": expr_id})
        return step, engine.expressions[step.output_refs[0]].canonical

    def _round_trip(self, tmp_path, *latexes):
        """Register, compute, save, reload — return (live, reloaded) engines."""
        ids = [self._register(t).display_id for t in latexes]
        live = {eid: self._simplify(self.eng, eid)[1] for eid in ids}
        path = tmp_path / "s.json"
        self.eng.save(str(path))
        from notebook.engine import NotebookEngine
        reloaded = NotebookEngine.load(str(path))
        reloaded.start()
        return ids, live, reloaded

    def test_it_still_computes_and_gives_the_same_answer(self, tmp_path):
        ids, live, reloaded = self._round_trip(tmp_path, r"\sum_i x_i")
        step, out = self._simplify(reloaded, ids[0])
        assert step.status.value == "success", out
        assert out == live[ids[0]], (out, live[ids[0]])

    def test_the_shared_range_survives_too(self, tmp_path):
        """②-a and ②-c meet here: one written index, one range, across a save.

        Read off the RESULT the CAS returned, not `ir_ast`: the rebuild is lazy
        (an untouched expression still has no IR after a reload, exactly as
        before this fix), so the result canonical is where the ranges actually
        show up — and it is the artifact the next command has to send back."""
        ids, live, reloaded = self._round_trip(tmp_path, r"\sum_i x_i + \sum_i y_i")
        _, out = self._simplify(reloaded, ids[0])
        assert len(_placeholder_ranges(out)) == 1, out

    def test_a_sum_registered_after_the_reload_does_not_collide(self, tmp_path):
        ids, live, reloaded = self._round_trip(tmp_path, r"\sum_i x_i")
        canonical = reloaded._handler.latex_to_canonical(r"\sum_i z_i")
        fresh = reloaded.register_expression(canonical, latex=r"\sum_i z_i")
        before = _placeholder_ranges(self._simplify(reloaded, ids[0])[1])
        after = _placeholder_ranges(self._simplify(reloaded, fresh.display_id)[1])
        assert before and after and before.isdisjoint(after)

    def test_a_file_written_before_the_map_existed_still_loads(self, tmp_path):
        """Additive key: an older file has no map, and its sums are handed FRESH
        pairs on first use.  That is the safe direction — a new pair is never one
        another sum already holds, so nothing is claimed equal that was not."""
        import json
        ids, live, reloaded = self._round_trip(tmp_path, r"\sum_i x_i")
        path = tmp_path / "s.json"
        data = json.loads(path.read_text())
        data.pop("sum_bound_pairs", None)
        path.write_text(json.dumps(data))
        from notebook.engine import NotebookEngine
        older = NotebookEngine.load(str(path))
        older.start()
        step, out = self._simplify(older, ids[0])
        assert step.status.value == "success", out
        assert _placeholder_ranges(out), out


class TestABinderThatIsNotTheFirstFactor:
    r"""②-d — only the first factor of a product was examined.

    `\sum_i x_i \sum_j y_j` is ordinary writing, and the second Σ collapsed to
    the bare symbol `sum(j)`: `sum(x(i) * sum(j) * y(j), i, …)`, which Maxima
    refused (measured 2026-08-08).  A factor written BEFORE the binder was the
    same defect from the other side — `a \sum_i x_i` was not a sum at all.

    The reading is the conventional one: a binder takes everything written to
    its right, and what is written to its left is a factor beside it.
    """

    def setup_method(self):
        nb.start_engine(title="notfirst", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _value(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        return self.eng.expressions[step.output_refs[0]].canonical.strip()

    def _code(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        return self.eng._handler.generate_backend_code(ex.ir_ast)

    @pytest.mark.parametrize("text,value", [
        (r"\sum_{i=1}^{3} \sum_{j=1}^{2} i j", "18"),   # 6 * 3
        (r"2 \int_0^1 x dx", "1"),
        (r"\int_0^1 x dx \sum_{i=1}^{3} i", "3"),       # 1/2 * 6
    ])
    def test_it_computes(self, text, value):
        assert self._value(text) == value

    def test_a_factor_before_the_sum_stays_outside_it(self):
        code = self._code(r"a \sum_i x_i")
        assert code.startswith("a * sum("), code

    def test_the_second_sum_is_a_sum(self):
        r"""It used to be the bare symbol `sum(j)` inside the first sum's body."""
        code = self._code(r"\sum_i x_i \sum_j y_j")
        assert code.count("sum(") == 2 and "sum(j)" not in code, code
        assert len(_placeholder_ranges(code)) == 2, code   # different letters

    def test_an_integral_inside_a_sum_is_an_integral(self):
        assert self._value(r"\sum_i \int_0^1 x_i t dt").replace(" ", "") \
            == "('sum(x(i),i,nbSumLo1,nbSumHi1))/2"

    def test_a_binder_with_nothing_to_bind_is_still_refused(self):
        r"""`\sum_j` alone is not a sum over an empty body.  Answering
        `sum(1, j, …)` would be plausible and invented; the CAS refusing the
        command out loud is the better failure, so this stays as it was."""
        with pytest.raises(Exception):
            self._value(r"\sum_j")


class TestTheCASNounFormCanBeReadBack:
    r"""②-e — `'sum(…)` is what the CAS returns, and it could not be parsed.

    A leading quote is the noun form ("this call, unevaluated").  Only `'diff`
    was read; `'sum` / `'integrate` raised, and the failure took the WHOLE
    expression down to a single Symbol.  That surfaced the moment such a result
    was referenced back into a formula — `E1 - E2` re-registers the spliced
    text — so the left half never reached the CAS as a sum and the answer came
    back as a SUCCESS (measured 2026-08-08).
    """

    def setup_method(self):
        nb.start_engine(title="noun", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    @pytest.mark.parametrize("canonical", [
        "'sum(x(i),i,a,b)", "'integrate(f,x)", "'limit(f,x,0)", "'diff(y,x,2)",
    ])
    def test_a_noun_call_round_trips(self, canonical):
        """The mark stays ON the name: unquoted, `integrate(f,x)` is a different
        request from the `'integrate(f,x)` the CAS answered with."""
        ex = self.eng.register_expression(canonical)
        code = self.eng._handler.generate_backend_code(ex.ir_ast)
        assert code.replace(" ", "") == canonical.replace(" ", ""), code

    def test_a_referenced_sum_result_still_reaches_the_cas_as_a_sum(self):
        from ui.interpret_pipeline import _expand_expr_ids
        h = self.eng._handler
        src = self.eng.register_expression(h.latex_to_canonical(r"\sum_i x_i"),
                                           latex=r"\sum_i x_i")
        step, _ = self.eng.run_operation("simplify", {"expr": src.display_id})
        result = self.eng.expressions[step.output_refs[0]]
        assert result.canonical.startswith("'sum("), result.canonical

        formula = f"{src.display_id} - {result.display_id}"
        expanded = _expand_expr_ids(formula, self.eng)
        combined = self.eng.register_expression(
            h.latex_to_canonical(expanded) or expanded)
        code = h.generate_backend_code(combined.ir_ast)
        # BOTH halves are four-argument sums; before, neither was — the whole
        # expression stayed the raw text it was spliced from.
        assert code.count("sum(") == 2, code
        assert "\\sum" not in code, code

    @pytest.mark.parametrize("text", ["'(x+1)", "'2", "'"])
    def test_a_quote_on_something_that_is_not_a_name_is_not_a_noun(self, text):
        """A noun form quotes a CALL.  Anything else keeps the parser's existing
        answer — the whole text as one opaque Symbol, which the CAS then refuses
        — rather than being invented into a call that was never written."""
        from notebook.models.ir import Symbol as IRSymbol
        from notebook.runtime.shared_parser import SharedParser
        node = SharedParser().parse(text)
        assert isinstance(node, IRSymbol) and node.name == text, repr(node)

    def test_a_trailing_prime_is_untouched(self):
        r"""`x'` is the creation-operator / derivative prime and is POSTFIX;
        only a LEADING quote is a noun mark."""
        from notebook.models.ir import PrimeExpr
        from notebook.runtime.shared_parser import SharedParser
        assert isinstance(SharedParser().parse("x'"), PrimeExpr)


class TestHowTheMeasureIsWritten:
    r"""②-f — `\int dx f`, `\int \frac{d\omega}{2\pi} f`, `\int dx dy f`.

    Only two spellings were read: one trailing `dx`, or one leading `dx`.  The
    ones a Green's function calculation is actually written in were not, and
    each came back as a SUCCESS (measured 2026-08-08):

    * ``\int \frac{d\omega}{2\pi} G(\omega)`` → ``(d·int·omega·G(omega))/(2π)``.
      ``\frac{dk}{2\pi}`` worked only because `dk` is ONE token; a command-spelled
      variable must be separated from the `d` (`_separate_adjacent_commands`,
      CLAUDE.md), so the measure arrived as two symbols.
    * ``\int dx dy x y`` → ``(dy·x²·y)/2`` — the second differential stayed a
      FACTOR.
    * ``\int \frac{d^3k}{(2\pi)^3} f(k)`` → ``(d³·int·k·f(k))/(8π³)``.  There is
      no d-dimensional integral to lower this to, so it is refused BY NAME
      rather than computed as a product.
    """

    def setup_method(self):
        nb.start_engine(title="measure", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _value(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        return self.eng.expressions[step.output_refs[0]].canonical.strip()

    @pytest.mark.parametrize("text,value", [
        # the spellings that already worked — pinned so they stay working
        (r"\int_0^1 x^2 dx", "1/3"),
        (r"\int_0^1 dx x^2", "1/3"),
        (r"\int_0^1 \frac{dx}{2\pi} x^2", "1/(6*%pi)"),
        # the ones that were silently wrong
        # The written range decorates the OUTER ∫ (the sign it was written on);
        # the inner one is an antiderivative, so `y` survives.
        (r"\int_0^1 dx dy x y", "y^2/4"),
        (r"\int_0^1 \int_0^1 x y dx dy", "1/4"),      # must not regress
        (r"\int_0^1 dx \int_0^1 dy x y", "1/4"),
    ])
    def test_it_computes(self, text, value):
        assert self._value(text).replace(" ", "") == value

    def test_a_command_spelled_variable_is_still_a_differential(self):
        r"""`d\omega` — the measure of every frequency integral."""
        out = self._value(r"\int \frac{d\omega}{2\pi} G(\omega)").replace(" ", "")
        assert out == "('integrate(G(omega),omega))/(2*%pi)", out

    def test_two_leading_differentials_are_two_integrals(self):
        canonical = self.eng._handler.latex_to_canonical(r"\int dx dy x y")
        ex = self.eng.register_expression(canonical, latex=r"\int dx dy x y")
        code = self.eng._handler.generate_backend_code(ex.ir_ast)
        assert code == "integrate(integrate(x * y, y), x)", code

    @pytest.mark.parametrize("text,dim", [
        (r"\int \frac{d^3k}{(2\pi)^3} f(k)", 3),
        (r"\int d^3k f(k)", 3),
        (r"\int \frac{d^2k}{(2\pi)^2} f(k)", 2),
    ])
    def test_a_volume_measure_is_read_and_kept(self, text, dim):
        """It REGISTERS — the researcher can write it and see it — and the IR
        carries the dimension.  Refusing at the parser would have made the
        expression unwritable, which is not an answer to `\\int d^3k`."""
        canonical = self.eng._handler.latex_to_canonical(text) or text
        ex = self.eng.register_expression(canonical, latex=text)
        assert ex.latex == text
        assert getattr(ex.ir_ast, "measure_dim", None) == dim

    @pytest.mark.parametrize("text", [
        r"\int \frac{d^3k}{(2\pi)^3} f(k)",
        r"\int d^3k f(k)",
    ])
    def test_a_volume_measure_is_refused_by_name_at_the_cas(self, text):
        """There is no d-dimensional integral to hand Maxima, so it is refused
        where it would be lowered — by name, with what to do about it."""
        canonical = self.eng._handler.latex_to_canonical(text) or text
        ex = self.eng.register_expression(canonical, latex=text)
        with pytest.raises(Exception) as exc:
            self.eng._handler.generate_backend_code(ex.ir_ast)
        assert "multidim_measure" in str(exc.value), str(exc.value)

    def test_the_dimension_survives_the_resolver_walks(self):
        r"""Every walker that REBUILDS an `Integral` used to drop a field it did
        not know about, so `measure_dim` arrived as None and `\int d^3k f(k)`
        lowered to the ONE-variable `integrate(f(k), k)` — silently wrong in a
        new way.  They rebuild with `model_copy` now."""
        from notebook.models.ir import Integral, Symbol as S
        from notebook.runtime.index_contractor import alphatize_indices, contract_indices
        node = Integral(expr=S(name="f"), var=S(name="k"), measure_dim=3)
        assert alphatize_indices(node).measure_dim == 3
        assert contract_indices(node, SymbolContext()).measure_dim == 3

    def test_a_measure_is_not_reported_as_an_unreadable_derivative(self, caplog):
        r"""`\frac{dk}{2\pi}` is a MEASURE.  Warning on numerator evidence alone
        put "could not read this as a derivative" under an integral that was
        read perfectly — a derivative needs a differential on BOTH sides."""
        import logging
        with caplog.at_level(logging.WARNING):
            self.eng._handler.latex_to_canonical(r"\frac{dk}{2\pi}")
            self.eng._handler.latex_to_canonical(r"\frac{d\omega}{2\pi}")
        assert "deriv" not in caplog.text and "微分の表記" not in caplog.text, caplog.text

    def test_a_real_unreadable_derivative_still_warns(self, caplog):
        r"""BP-9's own case: `\frac{delta}{dx}` became `'diff(elta,x,1)`."""
        import logging
        with caplog.at_level(logging.WARNING):
            self.eng._handler.latex_to_canonical(r"\frac{delta}{dx}")
        assert "微分の表記" in caplog.text, caplog.text

    @pytest.mark.parametrize("text,expected", [
        (r"\frac{dy}{dx}", "'diff(y,x,1)"),
        (r"\frac{\partial phi}{\partial x}", "'diff(phi,x,1)"),
    ])
    def test_ordinary_derivatives_are_untouched(self, text, expected):
        assert self.eng._handler.latex_to_canonical(text) == expected


class TestTheDisplayedSumCanBeReadBack:
    r"""SUM-DUMMY-NAMES, second half: what a perturbation expansion looks like.

    Expanding perturbatively multiplies the factors, and each contraction adds a
    summation index.  Two things made the result unpublishable — and the first
    of them was not a matter of taste:

    **The indices came back with no separator.**  QuantumAlgebra writes a term's
    indices concatenated (`V(#₁#₂#₃#₄)`), so substituting names into that gave
    `V_{mnpq}` and `\sum_{mnpqk_{5}}`.  By this project's OWN rule that is one
    index named `mnpq` (CLAUDE.md, AUDIT-SUBSCRIPT-SPLIT) — so the display did
    not round-trip: feeding `\sum_{mn} h_{mn}` back in gave a SINGLE sum over an
    index called `mn` (measured 2026-08-08).  "What is displayed must be what
    can be computed with" is the invariant, and this broke it.

    **A dummy the BACKEND introduced was named from another series.**  The
    researcher's own `m, n, p, q` were recovered correctly; only the index QA
    added for the contraction was neutral — and `k_{5}` standing beside
    `m, n, p, q` is not a line anyone would put in a paper.  It continues the
    family instead (`r`), skipping every name already in play.
    """

    def setup_method(self):
        nb.start_engine(title="ptnames", mode="second_quantization",
                        language="ja", domains=["condensed_matter"])
        self.eng = nb.get_engine()
        self.eng.run_operation("define_fermion_operator", {"name": "c"})

    def teardown_method(self):
        nb.stop_engine()

    def _commutator_latex(self):
        h0 = self.eng.register_expression(r"\sum_{m,n} h_{m,n} c'_{m} c_{n}")
        v = self.eng.register_expression(
            r"\sum_{m,n,p,q} V_{m,n,p,q} c'_{m} c'_{n} c_{p} c_{q}")
        step, _ = self.eng.run_operation(
            "commutator", {"a": h0.display_id, "b": v.display_id})
        return self.eng.expressions[step.output_refs[0]].latex

    def test_the_indices_are_separated(self):
        out = self._commutator_latex()
        assert "V_{m,n,p,q}" in out, out
        assert "V_{mnpq}" not in out, out

    def test_a_backend_introduced_dummy_spends_no_new_letter(self):
        r"""A bound index lives inside one sum and means nothing outside it, so
        taking a fresh letter for it costs the researcher something real for
        nothing — after `m, n, p, q` came `r`, then `s`, then `t`, and the
        alphabet left for actual quantities shrank with every commutator
        (raised by the researcher).  The extras are their OWN letters,
        subscripted and cycled: `m_1, n_1, p_1, q_1, m_2, …`"""
        out = self._commutator_latex()
        assert r"\sum_{m,n,p,q,m_{1}}" in out, out
        assert "k_{5}" not in out and "k_{1}" not in out, out
        for spent in ("r", "s", "t"):
            assert f",{spent}}}" not in out, (spent, out)

    def test_the_researchers_own_names_are_still_recovered(self):
        """The half that already worked must keep working."""
        h0 = self.eng.register_expression(r"\sum_{m,n} h_{m,n} c'_{m} c_{n}")
        step, _ = self.eng.run_operation("normal_order", {"expr": h0.display_id})
        out = self.eng.expressions[step.output_refs[0]].latex
        assert r"\sum_{m,n}" in out and "h_{m,n}" in out, out


class TestTheDisplayRoundTrips:
    """The displayed sum, fed back in, is the same sum."""

    def setup_method(self):
        nb.start_engine(title="rt", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _cas_of(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        return self.eng._handler.generate_backend_code(ex.ir_ast)

    def test_a_separated_two_index_sum_is_a_double_sum(self):
        code = self._cas_of(r"\sum_{m,n} h_{m,n}")
        assert code.count("sum(") == 2 and "h(m, n)" in code, code

    def test_the_unseparated_form_is_a_SINGLE_index_and_that_is_why(self):
        r"""Not a bug — the rule.  `h_{mn}` IS one index called `mn`, which is
        exactly why the display may not write it that way."""
        code = self._cas_of(r"\sum_{mn} h_{mn}")
        assert code.count("sum(") == 1 and "h(mn)" in code, code


class _Reg:
    def __init__(self, names):
        self.registry = {n: None for n in names}


class TestFreshDummyNames:
    """The naming, in isolation: spend no letters, collide with nothing."""

    def _mixin(self):
        """The mixin alone — the naming needs no backend, only the text."""
        from notebook.runtime.julia_latex_mixin import JuliaLatexMixin
        return JuliaLatexMixin()

    def test_it_cycles_the_written_letters(self):
        r"""`m_1, n_1, p_1, q_1, m_2, …` — cycling keeps the SUBSCRIPTS small,
        which is what a reader has to hold in their head."""
        assert self._mixin()._fresh_dummy_names("", ["m", "n", "p", "q"])[:5] == [
            "m_{1}", "n_{1}", "p_{1}", "q_{1}", "m_{2}"]

    def test_it_spends_no_letter_the_researcher_could_still_use(self):
        fresh = self._mixin()._fresh_dummy_names("", ["m", "n", "p", "q"])[:8]
        assert not any(len(n) == 1 for n in fresh), fresh

    def test_a_declared_symbol_is_skipped(self):
        r"""A researcher who declared `r` as a radius had it taken by a dummy;
        the same must not happen to a declared `m_1`."""
        fresh = self._mixin()._fresh_dummy_names("", ["m", "n"], _Reg(["m_1"]))
        assert fresh[0] == "n_{1}", fresh

    def test_a_literal_index_such_as_a_spin_label_is_skipped(self):
        """The `u` in `c_{m,u}` is an index too, so it cannot become a base."""
        fresh = self._mixin()._fresh_dummy_names(r"{c}_{m,u}", ["m"])
        assert all(n.startswith("m_") for n in fresh[:3]), fresh

    def test_with_nothing_written_it_takes_one_letter_and_subscripts_that(self):
        r"""A sum the backend produced from nothing has no letter to build on.
        Only here is a letter spent, and only one."""
        fresh = self._mixin()._fresh_dummy_names("", [])
        assert fresh[:2] == ["i_{1}", "i_{2}"], fresh

    def test_a_latex_command_does_not_consume_its_letters(self):
        r"""`\sum` must not take `s`, `u` and `m` out of play."""
        fresh = self._mixin()._fresh_dummy_names(r"\sum_{} \psi", [])
        assert fresh[0] == "i_{1}", fresh
