"""Indexed FUNCTIONS: ``a_i(k)``, ``A_{ij}(x)`` (2026-07-30).

A subscripted name applied to arguments reaches the resolver as
``Subscript(base, FunctionCall(...))`` — the parser is semantics-free, so it
cannot tell ``a_i(k)`` (a call on ``a_i``) from ``a`` subscripted by ``i(k)``.
The unbraced case was folded into a function name back in 2026-07-10
(``bessel_j(0,x)``); the BRACED case was not, and fell through to
``_extract_indices``, crashing with a raw pydantic ValidationError in the
researcher's face.

DESIGN DECISION (user, 2026-07-30): ``a_i`` and ``a_i(k)`` are DIFFERENT objects
and stay that way.  The subscript is an INDEX on a bare symbol (``a_i`` → ``a(i)``)
and part of the NAME when applied (``a_i(k)`` → ``a_i(k)``).  In physics these are
routinely distinct — a component versus a mode function — so this is not unified.
Substituting one must therefore leave the other alone; that is pinned below so a
later "consistency fix" cannot quietly merge them.
"""
import pytest

from notebook.engine import NotebookEngine


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="indexed fn") as e:
        yield e


class TestLowering:

    @pytest.mark.parametrize("expr,expected", [
        ("a_i(k)", "a_i(k)"),                  # single index folds into the name
        ("a_{i}(k)", "a_i(k)"),                # braced spelling of the same
        ("a_{1}(k)", "a_1(k)"),                # a digit run is one index too
        ("bessel_j(0,x)", "bessel_j(0, x)"),   # the 2026-07-10 case still folds
        # AUDIT-SUBSCRIPT-SPLIT made an unseparated run ONE index, so the
        # applied form folds like any other single index.  It was refused as
        # "multi-index" until 2026-07-30, because this branch restated the
        # splitting rule instead of asking it.
        ("A_{ij}(x)", "A_ij(x)"),
        ("T_{ijk}(x)", "T_ijk(x)"),
        (r"a_{\sigma}(k)", "a_sigma(k)"),
        # A BARE digit subscript.  It tokenizes as a Number rather than a Symbol,
        # and the bare branch used to test for a Symbol, so these were refused —
        # while the braced spelling of the same quantity was accepted.  Same
        # tokenization as the archived SUBST-DIGIT-SUB.
        ("G_0(k, omega)", "G_0(k, omega)"),
        ("H_0(t)", "H_0(t)"),
        ("omega_0(k)", "omega_0(k)"),
        ("G_12(x)", "G_12(x)"),
    ])
    def test_single_index_folds_into_a_function_name(self, engine, expr, expected):
        assert engine._handler.to_backend_code(expr).code == expected

    @pytest.mark.parametrize("bare,braced", [
        ("G_0(k, omega)", "G_{0}(k, omega)"),
        ("G_i(k)",        "G_{i}(k)"),
        ("G_12(x)",       "G_{12}(x)"),
    ])
    def test_the_braced_and_bare_spellings_agree(self, engine, bare, braced):
        """Two spellings of one quantity must not disagree — `G_0(k,omega)` was
        refused while `G_{0}(k,omega)` was accepted.  The free Green's function
        with arguments is about as basic as many-body notation gets."""
        assert (engine._handler.to_backend_code(bare).code
                == engine._handler.to_backend_code(braced).code)

    @pytest.mark.parametrize("latex,canonical", [
        # digits that are part of a NAME — a call, not a product
        (r"A_{3}(x)",                  "A_3(x)"),
        (r"A_{12}(x)",                 "A_12(x)"),
        (r"G_{0}\left(k , \omega\right)", "G_0(k , omega)"),
        (r"A_{3}(x)\,B_{3}(x)",        "A_3(x)*B_3(x)"),
        # digits that really are numeric literals — still multiply
        (r"2(x+1)",                    "2*(x+1)"),
        (r"2 (x+1)",                   "2*(x+1)"),
        (r"x^2(y)",                    "x^2*(y)"),
    ])
    def test_a_digit_subscript_reads_back_as_a_call_not_a_product(
            self, engine, latex, canonical):
        r"""Copying a displayed `G_{0}(k,\omega)` back into the input box must
        give the same expression.  `_implicit_multiplication`'s "digit before
        `(` is a product" rule fired on the last digit of the NAME, so it came
        back as `G_0*(k, omega)` — silently a different expression, and exactly
        the shape expanding a sum produces (`A_{3}(x)`)."""
        assert engine._handler.latex_to_canonical(latex) == canonical

    def test_the_displayed_result_round_trips(self, engine):
        """End to end: register, operate, then re-read what the timeline showed."""
        for src in ("G_0(k, omega)", "H_0(t) + H_1(t)", "G_i(k)"):
            ex = engine.register_expression(src)
            step, _ = engine.run_operation("expand", {"expr": ex.display_id})
            shown = engine.expressions[step.output_refs[0]].latex
            back = engine._handler.latex_to_canonical(shown)
            assert "*(" not in back, f"{src}: {shown!r} came back as {back!r}"

    def test_the_fold_follows_the_session_splitting_rule(self):
        """The splitting rule and the fold are ONE rule, so the session decides.

        Outside an einstein session ``A_{ij}`` is one index, so the applied form
        folds into the name ``A_ij``.  Inside one it is TWO, so the very same
        spelling becomes a subscripted function ``A[i,j](x)`` — the multi-index
        branch — without either place restating what "one index" means."""
        with NotebookEngine(title="einstein fn") as e:
            e.symbol_context.einstein_summation = True
            assert e._handler.to_backend_code("A_{ij}").code == "A(i, j)"
            for spelling in ("A_{ij}(x)", "A_ij(x)"):   # braced AND bare
                assert e._handler.to_backend_code(spelling).code == "A[i, j](x)"

    @pytest.mark.parametrize("expr,expected", [
        ("a_i", "a(i)"),                 # bare: the subscript is an INDEX
        ("A_{i,j}", "A(i, j)"),          # separated → two indices
        ("A_{i j}", "A(i, j)"),
        ("A_{ij}", "A(ij)"),             # unseparated → ONE index
        ("T_{ijk}", "T(ijk)"),
    ])
    def test_bare_subscript_is_still_an_index(self, engine, expr, expected):
        assert engine._handler.to_backend_code(expr).code == expected


class TestMultiIndexKeepsItsIndices:
    r"""A MULTI-index subscript applied to arguments is Maxima's own *subscripted
    function* — `D_{i,j}(x)` → `D[i,j](x)` (2026-07-31).

    These were refused until the Maxima manual settled it: the manual calls
    `dd[x](y) := ...` "one of a family of subscripted functions selected by a
    particular subscript value", which is exactly the notation.  Measured on real
    Maxima: undefined it has no side effect (`arrays` stays empty), `subst`
    reaches the index, a sum over the index expands term by term, and `tex1`
    returns `D_{i,j}(x)` — the researcher's own spelling, so the round trip
    closes with no display bridge.

    Two indices is the safe line.  A LABEL is never written with a separator
    (nobody spells the joint distribution `p_{x,y}`), so a separated subscript is
    unambiguously an index.  The SINGLE-index case is genuinely ambiguous and is
    pinned as unchanged in `TestSingleIndexStillFolds`.
    """

    @pytest.mark.parametrize("expr,expected", [
        ("A_{i,j}(x)",          "A[i, j](x)"),      # separated → two indices
        ("A_{i j}(x)",          "A[i, j](x)"),      # space is the same separator
        ("A_{i,j,k}(x)",        "A[i, j, k](x)"),
        ("M_{i,j}(x)",          "M[i, j](x)"),
        (r"a_{i,\sigma}(k)",    "a[i, sigma](k)"),  # 2nd quantization: site + spin
        (r"g_{\mu\nu}(x)",      "g[mu, nu](x)"),    # the metric tensor field
    ])
    def test_lowered_to_a_subscripted_function(self, engine, expr, expected):
        assert engine._handler.to_backend_code(expr).code == expected

    def test_the_display_is_the_researchers_own_notation(self, engine):
        """`tex1` of the bracket form gives the subscript back, so nothing has to
        restore it — unlike `D(i,j)(x)`, whose tex1 leaks CAS syntax AND comes
        back as a product."""
        code = engine._handler.to_backend_code("D_{i,j}(x)").code
        shown = engine._adapter.execute(f"tex1({code})").value
        assert shown == "D_{i,j}(x)"
        assert engine._handler.latex_to_canonical(shown) == "D_{i,j}(x)"

    def test_the_index_stays_an_index(self, engine):
        """The point of not folding into a name: substitution and summation
        still reach the subscript."""
        code = engine._handler.to_backend_code("D_{i,j}(x)").code
        assert engine._adapter.execute(f"subst(i=1, {code})").value == "D[1,j](x)"
        summed = engine._adapter.execute(
            "sum(A[i](x)*B[i](x), i, 1, 3)").value.replace(" ", "")
        assert summed.count("A[") == 3 and "3*A[i]" not in summed

    def test_a_chained_subscript_base_is_still_refused(self, engine):
        """`A_i_j(x)` — the base is itself a Subscript, which has no reading
        here.  It used to raise a raw pydantic ValidationError."""
        with pytest.raises(ValueError) as exc:
            engine.register_expression("A_i_j(x)")
        assert "ValidationError" not in str(exc.value)
        assert "unsupported_notation" not in str(exc.value), "key not translated"

    def test_the_message_exists_in_both_languages(self):
        from notebook.runtime.i18n import t
        ja = t("parser.unsupported_notation.called_subscript", "ja")
        en = t("parser.unsupported_notation.called_subscript", "en")
        assert ja != en
        for msg in (ja, en):
            assert msg and "unsupported_notation" not in msg

    def test_the_bare_form_is_unchanged(self, engine):
        """Bare and applied stay DIFFERENT objects (CLAUDE.md), and the bare
        lowering is untouched by this change."""
        assert engine._handler.to_backend_code(r"a_{i,\sigma}").code == "a(i, sigma)"
        assert engine._handler.to_backend_code("M_{i,j}").code == "M(i, j)"


class TestSingleIndexStillFolds:
    r"""The single-index applied form is NOT decided by this change.

    `C_i(t)` wants an index and `p_x(x)` wants a label, and they are the same
    shape.  Measured 2026-07-31: treating the single case as an index destroys
    meaning — with `p[x](x)` and `p[y](y)`, substituting `x=1, y=1` gives
    `p[1](1)^2`, collapsing the x-marginal and the y-marginal into one object
    with no error (it broke the mutual-information formula).  So it keeps
    folding until a discriminator exists (open_items AUDIT-INDEXED-CALL).
    """

    @pytest.mark.parametrize("expr,expected", [
        ("C_i(t)",          "C_i(t)"),
        ("p_x(x)",          "p_x(x)"),
        ("p_xy(x,y)",       "p_xy(x, y)"),
        ("bessel_j(0,x)",   "bessel_j(0, x)"),
        ("G_0(k, omega)",   "G_0(k, omega)"),
        ("A_{ij}(x)",       "A_ij(x)"),
    ])
    def test_unchanged(self, engine, expr, expected):
        assert engine._handler.to_backend_code(expr).code == expected

    def test_the_marginals_that_would_have_collapsed(self, engine):
        """The concrete regression the narrow line avoids."""
        both = engine._adapter.execute(
            "subst([x=1,y=1], p_x(x)*p_y(y))").value.replace(" ", "")
        assert both == "p_x(1)*p_y(1)"


class TestBoundIndexIsAnIndex:
    r"""Inside `\sum_i`, a single subscript naming `i` IS an index (2026-07-31).

    The sum says so — no guessing, which is what makes this discriminator usable
    where the bare `C_i(t)` still has none.  Before it, the componentwise dot
    product of vector analysis was a silent wrong answer:

        sum(A(i)*B(i),     i, 1, 3)  ->  A(3)B(3)+A(2)B(2)+A(1)B(1)   correct
        sum(A_i(x)*B_i(x), i, 1, 3)  ->  3*A_i(x)*B_i(x)              wrong

    Reading the dummy BEFORE the body is what this needed; two of the three
    IndexSum construction sites were written the other way round.
    """

    @pytest.mark.parametrize("expr,expected", [
        (r"\sum_i A_i(x)*B_i(x)", "sum(A[i](x) * B[i](x), i)"),
        (r"\sum_i C_i(t)",        "sum(C[i](t), i)"),
        (r"\sum_k E_k(x)^2",      "sum(E[k](x)^2, k)"),
    ])
    def test_the_bound_subscript_becomes_an_index(self, engine, expr, expected):
        assert engine._handler.to_backend_code(expr).code == expected

    def test_the_dot_product_expands_term_by_term(self, engine):
        """The physics this exists for."""
        out = engine._adapter.execute(
            "sum(A[i](x)*B[i](x), i, 1, 3)").value.replace(" ", "")
        assert out.count("A[") == 3 and "3*A[i]" not in out

    def test_only_the_bound_name_is_affected(self, engine):
        r"""`\sum_i` binds `i`, not `x` — the marginal keeps its label."""
        assert engine._handler.to_backend_code(r"\sum_i p_x(x)").code == \
            "sum(p_x(x), i)"

    def test_outside_a_sum_nothing_changed(self, engine):
        assert engine._handler.to_backend_code("A_i(x)").code == "A_i(x)"

    def test_the_binding_does_not_leak_past_the_sum(self, engine):
        r"""The stack must unwind: `\sum_i A_i(x)` followed by a bare `C_i(t)`
        in the same expression must leave the bare one folded."""
        code = engine._handler.to_backend_code(r"\sum_i A_i(x) + C_i(t)").code
        assert "A[i](x)" in code and "C_i(t)" in code


class TestOperationsWork:

    def test_expand_and_display(self, engine):
        ex = engine.register_expression("a_i(k) + b")
        st, val = engine.run_operation("expand", {"expr": ex.display_id})
        assert st.status.value == "success"
        assert val.replace(" ", "") == "a_i(k)+b"
        # The researcher sees their own notation, not the CAS name.
        assert engine.expressions[st.output_refs[0]].latex == \
            r"a_{i}\left(k\right)+b"

    def test_substitute_into_a_called_subscript(self, engine):
        ex = engine.register_expression("a_i(k) + b")
        st, val = engine.run_operation(
            "substitute", {"expr": ex.display_id, "lhs": "a_i(k)", "rhs": "7"})
        assert st.status.value == "success"
        assert val.replace(" ", "") == "b+7"


class TestBareAndCalledStayDistinct:
    """The design decision, pinned."""

    def test_they_lower_to_different_cas_objects(self, engine):
        assert engine._handler.to_backend_code("a_i + a_i(k)").code == \
            "a(i) + a_i(k)"

    def test_substituting_the_bare_symbol_leaves_the_function_alone(self):
        with NotebookEngine(title="distinct") as e:
            ex = e.register_expression("a_i + a_i(k)")
            st, val = e.run_operation(
                "substitute", {"expr": ex.display_id, "lhs": "a_i", "rhs": "5"})
            assert st.status.value == "success"
            assert val.replace(" ", "") == "a_i(k)+5"


class TestSingleIndexResolutionOrder:
    r"""What the researcher has already said beats a default (2026-07-31).

    Until stage 3, a single-index applied form was folded no matter what — a
    declared matrix base, an explicit `index_lowering_style`, even a recorded
    answer were all ignored.  The order now matches the notation-confirmation UI
    documented in CLAUDE.md: builtin family > sum binding > declaration >
    remembered answer > fold.
    """

    def _code(self, src, setup=None, **kw):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        ctx = SymbolContext()
        if setup:
            setup(ctx)
        h = MaximaBackendHandler(None)
        h.set_symbol_context(ctx)
        return h.to_backend_code(src).code

    def test_nothing_said_still_folds(self):
        assert self._code("C_i(t)") == "C_i(t)"

    @pytest.mark.parametrize("algebra", ["matrix", "vector", "tensor"])
    def test_a_declared_indexed_base_makes_it_an_index(self, algebra):
        from notebook.models.quantity import MathType, Quantity
        code = self._code("C_i(t)", lambda c: c.register(
            "C", Quantity(name="C", kind="anonymous",
                          math_type=MathType(algebra=algebra))))
        assert code == "C[i](t)"

    def test_an_explicit_lowering_style_makes_it_an_index(self):
        from notebook.models.quantity import Quantity
        code = self._code("C_i(t)", lambda c: c.register(
            "C", Quantity(name="C", kind="anonymous",
                          index_lowering_style="array_access")))
        assert code == "C[i](t)"

    def test_a_declared_scalar_base_is_not_an_answer(self):
        """`p` is a scalar and `p_x(x)` still wants its subscript as a label."""
        from notebook.models.quantity import Quantity
        code = self._code("p_x(x)", lambda c: c.register(
            "p", Quantity(name="p", kind="scalar")))
        assert code == "p_x(x)"

    @pytest.mark.parametrize("role,expected", [
        ("index", "C[i](t)"), ("name", "C_i(t)")])
    def test_a_remembered_answer_is_honoured(self, role, expected):
        from notebook.runtime.subscript_policy import APPLIED_SUBSCRIPT_KEY
        code = self._code("C_i(t)", lambda c: c.notation_usage.update(
            {APPLIED_SUBSCRIPT_KEY + "C": role}))
        assert code == expected

    def test_a_builtin_family_is_never_reinterpreted(self):
        """Even with a recorded 'index' for the base, `bessel_j` is a function."""
        from notebook.runtime.subscript_policy import APPLIED_SUBSCRIPT_KEY
        code = self._code("bessel_j(0,x)", lambda c: c.notation_usage.update(
            {APPLIED_SUBSCRIPT_KEY + "bessel": "index"}))
        assert code == "bessel_j(0, x)"

    def test_the_family_list_comes_from_the_backend_spec(self):
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        known = MaximaSemanticResolver.known_function_names
        for name in ("bessel_j", "legendre_p", "wigner_3j", "cdf_student_t",
                     "annuity_fv", "maximize_lp"):
            assert name in known, name


class TestTheDialogAsksOnlyWhenNothingElseAnswers:
    """Stage 3's confirmation UI — the last resort, not the first."""

    def _ask(self, text, setup=None, lang="ja"):
        from ui.interpret_notation import _applied_subscript_candidates
        with NotebookEngine(title="dlg") as e:
            if setup:
                setup(e.symbol_context)
            return _applied_subscript_candidates(e, lang, text, [])

    def test_it_asks_for_the_undecidable_case(self):
        got = self._ask("C_i(t)")
        assert got and got["status"] == "needs_confirmation"
        assert [c["args"]["role"] for c in got["candidates"]] == ["index", "name"]
        assert got["args"]["token"] == "applied_subscript:C"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_both_languages_are_filled_in(self, lang):
        got = self._ask("C_i(t)", lang=lang)
        for text in [got["confirmation_prompt"]] + \
                [c[f"display_{lang}"] for c in got["candidates"]]:
            assert "{" not in text and "notation." not in text

    @pytest.mark.parametrize("text,why", [
        ("bessel_j(0,x)",         "CAS builtin family"),
        ("G_0(k, omega)",         "a digit subscript is settled by the naming rule"),
        (r"\sum_i A_i(x)*B_i(x)", "the sum already answers it"),
        ("D_{i,j}(x)",            "multi-index is not ambiguous"),
        ("a*x + b",               "no applied subscript at all"),
    ])
    def test_it_stays_quiet_when_something_else_answers(self, text, why):
        assert self._ask(text) is None, why

    def test_a_declaration_silences_it(self):
        from notebook.models.quantity import MathType, Quantity
        assert self._ask("C_i(t)", lambda c: c.register(
            "C", Quantity(name="C", kind="anonymous",
                          math_type=MathType(algebra="matrix")))) is None

    def test_it_never_asks_twice(self):
        from notebook.runtime.subscript_policy import APPLIED_SUBSCRIPT_KEY
        assert self._ask("C_i(t)", lambda c: c.notation_usage.update(
            {APPLIED_SUBSCRIPT_KEY + "C": "name"})) is None
