"""Complex conjugation must be INPUTTABLE, in the spellings physics uses.

A conjugate appears without any operator algebra — a wave function in a
magnetic field is the everyday case — so this is not second-quantization
territory.  Measured 2026-08-09, one spelling out of six reached the CAS
correctly and the rest were silently wrong:

    \\psi^{\\ast}(x)   → psi^[ast](x)     a symbol raised to a LIST
    \\psi^{*}(x)      → psi^(*)*(x)      braces gone, star read as a factor
    \\psi^\\ast(x)     → psi^\\ast(x)      a backslash reaching the CAS
    \\psi^*_{n}(x)    → REFUSED          "multiple indices" — it has one
    \\overline{f(x)}  → f(x)_bar         conjugation dropped, no error
    \\overline{a+b}   → a+b_bar          and only `b` decorated

The braced spellings matter most: a TeX editor emits them, and so does this
system's own display, so `\\psi^{\\ast}(x)` must type back in.
"""

import pytest

from notebook.engine import NotebookEngine
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser


def _cas(text: str) -> str:
    """The CAS code the 3-layer pipeline produces for a canonical string."""
    resolved = MaximaSemanticResolver().resolve(SharedParser().parse(text),
                                                SymbolContext())
    return MaximaCodeGenerator().generate(resolved)


def _canonical(latex: str) -> str:
    return MaximaBackendHandler(None).latex_to_canonical(latex)


class TestConjugateMarkSpellings:
    """Every TeX spelling of the conjugate superscript is one canonical."""

    @pytest.mark.parametrize("latex", [
        r"\psi^*(x)", r"\psi^{*}(x)", r"\psi^\ast(x)",
        r"\psi^{\ast}(x)", r"\psi^{\star}(x)", r"\psi^\star(x)",
    ])
    def test_applied_spellings_agree(self, latex):
        assert _canonical(latex) == "psi^*(x)"

    @pytest.mark.parametrize("latex,expect", [
        (r"z^*", "z^*"), (r"z^{\ast}", "z^*"), (r"z^{*}", "z^*"),
        (r"\psi(x)^{\ast}", "psi(x)^*"),
    ])
    def test_bare_and_trailing_spellings_agree(self, latex, expect):
        assert _canonical(latex) == expect

    def test_an_exponent_is_not_a_conjugate(self):
        # A LONE star only.  `^{*2}` is an exponent and `^\dagger` is the
        # ADJOINT — a different IR node — so neither may be swept in.
        assert _canonical(r"x^{*2}") == "x^(*2)"
        assert "*" not in _canonical(r"a^\dagger")


class TestConjugateReachesTheCas:
    """…and the canonical lowers to `conjugate(...)`, not to garbage."""

    @pytest.mark.parametrize("canonical", [
        "z^*", "psi^*(x)", "psi(x)^*", "(psi(x))^*",
    ])
    def test_conjugate_survives_lowering(self, canonical):
        assert "conjugate(" in _cas(canonical)

    def test_product_with_its_own_conjugate(self):
        assert _cas("psi^*(x)*psi(x)") == "conjugate(psi(x)) * psi(x)"

    def test_nested_argument_is_not_cut_at_the_first_paren(self):
        assert _cas("f^*(g(x))") == "conjugate(f(g(x)))"

    @pytest.mark.parametrize("canonical", ["x^2(y)", "f^2(x)"])
    def test_a_numeric_script_is_left_alone(self, canonical):
        # `f^{2}(x)` is a genuine ambiguity (the square of the value, or the
        # second iterate).  Reading it is not this rule's business.
        assert "conjugate(" not in _cas(canonical)


class TestConjugateAgreesWithThePlainForm:
    """The conjugated and plain spellings of one quantity must lower to the
    SAME object, or the two halves of `\\varphi^*_n \\varphi_n` never pair up.
    """

    @pytest.mark.parametrize("plain,starred", [
        ("psi_n(x)", "psi^*_n(x)"),
        ("psi_n(x)", "psi_n^*(x)"),
        ("psi_n(x)", "psi_n(x)^*"),
        ("phi_{m,s}(x)", "phi^*_{m,s}(x)"),
        ("phi_{m,s}(x)", "phi_{m,s}(x)^*"),
    ])
    def test_same_object_under_the_star(self, plain, starred):
        assert _cas(starred) == f"conjugate({_cas(plain)})"

    def test_single_index_is_not_re_decided(self):
        # Regression: rebuilding the call inside the conjugate branch made the
        # starred form answer `psi[n](x)` where the plain form answers
        # `psi_n(x)` — two different CAS objects for one basis function.
        assert _cas("psi^*_n(x)") == "conjugate(psi_n(x))"
        assert _cas("phi^*_{m,s}(x)") == "conjugate(phi[m, s](x))"


class TestBarOverCompound:
    r"""`\overline{}` over a NAME is H-1's naming convention; over a compound
    it is conjugation."""

    @pytest.mark.parametrize("latex,expect", [
        (r"\overline{z}", "z_bar"),
        (r"\bar{\phi}", "phi_bar"),
        (r"\bar{H}", "H_bar"),
        (r"\bar{\phi_{n}}", "phi_n_bar"),
    ])
    def test_a_bar_over_a_name_is_still_a_name(self, latex, expect):
        assert _canonical(latex) == expect

    @pytest.mark.parametrize("latex,expect", [
        (r"\overline{\psi(x)}", "(psi(x))^*"),
        (r"\overline{a+b}", "(a+b)^*"),
        (r"\overline{z w}", "(z*w)^*"),
        (r"\overline{\psi_{n}(x)}", "(psi_n(x))^*"),
    ])
    def test_a_bar_over_a_compound_is_conjugation(self, latex, expect):
        assert _canonical(latex) == expect

    def test_both_in_one_expression(self):
        assert _canonical(r"\bar{x} + \overline{f(y)}") == "x_bar + (f(y))^*"


class TestConjugateDisplay:
    """What is displayed types back in, and the star sits on the function."""

    @pytest.fixture(scope="class")
    def engine(self):
        with NotebookEngine(title="conjugate display") as e:
            e.start()
            yield e

    @pytest.mark.parametrize("latex,expect", [
        (r"\psi(x)^*", r"\psi^{\ast}(x)"),
        (r"\psi(x)^{\ast}", r"\psi^{\ast}(x)"),
        (r"\psi^{\ast}(x)", r"\psi^{\ast}(x)"),
        (r"\varphi_{m,s}(x)^*", r"\varphi^{\ast}_{m,s}(x)"),
        (r"f(g(x))^*", r"f^{\ast}(g(x))"),
    ])
    def test_star_moves_onto_the_name(self, engine, latex, expect):
        expr = engine.register_expression(_canonical(latex), latex=latex)
        assert expr.latex == expect

    def test_a_conjugated_expression_does_not_move(self, engine):
        # `(A+B)^{\ast}` is a conjugated EXPRESSION, not a function call.
        expr = engine.register_expression(_canonical(r"(A+B)^{\ast}"),
                                          latex=r"(A+B)^{\ast}")
        assert expr.latex == r"(A+B)^{\ast}"

    def test_display_types_back_in(self, engine):
        # The round trip that was broken: our own output, re-entered.
        for source in (r"\psi(x)^*", r"\varphi_{m,s}(x)^*", r"\psi^*_{n}(x)"):
            expr = engine.register_expression(_canonical(source), latex=source)
            assert _cas(_canonical(expr.latex)) == _cas(_canonical(source))


class TestEnglishAndJapaneseSessionsAgree:
    """The conversion is notation, not language — pinned both ways because a
    reading rule that differs by session language is a bug by construction."""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_same_canonical_either_language(self, lang):
        with NotebookEngine(title="lang") as e:
            e.start()
            e.session.language = [lang]
            expr = e.register_expression(_canonical(r"\psi^{\ast}(x)"),
                                         latex=r"\psi^{\ast}(x)")
            assert expr.canonical == "psi^*(x)"
            assert expr.latex == r"\psi^{\ast}(x)"
