"""Unit tests for latex_to_canonical() converter."""
import pytest
from notebook.runtime.latex_converter import restore_decorations
from notebook.runtime.maxima_latex_converter import latex_to_canonical


class TestDelimiters:
    def test_dollar_stripped(self):
        assert latex_to_canonical("$x^2$") == "x^2"

    def test_double_dollar_stripped(self):
        assert latex_to_canonical("$$x+1$$") == "x+1"

    def test_bracket_delimiters_stripped(self):
        assert latex_to_canonical(r"\[x+1\]") == "x+1"

    def test_paren_delimiters_stripped(self):
        assert latex_to_canonical(r"\(x+1\)") == "x+1"


class TestFrac:
    def test_simple_frac(self):
        assert latex_to_canonical(r"\frac{x}{y}") == "((x)/(y))"

    def test_frac_with_expressions(self):
        result = latex_to_canonical(r"\frac{x^2-1}{x-1}")
        assert "(x^2-1)" in result
        assert "(x-1)" in result
        assert "/" in result

    def test_nested_frac(self):
        result = latex_to_canonical(r"\frac{\frac{a}{b}}{c}")
        assert "(a)/(b)" in result
        assert "/c" in result or "/(c)" in result

    def test_frac_pi(self):
        result = latex_to_canonical(r"\frac{\pi}{2}")
        assert "%pi" in result
        assert "2" in result


class TestSqrt:
    def test_sqrt_simple(self):
        assert latex_to_canonical(r"\sqrt{x}") == "sqrt(x)"

    def test_sqrt_expression(self):
        result = latex_to_canonical(r"\sqrt{x^2+1}")
        assert "sqrt(" in result

    def test_nth_root(self):
        result = latex_to_canonical(r"\sqrt[3]{x}")
        assert "1/3" in result or "(1/(3))" in result

    def test_sqrt_bare(self):
        result = latex_to_canonical(r"\sqrt x")
        assert "sqrt(x)" in result


class TestSuperscript:
    def test_single_char_exp(self):
        assert latex_to_canonical("x^{2}") == "x^2"

    def test_multi_char_exp(self):
        result = latex_to_canonical("x^{n+1}")
        assert "x^(n+1)" == result

    def test_negative_exp(self):
        result = latex_to_canonical("x^{-1}")
        assert "x^(-1)" == result

    def test_no_braces_unchanged(self):
        assert latex_to_canonical("x^2") == "x^2"


class TestTrigFunctions:
    def test_sin(self):
        assert latex_to_canonical(r"\sin(x)") == "sin(x)"

    def test_cos(self):
        assert latex_to_canonical(r"\cos(x)") == "cos(x)"

    def test_tan(self):
        assert latex_to_canonical(r"\tan(x)") == "tan(x)"

    def test_log(self):
        assert latex_to_canonical(r"\log(x)") == "log(x)"

    def test_ln_to_log(self):
        assert latex_to_canonical(r"\ln(x)") == "log(x)"

    def test_exp(self):
        assert latex_to_canonical(r"\exp(x)") == "exp(x)"

    def test_inverse_trig_uses_the_name_the_cas_knows(self):
        r"""Was: `\arctan(x)` -> `arctan(x)`.

        Maxima does NOT know `arctan`.  Measured 2026-08-18 it treated the name
        as an undefined function: `arcsin(1)` stayed `arcsin(1)` instead of
        %pi/2, `diff` returned a noun form, `integrate` did nothing, and `tex1`
        gave `{\it arcsin}(x)` — italic prose in a published paper.  Every
        inverse-trig calculation quietly did nothing.  `atan` computes, and its
        `tex1` is `\arctan x`, so the researcher still sees what they wrote.
        """
        assert latex_to_canonical(r"\arctan(x)") == "atan(x)"
        assert latex_to_canonical(r"\arcsin(x)") == "asin(x)"
        assert latex_to_canonical(r"\arccos(x)") == "acos(x)"


class TestGreekAndConstants:
    def test_pi(self):
        assert latex_to_canonical(r"\pi") == "%pi"

    def test_pi_in_expression(self):
        result = latex_to_canonical(r"2\pi")
        assert "%pi" in result

    def test_infty(self):
        assert latex_to_canonical(r"\infty") == "inf"

    def test_alpha(self):
        assert latex_to_canonical(r"\alpha") == "alpha"

    def test_omega(self):
        assert latex_to_canonical(r"\omega") == "omega"

    def test_lambda(self):
        assert latex_to_canonical(r"\lambda") == "lambda"

    def test_varepsilon(self):
        assert latex_to_canonical(r"\varepsilon") == "epsilon"

    def test_hbar(self):
        assert latex_to_canonical(r"\hbar") == "hbar"


class TestOperators:
    def test_cdot(self):
        assert latex_to_canonical(r"a\cdot b") == "a . b"

    def test_times(self):
        assert latex_to_canonical(r"a\times b") == "a ~ b"

    def test_div(self):
        # 2026-09-10: `\div` の置換が `replace_tex_command`（名前の終わりで
        # だけ置き換える）経由になり、前後の空白を食わなくなった。`a/b` と
        # `a / b` は**同じ IR に解決する**（実測）。実在の文字列 16710 件で
        # `\div` の出現は 0 件——綴りが変わった保存物は無い。
        assert latex_to_canonical(r"a\div b") == "a / b"


class TestLeftRight:
    def test_left_paren(self):
        result = latex_to_canonical(r"\left(x+1\right)")
        assert result == "(x+1)"

    def test_left_bracket(self):
        result = latex_to_canonical(r"\left[a,b\right]")
        assert result == "[a,b]"


class TestBraceRemoval:
    def test_simple_braces_removed(self):
        assert latex_to_canonical("{x}") == "x"

    def test_braces_around_group(self):
        result = latex_to_canonical("{a+b}")
        assert result == "a+b"


class TestMultiPass:
    def test_nested_frac_resolves(self):
        # \frac{\frac{1}{x}}{y} → ((1)/(x))/(y)
        result = latex_to_canonical(r"\frac{\frac{1}{x}}{y}")
        assert "(1)/(x)" in result
        assert "y" in result

    def test_sqrt_with_frac(self):
        result = latex_to_canonical(r"\sqrt{\frac{x}{y}}")
        assert "sqrt(" in result
        assert "(x)/(y)" in result


class TestPhysicsExpressions:
    def test_schrodinger_kinetic(self):
        result = latex_to_canonical(r"\frac{\hbar^2}{2m}")
        assert "hbar" in result
        assert "2" in result
        assert "m" in result

    def test_gaussian(self):
        result = latex_to_canonical(r"e^{-x^2}")
        # e is just "e" (user's variable); exponent should be expanded
        assert "(-x^2)" in result or "(-x" in result

    def test_taylor_term(self):
        result = latex_to_canonical(r"\frac{x^{n}}{n!}")
        assert "x^n" in result

    def test_trig_in_frac(self):
        result = latex_to_canonical(r"\frac{\sin(x)}{x}")
        assert "sin(x)" in result
        assert "x" in result


class TestFunctionApplication:
    """Known built-in functions receive their argument even without parens."""

    # --- sin ---
    def test_sin_space_var(self):
        assert latex_to_canonical(r"\sin x") == "sin(x)"

    def test_sin_braces(self):
        # \sin{x} — LaTeX argument braces converted to parens
        assert latex_to_canonical(r"\sin{x}") == "sin(x)"

    def test_sin_parens_unchanged(self):
        assert latex_to_canonical(r"\sin(x)") == "sin(x)"

    def test_sin_compound_arg(self):
        # multi-token argument: resolved over two passes
        assert latex_to_canonical(r"\sin 2x") == "sin(2*x)"

    def test_sin_superscript_arg(self):
        assert latex_to_canonical(r"\sin x^2") == "sin(x^2)"

    def test_sin_greek_arg(self):
        assert latex_to_canonical(r"\sin \theta") == "sin(theta)"

    def test_sin_braces_compound(self):
        assert latex_to_canonical(r"\sin{2x}") == "sin(2*x)"

    # --- other functions ---
    def test_cos_space_var(self):
        assert latex_to_canonical(r"\cos x") == "cos(x)"

    def test_log_space_var(self):
        assert latex_to_canonical(r"\log x") == "log(x)"

    def test_log_compound_arg(self):
        assert latex_to_canonical(r"\log 2x") == "log(2*x)"

    def test_exp_space_var(self):
        assert latex_to_canonical(r"\exp x") == "exp(x)"

    # --- in expressions ---
    def test_multiple_functions_in_sum(self):
        result = latex_to_canonical(r"\sin x + \cos x")
        assert result == "sin(x) + cos(x)"

    def test_function_definition(self):
        result = latex_to_canonical(r"f(x) = \sin x + \cos x")
        assert result == "f(x) = sin(x) + cos(x)"

    def test_compound_args_in_sum(self):
        result = latex_to_canonical(r"\sin x + \cos 2x")
        assert result == "sin(x) + cos(2*x)"

    # --- user-defined functions unaffected ---
    def test_user_fn_with_parens_unchanged(self):
        assert latex_to_canonical(r"g(x) = x^2") == "g(x) = x^2"

    def test_user_fn_multivar_unchanged(self):
        assert latex_to_canonical(r"h(x,y) = x + y") == "h(x,y) = x + y"


class TestImplicitMultiplication:
    """Explicit * is inserted for all implicit-multiplication patterns."""

    # --- no-space digit × letter / constant ---
    def test_digit_letter_nospace(self):
        assert latex_to_canonical("2x") == "2*x"

    def test_digit_percent_nospace(self):
        assert latex_to_canonical(r"2\pi") == "2*%pi"

    # --- space-separated digit × identifier ---
    def test_digit_space_letter(self):
        assert latex_to_canonical("2 x") == "2*x"

    def test_digit_space_pi_r(self):
        assert latex_to_canonical(r"2 \pi r") == "2*%pi*r"

    # --- digit × paren group ---
    def test_digit_paren_nospace(self):
        assert latex_to_canonical("2(x+1)") == "2*(x+1)"

    def test_digit_space_paren(self):
        assert latex_to_canonical("2 (x+1)") == "2*(x+1)"

    def test_digit_paren_squared(self):
        assert latex_to_canonical("2 (x+1)^2") == "2*(x+1)^2"

    # --- adjacent paren groups ---
    def test_paren_times_paren(self):
        assert latex_to_canonical("(x+1)(x-1)") == "(x+1)*(x-1)"

    def test_frac_times_paren(self):
        result = latex_to_canonical(r"\frac{1}{2}(x+1)")
        assert result == "((1)/(2))*(x+1)"

    # --- closing paren × variable ---
    def test_frac_times_vars(self):
        result = latex_to_canonical(r"\frac{1}{2} m v^2")
        assert result == "((1)/(2))*m*v^2"

    # --- letter × letter (adjacent variable names) ---
    def test_single_letter_product(self):
        assert latex_to_canonical("m v") == "m*v"

    def test_greek_product(self):
        assert latex_to_canonical(r"\alpha \beta") == "alpha*beta"

    def test_pi_times_r_squared(self):
        assert latex_to_canonical(r"\pi r^2") == "%pi*r^2"

    # --- unary minus is NOT converted ---
    def test_unary_minus_unchanged(self):
        assert latex_to_canonical("-(x-1)") == "-(x-1)"


class TestPassthrough:
    def test_plain_expression_unchanged(self):
        assert latex_to_canonical("x^2+1") == "x^2+1"

    def test_already_canonical(self):
        assert latex_to_canonical("(x^2-1)/(x-1)") == "(x^2-1)/(x-1)"

    def test_empty_string(self):
        assert latex_to_canonical("") == ""


class TestVectorCalculus:
    def test_nabla_dot_divergence(self):
        assert latex_to_canonical(r"\nabla \cdot F") == "nabla . F"

    def test_nabla_cross_curl(self):
        assert latex_to_canonical(r"\nabla \times F") == "nabla ~ F"

    def test_nabla_gradient(self):
        assert latex_to_canonical(r"\nabla \phi") == "nabla*phi"

    def test_nabla_laplacian(self):
        assert latex_to_canonical(r"\nabla^2 \phi") == "nabla^2*phi"
        assert latex_to_canonical(r"\Delta \phi") == "Delta*phi"


class TestDecorations:
    """B-6: LaTeX decoration commands."""

    # Hat: \hat creates a distinct symbol — \hat{H} ≠ H
    def test_hat_braced(self):
        assert latex_to_canonical(r"\hat{H}") == "H_hat"

    def test_hat_bare(self):
        assert latex_to_canonical(r"\hat H") == "H_hat"

    def test_widehat_single(self):
        assert latex_to_canonical(r"\widehat{H}") == "H_hat"

    def test_widehat_compound(self):
        assert latex_to_canonical(r"\widehat{H_0}") == "H_0_hat"

    def test_hat_expression(self):
        assert latex_to_canonical(r"\hat{H} + \hat{V}") == "H_hat + V_hat"

    # Tilde
    def test_tilde_plain_symbol(self):
        assert latex_to_canonical(r"\tilde{x}") == "x_tilde"

    def test_tilde_greek(self):
        assert latex_to_canonical(r"\tilde{\omega}") == "omega_tilde"

    def test_widetilde(self):
        assert latex_to_canonical(r"\widetilde{H_0}") == "H_0_tilde"

    def test_tilde_uppercase_greek(self):
        assert latex_to_canonical(r"\tilde{\Delta}") == "Delta_tilde"

    # Bar / overline
    def test_bar_plain_symbol(self):
        assert latex_to_canonical(r"\bar{x}") == "x_bar"

    def test_bar_uppercase(self):
        assert latex_to_canonical(r"\bar{H}") == "H_bar"

    def test_bar_greek(self):
        assert latex_to_canonical(r"\bar{\xi}") == "xi_bar"

    def test_overline_single(self):
        assert latex_to_canonical(r"\overline{\psi}") == "psi_bar"

    def test_overline_compound(self):
        # \overline{AB} → AB_bar (AB is a distinct decorated quantity)
        assert latex_to_canonical(r"\overline{AB}") == "AB_bar"

    # Dot: \dot{x} ≠ x  (time derivative notation)
    def test_dot_creates_new_symbol(self):
        assert latex_to_canonical(r"\dot{x}") == "x_dot"

    def test_ddot_creates_new_symbol(self):
        assert latex_to_canonical(r"\ddot{q}") == "q_ddot"

    # Underline: text emphasis only, stripped
    def test_underline_stripped(self):
        assert latex_to_canonical(r"\underline{F}") == "F"

    def test_vec_unchanged(self):
        # \vec is handled separately (keeps vec() wrapper)
        assert latex_to_canonical(r"\vec{F}") == "vec(F)"


class TestRestoreDecorations:
    """B-2 Display Bridge: canonical suffix → LaTeX decoration command."""

    # Basic round-trips (inverse of TestDecorations above)
    def test_hat_plain(self):
        assert restore_decorations("H_hat") == r"\hat{H}"

    def test_hat_greek(self):
        assert restore_decorations("omega_hat") == r"\hat{\omega}"

    def test_tilde_plain(self):
        assert restore_decorations("x_tilde") == r"\tilde{x}"

    def test_tilde_greek(self):
        assert restore_decorations("omega_tilde") == r"\tilde{\omega}"

    def test_bar_plain(self):
        assert restore_decorations("x_bar") == r"\bar{x}"

    def test_bar_greek(self):
        assert restore_decorations("xi_bar") == r"\bar{\xi}"

    def test_dot_creates_dot(self):
        assert restore_decorations("x_dot") == r"\dot{x}"

    def test_ddot_creates_ddot(self):
        assert restore_decorations("q_ddot") == r"\ddot{q}"

    # Subscripted base
    def test_hat_with_subscript_base(self):
        assert restore_decorations("H_0_hat") == r"\hat{H_0}"

    def test_tilde_with_subscript_base(self):
        assert restore_decorations("H_0_tilde") == r"\tilde{H_0}"

    def test_greek_with_subscript_base(self):
        assert restore_decorations("xi_k_bar") == r"\bar{\xi_k}"

    # In expressions
    def test_sum_of_decorated(self):
        assert restore_decorations("H_hat + V_hat") == r"\hat{H} + \hat{V}"

    def test_mixed_expression(self):
        result = restore_decorations("J * omega_tilde + H_bar")
        assert r"\tilde{\omega}" in result
        assert r"\bar{H}" in result

    # Non-decorated identifiers are unchanged
    def test_plain_symbol_unchanged(self):
        assert restore_decorations("H") == "H"

    def test_subscript_symbol_unchanged(self):
        assert restore_decorations("H_0") == "H_0"

    def test_greek_unchanged(self):
        # Greek names WITHOUT decoration are left to the CAS/display pipeline
        assert restore_decorations("omega") == "omega"

    # Uppercase Greek base
    def test_uppercase_greek_base(self):
        assert restore_decorations("Delta_tilde") == r"\tilde{\Delta}"

    # ddot before dot (no double-match)
    def test_ddot_not_matched_as_dot(self):
        # q_ddot should produce \ddot{q}, not \dot{q_d}
        assert restore_decorations("q_ddot") == r"\ddot{q}"


class TestNestedBracesAndCommandBoundary:
    r"""Two silent-wrongs in the LaTeX→canonical rules (EQ-EMBEDDED-NOTATION,
    measured 2026-08-01 while checking the S-c handoff's registered forms).

    Both produced a canonical the CAS ACCEPTS, so nothing failed anywhere:

    * ``\frac{1}{\sqrt{x^2+1}}`` → ``\frac1*sqrt(x^2+1)`` — a PRODUCT of an
      undefined symbol ``frac1``.  The non-nested ``\{([^{}]*)\}`` could not
      read a denominator carrying its own braces, and ``_remove_remaining_braces``
      then stripped the braces in the SAME pass, so no later pass could
      recover it.
    * ``\int_0^1 x^2 dx`` → ``%int_0^1*x^2*dx`` — likewise a product.  The
      culprit was ``\i`` → ``%i`` (Maxima's imaginary unit) matching the
      PREFIX of ``\int``.  Sorting the table by length only protects commands
      that are in the table, and ``\int`` is not.

    ``\int`` and ``\sum`` are deliberately left as-is for the parser, which
    turns them into Integral / IndexSum nodes — that is what "canonical is the
    researcher's own writing" means (CLAUDE.md).
    """

    def test_frac_with_braced_command_in_denominator(self):
        assert latex_to_canonical(r"\frac{1}{\sqrt{x^2+1}}") == "((1)/(sqrt(x^2+1)))"

    def test_frac_with_braced_command_in_numerator(self):
        assert latex_to_canonical(r"\frac{\sqrt{x}}{2}") == "((sqrt(x))/(2))"

    def test_frac_nested_in_frac(self):
        assert latex_to_canonical(r"\frac{1}{\frac{a}{b}}") == "((1)/(((a)/(b))))"

    def test_frac_greek_denominator_still_works(self):
        assert latex_to_canonical(r"\frac{1}{\sqrt{\xi^2 + \Delta^2}}") == \
            "((1)/(sqrt(xi^2 + Delta^2)))"

    def test_frac_without_two_braced_operands_is_left_alone(self):
        """The rule converts what it can read and never guesses."""
        assert r"\frac" in latex_to_canonical(r"\frac12")

    def test_int_is_not_eaten_by_the_imaginary_unit_rule(self):
        got = latex_to_canonical(r"\int_0^1 x^2 dx")
        assert "%int" not in got
        assert got.startswith(r"\int")

    def test_imaginary_unit_still_converts(self):
        assert latex_to_canonical(r"\mathrm{i}") == "%i"
        assert latex_to_canonical(r"\i") == "%i"

    def test_infty_still_converts(self):
        assert latex_to_canonical(r"\infty") == "inf"

    def test_sinh_not_shadowed_by_sin(self):
        """The same prefix hazard, on the function table."""
        assert latex_to_canonical(r"\sinh{x}") == "sinh(x)"
        assert latex_to_canonical(r"\sin{x}") == "sin(x)"


class TestIntegralReachesTheCasCorrectly:
    r"""End-to-end: the canonical must LOWER to a real integral, and the value
    must match.  A string check would have passed on the broken form too —
    ``%int_0^1*x^2*dx`` is a perfectly good product."""

    def _code(self, latex: str) -> str:
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        canon = latex_to_canonical(latex)
        ir = MaximaSemanticResolver().resolve(SharedParser().parse(canon),
                                              SymbolContext())
        return MaximaCodeGenerator().generate(ir)

    def test_definite_integral_lowers_to_integrate(self):
        assert self._code(r"\int_0^1 x^2 dx") == "integrate(x^2, x, 0, 1)"

    def test_integral_inside_an_equation(self):
        assert self._code(r"A = \int_0^1 x^2 dx") == "A=integrate(x^2, x, 0, 1)"

    def test_sum_inside_an_equation(self):
        assert self._code(r"S = \sum_{i} x_i") == "S=sum(x(i), i)"

    def test_value_agrees_with_the_cas(self):
        """Difference against the known answer, not string equality."""
        from notebook.runtime.maxima_adapter import MaximaAdapter
        ad = MaximaAdapter(timeout=30.0)
        ad.start()
        try:
            code = self._code(r"\int_0^1 x^2 dx")
            assert ad.execute(f"ratsimp(({code}) - 1/3)").value.strip() == "0"
            frac = self._code(r"\frac{1}{\sqrt{x^2+1}}")
            assert ad.execute(
                f"ratsimp(({frac}) - 1/sqrt(x^2+1))").value.strip() == "0"
        finally:
            ad.close()


class TestCanonicalIsNeverARegexTemplate:
    r"""A canonical keeps the researcher's TeX (``\sum_i*x_i``), so passing it
    as a re.sub REPLACEMENT made ``re`` read ``\s`` as an escape and raise
    PatternError — registering "S = \sum_i x_i" and then asking for its
    right-hand side crashed the whole request (measured 2026-08-01)."""

    def _expand(self, text: str, equation: str) -> str:
        """Register *equation* in a real session, then expand *text*."""
        import asyncio
        import ui.notebook_bridge as nb
        from ui.interpret_common import _expand_symbol_refs
        from ui.interpret_pipeline import (InterpretRequest, InputToken,
                                           run_interpret_pipeline)
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        nb.start_engine(title="tmpl", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend=eng.session.primary_backend)
        try:
            tok = InputToken(type="math", value=equation, latex=equation,
                             canonical="")
            asyncio.run(run_interpret_pipeline(
                InterpretRequest(tokens=[tok]), eng, sup))
            return _expand_symbol_refs(text, eng)
        finally:
            nb.stop_engine()

    def test_tex_bearing_canonical_does_not_crash(self):
        got = self._expand("S + 1", r"S = \sum_{i} x_i")
        assert "\\sum" in got and got.endswith("+ 1")

    def test_backslash_is_substituted_literally(self):
        got = self._expand("T", r"T = \int_0^1 x^2 dx")
        assert "\\int" in got


class TestFractionIsOneOperand:
    r"""A fraction must be emitted as ONE unit, ``((num)/(den))``.

    ``(num)/(den)`` is not one: anything binding tighter than ``/`` on either
    side takes a piece of it, and every case below was silently wrong rather
    than an error.  Found while rewriting the information-theory scenario onto
    general operations (open_items SCENARIO-MACRO): the entropy of a fair die,
    written ``-\sum_{x=1}^{6}\frac16\log\frac16``, evaluated to **0** and came
    back a success.  The `sum`-template macros there had been hiding it — they
    build the ``log`` call themselves, so the researcher's own spelling never
    reached this converter.
    """

    @pytest.mark.parametrize("latex,wrong,why", [
        (r"\log \frac{1}{6}",       "log (1)/(6)",     "log(1)/6 = 0, not log(1/6)"),
        (r"\sin \frac{\pi}{2}",     "sin (%pi)/(2)",   "sin(pi)/2 = 0, not 1"),
        (r"\log \frac{p(x)}{q(x)}", "log (p(x))/(q(x))", "the KL-divergence integrand"),
        (r"\frac{a}{b}^2",          "(a)/(b)^2",       "a/b^2, not (a/b)^2"),
        (r"x^\frac{1}{2}",          "x^(1)/(2)",       "x/2, not sqrt(x)"),
    ])
    def test_function_and_script_operands_keep_the_whole_fraction(self, latex, wrong, why):
        got = latex_to_canonical(latex)
        assert got != wrong, why
        # The denominator must stay inside the fraction's own parentheses.
        assert "))" in got or got.count("(") >= 3, got

    def test_the_measured_values(self):
        """Meaning, not spelling: ask the CAS (the string round-trip is what
        let these through for months)."""
        import ui.notebook_bridge as nb
        nb.start_engine(title="frac_unit", mode="standard", language="en",
                        domains=["information_theory"])
        eng = nb.get_engine()
        try:
            for latex, expect in ((r"\sin \frac{\pi}{2}", "1"),
                                  (r"\log \frac{1}{6}", "-log(6)"),
                                  (r"x^\frac{1}{2}", "sqrt(x)")):
                cano = eng._handler.latex_to_canonical(latex)
                r = eng._adapter.execute(f"ratsimp({cano} - ({expect}))")
                assert r.ok and r.value.strip() == "0", (latex, cano, r.value)
        finally:
            nb.stop_engine()


class TestADecoratedGreekNameThatIsAlreadyACommand:
    r"""``\theta_dot`` → ``\dot{\theta}``, never ``\\dot{\theta}``.

    Maxima's ``tex1`` writes the canonical ``theta_dot`` as ``\theta_dot``: it
    converts the Greek prefix and leaves the decoration suffix behind.
    ``restore_decorations`` scanned bare identifier tokens, so it matched
    ``theta_dot`` INSIDE the command and left the original backslash standing in
    front of its own output — ``\\dot{\theta}``, and ``\\`` is a LINE BREAK in
    LaTeX, not a decoration.

    Measured 2026-08-23 on an analytical-mechanics substitution label, where
    ``\dot\theta`` is the most ordinary quantity there is; ``\tilde\omega`` and
    ``\hat\Sigma`` are the same shape.  The function's own docstring already
    claimed LaTeX commands pass through unchanged — true except for a command
    whose NAME carries a decoration suffix, which is every decorated Greek.
    """

    @pytest.mark.parametrize("src,want", [
        (r"\theta_dot",   r"\dot{\theta}"),
        (r"\omega_tilde", r"\tilde{\omega}"),
        (r"\Sigma_hat",   r"\hat{\Sigma}"),
        (r"-\frac{2 \dot{r} \theta_dot}{r}", r"-\frac{2 \dot{r} \dot{\theta}}{r}"),
    ])
    def test_the_command_form_restores_without_doubling(self, src, want):
        assert restore_decorations(src) == want

    @pytest.mark.parametrize("src", [
        r"\frac{a}{b}", r"\left(x\right)", r"\sin x", r"\alpha", r"\theta",
        r"\dot{r}",
    ])
    def test_an_ordinary_command_is_untouched(self, src):
        """The backslash is consumed only when a restoration actually happens —
        otherwise `\frac` would lose its own."""
        assert restore_decorations(src) == src

    @pytest.mark.parametrize("src,want", [
        (r"theta_dot", r"\dot{\theta}"),
        (r"H_hat",     r"\hat{H}"),
        (r"H_0_hat",   r"\hat{H_0}"),
        (r"psi_hat_dagger_sigma", r"\hat{\psi}^{\dagger}_{\sigma}"),
    ])
    def test_the_bare_form_is_unchanged_by_the_fix(self, src, want):
        assert restore_decorations(src) == want

    def test_no_output_ever_contains_a_double_backslash(self):
        """The invariant, stated once: `\\` in a restoration is always wrong."""
        for src in (r"\theta_dot", r"\omega_tilde", r"\Sigma_hat", r"theta_dot",
                    r"-\frac{2 \dot{r} \theta_dot}{r}", r"\phi_ddot"):
            assert "\\\\" not in restore_decorations(src), src
