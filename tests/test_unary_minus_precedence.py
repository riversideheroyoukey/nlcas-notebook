r"""`-x^2` is `-(x^2)` — the sign binds LOOSER than the exponent.

Read the other way round, which is what the grammar did until 2026-08-08,
`e^{-x^2}` reached the CAS as `%e^(-x)^2`: **the Gaussian with its sign
flipped**, silently.

    -x^2                                 simplify  →  x^2
    e^{-x^2}                             simplify  →  %e^x^2
    \int_{-\infty}^{\infty} e^{-x^2} dx            →  a divergent integral

Gaussians, Boltzmann factors `e^{-E/(k_B T)}`, `e^{-r^2/(2\sigma^2)}` — every
`-<base>^<exp>` was affected.

**Why it stayed hidden for so long**, which is the part worth keeping: the
operations that pass the CANONICAL TEXT straight to Maxima never build this AST,
and Maxima's own parser reads `-x^2` correctly.  The transform family works that
way, and that is exactly where the Gaussian scenarios live — so
`scenario_fourier_convolution` published CORRECT answers
(`\mathcal F\{e^{-t^2}\} = \sqrt\pi e^{-\omega^2/4}`) while `simplify` on the
same expression was wrong.  A committed example being right is not evidence that
the parser is.

The fix moves the unary sign below `^`, which changes what a subscript and an
exponent may swallow.  Both directions bit immediately and both are pinned here:
`x_1^2` must not become `x_(1^2)`, and `a^b_c` must not become `a^(b_c)`.
"""
import pytest

import ui.notebook_bridge as nb
from notebook.models.ir import Number, Subscript, Superscript
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser


def _cas(text):
    ctx = SymbolContext()
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(text), ctx)
    return MaximaCodeGenerator(context=ctx).generate(ir)


class TestTheSignBindsLooserThanTheExponent:
    @pytest.mark.parametrize("text,expected", [
        ("-x^2", "-x^2"),
        ("%e^(-x^2)", "%e^(-x^2)"),
        ("exp(-x^2)", "exp(-x^2)"),
        ("-x^3", "-x^3"),
    ])
    def test_the_sign_stays_outside(self, text, expected):
        assert _cas(text) == expected

    def test_explicit_parentheses_still_mean_what_they_say(self):
        """`(-x)^2` IS the square of a negative — only the unparenthesised
        spelling was being misread."""
        assert _cas("(-x)^2") == "(-x)^2"

    def test_a_negative_number_base(self):
        """`-2^2` is -4, the ordinary reading of the notation and Maxima's own."""
        assert _cas("-2^2") == "-2^2"

    def test_a_bare_negative_number_still_folds(self):
        assert SharedParser().parse("-2") == Number(value=-2)

    def test_a_sign_in_the_exponent(self):
        assert _cas("2^-x") == "2^(-x)"


class TestWhatTheMoveMustNotDisturb:
    """The two operands whose density had to be kept: subscript and exponent."""

    def test_a_subscript_does_not_swallow_the_exponent(self):
        r"""`x_1^2` is `(x_1)^2`.  Swallowed, it became `x_(1^2)` — rejected
        outright by IndexedSymbol, so at least it was loud."""
        assert _cas("x_1^2") == "x(1)^2"

    def test_an_exponent_does_not_swallow_the_subscript(self):
        """`a^b_c` is `(a^b)_c` — pinned separately in
        tests/test_power_division_codegen.py, and broken by the first attempt."""
        assert isinstance(SharedParser().parse("a^b_c"), Subscript)
        assert isinstance(SharedParser().parse("a^b_c").base, Superscript)

    def test_a_negative_subscript_still_parses(self):
        r"""`\int_{-\infty}`, `c'_{-k}`, the einstein indices.  Dropping the
        sign from the subscript operand broke all of them at once."""
        assert _cas("f_-1") == "f(-1)"

    @pytest.mark.parametrize("text,expected", [
        ("2^3^2", "2^3^2"),          # `^` stays RIGHT-associative
        ("k/r^2", "k * (r^2)^-1"),   # the pinned k/r^2 → k*sqrt(r) repair
    ])
    def test_the_other_pinned_readings(self, text, expected):
        assert _cas(text) == expected


class TestItReachesTheCAS:
    """The value, not the string: this is what a researcher would have seen."""

    def setup_method(self):
        nb.start_engine(title="unary", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()

    def teardown_method(self):
        nb.stop_engine()

    def _value(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        ex = self.eng.register_expression(canonical, latex=latex)
        step, _ = self.eng.run_operation("simplify", {"expr": ex.display_id})
        return self.eng.expressions[step.output_refs[0]].canonical.replace(" ", "")

    def test_the_gaussian_keeps_its_sign(self):
        """`e^{-x^2}` simplified to `%e^x^2` — the wrong function entirely."""
        assert self._value(r"e^{-x^2}") == "%e^-x^2"

    def test_the_gaussian_integral(self):
        r"""`\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt\pi`.  It used to send a
        DIVERGENT integral and come back as an error."""
        assert self._value(r"\int_{-\infty}^{\infty} e^{-x^2} dx") == "sqrt(%pi)"

    def test_a_boltzmann_factor(self):
        assert self._value(r"e^{-x^2/2}") == "%e^-(x^2/2)"
