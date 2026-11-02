r"""A written function is APPLIED, or it is not a quantity at all.

open_items FN-SUBSCRIPT-NO-READING named one spelling (`\sin_2 x`).  Measuring
the family on 2026-08-17 found the neighbour it did not name is far worse and
far more common:

    \sin^2 x                     → sin^2 * x            (`sin` an ordinary SYMBOL)
    \cos^2\theta + \sin^2\theta  → theta*cos^2 + theta*sin^2   — never 1
    d/dx of \sin^2 x             → sin^2                — sin held constant

`(\sin x)^2` and `\sin(x)^2` were always right; only the spelling everyone
actually writes was wrong, and Maxima accepts `sin` as a symbol without a
murmur, so every one of these came back a success.

The split: `\sin^n x` has a conventional reading and is now READ; the two that
the notation genuinely does not decide are refused, each in its own words.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.i18n import t
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError


def _cas(latex: str, ctx=None) -> str:
    canonical = MaximaBackendHandler(None).latex_to_canonical(latex)
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                          ctx or SymbolContext())
    return MaximaCodeGenerator().generate(ir)


class TestThePowerIsRead:
    @pytest.mark.parametrize("latex,expect", [
        (r"\sin^2 x", "sin(x)^2"),
        (r"\sin^{2} x", "sin(x)^2"),
        (r"\sin^2(x)", "sin(x)^2"),
        (r"\sin^2 \theta", "sin(theta)^2"),
        (r"\tan^2 x", "tan(x)^2"),
        (r"\sinh^2 x", "sinh(x)^2"),
        (r"\sin^n x", "sin(x)^n"),
        (r"\sin^{n+1} x", "sin(x)^(n + 1)"),
    ])
    def test_exponent_applies_to_the_value(self, latex, expect):
        assert _cas(latex) == expect

    def test_the_pythagorean_identity_survives(self):
        r"""About as ordinary as physics notation gets, and it used to reach the
        CAS as `theta*cos^2 + theta*sin^2`, which no amount of `trigsimp` folds."""
        assert _cas(r"\cos^2\theta + \sin^2\theta") == (
            "cos(theta)^2 + sin(theta)^2")

    def test_and_the_cas_now_folds_it(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()
        got = _cas(r"\cos^2\theta + \sin^2\theta")
        res = adapter.execute(f"trigsimp({got})")
        assert str(getattr(res, "value", res)).strip() == "1"

    @pytest.mark.parametrize("latex,expect", [
        (r"\sin x", "sin(x)"),
        (r"\sin(x)", "sin(x)"),
        (r"\sin(x)^2", "sin(x)^2"),
        (r"\sin x \cos y", "sin(x) * cos(y)"),
        (r"\log_2 x", "log(x) * log(2)^-1"),
        (r"\sqrt{x}", "sqrt(x)"),
        (r"x^2", "x^2"),
    ])
    def test_what_already_worked_is_untouched(self, latex, expect):
        assert _cas(latex) == expect


class TestTheInverseIsSettledByTheRECIPROCALSName:
    r"""`\sin^{-1}` is arcsine, and the reason is that `1/sin` ALREADY HAS A
    NAME.

    Refused as ambiguous until 2026-08-18, when the researcher pointed out the
    discriminator: nobody writes `\sin^{-1}` for the reciprocal because that
    quantity is called `cosec` — likewise sec (1/cos) and cot (1/tan).  The
    system's own `_KNOWN_FUNCTIONS` already carried `csc`, `sec`, `cot` beside
    the inverses, which is the same argument in data form.

    The names are Maxima's, measured live: `asin(1)` is %pi/2 while `arcsin(1)`
    is an UNDEFINED function that stays put.
    """

    @pytest.mark.parametrize("latex,expect", [
        (r"\sin^{-1} x", "asin(x)"),
        (r"\cos^{-1} x", "acos(x)"),
        (r"\tan^{-1} x", "atan(x)"),
        (r"\sec^{-1} x", "asec(x)"),
        (r"\csc^{-1} x", "acsc(x)"),
        (r"\cot^{-1} x", "acot(x)"),
        (r"\sinh^{-1} x", "asinh(x)"),
        (r"\cosh^{-1} x", "acosh(x)"),
        (r"\tanh^{-1} x", "atanh(x)"),
        (r"\sin^{-1}(x)", "asin(x)"),
    ])
    def test_reads_as_the_inverse_function(self, latex, expect):
        assert _cas(latex) == expect

    def test_and_the_cas_evaluates_it(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()
        for latex, want in ((r"\sin^{-1} 1", "%pi/2"), (r"\tan^{-1} 1", "%pi/4")):
            res = adapter.execute(_cas(latex))
            assert str(getattr(res, "value", res)).strip() == want


class TestTheInverseTrigNamesReachTheCAS:
    r"""`\arcsin` used to lower to `arcsin`, which MAXIMA DOES NOT KNOW.

    Found while implementing the above, and independent of it: measured
    2026-08-18, `arcsin(1)` stayed `arcsin(1)` instead of %pi/2, `diff` returned
    a noun form, `integrate` did nothing, `sin(arcsin(x))` would not simplify,
    and `tex1` gave `{\it arcsin}(x)` — italic prose in a published paper.
    Every inverse-trig calculation quietly did nothing.
    """

    @pytest.mark.parametrize("latex,expect", [
        (r"\arcsin(x)", "asin(x)"),
        (r"\arccos(x)", "acos(x)"),
        (r"\arctan(x)", "atan(x)"),
        (r"\arcsin x", "asin(x)"),
    ])
    def test_lowers_to_the_name_the_cas_knows(self, latex, expect):
        assert _cas(latex) == expect

    def test_the_cas_evaluates_and_differentiates_it(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()
        assert str(getattr(adapter.execute(_cas(r"\arcsin(1)")), "value", "")).strip() \
            == "%pi/2"
        assert "sqrt" in str(getattr(
            adapter.execute(f"diff({_cas(r'\arcsin(x)')}, x)"), "value", ""))

    def test_the_display_round_trip_closes(self):
        """`tex1(asin(x))` is `\\arcsin x` — the researcher's own notation."""
        from notebook.runtime.maxima_adapter import MaximaAdapter
        adapter = MaximaAdapter()
        adapter.start()
        shown = str(getattr(adapter.execute("tex1(asin(x))"), "value", "")).strip()
        assert shown == r"\arcsin x"
        assert _cas(shown) == "asin(x)"


class TestWhatTheNotationDoesNotDecide:
    @pytest.mark.parametrize("latex,key", [
        # No named reciprocal, so the discriminator above does not apply and the
        # two readings still compete.  Decided 2026-08-18 to keep refusing:
        # reading `\log^{-1}` as 1/log silently misanswers anyone who meant exp.
        (r"\log^{-1} x", "function_inverse_script"),
        (r"\exp^{-1} x", "function_inverse_script"),
        # A negative exponent other than -1 is written for neither reading.
        (r"\sin^{-2} x", "function_negative_power"),
        # A subscript on anything but the logarithm has no reading at all.
        (r"\sin_2 x", "function_subscript_no_reading"),
        (r"\exp_2 x", "function_subscript_no_reading"),
        (r"\cos_n x", "function_subscript_no_reading"),
    ])
    def test_refused_by_name_not_silently_mangled(self, latex, key):
        with pytest.raises(UnknownNotationError) as exc:
            _cas(latex)
        assert exc.value.args[0] == f"parser.unsupported_notation.{key}"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("key", ["function_inverse_script",
                                     "function_negative_power",
                                     "function_subscript_no_reading"])
    def test_the_reason_is_translated_and_names_a_way_forward(self, lang, key):
        msg = t(f"parser.unsupported_notation.{key}", lang, name="sin")
        assert msg != f"parser.unsupported_notation.{key}"
        assert "sin" in msg

    def test_the_researchers_own_f_2_is_not_caught(self):
        """The ticket asked for this to be measured BEFORE refusing: `f` is not
        one of the backend's function names, so a name-with-subscript of the
        researcher's own can never match."""
        assert _cas(r"f_2(x)") == "f_2(x)"
        assert _cas(r"x_2") == "x(2)"

    def test_an_explicit_reciprocal_is_fine(self):
        """`(\\sin x)^{-1}` says which reading is meant, so it is not refused."""
        assert _cas(r"(\sin x)^{-1}") == "sin(x)^-1"

    def test_the_reason_explains_why_trig_is_different(self):
        from notebook.runtime.i18n import t
        for lang in ("ja", "en"):
            msg = t("parser.unsupported_notation.function_inverse_script", lang,
                    name="log")
            assert "cosec" in msg or "cot" in msg
