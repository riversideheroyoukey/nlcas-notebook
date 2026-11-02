r"""How far a written function reaches for its argument — one meaning, every spelling.

`\sin x` is a call and `\sin x \cos y` is a product: a bare function takes
exactly ONE token.  That rule was right; what was missing was every case where
"one token" is not the whole answer, and each miss was silent and produced
VALID CAS text — which is what makes them worse than an error.  Measured
2026-08-16, before the repair:

    \log \sqrt{x}       → log(sqrt) * x        the NAME taken as the argument
    \sin \cos x         → sin(cos) * x
    \log \sin(x)        → log(sin) * x
    \sin{\cos x}        → sin(cos * x)         a PRODUCT inside the sine
    \log \sin(\cos y)   → log(sin(cos * y))
    \log_2 x            → log(2) * x           the logarithm OF 2, times x
    \log_{10} x         → log(10) * x
    \log_2(x)           → log_2(x)             an undefined function
    \log_2 \frac{1}{6}  → log_2(1*6^-1)

`\log_2 6` therefore answered ≈ 4.16 where the value is ≈ 2.58.

One entry per MEANING with all the spellings that mean it, judged by asking
the CAS rather than comparing strings — the two lessons this repo keeps
relearning (docs/32 §Phase 5).  The last class is the one that must NOT move:
`\sin x y` is `sin(x)·y`, and rewriting it would be its own silent-wrong.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_adapter import MaximaAdapter
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler


@pytest.fixture(scope="module")
def handler():
    h = MaximaBackendHandler(None)
    h._context = SymbolContext()
    return h


@pytest.fixture(scope="module")
def cas():
    a = MaximaAdapter()
    a.start()
    yield a
    a.close()


def code(h, latex: str) -> str:
    return h.to_backend_code(h.latex_to_canonical(latex)).code


def _value(cas, command: str) -> "str | None":
    r = cas.execute(command)
    if not (r.ok and r.value and not r.value.startswith("ERROR")):
        return None
    return r.value.strip()


def _same(cas, a: str, b: str) -> bool:
    """Equal as MATHEMATICS.  `radcan` is needed because a change of base is a
    quotient of logarithms and `ratsimp` does not apply the logarithm laws —
    `log(8)/log(2)` stays put under it, which is Maxima not simplifying, not
    this system being wrong."""
    return _value(cas, f"radcan(({a}) - ({b}))") == "0"


#: (meaning, [spellings]).  Every spelling in a family is a way to write the
#: SAME expression, so all of them must reach the CAS as the same expression.
CORPUS: list[tuple[str, list[str]]] = [
    ("sine of a cosine", [
        r"\sin \cos x",
        r"\sin\cos x",          # no space — `_separate_adjacent_commands` puts one in
        r"\sin(\cos(x))",
        r"\sin (\cos x)",
        r"\sin{\cos x}",
    ]),
    ("logarithm of a square root", [
        r"\log \sqrt{x}",
        r"\log\sqrt{x}",
        r"\log(\sqrt{x})",
        r"\log{\sqrt{x}}",
    ]),
    ("three deep", [
        r"\sin \cos \tan x",
        r"\sin(\cos(\tan(x)))",
        r"\sin{\cos{\tan x}}",
    ]),
    ("a function inside a consumed group", [
        r"\log \sin(\cos y)",
        r"\log(\sin(\cos(y)))",
        r"\log{\sin{\cos y}}",
    ]),
    ("hyperbolic of a root", [
        r"\tanh \sqrt{x}",
        r"\tanh(\sqrt{x})",
    ]),
    ("base-2 logarithm of x", [
        r"\log_2 x",
        r"\log_{2} x",
        r"\log_2(x)",
        r"\log_{2}(x)",
        r"\log_2 {x}",
    ]),
    ("base-10 logarithm of x", [
        r"\log_{10} x",
        r"\log_{10}(x)",
    ]),
    ("symbolic base", [
        r"\log_b x",
        r"\log_{b} x",
        r"\log_b(x)",
    ]),
    ("base-2 logarithm of a compound argument", [
        r"\log_2 \frac{1}{6}",
        r"\log_2 (\frac{1}{6})",
        r"\log_{2}\left(\frac{1}{6}\right)",
    ]),
    ("base-2 logarithm of a function", [
        r"\log_2 \sin x",
        r"\log_2 (\sin x)",
        r"\log_2 {\sin x}",
    ]),
]

#: Spellings whose reading must NOT change: the ONE-TOKEN rule, the spaced
#: call, and the forms a subscript is legitimately part of a NAME in.
UNCHANGED: list[tuple[str, str]] = [
    (r"\sin x \cos y",   "sin(x) * cos(y)"),
    (r"\sin x + \cos y", "sin(x) + cos(y)"),
    (r"\sin x y",        "sin(x) * y"),
    (r"\sin 2x",         "sin(2 * x)"),
    (r"\sin 2 x",        "sin(2) * x"),
    (r"\sin x^2",        "sin(x^2)"),
    (r"2 \sin x",        "2 * sin(x)"),
    (r"r \sin \theta",   "r * sin(theta)"),
    (r"\log x",          "log(x)"),
    (r"\log(x)",         "log(x)"),
    (r"\log{x}",         "log(x)"),
    (r"\log (x+1)",      "log(x + 1)"),
    (r"\sqrt{\sin x}",   "sqrt(sin(x))"),
    (r"\sin x \, dx",    "sin(x) * dx"),
    (r"\det A",          "determinant(A)"),
    (r"\exp(-x^2)",      "exp(-x^2)"),
    ("bessel_j(0, x)",   "bessel_j(0, x)"),
    (r"p_x(x)",          "p_x(x)"),
    (r"C_i(t)",          "C_i(t)"),
]


class TestOneMeaningOneReading:
    @pytest.mark.parametrize("meaning,spellings", CORPUS,
                             ids=[m for m, _ in CORPUS])
    def test_every_spelling_reaches_the_cas_as_the_same_expression(
            self, handler, cas, meaning, spellings):
        first, rest = spellings[0], spellings[1:]
        base = code(handler, first)
        assert _value(cas, base) is not None, f"{meaning}: CAS refused {base}"
        for latex in rest:
            other = code(handler, latex)
            assert _same(cas, base, other), (
                f"{meaning}\n  {first!r} → {base}\n  {latex!r} → {other}")


class TestTheValueIsRight:
    r"""Agreeing with each other is not enough — they could agree on a wrong
    reading, which is exactly what `log(2)*x` and `log_2(x)` each did."""

    @pytest.mark.parametrize("latex,expected", [
        (r"\log_2 8",        "3"),
        (r"\log_2 32",       "5"),
        (r"\log_{10} 1000",  "3"),
        (r"\log_4 16",       "2"),
        (r"\log_2 x",        "log(x)/log(2)"),
        (r"\log_2 \frac{1}{6}", "-log(6)/log(2)"),
        (r"\sin \cos x",     "sin(cos(x))"),
        (r"\log \sqrt{x}",   "log(sqrt(x))"),
        (r"\sin{\cos x}",    "sin(cos(x))"),
    ])
    def test_value(self, handler, cas, latex, expected):
        assert _same(cas, code(handler, latex), expected), (
            f"{latex!r} → {code(handler, latex)}, expected {expected}")


class TestTheOneTokenRuleIsUnchanged:
    @pytest.mark.parametrize("latex,expected", UNCHANGED,
                             ids=[a for a, _ in UNCHANGED])
    def test_reading_did_not_move(self, handler, latex, expected):
        assert code(handler, latex) == expected


class TestTheExtremumBinderIsRefused:
    r"""`\max_i x_i` has no computable form — and three wrong ones.

    Measured 2026-08-16: the range-less form lowered to ``max(i) * x(i)``, and
    Maxima's ``max`` of one argument IS that argument, so it computed ``i·x(i)``
    — valid CAS text, a different quantity, silent.  ``\sup_i`` lowered to a
    function Maxima does not have; ``\max_{i=1}^{3}`` failed with a raw pydantic
    ValidationError as its reason.  One readable refusal replaces all three.
    """

    @pytest.mark.parametrize("latex", [
        r"\max_i x_i",
        r"\min_i x_i",
        r"\sup_i x_i",
        r"\max_x f(x)",
        r"\max_{i=1}^{3} x_i",
        r"\max_{i} x_i",
    ])
    def test_refused_with_a_reason(self, handler, latex):
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError) as exc:
            code(handler, latex)
        assert exc.value.i18n_key == "parser.unsupported_notation.extremum_binder"

    @pytest.mark.parametrize("latex,expected", [
        (r"\max(a, b)",   "max(a, b)"),
        (r"\max\{a, b\}", "max(a, b)"),
        (r"\min(a, b)",   "min(a, b)"),
    ])
    def test_the_ordinary_call_still_computes(self, handler, latex, expected):
        """What the refusal message points at has to work."""
        assert code(handler, latex) == expected


class TestNoFunctionNameIsEverAnArgument:
    r"""The shape every nested defect had: `log(sqrt)`, `sin(cos)`.

    A bare known-function NAME can never be an argument — it is not a value —
    so this is decidable without the CAS and cheap enough to run on every row.
    """

    _NAMES = ("sin", "cos", "tan", "sqrt", "exp", "log", "sinh", "cosh", "tanh")

    @pytest.mark.parametrize("latex",
                             [s for _m, sp in CORPUS for s in sp]
                             + [a for a, _ in UNCHANGED])
    def test_no_bare_name_as_argument(self, handler, latex):
        out = code(handler, latex)
        for name in self._NAMES:
            assert f"({name})" not in out and f"({name} " not in out, (
                f"{latex!r} → {out}: `{name}` is being used as a value")
