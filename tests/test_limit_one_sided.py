r"""LIMIT-ONE-SIDED / GF-NOTATION 残1族 (2026-09-11).

Three layers were broken and each one alone was enough to give a wrong answer:

* **the notation was unread** — `\lim` reached the CAS as a subscripted function
  or as a product, and the equal-time Green's function as one undefined symbol::

      \lim_{x \to 0} \frac{\sin x}{x}  →  lim[x, to, 0](sin(x) * x^-1)
      \lim_{n \to \infty} a_n          →  lim(n, to, inf) * a(n)      ← a PRODUCT
      \lim_{t' \to t^+} G(t,t')        →  raw LaTeX

* **the side never reached the CAS** — the template had no place for
  `direction`, so a one-sided limit was answered with the two-sided one while
  the DISPLAY said `\lim_{x \to 0^{+}}`.  Measured against a live Maxima,
  `limit(abs(x)/x, x, 0)` is `ind` and the two sides are `1` and `-1`.

* **the side could not be written** — `direction=plus` was dropped (pre-pass 0
  filled REQUIRED fields only, though BP-6's rule is about naming a field at
  all), `x → 0^+` took the point down with it, and `右極限` was not an alias.

The CAS could do all of it from the start: `limit(f, x, a, plus|minus)`.
"""
import pytest

from notebook.models.ir import CasOperation
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.backend_factory import make_backend
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError


def _maxima_code(latex: str) -> str:
    ctx = SymbolContext()
    h = MaximaBackendHandler(None)
    h.set_symbol_context(ctx)
    ir = MaximaSemanticResolver().resolve(
        SharedParser().parse(h.latex_to_canonical(latex)), ctx)
    return MaximaCodeGenerator(context=ctx).generate(ir)


def _shown(latex: str) -> str:
    h = MaximaBackendHandler(None)
    h.set_symbol_context(SymbolContext())
    return h.normalize_display_latex(h.latex_to_canonical(latex))


class TestTheNotationReachesTheCAS:
    @pytest.mark.parametrize("latex,command", [
        (r"\lim_{x \to 0} \frac{\sin x}{x}", "limit(sin(x) * x^-1, x, 0)"),
        (r"\lim_{n \to \infty} a_n",         "limit(a(n), n, inf)"),
        (r"\lim_{x \to a} (f(x) + g(x))",    "limit(f(x) + g(x), x, a)"),
        # the equal-time Green's function — GF-NOTATION's last family
        (r"\lim_{t' \to t^+} G(t,t')",       "limit(G(t, t\\'), t\\', t, plus)"),
        (r"\lim_{x \to 0^-} f(x)",           "limit(f(x), x, 0, minus)"),
    ])
    def test_command(self, latex, command):
        assert _maxima_code(latex) == command

    def test_the_binder_takes_the_term_not_the_sum(self):
        """The convention this project already applies to Σ:
        `\\sum_i x_i + y` is `sum(x(i), i) + y`."""
        assert _maxima_code(r"\lim_{x \to 0} f(x) + g(x)") == \
            "limit(f(x), x, 0) + g(x)"

    def test_a_factor_written_before_the_binder_is_not_under_it(self):
        assert _maxima_code(r"2 \lim_{x \to 0} f(x)") == "2 * limit(f(x), x, 0)"

    @pytest.mark.parametrize("latex", [r"\liminf_{n} a_n",
                                       r"\limsup_{n \to \infty} a_n"])
    def test_liminf_is_a_different_binder_and_is_left_alone(self, latex):
        """No explicit exclusion does this — the "is the next character `_`?"
        test already answers it, which a surviving mutation is what showed."""
        assert "limit(" not in _maxima_code(latex)

    @pytest.mark.parametrize("latex,command", [
        (r"a \int f(x) dx",   "a * integrate(f(x), x)"),
        (r"2 \int_0^1 x dx",  "2 * integrate(x, x, 0, 1)"),
    ])
    def test_the_same_rule_holds_for_the_integral(self, latex, command):
        """`_split_at_prefix`'s lead handling is shared with ∫, and nothing
        pinned it there — found by a mutation that hit the integral's copy of
        this branch by mistake and left the suite green (2026-09-11)."""
        assert _maxima_code(latex) == command


class TestTheSideIsPartOfTheQuestion:
    """`limit(abs(x)/x, x, 0)` is `ind`; the sides are 1 and -1."""

    @pytest.mark.parametrize("direction,expected", [
        (None,    "limit(abs(x)/x,x,0)"),
        ("plus",  "limit(abs(x)/x,x,0,plus)"),
        ("minus", "limit(abs(x)/x,x,0,minus)"),
    ])
    def test_direction_picks_the_template(self, direction, expected):
        _, builder, _ = make_backend("maxima")
        args = {"expr": "abs(x)/x", "var": "x", "point": "0"}
        if direction:
            args["direction"] = direction
        result = builder.build(CasOperation(operation="limit", args=args))
        assert result.ok and result.command == expected


class TestWhatIsShownIsWhatWasWritten:
    @pytest.mark.parametrize("latex,expected", [
        (r"\lim_{x \to 0} f(x)",     r"\lim_{x \to 0} f(x)"),
        (r"\lim_{x \to 0^+} f(x)",   r"\lim_{x \to 0^{+}} f(x)"),
        (r"\lim_{x \to 0^-} f(x)",   r"\lim_{x \to 0^{-}} f(x)"),
        (r"\lim_{n \to \infty} a_n", r"\lim_{n \to \infty} a_n"),
        (r"\lim_{t' \to t^+} G(t,t')", r"\lim_{t' \to t^{+}} G(t,t')"),
    ])
    def test_the_internal_spelling_never_shows(self, latex, expected):
        assert _shown(latex) == expected

    def test_both_display_entrances_agree(self):
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        assert beautify_bare_canonical_latex(r"\lim_{x,to,0} f(x)") == \
            r"\lim_{x \to 0} f(x)"

    @pytest.mark.parametrize("tex1,expected", [
        (r"\lim_{{\it t'}\downarrow t}{G\left(t\right)}",
         r"\lim_{t' \to t^{+}}{G\left(t\right)}"),
        (r"\lim_{{\it t'}\uparrow t}{G\left(t\right)}",
         r"\lim_{t' \to t^{-}}{G\left(t\right)}"),
    ])
    def test_maximas_own_arrow_spelling_is_given_back(self, tex1, expected):
        r"""Maxima's `tex1` writes a one-sided limit as `x\downarrow a`."""
        h = MaximaBackendHandler(None)
        ctx = SymbolContext()
        h.set_symbol_context(ctx)
        assert h.postprocess_result_latex("x", tex1, ctx) == expected

    def test_the_two_sided_arrow_is_left_alone(self):
        h = MaximaBackendHandler(None)
        ctx = SymbolContext()
        h.set_symbol_context(ctx)
        out = h.postprocess_result_latex(
            "x", r"\lim_{x\rightarrow 0}{f\left(x\right)}", ctx)
        assert r"\rightarrow" in out


class TestJuliaRefusesWithAReason:
    r"""Write it down first, compute later: it registers and displays, and is
    refused at the point of USE."""

    def test_refusal(self):
        ctx = SymbolContext()
        h = JuliaBackendHandler(None)
        h.set_symbol_context(ctx)
        ir = JuliaSemanticResolver().resolve(
            SharedParser().parse(h.latex_to_canonical(r"\lim_{x \to 0} f(x)")), ctx)
        with pytest.raises(UnknownNotationError) as exc:
            JuliaCodeGenerator(context=ctx).generate(ir)
        assert "julia_limit" in str(exc.value)

    def test_the_message_exists_in_both_languages(self):
        import json
        msgs = json.load(open("i18n/messages.json", encoding="utf-8"))
        entry = msgs["parser.unsupported_notation.julia_limit"]
        assert entry["ja"] and entry["en"]


class TestTheInstructionNamesTheSide:
    """Bilingual: every spelling that routes, measured in both languages."""

    @pytest.mark.parametrize("text,lang,direction", [
        ("E1 の x → 0 の極限を求めて",              "ja", None),
        ("E1 の x → 0+ の極限を求めて",             "ja", "plus"),
        ("E1 の x → 0 の右極限を求めて",            "ja", "plus"),
        ("E1 の x → 0 の右側極限を求めて",          "ja", "plus"),
        ("E1 の x → 0 の左極限を求めて",            "ja", "minus"),
        ("E1 の x → 0 の左側極限を求めて",          "ja", "minus"),
        ("E1 の極限を求めて var=x, point=0, direction=minus", "ja", "minus"),
        ("take the limit of E1 as x -> 0",          "en", None),
        ("the limit of E1 as x -> 0^+",             "en", "plus"),
        ("the right-hand limit of E1 as x -> 0",    "en", "plus"),
        ("the left-hand limit of E1 as x -> 0",     "en", "minus"),
        ("the limit of E1 as x -> 0 from the right", "en", "plus"),
        ("the limit of E1 as x -> 0 from the left",  "en", "minus"),
        ("the limit of E1 as x -> 0 from above",     "en", "plus"),
        ("the limit of E1 as x -> 0 from below",     "en", "minus"),
        ("limit of E1 with var=x, point=0, direction=plus", "en", "plus"),
    ])
    def test_reading(self, text, lang, direction):
        result = LLMSupervisor(primary_backend="maxima").interpret(text, lang=lang)
        assert result.operation == "limit"
        assert result.args.get("expr") == "E1"
        assert result.args.get("var") == "x"
        assert result.args.get("point") == "0"
        assert result.args.get("direction") == direction


class TestANamedOptionalFieldIsFilled:
    """BP-6's rule — "a researcher who names a field has removed the ambiguity"
    — applied to the half that did not have it.  The CHOOSING half already read
    required ∪ optional; the FILLING half read only required."""

    def test_an_optional_field_written_by_name_lands(self):
        from notebook.runtime.token_field_mapping import map_tokens_to_fields
        args = map_tokens_to_fields(
            ["E1", "direction=plus"], ["expr", "var", "point"], {"E1"},
            "en", optional_fields=["direction"])
        assert args.get("direction") == "plus"

    def test_a_field_no_operation_holds_is_not_read(self):
        from notebook.runtime.token_field_mapping import map_tokens_to_fields
        args = map_tokens_to_fields(
            ["E1", "nonesuch=7"], ["expr"], {"E1"}, "en", optional_fields=[])
        assert "nonesuch" not in args
