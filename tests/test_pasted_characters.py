r"""What a word processor does to a formula, and what the parser must do back.

Measured 2026-08-15, starting from open_items BARE-SYMBOL-PREDICATE-SPLIT — a
ticket about two regexes disagreeing.  The measurement said the ticket was a
symptom: `shared_parser._tokenize` **silently discarded every character it could
not match**, and `to_backend_code` reported `lowered` (success) on the result.
Four shapes reached the CAS as something other than what was written:

    y = x – 1     (U+2013 EN DASH)   → y = x * 1     SUBTRACTION → MULTIPLICATION
    z = f′(x)     (U+2032 PRIME)     → z = f(x)      the derivative mark vanished
    café = 2 x                       → caf = 2*x     name truncated, display kept `café`
    x温  = 2 x                        → x   = 2*x     name truncated onto the VARIABLE x
                                                     (published display: `3温 = 6`)

Two answers, because the two halves are different questions:

* a character with ONE mathematical meaning is normalised at the entrance
  (`ui/interpret_pipeline._fullwidth_to_halfwidth`), joining `_conjugate_mark`
  and `_subscript_separator` — the pattern CLAUDE.md describes as "made one
  spelling at the converter's entrance, so downstream has one to keep working";
* anything still unreadable is REFUSED with the character in the message, not
  dropped.

**Why the refusal is a complete check and not a longer list.** The guard it
replaces enumerated the characters it knew (`|`, `#`), so every character nobody
had thought of — including all four above — went on being dropped.  It is now
inverted: the tokenizer refuses whatever it did not consume, and only `\` (TeX
command prefix) and `†` (adjoint mark) are exempt.  Those two were MEASURED over
the full suite as the entire set the corpus relies on being skipped.

**Unicode identifiers were considered and rejected** (Maxima itself accepts
`café: 5` and `温度: 300`).  Three side effects, two of them blocking: the NL
field mapper excludes CJK tokens from operands by design, so a kanji-named
symbol could be registered and displayed but never named in an instruction
("displayed but not usable"); and the published `.tex` preamble deliberately
omits `inputenc`/`fontenc`, so an accented letter would not typeset.  Refusing
with a reason is the honest answer; open_items carries the reasoning.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.shared_parser import UnknownNotationError, _tokenize
from ui.interpret_pipeline import (
    InputToken, InterpretRequest, _fullwidth_to_halfwidth, run_interpret_pipeline)


@pytest.fixture
def session():
    nb.start_engine(title="pasted", mode="standard", language="ja", domains=[])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=[], primary_backend="maxima")
    try:
        yield eng, sup
    finally:
        nb.stop_engine()


def _math(eng, sup, latex):
    return asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
        InterpretRequest(tokens=[InputToken(type="math", value=latex,
                                            latex=latex, canonical="")]), eng, sup))


def _cas(eng, result):
    eid = result.get("expr_id") or (
        result.get("registered_expressions") or [{}])[-1].get("expr_id")
    assert eid, result
    return eng._handler.to_backend_code(eng.expressions[eid].canonical)


class TestNormalisedAtTheEntrance:
    """One mathematical meaning ⇒ one spelling, decided before anything reads it."""

    @pytest.mark.parametrize("pasted,expected", [
        ("y = x – 1", "y=x - 1"),        # U+2013 EN DASH
        ("y = x — 1", "y=x - 1"),        # U+2014 EM DASH
        ("y = x − 1", "y=x - 1"),        # U+2212 MINUS SIGN (already handled)
        ("y = x ＋ 1", "y=x + 1"),       # full-width plus (already handled)
    ])
    def test_a_pasted_dash_is_a_minus(self, session, pasted, expected):
        eng, sup = session
        low = _cas(eng, _math(eng, sup, pasted))
        assert low.status == "lowered", low
        assert low.code.replace("  ", " ") == expected, low.code

    @pytest.mark.parametrize("pasted,expected", [
        ("z = f′(x)", "'diff(f(x),x,1)"),     # U+2032 PRIME
        ("w = g″(t)", "'diff(g(t),t,2)"),     # U+2033 DOUBLE PRIME
    ])
    def test_a_typographic_prime_is_a_derivative(self, session, pasted, expected):
        eng, sup = session
        low = _cas(eng, _math(eng, sup, pasted))
        assert expected in low.code, low.code

    def test_prose_keeps_its_punctuation(self):
        """The relaxed rules are for MATH tokens only.

        An em dash between words is punctuation, and a prime after a number is a
        minute mark.  Both are left alone in a text token — which is why the
        normaliser takes a flag instead of being made unconditional.
        """
        for s in ("これは — 重要な注意", "5′ の角度"):
            assert _fullwidth_to_halfwidth(s) == s, s

    def test_a_spaced_dash_is_what_a_paste_looks_like(self):
        """The rule that existed required the operands to TOUCH the dash.

        `x–1` was converted and `x – 1` was not — and a formula pasted from a
        PDF is spaced.  That single gap is what let a subtraction become a
        product.
        """
        assert _fullwidth_to_halfwidth("x–1") == "x-1"
        assert _fullwidth_to_halfwidth("x – 1", math=True) == "x - 1"


class TestRefusedRatherThanDropped:
    def test_an_unreadable_character_raises(self):
        for s in ("café", "x温", "a é b"):
            with pytest.raises(UnknownNotationError):
                _tokenize(s)

    def test_the_message_names_the_character(self):
        with pytest.raises(UnknownNotationError) as exc:
            _tokenize("café")
        assert exc.value.fmt_kwargs.get("token") == "é", exc.value.fmt_kwargs

    def test_the_lowering_door_stops_instead_of_truncating(self, session):
        """`to_backend_code` is the one door to the CAS and it claimed SUCCESS.

        `x温` truncated to `x`, which collided with a variable of that name:
        the command became `subst(x=3, x=2*x)` and the notebook published
        `3温 = 6` — a substitution applied INSIDE a symbol's name.

        The door RE-RAISES rather than returning `refused`, deliberately: an
        `UnknownNotationError` carries an i18n key, and the engine boundary is
        where the session language is known.  Swallowing it into a `refused`
        reason string would hand the researcher an untranslated one.
        """
        eng, _sup = session
        for text in ("café = 2*x", "x温 = 2*x"):
            with pytest.raises(UnknownNotationError):
                eng._handler.to_backend_code(text)

    def test_the_researcher_sees_a_translated_reason(self, session):
        """End of the contract: the message reaches the researcher, translated.

        `register_expression` translates the i18n key with the session language
        and RAISES — the same path `|A|` and `#` have always taken — and
        `ui/routes/advisory.py` turns it into a failed response.  Asserted at
        the raise, because that is where the text is decided; the route's own
        wrapping is a separate concern.
        """
        eng, sup = session
        with pytest.raises(Exception) as exc:
            _math(eng, sup, "café = 2 x")
        msg = str(exc.value)
        assert "é" in msg, msg
        assert "数式として読めません" in msg, msg

    def test_ordinary_notation_is_untouched(self):
        for s in (r"\sum_{i} x_i", "c†(k)", "a'(1) a(1)", "x + 1", "f(x, t)"):
            _tokenize(s)          # must not raise

    def test_the_skippable_set_is_only_what_was_measured(self):
        """Two characters, both with a reason, both measured over the corpus.

        Pinned so that widening the exemption is a visible decision rather than
        a quiet way to make a failing input pass.
        """
        from notebook.runtime.shared_parser import _SKIPPABLE_CHARS
        assert _SKIPPABLE_CHARS == frozenset("\\†")
