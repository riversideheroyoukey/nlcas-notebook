"""The supervisor's lexicon after it moved to ``i18n/nl_patterns.json`` (NL-GRAMMAR ③).

The move itself was behaviour-preserving and checked that way — HEAD's compiled
patterns against the new ones over the corpus's 760 real spellings, their 979
tokens and 68 probes, comparing span and every group.  What this file pins is
what that check could NOT see, because it is a property of the CALLING CODE
rather than of a pattern:

* the ORDER the sign keys are tried in — it stayed in Python deliberately
  ("順序は構造であって言語の内容ではない"), and nothing tested it.  Measured
  2026-08-27: no test in the repo mentioned 非負/非正/nonneg/nonpos at all, and
  a reorder turns 「非正の実数」 into *positive* silently — a sign that is
  forwarded to Maxima's ``assume()``.
* that the domain and the sign are read INDEPENDENTLY (「正の整数」 keeps both).
* that ``nlp_sub`` applies each DISTINCT pattern once.  A key with no English
  sibling falls back to the Japanese one, so a naive loop over the languages
  substitutes twice, and a ``$``-anchored strip is not idempotent.
* that "X で 微分" still names the differentiation VARIABLE rather than letting
  positional mapping take the expression's left-hand side.

Both languages throughout, per the bilingual-unit-test rule.
"""
import pytest

from notebook.runtime.llm_supervisor import (
    _extract_domain_kind,
    _extract_domain_sign,
    _tokenize,
)
from notebook.runtime.nl_patterns import nlp, nlp_sub, patterns_all


class TestTheSignIsReadBeforeTheWordItContains:
    """「非正」 contains 「正」 and 「非負の」 contains 「負の」; English is the same
    shape ("non-positive" contains "positive").  The negated readings must be
    tried first, and this is the only thing that says so."""

    @pytest.mark.parametrize("text,expected", [
        ("c を非正の実数とする", "nonpos"),
        ("Define c as a non-positive real number", "nonpos"),
        ("Define c as a nonpositive real number", "nonpos"),
        ("b を非負の実数とする", "nonneg"),
        ("Define b as a non-negative real number", "nonneg"),
        ("Define b as a nonnegative real number", "nonneg"),
        ("T を正の実数とする", "positive"),
        ("Define T as a positive real number", "positive"),
        ("a を負の実数とする", "negative"),
        ("Define a as a negative real number", "negative"),
    ])
    def test_sign(self, text, expected):
        assert _extract_domain_sign(text) == expected


class TestTheDomainAndTheSignAreReadIndependently:
    """「正の整数」 must keep BOTH — the kind pattern winning and dropping the
    sign is the defect ``_extract_domain_sign`` was split out for."""

    @pytest.mark.parametrize("text,kind,sign", [
        ("n を正の整数とする", "integer", "positive"),
        ("Define n as a positive integer", "integer", "positive"),
        ("N を正の実数パラメータとして定義して", "real", "positive"),
        ("Define N as a positive real parameter", "real", "positive"),
        ("z を複素数とする", "complex", None),
        ("Define z as a complex number", "complex", None),
        ("m を自然数とする", "natural", None),
        ("Define m as a natural number", "natural", None),
        ("x をスカラーとする", None, None),
        ("Define x as a scalar", None, None),
    ])
    def test_kind_and_sign(self, text, kind, sign):
        assert (_extract_domain_kind(text), _extract_domain_sign(text)) == (kind, sign)


class TestASubstitutionRunsOncePerDistinctPattern:
    """``nlp(key, "en")`` falls back to the ``ja`` entry, so a loop over the
    languages would apply the same strip twice — and these strips are anchored
    at ``$``, which makes them non-idempotent."""

    def test_the_verb_tail_is_stripped_once(self):
        assert nlp("verb_tail_strip", "ja").pattern == nlp("verb_tail_strip", "en").pattern
        assert len(patterns_all("verb_tail_strip")) == 1
        # Applied twice this is "計算" — one verb phrase too many.
        assert nlp_sub("verb_tail_strip", "計算して求めて") == "計算して"

    def test_a_key_with_two_languages_keeps_both(self):
        assert len(patterns_all("domain_kind_real")) == 2


class TestTheOperationNounSurvivesTheVerb:
    """The aliases are noun-form, so the て-conjugated input has to be stripped
    back to the noun — in Japanese, where there is no space to do it."""

    @pytest.mark.parametrize("text,noun", [
        ("E1 の実部を求めて", "実部"),
        ("E1 の虚部を取り出して", "虚部"),
        ("E1 の固有値を算出する", "固有値"),
    ])
    def test_japanese_verb_tail(self, text, noun):
        assert noun in _tokenize(text)

    @pytest.mark.parametrize("text,noun", [
        ("Find the realpart of E1", "realpart"),
        ("Extract the imaginary part of E1", "imaginary"),
    ])
    def test_english_needs_no_strip(self, text, noun):
        """English separates the verb with a space, which is why the key has no
        English sibling — not an oversight."""
        assert noun in _tokenize(text)


class TestTheDifferentiationVariableIsTheOneAfterDe:
    """Positional mapping would take the first ASCII token, which in
    「L を q で微分して」 is the expression's left-hand side."""

    def test_japanese_particle_names_the_variable(self):
        m = nlp("diff_var_particle", "ja").search("L を q で微分して")
        assert m and m.group(1) == "q"

    def test_partial_derivative_too(self):
        m = nlp("diff_var_particle", "ja").search("H を p_x で偏微分して")
        assert m and m.group(1) == "p_x"

    def test_english_is_ordered_and_needs_no_pattern(self):
        """"differentiate f with respect to x" is handled by field mapping; the
        key is Japanese-only on purpose."""
        assert nlp("diff_var_particle", "en").pattern == nlp("diff_var_particle", "ja").pattern
