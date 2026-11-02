r"""The alias scan reads PHRASES, and the longest one wins.

open_items EN-ALIAS-MULTIWORD.  `LLMSupervisor._try_structured_parse` used to
hand `resolve_alias` one token at a time.  English is written with spaces, so an
alias containing one could never match — measured on the table as it stood
(2026-08-15): **126 of the aliases are multi-word and non-CJK**, leaving 21
operations with no reachable English spelling whatsoever.

Unreachable would have been the harmless half.  **48 of those phrases contain a
shorter alias that belongs to a DIFFERENT operation**, and that one matched
instead, so the researcher got the wrong operation reported as a success:

    Take the inverse Laplace transform of E1 from s to t
        → operation=laplace_transform        (the FORWARD transform)
        → registered `1/((s^2+4)*the)`       ("the" as a variable)

Three separate things had to be true for the sentence above to work, and each is
pinned here, because each can regress on its own:

1. the scan matches n-grams (this file's first class);
2. a LONGER match beats a shorter one contained in it (second class);
3. the phrase exists in the table at all — `inverse laplace transform` did not
   (third class).  No scan can match what is not there, and this is the half the
   ticket did not name.

Japanese is unaffected by 1 and 2 (its aliases are single tokens), which is
exactly why a JP-only scenario could never surface any of it.
"""
import re

import pytest

from notebook.runtime.llm_supervisor import _tokenize, scan_aliases
from notebook.runtime.operation_validator import OperationValidator


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


def _scan(v, text):
    return scan_aliases(_tokenize(text), v.resolve_alias, v.max_alias_words)


class TestPhrasesMatch:
    @pytest.mark.parametrize("text,expected", [
        ("find the pure Nash equilibria of E1 and E2", "pure_nash_equilibria"),
        ("multiply both sides of E1 by 2", "multiply_both_sides"),
        ("apply the mean field decoupling to E1", "pair_substitute"),
        ("take the inverse Laplace transform of E1 from s to t", "inverse_laplace"),
    ])
    def test_a_multiword_alias_is_found(self, v, text, expected):
        assert [c for _s, _e, c in _scan(v, text)] == [expected], _scan(v, text)


class TestTheLongestMatchWins:
    def test_inverse_beats_the_forward_transform(self, v):
        """`laplace` alone is an alias of the FORWARD transform and sits inside
        the phrase.  This is the measured silent-wrong, at the layer it came
        from."""
        assert [c for _s, _e, c in
                _scan(v, "take the inverse Laplace transform of E1 from s to t")
                ] == ["inverse_laplace"]

    def test_both_sides_beats_bare_multiply(self, v):
        """`multiply` alone means `multiply_expressions` — a different operation
        that would multiply two expressions rather than an equation's sides."""
        assert [c for _s, _e, c in _scan(v, "multiply both sides by 2")
                ] == ["multiply_both_sides"]

    def test_the_span_is_consumed_not_just_the_first_token(self, v):
        """A shorter alias inside the matched span must not ALSO match.

        `substitute` is an alias of its own operation and sits inside
        `operator substitute`; if only the first token were marked used, both
        would be offered as candidates and the wrong one could win."""
        spans = _scan(v, "operator substitute a(1) in E1")
        assert [c for _s, _e, c in spans] == ["operator_substitute"], spans
        start, end, _c = spans[0]
        assert end - start == 2, spans


class TestJapaneseIsUnchanged:
    @pytest.mark.parametrize("text,expected", [
        ("E1 を s から t へ逆ラプラス変換して", "inverse_laplace"),
        ("E1 の両辺に 2 をかけて", "multiply_both_sides"),
        ("E1 と E2 の純戦略ナッシュ均衡を求めて", "pure_nash_equilibria"),
    ])
    def test_single_token_japanese_aliases_still_match(self, v, text, expected):
        assert expected in [c for _s, _e, c in _scan(v, text)], _scan(v, text)


class TestTheTableHasTheEnglishPhrase:
    """The scan cannot match what nobody wrote down.

    `inverse_laplace`'s English aliases were `inverse_laplace` and `ilt` — no
    spelling a researcher would type.  Fixing only the scan left the headline
    sentence resolving to the forward transform, because the phrase did not
    exist.  Pinned as a property of the whole table, not of one operation, so a
    new operation added with only a snake_case English alias trips it.
    """

    def test_every_multiword_operation_can_be_spelled_with_spaces(self, v):
        missing = [
            op for op in v.known_operations
            if "_" in op and v.resolve_alias(op.replace("_", " ")) != op
        ]
        assert missing == [], missing

    def test_the_transform_pair_has_its_natural_english(self, v):
        assert v.resolve_alias("inverse laplace transform") == "inverse_laplace"
        assert v.resolve_alias("laplace transform") == "laplace_transform"


class TestGrammarWordsAreNotOperands:
    """The positional field mapper filled `s`/`t` from `Take` and `the`.

    Single letters are deliberately NOT stopwords: `a` is an article and also
    QuantumAlgebra's boson constructor, `i` an index.  A one-letter token stays
    an operand.
    """

    def test_english_function_words_are_dropped(self):
        from notebook.runtime.nl_patterns import nlp
        stop = nlp("operand_stopwords", "en")
        for word in ("the", "of", "from", "to", "take", "with", "respect"):
            assert stop.match(word), word
        for word in ("a", "i", "s", "t", "x", "E1", "omega"):
            assert not stop.match(word), word

    def test_japanese_drops_nothing(self):
        from notebook.runtime.nl_patterns import nlp
        stop = nlp("operand_stopwords", "ja")
        for word in ("x", "の", "式", "to", "the"):
            assert not stop.match(word), word


class TestTheScanItself:
    def test_matches_are_returned_in_text_order(self, v):
        spans = _scan(v, "multiply both sides of E1 by 2")
        assert spans == sorted(spans)

    def test_overlapping_matches_are_not_returned(self, v):
        spans = _scan(v, "take the inverse Laplace transform of E1 from s to t")
        seen: set[int] = set()
        for start, end, _c in spans:
            assert not (seen & set(range(start, end)))
            seen |= set(range(start, end))

    def test_the_ceiling_comes_from_the_table(self, v):
        """A five-word alias must work the day it is added.  A hardcoded ceiling
        is the same failure this whole file is about, one size up."""
        longest = max(v.known_aliases, key=lambda a: len(a.split()))
        assert v.max_alias_words >= len(longest.split())
        assert [c for _s, _e, c in _scan(v, longest)] == [
            v.resolve_alias(longest)]
