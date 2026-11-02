"""The last three NL vocabularies to leave the source, and what holds them.

NL-GRAMMAR residual ③ finished here (2026-08-27): ``spin_def_op_text``,
``order_suffix_token``, ``expr_ref_prefix`` and ``fermion_kind_kw`` moved to
``i18n/nl_patterns.json``.  The fourth was not on the list — it had been
labelled infrastructure ("a guard that reads both languages"), and re-measured
it is plainly VOCABULARY: ``フェルミ``/``fermion`` is the word that decides
which STATISTICS a bare operator declaration gets.  The move was behaviour-preserving — 1,750 inputs
compared old against new, and the 523-row instruction corpus replayed with no
drift — but the mutation pass found something the corpus cannot see: **breaking
any of the three failed no test in the repository.**

That is what this file is for.  Each class below breaks in exactly one of the
three ways the mutation produced, and each aims at the vocabulary's DIRECT
consumer, because aiming higher is what made two of the three look dead:

* ``spin_def_op_text`` first measured as "nothing moves" — the probes were
  COMPLETE declarations (``S をスピン演算子として定義して``), which the base
  ``decl_stmt`` guard already answers.  What only this key answers is a spin
  type word WITHOUT the declaration verb.
* ``order_suffix_token`` only runs when the tokenizer did not split ``4次`` into
  ``4`` + ``次…``; the standalone-digit branch above it takes the split case, so
  a scenario spelling never reaches the fallback.
* ``expr_ref_prefix`` fired on no corpus row in either language, because the
  prefix is not typed — ``ui/interpret_pipeline`` prepends it.  **That key is gone
  as of 2026-09-10**: it was read where the prefix never arrived, and the record
  is fixed at the source instead (see the note below where its class was).
"""
import pytest

from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.token_field_mapping import map_tokens_to_fields


class TestASpinTypeWordAloneStillBypassesAutoRegistration:
    """``should_bypass_auto_register_formula`` is a GUARD, and its two mistakes
    are not symmetric: a false negative registers a declaration as an ordinary
    expression and reports success.  Both languages, unconditionally — the
    handler has no reliable session language (``backend_handler._nl_any``)."""

    def _handler(self):
        return JuliaBackendHandler(None)

    @pytest.mark.parametrize("jp,en", [
        ("スピン演算子",      "spin operator"),
        ("パウリ演算子",      "Pauli operator"),
        ("パウリ行列",        "Pauli matrix"),
        ("スピン演算子 S",    "spin operator S"),
        ("S をスピン演算子に", "make S a spin operator"),
    ])
    def test_japanese(self, jp, en):
        assert self._handler().should_bypass_auto_register_formula("S", jp)

    @pytest.mark.parametrize("jp,en", [
        ("スピン演算子",      "spin operator"),
        ("パウリ演算子",      "Pauli operator"),
        ("パウリ行列",        "Pauli matrix"),
        ("スピン演算子 S",    "spin operator S"),
        ("S をスピン演算子に", "make S a spin operator"),
    ])
    def test_english(self, jp, en):
        assert self._handler().should_bypass_auto_register_formula("S", en)

    def test_an_ordinary_formula_is_not_bypassed(self):
        """A positive control: without it the assertions above would pass if the
        guard started answering True for everything."""
        h = self._handler()
        assert not h.should_bypass_auto_register_formula("x", "x^2 + 1")


class TestTheUnsplitOrderTokenStillYieldsTheOrder:
    """``4次`` reaching the field mapper as ONE token — the case the tokenizer's
    script-boundary split did not separate."""

    @pytest.mark.parametrize("tokens,order", [
        (["4次", "テイラー展開"], "4"),
        (["2次", "テーラー展開"], "2"),
        (["40次", "テイラー展開"], "40"),
        (["4", "テイラー展開"],   "4"),      # the split case, taken by the branch above
    ])
    def test_order_is_read(self, tokens, order):
        args = map_tokens_to_fields(list(tokens), ["expr", "var", "point", "order"],
                                    {"E1"}, "ja")
        assert args.get("order") == order

    @pytest.mark.parametrize("tokens", [["テイラー展開"], ["x", "テイラー展開"]])
    def test_no_order_is_invented(self, tokens):
        args = map_tokens_to_fields(list(tokens), ["expr", "var", "point", "order"],
                                    {"E1"}, "ja")
        assert "order" not in args


# ``TestTheInjectedExpressionReferenceIsStrippedFromTheBubble`` was here, and the
# vocabulary it pinned (``expr_ref_prefix``) is gone with it (2026-09-10,
# DECL-BARE-NOUN-EN（残）②).  It trimmed a leading ``E1 に対して`` off the YOU
# bubble for ten hand-listed definition operations, on the stated ground that the
# pipeline PREPENDS that prefix.  Measured, the prefix never reached the step this
# read: the scaffold sat on the ``llm_supervision`` step, and all six occurrences
# in the committed published examples were there.  What it COULD have trimmed was
# the researcher's own writing — ``E5 の x をスカラーとする`` starts with a real
# ``E5 の`` and such lines exist in saved sessions.  The record is fixed at the
# source now (``interpret_pipeline`` records what was typed), so nothing needs
# trimming; see ``tests/test_injected_scaffold_is_not_the_record.py``.


class TestABareOperatorDeclarationGetsTheRightStatistics:
    """``fermion_kind_kw`` classifies the ``kind`` that
    ``bare_quantum_op_decl`` captured.  Getting it wrong does not fail: it
    registers a BOSON where the researcher wrote a fermion, and the
    anticommutator that follows comes back as the wrong algebra."""

    def _handler(self):
        return JuliaBackendHandler(None)

    @pytest.mark.parametrize("text,lang,operation", [
        ("b をフェルミオン演算子とする",      "ja", "define_fermion_operator"),
        ("b をフェルミ演算子とする",          "ja", "define_fermion_operator"),
        ("b をボゾン演算子とする",            "ja", "define_boson_operator"),
        ("b をボソン演算子とする",            "ja", "define_boson_operator"),
        ("define b as a fermion operator", "en", "define_fermion_operator"),
        ("define b as a boson operator",   "en", "define_boson_operator"),
    ])
    def test_statistics(self, text, lang, operation):
        r = self._handler().extra_pre_phase1(text, lang)
        assert r is not None and r.operation == operation
