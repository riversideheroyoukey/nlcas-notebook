r"""A colliding alias must not demand arguments its reading cannot have.

ALIAS-COLLISION-UNREACHABLE.  `OperationValidator._alias_index` is a
``dict[str, str]`` resolved first-come, so when two operations list the same
alias word only one is reachable by it.  BP-6 fixed the half where the
researcher NAMED a deciding field.  This is the other half: they named nothing,
and the first-come winner is an operation the sentence cannot possibly
complete.  Measured 2026-08-17::

    product of E1 and E2   →  failed: `var`, `lower`, `upper` are missing
    E1 と E2 の積を求める     →  failed: same

"The product of a and b" is a multiplication; ∏ is what wins the word, and it
demands an index and bounds that the sentence has no place for.  Meanwhile
`multiply_expressions` is completely specified by what was already written.

**The sibling is offered, not substituted.**  Deciding here would turn a safe
failure into a silent wrong answer — a missing argument and a different
intended operation look identical from the text.  So the caller gets a second
candidate and asks.  That is the same stance as `bound_name_collision`: where
nothing decides, report and get out of the way.

**Measuring the alias TABLE is not measuring what is looked up.**  Eight alias
words collide in the schema, but the tokenizer strips Japanese verb tails
before the index is consulted, and `積を求める` arrives as `積` — a word only
`product_expr` listed, so the family had no sibling to offer.  The schema
already claimed the longer spelling for `multiply_expressions`; `積` was added
so the claim is true at the level actually looked up.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.token_field_mapping import (
    alias_siblings_that_complete, refine_alias_candidates, scan_aliases,
)
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestTheSpellingIsAskedAboutInsteadOfRefused:
    @pytest.mark.parametrize("lang,phrase,offered", [
        ("en", "product of E1 and E2", "multiply_expressions"),
        ("en", "take the product of E1 and E2", "multiply_expressions"),
        ("ja", "E1 と E2 の積を求める", "multiply_expressions"),
    ], ids=["en_product", "en_take_product", "ja_product"])
    def test_the_completable_sibling_is_offered(self, lang, phrase, offered):
        sup = _sup()
        r = sup.interpret(phrase, known_expr_ids={"E1", "E2"}, lang=lang)
        assert r.ok and r.requires_confirmation, (r.ok, r.error)
        assert r.operation == offered, r.operation
        assert [c["operation"] for c in r.candidates] == [offered]

    def test_it_stops_for_the_answer_rather_than_running(self):
        """An ask that runs anyway is not an ask."""
        nb.start_engine(title="coll", mode="standard", language="en", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "x+1", "math")
            _run(eng, sup, "x-1", "math")
            r = _run(eng, sup, "product of E1 and E2")
            assert r["status"] == "needs_confirmation", r["status"]
            assert "expr_id" not in r or not r.get("expr_id")
        finally:
            nb.stop_engine()


class TestTheOriginalReadingIsUntouched:
    r"""The winner must keep working wherever the sentence does complete it —
    the rule is about sentences that CANNOT run, not about preferring siblings."""

    @pytest.mark.parametrize("lang,phrase,operation", [
        ("en", "product E1 with var=i, lower=1, upper=5", "product_expr"),
        ("en", "laplace transform of E1 with t=t, s=s", "laplace_transform"),
        ("en", "laplacian of E1", "scalar_laplacian"),
        ("ja", "E1 を x で積分して", "integrate_indefinite"),
        ("ja", "E1 と E2 を掛け合わせる", "multiply_expressions"),
    ], ids=["product_expr", "laplace_transform", "laplacian", "integrate",
            "multiply"])
    def test_a_complete_sentence_runs_without_asking(self, lang, phrase, operation):
        sup = _sup()
        r = sup.interpret(phrase, known_expr_ids={"E1", "E2"}, lang=lang)
        assert r.operation == operation, r.operation
        assert not r.requires_confirmation

    def test_a_word_with_no_family_offers_nothing(self):
        """`総乗` names only ∏, so an incomplete sentence stays an ordinary
        missing-argument failure — pinned because that path is what the rule
        must NOT start reaching into."""
        sup = _sup()
        r = sup.interpret("E1 を i で 1 から 5 まで総乗",
                          known_expr_ids={"E1"}, lang="ja")
        assert not r.ok


class TestItOffersNothingWhereNothingIsDecidable:
    def test_two_qualifying_siblings_offer_nothing(self):
        class _ThreeWay:
            def get_spec(self, op):
                return {"supported_backends": ["maxima"],
                        "required_fields": [] if op != "one" else ["missing"]}

            def operations_for_alias(self, name):
                return ["one", "two", "three"]

        got = alias_siblings_that_complete(
            ["w"], ["one"], {"one": (0, 1)}, _ThreeWay(), "maxima", set(), "en")
        assert got == []

    def test_a_sibling_the_backend_does_not_support_is_not_offered(self):
        v = OperationValidator()
        got = alias_siblings_that_complete(
            ["平均場デカップリング", "E1"], ["pair_substitute"],
            {"pair_substitute": (0, 1)}, v, "maxima", {"E1"}, "ja")
        assert got == []

    def test_the_alias_word_itself_is_not_offered_as_an_argument(self):
        r"""The matched span is removed before the fields are tried.  Left in,
        the operation's own NAME lands in a field: measured on the `laplace`
        family (before that collision was removed as bad data),
        `laplace E1 s` filled `t='laplace'`, which made the winner look
        complete AND would have run the transform with the real `s` dropped.
        No surviving family reaches it with real data, so a stub stands in —
        the guard is right regardless of which families exist today.  Found by
        mutation."""
        class _WordEatingField:
            def get_spec(self, op):
                return {"supported_backends": ["maxima"],
                        "required_fields": ["a", "b"] if op == "one" else []}

            def operations_for_alias(self, name):
                return ["one", "two"]

        # 'w' is the alias; with it excluded, `one` cannot fill both a and b.
        got = alias_siblings_that_complete(
            ["w", "E1"], ["one"], {"one": (0, 1)}, _WordEatingField(),
            "maxima", {"E1"}, "en")
        assert got == ["two"]

    def test_an_operation_needing_nothing_counts_as_complete(self):
        class _NoFields:
            def get_spec(self, op):
                return {"supported_backends": ["maxima"],
                        "required_fields": ["missing"] if op == "one" else []}

            def operations_for_alias(self, name):
                return ["one", "two"]

        got = alias_siblings_that_complete(
            ["w"], ["one"], {"one": (0, 1)}, _NoFields(), "maxima", set(), "en")
        assert got == ["two"]

    def test_one_sibling_reached_from_two_candidates_is_offered_once(self):
        r"""Defensive: needs an alias with three owners, and every family in
        the schema is a pair.  Counted twice, `len(out) == 1` fails and the
        offer is lost."""
        class _Shared:
            def get_spec(self, op):
                return {"supported_backends": ["maxima"],
                        "required_fields": [] if op == "third" else ["missing"]}

            def operations_for_alias(self, name):
                return ["one", "two", "third"]

        got = alias_siblings_that_complete(
            ["w"], ["one", "two"], {"one": (0, 1), "two": (0, 1)},
            _Shared(), "maxima", set(), "en")
        assert got == ["third"]

    def test_a_completable_sibling_on_another_backend_is_not_offered(self):
        class _OtherBackend:
            def get_spec(self, op):
                return {"supported_backends": ["julia_full"] if op == "two"
                        else ["maxima"],
                        "required_fields": ["missing"] if op == "one" else []}

            def operations_for_alias(self, name):
                return ["one", "two"]

        got = alias_siblings_that_complete(
            ["w"], ["one"], {"one": (0, 1)}, _OtherBackend(), "maxima",
            set(), "en")
        assert got == []

    def test_a_candidate_already_in_the_list_is_not_added_twice(self):
        v = OperationValidator()
        got = alias_siblings_that_complete(
            ["product", "E1", "E2"],
            ["product_expr", "multiply_expressions"],
            {"product_expr": (0, 1), "multiply_expressions": (0, 1)},
            v, "maxima", {"E1", "E2"}, "en")
        assert got == []


class TestTheTokenizerDecidesWhatIsLookedUp:
    r"""The measurement that corrected the ticket: a collision in the alias
    TABLE is not necessarily a collision at the word the index is asked for."""

    def _looked_up(self, word: str) -> str:
        v = OperationValidator()
        toks = _tokenize(word)
        hits = list(scan_aliases(toks, v.resolve_alias, v.max_alias_words))
        return " ".join(toks[hits[0][0]:hits[0][1]]) if hits else ""

    def test_a_japanese_verb_tail_is_stripped_before_the_lookup(self):
        assert self._looked_up("積を求める") == "積"
        assert self._looked_up("積分する") == "積分"

    def test_every_colliding_word_still_collides_after_tokenization(self):
        r"""A ratchet.  If a future alias collides only in its unstripped form,
        the family lookup silently has no sibling to offer — which is exactly
        how `積を求める` hid."""
        import collections
        import json
        v = OperationValidator()
        with open("schemas/common_operations.schema.json", encoding="utf-8") as f:
            data = json.load(f)
        owners: dict[str, set[str]] = collections.defaultdict(set)
        for op in data["operations"]:
            for aliases in op.get("aliases", {}).values():
                for a in aliases:
                    owners[a.lower()].add(op["operation"])
        colliding = {w for w, ops in owners.items() if len(ops) > 1}
        lost = {w for w in colliding
                if len(owners.get(self._looked_up(w).lower(), set())) < 2}
        assert not lost, lost


class TestBothHalvesAnswerThroughOneCall:
    r"""`refine_alias_candidates` is the single door: the named-field swap
    (BP-6) and the offered sibling are the same question about the same family,
    and answering it in two places is the defect shape this area keeps
    producing."""

    def test_a_named_field_still_swaps(self):
        v = OperationValidator()
        got = refine_alias_candidates(
            ["integrate", "lower=0", "upper=1", "E1"], ["integrate_indefinite"],
            {"integrate_indefinite": (0, 1)}, v, "maxima", {"E1"}, "en")
        assert got == ["integrate_definite"]

    def test_an_incomplete_winner_gains_a_second_candidate(self):
        v = OperationValidator()
        got = refine_alias_candidates(
            ["product", "E1", "E2"], ["product_expr"],
            {"product_expr": (0, 1)}, v, "maxima", {"E1", "E2"}, "en")
        assert got == ["product_expr", "multiply_expressions"]

    def test_a_complete_winner_gains_nothing(self):
        v = OperationValidator()
        got = refine_alias_candidates(
            ["product", "var=i", "lower=1", "upper=5", "E1"], ["product_expr"],
            {"product_expr": (0, 1)}, v, "maxima", {"E1"}, "en")
        assert got == ["product_expr"]
