r"""Naming a field chooses the operation, not just fills it.

BP-6's remaining half.  `map_tokens_to_fields`' pre-pass 0 (2026-08-06) states
the principle already — *a researcher who names the fields has removed the
ambiguity; reading the names is strictly better than guessing from position* —
and applied it to FILLING a field.  Choosing the operation was left to the
alias index, which is a ``dict[str, str]`` resolved first-come, so::

    integrate E1 with var=x, lower=0, upper=1  →  integrate(x^2, x)   completed
    integrate E1 over x from 0 to 1            →  integrate(x^2,x,0,1)

Same request, two answers.  The word ``integrate`` is listed by BOTH
`integrate_indefinite` and `integrate_definite`, the index can hold only one,
and the bounds the researcher had written were discarded with no error at all.

**The working spelling never reached the alias table.**  Measured 2026-08-16,
`integrate E1 over x from 0 to 1` is claimed by the `def_integral` NL pattern
in an interceptor, which requires ``from … to …``; handed to the supervisor
alone the same text yields ``var='over'``.  So the alias scan had no notion of
this family whatsoever and the interceptor was carrying it single-handed — the
field=value form is simply the spelling with no interceptor behind it.

Eight alias words are owned by two operations.  Where no field name tells the
members apart the index's first-come answer stands deliberately — a rule that
fires where nothing decides is a guess.

(A ninth, ``laplace``, was on this list until 2026-08-17, when the researcher
pointed out that "laplace of E1" does not mean the Laplacian in English.  It
does not: ``scalar_laplacian`` had simply listed the wrong word, and the alias
was removed rather than disambiguated.  A collision between a right reading and
a wrong one is not a collision.)
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.token_field_mapping import (
    named_fields, prefer_operations_holding_named_fields,
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


class TestTheWrittenBoundsSurvive:
    r"""The defect, end to end, in both languages — and against the spelling
    that already worked, which is the whole point: one request, one answer."""

    @pytest.mark.parametrize("lang,phrase", [
        ("en", "integrate E1 with var=x, lower=0, upper=1"),
        ("en", "integrate E1 over x from 0 to 1"),
        ("ja", "E1 を var=x, lower=0, upper=1 で積分して"),
    ], ids=["en_named", "en_from_to", "ja_named"])
    def test_a_definite_integral_is_definite(self, lang, phrase):
        nb.start_engine(title="bp6", mode="standard", language=lang, domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "x^2", "math")
            r = _run(eng, sup, phrase)
            assert r["operation"] == "integrate_definite", r["operation"]
            assert eng.expressions[r["expr_id"]].canonical.strip() == "1/3"
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,phrase", [
        ("en", "integrate E1 with respect to x"),
        ("ja", "E1 を x で積分して"),
    ], ids=["en", "ja"])
    def test_no_bounds_written_stays_indefinite(self, lang, phrase):
        """The rule must not fire where the researcher named nothing."""
        nb.start_engine(title="bp6b", mode="standard", language=lang, domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "x^2", "math")
            r = _run(eng, sup, phrase)
            assert r["operation"] == "integrate_indefinite", r["operation"]
        finally:
            nb.stop_engine()

    def test_a_partly_named_request_is_not_promoted(self):
        """`var=x` alone is held by BOTH members, so nothing is decided."""
        sup = _sup()
        r = sup.interpret("integrate E1 with var=x", known_expr_ids={"E1"},
                          lang="en")
        assert r.operation == "integrate_indefinite", r.operation


class TestItOnlyReadsFieldsThatExist:
    def test_an_unknown_name_is_not_a_named_field(self):
        known = OperationValidator().all_field_names()
        assert named_fields(["n=0", "into", "E1"], known) == set()
        assert named_fields(["lower=0", "upper=1"], known) == {"lower", "upper"}

    def test_a_name_with_no_value_is_not_a_named_field(self):
        known = OperationValidator().all_field_names()
        assert named_fields(["lower=", "upper=,"], known) == set()

    def test_the_case_of_the_name_does_not_matter(self):
        known = OperationValidator().all_field_names()
        assert named_fields(["Lower=0"], known) == {"lower"}

    def test_ordinary_mathematics_does_share_names_with_fields(self):
        r"""Written first as "`x=2` is mathematics, not a field name", which is
        FALSE — measured, nine field names are one or two characters (`a`, `b`,
        `s`, `t`, `x`, `y`, `op`, `sm`, `sp`), and `solve_ode` really does have
        fields called `x` and `y`.  So the name filter is NOT what keeps a
        substitution from steering the operation.  What does is the swap rule
        itself, pinned exhaustively below."""
        known = OperationValidator().all_field_names()
        assert named_fields(["x=2"], known) == {"x"}


class TestEveryPossibleSwapIsEnumerated:
    r"""A ratchet over the SCHEMA, because the safety argument is a property of
    the data, not of the code.

    A swap needs a field the candidate lacks and exactly one alias-sibling
    holds.  Measured 2026-08-17 that is reachable for eight alias words only,
    and no short field name is among the discriminators — `x`/`y` belong to
    `solve_ode` and `s`/`t` to `laplace_transform`, and neither operation
    collides with anything.  That is why `substitute x=2 into E1` cannot be
    steered.

    Add a colliding alias whose members are told apart by a one-letter field
    and this fails — which is the point: the danger would otherwise be
    invisible.
    """

    def _swaps(self) -> set[tuple[str, str, str]]:
        import itertools
        v = OperationValidator()
        families: dict[str, set[str]] = {}
        for canonical in v.known_operations:
            spec = v.get_spec(canonical) or {}
            for al in spec.get("aliases", {}).values():
                for a in al:
                    families.setdefault(a.lower(), set()).add(canonical)
        out: set[tuple[str, str, str]] = set()
        for word, members in families.items():
            if len(members) < 2:
                continue
            first = v.resolve_alias(word)
            fields = set().union(*(v.field_names(m) for m in members))
            for r in (1, 2):
                for combo in itertools.combinations(sorted(fields), r):
                    got = prefer_operations_holding_named_fields(
                        [word] + [f"{f}=1" for f in combo],
                        [first], {first: (0, 1)}, v, "maxima")
                    if got != [first]:
                        for f in combo:
                            if prefer_operations_holding_named_fields(
                                    [word, f"{f}=1"], [first],
                                    {first: (0, 1)}, v, "maxima") != [first]:
                                out.add((word, f, got[0]))
        return out

    def test_the_discriminating_fields_are_exactly_these(self):
        got = {(w, f) for w, f, _ in self._swaps()}
        assert got == {
            ("integrate", "lower"), ("integrate", "upper"),
            ("integral", "lower"), ("integral", "upper"),
            ("積分", "lower"), ("積分", "upper"),
            ("積分する", "lower"), ("積分する", "upper"),
            ("product", "a"), ("product", "b"),
            ("積を求める", "a"), ("積を求める", "b"),
            # `積` joined on 2026-08-17 (ALIAS-COLLISION-UNREACHABLE): the
            # tokenizer strips `を求める`, so the longer spelling the schema
            # already claimed for `multiply_expressions` was never the word
            # looked up.  This enumeration is what noticed.
            ("積", "a"), ("積", "b"),
            ("式を掛け合わせる", "expr_a"), ("式を掛け合わせる", "expr_b"),
            ("式を掛け合わせる", "op"),
        }, got

    def test_no_field_a_researcher_writes_as_mathematics_can_swap(self):
        r"""`a`/`b` discriminate `product`, and they ARE short — but they are
        `multiply_expressions`' own operand fields, so writing them names that
        operation.  The names that appear in ordinary mathematics beside these
        words (`x`, `y`, `s`, `t`) must not appear here."""
        assert not {f for _, f, _ in self._swaps()} & {"x", "y", "s", "t"}

    def test_laplace_names_the_transform_and_nothing_else(self):
        r"""Reported here on 2026-08-16 as "a dead alias revived" — naming a
        coordinate system made `laplace` reach `scalar_laplacian`.  The
        researcher asked whether English works that way.  It does not: "the
        Laplace of f" is the transform (or the person), and the Laplacian is
        "the Laplacian of f" / ∇²f.  `scalar_laplacian` had listed the wrong
        word, so the revival was reviving a mistake — the alias was removed
        instead.  Japanese was never affected (`ラプラシアン` and
        `ラプラス演算子` are both right, and bare `ラプラス` was never listed)."""
        v = OperationValidator()
        assert v.operations_for_alias("laplace") == ["laplace_transform"]
        assert v.resolve_alias("laplacian") == "scalar_laplacian"
        assert v.resolve_alias("ラプラス演算子") == "scalar_laplacian"


class TestItRefusesToGuessWhereNothingDecides:
    r"""The five alias collisions that predate BP-6 keep first-come."""

    def test_an_undecidable_collision_is_left_alone(self):
        v = OperationValidator()
        assert set(v.operations_for_alias("平均場デカップリング")) == {
            "pair_substitute", "wick_contract"}
        # `expr` is held by both, so naming it decides nothing and the
        # first-come answer stands.
        got = prefer_operations_holding_named_fields(
            ["平均場デカップリング", "expr=E1"], ["pair_substitute"],
            {"pair_substitute": (0, 1)}, v, "julia_full")
        assert got == ["pair_substitute"]

    def test_a_sibling_the_backend_does_not_support_is_not_chosen(self):
        v = OperationValidator()
        got = prefer_operations_holding_named_fields(
            ["integrate", "lower=0", "upper=1"], ["integrate_indefinite"],
            {"integrate_indefinite": (0, 1)}, v, "julia")
        assert got == ["integrate_indefinite"]

    def test_a_field_no_member_holds_decides_nothing(self):
        r"""`point` belongs to the Taylor family, and NEITHER integrate holds
        it.  Without the sibling's own field check this swapped anyway — the
        families are pairs today, so "the other one" is always exactly one
        operation and the rule would fire on any unrecognised field at all.
        Found by mutation; nothing else in this file reached it."""
        v = OperationValidator()
        got = prefer_operations_holding_named_fields(
            ["integrate", "point=0"], ["integrate_indefinite"],
            {"integrate_indefinite": (0, 1)}, v, "maxima")
        assert got == ["integrate_indefinite"]

    def test_two_siblings_that_both_qualify_decide_nothing(self):
        r"""Defensive: every alias family in the schema is a PAIR today, so no
        real spelling can reach this.  A stub stands in for the third member,
        because a guard nothing exercises is a guard nothing has tested."""
        class _ThreeWay:
            def all_field_names(self):
                return frozenset({"lower"})

            def field_names(self, op):
                return frozenset() if op == "one" else frozenset({"lower"})

            def operations_for_alias(self, name):
                return ["one", "two", "three"]

            def get_spec(self, op):
                return {"supported_backends": ["maxima"]}

        got = prefer_operations_holding_named_fields(
            ["w", "lower=0"], ["one"], {"one": (0, 1)}, _ThreeWay(), "maxima")
        assert got == ["one"]

    def test_a_multi_word_alias_is_matched_whole(self):
        r"""Also defensive — no colliding alias is multi-word today.  Reading
        only the span's first token would look up `definite`, a different word
        with a different family."""
        class _TwoWord:
            def all_field_names(self):
                return frozenset({"lower"})

            def field_names(self, op):
                return frozenset() if op == "one" else frozenset({"lower"})

            def operations_for_alias(self, name):
                return ["one", "two"] if name == "take the integral" else []

            def get_spec(self, op):
                return {"supported_backends": ["maxima"]}

        got = prefer_operations_holding_named_fields(
            ["take", "the", "integral", "lower=0"], ["one"], {"one": (0, 3)},
            _TwoWord(), "maxima")
        assert got == ["two"]

    def test_the_alias_word_bounds_the_search(self):
        r"""`summation` holds `lower`/`upper` too — it must not be reachable
        from the word `integrate`."""
        v = OperationValidator()
        got = prefer_operations_holding_named_fields(
            ["integrate", "lower=0", "upper=1"], ["integrate_indefinite"],
            {"integrate_indefinite": (0, 1)}, v, "maxima")
        assert got == ["integrate_definite"]


class TestTheFamilyIsReadFromTheSpecs:
    def test_the_index_answers_one_and_the_specs_answer_both(self):
        r"""The mechanism: `resolve_alias` is single-valued by construction, so
        the family relation has to come from the specs."""
        v = OperationValidator()
        assert v.resolve_alias("integrate") == "integrate_indefinite"
        assert set(v.operations_for_alias("integrate")) == {
            "integrate_indefinite", "integrate_definite"}

    def test_an_unknown_word_has_no_family(self):
        assert OperationValidator().operations_for_alias("nonesuch") == []

    def test_a_canonical_name_is_its_own_alias(self):
        r"""Twenty-one operations do not list their own canonical name among
        their aliases — `define_function` is one — so the family lookup has to
        add it, or typing an operation's real name finds nothing."""
        v = OperationValidator()
        assert v.operations_for_alias("define_function") == ["define_function"]
        assert "integrate_definite" in v.operations_for_alias("integrate_definite")

    def test_field_names_covers_optional_fields_too(self):
        r"""Required-only would miss every field a researcher may name but need
        not — `assign_to` is optional on both integrals, and a rule that cannot
        see it would treat naming it as evidence for the sibling."""
        v = OperationValidator()
        assert {"expr", "var", "lower", "upper", "assign_to"} <= v.field_names(
            "integrate_definite")
        assert "assign_to" in v.field_names("integrate_indefinite")
        assert v.field_names("no_such_operation") == frozenset()

    def test_naming_only_an_optional_field_does_not_swap(self):
        v = OperationValidator()
        got = prefer_operations_holding_named_fields(
            ["integrate", "assign_to=F"], ["integrate_indefinite"],
            {"integrate_indefinite": (0, 1)}, v, "maxima")
        assert got == ["integrate_indefinite"]
