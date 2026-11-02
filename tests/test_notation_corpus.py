"""第1期: one MEANING × every SPELLING of it, checked against a live CAS.

Why this file exists (docs/handoffs/one_question_many_answers_handoff.md §2.1):
``tests/test_user_cas_invariants.py`` verifies the boundary over ``sessions/*.json``
— real research data, but by construction the set of expressions that ALREADY
WENT THROUGH.  The defects fixed on 2026-08-09/10 were all of the other shape:
*a spelling the researcher can type that is read differently*.  Those are
invisible to a corpus of accepted expressions, and they are mechanical to find
once the variants are written down.

Two lessons carried over from docs/32 shape how this is written:

* **Ask the CAS, not the strings.**  Two silent-wrong bugs survived string
  round-trip checks for months (``k/r^2`` → ``k*sqrt(r)``, ``^`` associativity).
  Equivalence here is ``ratsimp(a - b) = 0`` whenever both sides reach the CAS.

* **Assert through the researcher's own route.**  Measuring
  ``register_expression`` directly is how a defect got filed against a path
  nobody takes (handoff §2.2).  ``TestThroughThePipeline`` re-checks the gaps
  through ``run_interpret_pipeline`` with the rule-based supervisor.

The corpus is data (``tests/corpus/notation_corpus.json``) so that adding a
spelling is a one-line edit, not a code change.
"""
import asyncio
import json
import pathlib

import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.shared_parser import UnknownNotationError

_CORPUS_PATH = pathlib.Path(__file__).parent / "corpus" / "notation_corpus.json"
_FAMILIES = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))["families"]

_PINNED = [f for f in _FAMILIES if f["status"] == "pinned"]
_GAPS = [f for f in _FAMILIES if f["status"] == "gap"]


def _ids(families):
    return [f["id"] for f in families]


def test_the_corpus_is_loaded_and_populated():
    """Guard: a corpus that silently failed to load makes every test below
    vacuous, and pytest reports that as a clean run.

    ``TestKnownGapsAreStillGaps`` legitimately parametrises over an EMPTY list
    once every measured gap is closed (as of 2026-08-11 it is), which pytest
    reports as a skip — that skip is the good outcome, and this guard is what
    distinguishes it from the whole file having evaporated.
    """
    assert len(_FAMILIES) >= 25, f"only {len(_FAMILIES)} families loaded"
    assert _PINNED, "no pinned families — the corpus is not asserting anything"
    assert {f["kind"] for f in _FAMILIES} <= {"equivalent", "distinct"}
    assert {f["status"] for f in _FAMILIES} <= {"pinned", "gap"}


@pytest.fixture(scope="module")
def engine():
    """One CAS process for the whole file: every check is a pure conversion plus
    a ``ratsimp``, none of them leaves state behind."""
    with NotebookEngine(title="notation corpus") as e:
        yield e


def _canonical_via_pipeline(latex: str) -> "str | None":
    """What the researcher's own typing produces, interceptors and all."""
    from tests.test_debug_repl import _sup
    from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
    with NotebookEngine(title="route") as e:
        req = InterpretRequest(tokens=[{"type": "math", "latex": latex}])
        res = asyncio.run(run_interpret_pipeline(req, e, _sup()))
        rows = res.get("registered_expressions") or []
        return rows[0].get("canonical") if rows else None


def _outcome(engine, latex: str, route: str = "handler") -> tuple[str, str]:
    """(status, payload) for one spelling, as the CAS would receive it.

    ``refused`` is a first-class outcome, not an error: a spelling that is
    refused with a reason is BETTER than one silently mistranslated, and two
    spellings of one meaning must agree even about being refused.

    *route* names WHERE the reading happens, because that is a real distinction
    in this system and not a testing detail (handoff §2.2).  ``handler`` asks
    the converter directly — fast, and right for anything the converter owns.
    ``pinned`` families whose rule lives in the request pipeline must say
    ``pipeline``: Unicode Greek in a math token is normalised by
    ``normalize_request_greek``, which the handler route never runs, so probing
    it the fast way would report a fixed defect as still broken.
    """
    h = engine._handler
    try:
        canonical = (_canonical_via_pipeline(latex) if route == "pipeline"
                     else h.latex_to_canonical(latex))
    except UnknownNotationError as exc:
        return "refused", getattr(exc, "i18n_key", str(exc))
    except Exception as exc:                       # noqa: BLE001 — reported, not swallowed
        return "error", f"{type(exc).__name__}"
    if canonical is None:
        return "error", "pipeline registered nothing"
    try:
        lowered = h.to_backend_code(canonical)
    except UnknownNotationError as exc:
        return "refused", getattr(exc, "i18n_key", str(exc))
    except Exception as exc:                       # noqa: BLE001
        return "error", f"{type(exc).__name__}"
    if lowered.status == "refused":
        return "refused", lowered.reason or ""
    return "lowered", lowered.code


def _same_value(engine, a: str, b: str) -> bool:
    """True when the CAS says the two lowered codes are the same value.

    Falls back to text equality when the difference cannot be evaluated (a
    display-only object, an unevaluated sum): there, identical code is the only
    evidence available, and it is still stronger than nothing.
    """
    if a == b:
        return True
    try:
        diff = engine._adapter.execute(f"ratsimp(({a}) - ({b}))")
    except Exception:                              # noqa: BLE001
        return False
    if not diff.ok:
        return False
    return (diff.value or "").strip().strip("[]() ,;=0") == ""


def _disagreements(engine, family) -> list[str]:
    """Every variant of *family* that differs from its first variant."""
    variants = family["variants"]
    route = family.get("route", "handler")
    head_status, head_payload = _outcome(engine, variants[0], route)
    out = []
    for spelling in variants[1:]:
        status, payload = _outcome(engine, spelling, route)
        if status != head_status:
            out.append(f"{spelling!r} is {status} ({payload!r}) but "
                       f"{variants[0]!r} is {head_status} ({head_payload!r})")
        elif status == "lowered" and not _same_value(engine, head_payload, payload):
            out.append(f"{spelling!r} → {payload!r} but "
                       f"{variants[0]!r} → {head_payload!r}")
    return out


class TestTheRelationScannerItself:
    r"""``find_top_level_relation`` is the SINGLE definition of where a relation
    splits — and its exclusions had no test at all.

    Found 2026-08-11 by mutation: breaking the ``:=`` exclusion (so ``a := b``
    reports an ``=`` at the colon) was caught by NOTHING in the 4,200-test
    suite, early files and late alike.  A rule that everything else defers to
    has to be pinned where it is defined, not only through its callers.
    """

    @pytest.mark.parametrize("text,expected", [
        ("a = b",        (2, "=")),
        ("a <= b",       (2, "<=")),
        ("a >= b",       (2, ">=")),
        ("a < b",        (2, "<")),
        ("a # b",        (2, "#")),        # every spelling of "not equal"
        # …and the exclusions, which is what the mutation exposed:
        ("f(x) := x^2",  None),           # a DEFINITION, not a relation
        ("a == b",       None),           # equality TEST, not an equation
        ("a + b",        None),
        # bracket-aware: a bounded sum's own subscript is not the equation
        (r"\sum_{i=1}^{3} i^2",       None),
        (r"\sum_{i=1}^{3} i^2 = 14",  (19, "=")),
        ("f(a=1)",       None),
        ("[a=1, b=2]",   None),
    ])
    def test_where_a_relation_splits(self, text, expected):
        from notebook.runtime.shared_parser import find_top_level_relation
        assert find_top_level_relation(text) == expected, text

    @pytest.mark.parametrize("text,lhs,rhs", [
        ("a = b",                     "a", "b"),
        (r"\sum_{i=1}^{3} i^2 = 14",  r"\sum_{i=1}^{3} i^2", "14"),
        ("f(x) := x^2",               None, None),      # not a relation
    ])
    def test_strip_lhs_and_split_relation_agree_with_it(self, text, lhs, rhs):
        from notebook.runtime.shared_parser import split_relation, strip_lhs
        got = split_relation(text)
        if lhs is None:
            assert got is None and strip_lhs(text) == text
        else:
            assert got == (lhs, "=", rhs)
            assert strip_lhs(text) == rhs


class TestOneMeaningOneReading:
    """Every spelling of one meaning must reach the CAS as the same value."""

    @pytest.mark.parametrize("family",
                             [f for f in _PINNED if f["kind"] == "equivalent"],
                             ids=_ids([f for f in _PINNED if f["kind"] == "equivalent"]))
    def test_all_spellings_agree(self, engine, family):
        bad = _disagreements(engine, family)
        assert not bad, (
            f"{family['id']} ({family['meaning']}) is read differently "
            f"depending on how it is written:\n  " + "\n  ".join(bad))


class TestDeliberateDistinctions:
    """Spellings this project has decided are DIFFERENT objects.

    Pinned so that a later "unification" cannot quietly merge them — the
    subscript auto-split that turned ``E_{rot}`` into ``E(r,o,t)`` was exactly
    such a merge, and it was documented nowhere.
    """

    @pytest.mark.parametrize("family",
                             [f for f in _PINNED if f["kind"] == "distinct"],
                             ids=_ids([f for f in _PINNED if f["kind"] == "distinct"]))
    def test_the_variants_stay_different(self, engine, family):
        assert _disagreements(engine, family), (
            f"{family['id']} ({family['meaning']}): the variants now read the "
            f"same. If that is intended, move this family to kind=equivalent "
            f"with a reason — do not delete it.\n  note: {family.get('note', '')}")


class TestKnownGapsAreStillGaps:
    """Measured broken on 2026-08-10 and not yet repaired.

    Asserting the gap (rather than skipping it) makes fixing one a visible,
    deliberate change, and stops a NEW breakage from hiding among them — the
    idiom ``tests/test_user_cas_invariants.py`` already uses.
    """

    @pytest.mark.parametrize("family", _GAPS, ids=_ids(_GAPS))
    def test_the_gap_is_unchanged(self, engine, family):
        assert _disagreements(engine, family), (
            f"{family['id']} now reads consistently — remove its `status: gap` "
            f"and its `gap_reason` from tests/corpus/notation_corpus.json.\n"
            f"  recorded: {family['gap_reason']}")


class TestOperandsReachTheCasLowered:
    r"""第3期: an operation must never hand the CAS the researcher's own form.

    ``\sum_{i=1}^{3} i^2`` has canonical ``\sum_{i=1}^3*i^2`` — LaTeX-shaped on
    purpose, because a canonical IS the researcher's writing (CLAUDE.md, "What
    ``Expression.canonical`` Is").  ``to_backend_code`` is the one door that
    turns it into ``sum(i^2,i,1,3)``.

    Measured 2026-08-11 by sweeping the operations: every one went through that
    door except ``solve_equation``, which is dispatched BEFORE
    ``build_command``'s per-argument loop.  It sent Maxima the LaTeX, which
    Maxima cannot read, so ``_res_`` — the ADAPTER'S OWN SCRATCH NAME — came
    back and was registered as the answer.
    """

    SUM = r"\sum_{i=1}^{3} i^2"          # value 14

    @pytest.fixture(scope="class")
    def engine(self):
        with NotebookEngine(title="lowered") as e:
            yield e

    @pytest.mark.parametrize("operation,extra", [
        ("simplify", {}), ("expand", {}), ("factor", {}),
        ("differentiate", {"var": "x"}),
        ("taylor_expand", {"var": "x", "point": "0", "order": "2"}),
    ])
    def test_no_operation_sends_latex_to_the_cas(self, engine, operation, extra):
        expr = engine.register_expression(
            engine._handler.latex_to_canonical(self.SUM), latex=self.SUM)
        step, _ = engine.run_operation(operation, {"expr": expr.display_id, **extra})
        assert "\\" not in (step.rendered_command or ""), (
            f"{operation} handed the CAS un-lowered text: {step.rendered_command!r}")

    def test_solve_lowers_the_equation_it_was_given(self, engine):
        src = r"S = x \sum_{i=1}^{3} i^2"
        expr = engine.register_expression(
            engine._handler.latex_to_canonical(src), latex=src)
        step, _ = engine.run_operation(
            "solve_equation", {"equation": expr.display_id, "var": "x"})
        assert "\\" not in (step.rendered_command or ""), step.rendered_command
        value = engine.expressions[step.output_refs[0]].canonical
        assert "_res_" not in value, (
            "the adapter's scratch name was registered as a result — the CAS "
            "could not read what it was sent")
        assert value.replace(" ", "") == "S/14", value

    def test_both_solve_operands_are_lowered_together(self, engine):
        """Never one side: lowering rewrites indexed names, so an equation
        lowered against a raw variable would be solved for a symbol that is no
        longer in it."""
        build = engine._handler.build_command(
            "solve_equation", {"equation": "a_i = b_i", "var": "a_i"}, {})
        assert build.ok, build.error
        assert "a(i)" in build.command and "a_i" not in build.command, build.command

    def test_an_unlowerable_operand_fails_the_build(self, engine):
        """R4: what cannot be lowered must fail the operation, never reach the
        CAS as raw text."""
        build = engine._handler.build_command(
            "solve_equation", {"equation": "|A| = 0", "var": "x"}, {})
        assert not build.ok and build.error, build


class TestAnEqualsInsideBracketsIsNotAnEquation:
    r"""第3期: the `=` in `\sum_{i=1}^{3}` is not the equation's `=`.

    Repaired in ``engine.py`` and ``lhs_formatting`` on 2026-08-08/10, and found
    STILL LIVE in the equation-op interceptors on 2026-08-11 — the same question
    answered in a place nobody had counted.  The sum has to be on the LEFT for
    it to show, because when it sits on the right the equation's own `=` happens
    to come first and ``split("=", 1)`` is right by luck:

        \sum_{i=1}^{3} i^2 = 14   →  trigsimp(ratsimp(1}^3*i^2 = 14))

    Maxima cannot read that, so ``_res_`` — the adapter's scratch name — came
    back and was registered as the answer.
    """

    @staticmethod
    def _follow_up(register: str, instruction: str) -> str:
        from tests.test_debug_repl import _sup, _handle_interpret, _run
        with NotebookEngine(title="eq") as e:
            sup = _sup()
            _run(_handle_interpret(register, e, sup, no_llm=True))
            before = len(e.expressions)
            _run(_handle_interpret(instruction, e, sup, no_llm=True))
            if len(e.expressions) <= before:
                return "(nothing produced)"
            out = list(e.expressions.values())[-1]
            return out.latex or out.canonical or ""

    @pytest.mark.parametrize("register,instruction,expect", [
        (r"\sum_{i=1}^{3} i^2 = 14", "E1 の右辺を計算して", "14"),
        (r"S = \sum_{i=1}^{3} i^2",  "E1 の右辺を計算して", "14"),
    ])
    def test_the_rhs_of_a_bounded_sum_equation(self, register, instruction, expect):
        got = self._follow_up(register, instruction)
        assert "_res_" not in got, (
            f"the adapter's scratch name reached the notebook: {got!r}")
        assert expect in got, got

    def test_both_sides_keeps_the_whole_sum(self):
        got = self._follow_up(r"\sum_{i=1}^{3} i^2 = 14", "E1 の両辺を 2 で割る")
        assert "_res_" not in got, got
        # the head of the sum must still be there — `\sum_{i` alone is the bug
        assert r"\sum_{i=1}" in got, f"the sum lost its head: {got!r}"

    def test_an_ordinary_equation_is_unaffected(self):
        got = self._follow_up("U = a*x+b", "E1 の両辺を 2 で割る")
        assert "U" in got and "2" in got, got

    @pytest.mark.parametrize("register,instruction", [
        (r"S = \sum_{i=1}^{3} i^2",  "E1 の両辺を 2 で割る"),
        (r"\sum_{i=1}^{3} i^2 = 14", "E1 の両辺を 2 で割る"),
        (r"\sum_{i=1}^{3} i^2 = 14", "E1 の両辺に 3 をかける"),
        ("U = a*x+b",                "E1 の両辺を 2 で割る"),
    ])
    def test_a_both_sides_result_is_one_equation_in_the_researchers_own_writing(
            self, register, instruction):
        r"""Two defects met in this one string (measured 2026-08-11).

        The result carried TWO `=` — ``\frac{S}{2} = (\sum_{i=1}^3*i^2)/(2) = 7``
        — because a second LHS label was prefixed onto a result that already had
        one, showing the researcher the un-evaluated middle.  And the label was
        read off the CANONICAL, so it published the CAS spelling (literal `*`)
        of notation the researcher had written properly.
        """
        got = self._follow_up(register, instruction)
        # the `=` of a sum's own subscript is not the equation's
        from notebook.runtime.shared_parser import split_relation
        rel = split_relation(got)
        assert rel is not None and rel[1] == "=", got
        assert split_relation(rel[2]) is None, f"the result carries two `=`: {got!r}"
        assert "*" not in got, f"CAS spelling reached the display: {got!r}"


class TestEverySpellingTakesOneRoute:
    r"""第2期: spellings of one notation must be claimed by the SAME interceptor.

    24 interceptors branch on the raw text before the general pipeline sees
    anything.  That is correct design — a lattice sum has to be claimed before
    semantic resolution, or its constraint is lost — but it means a notation
    can be split: on 2026-08-09 the lattice regex held ``\neq`` and not ``\ne``,
    so one spelling worked and its twin silently did something else.

    Nothing could catch that until ``Step.route`` existed, because the
    attribution was not recorded anywhere.  Now the question is one assertion.
    """

    @staticmethod
    def _route(latex: str) -> "str | None":
        from tests.test_debug_repl import _sup
        from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
        with NotebookEngine(title="route") as e:
            req = InterpretRequest(tokens=[{"type": "math", "latex": latex}])
            res = asyncio.run(run_interpret_pipeline(req, e, _sup()))
        return res.get("route")          # None = no interceptor claimed it

    @pytest.mark.parametrize("meaning,spellings", [
        ("a sum over ordered pairs",
         [r"\sum_{i \ne j} V_{i,j}", r"\sum_{i \neq j} V_{i,j}"]),
        ("a nearest-neighbour bond sum",
         [r"\sum_{\langle ij \rangle} V_{i,j}", r"\sum_{\langle i,j \rangle} V_{i,j}"]),
        ("an ordinary expression, claimed by nobody",
         ["a*x+b", r"a \cdot x + b"]),
    ])
    def test_all_spellings_are_claimed_by_the_same_interceptor(self, meaning,
                                                               spellings):
        routes = {s: self._route(s) for s in spellings}
        assert len(set(routes.values())) == 1, (
            f"{meaning}: the spellings take DIFFERENT routes, so they can "
            f"behave differently while looking the same — {routes}")


class TestThroughThePipeline:
    """The same question asked through the researcher's actual route.

    A defect measured only at ``register_expression`` describes a path nobody
    takes (handoff §2.2, where that mistake produced a wrong ticket).  These
    confirm the gaps survive the interceptor chain rather than being claimed by
    one of it.
    """

    @staticmethod
    def _canonical_via_pipeline(latex: str) -> "str | None":
        from tests.test_debug_repl import _sup
        from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
        with NotebookEngine(title="route") as e:
            req = InterpretRequest(tokens=[{"type": "math", "latex": latex}])
            res = asyncio.run(run_interpret_pipeline(req, e, _sup()))
            rows = res.get("registered_expressions") or []
            return rows[0].get("canonical") if rows else None

    @pytest.mark.parametrize("latex,gone", [
        # Each of these registered SUCCESSFULLY on 2026-08-10 with the command
        # sitting in the canonical as an ordinary symbol, so it multiplied
        # whatever stood beside it.  The check is that the symbol is gone from
        # what the researcher's own typing produces — measured here rather than
        # at the converter, because a fix that never reaches the route the
        # researcher takes is not a fix (handoff §2.2).
        (r"\big(a+b\big)*c", "big"),
        (r"\dfrac{a}{b}",    "dfrac"),
        (r"\mathit{a}*x",    "mathit"),
        (r"{{a}\over{b}}",   "over"),
        (r"\partial_x f",    "partial"),
        (r"\sum\limits_{i=1}^{3} i^2", "limits"),
    ])
    def test_the_command_no_longer_survives_as_a_symbol(self, latex, gone):
        canonical = self._canonical_via_pipeline(latex)
        assert canonical is not None, f"{latex!r} registered nothing"
        assert gone not in canonical, (
            f"{latex!r} still carries {gone!r} in its canonical ({canonical!r}) — "
            f"it reaches the CAS as a symbol and multiplies its neighbour.")

    @pytest.mark.parametrize("lang,text", [
        # Both languages, because the trigger is looked up BY SESSION LANGUAGE
        # (i18n/nl_patterns.json) — a fix verified in one language says nothing
        # about the other.
        ("ja", r"a \neq b と仮定する"), ("ja", r"a \ne b と仮定する"),
        ("ja", "a != b と仮定する"),
        ("en", r"assume a \neq b"), ("en", "assume a != b"),
        ("en", r"assume that a \ne b"),
    ])
    def test_not_equal_works_where_the_cas_can_hold_it(self, lang, text):
        r"""Refused as an EXPRESSION, accepted as an ASSUMPTION.

        That split is not a compromise, it is what the CAS supports: measured
        2026-08-11, ``assume(a # b)`` fails and ``assume(notequal(a,b))``
        registers the fact, after which Maxima answers ``is(equal(a,b))`` with
        false.  So the three spellings all reach a place where "a is not equal
        to b" actually means something, instead of the products they used to
        become.
        """
        from tests.test_debug_repl import _sup
        from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
        with NotebookEngine(title="assume") as e:
            e.session.language = [lang]
            req = InterpretRequest(tokens=[{"type": "text", "value": text}])
            res = asyncio.run(run_interpret_pipeline(req, e, _sup()))
            facts = e._adapter.execute("facts()").value
        assert res.get("status") == "assumption_added", res
        assert "notequal(a,b)" in (facts or ""), (
            f"the assumption did not reach the CAS in a spelling it holds: {facts!r}")

    @pytest.mark.parametrize("latex", [r"a \neq b", r"a \ne b", "a != b"])
    def test_a_not_equal_relation_is_refused_not_multiplied(self, latex):
        """Each spelling used to register SUCCESSFULLY as something else.

        ``a \\neq b`` became the product ``a * neq * b`` and ``a != b`` became
        ``factorial(a) = b``.  All three are refused now rather than supported:
        what Maxima accepts is ``assume(notequal(a,b))`` — an assumption, not an
        expression — so the expression path says so instead of half-building a
        relation the rest of the system could not consume.

        The refusal is asserted as the researcher meets it: ``/interpret``
        catches it and returns a failed response carrying the reason, which is
        how every parser refusal in this codebase surfaces.
        """
        from tests.test_debug_repl import _sup
        from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
        with NotebookEngine(title="route") as e:
            req = InterpretRequest(tokens=[{"type": "math", "latex": latex}])
            try:
                res = asyncio.run(run_interpret_pipeline(req, e, _sup()))
            except Exception as exc:                     # what the route catches
                res = {"status": "failed", "error": repr(exc),
                       "registered_expressions": []}
        assert res.get("status") == "failed", res
        # translated for the researcher, not a bare i18n key.  The message DOES
        # name the spellings it refuses, so it contains `\neq` — that is the
        # message doing its job, not a leak.
        assert "parser.unsupported_notation" not in res["error"], res["error"]
        rows = res.get("registered_expressions") or []
        assert not rows, f"a not-equal relation still registered: {rows}"
