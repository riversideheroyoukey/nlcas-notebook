"""NL-GRAMMAR ③: one INSTRUCTION × both languages, measured against the system.

``tests/test_notation_corpus.py`` does this for what a researcher WRITES (one
meaning, every spelling).  This file does it for what a researcher SAYS — the
instruction that names an operation and its operands.

**Why the corpus is lifted from the scenario docs and not written here.**  On
2026-08-22 three regressions walked past 5813 tests and were caught only by
regenerating the worked examples.  The route being measured was right; the INPUT
was invented rather than taken from what researchers actually type.  Every row
here is a verbatim line from a ```scenario block in ``docs/scenarios/*.md``.

**What it can and cannot find.**  Every committed scenario line passes by
construction — ``scripts/generate_scenario_examples.py`` asserts on the first
failure, so a corpus of scenario lines can never report a failing status
(measured 2026-08-23: 1046/1046 succeeded, 0 failures — a vacuous number).  The
signal is not "does it run" but **do the two languages agree**: same operation,
same operands, same expressions created.  Nothing forces that, and measuring it
found 14 divergences in 523 rows, from two root causes.

Two checks, deliberately split by cost:

``TestTheCorpusDescribesTheScenarios``
    Instant, no CAS, always runs.  The corpus must still be the scenario docs
    verbatim, so editing a scenario without re-measuring fails here.

``TestBothLanguagesReachTheSameOperation``
    Replays every scenario through a live CAS (~4 min), so it is opt-in behind
    ``--run-scenarios`` / ``RUN_SCENARIOS=1`` — the same gate and the same
    reasoning as ``tests/test_examples_up_to_date.py``.
"""
import collections
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_CORPUS_PATH = REPO / "tests" / "corpus" / "instruction_corpus.json"
_CORPUS = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
_ROWS = _CORPUS["rows"]
_PINNED = [r for r in _ROWS if r["status"] == "pinned"]
_GAPS = [r for r in _ROWS if r["status"] == "gap"]


def test_the_corpus_is_loaded_and_populated():
    """Guard: a corpus that failed to load makes everything below vacuous, and
    pytest reports that as a clean run (the notation corpus carries the same
    guard, for the same reason)."""
    assert len(_ROWS) >= 400, f"only {len(_ROWS)} rows loaded"
    assert _PINNED, "no pinned rows — the corpus is not asserting anything"
    assert {r["status"] for r in _ROWS} <= {"pinned", "gap"}
    assert {r["kind"] for r in _ROWS} <= {"text", "math", "op"}


class TestTheCorpusDescribesTheScenarios:
    """The corpus must still BE the scenario docs — verbatim, and paired.

    Without this the corpus rots silently: a scenario line gets reworded, the
    corpus keeps pinning the old spelling, and the ratchet goes on passing while
    asserting about text nobody types any more.
    """

    @staticmethod
    def _scenario_lines():
        from scripts.generate_scenario_examples import (_parse_scenario_md,
                                                        SCENARIOS_DIR)
        out = {}
        for jp in sorted(SCENARIOS_DIR.glob("scenario_*_jp.md")):
            title = jp.stem[:-3]
            parsed = _parse_scenario_md(jp)
            if not parsed:
                continue
            en = SCENARIOS_DIR / f"{title}_en.md"
            ep = _parse_scenario_md(en) if en.exists() else None
            out[title] = (parsed[1], ep[1] if ep else None)
        return out

    @staticmethod
    def _text_of(cmd):
        if isinstance(cmd, dict):
            if "note" in cmd:
                return None                      # notes are prose, not instructions
            if "op" in cmd:
                return f"@op: {cmd['op']}"
            return cmd["math"]
        return cmd

    def test_every_scenario_instruction_is_in_the_corpus(self):
        rows = {(r["scenario"], r["index"]): r for r in _ROWS}
        missing, wrong = [], []
        for title, (jp_cmds, en_cmds) in self._scenario_lines().items():
            for idx, cmd in enumerate(jp_cmds):
                text = self._text_of(cmd)
                if text is None:
                    continue
                row = rows.get((f"{title}", idx)) or rows.get((title, idx))
                if row is None:
                    missing.append((title, idx, text[:60]))
                elif row["jp"] != text:
                    wrong.append((title, idx, row["jp"], text))
                elif en_cmds is not None and idx < len(en_cmds):
                    en_text = self._text_of(en_cmds[idx])
                    if en_text is not None and row["en"] != en_text:
                        wrong.append((title, idx, row["en"], en_text))
        assert not missing, (
            f"{len(missing)} scenario instruction(s) are not in the corpus — "
            f"re-run scripts/measure_instruction_corpus.py + "
            f"scripts/build_instruction_corpus.py: {missing[:5]}")
        assert not wrong, (
            f"{len(wrong)} corpus row(s) no longer match the scenario doc — "
            f"the corpus is pinning text nobody types: {wrong[:5]}")

    def test_no_corpus_row_refers_to_a_scenario_that_is_gone(self):
        titles = set(self._scenario_lines())
        orphans = sorted({r["scenario"] for r in _ROWS} - titles)
        assert not orphans, f"corpus rows for deleted scenarios: {orphans}"


class TestTheMeasuredDivergencesAreRecorded:
    """The gaps are DATA, and each one names its root cause.

    A gap that gets fixed must fail here, so the corpus is updated in the commit
    that fixes it — the notation corpus's contract, and the reason
    ``docs/refusals.md`` grew stale rows describing refusals already lifted.
    """

    def test_gaps_are_all_of_the_two_known_shapes(self):
        assert {r["gap"] for r in _GAPS} <= {"operands", "operation",
                                             "created_expressions", "route",
                                             "card_label", "unpaired"}

    def test_gap_rows_record_both_readings(self):
        for r in _GAPS:
            if r["gap"] == "unpaired":
                continue
            assert "operands_jp" in r and "operands_en" in r, r["id"]
            if r["gap"] == "route":
                assert r["route_jp"] != r["route_en"], r["id"]

    def test_the_gap_count_has_not_grown(self):
        """Ratchet.  Measured 2026-08-23: 14 of 523, then **3** the same day.

        Two root causes.  The first — ``heisenberg_equation`` registering its
        operand as an expression in Japanese but not in English, which shifted
        every later E-id in that scenario — is FIXED
        (``operand_boundary.formula_is_an_operation_operand``): 11 rows closed
        and the created-expression divergence went 8 → 0.  What remains is the
        second: ``substitute`` is handed raw LaTeX in Japanese and lowered
        canonical in English.  Both currently produce the same result, so it is
        latent rather than wrong — but it is one question with two answers.

        Lower this when it is fixed; raising it needs a reason in the commit.

        Then measured again with the ROUTE compared (NL-GRAMMAR ②): **6**.

        All six are now route-ONLY: the operand divergence was closed by making
        `substitute` record its two sides the way `substitute_into` always has.
        Measured first, on the six real spellings the corpus holds: raw and
        converted give the IDENTICAL result every time, so what differed was the
        HISTORY, not the answer — and three of the four substitution routes
        already recorded the converted form.

        What is left is that substitution is claimed by two rules whose split
        does not line up between the languages (`information_theory:13` is the
        MIRROR of the other three), and that "set the initial conditions"
        reaches `specialize_conditions` in Japanese and `substitute_free_params`
        in English.  Both rules now answer identically, so this is a DRIFT
        detector rather than a live defect: if an edit to one is not mirrored in
        the other, a row moves from `route` to `operands` or `operation` and
        this file has to say so.

        `routes` (the whole chain, not just the winner) is compared too, because
        a line can be touched by more than one rule — step 0 registers the
        leading formula and an interceptor then handles the rest.  Recording
        only the winner hid the step-0 attribution on every such line."""
        assert len(_GAPS) <= 6, (
            f"{len(_GAPS)} divergences, was 14 — a new instruction is read "
            f"differently in the two languages: "
            f"{[r['id'] for r in _GAPS if r['id'] not in _KNOWN_GAP_IDS]}")

    def test_the_known_gaps_are_the_ones_recorded(self):
        assert {r["id"] for r in _GAPS} == _KNOWN_GAP_IDS


#: The 6 left, listed so that closing one is visible as a NAMED change rather
#: than a number going down.  All six are route-only: two rules answer one
#: request and both now answer it the same way.
_KNOWN_GAP_IDS = {
    "central_force_conservation:9", "central_force_conservation:15",
    "central_force_conservation:17", "pendulum_lagrangian:16",
    "information_theory:13",          # the mirror: jp and en swap rules
    "damped_oscillator_numeric:15",   # two entirely different rules
}


#: Operation rows whose cards carry NO operation label.  Measured 28 on
#: 2026-08-27, then 10, and **0 on 2026-09-05** — every line that reaches an
#: operation now draws a labelled card, in both languages.  It is a ratchet, so
#: raising it needs a reason in the commit.
_UNLABELLED_CEILING = 0

#: Rows that never reached an operation — a registration or a declaration draws
#: the expression itself, and an "operation label" would be inventing one.
_NOT_AN_OPERATION = ("(expression)",)


def _operation_rows():
    for r in _ROWS:
        op = r.get("operation")
        if not op or op in _NOT_AN_OPERATION:
            continue
        if op.startswith(("define_", "declare_")):
            continue
        yield r


class TestTheCardTheResearcherSeesIsMeasuredToo:
    """The corpus compares what the screen shows, not only what was computed.

    Operation, operands, created expressions and route can all agree while the
    SCREEN differs — measured 2026-08-27, substitution answered identically in
    the two languages and drew a labelled card in one and an unlabelled one in
    the other, and every one of the 30 solve lines drew an unlabelled card in
    BOTH.  Neither was visible to any field this file compared, and both had
    been published in the worked examples for a long time.

    ``labelled_*`` is ANY, not ALL: an operation may answer with several
    expressions (solve returns every root), and those later cards are further
    answers to the same operation rather than further operations.
    """

    def test_the_column_is_actually_populated(self):
        """A guard: if the measurement stopped recording it, everything below
        would pass vacuously."""
        seen = [r for r in _ROWS if r.get("labelled_jp") is not None]
        assert len(seen) >= 400, f"only {len(seen)} rows carry the column"

    def test_the_two_languages_agree_on_whether_the_card_is_labelled(self):
        bad = [r["id"] for r in _ROWS
               if r.get("labelled_jp") is not None
               and r.get("labelled_en") is not None
               and r["labelled_jp"] != r["labelled_en"]]
        assert not bad, (
            "one language draws a labelled card and the other does not — the "
            f"same request, a different screen: {bad}")

    def test_the_unlabelled_operation_count_has_not_grown(self):
        """Ratchet: 28 → 10 → **0**.

        The last 10 were lines whose operation runs on a SCAFFOLD (`_T1`), so
        the label built from the operation's own args named a key the researcher
        has never seen.  Closed by letting the handler say what the scaffold
        stands for — it is the only thing that knows — while the WORDING stays
        in `_visible_action` (open_items SCAFFOLD-LABEL, closed 2026-09-05).
        """
        rows = list(_operation_rows())
        assert len(rows) >= 150, f"only {len(rows)} operation rows — check the filter"
        unlabelled = [r["id"] for r in rows
                      if r.get("labelled_jp") is False or r.get("labelled_en") is False]
        assert len(unlabelled) <= _UNLABELLED_CEILING, (
            f"{len(unlabelled)} operation rows draw no labelled card, was "
            f"{_UNLABELLED_CEILING}: {unlabelled[:8]}")


@pytest.mark.scenarios
class TestBothLanguagesReachTheSameOperation:
    """The corpus, re-measured against a live CAS.  Opt-in: ``--run-scenarios``.

    Parametrised per scenario so a failure names the scenario rather than the
    whole corpus, and so one broken scenario does not hide the other 47 (the
    2026-08-22 abort-on-first-failure lesson, in test form).
    """

    @pytest.mark.parametrize("scenario", sorted({r["scenario"] for r in _ROWS}))
    def test_scenario_matches_the_corpus(self, scenario):
        import asyncio
        from scripts.generate_scenario_examples import (_parse_scenario_md,
                                                        SCENARIOS_DIR)
        from scripts.measure_instruction_corpus import _replay
        from scripts.build_instruction_corpus import build

        jp = SCENARIOS_DIR / f"{scenario}_jp.md"
        parsed = _parse_scenario_md(jp)
        assert parsed, f"{scenario} has no scenario block"
        domain, commands = parsed
        records: list = []

        async def run():
            await _replay(scenario, domain, commands, "ja", "jp", records)
            en = SCENARIOS_DIR / f"{scenario}_en.md"
            ep = _parse_scenario_md(en) if en.exists() else None
            if ep:
                await _replay(scenario, ep[0] or domain, ep[1], "en", "en",
                              records)

        asyncio.run(run())
        fresh = {r["id"]: r for r in build(records)["rows"]}
        pinned = {r["id"]: r for r in _ROWS if r["scenario"] == scenario}
        assert set(fresh) == set(pinned), (
            f"{scenario}: the lines changed — re-run the corpus scripts")
        drift = []
        for rid, want in pinned.items():
            got = fresh[rid]
            # `labelled_*` is in this list on purpose: a column the corpus
            # records but the replay does not compare is a column that cannot
            # catch anything (the "wire it to the screen or do not add it"
            # rule).  It is what makes the solve/substitute label fixes stick.
            for field in ("operation", "status", "route", "routes", "operands",
                          "operands_jp", "operands_en", "route_jp", "route_en",
                          "routes_jp", "routes_en",
                          "labelled_jp", "labelled_en"):
                if want.get(field) != got.get(field):
                    drift.append((rid, field, want.get(field), got.get(field)))
        assert not drift, (
            f"{scenario}: {len(drift)} instruction(s) no longer read the same "
            f"way: {drift[:4]}")
