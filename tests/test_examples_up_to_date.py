"""EXAMPLES-DRIFT: the committed worked examples must match what the code does NOW.

**Why this test exists.**  `docs/scenarios/*.md` is the single source of each
scenario's commands, and `scripts/generate_scenario_examples.py` turns them into
`domains/<domain>/examples/*.json` + `*.tex` — the published worked examples, and
the "log is proof" artifacts of the practice loop.  Nothing checked that they
still matched the code.

Measured 2026-08-06: regenerating on a clean tree changed **90 files**.  The
examples had fallen behind roughly five root-cause fixes, and the published
`.tex` still showed `\\dot{theta}` (which typesets as the italic word "theta",
not θ) and literal `*` in the middle of formulae.  Nobody had noticed, nothing
was filed, and it surfaced only as a by-product of an unrelated change — where
it very nearly masked a real regression (7 scenarios had genuinely broken, in a
diff of 124 files of which 90 were unrelated staleness).

Three chances to catch it were all missed: the author of each fix did not
regenerate; the scenario docs state expected results by hand but nothing
compares them to the output; and the 3700-test suite never runs the generator.
This test is the third one.

**What it asserts.**  Regenerating into a scratch tree reproduces the committed
files byte-for-byte, once the genuinely per-run values are removed.  That set is
not guessed — it is derived by generating TWICE and keeping whatever differed
(see `_VOLATILE_KEYS`).

**It is opt-in** (`--run-scenarios` / `RUN_SCENARIOS=1`, gate in
tests/conftest.py) because it drives every scenario through a live CAS and takes
minutes, the same treatment the Playwright E2E tests get.  Run it before merging
a change to the parser, the LaTeX converter, the display bridge or the export —
those are exactly the layers whose fixes went missing.

**When it fails**, the fix is normally to regenerate and commit the result:

    PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py

Read the diff first, though: this test cannot tell "the examples got better"
from "the examples got wrong".  It only says they no longer describe the code.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DOMAINS = ROOT / "domains"

#: Values that legitimately differ between two runs of the same input.
#:
#: DERIVED, NOT GUESSED — generated twice into separate trees (2026-08-06, 84
#: json files) and kept exactly what differed: timestamp (2421 occurrences),
#: expr_uid (702), created_at (168), session_id (84).  Nothing else differed.
#: Do NOT widen this list to make a failure go away: anything outside it means
#: the examples no longer describe the code, which is what the test is for.
_VOLATILE_KEYS = frozenset({
    "timestamp",       # wall clock
    "expr_uid",        # random per registration
    "created_at",      # wall clock
    "session_id",      # random per session
})

# The .tex export needs NO normalisation — measured over the same two runs, all
# 84 exports were byte-identical.  A date line was expected and is not there, so
# TeX is compared verbatim: an unmeasured "volatile" rule would only be a place
# for a real difference to hide.


def _strip_volatile(node):
    """Recursively drop per-run values so two runs of the same input compare equal."""
    if isinstance(node, dict):
        return {k: _strip_volatile(v) for k, v in node.items()
                if k not in _VOLATILE_KEYS}
    if isinstance(node, list):
        return [_strip_volatile(v) for v in node]
    return node


def _norm_json(path: pathlib.Path):
    return _strip_volatile(json.loads(path.read_text(encoding="utf-8")))


def _norm_tex(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


#: Committed examples the generator does NOT own, listed explicitly rather than
#: loosening the check.  These come from a LIVE case-study run (APIEMS 2026,
#: docs/scenarios/scenario_or_production_planning_en.md — a doc with no
#: ```scenario``` block, which says "手動配置"): a base run, a re-run with
#: R_1=120, and a fork-point snapshot holding E1 only.  They exercise live
#: branch fork/switch and the numeric layer, which the generator cannot drive.
#:
#: Anything added here is a file NOBODY re-derives, so it can silently rot the
#: way the generated ones did — add only with a reason, and prefer making the
#: scenario generator-runnable instead.
_NOT_GENERATOR_OWNED = frozenset({
    "operations_research/examples/scenario_or_production_planning_base_en.json",
    "operations_research/examples/scenario_or_production_planning_base_en.tex",
    "operations_research/examples/scenario_or_production_planning_increased_r1_en.json",
    "operations_research/examples/scenario_or_production_planning_increased_r1_en.tex",
    "operations_research/examples/scenario_or_production_planning_fork_point_en.json",
})


def _committed_examples() -> "list[pathlib.Path]":
    """Every committed example the GENERATOR is responsible for."""
    found = sorted(DOMAINS.glob("*/examples/scenario_*.json")) + \
        sorted(DOMAINS.glob("*/examples/scenario_*.tex"))
    return [p for p in found
            if p.relative_to(DOMAINS).as_posix() not in _NOT_GENERATOR_OWNED]


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Regenerate every worked example into a scratch domains root.

    The generator writes under `out_root/<domain>/examples/`, so the committed
    examples are never touched — finding out whether they are stale must not
    itself make them fresh.
    """
    import generate_scenario_examples as gen
    out = tmp_path_factory.mktemp("examples")
    asyncio.get_event_loop().run_until_complete(gen.main(out))
    return out


class TestExamplesDescribeTheCurrentCode:
    def test_every_committed_example_was_regenerated(self, regenerated):
        """A scenario doc that was renamed or deleted without regenerating leaves
        an orphan example behind — and an orphan is stale by construction."""
        produced = {p.relative_to(regenerated).as_posix()
                    for p in regenerated.rglob("scenario_*") if p.is_file()}
        committed = {p.relative_to(DOMAINS).as_posix() for p in _committed_examples()}
        orphans = sorted(committed - produced)
        assert not orphans, (
            "Committed example(s) that the generator no longer produces — the "
            "scenario doc was renamed, deleted or moved to another domain "
            "without regenerating:\n  " + "\n  ".join(orphans))

    def test_no_new_example_is_missing_from_the_repo(self, regenerated):
        produced = {p.relative_to(regenerated).as_posix()
                    for p in regenerated.rglob("scenario_*")
                    if p.is_file() and p.suffix in (".json", ".tex")}
        committed = {p.relative_to(DOMAINS).as_posix() for p in _committed_examples()}
        missing = sorted(produced - committed)
        assert not missing, (
            "The generator produces example(s) that are not committed — run "
            "scripts/generate_scenario_examples.py and add them:\n  "
            + "\n  ".join(missing))

    def test_committed_examples_match_a_fresh_run(self, regenerated):
        """The check that was missing.  A failure means the examples no longer
        describe the code — usually because a fix landed without regenerating."""
        stale = []
        for committed in _committed_examples():
            fresh = regenerated / committed.relative_to(DOMAINS)
            if not fresh.exists():
                continue                      # reported by the orphan test above
            try:
                if committed.suffix == ".json":
                    same = _norm_json(committed) == _norm_json(fresh)
                else:
                    same = _norm_tex(committed) == _norm_tex(fresh)
            except (OSError, ValueError) as exc:
                stale.append(f"{committed.relative_to(ROOT)}: unreadable ({exc})")
                continue
            if not same:
                stale.append(str(committed.relative_to(ROOT)))
        assert not stale, (
            f"{len(stale)} committed example(s) no longer match what the code "
            "produces.  Regenerate and READ THE DIFF before committing — this "
            "test cannot tell an improvement from a regression:\n"
            "  PATH=\"$HOME/.juliaup/bin:$PATH\" .venv/bin/python "
            "scripts/generate_scenario_examples.py\n  "
            + "\n  ".join(stale[:40])
            + (f"\n  … and {len(stale) - 40} more" if len(stale) > 40 else ""))
