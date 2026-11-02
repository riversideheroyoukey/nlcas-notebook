"""NL-GRAMMAR ①: the two inline-formula halves register through ONE function.

`_auto_register_inline_formula` answers "is the formula at the head of this
instruction an expression to register?" twice — once for Japanese, where the
operation's name follows the formula and the boundary is the script transition
(`_FORMULA_OP_RE`), and once for English, where the name PRECEDES it and the
boundary is the operation's own alias (`_en_verb_formula_re`).  Those two
readings are legitimately different.

What was written twice and should not have been is the half AFTER the reading:
expand the E-ids, convert, find-or-register, attribute the step, report it.  The
two copies had already drifted — the English one dropped `warnings`, so a
bound-name collision on a formula typed in English was recorded on the
Expression and never reached the screen (`feedback_warn_when_you_cannot_decide`:
a warning nobody renders is not a warning).

Extracted 2026-08-23 into `_register_leading_formula`; the corpus (523 rows)
showed zero behavioural difference.  These tests pin the invariant that made the
drift possible.
"""
import asyncio
import inspect

import pytest

from notebook.engine import NotebookEngine
from ui.interpret_pipeline import _register_leading_formula


_REPORTED_KEYS = {"expr_id", "canonical", "latex", "step_id",
                  "undefined_symbols", "warnings"}


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="inline formula") as e:
        yield e


def _register(engine, latex):
    registered, canonical_to_id = [], {}
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(_register_leading_formula(
            engine, loop, latex, registered, canonical_to_id))
    finally:
        loop.close()
    return out, registered, canonical_to_id


class TestOneRegistrationForBothLanguages:

    def test_both_leading_formula_branches_go_through_the_helper(self):
        """Two call sites — the Japanese branch and the English one.

        This is the check that would have failed while the two copies existed,
        and the one that fails again if somebody re-inlines either.

        Two more branches register in the same function — H-1's named equation
        and the E-id-expansion branch — and they are deliberately NOT folded
        into this helper: each computes its canonical differently (the leading
        formula converts the researcher's LaTeX, H-1 may CAS-evaluate an
        expanded right-hand side, the E-id branch splices without evaluating).
        What all three DO share is what happens once a canonical exists, and
        that went to `_register_and_report` — so no branch calls
        `engine.register_expression` itself any more.
        """
        import ui.interpret_pipeline as ip
        src = inspect.getsource(ip._auto_register_inline_formula)
        assert src.count("_register_leading_formula") == 2, (
            "both the Japanese and the English branch must go through the "
            f"helper — found {src.count('_register_leading_formula')} call sites")
        assert "engine.register_expression" not in src, (
            "every branch must register through `_register_and_report`; a "
            "direct call is how `warnings` and the route attribution went "
            "missing from some branches and not others")
        assert "_attribute_new_steps" not in src, (
            "attribution belongs to `_register_and_report`, so it cannot be "
            "written into two branches out of three")

    def test_the_reported_entry_carries_every_field(self, engine):
        """Including `warnings` — the field the English copy had dropped."""
        (expr_id, canonical), registered, _ = _register(engine, "a x + b")
        assert expr_id is not None
        assert len(registered) == 1
        assert set(registered[0]) == _REPORTED_KEYS, set(registered[0])

    def test_an_identical_formula_is_reused_not_re_registered(self, engine):
        """An id comes back both ways, because the caller's next move is the
        same — only a FAILURE is None."""
        (first, canonical), reg1, map1 = _register(engine, "u p + v q")
        (second, canonical2), reg2, map2 = _register(engine, "u p + v q")
        assert first == second
        assert canonical == canonical2
        assert reg2 == [], "the second call must not report a new expression"
        assert map2[canonical2] == first

    def test_the_step_is_attributed_to_inline_formula(self, engine):
        """Un-attributed, a rule that starts claiming a line it used to decline
        is invisible to the instruction corpus (NL-GRAMMAR ②)."""
        before = set(engine.steps)
        (expr_id, _), _, _ = _register(engine, "k_1 y + k_2 z")
        new = [engine.steps[s] for s in engine.steps if s not in before]
        assert new, "registration recorded no step"
        assert any(getattr(s, "route", None) == "inline_formula" for s in new), (
            [getattr(s, "route", None) for s in new])

    def test_the_route_name_carries_no_language_suffix(self, engine):
        """`_ja`/`_en` would duplicate what the row's language already says, and
        made the two sides disagree by construction — three false divergences
        when it was tried (measured 2026-08-23)."""
        import ui.interpret_pipeline as ip
        src = inspect.getsource(ip)
        assert "inline_formula_ja" not in src
        assert "inline_formula_en" not in src
