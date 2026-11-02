"""INTERNAL-INTERMEDIATE-EID-GAP: the researcher's E numbers have no holes.

Several operations build a temporary expression, run a CAS operation on it and
then hide it.  While those temporaries took their key from the E counter, every
one of them punched a hole in the visible numbering — `E1` followed by `E3` — and
a hole reads as "there is an E2 somewhere".  Measured on the ticket: three
misreadings in a few days, one of which wrote a substitute command against the
empty number and hit a different defect (EXPR-REF silent-wrong) on the way.

Two causes, measured separately, because they are NOT the same mechanism and the
ticket described them as one:

1. **scaffolding** (H-8 両辺操作, F-1 lattice ops) — a real expression that is
   hidden afterwards.  Fixed by keying it `_T1` at birth so it never takes a
   number.
2. **solve / build_inverse_transform** — the result slot was allocated before the
   CAS ran, the first solution row was registered under a FRESH number and then
   copied back into the slot, and the fresh key popped.  The copy left the
   counter advanced (two solutions came out E4 and E6) and left a registration
   step whose output_refs named a deleted expression.  Fixed by registering the
   first row into the pre-allocated slot instead.

The invariant these pin is the researcher-facing one: **what the timeline shows
is E1, E2, E3, … with nothing missing.**
"""
import json
import tempfile
from pathlib import Path

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.runtime.expr_ref_fields import is_expr_ref
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline
from ui.timeline_serializer import build_timeline_entries


def _run(engine, text):
    import asyncio
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(req, engine, nb._supervisor))


def _timeline_ids(engine):
    return [e["expr_id"] for e in build_timeline_entries(engine) if "expr_id" in e]


def _assert_contiguous(engine):
    """The visible numbers are E1, E2, … with no hole — the whole point."""
    ids = _timeline_ids(engine)
    assert ids == [f"E{i}" for i in range(1, len(ids) + 1)], ids


def _dead_refs(engine):
    return [(s.step_id, r) for s in engine.steps.values()
            for r in (s.output_refs or []) if r not in engine.expressions]


@pytest.fixture
def engine():
    nb.start_engine(title="eid-gap", mode="standard", language="ja")
    try:
        yield nb.get_engine()
    finally:
        nb.stop_engine()


class TestScaffoldingTakesNoNumber:
    def test_both_sides_op_numbers_are_contiguous(self, engine):
        """H-8: `E1 の両辺に 2 をかけて` used to show E1 then E3 — the temp RHS
        took E2.  Chained twice so a single off-by-one cannot pass."""
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて", "E2 の両辺を c^2 で割って"]:
            _run(engine, cmd)
        assert _timeline_ids(engine) == ["E1", "E2", "E3"]
        _assert_contiguous(engine)

    def test_the_result_is_still_correct(self, engine):
        """Closing the gap must not change the mathematics.  The intermediate
        is a real expression the CAS operates on — an id the builder fails to
        resolve is passed through as a bare symbol, which is exactly what
        happened first: the answer came back `2 E = _{T}_{1}`, the KEY rendered
        as LaTeX, with no error anywhere."""
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて"]:
            _run(engine, cmd)
        latex = engine.expressions["E2"].latex
        assert "_T" not in latex and "T_" not in latex, latex
        assert "m" in latex and "c" in latex, latex

    def test_scaffolding_is_keyed_outside_the_e_space(self, engine):
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて"]:
            _run(engine, cmd)
        internal = [k for k in engine.expressions if k.startswith("_T")]
        assert internal, list(engine.expressions)
        assert all(engine.expressions[k].internal for k in internal)

    def test_scaffolding_never_reaches_the_timeline(self, engine):
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて"]:
            _run(engine, cmd)
        assert not [i for i in _timeline_ids(engine) if i.startswith("_T")]


class TestSolveResultsAreContiguous:
    def test_inverse_transform_two_solutions(self, engine):
        """Measured before the fix: E3 and E5, with E4 nowhere."""
        for cmd in ["X = a p + b q", "Y = c p + d q",
                    "E1 と E2 を p, q について逆変換して"]:
            _run(engine, cmd)
        assert _timeline_ids(engine) == ["E1", "E2", "E3", "E4"]

    def test_no_step_points_at_a_deleted_expression(self, engine):
        """The promote-and-delete also left a registration step whose
        output_refs named the popped key.  Inert (save/load/replay all
        survived it), but it is a history record referring to nothing."""
        for cmd in ["X = a p + b q", "Y = c p + d q",
                    "E1 と E2 を p, q について逆変換して"]:
            _run(engine, cmd)
        assert _dead_refs(engine) == []

    def test_lhs_aliases_point_at_the_right_expressions(self, engine):
        """The promotion had to redirect lhs_alias_map; not allocating the
        extra id removes the need, so check the aliases really do land."""
        for cmd in ["X = a p + b q", "Y = c p + d q",
                    "E1 と E2 を p, q について逆変換して"]:
            _run(engine, cmd)
        assert engine.lhs_alias_map.get("p") == "E3"
        assert engine.lhs_alias_map.get("q") == "E4"
        assert "p" in engine.expressions["E3"].latex


class TestMixedSessionStaysContiguous:
    def test_scaffolding_and_solve_together(self, engine):
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて",
                    "X = a p + b q", "Y = c p + d q",
                    "E3 と E4 を p, q について逆変換して"]:
            _run(engine, cmd)
        _assert_contiguous(engine)
        assert _dead_refs(engine) == []


class TestPersistence:
    def test_internal_key_survives_save_and_load(self, engine):
        for cmd in ["E = m c^2", "E1 の両辺に 2 をかけて"]:
            _run(engine, cmd)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            engine.save(p)
            assert json.loads(p.read_text(encoding="utf-8"))["internal_counter"] == 1
            loaded = NotebookEngine.load(p)
        assert loaded._internal_counter == 1
        assert _timeline_ids(loaded) == ["E1", "E2"]

    def test_a_file_without_the_key_loads_as_zero(self, engine):
        """Additive key: every session saved before this change lacks it, and
        its intermediates were keyed from the E space and must stay that way."""
        _run(engine, "E = m c^2")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.json"
            engine.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            del data["internal_counter"]                  # simulate an older file
            p.write_text(json.dumps(data), encoding="utf-8")
            loaded = NotebookEngine.load(p)
        assert loaded._internal_counter == 0


class TestTheRefPredicateAnswersOneQuestion:
    """`is_expr_ref` is the RESOLUTION-side question and must not be confused
    with "did the researcher type an E number" — the NL side keeps its own
    `E\\d+` patterns so it cannot match a `_T3` nobody types.
    """

    @pytest.mark.parametrize("value,expected", [
        ("E1", True), ("E12", True), ("_T1", True), ("_T99", True),
        ("E", False), ("_T", False), ("k_B", False), ("", False),
        (None, False), (7, False), ("E1x", False),
    ])
    def test_shape(self, value, expected):
        assert is_expr_ref(value) is expected

    def test_the_nl_side_still_refuses_an_internal_key(self):
        """If the user-text patterns accepted `_T3`, scaffolding would become
        addressable by the researcher — the opposite of the point."""
        from ui.interpret_common import _EXPR_ID_RE as ui_re
        from notebook.runtime.llm_supervisor import _EXPR_ID_RE as sup_re
        for rx in (ui_re, sup_re):
            assert rx.match("E1")
            assert not rx.match("_T1")
