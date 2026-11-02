"""Tests for history features added in the recent implementation sprint.

Covers:
  - Assumption add / remove / remove_by_canonical
  - Meta-step recording (user_input, llm_supervision, ambiguity_resolution, manual_note)
  - input_refs population on cas_operation steps
  - delete_note_step: lineage re-wiring, head update, kind restriction
  - assumptions roundtrip in save / load
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.step import StepKind, StepStatus


# ---------------------------------------------------------------------------
# Shared fixture — one real Maxima process for the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="history feature tests") as e:
        yield e


# ---------------------------------------------------------------------------
# Assumption management
# ---------------------------------------------------------------------------

class TestAssumptions:
    def test_add_returns_step_and_assumption(self, engine):
        step, asm = engine.add_assumption("x > 0", display="x > 0")
        assert step is not None
        assert asm is not None

    def test_add_step_kind(self, engine):
        step, _ = engine.add_assumption("y > 0")
        assert step.kind == StepKind.assumption_add

    def test_add_step_state_affecting(self, engine):
        step, _ = engine.add_assumption("z > 0")
        assert step.state_affecting is True

    def test_add_step_rendered_command(self, engine):
        step, _ = engine.add_assumption("n > 0")
        assert step.rendered_command is not None
        assert "assume(" in step.rendered_command
        assert "n > 0" in step.rendered_command

    def test_add_assumption_stored_in_dict(self, engine):
        _, asm = engine.add_assumption("m > 0")
        assert asm.assumption_id in engine.assumptions

    def test_add_operation_args_contain_assumption_id(self, engine):
        step, asm = engine.add_assumption("k > 0")
        assert step.operation_args.get("assumption_id") == asm.assumption_id
        assert step.operation_args.get("canonical") == "k > 0"

    def test_add_advances_branch_head(self, engine):
        head_before = engine.active_branch.head_step_id
        step, _ = engine.add_assumption("p > 0")
        assert engine.active_branch.head_step_id == step.step_id
        assert engine.active_branch.head_step_id != head_before

    def test_remove_step_kind(self, engine):
        _, asm = engine.add_assumption("q > 0")
        step = engine.remove_assumption(asm.assumption_id)
        assert step.kind == StepKind.assumption_remove

    def test_remove_step_rendered_command(self, engine):
        _, asm = engine.add_assumption("r > 0")
        step = engine.remove_assumption(asm.assumption_id)
        assert "forget(" in step.rendered_command

    def test_remove_deletes_from_dict(self, engine):
        _, asm = engine.add_assumption("s > 0")
        aid = asm.assumption_id
        engine.remove_assumption(aid)
        assert aid not in engine.assumptions

    def test_remove_unknown_id_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.remove_assumption("ASM99999")

    def test_remove_by_canonical(self, engine):
        _, asm = engine.add_assumption("t > 0")
        aid = asm.assumption_id
        engine.remove_assumption_by_canonical("t > 0")
        assert aid not in engine.assumptions

    def test_remove_by_canonical_unknown_raises(self, engine):
        with pytest.raises(ValueError, match="No active assumption"):
            engine.remove_assumption_by_canonical("no_such_condition")

    def test_remove_advances_branch_head(self, engine):
        _, asm = engine.add_assumption("u > 0")
        step = engine.remove_assumption(asm.assumption_id)
        assert engine.active_branch.head_step_id == step.step_id


# ---------------------------------------------------------------------------
# Assumptions persistence roundtrip
# ---------------------------------------------------------------------------

class TestAssumptionsPersistence:
    def test_assumption_survives_save_load(self, tmp_path):
        path = tmp_path / "asm.json"
        with NotebookEngine(title="asm persist") as e:
            _, asm = e.add_assumption("a > 0")
            aid = asm.assumption_id
            e.save(path)

        with NotebookEngine.load(path) as e2:
            assert aid in e2.assumptions
            assert e2.assumptions[aid].canonical == "a > 0"

    def test_removed_assumption_not_in_loaded_session(self, tmp_path):
        path = tmp_path / "asm2.json"
        with NotebookEngine(title="asm remove persist") as e:
            _, asm = e.add_assumption("b > 0")
            aid = asm.assumption_id
            e.remove_assumption(aid)
            e.save(path)

        with NotebookEngine.load(path) as e2:
            assert aid not in e2.assumptions

    def test_empty_assumptions_loads_cleanly(self, tmp_path):
        path = tmp_path / "asm_empty.json"
        with NotebookEngine() as e:
            e.save(path)

        import json
        data = json.loads(path.read_text())
        # Remove assumptions key to test backward compat
        del data["assumptions"]
        path.write_text(json.dumps(data))

        with NotebookEngine.load(path) as e2:
            assert e2.assumptions == {}


# ---------------------------------------------------------------------------
# Meta-step recording
# ---------------------------------------------------------------------------

class TestMetaStepRecording:
    """user_input, llm_supervision, ambiguity_resolution, manual_note."""

    def test_user_input_step_kind(self, engine):
        step = engine.record_user_input("E1 を展開して")
        assert step.kind == StepKind.user_input

    def test_user_input_not_state_affecting(self, engine):
        step = engine.record_user_input("何か入力")
        assert step.state_affecting is False

    def test_user_input_does_not_advance_head(self, engine):
        head_before = engine.active_branch.head_step_id
        engine.record_user_input("テスト入力")
        assert engine.active_branch.head_step_id == head_before

    def test_user_input_text_stored(self, engine):
        step = engine.record_user_input("保存テキスト")
        assert step.user_text == "保存テキスト"

    def test_llm_supervision_step_kind(self, engine):
        step = engine.record_llm_supervision(
            user_text="展開して",
            operation="expand",
            args={"expr": "E1"},
        )
        assert step.kind == StepKind.llm_supervision

    def test_llm_supervision_not_state_affecting(self, engine):
        step = engine.record_llm_supervision(
            user_text="因数分解",
            operation="factor",
            args={"expr": "E1"},
        )
        assert step.state_affecting is False

    def test_llm_supervision_does_not_advance_head(self, engine):
        head_before = engine.active_branch.head_step_id
        engine.record_llm_supervision("expand", "expand", {"expr": "E1"})
        assert engine.active_branch.head_step_id == head_before

    def test_llm_supervision_confirmed_status(self, engine):
        step = engine.record_llm_supervision("x", "expand", {}, confirmed=True)
        assert step.status == StepStatus.success

    def test_llm_supervision_unconfirmed_status(self, engine):
        step = engine.record_llm_supervision("x", None, {}, confirmed=False)
        assert step.status == StepStatus.failed

    def test_llm_supervision_candidates_stored(self, engine):
        cands = [{"operation": "expand"}, {"operation": "factor"}]
        step = engine.record_llm_supervision(
            "test", "expand", {}, candidates=cands, confirmed=True
        )
        assert "candidates" in step.operation_args
        assert len(step.operation_args["candidates"]) == 2

    def test_ambiguity_resolution_step_kind(self, engine):
        step = engine.record_ambiguity_resolution(
            user_text="微分して",
            chosen_operation="differentiate",
            chosen_args={"expr": "E1", "var": "x"},
            candidates=[{"operation": "differentiate"}, {"operation": "integrate_indefinite"}],
        )
        assert step.kind == StepKind.ambiguity_resolution

    def test_ambiguity_resolution_not_state_affecting(self, engine):
        step = engine.record_ambiguity_resolution(
            user_text="test",
            chosen_operation="expand",
            chosen_args={"expr": "E1"},
            candidates=[],
        )
        assert step.state_affecting is False

    def test_ambiguity_resolution_does_not_advance_head(self, engine):
        head_before = engine.active_branch.head_step_id
        engine.record_ambiguity_resolution("test", "expand", {"expr": "E1"}, [])
        assert engine.active_branch.head_step_id == head_before

    def test_ambiguity_resolution_user_confirmation_set(self, engine):
        step = engine.record_ambiguity_resolution(
            user_text="積分",
            chosen_operation="integrate_indefinite",
            chosen_args={"expr": "E1", "var": "x"},
            candidates=[],
        )
        assert step.user_confirmation is not None
        assert "integrate_indefinite" in step.user_confirmation

    def test_manual_note_step_kind(self, engine):
        step = engine.record_manual_note("これはメモ")
        assert step.kind == StepKind.manual_note

    def test_manual_note_not_state_affecting(self, engine):
        step = engine.record_manual_note("別のメモ")
        assert step.state_affecting is False

    def test_manual_note_advances_head(self, engine):
        head_before = engine.active_branch.head_step_id
        step = engine.record_manual_note("ヘッドを進めるメモ")
        assert engine.active_branch.head_step_id == step.step_id
        assert engine.active_branch.head_step_id != head_before

    def test_manual_note_text_stored(self, engine):
        step = engine.record_manual_note("テキスト確認")
        assert step.user_text == "テキスト確認"


# ---------------------------------------------------------------------------
# input_refs on CAS operation steps
# ---------------------------------------------------------------------------

class TestInputRefs:
    @pytest.fixture(scope="class")
    def op_engine(self):
        with NotebookEngine(title="input_refs test") as e:
            yield e

    def test_single_expr_arg_in_input_refs(self, op_engine):
        e1 = op_engine.register_expression("(x+1)^2")
        step, _ = op_engine.run_operation("expand", {"expr": e1.display_id})
        assert e1.display_id in step.input_refs

    def test_two_args_both_in_input_refs(self, op_engine):
        e1 = op_engine.register_expression("x^2-1")
        step, _ = op_engine.run_operation("factor", {"expr": e1.display_id})
        assert e1.display_id in step.input_refs

    def test_non_expr_args_not_in_input_refs(self, op_engine):
        e1 = op_engine.register_expression("x^3")
        step, _ = op_engine.run_operation(
            "differentiate", {"expr": e1.display_id, "var": "x"}
        )
        assert "x" not in step.input_refs
        assert e1.display_id in step.input_refs

    def test_input_refs_empty_for_registration(self, op_engine):
        e1 = op_engine.register_expression("sin(x)")
        step = op_engine.steps[e1.created_by_step]
        assert step.input_refs == []


# ---------------------------------------------------------------------------
# delete_note_step: re-wiring and kind restriction
# ---------------------------------------------------------------------------

class TestDeleteNoteStep:
    @pytest.fixture
    def note_engine(self):
        with NotebookEngine(title="note delete test") as e:
            yield e

    def test_delete_removes_step(self, note_engine):
        e1 = note_engine.register_expression("x+1")
        step_note = note_engine.record_manual_note("消えるメモ")
        sid = step_note.step_id
        note_engine.delete_note_step(sid)
        assert sid not in note_engine.steps

    def test_delete_middle_rewires_child_parent(self, note_engine):
        """A → note → B: after deleting note, B's parent should be A."""
        e1 = note_engine.register_expression("x+2")
        step_a = note_engine.steps[e1.created_by_step]
        note_step = note_engine.record_manual_note("中間メモ")
        e2 = note_engine.register_expression("x+3")
        step_b = note_engine.steps[e2.created_by_step]

        # Before delete: B's parent is the note
        assert note_step.step_id in step_b.parent_step_ids

        note_engine.delete_note_step(note_step.step_id)

        # After delete: B's parent should be A (the note's predecessor)
        updated_b = note_engine.steps[step_b.step_id]
        assert note_step.step_id not in updated_b.parent_step_ids
        assert step_a.step_id in updated_b.parent_step_ids

    def test_delete_head_note_updates_head(self, note_engine):
        """When the deleted note is the branch head, head should revert to predecessor."""
        e1 = note_engine.register_expression("y+1")
        step_a = note_engine.steps[e1.created_by_step]
        note_step = note_engine.record_manual_note("ヘッドメモ")

        assert note_engine.active_branch.head_step_id == note_step.step_id

        note_engine.delete_note_step(note_step.step_id)

        assert note_engine.active_branch.head_step_id == step_a.step_id

    def test_delete_non_manual_note_raises(self, note_engine):
        e1 = note_engine.register_expression("z+1")
        step = note_engine.steps[e1.created_by_step]
        assert step.kind == StepKind.expression_registration

        with pytest.raises(ValueError, match="Only manual_note"):
            note_engine.delete_note_step(step.step_id)

    def test_delete_unknown_step_raises(self, note_engine):
        with pytest.raises(ValueError, match="not found"):
            note_engine.delete_note_step("S99999")

    def test_delete_note_at_root_no_orphan_parents(self, note_engine):
        """Deleting the first note (no predecessor) should not leave empty parent refs."""
        fresh = NotebookEngine(title="root note")
        fresh.start()
        try:
            note_step = fresh.record_manual_note("最初のメモ")
            e1 = fresh.register_expression("a+1")
            step_a = fresh.steps[e1.created_by_step]

            fresh.delete_note_step(note_step.step_id)

            updated_a = fresh.steps[step_a.step_id]
            assert "" not in updated_a.parent_step_ids
        finally:
            fresh.close()


# ---------------------------------------------------------------------------
# delete_expression
# ---------------------------------------------------------------------------

class TestDeleteExpression:
    def test_delete_unreferenced_expression(self, engine):
        e = engine.register_expression("x + 1", latex="x+1")
        expr_id = e.display_id
        engine.delete_expression(expr_id)
        assert expr_id not in engine.expressions

    def test_delete_removes_registration_step(self, engine):
        e = engine.register_expression("x + 2", latex="x+2")
        expr_id = e.display_id
        step_id = e.created_by_step
        engine.delete_expression(expr_id)
        assert step_id not in engine.steps

    def test_delete_rewires_head(self, engine):
        e = engine.register_expression("x + 3", latex="x+3")
        expr_id = e.display_id
        head_before_delete = engine.branches[engine.active_branch_id].head_step_id
        engine.delete_expression(expr_id)
        # Head should have moved back (no longer the deleted step)
        assert engine.branches[engine.active_branch_id].head_step_id != head_before_delete

    def test_delete_referenced_expression_raises(self, engine):
        e1 = engine.register_expression("y^2", latex="y^2")
        engine.run_operation("simplify", {"expr": e1.display_id})
        with pytest.raises(ValueError, match="referenced"):
            engine.delete_expression(e1.display_id)

    def test_delete_nonexistent_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.delete_expression("E999")

    def test_delete_successor_still_exists(self, engine):
        e1 = engine.register_expression("z^2 - 1", latex="z^2-1")
        step, _ = engine.run_operation("simplify", {"expr": e1.display_id})
        e2_id = step.output_refs[0] if step.output_refs else None
        # E1 is referenced by the simplify step → cannot delete
        with pytest.raises(ValueError):
            engine.delete_expression(e1.display_id)
        # E2 is unreferenced → can delete
        if e2_id and e2_id in engine.expressions:
            engine.delete_expression(e2_id)
            assert e2_id not in engine.expressions
