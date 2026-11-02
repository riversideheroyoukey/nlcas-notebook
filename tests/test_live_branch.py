"""Live branch support: fork_branch / switch_branch (engine_history_mixin.py).

Closes the gap found while writing an APIEMS2026 OR case study: the only
existing "fork" (/session/fork) is a file-snapshot save whose new branch
always resets to branch_id="main", so numeric_branch_comparison() (which
groups NumericTasks by source_branch_id) could never have two branches to
compare. fork_branch() creates a SECOND branch that coexists with "main" in
the same loaded engine; switch_branch() moves the active cursor between them.
"""
import pytest

from notebook.engine import NotebookEngine


class TestForkBranch:
    def test_fork_creates_new_branch_and_switches(self):
        e = NotebookEngine(title="live branch")
        e1 = e.register_expression("a*b")
        branch = e.fork_branch(e1.created_by_step, label="approach_B")
        assert branch.branch_id != "main"
        assert branch.head_step_id == e1.created_by_step
        assert branch.forked_from_step_id == e1.created_by_step
        assert e.active_branch_id == branch.branch_id

    def test_main_branch_left_unchanged(self):
        e = NotebookEngine(title="live branch")
        e1 = e.register_expression("a*b")
        main_head_before = e.branches["main"].head_step_id
        e.fork_branch(e1.created_by_step, label="approach_B")
        assert e.branches["main"].head_step_id == main_head_before

    def test_fork_from_unknown_step_raises(self):
        e = NotebookEngine(title="live branch")
        with pytest.raises(ValueError):
            e.fork_branch("S999")


class TestSwitchBranch:
    def test_switch_to_unknown_branch_raises(self):
        e = NotebookEngine(title="live branch")
        with pytest.raises(ValueError):
            e.switch_branch("nope")

    def test_switch_back_to_main(self):
        e = NotebookEngine(title="live branch")
        e1 = e.register_expression("a*b")
        branch = e.fork_branch(e1.created_by_step)
        assert e.active_branch_id == branch.branch_id
        e.switch_branch("main")
        assert e.active_branch_id == "main"


class TestExpressionDivergence:
    def test_expressions_diverge_across_branches(self):
        e = NotebookEngine(title="live branch diverge")
        e1 = e.register_expression("a*b")
        branch = e.fork_branch(e1.created_by_step, label="B")

        e2b = e.register_expression("a+b")  # registered on the new branch
        e.switch_branch("main")
        e2m = e.register_expression("a-b")  # registered back on main

        active_on_main = e._active_expr_ids()
        assert e1.display_id in active_on_main
        assert e2m.display_id in active_on_main
        assert e2b.display_id not in active_on_main

        e.switch_branch(branch.branch_id)
        active_on_branch = e._active_expr_ids()
        assert e1.display_id in active_on_branch
        assert e2b.display_id in active_on_branch
        assert e2m.display_id not in active_on_branch


class TestPersistence:
    def test_save_load_round_trip_preserves_branches(self, tmp_path):
        e = NotebookEngine(title="live branch persist")
        e1 = e.register_expression("a*b")
        branch = e.fork_branch(e1.created_by_step, label="approach_B")
        e.register_expression("a+b")

        path = tmp_path / "session.json"
        e.save(str(path))

        loaded = NotebookEngine.load(str(path))
        assert set(loaded.branches.keys()) == {"main", branch.branch_id}
        assert loaded.active_branch_id == branch.branch_id
        assert loaded._branch_counter == e._branch_counter


class TestNumericBranchComparison:
    def test_two_branch_numeric_forks_are_comparable(self):
        e = NotebookEngine(title="numeric branch cmp")
        ex = e.register_expression("a*b")
        e.fork_numeric_task(ex.display_id, problem_type="function_evaluation",
                             target_quantity="P",
                             fixed_values={"a": "1", "b": "2"})

        branch = e.fork_branch(ex.created_by_step, label="alt")
        ex2 = e.register_expression("a*b")
        e.fork_numeric_task(ex2.display_id, problem_type="function_evaluation",
                             target_quantity="P",
                             fixed_values={"a": "3", "b": "4"})

        cmp = e.numeric_branch_comparison()
        comparable = [g for g in cmp["groups"] if g["comparable"]]
        assert len(comparable) == 1
        assert set(comparable[0]["branches"].keys()) == {"main", branch.branch_id}
