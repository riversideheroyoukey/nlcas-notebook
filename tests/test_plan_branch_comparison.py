"""C-Ext-5 B-4 (read-only): plan_branch_comparison() groups already-recorded
plan_execution steps by (workflow/goal label, target expr) and branch.

Deliberately does NOT re-execute a plan on another branch — every
plan_execution step is already branch-tagged for free (branch_id=
active_branch_id at execution time, like any other Step), so this is a pure
read/group method.  Mirrors numeric_branch_comparison() exactly
(tests/test_live_branch.py::TestNumericBranchComparison).

Plan execution itself may fail (no live Maxima adapter here) — irrelevant to
this method, which groups plan_execution steps regardless of whether the
plan they summarize succeeded or stopped."""
from notebook.engine import NotebookEngine


class TestPlanBranchComparison:
    def test_two_branches_running_same_workflow_are_comparable(self):
        e = NotebookEngine(title="plan branch cmp")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = e.create_workflow_plan("eigensystem", ex.display_id)
        e.execute_plan(plan.plan_id)

        branch = e.fork_branch(ex.created_by_step, label="alt")
        e.switch_branch(branch.branch_id)
        plan2 = e.create_workflow_plan("eigensystem", ex.display_id)
        e.execute_plan(plan2.plan_id)

        cmp = e.plan_branch_comparison()
        comparable = [g for g in cmp["groups"] if g["comparable"]]
        assert len(comparable) == 1
        assert comparable[0]["label"] == "eigensystem"
        assert comparable[0]["target"] == ex.display_id
        assert set(comparable[0]["branches"].keys()) == {"main", branch.branch_id}
        assert set(cmp["branch_ids"]) == {"main", branch.branch_id}

    def test_branch_unique_plan_is_not_comparable(self):
        e = NotebookEngine(title="plan branch unique")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = e.create_workflow_plan("eigensystem", ex.display_id)
        e.execute_plan(plan.plan_id)

        e.fork_branch(ex.created_by_step, label="alt")   # never runs anything

        cmp = e.plan_branch_comparison()
        assert len(cmp["groups"]) == 1
        assert cmp["groups"][0]["comparable"] is False
        assert set(cmp["groups"][0]["branches"].keys()) == {"main"}

    def test_different_target_expr_not_grouped_together(self):
        e = NotebookEngine(title="plan branch diff target")
        ex1 = e.register_expression("[2, 1; 1, 2]")
        plan1 = e.create_workflow_plan("eigensystem", ex1.display_id)
        e.execute_plan(plan1.plan_id)

        branch = e.fork_branch(ex1.created_by_step, label="alt")
        e.switch_branch(branch.branch_id)
        ex2 = e.register_expression("[3, 0; 0, 3]")
        plan2 = e.create_workflow_plan("eigensystem", ex2.display_id)
        e.execute_plan(plan2.plan_id)

        cmp = e.plan_branch_comparison()
        assert len(cmp["groups"]) == 2
        assert all(not g["comparable"] for g in cmp["groups"])
        targets = {g["target"] for g in cmp["groups"]}
        assert targets == {ex1.display_id, ex2.display_id}

    def test_empty_engine_shows_nothing(self):
        e = NotebookEngine(title="plan branch empty")
        cmp = e.plan_branch_comparison()
        assert cmp == {"groups": [], "branch_ids": []}

    def test_single_branch_single_plan_not_comparable(self):
        e = NotebookEngine(title="plan branch single")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = e.create_workflow_plan("eigensystem", ex.display_id)
        e.execute_plan(plan.plan_id)

        cmp = e.plan_branch_comparison()
        assert len(cmp["groups"]) == 1
        assert cmp["groups"][0]["comparable"] is False
        assert cmp["branch_ids"] == ["main"]
