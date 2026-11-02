"""D-5 (open_items.md): ui/timeline_serializer.py must surface a
`plan_execution` timeline entry for BOTH a completed and a stopped plan.

Before this fix, StepKind.plan_execution had no branch in
build_timeline_entries()'s elif chain, so the step (which DOES advance the
branch head, per engine_plan_mixin.py::_record_plan_summary) reached the loop
and was silently dropped — the live confirm-click render worked (it renders
the raw HTTP response, never touching this serializer), but a page reload
lost all trace that the plan had ever run."""
from notebook.engine import NotebookEngine
from notebook.models.plan import PlanStep, SupervisorPlan
from ui.timeline_serializer import build_timeline_entries


class TestPlanExecutionTimelineEntry:
    def test_completed_plan_entry_survives_reload_shape(self):
        with NotebookEngine(title="timeline plan completed") as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            plan = e.create_workflow_plan("eigensystem", ex.display_id)
            done = e.execute_plan(plan.plan_id)
            assert done.status == "completed"

            entries = build_timeline_entries(e)
            plan_entries = [en for en in entries if en["type"] == "plan_execution"]
            assert len(plan_entries) == 1
            entry = plan_entries[0]
            assert entry["status"] == "completed"
            assert entry["plan_id"] == plan.plan_id
            assert entry["message"]
            assert len(entry["plan_step_results"]) == 3
            assert all(r["status"] == "success" for r in entry["plan_step_results"])

    def test_stopped_plan_entry_reports_failed_status_and_partial_results(self):
        with NotebookEngine(title="timeline plan stopped") as e:
            ex = e.register_expression("(x+1)^2")
            plan = SupervisorPlan(
                plan_id="P1", goal_text="stop test",
                target_expr_id=ex.display_id,
                steps=[
                    PlanStep(step_id="s1", operation="expand",
                             args={"expr": "$target"}),
                    PlanStep(step_id="s2", operation="differentiate",
                             args={"expr": "E999", "var": "x"}),
                    PlanStep(step_id="s3", operation="factor",
                             args={"expr": "$prev"}),
                ])
            e.plans["P1"] = plan
            e._plan_counter = 1
            done = e.execute_plan("P1")
            assert done.status == "stopped"

            entries = build_timeline_entries(e)
            plan_entries = [en for en in entries if en["type"] == "plan_execution"]
            assert len(plan_entries) == 1
            entry = plan_entries[0]
            assert entry["status"] == "failed"
            assert entry["plan_id"] == "P1"
            results = entry["plan_step_results"]
            assert [r["status"] for r in results] == ["success", "failed", "skipped"]
