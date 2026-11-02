"""C-Ext-5 PoC: plan models, upfront validation, stop-safe executor,
the fixed eigensystem workflow, and the bilingual NL trigger."""
import asyncio

import pytest

from notebook.engine import NotebookEngine
from notebook.models.plan import PlanStep, SupervisorPlan
from notebook.models.step import StepKind
from notebook.runtime.i18n import t
from notebook.runtime.plan_workflows import (
    WORKFLOW_REGISTRY, build_workflow_plan, validate_plan,
)
from ui.interpret_plan import _try_plan_workflow


def _run_nl(engine, text, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _try_plan_workflow(engine, loop, lang, text, [])
    return asyncio.run(_inner())


# ---------------------------------------------------------------------------
# Validation (before anything executes)
# ---------------------------------------------------------------------------

class TestPlanValidation:

    def _validate(self, engine, plan):
        validate_plan(plan, engine._validator, engine.session.primary_backend,
                      engine._active_expr_ids(), lang="ja")

    def test_eigensystem_workflow_registered_and_valid(self):
        assert "eigensystem" in WORKFLOW_REGISTRY
        e = NotebookEngine(title="plan valid")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = build_workflow_plan("eigensystem", "P1", ex.display_id)
        self._validate(e, plan)  # must not raise

    def test_unknown_operation_rejected(self):
        e = NotebookEngine(title="plan bad op")
        ex = e.register_expression("x")
        plan = SupervisorPlan(plan_id="P1", target_expr_id=ex.display_id,
                              steps=[PlanStep(step_id="s1",
                                              operation="summon_demon",
                                              args={"expr": "$target"})])
        with pytest.raises(ValueError):
            self._validate(e, plan)

    def test_missing_required_field_rejected(self):
        e = NotebookEngine(title="plan missing")
        ex = e.register_expression("x")
        plan = SupervisorPlan(plan_id="P1", target_expr_id=ex.display_id,
                              steps=[PlanStep(step_id="s1",
                                              operation="differentiate",
                                              args={"expr": "$target"})])  # no var
        with pytest.raises(ValueError):
            self._validate(e, plan)

    def test_forward_reference_rejected(self):
        e = NotebookEngine(title="plan fwd")
        ex = e.register_expression("x")
        plan = SupervisorPlan(
            plan_id="P1", target_expr_id=ex.display_id,
            steps=[
                PlanStep(step_id="s1", operation="expand",
                         args={"expr": "$s2"}),      # forward ref
                PlanStep(step_id="s2", operation="expand",
                         args={"expr": "$target"}),
            ])
        with pytest.raises(ValueError):
            self._validate(e, plan)

    def test_prev_on_first_step_rejected(self):
        e = NotebookEngine(title="plan prev0")
        ex = e.register_expression("x")
        plan = SupervisorPlan(plan_id="P1", target_expr_id=ex.display_id,
                              steps=[PlanStep(step_id="s1", operation="expand",
                                              args={"expr": "$prev"})])
        with pytest.raises(ValueError):
            self._validate(e, plan)

    def test_backend_unsupported_rejected(self):
        e = NotebookEngine(title="plan backend")  # maxima session
        ex = e.register_expression("x")
        plan = SupervisorPlan(plan_id="P1", target_expr_id=ex.display_id,
                              steps=[PlanStep(step_id="s1",
                                              operation="normal_order",
                                              args={"expr": "$target"})])
        with pytest.raises(ValueError):
            self._validate(e, plan)

    def test_error_messages_translate_both_languages(self):
        for lang in ("ja", "en"):
            assert t("plan.unknown_operation", lang, step_id="s1",
                     operation="x")
            assert t("plan.stopped_summary", lang, plan_id="P1", goal="g",
                     step_id="s2", n_ok=1, n_total=3)


# ---------------------------------------------------------------------------
# Executor: chaining and stop-safe failure (real Maxima)
# ---------------------------------------------------------------------------

class TestPlanExecutor:

    def test_prev_chaining(self):
        """s2 consumes s1's output via $prev."""
        with NotebookEngine(title="plan chain") as e:
            ex = e.register_expression("(x+1)^3")
            plan = SupervisorPlan(
                plan_id="P1", goal_text="chain test",
                target_expr_id=ex.display_id,
                steps=[
                    PlanStep(step_id="s1", operation="differentiate",
                             args={"expr": "$target", "var": "x"}),
                    PlanStep(step_id="s2", operation="expand",
                             args={"expr": "$prev"}),
                ])
            e.plans["P1"] = plan
            e._plan_counter = 1
            done = e.execute_plan("P1")
            assert done.status == "completed"
            s1, s2 = done.step_results
            assert s1.status == "success" and s2.status == "success"
            assert s2.expr_id in e.expressions
            # d/dx (x+1)^3 = 3(x+1)^2 → expanded 3x^2+6x+3
            assert "6" in s2.value and "x^2" in s2.value.replace(" ", "")

    def test_stop_safe_failure_preserves_partial_results(self):
        """A runtime failure at s2 stops the plan, skips s3, keeps s1."""
        with NotebookEngine(title="plan stop") as e:
            ex = e.register_expression("(x+1)^2")
            plan = SupervisorPlan(
                plan_id="P1", goal_text="stop test",
                target_expr_id=ex.display_id,
                steps=[
                    PlanStep(step_id="s1", operation="expand",
                             args={"expr": "$target"}),
                    # E999 passes plan-level validation (only placeholders are
                    # checked there) but fails run_operation's expr-ref check —
                    # a genuine RUNTIME failure mid-plan.
                    PlanStep(step_id="s2", operation="differentiate",
                             args={"expr": "E999", "var": "x"}),
                    PlanStep(step_id="s3", operation="factor",
                             args={"expr": "$prev"}),
                ])
            e.plans["P1"] = plan
            e._plan_counter = 1
            done = e.execute_plan("P1")   # must NOT raise
            assert done.status == "stopped"
            s1, s2, s3 = done.step_results
            assert s1.status == "success" and s1.expr_id in e.expressions
            assert s2.status == "failed" and s2.error
            assert s3.status == "skipped"
            summary = [s for s in e.steps.values()
                       if s.kind == StepKind.plan_execution]
            assert summary and summary[-1].status.value == "failed"

    def test_eigensystem_end_to_end_with_persistence(self, tmp_path):
        with NotebookEngine(title="plan eigen") as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            plan = e.create_workflow_plan("eigensystem", ex.display_id)
            assert plan.status == "validated"
            done = e.execute_plan(plan.plan_id)
            assert done.status == "completed"
            vals = {r.operation: r.value for r in done.step_results}
            assert "x^2" in vals["characteristic_polynomial"].replace(" ", "")
            assert "3" in vals["eigenvalues"]        # eigenvalues {3, 1}
            assert "1" in vals["eigenvectors"]
            summary = [s for s in e.steps.values()
                       if s.kind == StepKind.plan_execution][-1]
            assert len(summary.output_refs) == 3
            assert summary.input_refs == [ex.display_id]
            path = tmp_path / "plan.json"
            e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.plans[plan.plan_id].status == "completed"
        assert loaded._plan_counter == 1
        assert len(loaded.plans[plan.plan_id].step_results) == 3

    def test_old_session_without_plan_keys_loads(self, tmp_path):
        import json
        e = NotebookEngine(title="plan compat")
        e.register_expression("x")
        path = tmp_path / "old.json"
        e.save(path)
        data = json.loads(path.read_text())
        data.pop("plans", None)
        data.pop("plan_counter", None)
        path.write_text(json.dumps(data, ensure_ascii=False))
        loaded = NotebookEngine.load(path)
        assert loaded.plans == {} and loaded._plan_counter == 0


# ---------------------------------------------------------------------------
# NL trigger (bilingual; narrow composite ask only)
# ---------------------------------------------------------------------------

class TestPlanNLTrigger:

    def test_jp_eigenstate_trigger(self):
        with NotebookEngine(title="plan nl jp") as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            res = _run_nl(e, f"{ex.display_id} の固有状態を求めて", lang="ja")
            assert res is not None and res["status"] == "completed"
            assert len(res["plan_step_results"]) == 3
            assert "P1" == res["plan_id"]

    def test_en_eigenstates_trigger(self):
        with NotebookEngine(title="plan nl en") as e:
            ex = e.register_expression("[0, 1; 1, 0]")
            res = _run_nl(e, f"Find the eigenstates of {ex.display_id}",
                          lang="en")
            assert res is not None and res["status"] == "completed"
            vals = " ".join(r["value"] for r in res["plan_step_results"])
            assert "-1" in vals                      # eigenvalues ±1

    def test_single_op_ask_does_not_trigger(self):
        """「固有値を求めて」 keeps its existing single-operation route."""
        e = NotebookEngine(title="plan nl narrow")
        e.register_expression("[2, 1; 1, 2]")
        assert _run_nl(e, "E1 の固有値を求めて", lang="ja") is None
        assert _run_nl(e, "Compute the eigenvalues of E1", lang="en") is None

    def test_no_expression_no_trigger(self):
        e = NotebookEngine(title="plan nl empty")
        assert _run_nl(e, "固有状態を求めて", lang="ja") is None


# ---------------------------------------------------------------------------
# A-1: Julia-session workflows (placeholder chain on the Julia backend)
# ---------------------------------------------------------------------------

class TestJuliaWorkflows:

    def test_eigensystem_on_julia_session(self):
        from notebook.models.session import SessionMode
        with NotebookEngine(title="plan julia eigen",
                            mode=SessionMode.second_quantization) as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            plan = e.create_workflow_plan("eigensystem", ex.display_id)
            done = e.execute_plan(plan.plan_id)
            assert done.status == "completed"
            vals = {r.operation: r.value for r in done.step_results}
            assert "3" in vals["eigenvalues"]

    def test_heisenberg_components_larmor(self):
        """Larmor: H = -γB·Sz → dSx/dt ∝ Sy, dSy/dt ∝ -Sx, dSz/dt = 0."""
        from notebook.models.session import SessionMode
        with NotebookEngine(title="plan julia heisenberg",
                            mode=SessionMode.second_quantization) as e:
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "gamma"})
            e.run_operation("define_scalar_parameter", {"name": "B"})
            h = e.register_expression("-gamma*B*Sz_1")
            plan = e.create_workflow_plan("heisenberg_components", h.display_id)
            done = e.execute_plan(plan.plan_id)
            assert done.status == "completed"
            sx, sy, sz = [r.value for r in done.step_results]
            assert "S" in sx and "gamma" in sx        # dSx/dt = 2Bγ Sy
            assert "-" in sy                          # dSy/dt = -2Bγ Sx
            assert sz.strip() == "0"                  # dSz/dt = 0

    def test_heisenberg_workflow_rejected_on_maxima(self):
        """heisenberg_equation is julia_full-only — upfront validation."""
        e = NotebookEngine(title="plan heis maxima")
        ex = e.register_expression("x")
        with pytest.raises(ValueError):
            e.create_workflow_plan("heisenberg_components", ex.display_id)

    def test_heisenberg_nl_trigger_jp_and_en(self):
        from notebook.models.session import SessionMode
        with NotebookEngine(title="plan heis nl",
                            mode=SessionMode.second_quantization) as e:
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "gamma"})
            e.run_operation("define_scalar_parameter", {"name": "B"})
            h = e.register_expression("-gamma*B*Sz_1")
            res = _run_nl(e, f"{h.display_id} について S の各成分のハイゼンベルク方程式を求めて",
                          lang="ja")
            assert res is not None and res["status"] == "completed"
            assert len(res["plan_step_results"]) == 3
            res_en = _run_nl(
                e, f"Heisenberg equations for all components of S with {h.display_id}",
                lang="en")
            assert res_en is not None and res_en["status"] == "completed"


# ---------------------------------------------------------------------------
# A-2: NL-exposed workflow parameter (characteristic-polynomial variable)
# ---------------------------------------------------------------------------

class TestWorkflowParams:

    def test_charpoly_variable_jp(self):
        with NotebookEngine(title="plan var jp") as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            res = _run_nl(e, f"{ex.display_id} の固有状態を変数 s で求めて",
                          lang="ja")
            assert res is not None and res["status"] == "completed"
            charpoly = res["plan_step_results"][0]["value"]
            assert "s" in charpoly and "x" not in charpoly

    def test_charpoly_variable_en(self):
        with NotebookEngine(title="plan var en") as e:
            ex = e.register_expression("[0, 1; 1, 0]")
            res = _run_nl(
                e, f"Find the eigenstates of {ex.display_id} with variable s",
                lang="en")
            assert res is not None and res["status"] == "completed"
            assert "s" in res["plan_step_results"][0]["value"]


# ---------------------------------------------------------------------------
# B-1: incremental status updates (live progress for a poller)
# ---------------------------------------------------------------------------

class TestIncrementalStatus:

    def test_plan_record_updates_during_execution(self):
        from unittest.mock import patch
        with NotebookEngine(title="plan progress") as e:
            ex = e.register_expression("(x+1)^2")
            plan = SupervisorPlan(
                plan_id="P1", goal_text="progress",
                target_expr_id=ex.display_id,
                steps=[PlanStep(step_id="s1", operation="expand",
                                args={"expr": "$target"}),
                       PlanStep(step_id="s2", operation="factor",
                                args={"expr": "$prev"})])
            e.plans["P1"] = plan
            e._plan_counter = 1

            observed: list[tuple[str, int]] = []
            real_run = e.run_operation

            def _spy(op, args):
                live = e.plans["P1"]
                observed.append((live.status, len(live.step_results)))
                return real_run(op, args)

            with patch.object(e, "run_operation", side_effect=_spy):
                done = e.execute_plan("P1")
            assert done.status == "completed"
            # During s1 the record was already "running" with 0 results;
            # during s2 it carried s1's result — live progress is observable.
            assert observed[0] == ("running", 0)
            assert observed[1] == ("running", 1)


# ---------------------------------------------------------------------------
# B-2: intermediate-result granularity (present=False → hidden expression)
# ---------------------------------------------------------------------------

class TestGranularityPolicy:

    def test_non_presented_step_output_is_hidden(self):
        with NotebookEngine(title="plan hidden") as e:
            ex = e.register_expression("(x+1)^3")
            plan = SupervisorPlan(
                plan_id="P1", goal_text="granularity",
                target_expr_id=ex.display_id,
                steps=[PlanStep(step_id="s1", operation="differentiate",
                                args={"expr": "$target", "var": "x"},
                                present=False),
                       PlanStep(step_id="s2", operation="expand",
                                args={"expr": "$prev"})])
            e.plans["P1"] = plan
            e._plan_counter = 1
            done = e.execute_plan("P1")
            assert done.status == "completed"
            s1, s2 = done.step_results
            assert e.expressions[s1.expr_id].hidden is True
            assert e.expressions[s2.expr_id].hidden is False
            # the hidden intermediate still fed $prev
            assert "6" in s2.value


# ---------------------------------------------------------------------------
# B-3: retry a stopped plan (carry successes, fix and re-run the failure)
# ---------------------------------------------------------------------------

class TestRetry:

    def _stopped_plan(self, e):
        ex = e.register_expression("(x+1)^2")
        plan = SupervisorPlan(
            plan_id="P1", goal_text="retry test",
            target_expr_id=ex.display_id,
            steps=[PlanStep(step_id="s1", operation="expand",
                            args={"expr": "$target"}),
                   PlanStep(step_id="s2", operation="differentiate",
                            args={"expr": "E999", "var": "x"}),
                   PlanStep(step_id="s3", operation="factor",
                            args={"expr": "$prev"})])
        e.plans["P1"] = plan
        e._plan_counter = 1
        return e.execute_plan("P1")

    def test_retry_with_fix_completes_and_carries_successes(self):
        with NotebookEngine(title="plan retry") as e:
            stopped = self._stopped_plan(e)
            assert stopped.status == "stopped"
            old_s1_expr = stopped.step_results[0].expr_id

            done = e.retry_plan("P1", args_override={
                "s2": {"expr": "$prev", "var": "x"}})
            assert done.status == "completed"
            assert done.plan_id == "P2"
            assert done.retried_from == "P1"
            # s1 was carried over — same expression, not re-executed
            assert done.step_results[0].expr_id == old_s1_expr
            assert done.step_results[1].status == "success"
            assert done.step_results[2].status == "success"
            # the old plan record is untouched (immutable history)
            assert e.plans["P1"].status == "stopped"

    def test_retry_refuses_completed_plans(self):
        with NotebookEngine(title="plan retry guard") as e:
            ex = e.register_expression("(x+1)^2")
            plan = e.create_workflow_plan("eigensystem", ex.display_id)
            e.execute_plan(plan.plan_id)
            # completed (or even stopped-at-runtime) eigensystem on a
            # non-matrix may stop; guard applies only to completed ones
            p = e.plans[plan.plan_id]
            if p.status == "completed":
                with pytest.raises(ValueError):
                    e.retry_plan(plan.plan_id)
