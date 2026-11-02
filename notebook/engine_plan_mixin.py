"""NotebookEngine mixin: multi-step plan execution (C-Ext-5 PoC).

The executor delegates every step to the EXISTING run_operation path —
plans add bookkeeping (upfront validation, placeholder chaining, stop-safe
failure handling, a summary timeline step), never new CAS authority.

Stop-safe contract: validation failures raise BEFORE anything executes;
an execution failure at step k records the error, marks the remaining
steps skipped, sets plan.status="stopped", and PRESERVES the partial
results (the researcher reviews them in the timeline).  No exception
escapes a mid-plan failure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from notebook.models.plan import (
    PLACEHOLDER_PREV, PLACEHOLDER_TARGET, PlanStepResult, SupervisorPlan,
)
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.i18n import t
from notebook.runtime.plan_workflows import build_workflow_plan, validate_plan


class EnginePlanMixin:
    """Plan creation and execution for NotebookEngine."""

    def _alloc_plan_id(self) -> str:
        self._plan_counter += 1
        return f"P{self._plan_counter}"

    # ------------------------------------------------------------------
    # Creation (fixed workflows take precedence over any LLM planner)
    # ------------------------------------------------------------------

    def create_workflow_plan(self, workflow_name: str, target_expr_id: str,
                             **kwargs) -> SupervisorPlan:
        plan = build_workflow_plan(workflow_name, self._alloc_plan_id(),
                                   target_expr_id, lang=self.lang, **kwargs)
        validate_plan(plan, self._validator, self.session.primary_backend,
                      self._active_expr_ids(), lang=self.lang)
        plan = plan.model_copy(update={"status": "validated"})
        self.plans[plan.plan_id] = plan
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_plan(self, plan_id: str,
                     _carry: "dict[str, PlanStepResult] | None" = None
                     ) -> SupervisorPlan:
        """Run a plan step by step.

        The plan record in ``self.plans`` is updated INCREMENTALLY (B-1):
        status="running" with partial step_results after every step, so a
        status poll (GET /plan/status/{id}) sees live progress.

        ``_carry`` (B-3, used by retry_plan) maps step_ids to results carried
        over from a previous stopped run — those steps are not re-executed;
        their recorded output expressions seed the placeholder chain.
        """
        plan = self.plans.get(plan_id)
        if plan is None:
            raise ValueError(t("plan.not_found", self.lang, plan_id=plan_id))
        # Re-validate against the CURRENT session state (expressions may have
        # been deleted since creation).
        validate_plan(plan, self._validator, self.session.primary_backend,
                      self._active_expr_ids(), lang=self.lang)

        carry = _carry or {}
        results: list[PlanStepResult] = []
        outputs: dict[str, str] = {}     # step_id -> output expr_id
        prev_expr: str = plan.target_expr_id
        failed_at: "str | None" = None

        def _publish(status: str) -> SupervisorPlan:
            updated = plan.model_copy(update={"status": status,
                                              "step_results": list(results)})
            self.plans[plan_id] = updated
            return updated

        _publish("running")
        for step in plan.steps:
            if failed_at is not None:
                results.append(PlanStepResult(
                    step_id=step.step_id, operation=step.operation,
                    status="skipped"))
                _publish("running")
                continue
            carried = carry.get(step.step_id)
            if carried is not None and carried.status == "success":
                # B-3: reuse the previous run's result without re-executing.
                results.append(carried)
                if carried.expr_id:
                    outputs[step.step_id] = carried.expr_id
                    prev_expr = carried.expr_id
                _publish("running")
                continue
            args = {
                k: self._resolve_plan_placeholder(v, plan, prev_expr, outputs)
                for k, v in step.args.items()
            }
            try:
                cas_step, value = self.run_operation(step.operation, args)
            except Exception as exc:
                results.append(PlanStepResult(
                    step_id=step.step_id, operation=step.operation,
                    status="failed", error=str(exc)))
                failed_at = step.step_id
                _publish("running")
                continue
            out_id = cas_step.output_refs[0] if cas_step.output_refs else ""
            outputs[step.step_id] = out_id
            if out_id:
                prev_expr = out_id
            results.append(PlanStepResult(
                step_id=step.step_id, operation=step.operation,
                status="success", expr_id=out_id, value=str(value)))
            _publish("running")

        plan = _publish("completed" if failed_at is None else "stopped")
        # B-2 granularity policy, applied AFTER the run: hiding an
        # intermediate mid-run would remove it from _active_expr_ids and
        # break the $prev chain of the very next step.  Stopped plans keep
        # everything visible on purpose — the researcher inspects the
        # partial results to decide how to retry.
        #
        # These intermediates DO still consume E numbers, unlike the ones fixed
        # under INTERNAL-INTERMEDIATE-EID-GAP, and that is deliberate rather
        # than an oversight: this is the one hiding site that cannot know up
        # front.  Whether a step's output is hidden depends on `failed_at`,
        # i.e. on whether a LATER step failed, so `run_operation(internal=True)`
        # cannot be decided when the operation runs.  Closing this one needs
        # renumbering after the fact, which is the option deliberately not
        # taken — an E number, once shown, is what the researcher writes down.
        if failed_at is None:
            for step in plan.steps:
                if step.present:
                    continue
                out_id = outputs.get(step.step_id, "")
                if out_id and out_id in self.expressions:
                    self.expressions[out_id] = self.expressions[out_id].model_copy(
                        update={"hidden": True})
        self._record_plan_summary(plan, failed_at)
        return plan

    def create_candidate_plan(self, goal_text: str, target_expr_id: str,
                              steps: "list", title: str = "") -> SupervisorPlan:
        """Turn a CHOSEN Adviser candidate into a validated plan (source="llm").

        Validation runs again here even though the candidate already passed
        the propose-time gate — defense in depth against a stale/forged
        candidate payload."""
        from notebook.models.plan import PlanStep
        plan = SupervisorPlan(
            plan_id=self._alloc_plan_id(),
            goal_text=goal_text or title,
            source="llm",
            workflow_name="",
            target_expr_id=target_expr_id,
            steps=[s if isinstance(s, PlanStep) else PlanStep.model_validate(s)
                   for s in steps],
        )
        validate_plan(plan, self._validator, self.session.primary_backend,
                      self._active_expr_ids(), lang=self.lang)
        plan = plan.model_copy(update={"status": "validated"})
        self.plans[plan.plan_id] = plan
        return plan

    def retry_plan(self, plan_id: str,
                   args_override: "dict[str, dict] | None" = None
                   ) -> SupervisorPlan:
        """Retry a stopped plan as a NEW plan (immutable history, B-3).

        Successful steps of the old run are carried over (their output
        expressions are reused, nothing re-executes) — except steps whose
        args are overridden, which re-execute with the fix.
        """
        old = self.plans.get(plan_id)
        if old is None:
            raise ValueError(t("plan.not_found", self.lang, plan_id=plan_id))
        if old.status not in ("stopped", "failed"):
            raise ValueError(t("plan.not_retryable", self.lang,
                               plan_id=plan_id, status=old.status))
        override = args_override or {}
        steps = [
            step.model_copy(update={
                "args": {**step.args, **override.get(step.step_id, {})}
            }) if step.step_id in override else step
            for step in old.steps
        ]
        from datetime import datetime as _dt, timezone as _tz
        new_plan = old.model_copy(update={
            "plan_id": self._alloc_plan_id(),
            "steps": steps,
            "status": "draft",
            "step_results": [],
            "retried_from": old.plan_id,
            "created_at": _dt.now(_tz.utc),
        })
        validate_plan(new_plan, self._validator, self.session.primary_backend,
                      self._active_expr_ids(), lang=self.lang)
        self.plans[new_plan.plan_id] = new_plan.model_copy(
            update={"status": "validated"})
        carry = {r.step_id: r for r in old.step_results
                 if r.status == "success" and r.step_id not in override}
        return self.execute_plan(new_plan.plan_id, _carry=carry)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_plan_placeholder(value, plan: SupervisorPlan,
                                  prev_expr: str, outputs: dict) -> object:
        if not isinstance(value, str) or not value.startswith("$"):
            return value
        if value == PLACEHOLDER_TARGET:
            return plan.target_expr_id
        if value == PLACEHOLDER_PREV:
            return prev_expr
        return outputs.get(value[1:], value)

    def _record_plan_summary(self, plan: SupervisorPlan,
                             failed_at: "str | None") -> None:
        ok = sum(1 for r in plan.step_results if r.status == "success")
        if failed_at is None:
            action = t("plan.executed_summary", self.lang,
                       plan_id=plan.plan_id, goal=plan.goal_text,
                       n_ok=ok, n_total=len(plan.steps))
        else:
            action = t("plan.stopped_summary", self.lang,
                       plan_id=plan.plan_id, goal=plan.goal_text,
                       step_id=failed_at, n_ok=ok, n_total=len(plan.steps))
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.plan_execution,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=action,
            operation="execute_plan",
            operation_args={
                "plan_id": plan.plan_id,
                "workflow_name": plan.workflow_name,
                "goal_text": plan.goal_text,
                "target_expr_id": plan.target_expr_id,
                "step_results": [r.model_dump(mode="json")
                                 for r in plan.step_results],
            },
            input_refs=[plan.target_expr_id] if plan.target_expr_id else [],
            output_refs=[r.expr_id for r in plan.step_results if r.expr_id],
            state_affecting=False,
            status=StepStatus.success if failed_at is None else StepStatus.failed,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)

    # ------------------------------------------------------------------
    # Branch comparison (C-Ext-5 B-4, read-only): mirrors
    # numeric_branch_comparison() (engine_numeric_mixin.py) exactly — never
    # re-executes a plan on another branch (that would touch the single
    # live CAS subprocess shared by all branches); it only groups
    # ALREADY-RECORDED plan_execution steps by branch for comparison.
    # ------------------------------------------------------------------

    def plan_branch_comparison(self) -> dict:
        """Group plan_execution steps by (workflow/goal label, target expr);
        a group is "comparable" when its steps span 2+ branches.  Every
        plan_execution step is already branch-tagged (branch_id=active_branch_id
        at execution time, like any other step) — this is a pure read/group
        method, no new tagging needed."""
        groups: dict = {}
        branch_ids: set = set()
        for step in sorted(self.steps.values(), key=lambda s: s.timestamp):
            if step.kind != StepKind.plan_execution:
                continue
            args = step.operation_args or {}
            label = args.get("workflow_name") or args.get("goal_text", "")
            target = args.get("target_expr_id", "")
            key = (label, target)
            g = groups.setdefault(key, {"label": label, "target": target,
                                        "branches": {}})
            g["branches"].setdefault(step.branch_id, []).append({
                "plan_id": args.get("plan_id", ""),
                "step_id": step.step_id,
                "status": step.status.value,
                "summary": step.visible_action,
            })
            branch_ids.add(step.branch_id)

        out = [{**g, "comparable": len(g["branches"]) >= 2}
               for g in groups.values()]
        return {"groups": out, "branch_ids": sorted(branch_ids)}
