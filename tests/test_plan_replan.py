"""Failure→replan loop (open_items C-Ext-5, 2026-07-18).

Unlike retry_plan (B-3: the SAME plan re-executed once the researcher fixes
the root cause), propose_replan asks the Adviser to propose a genuinely
DIFFERENT approach, informed by what failed.  Everything deterministic is
covered here with a FAKE chat callable — the failure-context text, the
JSON contract (shared with propose_plan_candidates, already covered by
test_plan_candidates.py), and graceful fallback on anything unresolvable.
What remains untested by design is whether a real LLM can turn a failure
into a sensible fix (docs/23 §11.1 — real usage is the only evaluator).
"""
import asyncio
import json

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.models.plan import PlanStep, PlanStepResult, SupervisorPlan
from notebook.runtime.plan_adviser import build_failure_context
from ui.interpret_plan import propose_replan
from ui.routes.session import NewSessionRequest, new_session, plan_replan


GOOD_REPLY = json.dumps([
    {"title": "対称性を使う", "summary": "固有値のみ直接計算",
     "steps": [{"operation": "eigenvalues", "args": {"expr": "$target"}}]},
])


def _run(coro):
    return asyncio.run(coro)


def _stopped_plan(engine, target_expr_id, goal_text="固有系を求めて"):
    steps = [
        PlanStep(step_id="s1", operation="characteristic_polynomial",
                 args={"expr": "$target", "var": "x"}),
        PlanStep(step_id="s2", operation="eigenvalues", args={"expr": "$target"}),
    ]
    results = [
        PlanStepResult(step_id="s1", operation="characteristic_polynomial",
                       status="success", expr_id="E2", value="x^2-4*x+3"),
        PlanStepResult(step_id="s2", operation="eigenvalues",
                       status="failed", error="something went wrong"),
    ]
    plan = SupervisorPlan(
        plan_id=engine._alloc_plan_id(), goal_text=goal_text, source="workflow",
        target_expr_id=target_expr_id, steps=steps, status="stopped",
        step_results=results,
    )
    engine.plans[plan.plan_id] = plan
    return plan


# ---------------------------------------------------------------------------
# build_failure_context: pure formatting, no LLM.
# ---------------------------------------------------------------------------

class TestBuildFailureContext:

    def test_includes_success_failure_and_skip(self):
        results = [
            PlanStepResult(step_id="s1", operation="op1", status="success", value="42"),
            PlanStepResult(step_id="s2", operation="op2", status="failed", error="boom"),
            PlanStepResult(step_id="s3", operation="op3", status="skipped"),
        ]
        plan = SupervisorPlan(plan_id="P1", goal_text="goal", step_results=results)
        ctx = build_failure_context(plan)
        assert "op1" in ctx and "42" in ctx
        assert "op2" in ctx and "FAILED" in ctx and "boom" in ctx
        assert "op3" in ctx and "skipped" in ctx


# ---------------------------------------------------------------------------
# propose_replan: deterministic skeleton via a fake Adviser chat.
# ---------------------------------------------------------------------------

class TestProposeReplan:

    def test_unknown_plan_id_returns_not_ok(self):
        e = NotebookEngine(title="replan missing")

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", "P999", chat=lambda s, u: "")
        res = _run(_inner())
        assert res["ok"] is False
        assert res["detail"]

    def test_non_stopped_plan_rejected(self):
        e = NotebookEngine(title="replan completed")
        ex = e.register_expression("x^2 - 1")
        plan = SupervisorPlan(plan_id=e._alloc_plan_id(), goal_text="goal",
                              target_expr_id=ex.display_id, status="completed")
        e.plans[plan.plan_id] = plan

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id, chat=lambda s, u: "")
        res = _run(_inner())
        assert res["ok"] is False
        assert res["detail"]

    def test_missing_target_expr_returns_not_ok(self):
        e = NotebookEngine(title="replan missing target")
        plan = _stopped_plan(e, target_expr_id="E999")

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id, chat=lambda s, u: "")
        res = _run(_inner())
        assert res["ok"] is False
        assert res["detail"]

    def test_no_candidates_from_adviser_returns_not_ok(self):
        e = NotebookEngine(title="replan no candidates")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = _stopped_plan(e, ex.display_id)

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id,
                                        chat=lambda s, u: "not json at all")
        res = _run(_inner())
        assert res["ok"] is False
        assert res["detail"]

    def test_successful_replan_includes_failure_context_in_prompt(self):
        e = NotebookEngine(title="replan prompt content")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = _stopped_plan(e, ex.display_id)
        seen = []

        def fake_chat(system, user):
            seen.append(user)
            return GOOD_REPLY

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id, chat=fake_chat)
        res = _run(_inner())
        assert res["ok"] is True
        assert len(seen) == 1
        assert "固有系を求めて" in seen[0]           # original goal
        assert "eigenvalues" in seen[0]              # the failed step's operation
        assert "FAILED" in seen[0]
        assert "something went wrong" in seen[0]     # the actual error text

    def test_successful_replan_response_shape(self):
        e = NotebookEngine(title="replan response shape")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = _stopped_plan(e, ex.display_id)

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id,
                                        chat=lambda s, u: GOOD_REPLY)
        res = _run(_inner())
        assert res["ok"] is True
        assert res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 1
        assert res["candidates"][0]["operation"] == "execute_plan_candidate"
        assert res["candidates"][0]["args"]["target_expr_id"] == ex.display_id

    def test_goal_text_in_response_is_clean_not_augmented(self):
        """The LLM-facing prompt carries the failure dump; the PERSISTENT
        goal shown to the researcher and stored on the eventual new plan
        must stay the clean original goal, not the raw failure text."""
        e = NotebookEngine(title="replan clean goal")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = _stopped_plan(e, ex.display_id, goal_text="固有系を求めて")

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id,
                                        chat=lambda s, u: GOOD_REPLY)
        res = _run(_inner())
        assert res["_user_text"] == "固有系を求めて"
        assert res["candidates"][0]["args"]["goal"] == "固有系を求めて"
        assert "FAILED" not in res["confirmation_prompt"]

    def test_prompt_wording_differs_from_manual_candidates(self):
        e = NotebookEngine(title="replan prompt wording")
        ex = e.register_expression("[2, 1; 1, 2]")
        plan = _stopped_plan(e, ex.display_id)

        async def _inner():
            loop = asyncio.get_event_loop()
            return await propose_replan(e, loop, "ja", plan.plan_id,
                                        chat=lambda s, u: GOOD_REPLY)
        res = _run(_inner())
        assert "失敗" in res["confirmation_prompt"]


# ---------------------------------------------------------------------------
# POST /plan/replan/{plan_id}: the HTTP surface (deterministic cases only —
# a real Adviser round-trip is exercised live, not in this unit suite).
# ---------------------------------------------------------------------------

class TestPlanReplanRoute:

    def test_unknown_plan_id_returns_ok_false(self):
        async def _inner():
            await new_session(NewSessionRequest(title="replan route missing", mode="standard", language="ja"))
            try:
                resp = await plan_replan("P999")
                assert resp["ok"] is False
                assert resp["detail"]
            finally:
                nb.stop_engine()
        _run(_inner())

    def test_non_stopped_plan_rejected(self):
        async def _inner():
            await new_session(NewSessionRequest(title="replan route completed", mode="standard", language="ja"))
            try:
                engine = nb.get_engine()
                ex = engine.register_expression("[2, 1; 1, 2]")
                plan = engine.create_workflow_plan("eigensystem", ex.display_id)
                done = engine.execute_plan(plan.plan_id)
                assert done.status == "completed"

                resp = await plan_replan(plan.plan_id)
                assert resp["ok"] is False
                assert resp["detail"]
            finally:
                nb.stop_engine()
        _run(_inner())
