"""D-5 (open_items.md): POST /plan/retry/{plan_id} — retry_plan's first HTTP
surface (previously engine-method-only, per B-3).  Mirrors the direct-route-call
test style already used in tests/test_bridge_session_routes.py."""
import asyncio

import ui.notebook_bridge as nb
from ui.routes.session import new_session, plan_retry, NewSessionRequest


def _run(coro):
    return asyncio.run(coro)


class TestPlanRetryRoute:
    def test_unknown_plan_id_returns_ok_false(self):
        async def _inner():
            await new_session(NewSessionRequest(title="retry route missing", mode="standard", language="ja"))
            try:
                resp = await plan_retry("P999")
                assert resp["ok"] is False
                assert resp["detail"]
            finally:
                nb.stop_engine()
        _run(_inner())

    def test_non_retryable_plan_rejected(self):
        """retry_plan only accepts a stopped/failed plan — a plan that
        completed cleanly must be rejected with a clear detail, not silently
        re-run."""
        async def _inner():
            await new_session(NewSessionRequest(title="retry route completed", mode="standard", language="ja"))
            try:
                engine = nb.get_engine()
                ex = engine.register_expression("[2, 1; 1, 2]")
                plan = engine.create_workflow_plan("eigensystem", ex.display_id)
                done = engine.execute_plan(plan.plan_id)
                assert done.status == "completed"

                resp = await plan_retry(plan.plan_id)
                assert resp["ok"] is False
                assert resp["detail"]
            finally:
                nb.stop_engine()
        _run(_inner())

    def test_stopped_plan_retried_successfully_after_fix(self):
        """Mirrors the real S-z scenario end to end: a plan stops on an
        undeclared spin operator, the researcher fixes the root cause, and
        the retry route resumes it as a new (immutable-history) plan."""
        async def _inner():
            await new_session(NewSessionRequest(
                title="retry route success", mode="second_quantization",
                language="en", domains=["spin_systems"]))
            try:
                engine = nb.get_engine()
                engine.run_operation("define_scalar_parameter", {"name": "gamma"})
                engine.run_operation("define_scalar_parameter", {"name": "B"})
                h = engine.register_expression(r"H = -\gamma * B * Sz_1")
                plan = engine.create_workflow_plan("heisenberg_components", h.display_id)
                stopped = engine.execute_plan(plan.plan_id)
                assert stopped.status == "stopped"  # Sz undeclared (mirrors S-z)

                engine.run_operation("define_spin_operator", {"name": "S"})
                resp = await plan_retry(plan.plan_id)
                assert resp["ok"] is True
                assert resp["status"] == "completed"
                assert resp["plan_id"] != plan.plan_id  # new, immutable-history plan
                assert len(resp["plan_step_results"]) == 3
                assert all(r["status"] == "success" for r in resp["plan_step_results"])
            finally:
                nb.stop_engine()
        _run(_inner())
