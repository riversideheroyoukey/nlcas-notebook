"""N-1 numeric-fork NL trigger (_try_numeric_fork_nl, ui/interpret_numeric.py).

Deliberately narrow scope: single sweep variable over an expression whose
ONLY free symbol is the swept one.  Multi-variable / list-matrix / non-matching
text must fall through (return None) to the existing modal, not fork.
"""
import asyncio

import pytest

from notebook.engine import NotebookEngine
from ui.interpret_numeric import _try_numeric_fork_nl
from ui.routes.operations import _handle_side_effect_operation, OperationRequest
from notebook.numeric.sandbox import SandboxRunner

needs_sandbox = pytest.mark.skipif(
    not SandboxRunner().available(),
    reason="macOS sandbox-exec or .venv-numeric python unavailable",
)


def _run_nl(engine, text, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _try_numeric_fork_nl(engine, loop, lang, text, [])
    return asyncio.run(_inner())


class TestTriggerFires:
    def test_jp_explicit_target(self):
        e = NotebookEngine(title="numeric nl jp")
        ex = e.register_expression("sin(omega)")
        res = _run_nl(e, f"{ex.display_id} の omega を 0 から 5 まで 200 点で数値的に掃引して",
                      lang="ja")
        assert res is not None and res["status"] == "needs_confirmation"
        assert res["operation"] == "numeric_fork_nl"
        assert res["args"]["expr_id"] == ex.display_id
        assert res["args"]["sweep_parameters"] == {
            "omega": {"min": 0.0, "max": 5.0, "points": 200}}

    def test_en_explicit_target(self):
        e = NotebookEngine(title="numeric nl en")
        ex = e.register_expression("sin(omega)")
        res = _run_nl(e, f"sweep omega from 0 to 5 with 200 points for {ex.display_id}",
                      lang="en")
        assert res is not None and res["status"] == "needs_confirmation"
        assert res["args"]["sweep_parameters"] == {
            "omega": {"min": 0.0, "max": 5.0, "points": 200}}

    def test_jp_target_falls_back_to_latest_expression(self):
        e = NotebookEngine(title="numeric nl jp fallback")
        e.register_expression("sin(omega)")
        res = _run_nl(e, "omega を 0 から 5 まで 掃引して", lang="ja")
        assert res is not None
        assert res["args"]["expr_id"] == "E1"

    def test_default_points_when_omitted(self):
        e = NotebookEngine(title="numeric nl default points")
        ex = e.register_expression("sin(omega)")
        res = _run_nl(e, f"sweep omega from 0 to 5 for {ex.display_id}", lang="en")
        assert res["args"]["sweep_parameters"]["omega"]["points"] == 200

    def test_zenkaku_digits_accepted(self):
        e = NotebookEngine(title="numeric nl zenkaku")
        ex = e.register_expression("sin(omega)")
        res = _run_nl(e, f"{ex.display_id} の omega を ０ から ５ まで 掃引して", lang="ja")
        assert res is not None
        assert res["args"]["sweep_parameters"]["omega"]["min"] == 0.0
        assert res["args"]["sweep_parameters"]["omega"]["max"] == 5.0


class TestTriggerDoesNotFire:
    def test_non_matching_text_returns_none(self):
        e = NotebookEngine(title="numeric nl no match")
        ex = e.register_expression("sin(omega)")
        assert _run_nl(e, f"{ex.display_id} を計算して", lang="ja") is None

    def test_multi_variable_expression_falls_through(self):
        """Two free symbols but only one is swept — must defer to the modal,
        not silently fork with the other symbol unbound."""
        e = NotebookEngine(title="numeric nl multi var")
        ex = e.register_expression("omega*gamma")
        assert _run_nl(e, f"{ex.display_id} の omega を 0 から 5 まで 掃引して",
                      lang="ja") is None

    def test_matrix_target_falls_through(self):
        e = NotebookEngine(title="numeric nl matrix")
        ex = e.register_expression("[2,1;1,2]")
        assert _run_nl(e, f"sweep omega from 0 to 5 for {ex.display_id}",
                      lang="en") is None

    def test_unknown_target_returns_none(self):
        e = NotebookEngine(title="numeric nl unknown target")
        assert _run_nl(e, "sweep omega from 0 to 5 for E99", lang="en") is None

    def test_empty_text_returns_none(self):
        e = NotebookEngine(title="numeric nl empty")
        assert _run_nl(e, "", lang="ja") is None


@needs_sandbox
class TestConfirmDispatchIntegration:
    def test_confirm_click_forks_generates_and_runs(self, tmp_path):
        import ui.session_io as session_io

        e = NotebookEngine(title="numeric nl dispatch")
        ex = e.register_expression("sin(omega)")

        async def _inner():
            await session_io.do_save(
                e, filename=str(tmp_path / "nl_dispatch.json"), overwrite=True)
            loop = asyncio.get_event_loop()
            req = OperationRequest(operation="numeric_fork_nl", args={
                "expr_id": ex.display_id, "problem_type": "parameter_sweep",
                "target_quantity": "",
                "sweep_parameters": {"omega": {"min": 0.0, "max": 5.0, "points": 50}},
                "fixed_values": {}})
            return await _handle_side_effect_operation(req, e, loop)

        resp = asyncio.run(_inner())
        assert resp is not None
        assert resp["ok"] is True
        assert resp["status"] == "confirmed"
        assert resp["operation"] == "numeric_fork_nl"
        assert resp["figure_artifact_id"]
        assert resp["text"]
        from notebook.models.step import StepKind
        assert any(s.kind == StepKind.numeric_result for s in e.steps.values())
