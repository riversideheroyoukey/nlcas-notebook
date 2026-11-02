"""C-Ext-5 priority 3': free-form plan trigger.

Reuses propose_plan_candidates/validate_plan/execute_plan_candidate entirely
unchanged (see tests/test_plan_candidates.py for the exhaustive coverage of
that shared machinery — contract parsing, the validate_plan gate, forged-
payload rejection). What's specific to this file: the new trigger phrase
fires/doesn't fire, and the Adviser is consulted with _FREEFORM_SYSTEM_PROMPT
(not _SYSTEM_PROMPT) — the only two things that actually differ.
"""
import asyncio
import json

from notebook.engine import NotebookEngine
from notebook.runtime.i18n import t
from notebook.runtime.plan_adviser import _FREEFORM_SYSTEM_PROMPT, _SYSTEM_PROMPT
from ui.interpret_plan import _try_plan_freeform
from ui.notebook_bridge import OperationRequest, _handle_side_effect_operation

GOOD_REPLY = json.dumps([
    {"title": "Factor then simplify", "summary": "basic approach",
     "steps": [{"operation": "factor", "args": {"expr": "$target"}}]},
])


def _run_nl(engine, text, chat, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _try_plan_freeform(engine, loop, lang, text, [],
                                        chat=chat)
    return asyncio.run(_inner())


def _run_bridge(engine, req):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _handle_side_effect_operation(req, engine, loop)
    return asyncio.run(_inner())


class TestFreeformTrigger:

    def test_jp_ask_yields_confirmation(self):
        e = NotebookEngine(title="freeform nl jp")
        ex = e.register_expression("x^2 - 1")
        res = _run_nl(e, f"{ex.display_id} の計画を立てて",
                      chat=lambda s, u: GOOD_REPLY, lang="ja")
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 1
        c = res["candidates"][0]
        assert c["operation"] == "execute_plan_candidate"
        assert c["args"]["target_expr_id"] == ex.display_id

    def test_en_ask_yields_confirmation(self):
        e = NotebookEngine(title="freeform nl en")
        ex = e.register_expression("x^2 - 1")
        res = _run_nl(e, f"plan this for {ex.display_id}",
                      chat=lambda s, u: GOOD_REPLY, lang="en")
        assert res is not None and res["status"] == "needs_confirmation"

    def test_non_freeform_ask_does_not_fire(self):
        e = NotebookEngine(title="freeform narrow")
        e.register_expression("x^2 - 1")
        assert _run_nl(e, "E1 を因数分解して",
                       chat=lambda s, u: GOOD_REPLY) is None

    def test_candidates_phrase_does_not_fire_freeform(self):
        """_CANDIDATES_RE and _FREEFORM_RE are distinct phrases — a
        candidates ask must not also trigger the freeform path."""
        e = NotebookEngine(title="freeform vs candidates")
        e.register_expression("x^2 - 1")
        assert _run_nl(e, "E1 の解法の候補を出して",
                       chat=lambda s, u: GOOD_REPLY) is None

    def test_uses_freeform_system_prompt(self):
        e = NotebookEngine(title="freeform system prompt")
        ex = e.register_expression("x^2 - 1")
        captured = {}

        def spy_chat(system, user):
            captured["system"] = system
            return GOOD_REPLY

        _run_nl(e, f"{ex.display_id} の計画を立てて", chat=spy_chat, lang="ja")
        assert captured["system"] == _FREEFORM_SYSTEM_PROMPT
        assert captured["system"] != _SYSTEM_PROMPT

    def test_messages_translate_both_languages(self):
        for lang in ("ja", "en"):
            assert t("plan.freeform_prompt", lang, goal="g")


class TestFreeformSelectionLoop:

    def test_selected_plan_executes_and_records(self):
        """End-to-end confirmation that execute_plan_candidate is fully
        shared with the Adviser candidates path — no branching needed."""
        with NotebookEngine(title="freeform select") as e:
            ex = e.register_expression("x^2 - 1")
            res = _run_nl(e, f"{ex.display_id} の計画を立てて",
                          chat=lambda s, u: GOOD_REPLY, lang="ja")
            chosen = res["candidates"][0]
            req = OperationRequest(
                operation=chosen["operation"], args=chosen["args"],
                user_text=res["_user_text"], candidates=res["candidates"])
            resp = _run_bridge(e, req)
            assert resp is not None and resp["ok"]
            assert len(resp["plan_step_results"]) == 1
            plan = e.plans[resp["plan_id"]]
            assert plan.source == "llm"
            assert plan.status == "completed"
