"""C-Ext-5 Supervisor auto-selection (2026-07-18, open_items).

The Adviser proposes (same propose_plan_candidates loop test_plan_candidates.py
covers); the Supervisor picks ONE of the already-validated candidates via
select_best_candidate — a different LLM role from the one that proposed
them, mirroring the existing Supervisor/Adviser split.  The chosen candidate
still goes through the normal needs_confirmation flow (a single-item
candidates modal) — nothing here bypasses the researcher's final approval
(docs/23 §2.4).
"""
import asyncio
import json

from notebook.engine import NotebookEngine
from notebook.models.plan import PlanCandidate, PlanStep
from notebook.runtime.plan_adviser import select_best_candidate
from ui.interpret_plan import _try_plan_auto_select


GOOD_REPLY = json.dumps([
    {"title": "固有値のみ", "summary": "固有値だけ素早く",
     "steps": [{"operation": "eigenvalues", "args": {"expr": "$target"}}]},
    {"title": "完全な固有系", "summary": "特性多項式から固有ベクトルまで",
     "steps": [
         {"operation": "characteristic_polynomial",
          "args": {"expr": "$target", "var": "x"}},
         {"operation": "eigenvalues", "args": {"expr": "$target"}},
         {"operation": "eigenvectors", "args": {"expr": "$target"}},
     ]},
])


def _candidates(n=2):
    steps = [PlanStep(step_id="s1", operation="eigenvalues", args={"expr": "$target"})]
    return [PlanCandidate(candidate_id=f"c{i}", title=f"title{i}", summary=f"summary{i}",
                          recommended_when="", risk="", steps=steps)
            for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# select_best_candidate: deterministic skeleton, no live model needed.
# ---------------------------------------------------------------------------

class TestSelectBestCandidate:

    def test_single_candidate_shortcuts_without_chat(self):
        """Nothing to choose between — must not consult the LLM at all."""
        cands = _candidates(1)

        def boom(system, user):
            raise AssertionError("chat must not be called for a single candidate")

        assert select_best_candidate(boom, "goal", cands) is cands[0]

    def test_empty_candidates_returns_none(self):
        assert select_best_candidate(lambda s, u: "", "goal", []) is None

    def test_valid_selection(self):
        cands = _candidates(3)
        reply = json.dumps({"selected_id": "c2"})
        assert select_best_candidate(lambda s, u: reply, "goal", cands) is cands[1]

    def test_fenced_reply_parses(self):
        cands = _candidates(2)
        reply = f"Sure!\n```json\n{json.dumps({'selected_id': 'c1'})}\n```"
        assert select_best_candidate(lambda s, u: reply, "goal", cands) is cands[0]

    def test_unknown_id_returns_none(self):
        """Never guess — a selected_id that matches nothing given aborts."""
        cands = _candidates(2)
        reply = json.dumps({"selected_id": "c99"})
        assert select_best_candidate(lambda s, u: reply, "goal", cands) is None

    def test_malformed_json_returns_none(self):
        cands = _candidates(2)
        assert select_best_candidate(lambda s, u: "not json at all", "goal", cands) is None

    def test_non_object_json_returns_none(self):
        cands = _candidates(2)
        assert select_best_candidate(lambda s, u: "[1, 2, 3]", "goal", cands) is None

    def test_empty_reply_returns_none(self):
        cands = _candidates(2)
        assert select_best_candidate(lambda s, u: "", "goal", cands) is None


# ---------------------------------------------------------------------------
# _try_plan_auto_select: NL trigger → Adviser candidates → Supervisor pick →
# single-item confirmation modal, or graceful fallback to all candidates.
# ---------------------------------------------------------------------------

def _run_auto_select(engine, text, adviser_chat, supervisor_chat=None, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _try_plan_auto_select(
            engine, loop, lang, text, [], None,
            adviser_chat=adviser_chat, supervisor_chat=supervisor_chat)
    return asyncio.run(_inner())


class TestAutoSelectTrigger:

    def test_jp_ask_with_single_candidate_needs_no_supervisor_chat(self):
        e = NotebookEngine(title="auto jp single")
        ex = e.register_expression("x^2 - 1")
        single_reply = json.dumps([
            {"title": "唯一の候補", "summary": "",
             "steps": [{"operation": "expand", "args": {"expr": "$target"}}]},
        ])

        def boom(system, user):
            raise AssertionError("supervisor chat must not fire for a single candidate")

        res = _run_auto_select(e, f"{ex.display_id} の解法を自動選んで",
                               adviser_chat=lambda s, u: single_reply,
                               supervisor_chat=boom)
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 1

    def test_en_ask_picks_supervisor_choice(self):
        e = NotebookEngine(title="auto en pick")
        ex = e.register_expression("[2, 1; 1, 2]")
        seen = []

        def supervisor_chat(system, user):
            seen.append(user)
            return json.dumps({"selected_id": "c2"})

        res = _run_auto_select(e, f"auto-select the best approach for {ex.display_id}",
                               adviser_chat=lambda s, u: GOOD_REPLY,
                               supervisor_chat=supervisor_chat, lang="en")
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 1
        assert res["candidates"][0]["display_ja"].startswith("完全な固有系")
        assert len(seen) == 1

    def test_falls_back_to_all_candidates_when_supervisor_unavailable(self):
        e = NotebookEngine(title="auto fallback none")
        ex = e.register_expression("[2, 1; 1, 2]")
        res = _run_auto_select(e, f"{ex.display_id} の解法を自動選択して",
                               adviser_chat=lambda s, u: GOOD_REPLY,
                               supervisor_chat=None)
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 2

    def test_falls_back_when_supervisor_reply_unparseable(self):
        e = NotebookEngine(title="auto fallback bad reply")
        ex = e.register_expression("[2, 1; 1, 2]")
        res = _run_auto_select(e, f"{ex.display_id} の解法を自動選択して",
                               adviser_chat=lambda s, u: GOOD_REPLY,
                               supervisor_chat=lambda s, u: "not json")
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 2

    def test_non_autoselect_ask_does_not_fire(self):
        """The plain "candidates" phrase (_CANDIDATES_RE) must not also
        trigger auto-select — the two are distinct, explicit triggers."""
        e = NotebookEngine(title="auto narrow")
        ex = e.register_expression("x^2 - 1")
        res = _run_auto_select(e, f"{ex.display_id} の解法の候補を出して",
                               adviser_chat=lambda s, u: GOOD_REPLY)
        assert res is None

    def test_prompt_wording_differs_from_manual_candidates(self):
        e = NotebookEngine(title="auto prompt wording")
        ex = e.register_expression("[2, 1; 1, 2]")
        res = _run_auto_select(e, f"{ex.display_id} の解法を自動選んで",
                               adviser_chat=lambda s, u: GOOD_REPLY,
                               supervisor_chat=lambda s, u: json.dumps({"selected_id": "c1"}))
        assert "Supervisor" in res["confirmation_prompt"]
