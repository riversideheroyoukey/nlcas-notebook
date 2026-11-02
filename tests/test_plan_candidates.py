"""C-Ext-5 priority 3: Adviser candidate loop.

Everything deterministic is covered here with a FAKE chat callable — the
JSON contract, the validate_plan gatekeeper (broken proposals never reach
the modal), the candidates response shape, and the full select→execute→
record loop through the execute_plan_candidate pseudo-operation.  What
remains untested by design is the real Adviser's proposal quality
(docs/23 §9 C).
"""
import asyncio
import json

import pytest

from notebook.engine import NotebookEngine
from notebook.models.step import StepKind
from notebook.runtime.i18n import t
from notebook.runtime.plan_adviser import (
    build_operations_catalog, parse_candidates, propose_plan_candidates,
)
from ui.interpret_plan import _narrative_context, _target_context, _try_plan_candidates
from ui.notebook_bridge import OperationRequest, _handle_side_effect_operation


GOOD_REPLY = json.dumps([
    {"title": "固有値のみ", "summary": "固有値だけ素早く",
     "recommended_when": "スペクトルだけ欲しいとき", "risk": "低",
     "steps": [{"operation": "eigenvalues", "args": {"expr": "$target"}}]},
    {"title": "完全な固有系", "summary": "特性多項式から固有ベクトルまで",
     "risk": "低",
     "steps": [
         {"operation": "characteristic_polynomial",
          "args": {"expr": "$target", "var": "x"}},
         {"operation": "eigenvalues", "args": {"expr": "$target"}},
         {"operation": "eigenvectors", "args": {"expr": "$target"}},
     ]},
])

MIXED_REPLY = json.dumps([
    {"title": "正当な候補",
     "steps": [{"operation": "expand", "args": {"expr": "$target"}}]},
    {"title": "未知操作の候補",
     "steps": [{"operation": "summon_demon", "args": {"expr": "$target"}}]},
    {"title": "必須引数欠落の候補",
     "steps": [{"operation": "differentiate", "args": {"expr": "$target"}}]},
])


def _run_nl(engine, text, chat, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _try_plan_candidates(engine, loop, lang, text, [],
                                          chat=chat)
    return asyncio.run(_inner())


def _run_bridge(engine, req):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _handle_side_effect_operation(req, engine, loop)
    return asyncio.run(_inner())


# ---------------------------------------------------------------------------
# JSON contract parsing
# ---------------------------------------------------------------------------

class TestContractParsing:

    def test_good_reply_parses(self):
        cands = parse_candidates(GOOD_REPLY)
        assert [c.candidate_id for c in cands] == ["c1", "c2"]
        assert cands[1].steps[1].operation == "eigenvalues"
        # step ids are assigned by US, linear depends_on
        assert [s.step_id for s in cands[1].steps] == ["s1", "s2", "s3"]
        assert cands[1].steps[2].depends_on == ["s2"]

    def test_fenced_reply_parses(self):
        cands = parse_candidates(f"Here you go:\n```json\n{GOOD_REPLY}\n```")
        assert len(cands) == 2

    def test_broken_json_yields_nothing(self):
        assert parse_candidates("sure! [ {title: broken") == []
        assert parse_candidates("no json at all") == []
        assert parse_candidates(json.dumps({"title": "not a list"})) == []

    def test_malformed_items_skipped(self):
        reply = json.dumps([
            {"no_title": True, "steps": [{"operation": "expand", "args": {}}]},
            {"title": "empty steps", "steps": []},
            {"title": "ok", "steps": [{"operation": "expand",
                                       "args": {"expr": "$target"}}]},
        ])
        cands = parse_candidates(reply)
        assert [c.title for c in cands] == ["ok"]


# ---------------------------------------------------------------------------
# validate_plan gatekeeper on proposals
# ---------------------------------------------------------------------------

class TestProposeGate:

    def test_invalid_candidates_rejected_with_reason(self):
        e = NotebookEngine(title="cand gate")
        ex = e.register_expression("(x+1)^2")
        valid, rejected = propose_plan_candidates(
            lambda s, u: MIXED_REPLY, "goal", ex.display_id, ex.canonical,
            e._validator, e.session.primary_backend, e._active_expr_ids())
        assert [c.title for c in valid] == ["正当な候補"]
        assert {t_ for t_, _ in rejected} == {"未知操作の候補", "必須引数欠落の候補"}
        assert all(reason for _, reason in rejected)

    def test_catalog_contains_only_backend_ops(self):
        e = NotebookEngine(title="cand catalog")
        catalog = build_operations_catalog(e._validator, "maxima")
        assert "eigenvalues(expr)" in catalog
        assert "normal_order" not in catalog     # julia_full-only

    def test_unknown_field_rejected_even_when_required_fields_present(self):
        """Regression for the gap found in S-ae's real-LLM run (docs/evaluations.md):
        validate_plan checked required fields were present but never checked for
        EXTRA fields the operation doesn't accept — so a candidate could pass this
        gate and only fail at actual execution time. define_matrix requires only
        'name'; an LLM-proposed 'expr' field must now be rejected here instead."""
        e = NotebookEngine(title="cand unknown field")
        ex = e.register_expression("x^2 - 1")
        reply = json.dumps([
            {"title": "bad args", "steps": [
                {"operation": "define_matrix",
                 "args": {"name": "M", "expr": "$target"}}]},
        ])
        valid, rejected = propose_plan_candidates(
            lambda s, u: reply, "goal", ex.display_id, ex.canonical,
            e._validator, e.session.primary_backend, e._active_expr_ids())
        assert valid == []
        assert len(rejected) == 1
        assert "expr" in rejected[0][1]


# ---------------------------------------------------------------------------
# C-Ext-5 §11.1 context-supply increment #1: target_summary must not be a
# bare, possibly-lossy canonical (docs/evaluations.md S-aa problem②: an H-1
# equation's canonical stores RHS-only per CLAUDE.md's H-1 rule, so a
# registered "dy/dx + y = x" surfaced to the Adviser as the bare string "x",
# hiding the differential structure and producing a nonsense plan).
# ---------------------------------------------------------------------------

class TestTargetContext:

    def test_identical_latex_and_canonical_returns_bare_string(self):
        e = NotebookEngine(title="ctx identical")
        ex = e.register_expression("x^2 + y^2")
        assert _target_context(e, ex.display_id) == ex.canonical

    def test_h1_rhs_only_canonical_surfaces_full_latex(self):
        """Mirrors what engine.register_expression itself does for H-1 input
        (notebook/engine.py: canonical=RHS, latex="LHS = RHS" via lhs_label)
        — canonical alone ("x") would hide the ODE's differential structure
        that only latex retains."""
        e = NotebookEngine(title="ctx h1 ode")
        ex = e.register_expression("x", latex="dy/dx + y = x")
        assert ex.canonical == "x"
        ctx = _target_context(e, ex.display_id)
        assert "dy/dx" in ctx
        assert "canonical/CAS form: x" in ctx

    def test_h1_target_reaches_adviser_prompt(self):
        """End-to-end through _try_plan_candidates: the fake chat must
        receive the differential structure, not just the bare RHS."""
        e = NotebookEngine(title="ctx h1 e2e")
        ex = e.register_expression("x", latex="dy/dx + y = x")
        seen = []

        def fake_chat(system, user):
            seen.append(user)
            return GOOD_REPLY

        res = _run_nl(e, f"{ex.display_id} の解法の候補を出して", chat=fake_chat)
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(seen) == 1
        assert "dy/dx" in seen[0]


# ---------------------------------------------------------------------------
# C-Ext-5 §11.1 context-supply increment #2: session narrative (declared
# symbols/assumptions/recent history) reused from ui/routes/advisory.py's
# _build_session_narrative (the same compressor /advise already relies on),
# capped smaller and reaching the Adviser prompt alongside target_summary.
# ---------------------------------------------------------------------------

class TestNarrativeContext:

    def test_fresh_session_yields_empty_narrative(self):
        e = NotebookEngine(title="narrative empty")
        assert _narrative_context(e, "some goal") == ""

    def test_explicit_max_items_overrides_config_default(self):
        e = NotebookEngine(title="narrative explicit override")
        for i in range(3):
            e.register_expression(f"x{i}", latex=f"x_{i}")
        ctx = _narrative_context(e, "goal", max_items=1)
        assert ctx.count("registered") == 1

    def test_default_max_items_reads_config(self):
        """max_items=None (the default) must resolve via
        config.config_utils.get_plan_narrative_max_items() rather than a
        hardcoded literal — this is what makes the item count tunable via
        config/default_settings.json without a code change."""
        e = NotebookEngine(title="narrative config default")
        for i in range(8):
            e.register_expression(f"x{i}", latex=f"x_{i}")
        from config.config_utils import get_plan_narrative_max_items
        expected = get_plan_narrative_max_items()
        ctx = _narrative_context(e, "goal")
        assert ctx.count("registered") == expected

    def test_assumption_and_registration_appear(self):
        with NotebookEngine(title="narrative basic") as e:
            e.register_expression("a", latex="a")
            e.add_assumption("a > 0")
            ctx = _narrative_context(e, "unrelated current text")
        assert "registered" in ctx
        assert "assumption" in ctx

    def test_current_trigger_text_excluded(self):
        """The current instruction is recorded as a user_input step BEFORE
        the NL trigger fires (ui/interpret_pipeline.py records the YOU-card
        first) — _narrative_context must not echo it back as if it were
        prior history, since it's already shown as "Goal:" in the prompt."""
        e = NotebookEngine(title="narrative dedup")
        e.register_expression("a", latex="a")
        e.record_user_input("解法の候補を出して")
        ctx = _narrative_context(e, "解法の候補を出して")
        assert "解法の候補を出して" not in ctx

    def test_narrative_reaches_adviser_prompt(self):
        with NotebookEngine(title="narrative e2e") as e:
            e.register_expression("a", latex="a")
            e.add_assumption("a > 0")
            ex = e.register_expression("x", latex="dy/dx + y = a*x")
            seen = []

            def fake_chat(system, user):
                seen.append(user)
                return GOOD_REPLY

            res = _run_nl(e, f"{ex.display_id} の解法の候補を出して", chat=fake_chat)
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(seen) == 1
        assert "Session context" in seen[0]
        assert "a > 0" in seen[0] or "assumption" in seen[0]


# ---------------------------------------------------------------------------
# NL trigger → candidates modal shape
# ---------------------------------------------------------------------------

class TestCandidatesTrigger:

    def test_jp_ask_yields_confirmation_with_candidates(self):
        e = NotebookEngine(title="cand nl jp")
        ex = e.register_expression("[2, 1; 1, 2]")
        res = _run_nl(e, f"{ex.display_id} の解法の候補を出して",
                      chat=lambda s, u: GOOD_REPLY, lang="ja")
        assert res is not None and res["status"] == "needs_confirmation"
        assert len(res["candidates"]) == 2
        c = res["candidates"][0]
        assert c["operation"] == "execute_plan_candidate"
        assert c["args"]["target_expr_id"] == ex.display_id
        assert c["display_ja"]
        assert "解法候補" in res["confirmation_prompt"]

    def test_en_ask_yields_confirmation(self):
        e = NotebookEngine(title="cand nl en")
        ex = e.register_expression("[2, 1; 1, 2]")
        res = _run_nl(e, f"Give me solution candidates for {ex.display_id}",
                      chat=lambda s, u: GOOD_REPLY, lang="en")
        assert res is not None and res["status"] == "needs_confirmation"
        assert "candidates" in res["confirmation_prompt"].lower()

    def test_non_candidate_ask_does_not_fire(self):
        e = NotebookEngine(title="cand narrow")
        e.register_expression("[2, 1; 1, 2]")
        assert _run_nl(e, "E1 の固有値を求めて",
                       chat=lambda s, u: GOOD_REPLY) is None

    def test_all_rejected_gives_translated_error(self):
        e = NotebookEngine(title="cand none")
        ex = e.register_expression("x")
        broken = json.dumps([{"title": "bad", "steps": [
            {"operation": "summon_demon", "args": {}}]}])
        res = _run_nl(e, f"{ex.display_id} の解法の候補を出して",
                      chat=lambda s, u: broken, lang="ja")
        assert res["status"] == "failed"
        assert "棄却" in res["error"]

    def test_adviser_unavailable_message(self):
        e = NotebookEngine(title="cand no adviser")
        e.register_expression("x")
        # chat=None and no bridge adviser configured in the test process
        res = _run_nl(e, "E1 の解法の候補を出して", chat=None, lang="ja")
        assert res["status"] == "failed"
        assert "Adviser" in res["error"]

    def test_messages_translate_both_languages(self):
        for lang in ("ja", "en"):
            assert t("plan.candidates_prompt", lang, goal="g")
            assert t("plan.no_candidates", lang, n_rejected=2)
            assert t("plan.adviser_unavailable", lang)


# ---------------------------------------------------------------------------
# Selection → execute_plan_candidate pseudo-op (full loop, real Maxima)
# ---------------------------------------------------------------------------

class TestSelectionLoop:

    def test_selected_candidate_executes_and_records(self):
        with NotebookEngine(title="cand select") as e:
            ex = e.register_expression("[2, 1; 1, 2]")
            res = _run_nl(e, f"{ex.display_id} の解法の候補を出して",
                          chat=lambda s, u: GOOD_REPLY, lang="ja")
            chosen = res["candidates"][1]        # 完全な固有系（3ステップ）
            req = OperationRequest(
                operation=chosen["operation"], args=chosen["args"],
                user_text=res["_user_text"], candidates=res["candidates"])
            resp = _run_bridge(e, req)
            assert resp is not None and resp["ok"]
            assert len(resp["plan_step_results"]) == 3
            plan = e.plans[resp["plan_id"]]
            assert plan.source == "llm"
            assert plan.status == "completed"
            assert "3" in plan.step_results[1].value    # eigenvalues {3,1}
            kinds = {s.kind for s in e.steps.values()}
            assert StepKind.ambiguity_resolution in kinds
            assert StepKind.plan_execution in kinds

    def test_forged_payload_rejected_by_second_validation(self):
        """Defense in depth: even if a tampered candidate reaches the
        pseudo-op, create_candidate_plan validates again."""
        e = NotebookEngine(title="cand forged")
        ex = e.register_expression("x")
        req = OperationRequest(
            operation="execute_plan_candidate",
            args={"goal": "g", "title": "forged",
                  "target_expr_id": ex.display_id,
                  "steps": [{"step_id": "s1", "operation": "summon_demon",
                             "args": {"expr": "$target"}}]})
        resp = _run_bridge(e, req)
        assert resp is not None and not resp["ok"]
        assert e.plans == {}                     # nothing was stored
