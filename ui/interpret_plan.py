"""C-Ext-5: rule-based triggers for fixed multi-step workflows.

Fixed workflows take precedence over any LLM planner.  The trigger is kept
NARROW on purpose: it fires only on the COMPOSITE ask (「固有状態」/
"eigenstates" / 「固有値と固有ベクトル」), never on single-operation
requests like 「固有値を求めて」 — those keep their existing 1-op route.
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operand_boundary import (
    EXPR_ID_IN_TEXT_RE, may_fall_back_to_the_recent_expression)

# The plan trigger grammars (eigensystem / Heisenberg-all / site / char-poly
# variable / candidates / free-form plan / auto-select) are externalized to
# i18n/nl_patterns.json (plan_*), looked up by session language.  All were
# already bilingual; the split keeps the same alternatives per language.  The
# Heisenberg base operator uses a shared group "base"; site/var use positional
# group 1 in both languages.
_ZENKAKU_DIGITS = {ord("０") + i: ord("0") + i for i in range(10)}


def _find_target(engine, text: str, consumed=()) -> "str | None":
    """An explicit E-id in the text, else the most recent expression.

    **名指していれば、それ以外で代用しない。** 実測 2026-09-11、この関数は
    どちらの半分も黙ってすり替えていた（日英とも）::

        G を行列とする（別に A = [2,1;1,2] が在る）
          G の固有値と固有ベクトルを求めて  →  completed  ← **A の固有系**
        E99 の固有値と固有ベクトルを求めて  →  completed  ← **A の固有系**

    落ちてよいかは族で1つの helper に訊く（`operand_boundary`）——この問いに
    5箇所が別々に答えていたのが、この形の誤りの根である。

    *consumed* は、呼び出し側の文法が**別の役で既に読んだ名前**。
    `S の各成分のハイゼンベルク方程式` の `S` は**対象ではなく、どの演算子の
    成分を発展させるか**で、対象はハミルトニアンである——渡さないと、いちばん
    普通の綴りで workflow が黙って降りる（実測 2026-09-11、guard を入れて初めて出た）。
    他の呼び出し側が読む名前は変数（スカラー）なので、そもそも数えられない。
    """
    m = EXPR_ID_IN_TEXT_RE.search(text)
    known = engine._active_expr_ids()
    if m and m.group(1) in known:
        return m.group(1)
    if not may_fall_back_to_the_recent_expression(engine.symbol_context, text,
                                                 consumed):
        return None
    return engine.most_recent_expr_id()


def _target_context(engine, target: str) -> str:
    """Target-expression summary handed to the Adviser (C-Ext-5 §11.1
    context-supply increment #1).  ``canonical`` alone is lossy for H-1
    equation-form expressions (CLAUDE.md H-1: canonical stores the RHS only,
    e.g. "x" for a registered "dy/dx + y = x" — the differential structure
    lives in ``latex``, not ``canonical``).  Sending both, and skipping the
    canonical line when it's identical to latex, avoids proposals built on
    an accidentally-truncated view of the target."""
    expr = engine.expressions[target]
    if expr.canonical == expr.latex:
        return expr.latex
    return f"{expr.latex}  (canonical/CAS form: {expr.canonical})"


def _narrative_context(engine, current_text: str, max_items: "int | None" = None) -> str:
    """Session narrative handed to the Adviser (C-Ext-5 §11.1 context-supply
    increment #2): declared symbols/assumptions/recent expression history.
    Reuses the same compressor /advise already relies on
    (ui/routes/advisory.py::_build_session_narrative) rather than building a
    second one, capped smaller here — the plan prompt needs orientation, not
    a full commentary transcript.  Lazy import avoids the module cycle
    advisory.py -> interpret_pipeline.py -> interpret_plan.py.

    ``max_items`` defaults to the config knob ``plan_narrative_max_items``
    (config/default_settings.json, config.config_utils.get_plan_narrative_max_items)
    rather than a hardcoded literal — the count was never measured, only
    chosen as "smaller than /advise's 15", so it needs to stay easy to tune
    without a code change while real usage data accumulates.

    The current trigger text itself is always the most recent user_input
    step by the time this runs (recorded before any NL trigger fires) and is
    filtered out — it's already shown as "Goal:" in the prompt."""
    if max_items is None:
        from config.config_utils import get_plan_narrative_max_items
        max_items = get_plan_narrative_max_items()
    from ui.routes.advisory import _build_session_narrative
    items = _build_session_narrative(engine, up_to_step_id=None,
                                     max_items=max_items + 1)
    lines: list[str] = []
    for item in items:
        kind = item["kind"]
        if kind == "user_input":
            if item.get("text", "").strip() == current_text.strip():
                continue
            lines.append(f"- instruction: {item['text']}")
        elif kind == "registration":
            lines.append(f"- registered {item['expr_id']}: {item['latex']}")
        elif kind == "cas_op":
            lines.append(f"- {item['expr_id']} = {item['operation']}"
                         f"({item.get('source_id') or ''}) → {item['latex']}")
        elif kind == "note":
            lines.append(f"- note: {item['text']}")
        elif kind == "assumption":
            lines.append(f"- assumption: {item['text']}")
    return "\n".join(lines[-max_items:])


def _resolve_adviser_chat():
    """Best-effort resolution of the configured Adviser LLM as a
    chat(system, user) -> str callable.  Lazy import avoids a module-load
    cycle with notebook_bridge (which imports this module's package)."""
    try:
        from ui.notebook_bridge import get_adviser
        adviser = get_adviser()
        llm = getattr(adviser, "_llm", None)
        if llm is None:
            return None

        def _chat(system: str, user: str) -> str:
            resp = llm.chat(system=system, user=user, temperature=0.2,
                            max_tokens=getattr(_chat, "max_tokens", 1024))
            _chat.last_error = "" if resp.ok else (getattr(resp, "error", "") or "empty_response")
            _chat.last_reply = resp.text or ""
            return resp.text if getattr(resp, "ok", False) else ""
        return _chat
    except Exception:
        return None


def _resolve_supervisor_chat(supervisor):
    """Best-effort resolution of the Supervisor LLM as a chat(system, user)
    -> str callable, mirroring _resolve_adviser_chat.  Returns None when the
    supervisor has no attached LLM client (e.g. --no-llm mode) — auto-select
    then falls back to showing every candidate for manual choice."""
    try:
        llm = getattr(supervisor, "_llm", None)
        if llm is None:
            return None

        def _chat(system: str, user: str) -> str:
            resp = llm.chat(system=system, user=user, temperature=0.2,
                            max_tokens=256)
            return resp.text if getattr(resp, "ok", False) else ""
        return _chat
    except Exception:
        return None


def _candidates_to_response(stripped, target, valid, registered, prompt_key, lang):
    """Build the needs_confirmation candidates-modal response shared by the
    Adviser candidate loop and the free-form planner — both produce the same
    PlanCandidate list shape, differing only in which prompt introduces it."""
    cands = []
    prompt_lines = [t(prompt_key, lang, goal=stripped)]
    for c in valid:
        label = c.title + (f" — {c.summary}" if c.summary else "")
        cands.append({
            "operation": "execute_plan_candidate",
            "args": {
                "goal": stripped,
                "title": c.title,
                "target_expr_id": target,
                "steps": [s.model_dump(mode="json") for s in c.steps],
            },
            "display_ja": label,
            "display_en": label,
        })
        detail = f"  ・{c.title}: {c.summary}"
        if c.risk:
            detail += f"（risk: {c.risk}）"
        prompt_lines.append(detail)
    return cands, prompt_lines


async def _try_plan_candidates(engine, loop, lang, text, registered,
                               chat=None) -> "dict | None":
    """Explicit 「解法の候補」 ask → Adviser consultation → candidates modal.

    Each surviving candidate becomes a button that posts the
    execute_plan_candidate pseudo-operation (same machinery as R-8's
    resolve_notation) — selection, execution and the ambiguity-resolution
    record all reuse the existing confirmation flow.
    """
    from notebook.runtime.plan_adviser import propose_plan_candidates

    stripped = (text or "").strip()
    if not stripped or not nlp("plan_candidates", lang).search(stripped):
        return None
    target = _find_target(engine, stripped)
    if not target:
        return None

    chat = chat or _resolve_adviser_chat()
    if chat is None:
        return {"status": "failed",
                "error": t("plan.adviser_unavailable", lang),
                "registered_expressions": registered}

    target_summary = _target_context(engine, target)
    narrative = _narrative_context(engine, stripped)
    valid, rejected = await loop.run_in_executor(None, partial(
        propose_plan_candidates, chat, stripped, target, target_summary,
        engine._validator, engine.session.primary_backend,
        engine._active_expr_ids(), lang, narrative_text=narrative))
    if not valid:
        return {"status": "failed",
                "error": t("plan.no_candidates", lang,
                           n_rejected=len(rejected)),
                "registered_expressions": registered}

    cands, prompt_lines = _candidates_to_response(
        stripped, target, valid, registered, "plan.candidates_prompt", lang)
    return {
        "status": "needs_confirmation",
        "operation": cands[0]["operation"],
        "args": cands[0]["args"],
        "confirmation_prompt": "\n".join(prompt_lines),
        "candidates": cands,
        "registered_expressions": registered,
        "_user_text": stripped,
    }


async def _try_plan_auto_select(engine, loop, lang, text, registered, supervisor,
                                adviser_chat=None, supervisor_chat=None) -> "dict | None":
    """Explicit auto-select ask → Adviser proposes candidates (same as
    _try_plan_candidates) → Supervisor picks ONE via select_best_candidate.

    The chosen candidate still goes through the normal needs_confirmation
    flow (a single-item candidates modal) — auto-selection only removes the
    "which of N" decision, never the researcher's final approval (docs/23
    §2.4: LLM-sourced plans always need confirmation).  Falls back to
    showing ALL candidates (identical to _try_plan_candidates) whenever the
    Supervisor's pick can't be resolved confidently — no Supervisor LLM
    configured, an unparseable reply, or an id that doesn't match any
    candidate — never guesses which one to run.
    """
    from notebook.runtime.plan_adviser import (
        propose_plan_candidates, select_best_candidate,
    )

    stripped = (text or "").strip()
    if not stripped or not nlp("plan_auto_select", lang).search(stripped):
        return None
    target = _find_target(engine, stripped)
    if not target:
        return None

    adviser_chat = adviser_chat or _resolve_adviser_chat()
    if adviser_chat is None:
        return {"status": "failed",
                "error": t("plan.adviser_unavailable", lang),
                "registered_expressions": registered}

    target_summary = _target_context(engine, target)
    narrative = _narrative_context(engine, stripped)
    valid, rejected = await loop.run_in_executor(None, partial(
        propose_plan_candidates, adviser_chat, stripped, target, target_summary,
        engine._validator, engine.session.primary_backend,
        engine._active_expr_ids(), lang, narrative_text=narrative))
    if not valid:
        return {"status": "failed",
                "error": t("plan.no_candidates", lang,
                           n_rejected=len(rejected)),
                "registered_expressions": registered}

    supervisor_chat = supervisor_chat or _resolve_supervisor_chat(supervisor)
    chosen = None
    if supervisor_chat is not None:
        chosen = await loop.run_in_executor(None, partial(
            select_best_candidate, supervisor_chat, stripped, valid))

    shown = [chosen] if chosen is not None else valid
    prompt_key = "plan.autoselect_prompt" if chosen is not None else "plan.candidates_prompt"
    cands, prompt_lines = _candidates_to_response(
        stripped, target, shown, registered, prompt_key, lang)
    return {
        "status": "needs_confirmation",
        "operation": cands[0]["operation"],
        "args": cands[0]["args"],
        "confirmation_prompt": "\n".join(prompt_lines),
        "candidates": cands,
        "registered_expressions": registered,
        "_user_text": stripped,
    }


async def _try_plan_freeform(engine, loop, lang, text, registered,
                             chat=None) -> "dict | None":
    """Explicit 「計画を立てて」/"plan this" ask → LLM plan proposal for a
    request that doesn't match any fixed workflow → same candidates modal.

    Reuses propose_plan_candidates/validate_plan/execute_plan_candidate
    unchanged (C-Ext-5 priority 3') — only the trigger phrase and the system
    prompt framing ("steps toward this goal" vs "1-3 alternatives") differ.
    """
    from notebook.runtime.plan_adviser import (
        _FREEFORM_SYSTEM_PROMPT, propose_plan_candidates,
    )

    stripped = (text or "").strip()
    if not stripped or not nlp("plan_freeform", lang).search(stripped):
        return None
    target = _find_target(engine, stripped)
    if not target:
        return None

    chat = chat or _resolve_adviser_chat()
    if chat is None:
        return {"status": "failed",
                "error": t("plan.adviser_unavailable", lang),
                "registered_expressions": registered}

    target_summary = _target_context(engine, target)
    narrative = _narrative_context(engine, stripped)
    valid, rejected = await loop.run_in_executor(None, partial(
        propose_plan_candidates, chat, stripped, target, target_summary,
        engine._validator, engine.session.primary_backend,
        engine._active_expr_ids(), lang,
        system_prompt=_FREEFORM_SYSTEM_PROMPT, narrative_text=narrative))
    if not valid:
        return {"status": "failed",
                "error": t("plan.no_candidates", lang,
                           n_rejected=len(rejected)),
                "registered_expressions": registered}

    cands, prompt_lines = _candidates_to_response(
        stripped, target, valid, registered, "plan.freeform_prompt", lang)
    return {
        "status": "needs_confirmation",
        "operation": cands[0]["operation"],
        "args": cands[0]["args"],
        "confirmation_prompt": "\n".join(prompt_lines),
        "candidates": cands,
        "registered_expressions": registered,
        "_user_text": stripped,
    }


async def _try_plan_workflow(engine, loop, lang, text, registered) -> "dict | None":
    """Composite workflow asks → fixed plan → stop-safe execution."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    workflow = None
    kwargs: dict = {}
    hm = nlp("plan_heisenberg_all", lang).search(stripped)
    if hm:
        workflow = "heisenberg_components"
        base = hm.group("base")
        if base:
            kwargs["base"] = base
        sm = nlp("plan_site", lang).search(stripped)
        if sm:
            kwargs["site"] = sm.group(1).translate(_ZENKAKU_DIGITS)
    elif nlp("plan_eigensystem", lang).search(stripped):
        workflow = "eigensystem"
        vm = nlp("plan_var", lang).search(stripped)
        if vm:
            kwargs["var"] = vm.group(1)
    if workflow is None:
        return None

    # 文法が `base` / `var` として読んだ名前は、対象ではない（上記）。
    target = _find_target(engine, stripped,
                          consumed=(kwargs.get("base"), kwargs.get("var")))
    if not target:
        return None

    try:
        plan = await loop.run_in_executor(
            None, partial(engine.create_workflow_plan, workflow, target,
                          **kwargs))
        plan = await loop.run_in_executor(
            None, partial(engine.execute_plan, plan.plan_id))
    except ValueError as exc:
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}

    ok_results = [r for r in plan.step_results if r.status == "success"]
    last = ok_results[-1] if ok_results else None
    summary_step = [s for s in engine.steps.values()
                    if s.kind.value == "plan_execution"][-1]
    return {
        "status": "completed" if plan.status == "completed" else "failed",
        "operation": "execute_plan",
        "plan_id": plan.plan_id,
        "expr_id": last.expr_id if last else None,
        "step_id": summary_step.step_id,
        "latex": (engine.expressions[last.expr_id].latex
                  if last and last.expr_id in engine.expressions else ""),
        "canonical": last.value if last else "",
        "operation_label": summary_step.visible_action,
        "plan_step_results": [r.model_dump(mode="json")
                              for r in plan.step_results],
        "registered_expressions": registered,
        **({} if plan.status == "completed"
           else {"error": summary_step.visible_action}),
    }


async def propose_replan(engine, loop, lang, plan_id, chat=None) -> dict:
    """Failure→replan loop (open_items C-Ext-5, 2026-07-18): the "🔄 別の方法で
    再計画" button on a stopped plan card → feed the failure into the Adviser
    → propose alternative candidates, returned in the SAME needs_confirmation
    shape as a fresh 「解法の候補」 ask (renderConfirmationHtml on the
    frontend, execute_plan_candidate on selection) — the chosen candidate
    still requires the researcher's confirmation before it runs (docs/23
    §2.4).

    Unlike retry_plan (B-3: the SAME plan re-executed once the researcher
    has fixed the root cause), this asks the LLM to propose a DIFFERENT
    approach.  Whether it can turn a failure into a sensible fix is,
    per docs/23 §11.1, "完全にLLM品質依存でCIでは検証不能" — real usage is
    the only evaluator; this function only owns the deterministic wiring.

    Unlike the NL-trigger functions above, this has no ``text``/regex match
    — it is called directly from a dedicated route (POST
    /plan/replan/{plan_id}), not through the interceptor pipeline, because
    its target (which plan, which failure) is already fully determined by
    the button that was clicked — nothing to infer from free text.
    """
    from notebook.runtime.plan_adviser import (
        _REPLAN_SYSTEM_PROMPT, build_failure_context, propose_plan_candidates,
    )

    plan = engine.plans.get(plan_id)
    if plan is None:
        return {"ok": False, "detail": t("plan.not_found", lang, plan_id=plan_id)}
    if plan.status not in ("stopped", "failed"):
        return {"ok": False, "detail": t("plan.not_retryable", lang,
                                         plan_id=plan_id, status=plan.status)}
    target = plan.target_expr_id
    if not target or target not in engine.expressions:
        return {"ok": False, "detail": t("plan.not_found", lang, plan_id=plan_id)}

    chat = chat or _resolve_adviser_chat()
    if chat is None:
        return {"ok": False, "detail": t("plan.adviser_unavailable", lang)}

    target_summary = _target_context(engine, target)
    narrative = _narrative_context(engine, plan.goal_text)
    goal_with_failure = f"{plan.goal_text}\n\n{build_failure_context(plan)}"

    valid, rejected = await loop.run_in_executor(None, partial(
        propose_plan_candidates, chat, goal_with_failure, target, target_summary,
        engine._validator, engine.session.primary_backend,
        engine._active_expr_ids(), lang,
        system_prompt=_REPLAN_SYSTEM_PROMPT, narrative_text=narrative))
    if not valid:
        return {"ok": False, "detail": t("plan.no_candidates", lang,
                                         n_rejected=len(rejected))}

    cands, prompt_lines = _candidates_to_response(
        plan.goal_text, target, valid, [], "plan.replan_prompt", lang)
    return {
        "ok": True,
        "status": "needs_confirmation",
        "operation": cands[0]["operation"],
        "args": cands[0]["args"],
        "confirmation_prompt": "\n".join(prompt_lines),
        "candidates": cands,
        "registered_expressions": [],
        "_user_text": plan.goal_text,
    }
