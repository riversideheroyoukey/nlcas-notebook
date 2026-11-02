"""Adviser consultation for plan candidates (C-Ext-5 priority 3).

The Adviser LLM proposes SOLUTION-METHOD candidates as JSON.  Its authority
ends there:

  - the JSON contract only lets it name operations and args — never CAS
    syntax (implementation memo §5.2);
  - every candidate is checked by ``validate_plan`` before it is shown;
    invalid candidates are dropped with a recorded reason;
  - the researcher picks; the chosen candidate is validated AGAIN when it
    becomes a SupervisorPlan (engine.create_candidate_plan).

The module is LLM-agnostic: ``propose_plan_candidates`` takes a
``chat(system, user) -> str`` callable, so tests inject a fake and the
deterministic skeleton (contract, gatekeeper, selection flow) is fully
covered without any model.
"""
from __future__ import annotations

import json
import re

from notebook.models.plan import PlanCandidate, PlanStep, SupervisorPlan
from notebook.runtime.plan_workflows import validate_plan

_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.S)

_SYSTEM_PROMPT = """You are the solution-method adviser of a theoretical-research notebook.
Given a goal and a catalog of available operations, propose 1-3 alternative
solution plans as a JSON array.  Rules:
- Use ONLY operations from the catalog, with exactly the listed required args.
- Expression-valued args may use the placeholders "$target" (the expression
  the user asked about) and "$prev" (the previous step's output).
- Output ONLY the JSON array, no prose.  Schema per candidate:
  {"title": str, "summary": str, "recommended_when": str, "risk": str,
   "steps": [{"operation": str, "args": {str: str}}]}
"""

# C-Ext-5 priority 3' (free-form planner): same JSON contract/schema as
# _SYSTEM_PROMPT above, reworded for a request that does not match any fixed
# workflow — one step-by-step plan toward the stated goal, not "alternatives".
# parse_candidates/validate_plan are unaware of which prompt produced the
# reply, so no other code needs to change.
_FREEFORM_SYSTEM_PROMPT = """You are the planning assistant of a theoretical-research notebook.
Given a goal and a catalog of available operations, propose a step-by-step
plan toward that goal as a JSON array (normally a single plan; propose a
second only if there are genuinely different reasonable approaches). Rules:
- Use ONLY operations from the catalog, with exactly the listed required args.
- Expression-valued args may use the placeholders "$target" (the expression
  the user asked about) and "$prev" (the previous step's output).
- Output ONLY the JSON array, no prose.  Schema per plan:
  {"title": str, "summary": str, "recommended_when": str, "risk": str,
   "steps": [{"operation": str, "args": {str: str}}]}
"""

# Failure→replan loop (C-Ext-5 open item, 2026-07-18): same JSON contract as
# _SYSTEM_PROMPT, reworded so the Adviser knows it is being asked for a
# DIFFERENT approach because a previous attempt failed, not a first pass.
_REPLAN_SYSTEM_PROMPT = """You are the solution-method adviser of a theoretical-research notebook.
A previous plan attempt for this goal failed partway through.  Given the
goal, the target expression, and details of what failed, propose 1-3
alternative solution plans that avoid the same failure, as a JSON array.
Rules:
- Use ONLY operations from the catalog, with exactly the listed required args.
- Expression-valued args may use the placeholders "$target" (the expression
  the user asked about) and "$prev" (the previous step's output).
- Output ONLY the JSON array, no prose.  Schema per candidate:
  {"title": str, "summary": str, "recommended_when": str, "risk": str,
   "steps": [{"operation": str, "args": {str: str}}]}
"""


def build_failure_context(plan: SupervisorPlan) -> str:
    """Compact failure summary appended to the goal for the replan prompt:
    which step(s) succeeded (so the Adviser doesn't need to guess what
    already worked), which one failed and why, and which were skipped as a
    result.  Built entirely from data already in ``plan.step_results`` —
    no re-inspection of the CAS or the expressions involved."""
    lines = ["A previous attempt at this goal failed partway through:"]
    for r in plan.step_results:
        if r.status == "success":
            lines.append(f"- step {r.step_id} ({r.operation}) succeeded, result: {r.value}")
        elif r.status == "failed":
            lines.append(f"- step {r.step_id} ({r.operation}) FAILED: {r.error}")
        elif r.status == "skipped":
            lines.append(f"- step {r.step_id} ({r.operation}) was skipped as a result")
    lines.append("Propose a genuinely different approach that avoids this specific failure.")
    return "\n".join(lines)


def build_operations_catalog(validator, backend: str) -> str:
    """Compact operation catalog for the prompt: name + required fields."""
    lines = []
    for name in sorted(validator.known_operations):
        spec = validator.get_spec(name)
        if not spec or backend not in spec.get("supported_backends", []):
            continue
        req = ", ".join(spec.get("required_fields", []))
        lines.append(f"- {name}({req})")
    return "\n".join(lines)


def build_user_prompt(goal_text: str, catalog: str,
                      target_summary: str, narrative_text: str = "") -> str:
    # C-Ext-5 §11.1 context-supply increment #2: declared symbols, assumptions,
    # and recent expression history, when the caller has any to offer — omitted
    # entirely for a fresh session so the prompt stays as lean as before.
    _narrative_block = (f"Session context (chronological, most recent last):\n"
                        f"{narrative_text}\n" if narrative_text else "")
    return (f"Goal: {goal_text}\n"
            f"Target expression ($target): {target_summary}\n"
            f"{_narrative_block}"
            f"Available operations:\n{catalog}\n"
            f"Respond with the JSON array only.")


def parse_candidates(raw_text: str) -> "list[PlanCandidate]":
    """Strictly parse the Adviser's reply into candidates.

    Anything malformed yields [] (or skips the malformed item) — a broken
    reply must never turn into a runnable plan.
    """
    if not raw_text:
        return []
    m = _JSON_BLOCK_RE.search(raw_text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    if not isinstance(data, list):
        return []

    candidates: list[PlanCandidate] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not item.get("title"):
            continue
        raw_steps = item.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            continue
        steps: list[PlanStep] = []
        ok = True
        for j, rs in enumerate(raw_steps, start=1):
            if (not isinstance(rs, dict) or not rs.get("operation")
                    or not isinstance(rs.get("args", {}), dict)):
                ok = False
                break
            steps.append(PlanStep(
                step_id=f"s{j}",                     # ids are OURS, not the LLM's
                operation=str(rs["operation"]),
                args={str(k): v for k, v in (rs.get("args") or {}).items()},
                description=str(rs.get("description", "")),
                depends_on=[f"s{j-1}"] if j > 1 else [],
            ))
        if not ok:
            continue
        candidates.append(PlanCandidate(
            candidate_id=f"c{i}",
            title=str(item.get("title", "")),
            summary=str(item.get("summary", "")),
            recommended_when=str(item.get("recommended_when", "")),
            risk=str(item.get("risk", "")),
            steps=steps,
        ))
    return candidates


def propose_plan_candidates(chat, goal_text: str, target_expr_id: str,
                            target_summary: str, validator, backend: str,
                            known_expr_ids: "set[str]", lang: str = "ja",
                            system_prompt: str = _SYSTEM_PROMPT,
                            narrative_text: str = "",
                            ) -> "tuple[list[PlanCandidate], list[tuple[str, str]]]":
    """Consult the Adviser and return (valid_candidates, rejected).

    ``rejected`` records (title, reason) for candidates that failed the
    validate_plan gate — kept for transparency (the researcher may want to
    know what the Adviser tried).

    ``system_prompt`` defaults to the "1-3 alternatives" framing; the
    free-form planner (ui/interpret_plan.py::_try_plan_freeform) passes
    _FREEFORM_SYSTEM_PROMPT instead — same JSON contract, different framing,
    zero duplication of parsing/validation.

    ``narrative_text`` is the C-Ext-5 §11.1 context-supply increment #2:
    declared symbols/assumptions/recent history, built by the caller via
    ui/interpret_plan.py::_narrative_context.  Optional and empty by default
    so existing callers/tests are unaffected.
    """
    catalog = build_operations_catalog(validator, backend)
    raw = chat(system_prompt, build_user_prompt(goal_text, catalog,
                                                target_summary, narrative_text))
    valid: list[PlanCandidate] = []
    rejected: list[tuple[str, str]] = []
    for cand in parse_candidates(raw or ""):
        probe = SupervisorPlan(plan_id="__probe__", goal_text=goal_text,
                               source="llm", target_expr_id=target_expr_id,
                               steps=cand.steps)
        try:
            validate_plan(probe, validator, backend, known_expr_ids, lang=lang)
        except ValueError as exc:
            rejected.append((cand.title, str(exc)))
            continue
        valid.append(cand)
    return valid, rejected


# ---------------------------------------------------------------------------
# Supervisor auto-selection (C-Ext-5 open item, 2026-07-18).
#
# The Adviser proposes; the Supervisor decides which ALREADY-VALIDATED
# candidate to run — a different LLM role from the one that proposed them,
# mirroring the existing Supervisor/Adviser split (llm_supervisor.py /
# llm_adviser.py).  This does not weaken core principle #4 (LLM is not the
# final authority for CAS generation): every candidate here already passed
# validate_plan before the Supervisor ever sees it, so selection only picks
# among options the deterministic gate already vouched for.  The chosen
# candidate STILL goes through the normal needs_confirmation flow — auto
# "selection" only removes the "which of N" decision, never the researcher's
# final approval (docs/23 §2.4: LLM-sourced plans always need confirmation).
# ---------------------------------------------------------------------------

_SELECT_SYSTEM_PROMPT = """You are the supervisor of a theoretical-research notebook,
choosing which ONE of several pre-validated solution-method candidates to run
for a goal.  Rules:
- You will be given the goal and a list of candidates, each with an id.
- Output ONLY a JSON object: {"selected_id": "<id>"}
- No prose, no markdown fences.
"""

_SELECT_JSON_RE = re.compile(r"\{.*\}", re.S)


def build_selection_prompt(goal_text: str, candidates: "list[PlanCandidate]") -> str:
    lines = [f"Goal: {goal_text}", "Candidates:"]
    for c in candidates:
        detail = f"- id={c.candidate_id}: {c.title} — {c.summary}"
        if c.recommended_when:
            detail += f" (recommended when: {c.recommended_when})"
        if c.risk:
            detail += f" (risk: {c.risk})"
        lines.append(detail)
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


def select_best_candidate(chat, goal_text: str,
                          candidates: "list[PlanCandidate]") -> "PlanCandidate | None":
    """Ask the Supervisor to pick ONE of the already-validated candidates.

    Returns None on anything it can't confidently resolve — no candidates,
    an unparseable/malformed reply, or a ``selected_id`` that doesn't match
    any given candidate — so the caller can fall back to showing all
    candidates for manual choice rather than guessing which one to run.
    A single candidate is returned directly without consulting the LLM
    (nothing to choose between).
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    raw = chat(_SELECT_SYSTEM_PROMPT, build_selection_prompt(goal_text, candidates))
    if not raw:
        return None
    m = _SELECT_JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    selected_id = data.get("selected_id")
    for c in candidates:
        if c.candidate_id == selected_id:
            return c
    return None
