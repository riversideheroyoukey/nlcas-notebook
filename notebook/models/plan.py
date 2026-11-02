"""Plan/Workflow data models (C-Ext-5): multi-step CAS plans.

One high-level instruction → a validated, step-by-step plan over EXISTING
common operations.  The plan layer adds NO new CAS capability — every step
goes through the same OperationValidator / run_operation path as a manual
instruction, so core principle #4 (the LLM is never the authority over CAS
commands) is preserved by construction: a plan only ever names operations
that the deterministic layer already knows how to build.

Field names follow docs/archive/experiments/theoretical_research_chatbot_implementation_memo.md
§5.1, adapted to this repo's pydantic style.  PlanCandidate (the Adviser
multi-candidate loop, priority 3) is defined below and used by
notebook/runtime/plan_adviser.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Placeholders allowed in expr-valued step args:
#   $target — the expression the plan was invoked on
#   $prev   — the output expression of the immediately preceding step
#   $s<k>   — the output expression of step with step_id "s<k>"
PLACEHOLDER_TARGET = "$target"
PLACEHOLDER_PREV = "$prev"


class PlanStep(BaseModel):
    """One step of a plan: an existing common operation with placeholder args."""
    step_id: str                       # "s1", "s2", …
    operation: str                     # must exist in common_operations schema
    args: dict = Field(default_factory=dict)
    description: str = ""              # human-readable purpose (i18n'd at build time)
    depends_on: list[str] = Field(default_factory=list)
    # Granularity policy (B-2): a step with present=False registers its output
    # as a HIDDEN expression — available for later steps via $prev/$sN and kept
    # in the plan record, but not shown on the timeline.  Long plans present
    # only their milestones.
    present: bool = True


class PlanCandidate(BaseModel):
    """One proposed solution method (Adviser consultation loop, priority 3).

    Candidates are PROPOSALS only: title/summary/risk are free text from the
    Adviser, but ``steps`` may name nothing beyond existing common operations
    — every candidate passes ``validate_plan`` before it is ever shown, and
    the chosen one is validated AGAIN when turned into a SupervisorPlan
    (defense in depth; the LLM never gains CAS authority).
    """
    candidate_id: str                  # "c1", "c2", …
    title: str
    summary: str = ""
    recommended_when: str = ""
    risk: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    auto_choice_score: "float | None" = None
    auto_choice_reason: str = ""


class PlanStepResult(BaseModel):
    """Execution record of one step (partial results survive a stop)."""
    step_id: str
    operation: str
    status: Literal["success", "failed", "skipped"]
    expr_id: str = ""                  # output expression, when produced
    value: str = ""                    # CAS value string, when produced
    error: str = ""                    # failure reason, when failed


class SupervisorPlan(BaseModel):
    """A validated multi-step plan and its execution state."""
    plan_id: str                       # "P1", "P2", …
    goal_text: str = ""
    source: Literal["workflow", "llm", "user"] = "workflow"
    workflow_name: str = ""
    target_expr_id: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    status: Literal[
        "draft", "validated", "running", "completed", "stopped", "failed"
    ] = "draft"
    step_results: list[PlanStepResult] = Field(default_factory=list)
    # Retry provenance (B-3): a retried plan is a NEW plan (immutable history);
    # this points back at the stopped plan it was seeded from.
    retried_from: str = ""
    created_at: datetime = Field(default_factory=_now)
