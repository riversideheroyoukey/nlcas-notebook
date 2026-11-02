from __future__ import annotations
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class StepKind(str, Enum):
    # --- Priority A ---
    cas_operation = "cas_operation"
    expression_registration = "expression_registration"
    assumption_add = "assumption_add"
    assumption_remove = "assumption_remove"
    llm_advisory = "llm_advisory"
    # --- Priority B ---
    user_input = "user_input"
    llm_supervision = "llm_supervision"
    ambiguity_resolution = "ambiguity_resolution"
    # --- Priority C ---
    manual_note = "manual_note"
    # --- Numeric layer (N-1) ---
    numeric_fork = "numeric_fork"      # theory expression → NumericTask
    numeric_result = "numeric_result"  # sandbox result refluxed into the note
    # --- Plan layer (C-Ext-5) ---
    plan_execution = "plan_execution"  # multi-step plan summary (per-step CAS
                                       # steps are recorded normally alongside)
    # --- Live branch (fork-in-session) ---
    branch_fork = "branch_fork"        # a new live Branch was created from a step
    # --- Cross-backend handoff (SCALAR-HANDOFF, docs/34 §5) ---
    scalar_handoff = "scalar_handoff"  # a scalar expression left for another
                                       # backend's session (a fork, not a switch)


class StepStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"


class Step(BaseModel):
    step_id: str
    kind: StepKind
    branch_id: str
    parent_step_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_text: Optional[str] = None
    visible_action: str
    operation: Optional[str] = None
    operation_args: dict = Field(default_factory=dict)
    backend: Optional[str] = None
    rendered_command: Optional[str] = None
    show_command_to_user: bool = False
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    state_affecting: bool = True
    status: StepStatus = StepStatus.pending
    user_confirmation: Optional[str] = None
    #: Which rule-based interceptor claimed the input that produced this step
    #: (``ui/interpret_registry``).  ``None`` means no interceptor did — the
    #: general/LLM path, a direct engine call, or a session saved before this
    #: field existed (additive key: an older file loads with None).
    #:
    #: 24 interceptors branch on the RAW text before the general pipeline sees
    #: anything, which is correct design — a lattice sum has to be claimed
    #: before semantic resolution — but until now *which* one claimed an input
    #: could not be asked, from a test or afterwards.  That cost real work:
    #: on 2026-08-09 a defect was filed against a path no researcher takes,
    #: because the measurement went through ``register_expression`` and missed
    #: the interceptor entirely (docs/handoffs/one_question_many_answers_handoff.md §2.2).
    route: Optional[str] = None
