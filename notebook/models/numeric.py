"""Numeric-layer data models (N-1): NumericTask / NumericArtifact / NumericExecution.

The numeric layer is a SEPARATE, cross-referenced layer beside the theory
layer (docs/archive/experiments/theory_to_numeric_design_principles.md): a theory
expression is FORKED into a NumericTask that carries its provenance
(expression ids + canonical snapshot, assumptions, branch), deterministic
code generation produces a code artifact, the sandbox runs it, and the
result flows back into the notebook timeline with full provenance.

Specs live here as structured objects — never only in conversation logs
(implementation memo §12).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
import uuid

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_execution_id() -> str:
    # Timestamp (not a random UUID) so directory names sort chronologically
    # and "which run is newer" is readable at a glance; microsecond precision
    # keeps back-to-back executions of the same task from colliding.
    return f"nx_{_now().strftime('%Y%m%dT%H%M%S%f')}"


# problem types with a deterministic whitelist template (implementation memo
# §7 lists six; phase 1 implemented the two the memo prioritizes — §11; G1 of
# the typed graph added function_evaluation, the chaining workhorse, and G2
# added minimize_1d: docs/26 §10).
PHASE1_PROBLEM_TYPES: frozenset[str] = frozenset(
    {"parameter_sweep", "plotting", "function_evaluation", "minimize_1d",
     "self_consistent_loop", "eigenvalue_problem"})

ProblemType = Literal[
    "function_evaluation", "numeric_integration", "root_finding",
    "eigenvalue_problem", "parameter_sweep", "plotting", "minimize_1d",
    "self_consistent_loop",
]

TaskStatus = Literal["draft", "code_generated", "executed", "failed"]


# ---------------------------------------------------------------------------
# Faceted numeric spec (docs/24) — the extensible IR the spec-elicitation
# "numeric ChatBot" (slice 2) fills in.  The computation's STRUCTURE is
# carried by orthogonal facets (intent / inputs / outputs / method /
# presentation), so a new computation type is DATA (a registry entry + input/
# output/method combos), not schema surgery.  Flexibility lives in the OPEN
# values (kind / shape / *.params are open strings/dicts — an unknown kind
# falls to the LLM codegen path); structure lives in the fixed frame (so the
# four design-principle benefits — solver-swap / code-regen / porting /
# cross-branch comparison — hold).  Additive: NumericTask.spec is Optional and
# absent in legacy tasks/save files.
# ---------------------------------------------------------------------------

# A variable's role is genuinely bounded (how it participates in the
# computation), so this stays a CLOSED set — validation and codegen dispatch
# key off it (docs/24 §8-2; extend by adding ONE role when a concrete scenario
# needs it, never by opening the type).
InputRole = Literal["sweep", "fixed", "free", "boundary"]


class IntentSpec(BaseModel):
    """What the researcher wants to compute.

    ``kind`` is an OPEN string backed by a registry (docs/24 §3.1): known
    kinds (parameter_sweep, root_finding, eigenvalue_problem, …) declare their
    default method / expected output shapes / codegen path; an unknown kind is
    not an error — it routes to the LLM codegen path.  ``goal`` is the
    LLM-verbalized natural-language objective (human-readable provenance).
    """
    kind: str = ""
    goal: str = ""


class InputVar(BaseModel):
    """One independent variable and how it participates.

    ``spec`` is role-dependent and open: sweep → {min,max,points,scale?};
    fixed → {value}; free → {initial}; boundary → {at,value}.  The COUNT of
    sweep inputs is the dimensionality — one sweep = 1-D, two = 2-D — so a 2-D
    plot needs no new schema, just a second sweep input (docs/24 §3.2).
    """
    name: str
    role: InputRole = "fixed"
    spec: dict = Field(default_factory=dict)


class OutputSpec(BaseModel):
    """One output quantity and its SHAPE (the expected_output_kind that was
    missing before, docs/24 §3.3).  ``outputs`` being a LIST is the point:
    a diagonalization emits [eigenvalues: vector, eigenvectors: matrix].
    ``over`` references input names (series over [x]; field over [x,y]; trace
    over [iteration]) so the axes' meaning survives as structure.
    """
    name: str
    shape: str = "scalar"    # scalar | series | field | vector | matrix | trace | … (open)
    over: list[str] = Field(default_factory=list)
    role: str = "primary"    # primary | diagnostic | … (open)


class MethodSpec(BaseModel):
    """Solver + stopping/tolerance.  ``params`` is open (tol, max_iter, step,
    seed …) so a new solver needs no schema change — "iterate to convergence"
    is method={name:"fixed_point", params:{tol:…, max_iter:…}} (docs/24 §3.4).
    """
    name: str = "direct_eval"
    params: dict = Field(default_factory=dict)


class PresentationSpec(BaseModel):
    """How the result is shown — deliberately separate from output shape
    (docs/24 §3.5): the same 2-D field can be a heatmap/contour/slice."""
    kind: str = "line"       # line | heatmap | contour | surface | table | scalar | convergence | … (open)
    options: dict = Field(default_factory=dict)


class NumericSpec(BaseModel):
    """The faceted numeric-computation IR (docs/24 §3).  Provenance stays on
    NumericTask; this describes only the computation."""
    intent: IntentSpec = Field(default_factory=IntentSpec)
    inputs: list[InputVar] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    method: MethodSpec = Field(default_factory=MethodSpec)
    presentation: PresentationSpec = Field(default_factory=PresentationSpec)


class NumericTask(BaseModel):
    """A numeric-computation spec forked from theory expressions.

    ``source_canonical`` / ``source_latex`` snapshot the expression at fork
    time so the provenance survives later edits to the theory layer.
    """
    task_id: str
    source_expression_ids: list[str] = Field(default_factory=list)
    source_canonical: str = ""
    source_latex: str = ""
    source_assumption_ids: list[str] = Field(default_factory=list)
    source_branch_id: str = "main"
    problem_type: ProblemType = "parameter_sweep"
    target_quantity: str = ""
    parameters: list[str] = Field(default_factory=list)
    fixed_values: dict[str, str] = Field(default_factory=dict)
    # sweep var -> {"min": float, "max": float, "points": int}
    sweep_parameters: dict[str, dict] = Field(default_factory=dict)
    boundary_conditions: dict = Field(default_factory=dict)
    approximations: list[str] = Field(default_factory=list)
    solver_candidates: list[str] = Field(default_factory=list)
    # Faceted IR (docs/24).  Additive/optional: None for legacy tasks and for
    # slice-1 parameter_sweep forks, which still populate the flat fields
    # above.  spec_projection.py maps flat <-> faceted so the deterministic
    # template keeps consuming the flat form untouched.
    spec: Optional[NumericSpec] = None
    # Typed-graph chaining (docs/26 §5): artifacts whose values were bound
    # into this task's inputs (InputVar.spec {"value_from": artifact_id}).
    # These refs ARE the artifact→task dependency edges — the graph is
    # derived from them, never stored separately (docs/26 §6).
    input_artifact_refs: list[str] = Field(default_factory=list)
    # Content hash of (source_canonical + snapshotted assumption canonicals)
    # at fork time (docs/26 §7).  Stale detection compares it against the
    # CURRENT hash of the same expression — no Expression versioning needed,
    # the snapshot stays the frozen form.  "" for legacy tasks (=> unknown,
    # never reported stale).
    source_hash: str = ""
    # Structured-loop body (docs/26 §8): the ordered per-iteration assignments
    # of a self_consistent_loop, snapshotted at fork time like
    # source_canonical.  Each entry: {"expr_id", "name", "canonical"} — the
    # fused template evaluates them in order, then applies the update map
    # (spec.method.params["update"]).  Empty for every other problem type.
    source_body: list[dict] = Field(default_factory=list)
    # Frozen expressions referenced by declarative plot layers.
    plot_sources: list[dict] = Field(default_factory=list)
    status: TaskStatus = "draft"
    created_by: str = "user"
    created_at: datetime = Field(default_factory=_now)


class NumericArtifact(BaseModel):
    """A product of the numeric layer: code, result, figure, table, or log.

    Small text artifacts (code / result JSON / log) live inline in
    ``content``; file artifacts (figures, tables) live on disk under
    ``<saved session stem>_numeric/`` and carry their ``path``.
    """
    artifact_id: str = Field(default_factory=lambda: f"na_{uuid.uuid4().hex[:8]}")
    task_id: str
    artifact_type: Literal["code", "result", "figure", "table", "log"]
    content: str = ""
    path: str = ""
    metadata: dict = Field(default_factory=dict)
    # Typed value fields (docs/26 §3) — set only for artifacts that can feed
    # a downstream task; "" / empty for code/figure/log artifacts.  NOT named
    # artifact_type (that field is the FILE kind above); value_kind is the
    # MATH kind and shares its vocabulary with OutputSpec.shape so the
    # spec-declared and actually-produced forms can be cross-checked.
    value_kind: str = ""      # scalar | series | vector | matrix | trace | …(open)
    value_type: str = ""      # real_float64 | complex_float64 | integer | …(open)
    shape: list = Field(default_factory=list)   # [] scalar | [n] | [m,n] | …
    ordering: dict = Field(default_factory=dict)  # only when order carries meaning
    value: Any = None         # small inline value; large values stay in `path`
    created_at: datetime = Field(default_factory=_now)


class NumericExecution(BaseModel):
    """One sandbox run of a task's code artifact."""
    execution_id: str = Field(default_factory=_new_execution_id)
    task_id: str
    code_artifact_id: str
    runtime: str = ""                # interpreter path
    package_info: dict[str, str] = Field(default_factory=dict)
    parameter_snapshot: dict = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    return_status: int = -1
    timed_out: bool = False
    duration_s: float = 0.0
    result_artifact_ids: list[str] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.return_status == 0 and not self.timed_out
