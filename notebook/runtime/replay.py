from __future__ import annotations
from dataclasses import dataclass, field

from notebook.models.branch import Branch
from notebook.models.expression import Expression
from notebook.models.step import Step, StepKind
from notebook.runtime.maxima_adapter import MaximaAdapter


@dataclass
class ReplayResult:
    """Expression table rebuilt after replaying a branch's history."""
    expr_values: dict[str, str]   # display_id -> canonical value from Maxima
    replayed_steps: int
    skipped_steps: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def compute_lineage(
    branch_id: str,
    steps: dict[str, Step],
    branches: dict[str, Branch],
) -> list[Step]:
    """Return steps in chronological order for the given branch.

    Walks backwards from the branch head through parent_step_ids links,
    crossing branch boundaries so that forked branches inherit the ancestor
    lineage automatically. The returned list is root-first.

    Args:
        branch_id: Branch to compute lineage for.
        steps: All known steps keyed by step_id.
        branches: All known branches keyed by branch_id.
    """
    branch = branches.get(branch_id)
    if branch is None or branch.head_step_id is None:
        return []

    ordered: list[Step] = []
    seen: set[str] = set()
    current_id: str | None = branch.head_step_id

    while current_id is not None:
        if current_id in seen:
            break  # cycle guard
        seen.add(current_id)

        step = steps.get(current_id)
        if step is None:
            break
        ordered.append(step)
        # Follow first parent (linear replay; DAG merge handled in future)
        current_id = step.parent_step_ids[0] if step.parent_step_ids else None

    ordered.reverse()
    return ordered


def replay(
    branch_id: str,
    steps: dict[str, Step],
    branches: dict[str, Branch],
    expressions: dict[str, Expression],
    adapter: MaximaAdapter,
) -> ReplayResult:
    """Rebuild expression values by replaying a branch's saved history.

    Rules:
    - Only state_affecting steps are re-executed.
    - expression_registration steps restore the canonical value from the
      saved Expression object (no CAS call needed).
    - cas_operation steps re-execute their saved rendered_command via the
      adapter. Natural language (user_text) is never re-interpreted.
    - Errors are collected without halting replay so that partial state
      can still be recovered.

    Args:
        branch_id: Branch to replay.
        steps: All saved steps keyed by step_id.
        branches: All saved branches keyed by branch_id.
        expressions: Saved expression objects keyed by display_id.
        adapter: Running MaximaAdapter instance (must be started before call).
    """
    lineage = compute_lineage(branch_id, steps, branches)

    expr_values: dict[str, str] = {
        display_id: expr.canonical
        for display_id, expr in expressions.items()
    }
    replayed = 0
    skipped = 0
    errors: list[str] = []

    for step in lineage:
        if not step.state_affecting:
            skipped += 1
            continue

        if step.kind == StepKind.expression_registration:
            # Canonical value already loaded from the expressions table.
            replayed += 1
            continue

        if step.rendered_command is None:
            skipped += 1
            continue

        result = adapter.execute(step.rendered_command)

        if result.ok:
            for ref in step.output_refs:
                expr_values[ref] = result.value
        else:
            errors.append(
                f"Step {step.step_id} ({step.visible_action}): {result.error}"
            )

        replayed += 1

    return ReplayResult(
        expr_values=expr_values,
        replayed_steps=replayed,
        skipped_steps=skipped,
        errors=errors,
    )
