"""Fixed workflow registry and plan validation (C-Ext-5, priorities 1–2).

Fixed workflows take precedence over any LLM planner (implementation memo
§5.6): a workflow is a deterministic function that assembles a SupervisorPlan
out of EXISTING common operations.  The first registered workflow is
``eigensystem`` — a thin wiring of the A-10 matrix operations
(characteristic_polynomial → eigenvalues → eigenvectors), no new CAS code.

``validate_plan`` checks a plan UPFRONT, before anything executes: every
operation must be known to the OperationValidator and supported by the
session backend, placeholder references may only point backwards, required
fields must be present, and args may not contain fields the operation does
not accept (mirrors OperationValidator.validate()'s own UNKNOWN_FIELDS check
— found missing here via S-ae's real-LLM run, docs/evaluations.md).  A plan
that fails validation never reaches the CAS.
"""
from __future__ import annotations

import re

from notebook.models.plan import (
    PLACEHOLDER_PREV, PLACEHOLDER_TARGET, PlanStep, SupervisorPlan,
)
from notebook.runtime.i18n import t

_STEP_REF_RE = re.compile(r"^\$s(\d+)$")

# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------

WORKFLOW_REGISTRY: dict[str, object] = {}


def register_workflow(name: str):
    def _deco(fn):
        WORKFLOW_REGISTRY[name] = fn
        return fn
    return _deco


@register_workflow("eigensystem")
def eigensystem_workflow(target_expr_id: str, lang: str = "ja",
                         var: str = "x") -> "list[PlanStep]":
    """H の固有状態を求める: 特性多項式 → 固有値 → 固有ベクトル.

    All three steps act on the ORIGINAL matrix ($target), not on each
    other's outputs — the chain value here is the plan bookkeeping, the
    single confirmation, and the summary step.
    """
    return [
        PlanStep(step_id="s1", operation="characteristic_polynomial",
                 args={"expr": PLACEHOLDER_TARGET, "var": var},
                 description=t("plan.step_charpoly", lang, var=var)),
        PlanStep(step_id="s2", operation="eigenvalues",
                 args={"expr": PLACEHOLDER_TARGET},
                 description=t("plan.step_eigenvalues", lang),
                 depends_on=["s1"]),
        PlanStep(step_id="s3", operation="eigenvectors",
                 args={"expr": PLACEHOLDER_TARGET},
                 description=t("plan.step_eigenvectors", lang),
                 depends_on=["s2"]),
    ]


@register_workflow("heisenberg_components")
def heisenberg_components_workflow(target_expr_id: str, lang: str = "ja",
                                   base: str = "S",
                                   site: str = "1") -> "list[PlanStep]":
    """各スピン成分のハイゼンベルク運動方程式: dS^α/dt = i[H, S^α] for α=x,y,z.

    The Larmor / spin-dynamics composite: three heisenberg_equation calls on
    the same Hamiltonian ($target), one per component.  Julia sessions only
    (heisenberg_equation is a julia_full operation — validate_plan enforces
    the backend, so invoking this in a Maxima session fails upfront).
    """
    steps = []
    for i, comp in enumerate(("x", "y", "z"), start=1):
        op_name = f"{base}{comp}_{site}"
        steps.append(PlanStep(
            step_id=f"s{i}", operation="heisenberg_equation",
            args={"operator": op_name, "hamiltonian": PLACEHOLDER_TARGET},
            description=t("plan.step_heisenberg", lang, component=op_name),
            depends_on=[f"s{i-1}"] if i > 1 else [],
        ))
    return steps


def build_workflow_plan(workflow_name: str, plan_id: str, target_expr_id: str,
                        lang: str = "ja", **kwargs) -> SupervisorPlan:
    builder = WORKFLOW_REGISTRY.get(workflow_name)
    if builder is None:
        raise ValueError(t("plan.workflow_not_found", lang,
                           workflow=workflow_name))
    steps = builder(target_expr_id, lang=lang, **kwargs)
    return SupervisorPlan(
        plan_id=plan_id,
        goal_text=t(f"plan.goal_{workflow_name}", lang),
        source="workflow",
        workflow_name=workflow_name,
        target_expr_id=target_expr_id,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Upfront validation
# ---------------------------------------------------------------------------

def validate_plan(plan: SupervisorPlan, validator, backend: str,
                  known_expr_ids: "set[str]", lang: str = "ja") -> None:
    """Raise ValueError (translated) on the first problem; return None if OK."""
    if not plan.steps:
        raise ValueError(t("plan.empty", lang, plan_id=plan.plan_id))
    if plan.target_expr_id and plan.target_expr_id not in known_expr_ids:
        raise ValueError(t("plan.target_missing", lang,
                           expr_id=plan.target_expr_id))

    seen_ids: set[str] = set()
    for idx, step in enumerate(plan.steps):
        spec = validator.get_spec(step.operation)
        if spec is None:
            raise ValueError(t("plan.unknown_operation", lang,
                               step_id=step.step_id, operation=step.operation))
        if backend not in spec.get("supported_backends", []):
            raise ValueError(t("plan.backend_unsupported", lang,
                               step_id=step.step_id, operation=step.operation,
                               backend=backend))
        for field in spec.get("required_fields", []):
            if field not in step.args or step.args[field] in ("", None):
                raise ValueError(t("plan.missing_field", lang,
                                   step_id=step.step_id,
                                   operation=step.operation, field=field))
        allowed_fields = (frozenset(spec.get("required_fields", []))
                          | frozenset(spec.get("optional_fields", [])))
        unexpected = sorted(k for k in step.args if k not in allowed_fields)
        if unexpected:
            raise ValueError(t("plan.unknown_field", lang,
                               step_id=step.step_id, operation=step.operation,
                               fields=", ".join(unexpected)))
        for value in step.args.values():
            if not isinstance(value, str) or not value.startswith("$"):
                continue
            if value == PLACEHOLDER_TARGET:
                if not plan.target_expr_id:
                    raise ValueError(t("plan.target_missing", lang,
                                       expr_id=PLACEHOLDER_TARGET))
                continue
            if value == PLACEHOLDER_PREV:
                if idx == 0:
                    raise ValueError(t("plan.bad_placeholder", lang,
                                       step_id=step.step_id, ref=value))
                continue
            m = _STEP_REF_RE.match(value)
            if m:
                if f"s{m.group(1)}" not in seen_ids:
                    raise ValueError(t("plan.bad_placeholder", lang,
                                       step_id=step.step_id, ref=value))
                continue
            raise ValueError(t("plan.bad_placeholder", lang,
                               step_id=step.step_id, ref=value))
        for dep in step.depends_on:
            if dep not in seen_ids:
                raise ValueError(t("plan.bad_placeholder", lang,
                                   step_id=step.step_id, ref=dep))
        seen_ids.add(step.step_id)
