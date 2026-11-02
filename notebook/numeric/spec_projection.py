"""Flat <-> faceted numeric-spec projections (docs/24 §5).

The faceted ``NumericSpec`` (intent/inputs/outputs/method/presentation) is the
extensible IR the slice-2 spec-elicitation ChatBot fills in.  Slice 1's
deterministic path, however, reads the FLAT ``NumericTask`` fields
(problem_type / target_quantity / fixed_values / sweep_parameters).  These two
pure projections bridge them so:

  * a slice-1 (flat) task gains a faceted view for free (``flat_to_spec``), and
  * a faceted spec that IS parameter-sweep-shaped can be lowered back to the
    flat fields (``spec_to_flat``) and fed to the UNCHANGED
    ``templates.generate_code`` — no rewrite of the deterministic template yet.

Only the parameter_sweep / plotting family is flat-representable; a
convergence / eigenvalue spec is not (``is_flat_representable`` says so).  The
projections are deliberately dumb and total — validation lives elsewhere
(``validate_numeric_spec``, a later commit).
"""
from __future__ import annotations

from notebook.models.numeric import (
    IntentSpec, InputVar, MethodSpec, NumericSpec, NumericTask,
    OutputSpec, PresentationSpec,
)

# Output shape implied by the number of swept variables (docs/24 §3.3):
# 0 swept -> a single value, 1 -> a curve, 2 -> a field.  Higher dims have no
# standard plot, so they carry no implied presentation.
_SWEEP_SHAPE = {0: "scalar", 1: "series", 2: "field"}
_SWEEP_PRESENTATION = {0: "scalar", 1: "line", 2: "heatmap"}


def flat_to_spec(task: NumericTask) -> NumericSpec:
    """Derive a faceted spec from a flat (slice-1) task's fields.

    Deterministic ordering: swept inputs first (in ``sweep_parameters`` order),
    then fixed inputs sorted by name — matching how ``templates.generate_code``
    emits fixed lines, so the round trip is stable.
    """
    sweep_names = list(task.sweep_parameters.keys())
    inputs: list[InputVar] = [
        InputVar(name=name, role="sweep", spec=dict(task.sweep_parameters[name]))
        for name in sweep_names
    ]
    inputs += [
        InputVar(name=name, role="fixed", spec={"value": task.fixed_values[name]})
        for name in sorted(task.fixed_values)
    ]

    n_sweep = len(sweep_names)
    target = task.target_quantity or "y"
    outputs = [OutputSpec(
        name=target,
        shape=_SWEEP_SHAPE.get(n_sweep, "series"),
        over=list(sweep_names),
        role="primary",
    )]

    return NumericSpec(
        intent=IntentSpec(kind=task.problem_type, goal=""),
        inputs=inputs,
        outputs=outputs,
        method=MethodSpec(name="direct_eval", params={}),
        presentation=PresentationSpec(
            kind=_SWEEP_PRESENTATION.get(n_sweep, "line"), options={}),
    )


def is_flat_representable(spec: NumericSpec) -> bool:
    """True if the deterministic template can consume this spec via
    ``spec_to_flat``: direct evaluation, inputs only sweep/fixed, at most one
    swept variable (one → sweep/plot; zero → point evaluation, docs/26 §10),
    a single primary output.  Anything else (free/boundary inputs, a solver
    method, 2-D, multiple outputs) is the LLM path's job.
    """
    if spec.method.name not in ("", "direct_eval"):
        return False
    roles = {v.role for v in spec.inputs}
    if not roles <= {"sweep", "fixed"}:
        return False
    n_sweep = sum(1 for v in spec.inputs if v.role == "sweep")
    if n_sweep > 1:
        return False
    return len(spec.outputs) <= 1


def spec_to_flat(spec: NumericSpec) -> dict:
    """Project a faceted spec back to the flat NumericTask fields the
    deterministic template reads.  Best-effort and total: callers gate on
    ``is_flat_representable`` first when they need the template to succeed.
    """
    fixed_values = {
        v.name: v.spec.get("value") for v in spec.inputs if v.role == "fixed"
    }
    sweep_parameters = {
        v.name: dict(v.spec) for v in spec.inputs if v.role == "sweep"
    }
    kind = spec.intent.kind
    if spec.presentation.kind == "cartesian":
        kind = "plotting"
    if kind in ("parameter_sweep", "plotting", "function_evaluation",
                "minimize_1d", "self_consistent_loop", "eigenvalue_problem"):
        problem_type = kind    # minimize_1d: the "sweep" input carries bounds
    elif not sweep_parameters:
        problem_type = "function_evaluation"   # 0 sweeps = point evaluation
    else:
        problem_type = "parameter_sweep"
    target_quantity = spec.outputs[0].name if spec.outputs else ""
    return {
        "problem_type": problem_type,
        "target_quantity": target_quantity,
        "fixed_values": {k: v for k, v in fixed_values.items() if v is not None},
        "sweep_parameters": sweep_parameters,
    }
