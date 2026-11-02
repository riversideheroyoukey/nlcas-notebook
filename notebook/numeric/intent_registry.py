"""Intent-kind registry for the faceted numeric spec (docs/24 §3.1, §8-1).

``IntentSpec.kind`` is an OPEN string: a novel intent is not an error, it just
routes to the LLM codegen path.  This registry records the KNOWN kinds and,
for each, its default method, the output shapes it typically produces, and
which codegen route can handle it today (deterministic template vs LLM).

Code-level (not a domains/ config file) on purpose: these are a small, fixed
technical vocabulary that drives dispatch — the same treatment as the plan
workflow registry (``plan_workflows``), not user-facing domain labels.  Adding
a kind is ONE entry here (data), never a schema change (docs/24 §6).
"""
from __future__ import annotations

from dataclasses import dataclass

CodegenRoute = str  # "deterministic" | "llm"


@dataclass(frozen=True)
class IntentKind:
    name: str
    default_method: str = "direct_eval"
    output_shapes: tuple[str, ...] = ()
    codegen: CodegenRoute = "llm"


# The six problem types the implementation memo enumerates (§7).
# parameter_sweep / plotting have a deterministic whitelist template (slice 1),
# and function_evaluation joined them in G1 of the typed graph (docs/26 §10 —
# the chaining workhorse); the rest are the LLM path's territory until a
# template is written.
_KINDS: dict[str, IntentKind] = {
    "cartesian_plot": IntentKind("cartesian_plot", "direct_eval", ("field",), "deterministic"),
    "function_evaluation": IntentKind(
        "function_evaluation", "direct_eval", ("scalar", "series"),
        "deterministic"),
    "parameter_sweep": IntentKind(
        "parameter_sweep", "direct_eval", ("series", "field"), "deterministic"),
    "plotting": IntentKind(
        "plotting", "direct_eval", ("series",), "deterministic"),
    "numeric_integration": IntentKind(
        "numeric_integration", "quadrature", ("scalar",), "llm"),
    "root_finding": IntentKind(
        "root_finding", "fixed_point", ("scalar", "trace"), "llm"),
    # G3 (docs/26 §10): promoted to the deterministic route — the template
    # evaluates the matrix canonical and diagonalizes in ONE fused script
    # (0 sweeps: eigenvalues+eigenvectors at a point; 1 sweep: eigenvalue
    # branches over the grid — the S-y avoided-crossing use case).
    "eigenvalue_problem": IntentKind(
        "eigenvalue_problem", "numpy_eig", ("vector", "matrix", "series"),
        "deterministic"),
    # G2 (docs/26 §10): bounded 1-D minimization — deterministic template via
    # scipy.optimize.minimize_scalar.  A registry addition, not a schema
    # change, exactly as docs/24 §6 prescribes.
    "minimize_1d": IntentKind(
        "minimize_1d", "brent_bounded", ("scalar", "scalar"), "deterministic"),
    # G2 (docs/26 §8): structured loop, region-fused into ONE sandbox run —
    # a multi-expression body evaluated per iteration with an update map and
    # a convergence criterion (the self-consistent physics loop: gap
    # equation, mean-field iteration).  Deterministic when the whole body is
    # whitelist-lowerable; a whitelist-external body raises and belongs to
    # the LLM path.
    "self_consistent_loop": IntentKind(
        "self_consistent_loop", "fixed_point", ("scalar", "trace"),
        "deterministic"),
}


def get_intent_kind(name: str) -> "IntentKind | None":
    return _KINDS.get(name)


def is_known_kind(name: str) -> bool:
    return name in _KINDS


def known_kinds() -> list[str]:
    return sorted(_KINDS)


def default_method_for(name: str) -> str:
    kind = _KINDS.get(name)
    return kind.default_method if kind else ""


def codegen_route(name: str) -> CodegenRoute:
    """'deterministic' if a whitelist template covers this kind, else 'llm'
    (also the answer for any unknown/novel kind)."""
    kind = _KINDS.get(name)
    return kind.codegen if kind else "llm"


def codegen_route_for_spec(spec):
    """Presentation structure takes precedence over legacy intent aliases."""
    if spec.presentation.kind == "cartesian":
        return "deterministic"
    return codegen_route(spec.intent.kind)
