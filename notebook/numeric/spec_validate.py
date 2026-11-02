"""Pre-execution gatekeeper for the faceted numeric spec (docs/24 §7).

``validate_numeric_spec`` returns the LIST of structural problems in a spec
(empty == valid).  A list, not raise-on-first, because the spec-elicitation
ChatBot wants to show the researcher every problem at once; the caller raises
if it prefers (the plan layer's ``validate_plan`` raises — same spirit, here we
defer the policy to the caller).

Each issue is ``{"code": <i18n key>, **kwargs}`` so the bridge/UI can translate
via ``t(code, lang, **kwargs)``.  The symbol-membership check is best-effort:
it only runs when the caller supplies the expression's free symbols (a
convergence/eigen spec may reference an expression the numpy whitelist cannot
lower, so free symbols aren't always available).
"""
from __future__ import annotations

from notebook.models.numeric import NumericSpec

# Output axes that are NOT input variables: a convergence trace is "over
# [iteration]", where iteration is an implicit run-time axis, not a declared
# input.  Referencing one of these in OutputSpec.over is legal.
_IMPLICIT_AXES = frozenset({"iteration"})


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _check_sweep(name: str, spec: dict, require_points: bool = True) -> list[dict]:
    lo, hi, pts = spec.get("min"), spec.get("max"), spec.get("points")
    issues: list[dict] = []
    if not _is_number(lo) or not _is_number(hi) or float(lo) >= float(hi):
        issues.append({"code": "numeric.spec.bad_sweep_range", "name": name})
    if not require_points:
        # minimize_1d: the "sweep" input carries BOUNDS, not a grid — a
        # points count is meaningless there (docs/26 §10).
        return issues
    try:
        if int(pts) < 2:
            issues.append({"code": "numeric.spec.too_few_points", "name": name})
    except (TypeError, ValueError):
        issues.append({"code": "numeric.spec.too_few_points", "name": name})
    return issues


def _check_method(method) -> list[dict]:
    issues: list[dict] = []
    params = method.params or {}
    if "tol" in params and (not _is_number(params["tol"]) or float(params["tol"]) <= 0):
        issues.append({"code": "numeric.spec.bad_method_param", "param": "tol"})
    if "max_iter" in params:
        try:
            if int(params["max_iter"]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append({"code": "numeric.spec.bad_method_param", "param": "max_iter"})
    return issues


def validate_numeric_spec(spec: NumericSpec,
                          free_symbols: "set[str] | None" = None) -> list[dict]:
    """Return the structural issues in *spec* (empty == valid).

    ``free_symbols`` — the expression's free symbols, when known; enables the
    "input name is actually a symbol of the expression" check.  Omit it when
    the expression isn't whitelist-lowerable (that check is then skipped).
    """
    issues: list[dict] = []
    input_names = {v.name for v in spec.inputs}

    if free_symbols is not None:
        for v in spec.inputs:
            if v.name not in free_symbols and not (spec.presentation.kind == "cartesian" and v.role == "sweep"):
                issues.append({"code": "numeric.spec.unknown_symbol", "name": v.name})

    for v in spec.inputs:
        sp = v.spec or {}
        # A value_from binding (docs/26 §5) defers the value to an upstream
        # artifact — the ENGINE resolves and type-checks it at fork time
        # (artifact existence/kind need engine state this pure validator
        # doesn't have), so the literal-number checks don't apply yet.
        bound = "value_from" in sp
        if v.role == "sweep":
            issues += _check_sweep(v.name, sp,
                                   require_points=(spec.intent.kind
                                                   != "minimize_1d"))
        elif v.role == "fixed":
            if not bound and not _is_number(sp.get("value")):
                issues.append({"code": "numeric.spec.bad_fixed_value", "name": v.name})
        elif v.role == "free":
            init = sp.get("initial")
            if not bound and init is not None and not _is_number(init):
                issues.append({"code": "numeric.spec.bad_free_initial", "name": v.name})

    if not spec.outputs:
        issues.append({"code": "numeric.spec.no_outputs"})
    for o in spec.outputs:
        for ref in o.over:
            if ref not in input_names and ref not in _IMPLICIT_AXES:
                issues.append({"code": "numeric.spec.dangling_over",
                               "output": o.name, "ref": ref})

    from notebook.numeric.plot_spec import is_cartesian, read_plot
    from notebook.runtime.shared_parser import UnknownNotationError
    if is_cartesian(spec) or spec.intent.kind == "cartesian_plot":
        try:
            read_plot(spec)
        except UnknownNotationError as exc:
            issues.append({"code": exc.i18n_key, **exc.fmt_kwargs})
    issues += _check_method(spec.method)
    issues += _check_loop(spec, input_names)
    return issues


def _check_loop(spec: NumericSpec, input_names: set) -> list[dict]:
    """Structural checks specific to self_consistent_loop (docs/26 §8):
    a non-empty body, a non-empty update map, and every update edge
    ``state_var ← body_output`` must connect a free input to a body name."""
    if spec.intent.kind != "self_consistent_loop":
        return []
    issues: list[dict] = []
    params = spec.method.params or {}
    body = params.get("body") or []
    if not body or not all(isinstance(b, dict) and b.get("name")
                           and b.get("expr_id") for b in body):
        issues.append({"code": "numeric.spec.loop_no_body"})
    update = params.get("update") or {}
    if not update:
        issues.append({"code": "numeric.spec.loop_no_update"})
    free_names = {v.name for v in spec.inputs if v.role == "free"}
    body_names = {b.get("name") for b in body if isinstance(b, dict)}
    for var, src in update.items():
        if var not in free_names or src not in body_names:
            issues.append({"code": "numeric.spec.loop_bad_update",
                           "var": str(var), "source": str(src)})
    return issues
