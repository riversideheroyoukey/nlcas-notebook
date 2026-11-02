"""Preparing a scalar expression to continue in another backend's session.

Pure logic only — no engine, no CAS, no file I/O — so the two decisions that
matter can be tested without starting either backend:

1. **Is this expression allowed to cross?**  Only if nothing operator-valued is
   left in it.  A handoff carries the researcher's own writing to a session
   whose CAS has no operator algebra, so an operator that slipped through would
   arrive as a meaningless symbol and evaluate to something plausible.  The
   refusal is deliberate and narrow: the researcher chooses the expression, so
   detecting a pure c-number sub-expression automatically (BP-4's hard part) is
   not needed here — checking what they chose is.

2. **What must the target session be told first?**  The declarations, WITH
   their signs.  Measured 2026-07-31 during the S-c handoff (open_items
   SIGN-DIALOG-ASSUME): declaring `omega_D` as a real without its sign left
   Maxima asking "Is omega_D positive, negative or zero?" mid-integral, which
   times out and takes the session with it.  Type alone is not enough to hand
   over a usable expression.

Reference: docs/developer_manuals/34 §5 (design + the manual measurement this
mechanises).
"""
from __future__ import annotations

from notebook.models.handoff import HandoffDeclaration


#: Quantity kinds that cannot cross into a scalar CAS session.
_OPERATOR_KINDS = frozenset({"operator"})


class HandoffRefused(Exception):
    """Raised when the chosen expression may not cross.

    Carries an i18n key like the parser's ``UnknownNotationError`` so the
    caller can translate it with the session language.
    """

    def __init__(self, i18n_key: str, **fmt_kwargs: object) -> None:
        super().__init__(i18n_key)
        self.i18n_key = i18n_key
        self.fmt_kwargs = fmt_kwargs


def operator_symbols(names: "set[str] | list[str]", context) -> "list[str]":
    """Names in *names* that the source session knows to be operator-valued."""
    found: list[str] = []
    for name in sorted(names):
        prop = context.get(name) if context else None
        if prop is None:
            continue
        if getattr(prop.quantity, "kind", None) in _OPERATOR_KINDS:
            found.append(name)
            continue
        stats = getattr(prop, "statistics", None)
        # A symbol declared through the particle-name path carries its statistics
        # rather than kind="operator" — both mean "not a c-number".
        if stats is not None and getattr(stats, "kind", None) in ("fermi",):
            found.append(name)
    return found


def split_export_form(text: str) -> "tuple[str, str]":
    """Split ``LABEL = VALUE`` at the first TOP-LEVEL ``=``; else ``("", text)``.

    What crosses a handoff is the VALUE.  A result can arrive wearing a label,
    and the two are not the same kind of thing:

    * ``H = J x y`` — an H-1 definition.  The label is a symbol NAME, and it is
      worth carrying: the target session gets the equation back.
    * ``c(-k,d) c(k,u) = uv/(u²+v²)²`` — a vacuum expectation value.  The label
      says what the value is OF, and it is an operator product, which is exactly
      what a scalar CAS has no meaning for.

    Measured 2026-08-03 (S-c re-run): without this split the operator check read
    the LABEL and refused the crossing — an expectation value is a c-number by
    definition, so the one expression the feature exists to carry was the one it
    turned away.  Splitting is not a loosening of the check: the check still runs,
    on the value, which is the thing that has to be computable in the target.

    Top-level means outside braces/brackets/parens, so a `\\sum_{k=1}` or a
    subscripted `_{i=j}` in the value is not mistaken for the separator.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0:
            # not <=, >=, !=, ==
            if i and text[i - 1] in "<>!=":
                continue
            if i + 1 < len(text) and text[i + 1] == "=":
                continue
            return text[:i].strip(), text[i + 1:].strip()
    return "", text


def collect_declarations(names: "set[str] | list[str]", context
                         ) -> "tuple[list[HandoffDeclaration], list[str]]":
    """Split *names* into (declarations to replay, names with no declaration).

    Undeclared names are NOT an error: `x`, `t`, `k` are ordinary free
    variables and both CASes handle them natively.  They are reported so the
    caller can ask about the ones that matter (the same distinction
    ``find_undefined_symbols`` draws for the input warning).
    """
    decls: list[HandoffDeclaration] = []
    undeclared: list[str] = []
    for name in sorted(names):
        prop = context.get(name) if context else None
        if prop is None:
            undeclared.append(name)
            continue
        q = prop.quantity
        decls.append(HandoffDeclaration(
            name=name,
            kind=getattr(q, "kind", "scalar") or "scalar",
            domain=getattr(q, "domain", None),
            sign=getattr(q, "sign", None),
        ))
    return decls, undeclared


def seed_operations(handoff) -> "list[tuple[str, dict]]":
    """The (operation, args) sequence that re-establishes *handoff* in a new session.

    Ordinary engine operations, executed for real in the target session — not a
    session file written behind the CAS's back.  That is what makes the target
    timeline honest: the declarations are steps the researcher can see, and the
    CAS state actually matches them.

    Sign travels as ``define_scalar_parameter``'s ``sign`` argument, which the
    engine already forwards to ``BackendHandler.assume_sign`` — the wiring
    SIGN-DIALOG-ASSUME repaired, reused rather than restated.
    """
    ops: list[tuple[str, dict]] = []
    for d in handoff.declarations:
        args: dict = {"name": d.name}
        if d.domain:
            args["domain"] = d.domain
        if d.sign:
            args["sign"] = d.sign
        ops.append(("define_scalar_parameter", args))
    return ops


def perform_handoff(handoff, sessions_dir,
                    filename: "str | None" = None) -> "str":
    """Create the TARGET session from *handoff* and save it; return its path.

    A file, not a live switch — the same shape as the existing file-snapshot
    fork (`/fork`), which also writes a new session and leaves the current one
    loaded.  Whether the app should then open it is a UX decision, not this
    function's to make.

    Everything is executed for real in a fresh engine of the target backend:
    the declarations become steps the researcher can see, the CAS state
    actually matches them, and the carried expression is registered from the
    researcher's own writing (`export_latex`) rather than from a Julia handle
    that names nothing outside its own session.
    """
    from notebook.engine import NotebookEngine
    from notebook.models.session import SessionMode

    mode = (SessionMode.second_quantization
            if handoff.target_backend == "julia_full" else SessionMode.standard_symbolic)
    title = handoff.source_title or "handoff"
    with NotebookEngine(title=f"{title} → {handoff.target_backend}",
                        mode=mode) as eng:
        eng.record_manual_note(
            f"[scalar handoff {handoff.handoff_id}] "
            f"source: {handoff.source_title or '(untitled)'} "
            f"({handoff.source_backend}), {', '.join(handoff.source_expression_ids)}"
            f", branch {handoff.source_branch_id}"
            # The operator label could not cross, so say here what the value is
            # OF.  Dropping it silently would leave the target session holding a
            # number with no account of where it came from.
            + (f" — value of ${handoff.source_label_latex}$"
               if handoff.source_label_latex else ""))
        for op, args in seed_operations(handoff):
            eng.run_operation(op, args)
        verdict: dict = {"status": "not_checkable", "reason": "no_expression"}
        if handoff.export_latex:
            canonical = eng._handler.latex_to_canonical(handoff.export_latex)
            eng.register_expression(canonical or handoff.export_latex,
                                    latex=handoff.export_latex)
            # Check the CROSSING itself: the same expression, valued twice from
            # the two canonicals.  Recorded in the target timeline whatever the
            # outcome — including "could not check", which must never be filed
            # as agreement.
            # The source side of the comparison is the source's own reading of
            # the exported LaTeX — never `source_canonical`, which for a Julia
            # result is the opaque `_nb_Ek` handle (see the field's docstring).
            _src = handoff.source_value_canonical or handoff.source_canonical or canonical
            if _src.strip().startswith("_nb_"):
                _src = ""      # a handle names nothing here; say so, don't guess
            verdict = (verify_crossing(_src, canonical or "", handoff.declarations)
                       if _src else
                       {"status": "not_checkable", "reason": "opaque_source_canonical",
                        "detail": handoff.source_canonical[:60]})
            eng.record_manual_note(
                f"[handoff verification] {verdict.get('status')}"
                + (f" — {verdict.get('reason')}: {verdict.get('detail', '')}"
                   if verdict.get("status") != "agreed" else
                   f" (values match at {verdict.get('points')} points)"))
        handoff.verification = verdict
        stem = filename or f"{handoff.handoff_id}_{handoff.target_backend}"
        stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:60]
        path = sessions_dir / (stem if stem.endswith(".json") else f"{stem}.json")
        eng.save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Verifying the crossing (docs/34 §5 item 4)
# ---------------------------------------------------------------------------
#
# A handoff crosses backends, so the usual differential check — ask the CAS
# whether `ratsimp(before - after)` is 0 — is unavailable: the two sides live in
# different processes with different notations.  What CAN be compared is the
# VALUE at a point, computed twice from the two canonicals through the numeric
# layer's whitelist-bound lowering (`canonical_to_numpy`).
#
# What this proves and what it does not, stated plainly: it checks that the
# CROSSING preserved meaning — that the export→re-read round trip did not
# quietly become a different expression.  That is the failure class this
# project keeps finding (docs/32: `\frac{1}{\sqrt{x}}` becoming the PRODUCT
# `frac1*sqrt(x)`; slice 2 here: an uncarried declaration turning `omega_D`
# into the indexed `omega(D)`).  It does NOT compare the two CASes against each
# other — that would need numeric evaluation inside each session, which is a
# separate piece of work.
#
# When the expression uses something outside the numeric whitelist (`asinh`,
# say) the check cannot run.  It then reports `not_checkable` WITH the reason:
# an unverified crossing must never be reported as a verified one.

#: Deterministic sample values, chosen to respect a declaration's sign and to
#: stay away from 0 (a singularity there would make both sides NaN and the
#: comparison meaningless).
_SAMPLE_POSITIVE = (1.7, 0.6, 2.3)
_SAMPLE_NEGATIVE = (-1.3, -0.8, -2.1)
_SAMPLE_FREE = (0.9, 1.4, 0.4)


class _MathAsNumpy:
    """The numpy names `canonical_to_numpy` can emit, backed by `math`.

    numpy lives in the dedicated `.venv-numeric`, never the app venv (CLAUDE.md
    — the same rule that keeps SymPy out).  A scalar spot check needs no arrays,
    so the handful of names the converter is able to emit are mapped onto the
    standard library instead of importing a dependency the app deliberately
    does not have.  Anything the converter could emit but this shim lacks
    raises AttributeError, which `verify_crossing` reports as "cannot check" —
    never as agreement.
    """
    import math as _m
    abs = staticmethod(abs)
    arccos, arcsin, arctan = staticmethod(_m.acos), staticmethod(_m.asin), staticmethod(_m.atan)
    arctan2 = staticmethod(_m.atan2)
    cos, cosh, exp = staticmethod(_m.cos), staticmethod(_m.cosh), staticmethod(_m.exp)
    floor, log = staticmethod(_m.floor), staticmethod(_m.log)
    round = staticmethod(round)
    sign = staticmethod(lambda x: (x > 0) - (x < 0))
    sin, sinh, sqrt = staticmethod(_m.sin), staticmethod(_m.sinh), staticmethod(_m.sqrt)
    tan, tanh = staticmethod(_m.tan), staticmethod(_m.tanh)
    e, pi = _m.e, _m.pi


def sample_bindings(free_names, declarations, point: int = 0) -> dict:
    """One assumption-respecting numeric point for *free_names*."""
    by_name = {d.name: d for d in (declarations or [])}
    out: dict[str, float] = {}
    for name in sorted(free_names):
        sign = getattr(by_name.get(name), "sign", None)
        if sign in ("positive", "nonneg"):
            pool = _SAMPLE_POSITIVE
        elif sign in ("negative", "nonpos"):
            pool = _SAMPLE_NEGATIVE
        else:
            pool = _SAMPLE_FREE
        out[name] = pool[point % len(pool)]
    return out


def verify_crossing(source_canonical: str, target_canonical: str,
                    declarations=None, points: int = 3) -> dict:
    """Compare the two sides' VALUES at a few points.

    Returns ``{"status": "agreed" | "differed" | "not_checkable", ...}``.
    """
    import math

    from notebook.numeric.expr_to_numpy import canonical_to_numpy

    try:
        src_expr, src_free = canonical_to_numpy(source_canonical)
        tgt_expr, tgt_free = canonical_to_numpy(target_canonical)
    except Exception as exc:            # noqa: BLE001 - reported, not swallowed
        # Deliberately broad: the converter refuses anything outside its
        # whitelist, and the point here is to report "cannot check" for every
        # such reason rather than to classify them.
        return {"status": "not_checkable", "reason": f"{type(exc).__name__}",
                "detail": str(exc)[:120]}

    if src_free != tgt_free:
        # Different free symbols is already a different expression — the shape
        # `frac1*sqrt(x)` takes when a `\frac` fails to convert.
        return {"status": "differed", "reason": "free_symbols",
                "detail": f"{sorted(src_free)} vs {sorted(tgt_free)}"}

    env = {"np": _MathAsNumpy, "__builtins__": {}}
    for i in range(max(1, points)):
        binding = sample_bindings(src_free, declarations, point=i)
        try:
            a = float(eval(src_expr, env, dict(binding)))       # noqa: S307
            b = float(eval(tgt_expr, env, dict(binding)))       # noqa: S307
        except (ArithmeticError, ValueError, AttributeError, NameError) as exc:
            # Enumerable on purpose: a singular point (ZeroDivisionError,
            # OverflowError), a domain error (`sqrt` of a negative), a name the
            # math shim lacks, or an unbound symbol.  Anything else is a bug in
            # this checker and must not be filed as "cannot check".
            return {"status": "not_checkable", "reason": "evaluation",
                    "detail": str(exc)[:120]}
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            continue
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            return {"status": "differed", "reason": "value",
                    "detail": f"at {binding}: {a} vs {b}"}
    return {"status": "agreed", "points": points}
