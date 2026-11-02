"""Unevaluated multiple definite-integral form (S-o change-of-variables workflow).

A *descriptor* is a plain dict carried on ``Expression.integral_descriptor``::

    {
      "integrand_latex":     "x^2+y^2",
      "integrand_canonical": "x^2+y^2",      # backend canonical (for eval)
      "bounds": [["y", "0", "1"], ["x", "0", "1"]],   # OUTER → INNER
      "region": {"kind": "rectangle"} | {"kind": "disk", "radius": "1"} | None,
    }

These helpers are pure (no CAS, no engine): they parse the LaTeX integral the user
typed, rebuild the display LaTeX, assemble the iterated CAS command for the eval
step, and derive new bounds for standard coordinate changes.  Integrand
LaTeX↔canonical conversion is backend-specific and stays with the handler/caller.
"""
from __future__ import annotations

import re

from notebook.runtime.latex_typography import braced_group


def latex_var(var: str, context=None) -> str:
    r"""Bare variable name → LaTeX (`theta` → `\theta`, `y_bar` → `\bar{y}`).

    Delegates to the ONE door (`backend_handler.canonical_name_to_latex`).
    This used to be a second Greek-only table beside
    `lhs_formatting._latex_var`, and the DERIVATIVE label asked neither
    (2026-09-24, DERIV-DISPLAY-CANONICAL-NAME).
    """
    from notebook.runtime.backend_handler import canonical_name_to_latex
    return canonical_name_to_latex(str(var).strip().lstrip("\\"), context)


#: The ``\int`` sign itself.  Its LIMITS are read separately, with BALANCED
#: braces — written into this regex they could not nest, and a bound as
#: ordinary as ``\frac{\pi}{2}`` or ``\sqrt{2}`` made the whole integral fall
#: through to the general route (measured 2026-08-16).  Nothing was wrong with
#: the answer, but the same notation became a different KIND of object: the
#: claimed spelling is a first-class integral form that can be transformed and
#: evaluated, the declined one an ordinary expression that cannot
#: (open_items INTEGRAL-CLAIM-ASYMMETRY).
_INT_HEAD_RE = re.compile(r"\\int\s*")
_DIFF_RE = re.compile(r"(?:\\,|\\;|\s)*(?:\\mathrm\{d\}|\\,?d|d)\s*(\\?[A-Za-z]\w*)\s*$")


def expand_multi_int_commands(s: str) -> str:
    r"""``\iiint`` → ``\int\int\int``, ``\iint`` → ``\int\int``.

    The multiple-integral commands are pure abbreviations — ``\iint f \, dx \,
    dy`` and ``\int \int f \, dx \, dy`` are the same integral — so making them
    ONE spelling at the entrance means nothing downstream has to know both.

    It was known in one place only.  This module expanded them for its own
    regex (route ①), and the general route did not, so a double integral
    WITHOUT explicit bounds — which ① declines, sending it to the general
    route — reached the CAS as ``iint * x * y * dx * dy``: a product with an
    undefined symbol named `iint`, and the ∫ gone (measured 2026-08-16).
    ``\int \int x y \, dx \, dy`` on the same page answered correctly.
    """
    return s.replace(r"\iiint", r"\int\int\int").replace(r"\iint", r"\int\int")


def _read_limit(s: str, i: int, mark: str) -> "tuple[str, int] | None":
    """Read one ``_…``/``^…`` limit at *i* — braced (balanced) or a bare token."""
    if i >= len(s) or s[i] != mark:
        return None
    k = i + 1
    group = braced_group(s, k)
    if group is not None:
        return group[0].strip(), group[1]
    m = re.match(r"\\?[A-Za-z0-9]+|[^{}\s]", s[k:])
    return (m.group(0), k + m.end()) if m else None


def _strip(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith("{") and tok.endswith("}"):
        tok = tok[1:-1]
    return tok.strip()


def parse_latex_definite_integral(latex: str) -> dict | None:
    """Parse ``\\int_a^b \\int_c^d <integrand> \\, dx \\, dy`` into a descriptor
    (without the canonical integrand — the caller fills that in).  Returns None if
    the string is not a (multiple) definite integral with explicit bounds.

    Pairing convention: the outermost (leftmost) ``\\int`` binds the *last-written*
    differential.  Bounds are returned OUTER → INNER.
    """
    s = expand_multi_int_commands(latex.strip())
    bounds_lr: list[tuple[str, str]] = []
    pos = 0
    while True:
        head = _INT_HEAD_RE.match(s, pos)
        if not head:
            break
        lo = _read_limit(s, head.end(), "_")
        hi = _read_limit(s, lo[1], "^") if lo else None
        if not (lo and hi):
            break
        bounds_lr.append((_strip(lo[0]), _strip(hi[0])))
        pos = hi[1]
        while pos < len(s) and s[pos] in " \t":
            pos += 1
    if not bounds_lr:
        return None
    body = s[pos:].strip()
    diffs_lr: list[str] = []
    while True:
        m = _DIFF_RE.search(body)
        if not m:
            break
        diffs_lr.append(m.group(1).lstrip("\\"))
        body = body[:m.start()].strip()
    diffs_lr.reverse()
    if len(diffs_lr) != len(bounds_lr) or not body:
        return None
    n = len(bounds_lr)
    # outer bound k binds differential (n-1-k) (last-written = outermost)
    bounds = [[diffs_lr[n - 1 - k], bounds_lr[k][0], bounds_lr[k][1]] for k in range(n)]
    integrand = body.strip()
    if integrand.startswith("(") and integrand.endswith(")"):
        integrand = integrand[1:-1].strip()
    return {"integrand_latex": integrand, "integrand_canonical": None,
            "bounds": bounds, "region": None}


def latex_bound(bound: str, format_const=None) -> str:
    r"""A backend integration bound → the LaTeX to show for it.

    Two steps, and BOTH are needed: the backend's own constants become commands
    (``2*%pi`` → ``2*\pi``, *format_const*), and the explicit multiplication
    between a number and a command comes out (``2*\pi`` → ``2\pi``), because
    nobody writes the `*` in a limit.

    One function because it was two.  ``build_integral_latex`` stripped the
    ``*`` and ``lhs_formatting`` did not, and the difference was invisible only
    because `normalize_bound` used to produce the INVALID `2%pi` — with no `*`
    in it there was nothing to strip, and the CAS answered `_res_`.  Fixing the
    bound spelling made the second site show `\int_{0}^{2*\pi}` (measured
    2026-08-16): a display defect that existed all along, uncovered by removing
    the thing that was hiding it.
    """
    _fc = format_const or (lambda s: s)
    return re.sub(r"(?<=[\w}])\s*\*\s*(?=[\\A-Za-z])", "", _fc(str(bound)))


def build_integral_latex(desc: dict, integrand_latex: str | None = None,
                         result_latex: str | None = None,
                         format_const=None) -> str:
    """Reconstruct ``\\int…\\int integrand \\, d(inner) … \\, d(outer)`` from a
    descriptor.  Bounds are OUTER → INNER; differentials are emitted INNER → OUTER.
    If *result_latex* is given, append ``= result``.  *format_const* (e.g.
    ``handler.format_constant_for_latex``) renders backend constants in bounds
    (``2*%pi`` → ``2\\pi``); defaults to identity."""
    def fc(b: str) -> str:
        return latex_bound(b, format_const)
    bounds = desc["bounds"]
    # The bounds stored on a descriptor are BACKEND text, and converting them
    # back cannot rebuild what was written: `\frac{\pi}{2}` normalizes to
    # `((%pi)/(2))` and comes back as `((\pi)/(2))` — correct, re-typeable, and
    # not the researcher's own writing.  When the input path recorded the
    # written spelling, show that instead (same rule as the extremum binder).
    # A descriptor built by a change of variables has no written form and falls
    # back, which is right: nobody wrote those bounds.
    written = desc.get("bounds_latex") or []
    body = integrand_latex if integrand_latex is not None else desc["integrand_latex"]
    def _lo_hi(k: int, lo: str, hi: str) -> "tuple[str, str]":
        if k < len(written) and written[k] and len(written[k]) == 2:
            return written[k][0], written[k][1]
        return fc(lo), fc(hi)
    ints = "".join(rf"\int_{{{_l}}}^{{{_h}}} "
                   for _l, _h in (_lo_hi(k, lo, hi)
                                  for k, (_v, lo, hi) in enumerate(bounds)))
    diffs = " ".join(rf"\, d{latex_var(v)}" for v, _lo, _hi in reversed(bounds))
    out = f"{ints}{body} {diffs}".strip()
    if result_latex is not None:
        out = f"{out} = {result_latex}"
    return out


def build_eval_command(integrand_canonical: str, bounds: list[list[str]]) -> str:
    """Assemble the iterated Maxima ``integrate(…)`` command (inner integral first)."""
    cmd = f"({integrand_canonical})"
    for var, lo, hi in reversed(bounds):          # INNER → OUTER
        cmd = f"integrate({cmd}, {var}, {lo}, {hi})"
    return cmd


# --- region + new-bound derivation for standard coordinate changes ----------

# Recognized substitution shapes (whitespace/`*` removed, compared on RHS set).
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace("*", "")


def classify_transform(old_vars: list[str], transform: list[str]) -> str | None:
    """Identify a standard coordinate change from its substitution RHS.  Returns
    'polar' | 'spherical' | None (None ⇒ caller falls back to LLM / user)."""
    rhs = {_norm(t) for t in transform}
    if len(old_vars) == 2 and rhs == {"rcos(theta)", "rsin(theta)"}:
        return "polar"
    if len(old_vars) == 3 and rhs == {
            "rsin(theta)cos(phi)", "rsin(theta)sin(phi)", "rcos(theta)"}:
        return "spherical"
    return None


def derive_new_bounds(kind: str, region: dict | None,
                      new_vars: list[str]) -> list[list[str]] | None:
    """New integration bounds (OUTER → INNER) for a standard transform over a
    standard region.  Returns None when it cannot be derived (⇒ LLM / user input).

    Polar over a disk of radius R:  θ:0..2π (outer), r:0..R (inner).
    Spherical over a ball radius R: φ:0..2π, θ:0..π, r:0..R.
    """
    if region is None:
        return None
    R = str(region.get("radius", "")) if region.get("kind") in ("disk", "ball") else ""
    if kind == "polar" and region.get("kind") == "disk" and R:
        return [["theta", "0", "2*%pi"], ["r", "0", R]]
    if kind == "spherical" and region.get("kind") == "ball" and R:
        return [["phi", "0", "2*%pi"], ["theta", "0", "%pi"], ["r", "0", R]]
    return None
