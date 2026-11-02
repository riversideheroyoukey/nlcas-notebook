"""LHS / derivative LaTeX formatting and equation-form parsing.

Pure helper functions extracted from engine.py (AUD-4 stage 1): derivative
and provenance LaTeX construction, simple-symbol-name extraction, and
equation-form (LHS = RHS) parsing.  No dependency on NotebookEngine — these
are stateless transforms re-exported by engine.py for backward compatibility.
"""
from __future__ import annotations

import re

from notebook.models.quantity import Quantity

def _latex_var(var: str, context=None) -> str:
    r"""Render a bare variable name as LaTeX (`theta` → `\theta`, `y_bar` → `\bar{y}`).

    **One door.** "How is a bare name spelled?" had THREE answers — this one,
    `integral_form.latex_var`, and `backend_handler.canonical_name_to_latex` —
    and the first two knew only the Greek letters, while the DERIVATIVE label
    asked none of them and wrote the internal canonical name (measured
    2026-09-24, DERIV-DISPLAY-CANONICAL-NAME: `\frac{df}{dtheta}` typesets as
    the italic word "dtheta" AND does not read back, so `d` and the variable
    cancelled into an ordinary fraction — the very BP-9 failure).  The full
    door also knows the researcher's declared `latex_name`, the decorations
    (`H_hat_0` → `\hat{H}_{0}`) and a Greek base with a subscript (`omega_0` →
    `\omega_{0}`); on every plain name (`x`, `r`, `t`, `k_B`, `E_total`) the
    three agreed, which is why folding them changes nothing else.
    """
    from notebook.runtime.backend_handler import canonical_name_to_latex
    return canonical_name_to_latex(str(var).strip().lstrip("\\"), context)


def _equation_form_latex(
    operation: str,
    args: dict,
    integrand_latex: str,
    result_latex: str,
    format_const=None,
) -> str | None:
    """Return equation-form LaTeX for integral and limit operations, or None.

    *format_const* is a callable that converts a backend canonical constant
    string to its LaTeX representation.  Callers should pass
    ``handler.format_constant_for_latex``; defaults to identity when omitted.
    """
    from notebook.runtime.integral_form import latex_bound
    if operation == "integrate_indefinite":
        var = _latex_var(args.get("var", "x"))
        return rf"\int {integrand_latex} \, d{var} = {result_latex}"
    if operation == "integrate_definite":
        var = _latex_var(args.get("var", "x"))
        lo  = latex_bound(args.get("lower", ""), format_const)
        hi  = latex_bound(args.get("upper", ""), format_const)
        return rf"\int_{{{lo}}}^{{{hi}}} {integrand_latex} \, d{var} = {result_latex}"
    # Σ and Π over a NAMED quantity read like the integral above: the operator
    # is written around the name, because the result is not that quantity —
    # `\sum_{i=1}^{3} Y` is `3Y`, so re-attaching a bare `Y =` would assert
    # something false.  Until 2026-08-16 these two sent the equation WHOLE and
    # registered the tautology `3 x^2+3 = 3 x^2+3` (open_items EQ-SOURCE-BARE-LHS).
    if operation in ("summation", "product_expr"):
        var = args.get("var", "i")
        lo  = latex_bound(args.get("lower", ""), format_const)
        hi  = latex_bound(args.get("upper", ""), format_const)
        sign = r"\sum" if operation == "summation" else r"\prod"
        return (rf"{sign}_{{{var}={lo}}}^{{{hi}}} {integrand_latex} = {result_latex}")
    if operation == "limit":
        var   = args.get("var", "x")
        point = latex_bound(args.get("point", "0"), format_const)
        dirn  = str(args.get("direction", "both"))
        if dirn == "plus":
            arrow = rf"{var} \to {point}^{{+}}"
        elif dirn == "minus":
            arrow = rf"{var} \to {point}^{{-}}"
        else:
            arrow = rf"{var} \to {point}"
        return rf"\lim_{{{arrow}}} {integrand_latex} = {result_latex}"
    return None


#: A canonical NAME (not an expression): the general branch spells it.
_NAME_SHAPED_RE = re.compile(r"^[A-Za-z_]\w*'*$")


def _format_provenance_lhs(origin: str, diff_vars: list[str],
                           is_partial: bool,
                           qty: Quantity | None = None,
                           parse_ir=None, context=None) -> str:
    """Return LaTeX for a (possibly iterated/mixed) partial or ordinary derivative.

    origin    — the original function label: "c", "f(a,b)", "y", or an expression
    diff_vars — list of variables in differentiation order, e.g. ["a"] or ["a","b"]
    is_partial — True when ∂ notation is needed
    qty       — optional Quantity for dot/ddot notation when time_dependent=True

    Examples
    --------
    ("c",   ["a"],      True)  → ∂c/∂a
    ("c",   ["a","b"],  True)  → ∂²c/∂a∂b
    ("f(a,b)", ["a"],   True)  → ∂/∂a f(a,b)
    ("f(a,b)", ["a","b"], True)→ ∂²/∂a∂b f(a,b)
    ("y",   ["x"],      False) → dy/dx
    ("f(x)",["x"],      False) → d/dx f(x)
    ("q(t)",["t"],      False, qty(time_dependent=True)) → dot{q}
    ("q(t)",["t","t"],  False, qty(time_dependent=True)) → ddot{q}
    """
    n = len(diff_vars)

    # Dot/ddot notation for time-dependent quantities (q̇, q̈, q⃛, q⁽ⁿ⁾).
    # Only when notation="dot" — switching to "leibniz" produces d/dt even
    # for time-dependent quantities, allowing mixed notation in one session.
    if (qty is not None
            and qty.time_dependent
            and qty.notation == "dot"
            and not is_partial
            and n >= 1
            and all(v == qty.time_parameter for v in diff_vars)):
        base = qty.latex_name or _latex_var(qty.name, context)
        if n == 1:
            return rf"\dot{{{base}}}"
        if n == 2:
            return rf"\ddot{{{base}}}"
        if n == 3:
            return rf"\dddot{{{base}}}"
        return rf"{base}^{{({n})}}"
    dep = r"\partial" if is_partial else "d"
    sp = " " if is_partial else ""

    # Build denominator, grouping repeated adjacent variables (∂a²∂b etc.)
    def _denom(vars_list: list[str]) -> str:
        # **The variable is spelled the way the researcher wrote it.** Left as
        # the canonical name this published `\frac{d}{dtheta}` — the italic
        # word "dtheta", not `d\theta` — and BP-9's inverse could not read it
        # back either, so `d` and the variable cancelled into an ordinary
        # fraction (measured 2026-09-24).
        if len(vars_list) == 1:
            return f"{dep}{sp}{_latex_var(vars_list[0], context)}"
        parts: list[str] = []
        i = 0
        while i < len(vars_list):
            v = _latex_var(vars_list[i], context)
            cnt = 1
            while i + cnt < len(vars_list) and vars_list[i + cnt] == vars_list[i]:
                cnt += 1
            parts.append(f"{dep}{sp}{v}^{{{cnt}}}" if cnt > 1 else f"{dep}{sp}{v}")
            i += cnt
        return "".join(parts)

    denom = _denom(diff_vars)
    numerator_order = f"{dep}^{{{n}}}" if n > 1 else dep
    stripped = origin.strip()

    # Use qty.kind as primary detector (supports latex_name); fall back to IR.
    if qty is not None and qty.kind == "function":
        _fn_label = qty.latex_name or _latex_var(qty.name, context)
        _fn_inner = ",".join(qty.natural_vars)
        return rf"\frac{{{numerator_order}}}{{{denom}}} {_fn_label}({_fn_inner})"
        
    from notebook.models.ir import FunctionCall, Symbol
    ir = parse_ir(stripped) if parse_ir is not None else None

    if ir is not None and isinstance(ir, FunctionCall) and isinstance(ir.func, Symbol):
        from notebook.runtime.expr_generator import ExprGenerator
        gen = ExprGenerator()
        name = _latex_var(ir.func.name, context)
        inner = ", ".join(gen.generate(a) for a in ir.args)
        return rf"\frac{{{numerator_order}}}{{{denom}}} {name}({inner})"
    if qty is not None and qty.kind == "scalar":
        _sc_label = qty.latex_name or _latex_var(qty.name, context)
        num = f"{dep}{sp}{_sc_label}" if n == 1 else f"{dep}^{{{n}}}{sp}{_sc_label}"
        return rf"\frac{{{num}}}{{{denom}}}"
    if ir is not None and isinstance(ir, Symbol):
        _sym_label = _latex_var(ir.name, context)
        num = (f"{dep}{sp}{_sym_label}" if n == 1
               else f"{dep}^{{{n}}}{sp}{_sym_label}")
        return rf"\frac{{{num}}}{{{denom}}}"
    # General expression: wrap in \left(\right).  A NAME-shaped origin is a
    # NAME, not an expression — `H_hat_0` parses as a Subscript, so it reaches
    # this branch and used to be published verbatim (`\left(H_hat_0\right)`),
    # the `\[ H_hat_0 \]` shape CLAUDE.md records for declarations.
    if _NAME_SHAPED_RE.match(stripped):
        stripped = _latex_var(stripped, context)
    return rf"\frac{{{numerator_order}}}{{{denom}}}\left({stripped}\right)"


def _format_derivative_lhs(func_lhs: str, var: str, order: int,
                            is_partial: bool = False,
                            qty: Quantity | None = None,
                            parse_ir=None, context=None) -> str:
    """Thin wrapper around _format_provenance_lhs for single-step derivatives."""
    return _format_provenance_lhs(func_lhs, [var] * order, is_partial,
                                  qty=qty, parse_ir=parse_ir, context=context)


# Detects a derivative LHS that is already a single-variable (ordinary or
# partial) derivative wrapper of the form \frac{d}{dk}\left(INNER\right) or
# \frac{d^{n}}{dk^{n}}\left(INNER\right).  Used to collapse repeated
# differentiation w.r.t. the same variable into a higher-order derivative
# (position → velocity → acceleration) instead of nesting d/dk(d/dk(...)).
# The variable may now carry a BRACE GROUP (`\bar{y}`, `\omega_{0}`), because the
# label spells the researcher's own name (DERIV-DISPLAY-CANONICAL-NAME).  Written
# as `[^{}^]+?` it stopped at the first brace and the order bump silently stopped
# collapsing — d/dk(d/dk(X)) would nest instead of becoming d²/dk².
_DERIV_BUMP_RE = re.compile(
    r'^\\frac\{(?P<dep1>d|\\partial)(?:\^\{(?P<numord>\d+)\})?\}'
    r'\{(?P<dep2>d|\\partial)\s*(?P<var>(?:[^{}^]|\{[^{}]*\})+?)'
    r'(?:\^\{(?P<denord>\d+)\})?\}'
    r'\\left\((?P<inner>.*)\\right\)$'
)


def _bump_same_var_derivative(src_lhs_latex: str, var: str, add_order: int,
                              parse_ir=None, context=None) -> "str | None":
    """If *src_lhs_latex* is already a derivative w.r.t. *var*, return the LHS
    with its order raised by *add_order* (collapsing the nesting).  Returns
    None when *src_lhs_latex* is not a same-variable derivative wrapper, so the
    caller can fall back to the default wrapping behavior.
    """
    m = _DERIV_BUMP_RE.match(src_lhs_latex.strip())
    if m is None:
        return None
    # Numerator and denominator differential operators must agree (both d or
    # both \partial); the denominator variable must equal the new diff var.
    if m.group("dep1") != m.group("dep2"):
        return None
    # The label carries the RESEARCHER's spelling and *var* the canonical name,
    # so the two are compared through the same door that wrote the label.
    if m.group("var").strip() != _latex_var(var, context).strip():
        return None
    cur = int(m.group("numord") or "1")
    inner = m.group("inner")
    is_partial = (m.group("dep1") == r"\partial")
    return _format_provenance_lhs(inner, [var] * (cur + add_order), is_partial,
                                  parse_ir=parse_ir, context=context)


def _format_wrt_qdot_lhs(source_label: str, wrt_qty: Quantity) -> str:
    """Return LaTeX for ∂source_label/∂q̇.

    Always uses dot notation in the denominator — ∂/∂q̇ is the standard
    Lagrangian-mechanics form regardless of the symbol's notation.
    """
    base = wrt_qty.latex_name or wrt_qty.name
    qdot = rf"\dot{{{base}}}"
    return rf"\frac{{\partial {source_label}}}{{\partial {qdot}}}"


import re as _re_engine  # local alias to avoid shadowing outer `re`
from notebook.runtime.shared_parser import find_top_level_relation

# Matches a "simple" symbol name that can be used as a SymbolContext key.
# Allows: plain letters (H, E), underscored names (H_eff), LaTeX-subscripted
# names (H_{BCS}), and optionally a single trailing prime (H').
# Does NOT match: function calls f(x), complex LaTeX (\hat{H}), multi-part exprs.
_SIMPLE_SYM_RE = _re_engine.compile(
    r'^\\?(?P<base>[A-Za-z][A-Za-z0-9]*)(?:_\{?(?P<sub>[A-Za-z0-9]+)\}?)?\'?$'
)

# Decorated forms: \hat{X}, \bar{X}, \tilde{X}, \vec{X}, \check{X}, etc.
# Captures: decoration, base symbol, optional subscript.
_DECORATED_SYM_RE = _re_engine.compile(
    r'^\\(?P<deco>hat|bar|tilde|vec|check|breve|acute|grave|dot|ddot|mathring)'
    r'\{(?P<base>[A-Za-z][A-Za-z0-9]*)\}'
    r'(?:_\{?(?P<sub>[A-Za-z0-9]+)\}?)?$'
)

# Expectation value forms: <X>, \langle X \rangle, \langle X \rangle_sub
# Also handles:
#   \langle \hat{X} \rangle  (strips the decorator)
#   <ab>                     (multi-character, treated as single base "ab")
#   \langle n_k \rangle      (subscripted base)
#   \langle a b \rangle      (space-separated product → concatenated as "ab")
_EXPECT_RE = _re_engine.compile(
    r'^(?:<|\\langle)\s*'
    r'(?:\\(?:hat|bar|tilde|vec|dot|ddot)\{)?'              # optional decorator
    r'(?P<base>[A-Za-z][A-Za-z0-9]*'                        # first word of base
    r'(?:_\{?[A-Za-z0-9]+\}?)?'                            # optional subscript on base
    r'(?:\s+[A-Za-z][A-Za-z0-9]*)*)'                       # optional space-sep continuation
    r'\}?'                                                   # close decorator brace
    r'\s*(?:>|\\rangle)'
    r'(?:_\{?(?P<sub>[A-Za-z0-9]+)\}?)?$'
)

# Time/partial derivative forms: \frac{dX}{dt}, \frac{\partial X}{\partial t}
# Captures the numerator symbol and denominator variable.
_DERIV_RE = _re_engine.compile(
    r'^\\frac\{(?:d|\\partial)\s*'
    r'(?:\\(?:hat|bar|tilde|vec)\{)?(?P<sym>[A-Za-z][A-Za-z0-9]*)\}?'
    r'\}\{(?:d|\\partial)\s*(?P<var>[A-Za-z][A-Za-z0-9]*)\}$'
)

# Suffix appended to SymbolContext key for each decoration
_DECO_SUFFIX: dict[str, str] = {
    "hat":       "hat",
    "bar":       "bar",
    "tilde":     "tilde",
    "vec":       "vec",
    "check":     "check",
    "breve":     "breve",
    "acute":     "acute",
    "grave":     "grave",
    "dot":       "dot",
    "ddot":      "ddot",
    "mathring":  "ring",
}


def _extract_simple_symbol_name(lhs_label: str) -> str | None:
    """Extract a SymbolContext-compatible name from an LHS label string.

    Handles plain identifiers, LaTeX-decorated forms, and expectation-value
    notation.  Returns ``None`` when the label cannot be mapped to a
    predictable key (e.g. function calls, multi-part expressions).

    Key derivation rules
    --------------------
    Plain:
        ``H``           → ``"H"``
        ``H_eff``       → ``"H_eff"``
        ``E_{kin}``     → ``"E_kin"``
    Decorated (``\\hat``, ``\\bar``, etc.):
        ``\\hat{H}``    → ``"H_hat"``
        ``\\bar{H}``    → ``"H_bar"``
        ``\\bar{H}_{BCS}`` → ``"H_BCS_bar"``
    Expectation value:
        ``<H>``                   → ``"H_exp"``
        ``\\langle H \\rangle``   → ``"H_exp"``
        ``\\langle H \\rangle_0`` → ``"H_exp_0"``
    Not mapped (returns ``None``):
        ``f(x)``        — function call
        ``a + b``       — multi-symbol expression
        ``\\frac{d}{dt}H`` — complex LaTeX
    """
    s = lhs_label.strip()

    # 0. Derivative form: \frac{dX}{dt} → X_dt, \frac{\partial H}{\partial x} → H_dx
    m = _DERIV_RE.match(s)
    if m:
        sym = m.group("sym")
        var = m.group("var")
        return f"{sym}_d{var}"

    # 1. Expectation value: <X> or \langle X \rangle
    m = _EXPECT_RE.match(s)
    if m:
        # Normalize base: strip spaces (e.g. "a b" → "ab"), keep subscript notation
        base_raw = m.group("base")
        base = _re_engine.sub(r'\s+', '', base_raw)  # "a b" → "ab"
        sub  = m.group("sub")
        # Key order: base_exp_sub (e.g. H_exp_0) — exp comes before subscript
        key  = f"{base}_exp_{sub}" if sub else f"{base}_exp"
        return key

    # 2. Decorated: \hat{X}, \bar{X}, etc.
    m = _DECORATED_SYM_RE.match(s)
    if m:
        deco = m.group("deco")
        base = m.group("base")
        sub  = m.group("sub")
        suffix = _DECO_SUFFIX.get(deco, deco)
        if sub:
            key = f"{base}_{sub}_{suffix}"
        else:
            key = f"{base}_{suffix}"
        return key

    # 3. Plain identifier
    m = _SIMPLE_SYM_RE.match(s)
    if m is None:
        return None
    base = m.group("base")
    sub  = m.group("sub")
    key = f"{base}_{sub}" if sub else base
    # A Greek stem with digits glued on is that letter SUBSCRIPTED — `phi0` is
    # φ₀.  The declaration path already splits it (BackendHandler.
    # extra_pre_phase1); an H-1 left-hand side is the OTHER way a name enters
    # the notebook and was still keeping the glued form, so `phi0 = \phi`
    # published as the italic word "phi0" (measured 2026-08-09).  Same answer,
    # both doors.
    from notebook.runtime.backend_handler import split_greek_digit_name
    return split_greek_digit_name(key)


def _normalize_lhs(lhs: str) -> str:
    """Normalize an LHS label string for use as a lhs_alias_map key.

    Strips leading/trailing whitespace and collapses interior whitespace so
    that ``"H + V"`` and ``"H  +  V"`` map to the same key.  Also strips
    Maxima/LaTeX thin-space commands (``\\,``) and outer parentheses that
    are purely cosmetic.
    """
    import re as _r
    s = lhs.strip()
    s = _r.sub(r'\\,', '', s)          # Maxima thin space
    s = _r.sub(r'\s+', ' ', s)         # collapse whitespace
    return s


def _lhs_latex_from_expr(latex: str) -> str:
    r"""Extract the LHS label from an expression's stored latex string.

    Handles both spaced form 'LHS = RHS' and compact form 'LHS=RHS'.
    Returns the full string when there is no top-level '=' (plain expression).

    An `=` inside brackets is NOT one: `latex.find("=")` read
    `\sum_{i=1}^{3} i^2` as an equation whose LHS is `\sum_{i`, and the result
    of summing it displayed as `\sum_{i = 14` (measured 2026-08-08).
    `find_top_level_relation` is the bracket-aware scan the parser already uses
    for this question.
    """
    rel = find_top_level_relation(latex)
    if rel is not None and rel[1] == "=":
        return latex[:rel[0]].strip()
    return latex


def _rhs_of_latex(latex: str) -> str:
    """Extract the RHS from an equation latex string, or return it unchanged.

    Used to strip the Maxima-generated LHS from a combined-equation result
    so the display does not end up with two '=' signs.
    """
    rel = find_top_level_relation(latex)
    if rel is not None and rel[1] == "=":
        return latex[rel[0] + 1:].strip()
    return latex


_COMBINE_OP_SYMBOLS: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": r"\cdot",
    "/": "/",
}


def _has_toplevel_additive(latex: str) -> bool:
    """Return True if *latex* has a top-level + or - (not inside braces/parens)."""
    depth = 0
    for ch in latex:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and ch in "+-":
            return True
    return False
