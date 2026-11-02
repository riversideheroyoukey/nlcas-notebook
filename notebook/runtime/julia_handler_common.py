"""Julia backend: module-level IR/operator helpers, regexes, and constants.

Extracted from julia_backend_handler.py (AUD-4 stage 2) to shrink the handler
module.  These are stateless helpers and compiled patterns used by
JuliaBackendHandler; they are re-imported there so existing references and
external imports (e.g. tests importing _adjoint_ir) keep working.
"""
from __future__ import annotations

import re

from notebook.runtime.backend_handler import GREEK_CMD

# Alias for internal use within this module (normalize_display_latex etc.)
_GREEK_CMD = GREEK_CMD

# Mapping from Quantity.domain (kind) values to Julia Symbolics.jl type
# annotations.  Julia has no Positive/Natural built-in types, and Symbolics.jl
# has no sign predicate either, so a sign qualifier (Quantity.sign:
# positive/negative/nonneg/nonpos) never reaches this mapping — only the
# domain KIND (real/integer/natural/complex) does; see BackendHandler.assume_sign.
_DOMAIN_TO_JULIA_TYPE: dict[str, str] = {
    "real":     "Real",
    "integer":  "Integer",
    "natural":  "Integer",
    # "complex" -> no annotation (Symbolics default)
}

# Operations that produce no meaningful return value on the Julia side.
_JULIA_VOID_OPS: frozenset[str] = frozenset({
    "define_boson_operator",
    "define_fermion_operator",
    "define_spin_operator",
    "define_scalar_parameter",
    "define_vector",
    "define_matrix",
    "define_tensor",
})

# ---------------------------------------------------------------------------
# Particle operator patterns (Phase 0.5)
# ---------------------------------------------------------------------------
# The particle-operator declaration grammar moved to i18n/nl_patterns.json
# (`particle_operator_decl`, `bare_quantum_op_decl`, `particle_def_op_text`) on
# 2026-08-23 — NL trigger patterns are language CONTENT and belong there, not in
# source (the `feedback_nl_input_i18n` rule; the anti-pattern it names is exactly
# the bilingual `を|as` + `フェルミオン演算子|fermion operator` regex that stood
# here).  The operator-expression fragment went with them, as `{{op_expr}}`.
#
# Checked before moving: over the 14 real spellings the instruction corpus holds,
# the new patterns read the same (op, particle, kind) as the five they replace.

# Strip base identifier from operator expression (e.g. "a'(k)" → "a")
_OP_BASE_NAME_RE = re.compile(r"^([A-Za-zσ][A-Za-z0-9_]*)")

# ---------------------------------------------------------------------------
# IR helpers shared by operator_substitute and relabel_index
# ---------------------------------------------------------------------------

def _collect_dummy_names(ir: "Expr") -> "list[str]":
    """Collect dummy index names from top-level IndexSum nodes in the IR."""
    from notebook.models.ir import IndexSum, Add
    if isinstance(ir, IndexSum):
        return [idx.name for idx in ir.dummy_indices]
    if isinstance(ir, Add):
        seen: set[str] = set()
        result: list[str] = []
        for arg in ir.args:
            for n in _collect_dummy_names(arg):
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result
    return []


def _relabel_index_in_canonical(canonical: str, old_idx: str, new_idx: str) -> str:
    """Swap old_idx and new_idx throughout a canonical expression string.

    Designed for BZ relabeling such as k → -k, where the sum dummy index
    is a single letter.  Performs a three-step token swap:

      1. neg-form (_-k / (-k))  → temporary marker
      2. pos-form (_k  /  (k))  → neg-form
      3. temporary marker       → pos-form
      4. Restore sum-variable occurrences (\\sum_-k → \\sum_k) because
         the sum domain is symmetric and Σ_{-k} = Σ_k.

    Only handles the case where new_idx == '-' + old_idx (standard BZ
    relabeling).  For other substitutions, fall back to simple subst.
    """
    # Only the k → -k (sign-flip) pattern is supported via swap.
    # For arbitrary relabeling, just do a direct substitution.
    if new_idx != f"-{old_idx}":
        # Simple directional substitution (no swap needed)
        s = canonical
        s = re.sub(rf"\({re.escape(old_idx)}\)", f"({new_idx})", s)
        s = re.sub(rf"_{re.escape(old_idx)}\b", f"_{new_idx}", s)
        return s

    TEMP = "__IDX_SWAP_TEMP__"
    s = canonical

    # Step 1: neg-form → TEMP  (to avoid double negation in step 2)
    s = s.replace(f"(-{old_idx})", f"({TEMP})")
    s = re.sub(rf"_-{re.escape(old_idx)}\b", f"_{TEMP}", s)

    # Step 2: pos-form → neg-form
    s = re.sub(rf"\({re.escape(old_idx)}\)", f"(-{old_idx})", s)
    s = re.sub(rf"_{re.escape(old_idx)}\b", f"_-{old_idx}", s)

    # Step 3: TEMP → pos-form
    s = s.replace(f"({TEMP})", f"({old_idx})")
    s = s.replace(f"_{TEMP}", f"_{old_idx}")

    # Step 4: restore sum variables changed in step 2
    # In memory the canonical has a real backslash: \sum_-k → \sum_k
    s = re.sub(rf"\\sum_-{re.escape(old_idx)}\b", rf"\\sum_{old_idx}", s)
    s = s.replace(f"\\sum_{{-{old_idx}}}", f"\\sum_{{{old_idx}}}")

    return s


# ---------------------------------------------------------------------------
# IR helpers for operator_substitute
# ---------------------------------------------------------------------------

def _collect_scalar_fn_names(expr: "Expr", ctx: "SymbolContext") -> "set[str]":
    """Return names of scalar parameters that appear as function calls in expr.

    These need `Symbolics.@syms name(..)` declarations before normal_form()
    so Julia treats them as callable symbolic functions rather than bare Syms.
    """
    from notebook.models.ir import (
        Add, Mul, NonCommutativeMul, Pow, FunctionCall, Adjoint,
        Symbol as _SymIR, IndexSum,
    )
    names: set[str] = set()
    if isinstance(expr, FunctionCall) and isinstance(expr.func, _SymIR):
        fn = expr.func.name.rstrip("'†")
        prop = ctx.get(fn) if ctx else None
        if prop and prop.quantity.kind in ("scalar", "anonymous"):
            names.add(fn)
        for a in expr.args:
            names |= _collect_scalar_fn_names(a, ctx)
    elif isinstance(expr, (Add, Mul, NonCommutativeMul)):
        for a in expr.args:
            names |= _collect_scalar_fn_names(a, ctx)
    elif isinstance(expr, Adjoint):
        names |= _collect_scalar_fn_names(expr.expr, ctx)
    elif isinstance(expr, Pow):
        names |= _collect_scalar_fn_names(expr.base, ctx)
        names |= _collect_scalar_fn_names(expr.exponent, ctx)
    elif isinstance(expr, IndexSum):
        names |= _collect_scalar_fn_names(expr.body, ctx)
    return names

def quantum_operator_names(
    ctx: "SymbolContext | None", *ir_nodes: "object"
) -> "set[str]":
    """The names this session treats as QUANTUM OPERATORS, with prime variants.

    The declaration is the discriminator — the same one the `^\\dagger` reading
    and `subscript_policy` already use — and measured over the published Julia
    sessions it is exactly right: every primed operator base (`c`, `g`, `a`,
    `b`) carries ``kind == "operator"``, and `sigma` — whose prime marks a
    DISTINCT VARIABLE, not an adjoint — is declared nowhere.

    ``ir_nodes`` add the bases that carry a prime in an APPLIED position but are
    absent from the context (a Bogoliubov `beta'(-k)` introduced by a
    substitution); pass them where an IR is at hand.

    Three sites asked this question and only two answered it: ``operator_substitute``
    and ``_adjoint_ir``'s caller each built the set inline, and the DISPLAY did not
    ask at all — so every prime became a dagger and `G(t, t')` was shown as
    `G(t, t^{\\dagger})` (PRIME-DISPLAY-DAGGER, 2026-09-10).
    """
    names: set[str] = set()
    if ctx is not None:
        for _name, _entry in ctx.registry.items():
            if _entry.quantity.kind == "operator":
                names.add(_name)
                names.add(_name + "'")
    for _node in ir_nodes:
        if _node is None:
            continue
        for _base in _prime_bases(_node):
            names.add(_base)
            names.add(_base + "'")
    return names



def integer_spin_operator_names(ctx: "SymbolContext | None") -> "set[str]":
    r"""符号のラベル（`⁺ ⁰ ⁻`）を持ちうる演算子——**整数スピン＝ボソン**——の名前。

    `quantum_operator_names` の一段狭い版で、隣に在るのはそれが同じ問いだから
    である。符号の射影は整数スピンのものなので、演算子であるだけを条件にすると
    電子の `c_{1,+}` に当たり、**その粒子が持たない射影**を黙って添字にする。
    **統計はスピンのパリティを決める**（CLAUDE.md「Statistics fixes the parity,
    not the magnitude」）ので、`statistics.kind == "bose"` が使える判別子である。

    **限界**: 宣言に残るのは統計までで**粒子そのものは残らない**（`registry` の
    項目は `quantity` と `statistics` だけ）ので、「その粒子の値集合に本当にこの
    射影が在るか」はここからは確かめられない——スピン0のボソンに `a_{1,+}` と
    書けば意味の無い添字が付く（open_items OPERAND-BOUNDARY ③ 残）。
    """
    if ctx is None:
        return set()
    names: set[str] = set()
    for name, entry in ctx.registry.items():
        q = getattr(entry, "quantity", None)
        st = getattr(entry, "statistics", None)
        if q is None or getattr(q, "kind", None) != "operator":
            continue
        if getattr(st, "kind", None) != "bose":
            continue
        names.add(name)
        names.add(name + "'")
    return names

def _adjoint_ir(expr: "Expr", operator_names: "set[str]") -> "Expr":
    """Compute the Hermitian adjoint of an IR expression.

    FunctionCall nodes whose func name is in `operator_names` are wrapped with
    Adjoint (quantum operator dagger).  All other FunctionCall nodes (scalar
    helper functions like u(k), v(k)) are assumed real and returned unchanged.

    Known pure-imaginary scalars (the Julia imaginary unit "im") are complex-
    conjugated: im† = -im.  All other scalar Symbols are assumed real.
    """
    from notebook.models.ir import (
        Add, Mul, NonCommutativeMul, Pow, FunctionCall, Adjoint, Symbol,
        Number, IndexedSymbol,
    )

    # Pure-imaginary unit symbols at the IR level.
    # "i" is parsed by SharedParser and converted to "im" only at code-gen time.
    # "im" may also appear if the user writes Julia-style imaginary directly.
    # Both must be negated on adjoint: conj(i) = -i, conj(im) = -im.
    _IMAG_UNIT_NAMES: "frozenset[str]" = frozenset({"i", "im"})

    if isinstance(expr, Add):
        return Add(args=[_adjoint_ir(a, operator_names) for a in expr.args])
    if isinstance(expr, (Mul, NonCommutativeMul)):
        cls = type(expr)
        return cls(args=[_adjoint_ir(a, operator_names) for a in reversed(expr.args)])
    if isinstance(expr, Adjoint):
        return expr.expr  # (A†)† = A
    if isinstance(expr, FunctionCall) and isinstance(expr.func, Symbol):
        fn = expr.func.name
        # Check prime-suffix BEFORE plain membership to avoid double-wrapping.
        if fn.endswith("'") and fn[:-1] in operator_names:
            # X'(k) is creation — adjoint = annihilation X(k)
            return FunctionCall(func=Symbol(name=fn[:-1]), args=expr.args)
        if fn in operator_names:
            return Adjoint(expr=expr)            # X(k) → X†(k)
        return expr  # scalar function — assume real
    if isinstance(expr, IndexedSymbol) and isinstance(expr.base, Symbol):
        bn = expr.base.name
        if bn.endswith("'") and bn[:-1] in operator_names:
            return IndexedSymbol(base=Symbol(name=bn[:-1]), indices=expr.indices)
        if bn in operator_names:
            return IndexedSymbol(base=Symbol(name=bn + "'"), indices=expr.indices)
        return expr  # scalar subscript — assume real
    if isinstance(expr, Symbol):
        if expr.name in _IMAG_UNIT_NAMES:
            return Mul(args=[Number(value=-1), expr])  # i† = -i, im† = -im
        return expr  # real scalar symbol — unchanged
    return expr  # Number → real


def _normalize_idx_for_sign(actual_idx: "Expr", formal_name: str) -> "Expr":
    """Normalize index representations to Mul(-1, Symbol) / Symbol form.

    The SharedParser + JuliaSemanticResolver produce different IR for the same
    -k index depending on input notation:
      c(-k)  → FunctionCall args contain Mul(Number(-1), Symbol("k"))
      c_-k   → IndexedSymbol indices contain IndexSymbol(name="-k")
      c_k    → IndexedSymbol indices contain IndexSymbol(name="k")

    This helper converts IndexSymbol("k") → Symbol("k") and
    IndexSymbol("-k") → Mul(Number(-1), Symbol("k")) so that _is_neg_idx
    and _extract_pos_base work uniformly across both notations.
    """
    from notebook.models.ir import Mul, Number, Symbol
    try:
        from notebook.models.ir import IndexSymbol as _IS
        if isinstance(actual_idx, _IS):
            if actual_idx.name == formal_name:
                return Symbol(name=formal_name)
            if actual_idx.name == "-" + formal_name:
                return Mul(args=[Number(value=-1), Symbol(name=formal_name)])
    except ImportError:
        pass
    return actual_idx


def _is_neg_idx(actual_idx: "Expr", formal_name: str) -> bool:
    """True if actual_idx represents -(formal_name).

    Handles both notations produced by the SharedParser + JuliaSemanticResolver:
      Mul(Number(-1), Symbol(formal_name))  — from c(-k)
      IndexSymbol(name="-k")               — from c_-k  (after normalization)
    """
    idx = _normalize_idx_for_sign(actual_idx, formal_name)
    from notebook.models.ir import Mul, Number, Symbol
    if isinstance(idx, Mul) and len(idx.args) == 2:
        a, b = idx.args
        if (isinstance(a, Number) and a.value == -1
                and isinstance(b, Symbol) and b.name == formal_name):
            return True
        if (isinstance(b, Number) and b.value == -1
                and isinstance(a, Symbol) and a.name == formal_name):
            return True
    return False


def _extract_pos_base(neg_idx: "Expr") -> "Expr":
    """Extract positive Symbol from a negative-index expression.

    Handles both Mul(-1, Symbol("k")) and IndexSymbol("-k") forms.
    Returns Symbol("k") in either case.
    """
    # First normalize IndexSymbol("-k") → Mul(-1, Symbol("k"))
    normalized = neg_idx
    try:
        from notebook.models.ir import IndexSymbol as _IS
        if isinstance(neg_idx, _IS) and neg_idx.name.startswith("-"):
            from notebook.models.ir import Mul, Number, Symbol
            normalized = Mul(args=[Number(value=-1), Symbol(name=neg_idx.name[1:])])
    except ImportError:
        pass
    from notebook.models.ir import Mul, Number
    if isinstance(normalized, Mul) and len(normalized.args) == 2:
        a, b = normalized.args
        if isinstance(a, Number) and a.value == -1:
            return b
        if isinstance(b, Number) and b.value == -1:
            return a
    return normalized


def _subst_quantum_op(
    expr: "Expr",
    op_name: str,
    rhs_ir: "Expr",
    formal_idx: str,
    operator_names: "set[str]",
    index_sign: "str | None" = None,
    static_index: "str | None" = None,
    exact_index: bool = False,
) -> "Expr":
    """Traverse IR and replace op_name(idx) / op_name†(idx) throughout.

    * op_name(actual_idx)  → rhs_ir[formal_idx ← actual_idx]
    * op_name†(actual_idx) → adjoint(rhs_ir[formal_idx ← actual_idx])

    index_sign controls which occurrences are substituted:
    - None (default): all occurrences
    - "positive": only c(k) where actual_idx == Symbol(formal_idx)
    - "negative": only c(-k) where actual_idx == -Symbol(formal_idx);
                  substitutes rhs_ir with formal_idx ← positive base of actual_idx

    static_index: optional spin/discrete label to filter on args[1].
    When set, only occurrences c(k, <static_index>) are substituted.
    1-index operators (no args[1]) are always matched regardless.
    """
    from notebook.models.ir import (
        Add, Mul, NonCommutativeMul, Pow, FunctionCall, Adjoint, Symbol,
        Number, IndexSum, IndexedSymbol, IndexSymbol,
    )
    from notebook.runtime.index_contractor import substitute_symbol_or_index as _ssi

    # exact_index: when True, formal_idx is a concrete site label (filter mode).
    #   Only occurrences where actual_idx == formal_idx are substituted, and
    #   the RHS is returned as-is (no formal-parameter substitution).
    # exact_index: when False (default), formal_idx is a template parameter.
    #   All occurrences are substituted; rhs_ir is instantiated with
    #   formal_idx ← actual_idx (Bogoliubov / uniform mean-field style).
    #
    # The caller sets exact_index via the operation arg "exact_index": true,
    # or the supervisor auto-detects it when formal_idx does not appear in RHS
    # AND the caller passes exact_index=True explicitly.

    def _idx_matches_formal(idx: "Expr") -> bool:
        """True if idx represents the same label as formal_idx."""
        if isinstance(idx, (Symbol, IndexSymbol)):
            return idx.name.lstrip(":") == formal_idx
        if isinstance(idx, Number):
            # Integer site labels: Sz(1) has Number(1) as index.
            # Match when formal_idx is the same integer string.
            try:
                return int(formal_idx) == int(idx.value)
            except (ValueError, AttributeError):
                pass
        return False

    def _apply_rhs(actual_idx: "Expr") -> "Expr":
        """Apply rhs_ir with formal_idx ← actual_idx (or ← positive base for -k)."""
        if index_sign == "negative":
            return _ssi(rhs_ir, formal_idx, _extract_pos_base(actual_idx))
        if exact_index:
            return rhs_ir  # concrete site: RHS is constant, no substitution needed
        return _ssi(rhs_ir, formal_idx, actual_idx)

    def _should_subst(actual_idx: "Expr") -> bool:
        """Return True if this occurrence should be substituted given index_sign."""
        if exact_index:
            return _idx_matches_formal(actual_idx)
        if index_sign is None:
            return True
        is_neg = _is_neg_idx(actual_idx, formal_idx)
        return (index_sign == "negative") == is_neg

    def _spin_match(args_list: "list") -> bool:
        """Return True if args[1] matches static_index (or no filter set)."""
        if static_index is None:
            return True
        if len(args_list) < 2:
            return True  # 1-index operator — no spin arg to check
        spin_arg = args_list[1]
        if isinstance(spin_arg, Symbol):
            return spin_arg.name.lstrip(":") == static_index
        return True  # conservative: match if structure is unknown

    _rec = lambda a: _subst_quantum_op(
        a, op_name, rhs_ir, formal_idx, operator_names, index_sign, static_index, exact_index
    )

    def _norm(idx: "Expr") -> "Expr":
        """Normalize IndexSymbol(-k)→Mul(-1,k) so _is_neg_idx / _apply_rhs work."""
        return _normalize_idx_for_sign(idx, formal_idx)

    if isinstance(expr, Adjoint):
        inner = expr.expr
        if isinstance(inner, FunctionCall) and isinstance(inner.func, Symbol):
            if inner.func.name == op_name:
                actual_idx = _norm(inner.args[0] if inner.args else Symbol(name=formal_idx))
                if _should_subst(actual_idx) and _spin_match(inner.args):
                    return _adjoint_ir(_apply_rhs(actual_idx), operator_names)
        if isinstance(inner, IndexedSymbol) and isinstance(inner.base, Symbol):
            if inner.base.name == op_name:
                actual_idx = _norm(inner.indices[0] if inner.indices else Symbol(name=formal_idx))
                if _should_subst(actual_idx) and _spin_match(list(inner.indices)):
                    return _adjoint_ir(_apply_rhs(actual_idx), operator_names)
        return Adjoint(expr=_rec(inner))

    if isinstance(expr, FunctionCall) and isinstance(expr.func, Symbol):
        fn = expr.func.name
        actual_idx = _norm(expr.args[0] if expr.args else Symbol(name=formal_idx))
        if fn == op_name and _should_subst(actual_idx) and _spin_match(expr.args):
            return _apply_rhs(actual_idx)
        if fn == op_name + "'" and _should_subst(actual_idx) and _spin_match(expr.args):
            return _adjoint_ir(_apply_rhs(actual_idx), operator_names)
        return FunctionCall(func=Symbol(name=fn), args=[_rec(a) for a in expr.args])

    if isinstance(expr, IndexedSymbol) and isinstance(expr.base, Symbol):
        bn = expr.base.name
        actual_idx = _norm(expr.indices[0] if expr.indices else Symbol(name=formal_idx))
        if bn == op_name and _should_subst(actual_idx) and _spin_match(list(expr.indices)):
            return _apply_rhs(actual_idx)
        if bn == op_name + "'" and _should_subst(actual_idx) and _spin_match(list(expr.indices)):
            return _adjoint_ir(_apply_rhs(actual_idx), operator_names)
        return IndexedSymbol(base=_rec(expr.base), indices=expr.indices)

    if isinstance(expr, Add):
        return Add(args=[_rec(a) for a in expr.args])

    if isinstance(expr, (Mul, NonCommutativeMul)):
        cls = type(expr)
        return cls(args=[_rec(a) for a in expr.args])

    if isinstance(expr, Pow):
        return Pow(base=_rec(expr.base), exponent=_rec(expr.exponent))

    if isinstance(expr, IndexSum):
        # model_copy, not a fresh IndexSum: rebuilding it listed the two fields
        # that existed when this was written and dropped every one added since
        # — the written range (`lower`/`upper`), the placeholder pair, and now
        # `op`, which would turn a Π back into a Σ with no error anywhere.
        # This is the `measure_dim` walker bug (CLAUDE.md) in another file.
        return expr.model_copy(update={"body": _rec(expr.body)})

    return expr  # Symbol, Number, IndexedSymbol, etc.


# Spin operator definition patterns (Phase 0.5)
#
# **The subject's boundary is the PARTICLE (を) / the KEYWORD (as), not the type
# word itself.**  Both patterns used to require the type word IMMEDIATELY after
# it, so one qualifier was enough to lose the whole declaration — and what it
# fell through to was `define_scalar_parameter`.  Measured 2026-08-22:
#
#     I を核スピン演算子として定義して          → I registered as a SCALAR, "success"
#     I を ^{13}C の核スピン演算子として定義して → ditto
#     define I as the nuclear spin operator     → ditto
#
# The spin-declaration READING lives once, in
# `backend_handler.read_spin_operator_declaration` (2026-08-22): the subject's
# boundary is the PARTICLE, not the type word, so one qualifier — `核`スピン演算子,
# which is how every NMR text writes it — used to lose the whole declaration and
# fall through to `define_scalar_parameter`, registering a SCALAR and reporting
# success.  A byte-identical copy of the old regexes stayed HERE, imported by
# `julia_backend_handler` and never called: dead since that day, and measured
# dead 2026-08-23 (no `Load` of either symbol anywhere).  Deleted rather than
# left, because a duplicate nobody calls is one a later reader edits or revives.

# The spin type words themselves are `spin_def_op_text` in
# i18n/nl_patterns.json, asked at the one place that asks them
# (`julia_backend_handler.should_bypass_auto_register_formula`) — a guard, so it
# reads BOTH languages.


# Expression ID pattern — never a valid operator name.  Imported by
# julia_backend_handler; `_T\d+` scaffolding keys are deliberately NOT
# matched here, because the two call sites ask whether a SPIN/OPERATOR
# argument is an E-id the researcher wrote, not whether a string is a key
# into expressions (that question is expr_ref_fields.is_expr_ref).
_EXPR_ID_RE = re.compile(r'^E\d+$')

def _prime_bases(ir_node: "object") -> "set[str]":
    """Collect base names of prime/dagger-suffix operators in an IR tree.

    Used by operator_substitute and operator_substitute_pair to ensure
    creation operators like 'beta'' are added to _op_names so _adjoint_ir
    correctly recognises them even when not registered in SymbolContext.
    """
    from notebook.models.ir import (
        FunctionCall as _FC, Symbol as _Sym,
        Add as _A, Mul as _M, NonCommutativeMul as _NC,
    )
    result: set[str] = set()
    if isinstance(ir_node, _FC) and isinstance(ir_node.func, _Sym):
        fn = ir_node.func.name
        if fn.endswith("'") or fn.endswith("†"):
            result.add(fn.rstrip("'†"))
        for a in ir_node.args:
            result |= _prime_bases(a)
    elif isinstance(ir_node, (_A, _M, _NC)):
        for a in ir_node.args:
            result |= _prime_bases(a)
    return result

# Matches Julia operator identifiers: name( or name†( or name'(
_JULIA_OP_RE = re.compile(r'\b([a-zA-Z][a-zA-Z0-9_]*)(?:[†\']+)?\(')
# QuantumAlgebra.jl and Symbolics.jl built-in function names to exclude.
_JULIA_BUILTINS = frozenset({
    "normal_form", "comm", "vacExpVal", "heisenberg_eom",
    "expand", "simplify", "factor", "real", "imag",
    "string", "println", "print", "show",
    "param", "params",
})


# Args that hold CAS expressions (may contain implicit multiplication).
# Names, format strings, and scalar parameters are excluded.
_EXPRESSION_ARG_FIELDS: frozenset[str] = frozenset({
    "a", "b", "expr", "operator", "hamiltonian", "equation",
})


def _split_top_level_commas(s: str) -> "list[str]":
    """Split *s* on commas at bracket depth 0, so a 2-argument operator call
    like ``c(k,u)`` in a comma-joined list (e.g. ``"c(k,u), c(-k,d)"``) is
    kept intact instead of being shredded at its own internal comma.

    Used wherever a comma-joined basis/equation list may itself contain
    multi-argument operator calls (e.g. L-1's build_inverse_transform).
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]
