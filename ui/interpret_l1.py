"""L-1: linear basis transforms, canonical & metric constraints (parsers).

Pure rule-based parsers/helpers extracted from interpret_pipeline.py (Phase 1
refactor). run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re

from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.shared_parser import split_relation, strip_lhs
from ui.interpret_common import _EIDS_RE, split_name_list


# L-1: "E1〜E3 の逆変換を作って" / "E1,E2 を q_1, q_2 について逆変換して" /
# "Invert E4, E5, E6, E7 for c(k,u), c(-k,d), ..." / "Build the inverse
# transform of E4, E5 for x, y".  A linear basis change new_i = Σ_j M_ij old_j
# is inverted by solving the forward system for the OLD basis: old = M^{-1}
# new.  We therefore delegate to solve_system, contributing only the
# auto-identification of the old basis.  Externalized to
# i18n/nl_patterns.json["inv_transform_vars"/"inv_transform"] (named groups
# eids, vars).  Deliberately checked (via _try_equation_intercepts, L-1)
# before the generic matrix-ops "invert"/"inverse" alias scan, which would
# otherwise misroute a multi-equation basis-transform request to the
# single-matrix inverse_matrix operation.
# Function/constant tokens that are never part of the old basis (rotation by an
# angle uses cos/sin; these are coefficients, not basis vectors).
_NON_BASIS_TOKENS = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc", "exp", "log", "ln",
    "sqrt", "sinh", "cosh", "tanh", "abs", "i", "e",
})


def _identifiers_in_order(s: str) -> list[str]:
    """First-appearance, de-duplicated identifiers in a canonical expression string."""
    seen: list[str] = []
    for m in re.finditer(r"[A-Za-z_]\w*", s or ""):
        tok = m.group(0)
        if tok not in seen:
            seen.append(tok)
    return seen


# L-1: explicit-matrix inverse transform.  "M = [[u,-v],[v,u]]、旧基底 c_k, c_mk、
# 新基底 α_k, α_mk の逆変換を求めて" — give M and the two bases directly instead of
# N forward equations.
_MATRIX_M_RE = re.compile(r'M\s*=\s*(\[\[.*?\]\])', re.U | re.S)
# In Japanese, the 旧基底/新基底 keywords themselves (kanji) fall outside this
# ASCII-only character class and naturally stop the greedy match, but "old
# basis"/"new basis" are plain ASCII words the greedy form would otherwise
# swallow into the FIRST list (e.g. "old basis: a, b, new basis: c, d"
# capturing "a, b, new basis: c, d" as the old basis) — each character is
# guarded by a negative lookahead so the match still stops right before the
# other basis keyword, while remaining greedy (and stopping at kanji / string
# end) everywhere else, unchanged from before.
_BASIS_LIST = (r"((?:(?!旧基底|新基底|old\s*basis|new\s*basis)"
               r"[A-Za-z0-9_,\s'()+\-])+)")
_OLD_BASIS_RE = re.compile(r'(?:旧基底|old\s*basis)\s*[:：]?\s*' + _BASIS_LIST, re.U | re.I)
_NEW_BASIS_RE = re.compile(r'(?:新基底|new\s*basis)\s*[:：]?\s*' + _BASIS_LIST, re.U | re.I)


def _parse_bracket_matrix(s: str) -> "list[list[str]] | None":
    """Parse "[[u, -v], [v, u]]" into [["u","-v"],["v","u"]] (reuses the matrix parser)."""
    s = (s or "").strip()
    m = re.match(r"^\[(.*)\]$", s, re.S)
    if not m:
        return None
    return _parse_maxima_matrix("matrix(" + m.group(1) + ")")


def _parse_matrix_transform(text: str) -> "dict | None":
    """Detect an explicit-matrix inverse-transform request → {M, old, new}."""
    t = text or ""
    # Regression: this gate was a plain Japanese-only substring check, so it
    # never let English text reach _OLD_BASIS_RE/_NEW_BASIS_RE below even
    # though those are already bilingual ("old basis"/"new basis").
    has_inverse = "逆" in t or re.search(r'(?i)\b(?:invert|inverse)\b', t)
    has_transform = "変換" in t or re.search(r'(?i)\btransform(?:ation)?\b', t)
    if not (has_inverse and has_transform):
        return None
    mm = _MATRIX_M_RE.search(text)
    mo = _OLD_BASIS_RE.search(text)
    mn = _NEW_BASIS_RE.search(text)
    if not (mm and mo and mn):
        return None
    M = _parse_bracket_matrix(mm.group(1))
    old = [x.strip() for x in re.split(r"[,、]\s*", mo.group(1).strip()) if x.strip()]
    new = [x.strip() for x in re.split(r"[,、]\s*", mn.group(1).strip()) if x.strip()]
    if not M or len(M) != len(new) or any(len(r) != len(old) for r in M):
        return None
    return {"M": M, "old": old, "new": new}


def _parse_inverse_transform(text: str, lang: str = "ja") -> "dict | None":
    """Detect an inverse-basis-transform request and return its E-ids and (optional)
    explicit old basis, or None.  The actual inversion reuses solve_system."""
    m = nlp("inv_transform_vars", lang).search(text)
    if m:
        eids = _EIDS_RE.findall(m.group("eids"))
        vars_raw = split_name_list(m.group("vars"), lang)
        return {"eids": eids, "vars": vars_raw,
                "named": bool(pick_group(m, "named"))}
    m = nlp("inv_transform", lang).search(text)
    if m:
        eids = _EIDS_RE.findall(m.group("eids"))
        # **名指していない要求は claim しない。** 英語の裸の動詞 `invert` は
        # `inverse_matrix` の別名でもあり（スキーマ）、日本語には対応物が無い
        # ——`逆変換` という名詞を必ず書くので。名指されておらず、しかも E-id が
        # 1つ（＝この操作が受け取れない形）なら、これは逆基底変換の要求ではない。
        # 実測 2026-09-07: `invert E1` が「逆変換には最低2つの変換式が必要です」
        # ——研究者が書いていない変換について語る断り文を返していた。数の判定は
        # 呼び出し側にも在るが、あちらは**名指された**要求に理由を返すためのもの。
        if not pick_group(m, "named") and len(eids) < 2:
            return None
        return {"eids": eids, "vars": [], "named": bool(pick_group(m, "named"))}
    return None


# Operator-aware atom: keeps c(1), c'(2), c_k whole instead of collapsing to "c".
_BASIS_ATOM_RE = re.compile(r"[A-Za-z_]\w*'*(?:\([^)]*\))?")


def _basis_atoms_in_order(s: str) -> list[str]:
    """First-appearance, de-duplicated atoms, keeping operator forms (c(1), c'(2))
    whole so distinct quantum operators are not collapsed to their base name."""
    seen: list[str] = []
    for m in _BASIS_ATOM_RE.finditer(s or ""):
        tok = m.group(0)
        if tok not in seen:
            seen.append(tok)
    return seen


def _atom_base(tok: str) -> str:
    """Base name of an atom: c'(2)→c, c(1)→c, c_k→c_k, q1→q1."""
    m = re.match(r"[A-Za-z_]\w*", tok or "")
    return m.group(0) if m else (tok or "")


def _forward_has_operators(engine, eids: list[str]) -> bool:
    """Whether the forward equations' RHS reference any declared operator."""
    ctx = getattr(engine, "symbol_context", None)
    if ctx is None:
        return False
    for eid in eids:
        expr = engine.expressions.get(eid)
        if expr is None:
            continue
        latex = expr.latex or ""
        # `strip_lhs` returns the text unchanged when it is not an equation,
        # which is the old `else latex` branch — bracket-aware, so a forward
        # equation whose LEFT side carries a bounded sum keeps its head.
        rhs = expr.canonical or strip_lhs(latex)
        for tok in _basis_atoms_in_order(rhs):
            prop = ctx.get(_atom_base(tok))
            if prop and prop.quantity.kind == "operator":
                return True
    return False


def _infer_old_basis(engine, eids: list[str]) -> "list[str] | None":
    """Infer the old basis of a linear transform from the forward equations E-ids.

    new basis = the LHS of each equation; old basis = the RHS atoms (first-appearance
    order) minus the new-basis names, known function/constant tokens, and any symbol
    declared as a scalar parameter (a transform *coefficient*, e.g. u, v, a_ij).
    Operator atoms (c(1), c'(2)) are kept distinct.  Returns the basis when exactly
    len(eids) atoms remain, else None (caller prompts)."""
    new_vars: set[str] = set()
    ordered: list[str] = []
    seen: set[str] = set()
    ctx = getattr(engine, "symbol_context", None)
    for eid in eids:
        expr = engine.expressions.get(eid)
        if expr is None:
            return None
        latex = expr.latex or ""
        _rel = split_relation(latex)
        if _rel is not None and _rel[1] == "=":
            for nm in _basis_atoms_in_order(_rel[0]):
                new_vars.add(nm)
        # H-1 canonical stores the RHS only; fall back to the post-= latex slice.
        rhs = expr.canonical or strip_lhs(latex)
        for tok in _basis_atoms_in_order(rhs):
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)

    def _is_scalar_coeff(tok: str) -> bool:
        if ctx is None:
            return False
        q = ctx.get_quantity(_atom_base(tok)) or ctx.get_quantity(tok)
        return bool(q and getattr(q, "kind", None) == "scalar")

    basis = [t for t in ordered
             if t not in new_vars
             and _atom_base(t) not in _NON_BASIS_TOKENS
             and not _is_scalar_coeff(t)]
    return basis if len(basis) == len(eids) else None


def _new_basis_symbols(engine, eids: list[str]) -> list[str]:
    """Canonical LHS (new-basis) symbol of each forward equation, for block-local
    unbinding in the inverse-transform solve (H-1 binds these to their RHS)."""
    handler = getattr(engine, "_handler", None)
    out: list[str] = []
    for eid in eids:
        expr = engine.expressions.get(eid)
        latex = (expr.latex or "") if expr else ""
        # This LHS becomes the `new_basis` argument, which the inverse-transform
        # command unbinds block-locally in the CAS — so a mis-split is a broken
        # SYMBOL in a Maxima block, not a display.  Measured 2026-08-13: on
        # `\sum_{i=1}^{3} c_i = u a + v b` the naive split reported the new basis
        # as `\sum_{i`.
        _rel = split_relation(latex)
        if _rel is None or _rel[1] != "=":
            continue
        lhs = _rel[0]
        canon = lhs
        if handler is not None:
            try:
                canon = handler.latex_to_canonical(lhs) or lhs
            except Exception:
                canon = lhs
        canon = canon.strip()
        if canon and canon not in out:
            out.append(canon)
    return out


# L-1 (step 4): "E1[, E2…] の正準拘束を求めて" / "derive the canonical constraints
# of E1, E2" — derive the canonical (anti)commutation constraints that make a
# linear operator transform canonical (Bogoliubov: u²±v²=1).  No schema
# operation names this (it's a rule-based-only interceptor,
# _try_eq_canonical_constraints, that computes the (anti)commutators directly),
# so a miss here has no LLM fallback in any language — English needs its own
# pattern, not just a fallback.
def _parse_canonical_constraints(text: str, lang: str = "ja") -> "list[str] | None":
    """Detect a canonical-constraint request and return the forward-def E-ids.

    Grammar externalized to nl_patterns.json["canon_constraint"/"verify_canon"];
    session-language only (docs/28 §3.1)."""
    m = (nlp("canon_constraint", lang).search(text or "")
         or nlp("verify_canon", lang).search(text or ""))
    if not m:
        return None
    return _EIDS_RE.findall(m.group(0))


def _is_verify_request(text: str, lang: str = "ja") -> bool:
    """Whether the request asks to *verify* canonicality (verdict) rather than just
    derive the constraints."""
    return bool(nlp("verify_canon", lang).search(text or ""))


# L-1 (step 4, apply): reduce an expression under derived canonical constraints,
# e.g. "E2 を E5 の拘束で簡約" / "reduce E2 using E5's constraints".  Grammar and
# the simplify-verb / constraint-keyword guards are externalized to nl_patterns.json
# ("apply_constraints", "constr_simplify_verb", "constr_keyword").
def _parse_apply_constraints(text: str, lang: str = "ja") -> "dict | None":
    """Detect a constraint-application request.  Returns {target, relations} where
    relations is a list of constraint E-ids, or None to mean "use the session's
    derived canonical constraints"."""
    text = text or ""
    if not nlp("constr_simplify_verb", lang).search(text):
        return None
    if not nlp("constr_keyword", lang).search(text):
        return None
    m = nlp("apply_constraints", lang).search(text)
    if m:
        return {"target": m.group(1), "relations": _EIDS_RE.findall(m.group(2))}
    eids = _EIDS_RE.findall(text)
    if len(eids) == 1:
        return {"target": eids[0], "relations": None}
    return None


# L-1 (non-operator metric): declare an inner product ⟨e_i,e_j⟩ = value, building
# the old-basis Gram matrix used by the metric-preservation condition M G Mᵀ = G.
_INNER_PRODUCT_RE = re.compile(
    r'(?:⟨|\\langle)\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*(?:⟩|\\rangle)\s*'
    r'=\s*(.+?)\s*$',
    re.U,
)
def _parse_inner_product(text: str) -> "tuple[str, str, str] | None":
    """Detect an inner-product declaration ⟨a,b⟩ = v → (a, b, v)."""
    m = _INNER_PRODUCT_RE.search((text or "").strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3).strip()


def _parse_metric_constraints(text: str, lang: str = "ja") -> "list[str] | None":
    """Detect a metric-preservation-constraint request → forward-transform E-ids.

    Grammar externalized to nl_patterns.json["metric_constraint"]."""
    m = nlp("metric_constraint", lang).search(text or "")
    return _EIDS_RE.findall(m.group("eids")) if m else None


def _gram_rows(gram: dict, basis: list[str]) -> str:
    """Backend-neutral Gram matrix over *basis* as "r00, r01; r10, r11"; undeclared
    diagonal defaults to 1, undeclared off-diagonal to 0 (the rest is orthonormal).
    Each backend formats this into its own matrix literal."""
    rows = []
    for a in basis:
        row = []
        for b in basis:
            val = gram.get((a, b)) or gram.get((b, a))
            if val is None:
                val = "1" if a == b else "0"
            row.append(val)
        rows.append(", ".join(row))
    return "; ".join(rows)


def _parse_maxima_matrix(s: str) -> "list[list[str]] | None":
    """Parse 'matrix([a,b],[c,d])' into [[a,b],[c,d]] (paren/bracket-aware split)."""
    s = (s or "").strip()
    m = re.match(r"^matrix\((.*)\)$", s, re.S)
    if not m:
        return None
    body = m.group(1)
    rows, depth, cur = [], 0, []
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            rows.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        rows.append("".join(cur))

    def _cells(r: str) -> list[str]:
        r = r.strip()
        rm = re.match(r"^\[(.*)\]$", r, re.S)
        if not rm:
            return []
        inner, d, c, out = rm.group(1), 0, [], []
        for ch in inner:
            if ch in "([":
                d += 1
            elif ch in ")]":
                d -= 1
            if ch == "," and d == 0:
                out.append("".join(c).strip()); c = []
            else:
                c.append(ch)
        if c:
            out.append("".join(c).strip())
        return out

    return [_cells(r) for r in rows]


# L-1 step 4: apply a transform (its equations) to a target expression, routing by
# type — commutative `substitute` for scalars/coordinates, `operator_substitute`
# (order-preserving) for operators.  "E5, E6 を E10 に代入して" / "substitute E5, E6
# into E10".  Grammar externalized to nl_patterns.json["transform_subst"] (named
# groups srcs1/tgt1 and the JA reverse order tgt2/srcs2).
def _parse_transform_subst(text: str, lang: str = "ja") -> "tuple[list[str], str] | None":
    """Detect "S1, S2 を T に代入" (≥2 sources) → (source E-ids, target E-id)."""
    m = nlp("transform_subst", lang).search(text or "")
    if not m:
        return None
    gd = m.groupdict()
    srcs = gd.get("srcs1") or gd.get("srcs2")
    tgt = gd.get("tgt1") or gd.get("tgt2")
    return _EIDS_RE.findall(srcs), tgt


def _subst_lhs_rhs(engine, sid: str) -> "tuple[str, str] | None":
    """Extract (lhs, rhs) of the equation stored at E-id *sid* for substitution.
    Priority: clean lhs_alias_map key → canonical split → latex split."""
    expr = engine.expressions.get(sid)
    if expr is None:
        return None

    def _strip(s: str) -> str:
        return re.sub(r"^\[+|\]+$", "", s or "").strip()

    canon = _strip(expr.canonical)
    # Both halves become CAS operands — `substitute(lhs, rhs)` and
    # `operator_substitute`'s op_rules (whose op_name/formal_index are read off
    # the LHS) — so this split is code generation, not display.  Measured
    # 2026-08-13 through the real `E1, E2 を E3 に代入して` route, the naive one
    # sent Maxima
    #     ratsimp(subst(sum(c(i), i, 1, 3)=1}^3*c_i = u*a + v*b, c(1) + d^2))
    # for a forward equation with a bounded sum on the left: the replacement is
    # a fragment that still carries an `=`.  The step failed, but blaming a
    # DIFFERENT symbol, so the reason the researcher was shown was not the cause.
    #
    # `"=" in op` rather than `op == "="`: the old guard `"=" in canon` also
    # fired on `<=` / `>=`, and an inequality does reach here, so which
    # relations count is kept as written and only the SCAN changes.  The LHS it
    # yields is no longer mangled, though — `y <= x^2 + a` gave the op name
    # `y <` before (measured), and `_op_name_index` was handed that.
    _rel_can = split_relation(canon)
    _is_eq_can = _rel_can is not None and "=" in _rel_can[1]
    cands = [k for k, v in getattr(engine, "lhs_alias_map", {}).items()
             if v == sid and "[" not in k]
    cands.sort(key=lambda k: (k.startswith("\\"), k))
    if cands:
        lhs = cands[0]
        if lhs.startswith("\\"):
            lhs = engine._handler.lhs_latex_to_canonical(lhs) or lhs
        return lhs, (_rel_can[2] if _is_eq_can else canon)
    if _is_eq_can:
        return _rel_can[0], _rel_can[2]
    lat = _strip(expr.latex)
    # `==` needs no guard of its own; find_top_level_relation excludes it.
    _rel_lat = split_relation(lat)
    if _rel_lat is not None and "=" in _rel_lat[1]:
        return _rel_lat[0], _rel_lat[2]
    return None


def _op_name_index(lhs: str) -> "tuple[str, str | None]":
    """Split an operator LHS into (base, index): c_k→(c,k), c(k)→(c,k), c_1→(c,1)."""
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(?:_\{?|\()([A-Za-z0-9+\-]+)\}?\)?$",
                 (lhs or "").strip())
    return (m.group(1), m.group(2)) if m else ((lhs or "").strip(), None)


def _is_operator_symbol(engine, lhs: str) -> bool:
    """Whether the base name of *lhs* is a declared operator in the SymbolContext."""
    ctx = getattr(engine, "symbol_context", None)
    if ctx is None:
        return False
    base, _ = _op_name_index(lhs)
    prop = ctx.get(base)
    return bool(prop and prop.quantity.kind == "operator")


def _operator_statistics(engine, canonical: str) -> str:
    """Return 'fermi' or 'bose' for the operators appearing in *canonical*, read
    from SymbolContext.  Bogoliubov preserves statistics, so the (anti)commutator
    to test is the anticommutator for fermions and the commutator for bosons."""
    ctx = getattr(engine, "symbol_context", None)
    if ctx is None:
        return "bose"
    toks = set(_identifiers_in_order(canonical))
    for name, entry in ctx.registry.items():
        if name in toks and entry.quantity.kind == "operator":
            if entry.statistics.kind == "fermi":
                return "fermi"
    return "bose"
