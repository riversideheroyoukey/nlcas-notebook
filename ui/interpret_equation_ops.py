"""Equation operations (H-5/H-7/H-8/H-9): solve, both-sides ops, and the
RHS-canonical → operation parser.

Pure rule-based parsers/helpers extracted from interpret_pipeline.py (Phase 1
refactor). run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re

from notebook.runtime.nl_patterns import fragment, nlp
from ui.interpret_common import _EIDS_RE, split_name_list


# H-5/H-7: "En の右辺を <操作> して" / "<verb> the right-hand side of En" —
# operate on the RHS of a registered equation.  Externalized to
# i18n/nl_patterns.json["rhs_op"] (named groups eid, keyword — same names
# despite the JA/EN word-order reversal, so callers don't need to know which
# language matched).
# Keyword → operation name.  None = use _parse_rhs_to_operation logic (計算/評価).
_RHS_OP_MAP: dict[str, "str | None"] = {
    "展開":       "expand",
    "因数分解":   "factor",
    "有理式簡約": "rational_simplify",
    "有理化":     "rational_simplify",
    "整理":       "simplify",
    "計算":       None,
    "評価":       None,
    "expand":      "expand",
    "factor":      "factor",
    "rationalize": "rational_simplify",
    "rationalise": "rational_simplify",
    "compute":     None,
    "evaluate":    None,
}

#: 簡約の綴りは**この表に書かない**——`simplify_verb_ja` / `simplify_verb_en` に
#: 訊く。`整理` だけは表に残る: あれは曖昧語なので fragment に入れないと決めてある
#: （三十七問目）。
_SIMPLIFY_FRAGMENTS = ("simplify_verb_ja", "simplify_verb_en")


def rhs_operation_for(keyword: str) -> "str | None":
    """右辺に対する keyword → 操作名。`None` は「計算/評価」経路を意味する。

    **正規表現を広げるだけでは足りない。** この表は値（研究者が書いた語そのもの）
    で引くので、`rhs_op` の alternation に簡約の語彙を差し込んでも、ここが知らな
    ければ `None` に落ちて `_parse_rhs_to_operation` が**式の形から操作を引き直す**
    ——実測 2026-09-07、`C = [A, B]` の右辺に対して `E1 の右辺を簡約して` が
    **`commutator` を実行した**（研究者は「簡約」と書いている）。
    綴りを書き足すと4箇所目になるので、共有 fragment に訊く。
    """
    if keyword in _RHS_OP_MAP:
        return _RHS_OP_MAP[keyword]
    for name in _SIMPLIFY_FRAGMENTS:
        if re.fullmatch(fragment(name), keyword, re.I):
            return "simplify"
    return None


#: 辺の綴り → `"left"` / `"right"`。`右辺`・`RHS`・`right side` は同じ辺である。
def side_of(match) -> str:
    """一致から**どちらの辺か**を読む。読めなければ右（既存の既定）。

    綴りは `右辺`/`左辺`、`right[- ]hand side`、`right side`、`RHS`/`LHS` と
    いろいろあるが、辺は2つしかない。`re` は alternation をまたいだ同名グループを
    禁じるので枝ごとに綴り分けてあり、`pick_group` が値の入っている枝を取る。
    """
    from notebook.runtime.nl_patterns import pick_group
    raw = (pick_group(match, "side") or pick_group(match, "sideB") or "").strip()
    low = raw.lower()
    if raw == "左辺" or low in ("left", "l"):
        return "left"
    return "right"


def _match_rhs_op(text: str, lang: str = "ja") -> "tuple[str, str, str] | None":
    """Return (eid, keyword, side) for an 'operate on one SIDE of En' request.

    `side` は 2026-09-05 に足した。それまで右辺しか読めず、`右辺` に当たらない
    綴り（`right side`・`RHS`・`left-hand side`）は**別名走査に落ちて式全体に
    操作が適用され、成功が返っていた**（`simplify the RHS of E1` の正準形は
    `x^2+a*x=0`——等式まるごと）。
    """
    m = nlp("rhs_op", lang).search(text)
    if not m:
        return None
    return (m.group("eid"), m.group("keyword").lower(), side_of(m))

# H-9: solve patterns, externalized to i18n/nl_patterns.json (named groups):
#   solve_system_vars — "E1 と E2 の連立方程式を X と Y について解いて" /
#                        "solve E1 and E2 for x and y"  (eids, vars)
#   solve_system      — "E1 と E2 の連立方程式を解いて" / "solve E1 and E2"  (eids)
#   solve_single      — "E1 を X について解いて" / "solve E1 for X"  (eid, var)


def _solve_vars_from(text: str) -> "list[str]":
    """Split a solve-for list — `x, y and z` / `x, y, z` — into names.

    `and` を別の `re.sub` で先に消してから切っていた（`および` は消えないので
    助詞が変数になった）。区切りの答えは `interpret_common.split_name_list` に
    1つだけ置いてある。"""
    return split_name_list(text)


def _parse_solve(text: str, lang: str = "ja",
                 alias_free_text: str = "") -> "tuple[str, dict] | None":
    """Detect solve patterns and return (operation, args) or None.

    *text* has been through 2a-sym, which rewrites the researcher's symbols
    into E-IDs so the supervisor can tell an expression from a name.  The
    equation list NEEDS that rewrite; the SOLVE-FOR VARIABLES must not have it,
    and a variable that happens to be an equation's left side was being
    replaced by its expression id — `solve E1 and E2 and E3 for a, b, c` sent
    `solve([…], [a, b, E3])` and answered for `a` and `b` as a success
    (measured 2026-08-16, with `c = 2` among the equations: an ordinary thing
    to write in a linear system).  So the variables are re-read from
    *alias_free_text* when it is given, and only they are.
    """
    # System with explicit solve-for vars
    m = nlp("solve_system_vars", lang).search(text)
    if m:
        # 語順が2通りある（`E1 と E2 を x と y について解いて` /
        # `x と y について E1 と E2 を解いて` / `solve for x and y in E1 and E2`）。
        # `re` は alternation をまたいだ同名グループを禁じるので枝ごとに綴り分けて
        # あり、`pick_group` が値の入っている枝を取る——`solve_single` が既に
        # そうしている。**英語の変数先行の語順には鍵が無く、別名走査が答えて
        # 1本目だけを解いて `completed` を返していた**（実測 2026-09-12、
        # `solve for x and y in E1 and E2` → `3-y`。正解は x=2, y=1）。
        from notebook.runtime.nl_patterns import pick_group
        eids = _EIDS_RE.findall(pick_group(m, "eids") or "")
        vars_raw = _solve_vars_from(pick_group(m, "vars") or "")
        if alias_free_text:
            raw = nlp("solve_system_vars", lang).search(alias_free_text)
            if raw is not None:
                vars_raw = _solve_vars_from(pick_group(raw, "vars") or "") or vars_raw
        return ("solve_system", {
            "equations": ", ".join(eids),
            "vars": ", ".join(vars_raw),
        })
    # System without explicit vars (Maxima auto-detects)
    m = nlp("solve_system", lang).search(text)
    if m:
        eids = _EIDS_RE.findall(m.group("eids"))
        # vars will be filled by pending_op prompt if needed
        return ("solve_system", {
            "equations": ", ".join(eids),
            "vars": "",   # empty → will prompt
        })
    # Single equation, NO variable named.  The system branch above has always
    # answered this by handing over an empty `vars` and letting the caller
    # prompt; the single-equation branch had no such reading at all, so
    # `E1 を解いて` reached no rule and came back "could not interpret" —
    # while English `solve E1` reached the operation through the alias scan and
    # said which argument was missing.  One question, three answers (measured
    # 2026-09-05); this is the third made to agree with the other two.
    m = nlp("solve_single_novar", lang).search(text)
    if m and not nlp("solve_single", lang).search(text):
        _eid = m.groupdict().get("eid") or m.groupdict().get("eid2")
        return ("solve_equation", {"equation": _eid, "var": ""})
    # 「E1 から x を求めて」/「find x from E1」、
    # 「E1 の解を x について求めて」/「find the solution of E1 for x」——
    # どちらも実在する言い方で、どの規則も読めず失敗していた（実測 2026-09-06）。
    # `solve_single` の**手前**に置く: 語順が違うだけで同じ要求なので、先に
    # 特定の形が答え、当たらなければ従来どおり `solve_single` が見る。
    for _key in ("solve_from_expr", "solve_solution_of_var"):
        _m2 = nlp(_key, lang).search(text)
        if _m2:
            _v = _m2.group("var").strip()
            if alias_free_text:
                _raw2 = nlp(_key, lang).search(alias_free_text)
                if _raw2 is not None:
                    _v = _raw2.group("var").strip() or _v
            return ("solve_equation", {"equation": _m2.group("eid"), "var": _v})
    # Single equation with explicit var
    m = nlp("solve_single", lang).search(text)
    if m:
        # 語順が2通りある（`E1 を x について解いて` / `x について E1 を解いて`）。
        # `re` は alternation をまたいだ同名グループを禁じるので枝ごとに綴り分けて
        # あり、`pick_group` が値の入っている枝を取る（二十問目の辺の読みと同じ）。
        from notebook.runtime.nl_patterns import pick_group
        var_raw = (pick_group(m, "var") or "").strip()
        # Re-read the variable from the researcher's own text — the same reason
        # the system branch above does, and it was fixed there ONLY.  2a-sym
        # rewrites a symbol that is some equation's left side into that
        # equation's E-id, and a solve-for variable is exactly such a symbol:
        # in a session that had defined `G`, `E33 を G^{(HF)}_{n\sigma}(\omega)
        # について解いて` reached Maxima as `solve(…, E1_psupHF[n,sigma](omega))`
        # and answered "no solution" (measured 2026-08-21).
        if alias_free_text:
            _raw_single = nlp("solve_single", lang).search(alias_free_text)
            if _raw_single is not None:
                var_raw = (pick_group(_raw_single, "var") or "").strip() or var_raw
        # Convert LaTeX time-derivative notation to canonical form:
        # \ddot{q} → q_ddot,  \dot{q} → q_dot  (then build_command applies time_dep_map)
        var_raw = re.sub(r'\\ddot\{([A-Za-z]\w*)\}', r'\1_ddot', var_raw)
        var_raw = re.sub(r'\\dot\{([A-Za-z]\w*)\}', r'\1_dot', var_raw)
        # Everything else is handed over AS WRITTEN.  The builder already lowers
        # this operand through the backend's own door
        # (`_solve_operand_to_backend` → `to_backend_code`), and the equation
        # goes through the SAME door — so a second, cruder lowering here could
        # only disagree with it, and did: stripping the backslashes merged
        # `G_{n\sigma}` into the single name `G_{nsigma}`, which appears nowhere
        # in the equation, and Maxima answered "no solution" for a variable the
        # researcher had written correctly (measured 2026-08-21).  The dot forms
        # above stay because they feed `apply_time_dep_map`, not the door.
        return ("solve_equation", {
            "equation": pick_group(m, "eid"),
            "var": var_raw,
        })
    return None

# H-8: "En の両辺を X で割る/にかける/に足す/から引く" — operate on both sides.
# NL patterns are externalized to i18n/nl_patterns.json (named-group regexes per
# language); each operation is tried independently.  Named groups: eid, operand.
_BOTH_SIDES_KEYS = (
    ("both_sides_divide",   "divide"),
    ("both_sides_multiply", "multiply"),
    ("both_sides_times",    "multiply"),
    ("both_sides_add",      "add"),
    ("both_sides_subtract", "subtract"),
)

def _parse_both_sides(text: str, lang: str = "ja") -> "tuple[str, str, str | None] | None":
    """Parse "En の両辺を X で割る" / "Divide both sides of En by X" style text.

    Returns (eid, operation, operand) where operation is one of:
      "divide", "multiply", "add", "subtract", "simplify"
    and operand is the canonical string (or None for simplify).
    Returns None when no pattern matches.
    """
    for key, op in _BOTH_SIDES_KEYS:
        m = nlp(key, lang).search(text)
        if m:
            return (m.group("eid"), op, m.group("operand").strip())
    m = nlp("both_sides_simplify", lang).search(text)
    if m:
        return (m.group("eid"), "simplify", None)
    return None

# LaTeX for new LHS after both-sides operation
def _both_sides_new_lhs(lhs: str, operation: str, operand: str | None) -> str:
    """Build display LaTeX for the transformed LHS."""
    if operation == "divide" and operand:
        return rf"\frac{{{lhs}}}{{{operand}}}"
    if operation == "multiply" and operand:
        return rf"{operand} {lhs}"
    if operation == "add" and operand:
        return rf"{lhs} + {operand}"
    if operation == "subtract" and operand:
        return rf"{lhs} - {operand}"
    return lhs  # simplify: LHS unchanged


# Detect commutator(a, b) in canonical
_COMMUTATOR_CANON_RE = re.compile(r'^commutator\((.+),\s*(.+)\)$', re.S)
# [A, B] bracket notation (produced by latex_to_canonical for bracket inputs)
_COMMUTATOR_BRACKET_RE = re.compile(r'^\[(.+),\s*(.+)\]$', re.S)
# Detect diff(expr, var) or diff(expr, var, n)
# Use [a-zA-Z_]\w* for var so the greedy .+ cannot steal an identifier-named var
# (e.g. diff(q, t, 2) → expr="q", var="t", order="2")
_DIFF_CANON_RE = re.compile(r'^diff\((.+),\s*([a-zA-Z_]\w*)(?:,\s*(\d+))?\)$', re.S)


def _parse_rhs_to_operation(rhs_canonical: str) -> "tuple[str, dict] | None":
    """Try to parse a RHS canonical string into a (operation, args) pair.

    Returns None when the RHS does not match a known single-operation pattern
    (caller should fall back to simplify).
    """
    s = rhs_canonical.strip()
    m = _COMMUTATOR_CANON_RE.match(s)
    if m:
        return ("commutator", {"a": m.group(1).strip(), "b": m.group(2).strip()})
    m = _COMMUTATOR_BRACKET_RE.match(s)
    if m:
        return ("commutator", {"a": m.group(1).strip(), "b": m.group(2).strip()})
    m = _DIFF_CANON_RE.match(s)
    if m:
        args: dict = {"expr": m.group(1).strip(), "var": m.group(2).strip()}
        if m.group(3):
            args["order"] = m.group(3)
        return ("differentiate", args)
    return None
