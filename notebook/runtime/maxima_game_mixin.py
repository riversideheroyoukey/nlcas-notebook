"""MaximaBackendHandler mixin: F-17 game-theory command builders
(game_theory domain).

Payoff/bimatrix REPRESENTATION needs no new code — the existing `[a,b;c,d]`
matrix literal already covers it (notebook/runtime/shared_parser.py).  Only
the equilibrium CONCEPTS below are new, and both stay within Maxima's own
list/comparison machinery so the whole check is ONE deterministic command
(no Python-side numeric evaluation of matrix entries — Maxima resolves
comparisons via ``is()``, which also degrades safely to an unresolved
condition for symbolic entries rather than silently guessing).

Scope (deliberately narrow, matching the "add only the missing primitive"
discipline used throughout F-14/F-16 this session): pure-strategy Nash
equilibria (any m×n bimatrix) and mixed-strategy indifference conditions
(the standard 2×2 case).  Iterated elimination of dominated strategies is
NOT implemented — it is a genuinely separate, multi-round algorithm rather
than a single missing primitive; deferred as a documented follow-up (see
docs/scenarios/scenario_matching_pennies.md's "対応範囲と制限").

Extracted as its own mixin (not appended directly to
notebook/runtime/maxima_backend_handler.py) for the same hygiene-ratchet
reason as MaximaEconMixin.
"""
from __future__ import annotations

from notebook.runtime.backend_handler import BuildResult
# `strip_lhs` REPLACES the `_strip_h1_lhs` this module used to define and
# maxima_signal_mixin used to copy verbatim.  Same three lines, same
# question — and both got it wrong on `\sum_{i=1}^{3} c_i = u*x + v*y`
# (measured 2026-08-13: `1}^{3} c_i = u*x + v*y`).  The comment that used
# to sit here called duplicating small helpers this codebase's precedent;
# it is the precedent ONE-QUESTION-MANY-ANSWERS exists to end.
from notebook.runtime.shared_parser import strip_lhs


def _split_top_level(s: str, sep: str = ",") -> "list[str]":
    """Split *s* on top-level occurrences of *sep* only (ignoring ones nested
    inside (), [], {}).  Duplicated (small, single-purpose) from
    maxima_econ_mixin.py rather than imported, to keep each domain mixin
    self-contained."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def parse_matrix_rows(canonical: str) -> "list[list[str]]":
    """Parse a Maxima ``matrix([a, b], [c, d], ...)`` canonical string into
    rows of entry strings.  Returns [] if *canonical* is not a matrix
    literal (structural parse only — no numeric evaluation)."""
    s = canonical.strip()
    if not (s.startswith("matrix(") and s.endswith(")")):
        return []
    inner = s[len("matrix("):-1]
    rows: list[list[str]] = []
    for row_str in _split_top_level(inner, ","):
        row_str = row_str.strip()
        if not (row_str.startswith("[") and row_str.endswith("]")):
            return []
        rows.append(_split_top_level(row_str[1:-1], ","))
    return rows


class MaximaGameMixin:
    """F-17: game-theory command builders (see module docstring)."""

    def _build_pure_nash_equilibria(self, args: dict, expr_lookup: dict) -> BuildResult:
        a_ref = str(args.get("payoff_a", ""))
        b_ref = str(args.get("payoff_b", ""))
        a_canon = strip_lhs(expr_lookup.get(a_ref, a_ref))
        b_canon = strip_lhs(expr_lookup.get(b_ref, b_ref))
        rows_a = parse_matrix_rows(a_canon)
        rows_b = parse_matrix_rows(b_canon)
        if not rows_a or not rows_b:
            return BuildResult(
                ok=False, command="",
                error="pure_nash_equilibria: payoff_a and payoff_b must both "
                      "be registered matrices (e.g. [2,0;0,1])")
        m, n = len(rows_a), len(rows_a[0])
        if len(rows_b) != m or any(len(r) != n for r in rows_a + rows_b):
            return BuildResult(
                ok=False, command="",
                error="pure_nash_equilibria: payoff_a and payoff_b must have "
                      "the same shape")
        command = (
            f"block([A: {a_canon}, B: {b_canon}, m: {m}, n: {n}],\n"
            f"  sublist(create_list([i,j], i,1,m, j,1,n),\n"
            f"    lambda([c], is(equal(A[c[1],c[2]], "
            f"lmax(makelist(A[k,c[2]], k,1,m))))\n"
            f"      and is(equal(B[c[1],c[2]], "
            f"lmax(makelist(B[c[1],k], k,1,n)))))))"
        )
        return BuildResult(ok=True, command=command)
