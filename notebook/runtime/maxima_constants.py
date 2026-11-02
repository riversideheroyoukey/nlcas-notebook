r"""Maxima's named constants, and their way back to LaTeX.

One direction only: a constant token that came OUT of Maxima (or that
``normalize_bound`` produced on the way in) → what the researcher should see.
The way in is ``MaximaBackendHandler.normalize_bound``.

Split out of ``maxima_backend_handler`` for the reason that file's hygiene
comment gives — the module boundary, not the ratchet, is what keeps it small —
and it is a clean unit: a table, the scan that applies it, and no other state.

Its callers pass integration BOUNDS and limit POINTS (``lhs_formatting``,
``integral_form``, ``engine_deferred_forms_mixin``), never a general
expression.  That is what makes a bare word safe to translate here.
"""
from __future__ import annotations

import re

MAXIMA_CONST_MAP: dict[str, str] = {
    "%pi":  r"\pi",
    "%inf": r"\infty",
    "%e":   r"e",
    "%i":   r"i",
    "minf": r"-\infty",
    "inf":  r"\infty",
}

# `minf` is listed BEFORE the bare `inf` so negative infinity is never read as a
# symbol `m` times infinity; the lookbehind refuses the same thing again.
#
# The bare spelling is the one that actually occurs.  `normalize_bound` writes
# `\infty` → `inf` (Maxima's own name for it — `%inf` is not Maxima syntax), and
# until 2026-08-15 only `minf` had a way back, so `\int_0^\infty f dt` was
# displayed, and exported, as `\int_{0}^{inf}`: the researcher's own upper bound
# returned to them as the italic word "inf".  It surfaced when the Laplace
# scenarios were rewritten to reach the transform through the general
# `integrate_definite` — the macro built its own display and never asked.
MAXIMA_CONST_RE = re.compile(
    r"%pi|%inf|%e|%i|minf|(?<![A-Za-z0-9_])inf(?![A-Za-z0-9_])")


def constant_to_latex(token: str) -> str:
    """Rewrite every Maxima constant in *token* to its LaTeX spelling."""
    return MAXIMA_CONST_RE.sub(lambda m: MAXIMA_CONST_MAP[m.group()], token)


#: Maxima EVALUATES `sum(...)` / `product(...)` the moment it reads them, so a
#: bounded aggregate becomes its terms before anything can look at it.  That is
#: one root with two faces, both measured 2026-08-15 and both reported as
#: successes:
#:
#:   * `subst(sum(c(i),i,1,3) = u*a+v*b, H)` never matches — the pattern is three
#:     terms by the time `subst` runs — so the target came back with its sums
#:     merely EXPANDED and no substitution done
#:     (open_items TRANSFORM-SUBST-CHAIN);
#:   * `subst(i=m, sum(c(i),i,1,3))` has no `i` left to rename, so
#:     `relabel_index` returned the expanded sum whether or not the index it was
#:     given even existed (open_items NOOP-REWRITE-FAMILY).
#:
#: Quoting makes the aggregate a NOUN, which is the form both operations need:
#: measured, `subst('sum(c(i),i,1,3) = u*a+v*b, 'sum(c(i),i,1,3) + y)` answers
#: `y+b*v+a*u` and `subst(i=m, 'sum(c(i),i,1,3))` answers `'sum(c(m),m,1,3)`.
#: `ev(…, nouns)` evaluates it later, `tex1` renders it as the researcher's own
#: `\sum_{i=1}^{3}{c(i)}`, and the reader for `'sum(…)` already exists (B-10 ②-e).
#:
#: `\b` keeps `nusum(`, `lsum(` and friends out: their `sum(` has no word
#: boundary in front of it.
_AGGREGATE_CALL_RE = re.compile(r"(?<!')\b(sum|product)\(")


def quote_aggregates(code: str) -> str:
    """Turn every unquoted ``sum(``/``product(`` in *code* into its noun form."""
    return _AGGREGATE_CALL_RE.sub(lambda m: "'" + m.group(1) + "(", code)
