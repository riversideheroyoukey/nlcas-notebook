"""Static scan of a generated LaTeX export.

The published examples (``domains/*/examples/*.tex``) are the practice loop's
own output, and until 2026-08-14 nothing looked at them: five defects were found
by a researcher READING one file, and every one of them has a mechanical shape —
a `%` inside math, a double subscript, `<` in text mode, an over-escaped `\\^\\{\\}`,
a Greek letter spelled as a bare word.  Four of the five had been sitting in
committed examples.

This module is that missing reader.  It needs no TeX engine: the questions are
about the SOURCE, so they are answered by scanning it.  Compiling would answer a
different (also useful) question and is not available everywhere — see
open_items TEX-EXPORT-SCAN for what a compile pass would add.

Each check returns findings as ``(line_number, code, detail)``.  A check that
cannot decide reports nothing: this guard runs over every committed example, so
a false positive costs more than a miss.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Splitting a .tex into MATH and TEXT regions.  Almost every check below asks
# its question of one region or the other, and asking it of the whole file is
# how a scan gets a reputation for false positives ("`_` is fine in text mode",
# "`<` is fine in math mode").
# ---------------------------------------------------------------------------
_MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*")
_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(re.escape(e) for e in _MATH_ENVS) + r")\}"
    r"(.*?)\\end\{\1\}", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def math_spans(text: str) -> "list[tuple[int, int]]":
    """(start, end) of every math region — display environments and inline `$…$`."""
    spans = [(m.start(2), m.end(2)) for m in _ENV_RE.finditer(text)]
    for m in _INLINE_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in spans):
            spans.append((m.start(1), m.end(1)))
    return sorted(spans)


def text_spans(text: str) -> "list[tuple[int, int]]":
    """The complement of :func:`math_spans` — everything typeset as prose."""
    out, prev = [], 0
    for s, e in math_spans(text):
        if s > prev:
            out.append((prev, s))
        prev = max(prev, e)
    if prev < len(text):
        out.append((prev, len(text)))
    return out


# ---------------------------------------------------------------------------
# The checks.  Each one exists because a specific defect reached a committed
# example; the docstring names it so a future reader can tell a real finding
# from a rule someone invented.
# ---------------------------------------------------------------------------

# `%` starts a LaTeX COMMENT.  Maxima writes `%e`/`%i`/`%pi`, so a canonical
# pasted into a math slot silently DELETES the rest of its line — measured in
# scenario_damped_oscillator_numeric_jp E6, where the whole first line of the
# published solution was gone with no error anywhere.
_COMMENT_IN_MATH = re.compile(r"(?<!\\)%")

# Two subscripts on one base: `x_{ic}_{1}`, a HARD LaTeX error ("Double
# subscript"), so the file does not typeset at all.  Maxima's tex1 splits a
# trailing digit run off an italicised identifier and produced exactly this.
_DOUBLE_SCRIPT = re.compile(r"_(?:\{[^{}]*\}|\\?[A-Za-z0-9])\s*_")

# `<` / `>` are not LaTeX specials, but the preamble deliberately omits fontenc
# (it breaks uplatex), so OT1 is in force and they set as `¡` / `¿`.
_ANGLE_IN_TEXT = re.compile(r"(?<!\\)[<>]")

# A replacement that got escaped again by the next row of a sequential escape
# table: `x^2` came out `x\^\{\}2`.
_OVER_ESCAPED = re.compile(r"\\(?:\^|textbackslash|textasciitilde)\\\{\\\}")

# CAS syntax that must never reach the researcher (Core Principle #1), plus the
# adapter's own scratch name, which has been registered as a RESULT before.
_CAS_LEAK = re.compile(r"%e\^|%i(?![A-Za-z])|%pi|_res_|_nb_E\d")

# A Greek letter spelled as a bare word inside math sets as the italic WORD
# "omega", not ω.  Measured in the assume cards (`$\displaystyle omega > 0$`)
# and, earlier, in `\dot{theta}` (EXAMPLES-DRIFT).
_GREEK = ("alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|"
          "mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega")
_BARE_GREEK = re.compile(r"(?<![\\A-Za-z])(" + _GREEK + r")(?![A-Za-z])")

# Maxima's product operator surviving into a math slot.  A SCRIPTED star is not
# one: `m^*` (effective mass), `Q^*` / `x^*` (optimal value) are ordinary physics
# and OR notation, and the first version of this check reported 23 of them and
# nothing else — a guard that runs over every committed example cannot afford
# that, so the star must sit BETWEEN OPERANDS to count.
_BARE_STAR = re.compile(r"(?<![\\{^_])\*(?!\})")


def scan(text: str) -> "list[tuple[int, str, str]]":
    """Every finding in *text*, as (line, code, detail)."""
    findings: list[tuple[int, str, str]] = []
    maths = math_spans(text)
    texts = text_spans(text)

    def add(pos: int, code: str, detail: str) -> None:
        findings.append((_line_of(text, pos), code, detail))

    for s, e in maths:
        body = text[s:e]
        for rx, code in ((_COMMENT_IN_MATH, "comment-in-math"),
                         (_DOUBLE_SCRIPT, "double-subscript"),
                         (_CAS_LEAK, "cas-syntax-leak"),
                         (_BARE_GREEK, "bare-greek-word"),
                         (_BARE_STAR, "bare-star-in-math")):
            for m in rx.finditer(body):
                lo = max(0, m.start() - 25)
                add(s + m.start(), code, body[lo:m.end() + 25].replace("\n", " "))

    for s, e in texts:
        body = text[s:e]
        for rx, code in ((_ANGLE_IN_TEXT, "angle-bracket-in-text"),
                         (_OVER_ESCAPED, "over-escaped")):
            for m in rx.finditer(body):
                lo = max(0, m.start() - 25)
                add(s + m.start(), code, body[lo:m.end() + 25].replace("\n", " "))

    # Whole-file well-formedness.  Only reported when clearly broken — an odd
    # `$` count means an unclosed inline math run, which swallows prose into a
    # formula (or the reverse) for the rest of the document.
    if len(re.findall(r"(?<!\\)\$", text)) % 2:
        findings.append((0, "unbalanced-dollar", "odd number of unescaped $"))
    if text.count("{") != text.count("}"):
        findings.append((0, "unbalanced-brace",
                         f"{text.count('{')} open vs {text.count('}')} close"))
    if r"\begin{aligned}" in text and r"\usepackage{amsmath" not in text:
        findings.append((0, "missing-package", "aligned without amsmath"))
    return sorted(findings)
