r"""An extremum written with a bound index — ``\max_i x_i``, ``\sup_{x \in X} f(x)``.

A *descriptor* is a plain dict carried on ``Expression.extremum_descriptor``::

    {"op": "max", "index_latex": "i", "body_latex": "x_i"}

**Why this is an object rather than an expression.**  Maxima has no "extremum
over an index": measured 2026-08-16, ``lmax(makelist(x[i],i,1,3))`` works and
``lmax(makelist(x[i],i,1,n))`` FAILS, so a symbolic range has no command at
all.  Read as an ordinary subscript it lowered to ``max(i) * x(i)`` — and
Maxima's ``max`` of one argument IS that argument, so ``\max_i x_i`` computed
as ``i·x(i)``: valid CAS text, a different quantity, silent.

**Why it is not simply refused.**  It was, for part of 2026-08-16, and that
stopped the notation being WRITABLE — a researcher could not put their own
objective function in their own notebook.  ``\max_i`` and ``\sup_x`` are
everyday notation in optimisation, analysis, statistics and machine learning,
so this is the same shape as ``lattice_descriptor`` and ``integral_descriptor``:
**write it down now, refuse at the point of use.**

**Why not `IndexSum.op`.**  That node's contract is that Σ and Π "differ ONLY
in the operator applied to the terms: same dummies, same ranges, same
placeholder pairs".  An extremum does not qualify — a range-less sum is given a
symbolic placeholder pair (``nbSumLo1``), and a symbolic range is exactly what
``lmax`` cannot take.  Riding that node would turn a silent wrong answer into a
silent failure.

These helpers are pure: no CAS, no engine.  The eventual computable case (a
concrete integer range → ``lmax(makelist(…))``) belongs here too when it is
built — see open_items EXTREMUM-BINDER.
"""
from __future__ import annotations

import re

from notebook.runtime.latex_typography import braced_group

#: The binder names.  ``\arg\max`` / ``\arg\min`` arrive with the ``\arg``
#: already separated (`_separate_adjacent_commands`), so they are matched as a
#: two-command prefix rather than one name.
_OPS: tuple[str, ...] = ("max", "min", "sup", "inf", "limsup", "liminf",
                         "argmax", "argmin")

#: ``\max``, ``\sup``, ``\arg\max`` — the binder NAME only.  Its subscript and
#: optional superscript are read separately, with balanced braces: written as
#: one regex the subscript could not nest (``\max_{i \in \{1,2,3\}}`` was not
#: claimed at all) and the superscript was left in the BODY
#: (``\max_{i=1}^{3} x_i`` gave a body of ``^{3} x_i`` — measured 2026-08-16,
#: in code written the same day).
_HEAD_RE = re.compile(
    r"(?<![A-Za-z])\\(?:(arg)\s*\\)?(" + "|".join(_OPS) + r")\s*(?=_)")


def _strip(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith("{") and tok.endswith("}"):
        tok = tok[1:-1]
    return tok.strip()


def _script(s: str, i: int, mark: str) -> "tuple[str, int] | None":
    """Read one ``_…``/``^…`` script at *i*, braced (balanced) or bare."""
    if i >= len(s) or s[i] != mark:
        return None
    k = i + 1
    group = braced_group(s, k)
    if group is not None:
        return group[0].strip(), group[1]
    m = re.match(r"\\?[A-Za-z0-9]+", s[k:])
    return (m.group(0), k + m.end()) if m else None


def parse_latex_extremum(latex: str) -> "dict | None":
    r"""Read ``\max_{range}^{upper} <body>`` into a descriptor, or None when
    *latex* is not an extremum binder.

    The body is everything after the scripts — an extremum, like Σ, binds what
    is written to its right.  Only the OUTERMOST binder is read: the descriptor
    is display-only, so nothing downstream needs the inner structure.

    ``index_latex`` / ``upper_latex`` keep **what was written**, so the display
    gives the researcher their own spelling back.  Whether that spelling names
    a computable range is a separate question, asked at evaluation time by
    :func:`read_range` — the same separation the rest of this system keeps
    between the user's writing and what the CAS is handed.
    """
    s = latex.strip()
    m = _HEAD_RE.search(s)
    if m is None:
        return None
    sub = _script(s, m.end(), "_")
    if sub is None:
        return None
    sup = _script(s, sub[1], "^")
    body = s[(sup[1] if sup else sub[1]):].strip()
    if not body:
        return None                      # `\max_i` with nothing bound
    op = ("arg" + m.group(2)) if m.group(1) else m.group(2)
    return {"op": op,
            "index_latex": _strip(sub[0]),
            "upper_latex": _strip(sup[0]) if sup else None,
            "body_latex": body,
            "prefix_latex": s[:m.start()].strip()}


#: ``i=1`` (paired with a superscript) and ``1 \le i \le 3`` — the two ways a
#: bounded extremum is written.  ``\le``/``\leq``/``<=`` and the strict forms
#: all reach here as written; the strict ones are deliberately NOT read (see
#: :func:`read_range`).
_EQ_RANGE_RE = re.compile(r"^\s*([A-Za-z]\w*)\s*=\s*(.+?)\s*$")
_LE_RANGE_RE = re.compile(
    r"^\s*(.+?)\s*(?:\\le|\\leq|<=)\s*([A-Za-z]\w*)\s*(?:\\le|\\leq|<=)\s*(.+?)\s*$")


def read_range(desc: dict) -> "tuple[str, str, str] | None":
    r"""``(dummy, lower, upper)`` when the descriptor names a range, else None.

    Two spellings, both ordinary: ``\max_{i=1}^{3}`` (Σ-style) and
    ``\max_{1 \le i \le 3}`` (the commoner one for an extremum).

    A STRICT inequality is not read.  ``\max_{1 < i < 3}`` over integers means
    ``i = 2``, but over reals it means something this has no form for, and the
    notation does not say which — the same "never guess" line the rest of the
    system holds.  A bare ``\max_i`` names no range and returns None here,
    which is not a defect: it is the display-only case.
    """
    sub = (desc.get("index_latex") or "").strip()
    upper = desc.get("upper_latex")
    if upper:
        m = _EQ_RANGE_RE.match(sub)
        if m is not None:
            return m.group(1), m.group(2), str(upper)
        return None
    m = _LE_RANGE_RE.match(sub)
    if m is not None:
        return m.group(2), m.group(1), m.group(3)
    return None


def is_concrete_range(rng: "tuple[str, str, str] | None") -> bool:
    """True when both bounds are plain integers — the case the CAS can take.

    Measured 2026-08-16: ``lmax(makelist(x(i),i,1,3))`` answers
    ``max(x(1),x(2),x(3))`` and ``lmax(makelist(x(i),i,1,n))`` FAILS.  A
    symbolic bound therefore stays display-only, and saying so at the point of
    use beats sending a command that errors.
    """
    if rng is None:
        return False
    _dummy, lo, hi = rng
    try:
        int(str(lo).strip())
        int(str(hi).strip())
    except ValueError:
        return False
    return True


#: ``i \in \{1,2,3\}`` — an EXPLICIT SET.  Written when the values are not a
#: contiguous run (``\{1,3,7\}``), which is the whole reason for the spelling.
_SET_RE = re.compile(
    r"^\s*([A-Za-z]\w*)\s*(?:\\in|∈)\s*\\?\{\s*(?P<items>[^{}]*?)\s*\\?\}\s*$")

#: A STRICT inequality range, read only far enough to say WHY it is refused.
_STRICT_RE = re.compile(
    r"^\s*(.+?)\s*(?:<|\\lt)\s*([A-Za-z]\w*)\s*(?:<|\\lt)\s*(.+?)\s*$")


def read_value_set(desc: dict) -> "tuple[str, list[str]] | None":
    r"""``(dummy, values)`` for ``\max_{i \in \{1,2,3\}}``, else None.

    Only INTEGER LITERALS are accepted, for the same measured reason
    :func:`is_concrete_range` gives: the CAS needs actual values.  Measured
    2026-08-18, ``lmax(makelist(x[i], i, [1,3,7]))`` answers
    ``max(x[1],x[3],x[7])`` — so a NON-CONTIGUOUS set works, which is what this
    spelling is for.  A range would be written as a range.

    Refused with a factually WRONG reason until now: the message said "the range
    is not concrete integers" about ``\{1,2,3\}``, which is nothing but concrete
    integers.  The set was simply never read (open_items EXTREMUM-EXPLICIT-SET).
    """
    sub = (desc.get("index_latex") or "").strip()
    if desc.get("upper_latex"):
        return None                      # a superscript means the Σ-style range
    m = _SET_RE.match(sub)
    if m is None:
        return None
    items = [it.strip() for it in m.group("items").split(",") if it.strip()]
    if not items:
        return None
    for it in items:
        try:
            int(it)
        except ValueError:
            return None                  # a symbolic member is not evaluable
    return m.group(1), items


def is_strict_range(desc: dict) -> bool:
    r"""Is this ``\max_{1 < i < 3}`` — the one that genuinely cannot be read?

    Over the integers it means ``i = 2``; over the reals it means something this
    has no form for, and the notation does not say which.  Told apart from an
    unreadable range so the refusal can give the real reason instead of claiming
    the bounds are not integers — here they plainly are.
    """
    if desc.get("upper_latex"):
        return False
    return _STRICT_RE.match((desc.get("index_latex") or "").strip()) is not None


def build_set_eval_command(op: str, body_code: str,
                           vset: "tuple[str, list[str]]") -> str:
    r"""``lmax(makelist(body, i, [1,2,3]))`` — Maxima's list form of makelist.

    The sibling of :func:`build_eval_command`; a set is not a range, so it takes
    the three-argument ``makelist`` rather than the four-argument one.
    """
    dummy, values = vset
    fn = "lmin" if op in ("min", "inf", "liminf", "argmin") else "lmax"
    return f"{fn}(makelist({body_code}, {dummy}, [{', '.join(values)}]))"


def build_eval_command(op: str, body_code: str, rng: "tuple[str, str, str]") -> str:
    """The CAS command for a concrete-range extremum.

    ``lmax``/``lmin`` take a LIST, and ``makelist`` builds it — there is no
    ``max(expr, i, lo, hi)`` in Maxima, which is the whole reason this notation
    needed an object of its own.
    """
    dummy, lo, hi = rng
    fn = "lmin" if op in ("min", "inf", "liminf", "argmin") else "lmax"
    return f"{fn}(makelist({body_code}, {dummy}, {lo}, {hi}))"


def build_extremum_latex(desc: dict) -> str:
    r"""Rebuild ``\max_{i} body`` from a descriptor.

    ``\arg\max`` is written back with its two commands: it is one operator to
    the reader and two to LaTeX.
    """
    op = desc["op"]
    head = rf"\arg\{op[3:]}" if op.startswith("arg") else rf"\{op}"
    sup = desc.get("upper_latex")
    script = rf"_{{{desc['index_latex']}}}" + (rf"^{{{sup}}}" if sup else "")
    out = rf"{head}{script} {desc['body_latex']}".strip()
    prefix = (desc.get("prefix_latex") or "").strip()
    return f"{prefix} {out}".strip() if prefix else out
