r"""What a LaTeX command says about SETTING a formula, not about its meaning.

Extracted from ``latex_converter`` on 2026-08-10, the split this project takes
when the material for one decision keeps growing (the precedent is
``subscript_policy`` and ``sum_integral_prefix``): sizing hints, font wrappers,
layout commands and TeX's infix ``\over`` are all one question — *is this mark
part of the mathematics?* — and the answer belongs in one module.

Why it matters that they are handled at all: a command the converter does not
know keeps its name and loses its backslash further down (the parser's
tokenizer ignores ``\`` so that ``\sum`` can reach it), so it arrives at the CAS
as an ordinary symbol and multiplies whatever stands beside it.  Measured
2026-08-10 — every one of these reported SUCCESS:

===========================  ==================================================
``\big(a+b\big)*c``          ``expand`` answered ``big(b*big+a)*c``
``\dfrac{a}{b}``             became the single symbol ``dfracab``
``\mathit{a}*x``             became ``mathita*x``
``{{a}\over{b}}``            became ``a`` times the symbol ``over`` times ``b``
===========================  ==================================================

The last two are not hypothetical.  ``sessions/*.json`` already holds 24
``\over`` and 8 font-wrapped symbols, all of them written by this system's own
display: Maxima's ``tex1`` emits ``{{a}\over{b}}`` for every fraction and
``\mathit{i}`` for the imaginary unit, so the display round trip (R3) did not
close for either.
"""
from __future__ import annotations

import re

#: Delimiter SIZE hints.  ``\left``/``\right`` have their own rule in the
#: converter; these are their fixed-size siblings.  One pattern covers
#: ``\big \Big \bigg \Bigg`` and their ``l``/``r``/``m`` variants.
_SIZING_RE = re.compile(r"\\[Bb]igg?[lrm]?(?![A-Za-z])\s*")

#: Commands that change how something is SET, not what it means; the wrapper's
#: content is kept verbatim.  ``\mathbf`` is deliberately absent — the
#: converter's ``_vector`` rule reads it as a vector.
_FONT_CMDS = ("mathrm", "mathit", "mathsf", "mathtt", "mathnormal",
              "textrm", "textit", "textbf", "text", "operatorname")

#: TeX-2 font switches written INSIDE the group they affect: ``{\rm asinh}``.
#: Maxima's own ``tex1`` emits this one.
_FONT_SWITCH_RE = re.compile(r"\{\s*\\(?:rm|it|bf|sf|tt|cal)\s+([^{}]*)\}")

#: Layout-only commands that carry no value at all.
_LAYOUT_CMDS = (r"\limits", r"\nolimits", r"\displaystyle", r"\textstyle",
                r"\scriptstyle", r"\mathstrut", r"\strut")


def upright_operator_spellings(names: "dict[str, str]") -> "dict[str, str]":
    r"""``{"tr": "mattrace"}`` → そのラッパ綴りを**全部**（`\text{tr}`・
    `\mathrm{tr}`・`\operatorname{tr}` …）。

    **綴りは並べずに生成する。** 立体で組まれた作用素名は、どのラッパで書くかが
    研究者と TeX エディタの好みでしかない（`\mathrm{tr}` と `\operatorname{tr}` は
    同じものである）。列挙すると片方だけに足して他方を忘れる——核種の綴りを
    (A, 記号) から生成したのと同じ理由。ラッパの一覧はここ（`_FONT_CMDS`）に
    1つだけあるので、そこから作る。

    **どこへ渡すか。** `strip_typography` の `keep`（ラッパを外さないため）と、
    バックエンドの関数表（`\text{tr}` を `mattrace` に置き換えるため）の**両方**。
    片方だけだと: `keep` に無ければ先に外れて `tr` になり関数表が空振りし、
    関数表に無ければラッパが残ったまま CAS に届く。実測 2026-09-06、前者が
    起きていて `\text{tr}(M)` は未定義の `tr(M)` として登録され、以後どの操作も
    黙って何もしなかった（`\det` は素の綴りなので通り、`\mathrm{det}` は落ちる、
    という同じ表の中での割れになっていた）。
    """
    return {f"\\{cmd}{{{name}}}": canonical
            for name, canonical in names.items()
            for cmd in _FONT_CMDS}


def braced_group(s: str, i: int) -> "tuple[str, int] | None":
    r"""Read the ``{...}`` group starting at *i*, honouring nesting.

    Returns ``(inner, index_after_closing_brace)``, or None when *i* is not an
    opening brace or the braces are unbalanced.  Commands whose operand can
    itself contain braces (``\frac{1}{\sqrt{x^2+1}}``) cannot be matched with
    ``\{([^{}]*)\}`` — the same conclusion BP-9 reached for derivative display
    forms, where scanning with balanced delimiters replaced a regex that
    silently dropped operands.
    """
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return None


def normalize_command_aliases(s: str) -> str:
    r"""Make the size-hinted spellings of a command one spelling: ``\frac``.

    Called from BOTH :func:`strip_typography` and
    ``MaximaBackendHandler.latex_to_canonical`` — deliberately one function, not
    one rule written twice.  The BP-9 derivative restorer runs BEFORE the
    converter, so a normalisation living only inside the converter is invisible
    to it: ``\dfrac{dy}{dx}`` was read as an ordinary fraction and lowered to
    ``dy * dx^-1``, a product of two undefined symbols, while ``\frac{dy}{dx}``
    became ``'diff(y,x,1)``.  Idempotent, so applying it twice is harmless.
    """
    return re.sub(r"\\[dt]frac(?![A-Za-z])", r"\\frac", s)


def unwrap_command(s: str, name: str, keep: "dict | None" = None) -> str:
    r"""Replace ``\name{X}`` with ``X``, scanning braces balanced.

    A regex cannot do this: ``\mathrm{g'}`` and ``\text{\alpha_{1}}`` both carry
    delimiters of their own.

    *keep* is the backend's own constant table.  A wrapper whose FULL spelling
    appears there is left for that table to read — an upright ``\mathrm{i}`` is
    the imaginary unit, not the symbol ``i``, and the backend is what knows so.
    Deciding it here instead would be this project's recurring defect: one
    question answered in two places, differently.
    """
    needle = "\\" + name
    out: list[str] = []
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        k = j + len(needle)
        if k < len(s) and s[k].isalpha():        # `\textit` is not `\text`
            out.append(s[i:k])
            i = k
            continue
        while k < len(s) and s[k] == " ":
            k += 1
        group = braced_group(s, k)
        if group is None:                        # no operand — leave it be
            out.append(s[i:k])
            i = k
            continue
        if keep and f"{needle}{{{group[0]}}}" in keep:
            out.append(s[i:group[1]])            # a named constant, not a font
            i = group[1]
            continue
        out.append(s[i:j])
        out.append(group[0])
        i = group[1]


def infix_over(s: str) -> str:
    r"""``{ X \over Y }`` → ``(X)/(Y)`` — TeX's infix fraction.

    This is what Maxima's ``tex1`` emits for every fraction, so leaving it
    unread breaks the display round trip for the system's OWN output.  The
    enclosing group is found by scanning braces in both directions, because the
    operands carry braces of their own (``{{omega_{D}}\over{\sinh(x)}}``).
    """
    while True:
        j = s.find(r"\over")
        if j < 0 or s[j + len(r"\over"):j + len(r"\over") + 1].isalpha():
            return s
        depth, start = 0, None
        for i in range(j - 1, -1, -1):           # back to this group's `{`
            if s[i] == "}":
                depth += 1
            elif s[i] == "{":
                if depth == 0:
                    start = i
                    break
                depth -= 1
        if start is None:
            return s                             # unbalanced — leave it alone
        group = braced_group(s, start)
        if group is None:
            return s
        inner, end = group
        head, _, tail = inner.partition(r"\over")
        s = f"{s[:start]}({head.strip()})/({tail.strip()}){s[end:]}"


def strip_typography(s: str, keep: "dict | None" = None) -> str:
    """Drop every mark that is about setting the formula rather than its value.

    Applied at the converter's entrance — the same place ``_subscript_separator``
    and ``_conjugate_mark`` make one spelling out of several — so the rest of the
    pipeline has a single spelling to work on rather than several to keep in
    step.  *keep* is passed through to :func:`unwrap_command`.
    """
    s = _SIZING_RE.sub("", s)
    for cmd in _LAYOUT_CMDS:
        s = s.replace(cmd, "")
    s = normalize_command_aliases(s)
    s = _FONT_SWITCH_RE.sub(r"\1", s)
    for cmd in _FONT_CMDS:
        s = unwrap_command(s, cmd, keep)
    return infix_over(s)
