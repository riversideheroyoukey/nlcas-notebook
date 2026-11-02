r"""Does the published `.tex` still say what the session JSON says?

`tex_sanity` reads the `.tex` ALONE, so it can only see defects it has a rule
for — "既に踏んだ欠陥しか見えない" (open_items TEX-EXPORT-SCAN).  The session
JSON carries every expression and step, so putting the two side by side answers
questions no amount of scanning one file can:

* a card that never reached the document (an expression the researcher can see
  in the notebook and not in the paper),
* an equation NUMBER that repeats or points at nothing,
* an expression whose latex changed **on the way out** — mangled by the export
  path alone, with the notebook still showing it correctly.

Both defects that prompted this were found by eye in one file: a scaffold
expression published as E6, and an E7→E6 gap in the numbering.  Both have a
mechanical shape, which is what this module is.

**The one normalisation, and why it is exactly one.**  Display deliberately
renders `*` as a space (CLAUDE.md, "Space=multiplication (bidirectional)"), so a
literal `*` in the JSON legitimately becomes a space in the `.tex`; measured over
the 90 committed pairs, that single rule accounts for **all 58** differences and
nothing else differs.  Whitespace is ignored because the exporter indents.
Anything beyond those two is reported.

Widening this list is how the check would stop working.  The precedent is in
this module's sibling: the first `bare-star` rule added to `tex_sanity` was
**23/23 false positives**, and on a check that runs over every published example
a false positive costs more than a miss — it trains the reader to skip the
output.  So the rule here was measured against the whole corpus BEFORE it was
written down, and it reports zero.
"""
from __future__ import annotations

import re

#: The math environments the exporter emits a numbered card in.
_EQ_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}", re.S)
_LABEL_RE = re.compile(r"\\label\{eq:([A-Za-z0-9]+)\}")


def _norm(latex: str) -> str:
    """Compare-form: drop the label, the `*`↔space bridge, and all whitespace."""
    return re.sub(r"\s+", "", _LABEL_RE.sub("", latex).replace("*", " "))


def _published(expr: dict) -> bool:
    """Is this expression meant to appear in the document?

    `internal` and `hidden` are the two flags the notebook already uses to keep
    scaffolding out of sight; an expression carrying either is not a missing
    card.  Measured over the corpus: 20 internal and 6 hidden out of 819.
    """
    return not expr.get("internal") and not expr.get("hidden")


def crosscheck(session: dict, tex: str) -> "list[tuple[str, str]]":
    """Every disagreement between *session* (parsed JSON) and *tex*.

    Returns ``(code, detail)`` pairs, sorted, so a caller can print them the way
    ``tex_sanity.scan``'s findings are printed.
    """
    exprs = session.get("expressions", {}) or {}
    labels = _LABEL_RE.findall(tex)
    findings: list[tuple[str, str]] = []

    shown = {eid for eid, e in exprs.items() if _published(e)}
    for eid in sorted(shown - set(labels)):
        findings.append(("card-missing",
                         f"{eid} is neither internal nor hidden, but no "
                         f"\\label{{eq:{eid}}} is in the document"))
    for lab in sorted({l for l in labels if labels.count(l) > 1}):
        findings.append(("label-duplicated",
                         f"eq:{lab} appears {labels.count(lab)} times — two "
                         f"equations carry one number"))
    for lab in sorted(set(labels) - set(exprs)):
        findings.append(("label-orphan",
                         f"eq:{lab} has no expression of that id in the session"))

    for m in _EQ_RE.finditer(tex):
        body = m.group(2)
        lm = _LABEL_RE.search(body)
        if lm is None or lm.group(1) not in exprs:
            continue
        want = exprs[lm.group(1)].get("latex") or ""
        if _norm(body) != _norm(want):
            findings.append((
                "latex-mangled",
                f"{lm.group(1)} differs from the session by more than the "
                f"`*`/whitespace bridge:\n      session: {want.strip()[:120]}"
                f"\n      export : {_LABEL_RE.sub('', body).strip()[:120]}"))
    return sorted(findings)
