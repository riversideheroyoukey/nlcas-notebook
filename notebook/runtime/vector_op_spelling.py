r"""How is a ∇ operation spelled?  One table, read by the single and the nested forms.

The same question had three answers (NESTED-VEC-OP-LATEX, measured 2026-09-17):

* the SINGLE reader (`_SINGLE_VEC_PATTERNS`, then in `llm_supervisor`) knew the LaTeX
  commands — `\nabla \cdot`, `\nabla \times`, `\nabla^2`, `\Delta` — as well as the
  Unicode symbols and the names;
* the NESTED reader (`nested_vec_2paren` / `1paren` / `noparen` in
  `nl_patterns.json`, over the fragment `vec_op_kw`) knew only the Unicode symbols,
  the names and the Japanese words;
* the operation schema's aliases were a third, and a comment beside the two tables
  already recorded that folding them was deferred (2026-08-27).

So `∇·(∇×A)` computed 0 and **`\nabla \cdot (\nabla \times \vec{A})` — what a
researcher writes and what this system prints — was registered without being
computed**: 70 of 70 LaTeX cells (7 operator pairs × 5 spellings × both languages).

The nested form is now BUILT from the single one: an outer operator prefix, then the
rest — with one pair of enclosing parentheses removed — read by the single reader.
Nothing about how an operator is spelled is written twice.  The Japanese genitive
chain (`A の回転の発散`) is a different grammar — nouns in a sentence — and stays a
key, now with its English mirror (`the divergence of the curl of A`).
"""
from __future__ import annotations

import re

#: A field is a single identifier; the caller checks it is a registered field.
_F = r"[A-Za-z_]\w*"

#: Each ∇ operator's spellings, most specific first (∇· / ∇× / ∇² before a bare ∇,
#: which is the gradient).
VEC_OP_PREFIXES: "list[tuple[str, str]]" = [
    (r"(?:\\nabla\s*\\cdot|∇\s*[·・])", "divergence"),
    (r"(?:\\nabla\s*\\times|∇\s*×)", "curl"),
    (r"(?:\\nabla\s*\^\s*\{?\s*2\s*\}?|∇\s*[²2]|\\Delta|∇\^2)", "scalar_laplacian"),
    (r"(?:\\nabla|∇)", "gradient"),
    (r"(?:divergence|div)", "divergence"),
    (r"(?:curl|rot)", "curl"),
    (r"(?:gradient|grad)", "gradient"),
    (r"(?:scalar_laplacian|laplacian)", "scalar_laplacian"),
]

_FLAGS = re.IGNORECASE | re.UNICODE
_SINGLE = [(re.compile(r"^\s*" + p + r"\s*\(?\s*(" + _F + r")\s*\)?\s*$", _FLAGS), op)
           for p, op in VEC_OP_PREFIXES]
_PREFIX = [(re.compile(r"^\s*" + p + r"\s*(?P<rest>.+?)\s*$", _FLAGS | re.S), op)
           for p, op in VEC_OP_PREFIXES]

#: The spelling the component builder (`_VEC_OP_RE`) reads for each operation's
#: RESULT — `curl(A)` is a vector, `divergence(A)` a scalar.  One table: the
#: operand handoff for the vector products (`vector_operand`) reads it too.
OPERATION_CALL = {
    "curl": "curl",
    "gradient": "gradient",
    "divergence": "divergence",
    "scalar_laplacian": "laplacian",
}

#: The noun each language uses in the genitive chain, and the operation it names.
#: Nouns are not operator spellings (nobody writes `発散(回転(A))`, measured: 0
#: corpus rows, 0 scenario lines, 0 test inputs), so they live apart from the prefixes.
_CHAIN_NOUN_OP = {
    "発散": "divergence", "回転": "curl", "勾配": "gradient", "ラプラシアン": "scalar_laplacian",
    "div": "divergence", "divergence": "divergence",
    "curl": "curl", "rot": "curl",
    "grad": "gradient", "gradient": "gradient",
    "laplacian": "scalar_laplacian",
}

#: ベクトルの印（`\vec{A}` / `\mathbf A`）。**この読みでは名前の飾りである**——
#: 「これはベクトルだ」は宣言が既に言っており、どの場かは名前が言う。剥がして
#: おかないと、研究者が**標準の綴り**で書いた `\nabla \times \vec{A}` が
#: 操作に届かず、式として登録されるだけになる（実測 2026-09-11。素の
#: `\nabla \times A` は通っており、電磁気の教科書の書き方だけが落ちていた）。
_VEC_MARK_RE = re.compile(r"\\(?:vec|mathbf)\s*(?:\{\s*([A-Za-z_]\w*)\s*\}|\s+([A-Za-z_]\w*))")


def strip_vector_mark(text: str) -> str:
    r"""``\nabla \times \vec{A}`` → ``\nabla \times A``."""
    return _VEC_MARK_RE.sub(lambda m: m.group(1) or m.group(2), text)


def read_single(text: str) -> "tuple[str, str] | None":
    """``(operation, field)`` for one ∇ operation applied to one field, else None."""
    text = strip_vector_mark(text)
    for rx, op in _SINGLE:
        m = rx.match(text)
        if m:
            return op, m.group(1)
    return None


def _without_enclosing_parens(text: str) -> str:
    r"""Remove ONE pair of parentheses — `( … )` or `\left( … \right)` — when it
    encloses the whole text.  Balanced, so `(\nabla f) + g` is left alone."""
    s = text.strip()
    for open_, close in ((r"\left(", r"\right)"), ("(", ")")):
        if not (s.startswith(open_) and s.endswith(close)):
            continue
        inner = s[len(open_):len(s) - len(close)]
        depth = 0
        for ch in inner:
            depth += ch == "("
            depth -= ch == ")"
            if depth < 0:
                break
        else:
            if depth == 0:
                return inner.strip()
    return s


def read_nested(text: str) -> "tuple[str, str] | None":
    r"""``(outer operation, inner call)`` for a ∇ operation applied to another, else None.

    `\nabla \cdot (\nabla \times \vec{A})`, `∇·(∇×A)`, `div(curl(A))`,
    `\nabla \times \nabla f` → the outer prefix, then the rest read as a SINGLE
    operation; and the genitive chain `A の回転の発散` / `the divergence of the curl
    of A`.  The inner call is spelled the way the component builder reads it
    (`curl(A)`), so the outer operation computes it.
    """
    text = strip_vector_mark(text)
    for rx, outer in _PREFIX:
        m = rx.match(text)
        if not m:
            continue
        inner = read_single(_without_enclosing_parens(m.group("rest")))
        if inner is not None:
            return outer, f"{OPERATION_CALL[inner[0]]}({inner[1]})"
    from notebook.runtime.nl_patterns import nlp_search
    m = nlp_search("nested_vec_chain", text)
    if m:
        outer = _CHAIN_NOUN_OP.get(m.group("outer").strip().lower())
        inner = _CHAIN_NOUN_OP.get(m.group("inner").strip().lower())
        if outer and inner:
            return outer, f"{OPERATION_CALL[inner]}({m.group('field').strip()})"
    return None
