r"""What are the components of this operand?  One answer for the vector products.

`_build_vec_components` (the Maxima handler) answered this with three shapes — a
list `[a, b, c]`, a declared vector field, a nested `curl(X)` — and for anything
else **invented a component name** by appending `_x`/`_y`/`_z` to whatever text it
was handed.  Measured 2026-09-17 over the dot/cross family (VEC-BINARY-OPERAND-
WHITESPACE), that last resort is where every broken spelling landed:

    \vec{A}                                  →  \vec{A}_x          raw CAS error
    \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} →  matrix([1],[2],[3])_x   raw CAS error
    u = (1, 2, 3)                            →  (1, 2, 3)_x        raw CAS error

**The norm had already answered the same question** (`_build_norm`): it lowers
the researcher's writing first, reads a one-row or one-column matrix as a vector
(a researcher writes a vector in TeX, and it arrives as a MATRIX), and refuses a
parenthesised tuple WITH A REASON — "a parenthesised tuple is not read as a
vector".  Two operations, one question, two answers; this module is the reading
the vector products now share with it.

The suffix is still the answer for a bare NAME — an undeclared `B` used as a
vector has always been read as `(B_x, B_y, B_z)`, and nothing measured says
otherwise.  What changes is that text which is NOT a name no longer gets a name
invented for it.

**そして「掛かり方」を文字列で決めない** (CROSS-PRODUCT-INSIDE-OPERAND,
2026-09-18)。外積の CAS 綴りは `nb_cross` だが、それが式のどこを置き換えるかは
構文の問題で、文字列の上では決まらない。ここは「いちばん外側の `~` で2つに割る」
という近似を持っていたので、`2 (v ~ w)` も `(v ~ w) + r` も `(v ~ w) ~ r` も
読めなかった。答えを持っているのはパーサなので、`cross_products_as_calls` が
**構文木の上で**書き換え、綴りはバックエンドの生成器が書く。
"""
from __future__ import annotations

import re

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def is_bare_name(text: str) -> bool:
    """A single identifier — the only text a component suffix can be glued to."""
    return bool(_NAME_RE.fullmatch((text or "").strip()))


def _split_top_level(inner: str) -> "list[str]":
    """Split on commas at bracket depth 0 (a row of `matrix(...)` holds commas)."""
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def three_entries(code: str) -> "tuple[str, str, str] | None":
    """The three entries of a one-column or one-row ``matrix(...)``, else None.

    A column vector written `\\begin{pmatrix} 1 \\\\ 2 \\\\ 3 \\end{pmatrix}`
    lowers to `matrix([1], [2], [3])`; a row vector to `matrix([1, 2, 3])`.
    Frobenius and the 2-norm coincide on either, which is why the norm reads them
    as vectors without asking — and the dot and cross products need exactly three.
    """
    s = (code or "").strip()
    if not (s.startswith("matrix(") and s.endswith(")")):
        return None
    rows = _split_top_level(s[len("matrix("):-1])
    unwrapped = []
    for r in rows:
        if not (r.startswith("[") and r.endswith("]")):
            return None
        unwrapped.append(_split_top_level(r[1:-1]))
    if len(unwrapped) == 1 and len(unwrapped[0]) == 3:          # one row
        return tuple(unwrapped[0])  # type: ignore[return-value]
    if len(unwrapped) == 3 and all(len(r) == 1 for r in unwrapped):  # one column
        return (unwrapped[0][0], unwrapped[1][0], unwrapped[2][0])
    return None


def vector_operation_operand(text: str) -> str:
    r"""`\nabla \times \vec{A}` written as an OPERAND → `curl(A)`; else unchanged.

    The reader for a single ∇ operation already exists (`_try_single_vec_op`), and
    the component builder already computes `curl(X)` / `gradient(X)` as a vector
    and refuses `divergence(X)` / `laplacian(X)` as a scalar.  What was missing was
    only the handoff: with the operand boundary right, `\nabla \times \vec{A} と
    \vec{B} の内積` hands over the text `\nabla \times \vec{A}`, and nothing turned
    it into the spelling both sides already agree on.  So `(∇×A)·B` is COMPUTED,
    not refused — the capability was all there.
    """
    from notebook.runtime.vector_op_spelling import OPERATION_CALL, read_nested, read_single
    single = read_single(text)
    if single is not None:
        return f"{OPERATION_CALL[single[0]]}({single[1]})"
    nested = read_nested(text)        # `\nabla \times \nabla \times \vec{A}` as an operand
    if nested is not None:
        return f"{OPERATION_CALL[nested[0]]}({nested[1]})"
    return text


_WORD_RE = re.compile(r"(?<![\w%])([A-Za-z_]\w*)(?!\w)(\s*\()?")


def expression_components(code: str, context, value_of, _depth: int = 0,
                          to_cas=None) -> "tuple[str, str, str] | None":
    r"""A vector-valued EXPRESSION's components — `(E)[1]`, `(E)[2]`, `(E)[3]` — else None.

    `2 v の回転を求めて` dropped the coefficient and computed `curl(v)`, `completed`
    (OPERAND-COEFFICIENT-DROPPED, measured 2026-09-17 over six operations × both
    languages).  With the whole operand registered, what remained was reading a
    scalar multiple, a sum or a coordinate-weighted vector as components — and the
    CAS already does it: a name given a value by H-1 is bound to a LIST, and Maxima's
    list arithmetic answers `(2*v)[1] = 2*y`, `(v+w)[3] = x`, `(x*v)[2] = -x^2`
    (measured).  Through the ∇ builders that gives `curl(2v) = [0,0,-4]`,
    `curl(v+w) = [0,-1,-2]`, `div(2r) = 6` and `curl(x v) = [0,0,-3x]` — the product
    rule included, because the derivative acts on the component expressions.

    Two things are needed first, both decided from NOTEBOOK state:

    * the expression is a VECTOR (`vector_op_types.infer_algebra`, which reads a
      name's type from its value — `2 v` is a vector, `v . w` a scalar, `v w` unknown);
    * every vector in it is something the CAS holds as a list.  A DECLARED field is an
      atom — `(2*A)[1]` stays a noun (measured) — so it is replaced by its component
      list first; an H-1 vector name by its (listified) value, so a value that itself
      mentions a declared field reaches the CAS as a list too.  A vector this cannot
      spell as a list returns None, and the caller refuses.

    **The value is spliced in its stored spelling, NOT lowered.**  A first version
    lowered it, and a mutation that removed the lowering failed nothing; measured,
    the lowering was HARMFUL — `v = [x y_1, x^2, 0]` gave `2 v の回転` as
    `[0,0,4x - 2·(d/dy y(1))·x]`, because `y_1` became `y(1)`, a function of `y`.
    Unlowered it is `[0,0,4x]`, twice `v の回転`, and the list path the builder
    already had never lowered either.

    **A cross product is spelled by the BACKEND, from the structure** (*to_cas*,
    CROSS-PRODUCT-INSIDE-OPERAND).  Maxima's `~` is not evaluable at all — measured
    2026-09-18, `((v) ~ (w))` is a raw `Maxima evaluation error`, with or without a
    list — and the system defines `nb_cross` for exactly that.  Which text the call
    replaces was decided here by splitting on a top-level `~`, so anything else
    (`2 (v ~ w)`, `(v ~ w) + r`, `(v ~ w) ~ r`) was refused; the parser is what
    knows that `v ~ w + r` is `nb_cross(v, w) + r`.  The operand is rewritten
    BEFORE the values are spliced, so the rule above still holds: the operand is
    the researcher's own writing, which every other operation lowers, and the
    values are not.  **`~` があるときだけ訊く guard は測って消した**——呼び出し側
    （`_build_vec_components`）が先に同じ lowering を済ませているので、ここでの
    2度目は恒等写像である。6件で実測し、CAS へ送る命令まで**1バイトも変わらな
    かった**（前周が有害と測った `v = [x y_1, x^2, 0]` を含む）。
    """
    from notebook.runtime.vector_op_types import infer_algebra
    if _depth > 8 or infer_algebra(code, context, value_of) != "vector":
        return None
    if to_cas is not None:
        code = to_cas(code) or code
    if "~" in code:                   # 綴りへ落とせなかった外積は読めない
        return None
    listified = _listify(code, context, value_of, _depth)
    if listified is None:
        return None
    return tuple(f"({listified})[{i}]" for i in (1, 2, 3))  # type: ignore[return-value]


def cross_products_as_calls(node, name: str):
    """*node* の外積をすべて ``name(左, 右)`` の呼び出しに書き換えた木を返す。

    **どこで置き換わるかは構文木が決める。** 以前はこの判断が文字列の上にあり、
    「いちばん外側の `~` で2つに割る」しかできなかったので、`2 (v ~ w)` ・
    `x (v ~ w)` ・ `(v ~ w) + r` ・ `(v ~ w) ~ r` はどれも読めず断られていた
    （CROSS-PRODUCT-INSIDE-OPERAND、実測 2026-09-18: 回転・発散・ノルムの3操作 ×
    10綴り × 日英の 60 セル中、通っていたのは 24——壊れていた6綴りが 36 セル）。
    掛かり方は構文の問題なので、答えを持って
    いるのはパーサである——`v ~ w + r` は `nb_cross(v, w) + r` であって
    `nb_cross(v, w + r)` ではない。

    書き換えは節点の種類を数え上げず、pydantic の欄をたどる。数え上げると、
    外積を包める節点が増えるたびにここを直し忘れる。
    """
    from notebook.models import ir
    if not isinstance(node, ir.Expr):
        return node
    if isinstance(node, ir.CrossProduct):
        return ir.FunctionCall(
            func=ir.Symbol(name=name),
            args=[cross_products_as_calls(node.left, name),
                  cross_products_as_calls(node.right, name)])
    updates = {}
    for field in type(node).model_fields:
        value = getattr(node, field, None)
        if isinstance(value, ir.Expr):
            updates[field] = cross_products_as_calls(value, name)
        elif isinstance(value, list) and any(isinstance(v, ir.Expr) for v in value):
            updates[field] = [cross_products_as_calls(v, name) for v in value]
    return node.model_copy(update=updates) if updates else node


def listify(code: str, context, value_of, depth: int = 0) -> "str | None":
    r"""式の中のベクトルを**成分のリスト**に開く（開けなければ None）。

    `nb_cross(A, B)` が正しいのは A・B がリストのときだけで、宣言された場の
    ような原子に当てると `A[2]*B[3]-…`——**研究者が書いていない名前**を返す
    （実測）。この関数がその前提を作る唯一の場所である。
    """
    return _listify(code, context, value_of, depth)


def _listify(code: str, context, value_of, depth: int) -> "str | None":
    from notebook.runtime.vector_op_types import infer_algebra
    failed = False

    def repl(m: "re.Match[str]") -> str:
        nonlocal failed
        name, call = m.group(1), m.group(2)
        if call:                         # a function call (`sin(`, `vec(`) — not a quantity
            return m.group(0)
        qty = context.get_quantity(name) if context is not None else None
        if qty is None:
            return m.group(0)
        comps = getattr(qty, "components", None)
        if getattr(qty, "kind", None) == "vector_field" and comps and len(comps) >= 3:
            return "[" + ",".join(comps[:3]) + "]"
        value = value_of(name) if (value_of and getattr(qty, "defining_expr_id", None)) else None
        if value and infer_algebra(value, context, value_of) == "vector":
            inner = _listify(value, context, value_of, depth + 1)
            if inner is None:
                failed = True
                return m.group(0)
            return f"({inner})"
        if getattr(getattr(qty, "math_type", None), "algebra", None) == "vector":
            failed = True                # declared vector with no component names
        return m.group(0)

    out = _WORD_RE.sub(repl, re.sub(r"\bvec\(\s*([A-Za-z_]\w*)\s*\)", r"\1", code))
    return None if failed else out


def not_a_vector_reason(code: str, context, value_of) -> "str":
    """Which refusal fits an operand that could not be read as a vector — its i18n key.

    A product of two VECTORS is not "not a vector": it is a dot or a cross product whose
    kind was not written, and the generic reason (write `[1, 2, 3]`, a tuple is not a
    vector) would answer a different question.
    """
    from notebook.models import ir
    from notebook.runtime.vector_op_types import _parse, _algebra_of_node
    node = _parse(code)
    factors = (node.args if isinstance(node, (ir.Mul, ir.NonCommutativeMul))
               else [node.left, node.right] if type(node).__name__ == "Juxtaposition" else [])
    n_vectors = sum(1 for f in factors if _algebra_of_node(f, context, value_of, 0) == "vector")
    if n_vectors >= 2:
        return "vector.product_of_vectors"
    return "vector.operand_not_a_vector"


def unwrap_vector_mark(text: str) -> "str | None":
    """The inside of ONE `vec(...)` that encloses the whole text, else None.

    The pattern `vec\\((.*)\\)` matched `vec(v) ~ vec(w)` as one wrapper around
    `v) ~ vec(w` — greedy across two — and the refusal named that broken text
    (found 2026-09-17 when `\\vec{v} \\times \\vec{w}` became a readable operand).
    """
    from notebook.runtime.vector_op_spelling import _without_enclosing_parens
    s = (text or "").strip()
    if not s.startswith("vec("):
        return None
    rest = s[3:]
    inner = _without_enclosing_parens(rest)
    return inner if inner != rest.strip() else None
