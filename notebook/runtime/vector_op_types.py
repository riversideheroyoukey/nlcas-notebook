"""ベクトル解析の操作が、その被演算子に適用できるか（H-6）。

`div` はベクトルに、`grad` はスカラーに定義される。型が合わない組み合わせは
**数学的に未定義**で、答えは存在しない。

**なぜ要るのか（実測 2026-08-06）。** 型を見ていなかったため、未定義の組み合わせが
`status=completed` で通り、もっともらしい式が返っていた:

===============================  =========================  ==================
入力                              返っていたもの              正しくは
===============================  =========================  ==================
スカラー `x^2+y^2` の**発散**      `y^2+x^2`（そのまま）        未定義
スカラー `x^2+y^2` の**回転**      `y^2+x^2`（そのまま）        未定義
ベクトル場の**勾配**               `[0=0, 0=0, 0=0]`          未定義
===============================  =========================  ==================

どれもエラーが出ないので、研究者は「発散を計算した」と思って先へ進む。これは
起票 H-6 が「着手条件: 型情報を使った操作が必要になったとき」と書いていた条件が
満たされた状態であり、`_reject_display_only_operands` が表示専用オブジェクトに
対してやっているのと同じ「**使う地点で理由付きで断る**」の型判定版である。

**推測はしない。** 型が確信を持って言えるときだけ断る。言えないときは従来どおり
通す——「たぶんスカラー」で拒否すると、正しい計算が理由なく止まる方が害が大きい。
確信を持てるのは次の2つだけ:

* 被演算子の記号がすべてスカラー（ベクトルが1つも無い）→ スカラー
* ベクトルがちょうど1つで、残りがすべてスカラー → ベクトル
  （`m a` は「スカラー×ベクトル＝ベクトル」で曖昧さがない）

ベクトルが2つ以上あると内積か外積かテンソル積か決まらないので **None**（不明）
を返し、断らない。

**型の在り処が2系統ある**のは既存の設計で、ここはそれを読むだけ:
`kind`（`vector_field` / `scalar_field`）と `math_type.algebra`
（`vector` / `matrix` / `tensor`）。宣言の仕方で入る場所が違う——
「ベクトル場として定義」は `kind=vector_field, algebra=scalar`、
「ベクトルとして定義」は `kind=scalar, algebra=vector`。
"""
from __future__ import annotations

import re

#: ベクトル解析の操作 → 被演算子に要求される型。下の出力型の表と対になるもので、
#: あちらは「何が返るか」、ここは「何を受け取れるか」。
VECTOR_OP_OPERAND: "dict[str, str]" = {
    "divergence":       "vector",
    "curl":             "vector",
    "gradient":         "scalar",
    "scalar_laplacian": "scalar",
}

#: ∇ の操作を**呼び出しの綴り**で書いた式（`curl(A)`・`grad(f)`）が返す型。
#: **この表は2026-09-17まで Maxima のハンドラに在って、どこからも読まれていなかった**
#: （`_VEC_OP_OUTPUT_TYPE`）。型の推論はそれを訊かず、`grad(f)` を「未宣言の名前
#: `grad` とスカラー場 `f` でできた式」＝スカラーと読んで、`curl(grad(f))`
#: （答えは `[0,0,0]`）・`div(grad(f))`（ラプラシアン）・`grad(div(A))` を、研究者が
#: 書いていない E1 についての理由で断っていた（NESTED-VEC-OP-LATEX）。
VECTOR_OP_RESULT: "dict[str, str]" = {
    "curl": "vector", "rot": "vector", "gradient": "vector", "grad": "vector",
    "divergence": "scalar", "div": "scalar",
    "laplacian": "scalar", "scalar_laplacian": "scalar",
}

_CALL_RE = re.compile(r"(\w+)\s*\((.*)\)", re.S)


def _operation_result(text: str) -> "str | None":
    """*text* 全体が ∇ の操作の呼び出しなら、その結果の型。

    外側の括弧は外す（登録された被演算子は `(grad(f))` の形で来る）。呼び出しの
    括弧が**全体を**囲んでいることは深さで確かめる——`grad(f)*div(A)` は呼び出し
    1つではない。
    """
    from notebook.runtime.vector_op_spelling import _without_enclosing_parens
    s = text.strip()
    while (inner := _without_enclosing_parens(s)) != s:
        s = inner
    m = _CALL_RE.fullmatch(s)
    if not m or m.group(1).lower() not in VECTOR_OP_RESULT:
        return None
    if _without_enclosing_parens(f"({m.group(2)})") != m.group(2).strip():
        return None
    return VECTOR_OP_RESULT[m.group(1).lower()]


#: 成分リスト `[a, b, c]` / `\left[ … \right]`。ベクトルを成分で書いた形。
_COMPONENT_LIST_RE = re.compile(r"^(?:\\left)?\s*\[.*\]\s*(?:\\right)?$", re.S)

#: 記号名らしきものを拾う。CAS コードでも利用者書式でも同じように働けばよい。
_NAME_RE = re.compile(r"[A-Za-z_]\w*")

#: 型判定に使ってはいけない名前（関数・定数）。
_NOT_A_QUANTITY = frozenset({
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "diff", "integrate",
    "sum", "matrix", "pi", "e", "i",
    # `vec` はこの系がベクトルを包む綴りであって、量の名前ではない
    # （CLAUDE.md「`vec(...)` is the system's wrapper, not a component name」）。
    "vec",
})

#: この系自身のベクトルの包み。**宣言と同じ資格で「これはベクトルだ」と言う**——
#: `\vec{B}` / `\mathbf{B}` はどちらもここへ降りる。
_VEC_WRAPPER_RE = re.compile(r"\bvec\s*\(")


def _algebra_of(name: str, context) -> "str | None":
    """宣言から読める型。読めなければ None。"""
    prop = context.get(name) if context else None
    if prop is None:
        return None
    qty = prop.quantity
    if getattr(qty, "kind", None) == "vector_field":
        return "vector"
    if getattr(qty, "kind", None) == "scalar_field":
        return "scalar"
    algebra = getattr(getattr(qty, "math_type", None), "algebra", None)
    if algebra in ("vector", "matrix", "tensor"):
        return "vector" if algebra == "vector" else algebra
    if algebra == "scalar" or getattr(qty, "kind", None) == "scalar":
        return "scalar"
    return None


def infer_algebra(text: str, context, value_of=None, _depth: int = 0) -> "str | None":
    """*text* が表す量の型。確信が持てなければ None。

    推測しないことが要件（モジュール docstring 参照）。ベクトルが2つ以上ある式は
    内積とも外積とも取れるので None を返す。

    *value_of* は「H-1 で値を与えた名前 → その値の文字列」。渡されれば、その名前の
    型は**値から**読む（記録された `algebra` は登録側の既定値で、宣言ではない）。
    """
    if not text or _depth > 8:
        return None
    # A COMPONENT LIST is a vector, whatever its components are made of.
    # Measured 2026-08-06: `∇·(A×B)` was refused because the cross product's
    # canonical is `[A_y*B_z-A_z*B_y, …]` and every component name is a declared
    # SCALAR function — the components of a vector are scalars, so counting them
    # says "scalar" about the very thing that is a vector.  The brackets are the
    # signal, and they are checked first.
    if _COMPONENT_LIST_RE.match(text.strip()):
        return "vector"
    called = _operation_result(text)
    if called is not None:
        return called
    names_all = set(_NAME_RE.findall(text))
    if not (names_all - _NOT_A_QUANTITY) and not names_all & {"vec"}:
        return None                      # 名前が無い（`3`）——判定しない
    tree = _parse(text)
    if tree is not None:
        return _algebra_of_node(tree, context, value_of, _depth)
    if names_all & set(VECTOR_OP_RESULT):
        # ∇ の呼び出しが式の**一部**にある（`grad(f)*div(A)`）。名前として数えると
        # 「未宣言の名前だけ＝スカラー」と言い切ってしまうが、確信は無い。
        return None
    names = {n for n in _NAME_RE.findall(text) if n not in _NOT_A_QUANTITY}
    if not names:
        return None
    kinds = {n: _algebra_of(n, context) for n in names}
    vectors = [n for n, v in kinds.items() if v == "vector"]
    # **この系でベクトルだと言えるのは3つだけ**——成分リスト（上で答え済み）、
    # 宣言、そしてこの系自身の包み `vec(...)`。包みは宣言と同じ資格で
    # 「これはベクトルだ」と言うので、ここで数える（実測 2026-09-17: これが無いと
    # 未宣言の `\vec{B}` の**勾配**が `[0,0,0]` を `completed` で返した——
    # このモジュールの docstring が「直した」と書いた形そのものが、
    # 宣言されていない側で生き残っていた）。
    wrapped = len(_VEC_WRAPPER_RE.findall(text))
    n_vectors = len(vectors) + wrapped
    if n_vectors == 1:
        return "vector"
    if n_vectors >= 2:
        return None                      # ベクトルが複数 → 内積か外積か決まらない
    # ここに来たのは「ベクトルだと言うものが何も無い」場合である。
    #
    # **未宣言の名前はベクトルではない。** 以前はここで `None`（判定しない）を返して
    # いたが、それは「推測しない」ではなく**測っていない**だった: 未宣言の記号は
    # この系では自由変数＝CAS の記号原子で（CLAUDE.md「Free variables」）、
    # ベクトルになるには宣言か成分リストか包みが要る——どれも無いなら
    # スカラーだと**言い切れる**。実測 2026-09-17、言わなかった代償は
    # `f = x^2 + y^2` の発散・回転が**生の `Maxima evaluation error`** を返すこと
    # だった（Core Principle #1）。このモジュール自身の docstring は、その
    # `x^2+y^2` を**断る例**として挙げている。
    return "scalar"


def value_lookup(context, text_of_expr):
    """``name -> その名前を定義した式の右辺`` を返す関数を作る。

    *text_of_expr* は「式番号 → その式の文字列」。エンジンは ``Expression.canonical``
    を、成分の組み立ては ``expr_lookup`` を渡す——**どちらもノートの状態**で、
    CAS に訊かない（判断が再現できるように）。
    """
    from notebook.runtime.shared_parser import strip_lhs

    def value_of(name: str) -> "str | None":
        qty = context.get_quantity(name) if context is not None else None
        eid = getattr(qty, "defining_expr_id", None) if qty is not None else None
        text = text_of_expr(eid) if eid else None
        return strip_lhs(text) if text else None
    return value_of


def _parse(text: str):
    try:
        from notebook.runtime.shared_parser import SharedParser
        return SharedParser().parse(text)
    except Exception:
        return None


def _algebra_of_name(name: str, context, value_of, depth: int) -> "str | None":
    """名前の型。**H-1 で値を与えた名前は値から読む**（H1-VECTOR-ALGEBRA-DEFAULT）。

    `v = [y, -x, 0]` の `v` は `kind='anonymous'`・`algebra='scalar'` で記録される。
    その `scalar` を宣言として読んでいたので、`v + w の回転を求めて` は「E3 は
    スカラーです」で断られていた（実測 2026-09-17、日英とも）。値が引けなければ
    確信が無いので None。
    """
    if name in ("pi", "e", "i", "%pi", "%e", "%i"):
        return "scalar"
    qty = context.get_quantity(name) if context is not None else None
    if (qty is not None and getattr(qty, "defining_expr_id", None)
            and getattr(qty, "kind", None) not in ("vector_field", "scalar_field")):
        value = value_of(name) if value_of is not None else None
        return infer_algebra(value, context, value_of, depth + 1) if value else None
    # 宣言か、未宣言（＝自由変数、スカラー）。
    return _algebra_of(name, context) or "scalar"


def _algebra_of_node(node, context, value_of, depth: int) -> "str | None":
    """式の**構造**から型を読む——名前を数えるのでは和と積を区別できない。

    数える方式は「ベクトルが2つ以上なら内積か外積か決まらない」と答えたので、
    `v + w`（和＝ベクトル）も `v . w`（内積＝スカラー）も None になり、`2 v` は
    名前 `v` の記録された既定値で「スカラー」になった。規則はどれも算術である:
    和はすべての項が同じ型ならその型、積はベクトルの因子が0個ならスカラー・1個なら
    ベクトル・2個以上なら判定しない、内積はスカラー、外積はベクトル（成分は CAS の
    リスト演算ではなく `nb_cross` で取る——`vector_operand`）。
    """
    from notebook.models import ir
    kind = type(node).__name__
    if isinstance(node, ir.Number):
        return "scalar"
    if isinstance(node, ir.Symbol):
        return _algebra_of_name(node.name, context, value_of, depth)
    if kind == "BracketExpr":
        return "vector"
    if isinstance(node, ir.FunctionCall):
        fname = getattr(node.func, "name", "") if node.func is not None else ""
        if fname == "vec":
            return "vector"
        if fname.lower() in VECTOR_OP_RESULT:
            return VECTOR_OP_RESULT[fname.lower()]
        args = [_algebra_of_node(a, context, value_of, depth) for a in node.args]
        return "scalar" if all(a == "scalar" for a in args) else None
    if isinstance(node, ir.Add):
        parts = {_algebra_of_node(a, context, value_of, depth) for a in node.args}
        return parts.pop() if len(parts) == 1 and None not in parts else None
    factors = None
    if isinstance(node, (ir.Mul, ir.NonCommutativeMul)):
        factors = node.args
    elif kind == "Juxtaposition":
        factors = [node.left, node.right]
    if factors is not None:
        algs = [_algebra_of_node(f, context, value_of, depth) for f in factors]
        if any(a not in ("scalar", "vector") for a in algs):
            return None
        n_vec = algs.count("vector")
        return "scalar" if n_vec == 0 else "vector" if n_vec == 1 else None
    if isinstance(node, ir.Pow) or kind == "Superscript":
        base = node.base
        exp = getattr(node, "exponent", None) or getattr(node, "script", None)
        both = (_algebra_of_node(base, context, value_of, depth),
                _algebra_of_node(exp, context, value_of, depth))
        return "scalar" if both == ("scalar", "scalar") else None
    if isinstance(node, (ir.DotProduct, ir.CrossProduct)):
        both = (_algebra_of_node(node.left, context, value_of, depth),
                _algebra_of_node(node.right, context, value_of, depth))
        if both != ("vector", "vector"):
            return None
        return "scalar" if isinstance(node, ir.DotProduct) else "vector"
    return None


def operand_mismatch(operation: str, text: str, context,
                     value_of=None) -> "tuple[str, str] | None":
    """型が合わなければ ``(要求された型, 実際の型)``、合う/不明なら None。"""
    required = VECTOR_OP_OPERAND.get(operation)
    if required is None:
        return None
    actual = infer_algebra(text, context, value_of)
    if actual is None or actual == required:
        return None
    return required, actual


def field_operand(name: str, context, expressions) -> "tuple[str | None, str]":
    """∇ の操作の場として書かれた名前の ``(型, 操作へ渡す被演算子)``。

    宣言された場はその種類が答える。**H-1 で値を与えた名前は値が答える**——
    `u = x^2 + y^2` の `u` は `kind='anonymous'` で記録され、ベクトルの値を持つ
    `v = [y, -x, 0]` も `algebra='scalar'` と記録される（登録側の既定値であって
    宣言ではない）。割り込み口は種類だけを見ていたので、LaTeX の `\\nabla u` ・
    `\\nabla \\times v` は数式として登録されて計算されず、同じ量を `∇u` ・
    `u の勾配を求めて` と書けば計算された（実測 2026-09-17、日英とも）。

    値で答えるときは**定義した式の番号**を渡す——文の経路（2a-sym が名前を
    式番号へ書き換える）と同じ被演算子になり、成分の組み立てが値を読む。名前の
    まま渡すと、ベクトルの値を持つ名前に成分名 `v_x` を発明する。
    """
    qty = context.get_quantity(name) if context is not None else None
    if qty is None:
        return None, name
    kind = getattr(qty, "kind", None)
    if kind == "vector_field":
        return "vector", name
    if kind in ("scalar_field", "scalar"):
        return "scalar", name
    eid = getattr(qty, "defining_expr_id", None)
    expr = expressions.get(eid) if eid else None
    if expr is None:
        return None, name
    from notebook.runtime.shared_parser import strip_lhs
    value_of = value_lookup(context, lambda e: getattr(expressions.get(e), "canonical", None))
    return infer_algebra(strip_lhs(expr.canonical), context, value_of), eid
