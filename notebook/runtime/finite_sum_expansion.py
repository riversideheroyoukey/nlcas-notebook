r"""具体的な範囲を持つ Σ / Π を、項に**展開する**。

QuantumAlgebra の `∑` は**形式和**である——添字を束縛はするが、範囲を置く場所が
無い。生成器はそこへ `\sum_{i=1}^{2} …` を素通しさせていたので、**書いた範囲が
黙って消えた**（実測 2026-08-22、`[U\sum_{i=1}^{2} n_{i\uparrow}n_{i\downarrow},
c_{1\uparrow}]` が `\sum_{i}[U c†_{1↓}c_{1↑}c_{1↓}]` を成功として返す——i を含まない
項に Σ_i が掛かっており、素直に読めば答えは倍になる）。

**「バックエンドが受け取らない」は理由にならない**（利用者の指摘 2026-08-22）:
範囲が具体的な整数なら、和は有限個の項であり、展開は**機械的で厳密**——推測が
入る余地はどこにも無い。この系は既に同じことを3箇所でしている
（`expand_neighbor_sum`・`constrained_sum`・`_sum_dummy_indices` の範囲書き換え）。

範囲が**書かれていない**ときは展開しない: そこは QA の形式和が書かれたものと
正確に一致する（`\sum_\sigma` は「σ の値ぜんぶ」であって、2項の和ではない）。
範囲が**記号**のときも展開できないので、呼び手が判断できるよう `None` を返す。
"""
from typing import Optional

from notebook.models.ir import Add, Expr, IndexSum, IndexSymbol, Mul, Number


def _integer_value(node: "Optional[Expr]") -> "Optional[int]":
    """``Number(3)`` → 3。整数でないもの・記号・None は None。"""
    if not isinstance(node, Number):
        return None
    v = node.value
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


#: 展開して作ってよい項数の上限。これを超えるものは「書いて眺める」対象で、
#: 展開すると生成コードが読めなくなるうえ、CAS 側で時間もかかる。上限に当たった
#: ことは呼び手に `None` として伝わり、形式和の経路へ落ちる。
EXPANSION_CAP = 32


def expand_finite_index_sum(node: IndexSum) -> "Optional[Expr]":
    r"""``\sum_{i=1}^{2} f(i)`` → ``f(1) + f(2)``。展開できなければ None。

    展開する条件（どれも「書かれていること」だけで決まる）:

    - 束縛添字が**ちょうど1つ**。`lower`/`upper` は添字ごとではないので
      (`IndexSum` の docstring 参照)、複数添字に範囲を割り当てるのは推測になる。
    - `lower`・`upper` の**両方が具体的な整数**。
    - 項数が `EXPANSION_CAP` 以下。

    `op` が ``"product"`` なら積に展開する——Σ と Π の違いは**項に applied する
    演算子だけ**であり、範囲も束縛も同じだからである。
    """
    if len(node.dummy_indices) != 1:
        return None
    lo, hi = _integer_value(node.lower), _integer_value(node.upper)
    if lo is None or hi is None:
        return None
    if hi < lo:
        # 空の和は 0、空の積は 1。書かれたとおりの意味で、推測ではない。
        return Number(value=0 if node.op == "sum" else 1)
    if hi - lo + 1 > EXPANSION_CAP:
        return None

    from notebook.runtime.index_contractor import substitute_symbol_or_index

    name = node.dummy_indices[0].name
    # **添字の位置に入るのは `IndexSymbol("1")`、式の位置に入るのは `Number(1)`。**
    # `IndexedSymbol.indices` は `IndexSymbol` しか受けない（整数サイト `c_{1,↑}` も
    # `IndexSymbol("1")` として持たれている）ので、置換は2回行って、添字として
    # 使えた側を採る。順序が逆だと `f_i` の添字が数になり pydantic が拒否する。
    terms = []
    for k in range(lo, hi + 1):
        try:
            terms.append(substitute_symbol_or_index(
                node.body, name, IndexSymbol(name=str(k))))
        except Exception:
            terms.append(substitute_symbol_or_index(
                node.body, name, Number(value=k)))
    if len(terms) == 1:
        return terms[0]
    return Add(args=terms) if node.op == "sum" else Mul(args=terms)
