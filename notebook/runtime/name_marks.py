r"""名前に付いた**印**が何を意味するか。

共役 ``\psi^*``、随伴 ``\psi^\dagger``、プライム ``x'``、そして代数的な添字
``c_{k+q}`` —— どれも「基底の名前に印が付いている」という一つの問いで、
2026-08-20〜21 の一日でこの族に6回続けて手が入った。読みは resolver の中に散らばり、
衛生ラチェットの上限を6回引き上げるあいだ「次に足すときは切り出す」と書き続けた末に、
足し続けてしまった。ここがその切り出し先である。

**分割の線は `subscript_policy` と同じ**: この module は**読み**（何を意味するか）を
持ち、**ノードの構築**は resolver に残す。resolver だけが ``_resolve_node`` を持つので、
解決が要る判断は callback で受け取る（``classify_called_subscript`` と同じ形）。

族の履歴（実測、いずれもエラー無しの誤答だった）:

* ``\psi^*(x)`` が ``psi^[ast](x)`` —— 記号のリスト乗（2026-08-09）
* ``\psi^\dagger`` の5つの綴りが**5通りの別物**（転置・``dagger(x)`` 乗・積…、08-20）
* ``x'`` が Maxima の構文エラーになり、アダプタがそれを**成功**として返す（08-20）
* ``v(\mathbf r-\mathbf r')`` が ``v(0)`` —— プライムが黙って捨てられる（08-20）
* ``c_{-k}`` が生の pydantic、``c_{k+q}`` が ``c(k) + q``（08-20）
"""
from __future__ import annotations

from notebook.models.ir import (
    BracketExpr, Decorated, Expr, FunctionCall, IndexSymbol, IndexedSymbol,
    Symbol,
)
from notebook.runtime.shared_parser import UnknownNotationError

#: 「これを共役に」と読む上付き、resolver に届くすべての綴り。
CONJUGATE_MARKS = ("ast", "star", "*")

#: …「随伴に」と読むもの。4つの綴りは変換器の入口で既に1つにされている
#: (`latex_converter._dagger_mark`)。ここに残るのはその成れの果て。
DAGGER_MARKS = ("dagger", "dag")

#: この系がベクトルを包む綴り。プライムは包みの**中**の名前に付く。
_VEC = "vec"


def script_mark(script: Expr) -> "str | None":
    r"""上付きが担っている素の名前、無ければ ``None``。

    ``^*``・``^{\ast}``・``^{\dagger}`` はどれもここに届く —— ``Symbol`` か、
    1要素の中括弧 ``BracketExpr`` か、``\mathrm{}`` に包まれた形。**読みは一つ**に
    保つ: 適用形（``\psi^*(x)``）と素の形（``\psi^*``）が別々に読んでいたら、
    同じ綴りが2つの意味に割れる。
    """
    if isinstance(script, Symbol):
        return script.name
    if isinstance(script, BracketExpr) and len(script.args) == 1:
        inner = script.args[0]
        if isinstance(inner, Symbol):
            return inner.name
        if (isinstance(inner, IndexedSymbol)
                and getattr(inner.base, "name", None) in ("mathrm", "text")
                and len(inner.indices) == 1):
            return inner.indices[0].name
    if (isinstance(script, IndexedSymbol)
            and getattr(script.base, "name", None) in ("mathrm", "text")
            and len(script.indices) == 1):
        return script.indices[0].name
    return None


def is_mark(mark: "str | None") -> bool:
    """共役か随伴の印か（どちらでもなければ False）。"""
    return mark in CONJUGATE_MARKS or mark in DAGGER_MARKS


def is_dagger(mark: "str | None") -> bool:
    return mark in DAGGER_MARKS


def dagger_reading(base_surface: Expr, algebra_of, *,
                   default_is_adjoint: bool) -> "tuple[str, Expr]":
    r"""``X^\dagger`` は **随伴** か、**装飾された名前** か。

    実測 2026-08-20、一つの量が5通りに書かれて5通りの別物になっていた ——
    ``'transpose(psi)``、``psi^dagger(x)``（psi の ``dagger(x)`` 乗）、
    ``'transpose(psi)(sigma)``、``'transpose(psi) * x``。どれも研究者が書いた
    生成演算子ではない。

    Maxima セッションには演算子代数が無い（open_items F-2）ので、そこでの
    ``\psi^\dagger`` は**名前**である —— ``\hat\psi`` とまったく同じもので、同じ
    ``Decorated`` の機構を通る（だから添字も引数リストも既に効く）。随伴が本当に
    意味を持つ場合、すなわち基底が**行列・ベクトル・テンソル・演算子と宣言された**
    ときは随伴のまま。判別材料が宣言なのは、隣の ``^T`` の分岐が既にそう読んで
    いたからで、同じ問いが隣り合う2行で別々に答えられていた。

    :param algebra_of: 引数なしで呼ぶと基底の代数種別を返す callable。解決は
        resolver にしかできないので受け取る（`classify_called_subscript` と同じ形）。
        遅延にしてあるのは、**解決に副作用がある**ため —— プライム付きの名前は
        解決時に登録簿へ登録されるので、二度解決すると二重登録になる
    :param default_is_adjoint: 未宣言の基底をどう読むか。True が共有の既定
        （Julia: QuantumAlgebra / Symbolics は任意の式に随伴を持つ）。Maxima は
        False —— Backend Encapsulation Rule に従い、その差はバックエンドの側に置く。
    :returns: ``("adjoint", 基底の**表層**ノード)`` か ``("name", 装飾ノード)``。
        随伴の側で表層を返すのは、解決を持っているのが呼び出し側だから ——
        resolver は ``algebra_of`` で既に解決した結果に差し替える。ここで
        解決済みを返せるように見えるのは ``algebra_of`` を呼んだ経路だけで、
        ``default_is_adjoint`` の経路は基底を一度も解決しない。
    """
    if default_is_adjoint:
        return "adjoint", base_surface
    if isinstance(base_surface, Symbol):
        if algebra_of() in ("matrix", "vector", "tensor", "operator"):
            return "adjoint", base_surface
    return "name", Decorated(base=base_surface, decoration="dagger")


def prime_on_non_symbol(inner_resolved: Expr, order: int) -> Expr:
    r"""素の記号でないものに付いたプライムを、**名前の上に置き直す**。

    プライムは素の記号に付いたときと同じ意味 —— 「同じ名前の別の量」 —— なので、
    置き場所は名前しかない。既定は**黙って落として**いた: ``v(\mathbf r-\mathbf r')``
    は Maxima に ``v(vec(r) - vec(r))``、つまり **``v(0)``** として届き、``c_k'`` は
    ``c(k)`` になっていた（実測 2026-08-20）。

    名前を持つ形は3つ。それ以外は**理由付きで断る** —— 置き場所の無いプライムは、
    書かれなかったプライムではない。
    """
    primes = "'" * order
    if isinstance(inner_resolved, IndexSymbol):
        # 束縛添字にもプライムは付く（``\sum_{k'}``・``c_{k'\sigma}``）。
        # 添字は名前なので、同じように受ける。
        return IndexSymbol(name=inner_resolved.name + primes)
    if (isinstance(inner_resolved, IndexedSymbol)
            and isinstance(inner_resolved.base, Symbol)):
        return IndexedSymbol(base=Symbol(name=inner_resolved.base.name + primes),
                             indices=inner_resolved.indices)
    # ``\mathbf{r}'`` —— 変換器は太字を ``vec(r)`` と包むので、研究者が名前に付けた
    # プライムが包みの外に落ちる。中へ戻す: ``vec(r)'`` → ``vec(r')``。
    if (isinstance(inner_resolved, FunctionCall)
            and isinstance(inner_resolved.func, Symbol)
            and inner_resolved.func.name == _VEC
            and len(inner_resolved.args) == 1
            and isinstance(inner_resolved.args[0], Symbol)):
        return FunctionCall(
            func=inner_resolved.func,
            args=[Symbol(name=inner_resolved.args[0].name + primes)])
    raise UnknownNotationError("parser.unsupported_notation.prime_on_compound")


def index_from_compound(node: Expr, is_up: bool) -> IndexSymbol:
    r"""**代数的な添字** —— ``c_{-k}``・``c_{k+q}``・``A_{i+1}`` —— を、書かれた式を
    名前とする1つの ``IndexSymbol`` にする。

    多体と格子の標準記法で、両 CAS がそのまま受ける（``c(-k)``・``c(k+q)``）。
    **Julia のリゾルバは最初からこう読んでいた**（「complex subscript は文字列
    IndexSymbol にする」）。共有側が持っていなかっただけで、Maxima 側では複合が
    ``IndexedSymbol(indices=[…])`` の型検査に当たり、研究者に生の pydantic
    ValidationError が出ていた —— ``c_{-k}``、BCS の対の相手で（実測 2026-08-20）。
    """
    from notebook.runtime.expr_generator import ExprGenerator
    return IndexSymbol(name=ExprGenerator().generate(node), is_up=is_up)
