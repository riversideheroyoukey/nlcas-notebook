r"""添字をどう読むかの方針（open_items SUBSCRIPT-POLICY-SPLIT）。

``BaseSemanticResolver._resolve_subscript`` から**判断だけ**を切り出したもの。
ノードを組むのは呼び出し側（リゾルバが ``_resolve_node`` を持っているため）で、
ここは「この綴りは名前か、添字か、読めないか」を答えるところまでを担う。

**なぜ別モジュールなのか。** 添字の読み方を決める材料はこの1年で増え続けており
（区切り規則・宣言の有無・CAS 組み込み関数族・添字の個数、これから入る「和に
束縛されているか」と確認ダイアログ）、そのたびに ``_resolve_subscript`` が伸びて
``semantic_resolver.py`` のハイジーン上限が **6回**上がった——1100→1155→1250→
1300→1350→1360、**6回とも同じ関数**。材料が増えること自体は設計どおりなので、
増える場所を分ける。

**方針は1箇所に書く。** ここに集めた判断は、どれも「規則を2箇所に書いて片方が
古くなった」履歴を持つ:

* 適用形の分岐が添字分割規則を**再掲**していたため、AUDIT-SUBSCRIPT-SPLIT が
  「無区切りは1添字」に変えたとき ``A_{ij}(x)`` だけが拒否されたまま残った。
* 裸の分岐が ``isinstance(_callee, Symbol)`` を自前で見ていたため、数字添字
  ``G_0(k,omega)`` が拒否される一方 ``G_{0}(k,omega)`` は通っていた——同じ量の
  2つの綴りが違う答えを出していた（アーカイブ SUBST-DIGIT-SUB と同じ
  トークン化）。

したがってここでも規則は**呼ぶ**。何個の添字かは呼び出し側が渡す
``count_indices``（＝``_parse_indices_from_symbol``）が答える。

**CAS 非依存**（Core Principle #10）: このモジュールに CAS 名・トークン・分岐は
1つも無い。バックエンド差はリゾルバのサブクラスが持つ。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from notebook.models.ir import (
    BracketExpr, Decorated, Juxtaposition, Number, PrimeExpr, Subscript,
    Symbol,
)
from notebook.runtime.script_label import is_label_name


#: ``SymbolContext.notation_usage`` のキー接頭辞。``C_i(t)`` の ``i`` が添字か
#: 名前の一部かを利用者が答えたら、``applied_subscript:C`` に ``"index"`` /
#: ``"name"`` を記録する。**基底ごと**なのは、``C_i(t)`` について答えたことは
#: ``C_j(t)`` にも当てはまるから——訊かれているのは量 ``C`` の性質である。
#: `notation_usage` は保存/復元されるので、答えはセッションを跨いで残る。
APPLIED_SUBSCRIPT_KEY = "applied_subscript:"


def read_subscript_name(node, unwrap_negation: Callable) -> "str | None":
    r"""添字の綴りを1つの名前として読む。読めなければ None。

    3つの形すべてを1つの関数で扱う——``Symbol``（``a_i``）、``Number``
    （``a_1``——数字は ``Symbol("1")`` ではなく ``Number(value=1)`` に
    トークン化される）、連鎖 ``Subscript``（``a_k_s`` → ``"a_k_s"``）。

    形ごとに別々に判定したのが SUBST-DIGIT-SUB とその再発の原因なので、
    読み取りの入口はここ1つに保つこと。
    """
    _, unwrapped = unwrap_negation(node)
    if isinstance(unwrapped, Symbol):
        return unwrapped.name
    if isinstance(unwrapped, Number):
        v = unwrapped.value
        return str(int(v)) if float(v).is_integer() else str(v)
    if isinstance(unwrapped, Subscript):
        base = read_subscript_name(unwrapped.base, unwrap_negation)
        sub = read_subscript_name(unwrapped.sub, unwrap_negation)
        if base and sub:
            return f"{base}_{sub}"
    if isinstance(unwrapped, PrimeExpr):
        # プライムは名前の一部（「同じ名前の別の量」の印）。無印の
        # ``\psi^\dagger_\sigma(x)`` は読めるのに ``\psi^\dagger_{\sigma'}(x)``
        # ——第二の場所・第二のスピンという、グリーン関数では必ず出てくる形——
        # だけが「連鎖添字が基底の形」として断られていた（実測 2026-08-20）。
        inner = read_subscript_name(unwrapped.expr, unwrap_negation)
        if inner:
            return inner + "'"
    if isinstance(unwrapped, Decorated):
        # 装飾は名前の一部であって、添字の区切りではない。パーサは既に
        # ``psi_hat_sigma`` を ``Subscript(Decorated(psi,"hat"), sigma)`` と
        # 構造で読んでいる——装飾はここでは接尾辞の綴りではなく**フィールド**
        # である——のに、読み取りの入口が ``Decorated`` を知らないので名前が
        # 読めず、``\hat\psi_\sigma(x)`` は「連鎖添字が基底の形」として
        # 断られていた。第二量子化でいちばん普通に書かれる形が、研究者の
        # 書いていない ``M_{i,j}(x)`` の話を理由に返していた（実測 2026-08-20）。
        base = read_subscript_name(unwrapped.base, unwrap_negation)
        if base:
            return f"{base}_{unwrapped.decoration}"
    return None


def declared_subscript_role(name: str, context) -> "str | None":
    """Has the researcher DECLARED this base to be an indexed quantity?

    Two existing signals, reused rather than restated: a matrix/vector/
    tensor algebra (which the Maxima generator already reads as
    ``array_access`` for the bare form), and an explicit
    ``index_lowering_style`` (docs/07 — saying how indices lower says there
    ARE indices).  A declared SCALAR base is deliberately NOT an answer:
    ``p`` is a scalar and ``p_x(x)`` still wants its subscript as a label.
    """
    prop = context.get(name) if name else None
    if prop is None:
        return None
    qty = prop.quantity
    style = getattr(qty, "index_lowering_style", None)
    if style and style != "backend_default":
        return "indexed"
    if getattr(qty.math_type, "algebra", None) in ("vector", "matrix", "tensor"):
        return "indexed"
    return None


#: Names whose SUBSCRIPT is a binder, not an index: `\max_i x_i` is "the
#: largest over i", not `max` indexed by `i`.  Spelled without the backslash
#: because that is how they arrive here.
_EXTREMUM_BINDERS: frozenset[str] = frozenset({
    "max", "min", "sup", "inf", "argmax", "argmin",
})


def extremum_binder(expr) -> "str | None":
    r"""The binder name when *expr* is ``\max_i`` / ``\min_i`` / ``\sup_i`` …

    None otherwise, including the ordinary two-argument ``\max(a, b)``, which
    is a plain function call and works.

    **Why this is refused rather than read.**  Measured 2026-08-16, the
    range-less form lowered to ``max(i) * x(i)`` — and Maxima's ``max`` of one
    argument IS that argument, so ``\max_i x_i`` computed as ``i · x(i)``:
    valid CAS text, a completely different quantity, no error anywhere.
    ``\sup_i`` was worse (``sup`` is not a Maxima function at all), and
    ``\max_{i=1}^{3} x_i`` failed with a raw pydantic ValidationError as its
    reason — three different wrong answers for one notation.

    **Why it is not given Σ's machinery.**  ``IndexSum.op`` carries Σ and Π
    together, and its own contract is that the two "differ ONLY in the operator
    applied to the terms: same dummies, same ranges, same placeholder pairs".
    An extremum does not qualify: measured, ``lmax(makelist(x[i],i,1,3))``
    works but ``lmax(makelist(x[i],i,1,n))`` FAILS, so a symbolic range has no
    form — and the range-less form's placeholder pair (``nbSumLo1``) is
    symbolic by construction.  Riding that node would turn a silent wrong
    answer into a silent failure.

    The bounded-and-concrete case IS computable and is worth having; it needs a
    representation of its own (open_items EXTREMUM-BINDER) rather than a seat
    on a node whose invariant it breaks.
    """
    base = getattr(expr, "base", None)
    if not isinstance(base, Symbol) or base.name not in _EXTREMUM_BINDERS:
        return None
    return base.name


@dataclass(frozen=True)
class CalledSubscript:
    """``X_sub(args)`` の読み。``kind`` が答えで、残りはその材料。

    * ``name``    — 添字は名前の一部。``fold_name`` がその関数名。
    * ``indexed`` — 添字は添字。``indices`` がその並び。
    * ``refuse``  — 読めない。呼び出し側が理由付きで断る。
    """
    kind: Literal["name", "indexed", "refuse"]
    fold_name: "str | None" = None
    indices: "list | None" = None
    base_name: "str | None" = None
    """基底の名前。読み手が決めた綴りをそのまま返す——``Decorated`` の基底では
    ``expr.base.name`` が存在しないので、呼び出し側が名前を作り直すと二か所が
    別々に答えることになる。"""


def classify_called_subscript(
    expr, context, *,
    read_name: Callable,
    count_indices: Callable,
    extract_indices: Callable,
    bound_indices: "frozenset[str] | set[str]" = frozenset(),
    base_role: "Callable | None" = None,
    is_builtin_family: "Callable | None" = None,
) -> CalledSubscript:
    r"""``X_sub(args)`` の添字は名前の一部か、添字か。

    パーサは意味を持たないので、``bessel_j(0,x)`` も ``a_i(k)`` も
    ``D_{i,j}(x)`` も同じ ``Subscript(base, FunctionCall(...))`` として届く。
    ここがその読み分けを決める唯一の場所。

    **単一添字は名前に畳む。** ``bessel_j(0,x)`` は「bessel を (j への呼び出し)
    で添字付けした」ではなく関数 ``bessel_j`` の呼び出しで、それ以外の読みは
    ない。畳まないと ``_extract_indices`` に流れて生の pydantic ValidationError
    になっていた（2026-07-10、数値層 S-ac の実践ループで発覚）。

    **多重添字は添字のまま。** ``D_{i,j}(x)`` / ``g_{\mu\nu}(x)`` /
    ``a_{i,\sigma}(k)``。バックエンドがその形に持つものへ落とす（Maxima は
    subscripted function ``D[i,j](x)``、Julia は QuExpr が呼び出せないので拒否）。

    **なぜ2添字が線なのか**（実測 2026-07-31）。ラベルを区切って書く人はいない
    ——同時分布を ``p_{x,y}`` とは書かない——ので、区切られた添字は曖昧さなく
    添字である。単一添字はそうではない: ``C_i(t)`` は添字にしたく、``p_x(x)``
    は名前にしたく、綴りが同型。単一も添字として扱う案を実装して測ったところ、
    ``p_x(x)`` と ``p_y(y)`` が ``x=1, y=1`` の代入で**どちらも ``p[1](1)`` に
    潰れ**、x の周辺分布と y の周辺分布が区別できなくなった（相互情報量の式が
    エラーなしで別物になる）。``p_xy`` は宣言も組み込みもされないので登録簿でも
    関数族リストでも救えない。**判別材料が出るまで単一添字は畳んだまま**にする。

    **例外は「和に束縛された添字」**（第2段、2026-07-31）。``\sum_i`` の内側では
    ``i`` は総和の走る添字だと**確定している**ので、単一でも添字として読む。
    推測が入らないのがこの材料の要点で、これで
    ``\sum_i A_i(x)B_i(x)`` が ``A(1)B(1)+A(2)B(2)+A(3)B(3)`` に展開されるように
    なる——それまでは ``3*A_i(x)*B_i(x)`` という**エラーの出ない誤答**で、電磁気の
    ベクトル解析で普通に踏む形だった。

    束縛の外に単独で書かれた ``C_i(t)`` はまだ畳まれる（第3段＝添字分類ダイアログの
    領分）。``\sum_x p_x(x)`` のように**ラベルと同じ文字で和を取る**と添字として
    読まれるが、その書き方自体が「x について和を取る」と宣言しているので、
    推測ではなく利用者の指定に従った結果である。

    :param read_name: 綴り → 名前（``read_subscript_name`` の束縛版）
    :param count_indices: 名前 → 添字の個数（``_parse_indices_from_symbol``）。
        **再掲せず呼ぶこと**——モジュール docstring の履歴を参照
    :param extract_indices: 綴り → ``IndexSymbol`` の並び
    :param bound_indices: いま囲っている和が束縛している添字名
    :param base_role: 基底名 → ``"indexed"`` / None（宣言を読む）
    :param is_builtin_family: フルネーム → CAS 組み込みの関数族か
    """
    callee = expr.sub.func
    if isinstance(expr.base, Symbol):
        base_name = expr.base.name
    elif isinstance(expr.base, Decorated):
        # 装飾された名前（``\hat\psi_\sigma(x)``）。装飾は名前の一部なので、
        # ここから先は素の名前とまったく同じに読む——「装飾を添字の区切りとして
        # 数えない」とはこのこと。連鎖添字の基底（下）とは違い、読みはある。
        base_name = read_name(expr.base)
    else:
        base_name = None
    if base_name is None:
        # 連鎖添字が基底の形（``A_i_j(x)``）。読みが無いので断る。
        return CalledSubscript(kind="refuse")

    indices = None
    try:
        indices = extract_indices(callee, context)
    except Exception:
        indices = None
    if indices and len(indices) >= 2:
        return CalledSubscript(kind="indexed", indices=indices,
                               base_name=base_name)

    # 添字が**角括弧で書かれていた**（`f[i](x)`）なら、単一でも添字である。
    # 下の畳み込みが答えているのは `_` の綴りの曖昧さ——`C_i(t)` は添字にしたく
    # `p_x(x)` は名前にしたい、が同型——という問いで、角括弧にその曖昧さは無い。
    # これはバックエンド自身の添字記法であり、**このシステムの生成器が書いた
    # 綴り**でもある（`FunctionCall(func=IndexedSymbol)` → `f[i](x)`、宣言に
    # よらず）。畳むと往復が閉じない: 実測 2026-08-16、
    # `simplify \int \sum_i f_i(x) dx` が組んだ `sum(f[i](x), i, nbSumLo1, …)`
    # を読み直す段で `f[i](x)` が `f_i(x)` になり、**和の走る添字に届かない
    # 別の CAS オブジェクト**に化けていた（そこに至る前は `f * [i](x)` という
    # リストとの積で、こちらは誤答を成功として返していた）。
    if getattr(expr, "bracketed", False) and indices:
        return CalledSubscript(kind="indexed", indices=indices,
                               base_name=base_name)

    # 単一添字: 名前に畳む。3つの綴り（裸シンボル・裸数字・中括弧）はどれも
    # 同じ1つの添字を指すので、読み取りは read_name に一本化する。
    # Juxtaposition は空白区切り（``A_{i j}``）で、構造上つねに多重添字。
    if isinstance(callee, BracketExpr):
        only = callee.args[0] if len(callee.args) == 1 else None
    else:
        only = callee
    name = (read_name(only)
            if only is not None and not isinstance(only, Juxtaposition) else None)
    if name and count_indices(name, context) == 1:
        fold = CalledSubscript(kind="name", base_name=base_name,
                               fold_name=f"{base_name}_{name}")
        as_index = (CalledSubscript(kind="indexed", indices=indices,
                                    base_name=base_name)
                    if indices else fold)
        # 単一添字の解決順序。既存の記法確認 UI（CLAUDE.md）と同じ順で、
        # **利用者が既に言ったことを推測より先に使う**:
        #   0. 上付きラベルの綴り       → 名前（`G_sup0`・`Sigma_psupHF`）
        #   1. CAS 組み込みの関数族     → 名前（`bessel_j(0,x)`。訊く余地なし）
        #   2. 和が束縛している         → 添字（第2段。推測ゼロ）
        #   3. 基底の宣言               → 添字（行列・テンソル等、または
        #                                 index_lowering_style の明示）
        #   4. notation_usage の記録    → 前に答えてもらった内容
        #   5. どれも無ければ畳む       → 従来どおり（`p_x(x)` が守られる）
        # 5 に落ちるのが「訊く」余地の残っている唯一の場所である。
        # 0. 上付きラベル（`G^0` → `G_sup0`、`\Sigma^{(HF)}` → `Sigma_psupHF`）は
        #    **このシステム自身が書いた名前**であって添字ではない。裸の形では
        #    パーサが既にそう読んでいる（`is_label_name` → `Decorated`）が、
        #    適用形はここを通るので、同じ問いにこちら側の答えが要る。無いと
        #    宣言が先に効いて名前が割れる: `\Sigma` を行列と宣言したセッションで
        #    `\Sigma^{(HF)}(n,\sigma)` が `Sigma[psupHF](n,sigma)` になり、
        #    そのうえ Maxima 側では `\Sigma = \Sigma^{(1)}` の束縛が基底を
        #    置き換えて `Sigma_psup1[psupHF]` ——**誰も書いていない量**——になった。
        #    続く代入は当然どこにも一致せず、**成功として**何もせず返った
        #    （実測 2026-08-21、AUDIT-SUBST-INDEXED と同じ形）。
        if is_label_name(name):
            return fold
        if is_builtin_family is not None and is_builtin_family(fold.fold_name):
            return fold
        if name in bound_indices:
            return as_index
        if base_role is not None and base_role(base_name, context) == "indexed":
            return as_index
        remembered = (context.notation_usage or {}).get(
            f"{APPLIED_SUBSCRIPT_KEY}{base_name}")
        if remembered == "index":
            return as_index
        if remembered == "name":
            return fold
        return fold
    return CalledSubscript(kind="refuse")
