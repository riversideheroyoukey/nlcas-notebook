"""被演算子がその操作を受けられるか（使う地点での断り）。

``NotebookEngine.run_operation`` から**断りの判断だけ**を切り出したもの。

**なぜ別モジュールなのか。** ここは材料が増え続ける場所である——F-9 の表示専用
状態（2026-07-28）、範囲つき定積分の表示専用オブジェクト（2026-08-03）、そして
ベクトル解析の被演算子型（H-6、2026-08-06）で3つ目。`subscript_policy` /
`expr_ref_fields` と同じで、増える場所を分ける。

**共通する規則は1つ**——「タイムラインに出したものは計算できるはず」なので、
**できないなら使う地点で理由付きで断る**。黙って通すと、もっともらしい答えが
返って研究者はそのまま先へ進む。実測された3つの形:

* 係数だけの角運動量状態に演算を掛けると ``unsupported_notation.pipe``
  （「パイプ記号は未対応」）という何も説明しないエラーになっていた
* 範囲つき定積分に一般操作を掛けると、アダプタ自身の作業変数名 ``_res_`` が
  **成功した式として登録**されていた
* スカラーの発散が ``y^2+x^2``（そのまま）、ベクトル場の勾配が ``[0=0, 0=0, 0=0]``
  として**成功**していた（H-6）

**断るのは確信があるときだけ。** 型が読めない被演算子は従来どおり通す——
正しい計算が理由なく止まる方が害が大きい（判定は ``vector_op_types``）。
"""
from __future__ import annotations

import re

from notebook.runtime.i18n import t
from notebook.runtime.operand_boundary import declared_object_names
from notebook.runtime.expr_ref_fields import VARIABLE_FIELDS
from notebook.runtime.token_field_mapping import EXPR_FIELDS
from notebook.runtime.vector_op_types import operand_mismatch, value_lookup

_EXPR_ID_RE = re.compile(r"^E\d+$")


#: 型宣言の操作 → このノートでの代数。**1つの定義**——エンジンの登録側もここを読む
#: （2箇所に書くと、片方が新しい型を知らないまま古い答えを出す）。
TYPE_DECL_ALGEBRA: dict[str, str] = {
    "define_scalar_parameter": "scalar",
    "define_vector":           "vector",
    "define_matrix":           "matrix",
    "define_tensor":           "tensor",
}


#: 結果が **CAS の値ではない**操作。汎用の ``run_operation`` からは走らせない。
#:
#: `heisenberg_operator` の `BuildResult` は「基底;…|行;…」という**内部プロトコル
#: 文字列**で、それを読めるのは `engine.heisenberg_operator`（QA に閉包を訊き、
#: スカラー側に指数を訊く）だけである。`ui/interpret_pair_rep.py` の冒頭は最初から
#: 「`run_operation` の一般経路には乗せない」と書いていたが、`run_operation` 側は
#: それを知らされていなかった——そして別名走査は `heisenberg operator` /
#: 「時間発展」をこの操作名に解決するので、`the Heisenberg operator of a(1) under E1`
#: は `a_{1} \vee 1 \mathrel{\ldotp\mkern-2.5mu\ldotp} - w`、つまりプロトコル文字列の
#: Latexify 表示を **`completed` として公開していた**（実測 2026-09-06）。
#:
#: 読みの側（`heisenberg_operator_nl`）は実測した綴りを全部拾うようにしたので、
#: ここへ来るのは測っていない綴りだけである。`\max_i` のときと同じで、**読みを
#: 広げてもガードは残す**——別名走査にも LLM にもこの操作名に辿り着く道がある。
STRUCTURE_ONLY_OPERATIONS: frozenset[str] = frozenset({"heisenberg_operator"})


def reject_structure_only_operation(operation: str, *, lang: str) -> None:
    """結果が構造で値でない操作を、汎用経路の入口で理由付きで断る。"""
    if operation in STRUCTURE_ONLY_OPERATIONS:
        raise ValueError(t("pair_rep.not_a_generic_operation", lang,
                           operation=operation))


def reject_conflicting_redeclaration(operation: str, args: dict, *,
                                     symbol_context, lang: str) -> None:
    r"""既に**別の型で**宣言されている名前を、黙って上書きさせない。

    宣言はこの系のほとんどの読みの判別材料である（添字が index か名前か、`^{-1}`
    が逆行列か逆数か、積が可換か）。だから型を変えるということは、**その名前を
    使って既に書いた式の意味が変わる**ということで、黙って起きてよいことではない。

    実測 2026-08-22（利用者報告の隣で発見）: `\hat H_0 をスカラーとする` のあとに
    `\hat H_0 を行列とする` と書くと、返るのは `Maxima evaluation error` ——CAS 内部の
    文字列で、何が起きたかも次に何をすればよいかも言っていない（Core Principle #1）。
    CAS が断ること自体は正しい（`scalar` と `nonscalar` は両立しない）。

    **既定は断る**（利用者の決定、2026-08-22）。上書きは「宣言し直して」と明示した
    ときだけで、そのときは前の宣言を CAS から取り消してから宣言する。
    """
    if operation not in TYPE_DECL_ALGEBRA or args.get("_redeclare"):
        return
    new_algebra = TYPE_DECL_ALGEBRA[operation]
    from notebook.runtime.latex_converter import restore_decorations
    names = args.get("_all_names") or [args.get("name", "")]
    for name in names:
        if not name:
            continue
        existing = symbol_context.get(name)
        if existing is None:
            continue
        old_algebra = getattr(existing.quantity.math_type, "algebra", None)
        if old_algebra and old_algebra != new_algebra:
            # 名前は**表示の扉**を通す。断り文に出すのが `H_hat_0` では、研究者が
            # 書いた `\hat H_0` と結びつかない（Core Principle #1）。
            raise ValueError(t(
                "declare.type_conflict", lang,
                name=restore_decorations(name),
                old=t(f"declare.algebra_{old_algebra}", lang),
                new=t(f"declare.algebra_{new_algebra}", lang)))


#: 素の名前が **CAS の値ではない**宣言（実測 2026-09-11）。ベクトル場は成分
#: （`A_x, A_y, A_z`）にしか居らず、行列・テンソル・ベクトルは `nonscalar` と
#: 宣言された記号でしかない——どちらも「A は x に依らない」と読まれる。
#: **スカラー場も入る**——`differentiate` だけは `'diff(A,x,1)` という正直な形を
#: 返すが（`depends` が宣言されているため）、`\int A\,dx` は `A*x`、
#: `\sum_{x=0}^{1} A` は `2*A` と、**依存を忘れた答え**を返す（実測 2026-09-11）。
#: 正直な `differentiate` は種別ごとの免除表（`declaration_reading_ops`）に載せる。
#: **演算子は入らない**: Julia では素の名前そのものが QuantumAlgebra の対象である。
_VALUELESS_KINDS: frozenset[str] = frozenset({"vector_field", "scalar_field"})
_VALUELESS_ALGEBRAS: frozenset[str] = frozenset({"matrix", "tensor", "vector"})


def _declared_type_label(quantity, lang: str) -> str:
    """宣言の種別を**研究者の言語で**呼ぶ。"""
    kind = getattr(quantity, "kind", None)
    if kind in _VALUELESS_KINDS or kind == "scalar_field":
        return t(f"declare.kind_{kind}", lang)
    algebra = getattr(getattr(quantity, "math_type", None), "algebra", None)
    return t(f"declare.algebra_{algebra}_a", lang)


def reject_declared_object_without_value(operation: str, args: dict, *,
                                         symbol_context, reads_declaration,
                                         operation_label: str,
                                         reading_ops_label,
                                         lang: str) -> None:
    r"""宣言されただけの対象を、**値を必要とする操作**に渡させない。

    名前を式クラスの欄に通せるようにした結果（OPERAND-BOUNDARY 残①）、
    **開いた面を全数で測ると 47 操作のうち宣言を読むのは 3 つだけ**だった
    （ベクトル場: `curl` / `divergence` / `norm` が読み、9 つが断り、
    **35 が宣言を無視して素の記号として CAS に渡す**）。素の記号として渡ると
    CAS は「A は x に依らない」と読むので、失敗していた綴りが**もっともらしい
    誤答**に変わる——`A を x で微分して` → `0`、`積分して` → `A*x`、
    `総和` → `2*A`。系自身の `divergence` が `'diff(A_x,x,1)` を返すことと
    矛盾している。

    **断り文は「できない」ではなく、満たされていない条件を述べる**（依頼者の
    指定）。条件は `defining_expr_id` そのもの——宣言だけなら `None`、
    `A = …` と値を与えると `'E1'` になる（実測）。だから断り文は
    「値が与えられていない」と言い、**測って通ることを確かめた道**だけを示す:

        A をベクトル場とする / A = \begin{pmatrix} x^2 \\ 0 \\ 0 \end{pmatrix}
          A を x で偏微分して  →  matrix([2*x],[0],[0])   ← 値を与える道
          A の発散を求めて      →  'diff(A_z,z,1)+…        ← 宣言だけで計算する道

    **免除は「操作」ではなく「操作と種別の組」で決まる**（実測 2026-09-11）。
    `gradient` / `scalar_laplacian` はスカラー場を読むが、**行列・テンソルに
    対しては宣言を無視して `[0,0,0]` / `0` を返す**——操作名だけで免除すると、
    塞いだはずの穴をその2つが開けたままにする。だから *reads_declaration* には
    種別も渡す。

    **免除には「型エラーで断る」組も入る。** `curl` にスカラー場を渡すと
    「'f' はスカラー場ですが、ベクトル場が必要です」という**より正確な理由**が
    返る——値を与えても `curl` はスカラー場に対して正しくないので、満たされて
    いない条件は値ではない。手前でこの guard が答えると、その理由を消してしまう
    （全スイートが落ちて気づいた）。一方 *reading_ops_label* が案内するのは
    **実際に計算できる**ものだけである（`declaration_computing_ops`）——免除の
    一覧をそのまま見せると、効かない回避策を勧めることになる。

    *reads_declaration* はバックエンドが答える（どの操作が宣言から組み立てるかは
    そのバックエンドの知識である——Backend Encapsulation）。
    """
    names = declared_object_names(symbol_context)
    if not names:
        return
    for field, value in args.items():
        if field not in EXPR_FIELDS or not isinstance(value, str) or value not in names:
            continue
        qty = symbol_context.get_quantity(value)
        if qty is None or getattr(qty, "defining_expr_id", None):
            continue
        kind = getattr(qty, "kind", None)
        algebra = getattr(getattr(qty, "math_type", None), "algebra", None)
        if kind not in _VALUELESS_KINDS and algebra not in _VALUELESS_ALGEBRAS:
            continue
        if reads_declaration(operation, kind, algebra):
            continue
        ops = reading_ops_label(kind, algebra)
        key = ("operand.declared_without_value" if ops
               else "operand.declared_without_value_no_op")
        raise ValueError(t(key, lang, name=value,
                           type_label=_declared_type_label(qty, lang),
                           operation=operation_label,
                           example=_value_example(value, kind, algebra),
                           reading_ops=ops))


#: 値の与え方の**例**（右辺だけ）。種別に合う形でなければ案内にならない
#: ——行列に列ベクトルを見せていた（実測 2026-09-11）。綴りは記法そのものなので
#: 言語を問わない。左辺は**研究者が書いた名前**を入れる。
_VALUE_EXAMPLE = {
    "scalar_field":  "x^2 + y^2",
    "vector_field": r"\begin{pmatrix} x^2 \\ 0 \\ 0 \end{pmatrix}",
    "vector":       r"\begin{pmatrix} 3 \\ 4 \end{pmatrix}",
    "matrix":       r"\begin{pmatrix} x^2 & 0 \\ 0 & 1 \end{pmatrix}",
    "tensor":       r"\begin{pmatrix} x^2 & 0 \\ 0 & 1 \end{pmatrix}",
}


def _value_example(name: str, kind, algebra) -> str:
    #: 右辺の無い「例: `f`」は**値の与え方の例になっていない**（自分で踏んだ、
    #: 実測 2026-09-11——スカラー場の欄が抜けていた）。表に無い種別が来たら
    #: 例を出さずに名前だけを返すのではなく、いちばん素直な式を出す。
    rhs = (_VALUE_EXAMPLE.get(kind) or _VALUE_EXAMPLE.get(algebra)
           or _VALUE_EXAMPLE["scalar_field"])
    return f"{name} = {rhs}"


def translate_value_bearing_variable_failure(
        operation: str, args: dict, *, symbol_context, cas_holds_value,
        value_of_name, operation_label: str, lang: str) -> None:
    r"""CAS が断ったあとに読み替える——**値を持つ名前は変数ではない**。

    H-1 の P2 は定義を CAS 側へ束縛する（`y : x^3`）。その名前が操作の**変数の欄**に
    立つと Maxima は断り、研究者に返るのは生の `Maxima evaluation error` ——原因も
    次の一手も言っていない文字列である（Core Principle #1）。実測 2026-09-19、
    `z^2 + y` を登録して `y = x^3` と書いたあと、日英とも:

        E1 を y で偏微分して / 全微分して          生の `Maxima evaluation error`
        E1 を y で 0 から 1 まで定積分して          同上
        E1 を y について 0 のまわりで 3 次まで展開     同上

    **∇ と同じ直し方（`block([…])` の影）は実測で却下されている。** 座標は操作自身が
    束縛するので隠してよいが、ここは研究者がその名前を文に書いている。しかも P2 は
    代入時に評価するので `r = \sqrt{x^2+y^2+z^2}` のあとの `f = 1/r` は `r` を含まない
    形で格納され、隠すと `diff(f,r)` は `-1/r^2` ではなく **`0`** を返す。大きな声の
    エラーが静かな誤答に変わるのは交換ではなく悪化である。

    **事前に断ってはならない。境界は欄ではなく CAS にある**（実測 2026-09-19、
    束縛された名前を変数の欄に置いて15操作）:

    ==================================  ==========================================
    断る                                 `diff` / 定積分 / `taylor` / `limit` / `residue`
    答える（**それが正しい**）                 `solve` / `sum` / `product` / `ratsimp` /
                                        `partfrac` / `charpoly` / `coeff`
    ==================================  ==========================================

    `solve` を欄で先回りして断ったら、**公開実例の本筋が1本止まった**——
    `scenario_coulomb_gauge` は `Az_z = \partial A_z/\partial z` と名前をつけてから
    ゲージ条件を `Az_z` について解いており、Maxima は「その名前が指すもの」について
    正しく答える（`'diff(A_z,z,1) = -'diff(A_y,y,1)-'diff(A_x,x,1)`）。コーパス差分が
    捕まえた。**断りの理由が入力に当たっているかは、族ごとに CAS へ訊いて測る。**

    **帳簿ではなく CAS が決める**のはここでも同じで、`defining_expr_id` は上位集合で
    ある（解の結果は id を持つのに束縛しない、実測）。帳簿は**安いふるい**として使い、
    普通の変数 `x` / `t` / `k` はそこで落ちるので CAS には訊かない。

    **断り文は満たされていない条件を述べ、測って通る道だけを名指す**（実測）:

        y = x^3 のあと
          E1 を u で偏微分して   →  0        ← 値を持たない名前を変数にする
          E1 を x で偏微分して   →  3*x^2    ← 値が書かれている変数で操作する
          y をスカラーとする        →  戻らない（宣言しても束縛は消えない）
    """
    for field, value in args.items():
        if field not in VARIABLE_FIELDS or not isinstance(value, str):
            continue
        # **形は見ない。** 帳簿の照会がそれより狭いから——`get_quantity("x + 1")`
        # は None である（変異試験で生き残り、測って削った 2026-09-19）。CAS へ
        # 送る前の形の検査はバックエンド側にあり、そこは訊く直前なので意味がある。
        name = value.strip()
        qty = symbol_context.get_quantity(name) if symbol_context else None
        eid = getattr(qty, "defining_expr_id", None) if qty else None
        if not eid or not cas_holds_value(name):
            continue
        # **名前は表示の扉を通さない**——`var` は研究者の綴りをそのまま運ぶ
        # （実測 2026-09-19、`\bar{y}` で微分すると `var='\\bar{y}'` が届く）。
        # 飾りのある名前はそもそも帳簿に当たらないので、ここへ来るのは素の
        # 識別子だけである（変異試験で `restore_decorations` が生き残り、測って
        # 削った）。飾りのある変数は**値の有無にかかわらず** `0` を返す別の
        # 欠陥に落ちている（起票 DECORATED-NAME-AS-DIFF-VAR）。
        raise ValueError(t("operand.variable_has_a_value", lang, name=name,
                           value=value_of_name(eid) or eid,
                           eid=eid, operation=operation_label))


def translate_indexed_variable_failure(
        operation: str, args: dict, *, lowered_of, operation_label: str,
        lang: str) -> None:
    r"""CAS が断ったあとに読み替える——**添字付きの名前は変数にならない操作がある**。

    `x_1` は宣言が無ければ**成分の添字**として読まれ、CAS へは `x(1)` ——関数
    呼び出し——として届く。実測 2026-09-19、Maxima はそれを操作によって受けたり
    断ったりする:

    ==============================  ========================================
    断る                             定積分 / `limit` / `taylor` / `sum`
    受ける（そして正しい）                   `diff` / 不定積分 / `residue` /
                                    `partfrac` / `coeff` / `solve` / `charpoly`
    ==============================  ========================================

    だからここも**事前に断らず、断られてから読み替える**（DEFINED-NAME-AS-VARIABLE
    と同じ形、同じ場所）。定積分だけは IR の段階で先に断っており
    （`integral.var_is_indexed`、`Integral.var` は `Symbol` を要求する）、その文面を
    操作名で一般化したのがここの断りである。

    **到達するようになったのは 2026-09-19** ——別名走査が `x_1` を変数の欄に入れる
    ようになった日で（DECORATED-VAR-NOT-TOKENIZED）、それまでは手前で
    「必要な引数が不足しています」と落ちていた。**出口は足すたびに数え直す。**
    """
    import re as _re
    for field, value in args.items():
        if field not in VARIABLE_FIELDS or not isinstance(value, str):
            continue
        name = value.strip()
        if not name:
            continue
        lowered = lowered_of(name)
        if not lowered or not _re.fullmatch(r"[A-Za-z_]\w*\([^()]*\)", lowered):
            continue
        raise ValueError(t("operand.variable_is_indexed", lang, var=name,
                           lowered=lowered, base=name.split("_")[0],
                           operation=operation_label))


def reject_unusable_operands(operation: str, args: dict, *,
                         expressions: dict, symbol_context, lang: str,
                         operation_label: "str | None" = None) -> None:
    """Stop an operation whose operand declares itself non-computable.

    Anything shown in the timeline looks computable, so an object that is
    not must say so **where the researcher tries to use it** — otherwise the
    failure surfaces much later as an opaque parser complaint (an anonymous
    coupled state's bra-ket latex reaching the CAS produced
    ``unsupported_notation.pipe``, which explains nothing and suggests
    nothing).

    This enforces the ``angular_descriptor["cas_usable"]`` contract.  No
    F-9 operation currently produces a False — coupled states are always
    registered with named basis symbols (auto-generated when the researcher
    does not choose them), precisely so that everything displayed IS
    computable — and this guard is what keeps that an invariant rather than
    a convention.
    """
    for value in args.values():
        if not isinstance(value, str) or not _EXPR_ID_RE.match(value):
            continue
        expr = expressions.get(value)
        descriptor = getattr(expr, "angular_descriptor", None) if expr else None
        if descriptor and not descriptor.get("cas_usable", True):
            raise ValueError(t("angular.display_only_state", lang,
                               expr_id=value, operation=operation))
        # A range-bearing definite integral is display-only by design: the
        # CAS is deliberately not invoked at input (INTEGRAL-COV), so its
        # canonical is LaTeX, not a Maxima expression.  Its own operations
        # (`change_variables_integral_form`, `evaluate_integral_form`) are
        # engine methods and never arrive here, so anything that DOES reach
        # this point is a generic operation being handed a picture.
        # Measured 2026-08-03: `changevar` on one came back as the literal
        # `_res_` — the adapter's own scratch name — and was registered as a
        # successful expression.
        if expr is not None and getattr(expr, "integral_descriptor", None):
            raise ValueError(t("integral.display_only_form", lang,
                               expr_id=value, operation=operation))
        # A PAIR REPRESENTATION (`\hat c_n(t)`) is display-only in the same
        # sense: its canonical is `e^{-i\varepsilon t} c_n`, which is a
        # coefficient the CAS cannot hold beside an operator it can (docs/33
        # §6.1).  Its own operation is `pair_bracket`, which passes the
        # OPERATOR parts and never arrives here.  Measured 2026-08-22, a
        # generic bracket handed one died with the CAS-internal
        # "Bracket expressions in Julia must have exactly 2 arguments" —
        # a message about the backend's parser, not about what was asked.
        if expr is not None and getattr(expr, "pair_representation", None):
            raise ValueError(t("pair_rep.display_only_pair", lang,
                               expr_id=value, operation=operation))
        # A symbolic BOND SUM is display-only for the same reason: its
        # canonical is the LaTeX it was written in, and the constraint that
        # makes it what it is (⟨ij⟩, i<j, i≠j) lives in the descriptor, not in
        # the text.  A generic CAS operation re-reads the text and loses it —
        # measured 2026-08-10, `simplify` on `\sum_{\langle ij\rangle} V_{i,j}`
        # returned `-(V(i,j)*nbSumLo1) + V(i,j)*nbSumHi1 + V(i,j)`: Maxima had
        # summed a CONSTANT over the placeholder range, and the answer came back
        # as a success.  `\sum_{i \ne j}` lost its constraint and became an
        # ordinary sum over `i` with `j` left free — the same shape as the
        # original, which is what makes it dangerous.
        #
        # Its own operations (expansion to N sites, dispersion, substitution
        # into the body) are engine methods that never arrive here.
        if expr is not None and getattr(expr, "lattice_descriptor", None):
            raise ValueError(t("lattice.display_only_bond_sum", lang,
                               expr_id=value, operation=operation))
        # An extremum over a bound index is display-only for a THIRD reason:
        # not deferred evaluation but no backend form at all — Maxima's `lmax`
        # needs a concrete list, so a symbolic range has no command (measured
        # 2026-08-16).  Read as an ordinary subscript it lowered to
        # `max(i) * x(i)`, and `max` of one argument IS that argument, so
        # `\max_i x_i` computed as `i·x(i)`: valid CAS text, a different
        # quantity, silent.  Refusing here rather than at registration is the
        # whole point — the notation stays writable.
        if expr is not None and getattr(expr, "extremum_descriptor", None):
            raise ValueError(t("extremum.display_only_form", lang,
                               expr_id=value, operation=operation))
        # H-6: `div` is defined on a vector and `grad` on a scalar; the other
        # combinations have no answer.  Measured 2026-08-06, they came back
        # as successes anyway — the divergence of the scalar `x^2+y^2`
        # returned `y^2+x^2` unchanged, and the gradient of a vector field
        # returned `[0=0, 0=0, 0=0]`.  No error, so the researcher moves on
        # believing a divergence was taken.  Refused only when the type is
        # CERTAIN (see vector_op_types — two vectors in one expression stay
        # unknown and are still allowed through).
        if expr is not None:
            _mm = operand_mismatch(
                operation, expr.canonical or expr.latex or "",
                symbol_context,
                value_lookup(symbol_context,
                             lambda e: getattr(expressions.get(e), "canonical", None)))
            if _mm is not None:
                # **断り文は研究者の語で書く**（Core Principle #1）。以前は
                # `{actual}` と `{operation}` に内部名を素で入れており、日本語では
                # 「E1 は scalar なので、divergence の対象にできません」と読めた。
                # この断りは 2026-09-17 まで**未宣言の記号を含む式には届いていな
                # かった**（型を判定しなかった）ので目に触れにくかったが、素の式にも
                # 届くようになった以上、内部名のままにはできない。語はどちらも既に
                # 在る——型は `declare.algebra_*`、操作名は呼び出し側の
                # `_operation_label`（隣の guard には渡っていて、ここには渡って
                # いなかった）。
                raise ValueError(t("vector_op.operand_type", lang,
                                   expr_id=value,
                                   operation=operation_label or operation,
                                   required=t(f"declare.algebra_{_mm[0]}", lang),
                                   actual=t(f"declare.algebra_{_mm[1]}", lang)))


_NUMBER = re.compile(r"^[-+]?[\d./]+$")


def _own_word_kind(piece: str, *, validator, lang_free: bool = True) -> "str | None":
    """その綴りは**システム自身の語**か。そうならその種類（i18n キーの一部）。

    見るのは4つ、どれもこの系が既に1箇所に持っている語彙である:
    文法語・並びの区切り・計算/導出の動詞・操作の別名。
    """
    from notebook.runtime.nl_patterns import fragment, nlp
    p = (piece or "").strip()
    if not p:
        return None
    if nlp("operand_stopwords", "en").match(p):
        return "grammar"
    for name in ("list_sep_ja", "list_sep_en"):
        if re.fullmatch(fragment(name).strip(), p, re.I):
            return "separator"
    for name in ("compute_verb_ja", "compute_verb_en",
                 "derive_verb_ja", "derive_verb_en"):
        if re.fullmatch(fragment(name), p, re.I):
            return "verb"
    if validator is not None and validator.operations_for_alias(p.lower()):
        return "operation_name"
    return None


def word_operand_warnings(operation: str, args: dict, *, validator,
                          lang: str) -> "list[str]":
    r"""②の共通の仕組み —— **オペランドは、システム自身の語ではない**。

    ②の本文は「操作を名指した文が、別の操作で答えられてはならない」だったが、
    2026-09-05 に実在1046綴りで測ると**そのままでは不変条件にならない**: 素朴に
    書くと 88/381 が誤検知で（`Define X as a scalar` は `define` を名指して
    `define_scalar_parameter` が答える——違反ではなく**特殊化**）、しかもこの日に
    見つけた欠陥5件のうち**捕まえるのは1件だけ**だった。

    5件に共通していたのは別のことで、**研究者が書いた語がオペランドの欄に入って
    いた**——そしてその語はいずれも**この系自身の語彙**だった:

        integrate E1 over x         → var='over'         `over` は文法語
        derive the inverse of E1    → var='inverse'      `inverse` は操作の別名
        … を x および y について解いて   → vars='x, および, y'  `および` は並びの区切り
        substitute x = 2 into E1    → `into` が CAS の関数に  `into` は文法語

    研究者の記号がこれらと一致することは、まず無い。実測: 3つの欠陥形をすべて
    捕まえ、正しいオペランド（`x` `Q` `S` `theta` `3x3` `foo` `dBdt` `mz` `T`）
    すべてで黙り、**コーパスの操作530行で発火0**。

    **測って捨てた候補が2つある**（どちらも雑音が多すぎた）:

    - 「答えた操作 ≠ 名指された操作」…… 88/381 が誤検知（特殊化を違反と数える）。
    - 「オペランドはセッションが既知の記号である」…… 実セッションの再生で34件。
      大文字1文字の変数（`Q` `S` `T` `X`）や `3x3` に鳴る。既存の未定義記号検出が
      **式について**同じ判定をしており、そちらを操作の応答に配線するほうが筋
      （下の「残っていること」）。

    **この判定が書けるのは、①で語彙を1箇所ずつに畳んだからである。** 2026-09-05
    の朝の時点では文法語は `operand_stopwords` と `_JP_PARTICLES_STRIP` に、区切りは
    6箇所に、計算動詞は ja 9・en 8箇所に散っていて、**1つの検査がそれらに訊く
    ことができなかった**。

    **警告であって拒否ではない。** `over` という名前の変数を研究者が本当に使う
    可能性は否定できない（`bound_name_collision` と同じ判断——判別できないなら
    事実だけ伝えて研究者に残す）。
    """
    out: list[str] = []
    for field, value in (args or {}).items():
        if field.startswith("_") or field == "registered_names" or value is None:
            continue
        for piece in re.split(r"\s*,\s*", str(value)):
            piece = piece.strip()
            # 記号を含む綴り（`a*x+b`, `x^2`, `\lambda`）を先に除く早道は**置いて
            # いない**。実測 2026-09-05、それは `w.r.t.` と `&` を黙らせる——
            # どちらもこの系自身の語で、鳴らすべきものだった。コーパスでは記号を
            # 含む値がシステムの語と一致した例は0件なので、速さのための除外に
            # 正しさを損なう価値は無い。
            if not piece or _NUMBER.match(piece) or _EXPR_ID_RE.match(piece):
                continue
            kind = _own_word_kind(piece, validator=validator)
            if kind is None:
                continue
            out.append(t(f"operand.own_word.{kind}", lang,
                         field=field, value=piece, operation=operation))
    return out
