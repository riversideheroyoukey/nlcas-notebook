"""オペランドの**境界**の規則を、一箇所に置く。

「どこでオペランドが終わるか」という一つの問いに、この系は三箇所で別々に答えて
いた——どれも空白とカンマによる近似だったので、凝縮系でいちばん基本的な綴りが
計算に届かなかった（実測 2026-08-22）:

    c(1,u) と c'(1,u) の反交換子   → b="c'(1"            生の ParseError
    [c(1,u), c'(1,u)]              → a="c(1"             黙って別の場所で切る
    n_{1↑} と n_{1↓} の交換子      → [c†_{1↑}, c_{1↑}]   別の演算子を成功として返す

境界は**括弧の深さ**（カンマ）と**助詞・キーワード**（と／の、of／and）である。
どちらも研究者が書いたものであって、空白ではない。

判別材料が増える見込みの族なので、規則の側に置く（`subscript_policy` と同じ理由
——SUBSCRIPT-POLICY-SPLIT）。次の材料もここへ足すこと。
"""
import re

from notebook.runtime.nl_patterns import fragment
from notebook.runtime.shared_parser import find_top_level_relation

#: 区切り文字（空白・カンマ・読点・句点・全角空白）。**括弧の外にあるときだけ**
#: 区切りとして働く。
_TOKEN_SEP_RE = re.compile(r"[\s,、。　]")


def _split_outside_brackets(text: str) -> "list[str]":
    r"""区切りで分けるが、``(…)`` / ``[…]`` / ``{…}`` の中では分けない。

    `c'(1,u)` は1つのトークンである。深さを数えるだけで、開き括弧の種類は問わない
    ——閉じ過ぎ（`)` が余る）でも負にはしない: 研究者の書き損じで文全体が1トークンに
    潰れるより、そこだけ区切れるほうが害が小さい。
    """
    out, buf, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and _TOKEN_SEP_RE.match(ch):
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return out


# Bracket notation for commutator/anticommutator:
#   [X, Y]  →  commutator(a=X, b=Y)
#   {X, Y}  →  anticommutator(a=X, b=Y)
#
# **どのカンマが区切りかは括弧の深さで決まる。**  これを正規表現の文字クラス
# （第1オペランドから `,` を除く）で近似していたため、凝縮系でいちばん基本的な
# 綴りである多添字演算子が壊れていた——しかも**エラーではなく、黙って違う場所で**:
#
#     [c(1,u), c'(1,u)]  →  a="c(1",  b="u), c'(1,u)"
#
# `_tokenize` の分割と同じ規則で、同じ理由（実測 2026-08-22）。
def split_leading_compute_verb(text: str) -> "tuple[str, str]":
    r"""先頭の計算動詞を `(飲んだぶん, 残り)` に分ける。無ければ `("", 文)`。

    **英語は動詞が前に来る。** `compute [A, B]` / `evaluate {A, B}` は日本語の
    `[A, B] を計算して` と同じ要求だが、先頭が括弧でないので括弧の読みがどれも
    当たらなかった（実測 2026-09-06）。語は共有一覧（`compute_verb_en`）——
    「計算せよという要求か」は十五問目が1箇所に畳んだ問いである。

    **飲んだぶんを返すのは、文を組み直す側がいるから**（2026-09-11、
    MATRIX-INLINE-OPERAND-EN）。`_en_verb_formula_re` は文頭に操作の別名を
    要求するので、`compute the determinant of [4, 0; 0, 1]` は当たらず、
    その場に書いた行列が**登録されないまま**「式を名指していない文」に見えて
    直前の式に落ちていた——実測、別の行列の行列式が `completed` で返る。
    動詞を飲んで判定し、**組み直すときは飲んだぶんを戻す**——落としても答えは
    同じだが（実測19綴り）、`matrix_trigger` が計算動詞を要求するので**答える
    規則が変わる**（行列のインターセプタが降り、別名走査が答える）。
    """
    import re as _re
    from notebook.runtime.nl_patterns import fragment
    stripped = (text or "").lstrip()
    m = _re.match(r"(?i)^(?:please\s+)?(?:" + fragment("compute_verb_en")
                  + r"|take)\s+(?:the\s+)?", stripped)
    if not m:
        return "", stripped
    return stripped[:m.end()], stripped[m.end():].lstrip()


def skip_leading_compute_verb(text: str) -> "str | None":
    r"""先頭の計算動詞を飛ばして**括弧から始まる残り**を返す。無ければ None。

    **1つの関数にしてある**のは、これを訊く場所が2つあるから
    （`starts_a_bracket_operand_form` と `_bracket_operands`）——同じ問いを
    2度書けば、次に語を足したとき片方だけが知ることになる。
    """
    stripped = (text or "").lstrip()
    if stripped.startswith(("[", "{", "\\{")):
        return stripped
    prefix, rest = split_leading_compute_verb(stripped)
    if not prefix:
        return None
    return rest if rest.startswith(("[", "{", "\\{")) else None


def starts_a_bracket_operand_form(text: str) -> bool:
    r"""Does *text* begin a two-operand bracket — `[X, Y]`, `{X, Y}`, `\{X, Y\}`?

    Three places asked this with `startswith(("[", "{"))` — the same question
    spelled three times — and all three missed the **LaTeX** brace: measured
    2026-08-22, `\{c_{1\uparrow}, c^\dagger_{1\uparrow}\}`, which is what a math
    chip and the handwriting panel produce, was auto-registered as an ordinary
    expression and nothing was computed.
    """
    return skip_leading_compute_verb(text) is not None


def is_bracket_operation_text(text: str) -> bool:
    r"""`[A, B]` / `\{A, B\}` が**操作の要求**か（関係式の左辺ではないか）。

    **括弧に `=` が続けば、それは関係式である。** `\{A,B\} = δ` は
    「これが成り立つ」と述べた式で、`\{A,B\}` は「求めよ」という要求である
    ——研究者が書いたものがそのまま判別材料になる。

    実測 2026-08-22（実例の再生成が捕まえた、私自身の退行）: `\{X, Y\}` を
    括弧形と認める修正を入れたとき、この区別を入れ忘れたため、場の演算子の
    反交換関係の行

        \{\hat\psi_\sigma(r), \hat\psi^\dagger_{\sigma'}(r')\} = \delta_{\sigma\sigma'}\delta(r-r')

    が「オペランドは揃っている」と判定されて式として登録されなくなり、
    グリーン関数のシナリオが**そこで止まった**。しかも再生成はその時点で
    中断するので、それ以降のシナリオは**古いまま公開され続けていた**。
    """
    return (starts_a_bracket_operand_form(text)
            and find_top_level_relation(text or "") is None)


def _bracket_operands(text: str, opener: str) -> "tuple[str, str, str] | None":
    r"""``[X, Y]`` / ``{X, Y}`` を先頭から読み、``(X, Y, 一致した全体)`` を返す。

    区切りは**深さ0のカンマ**——`(`/`[`/`{` の中のカンマは operand の一部である。
    カンマが複数あるときは最初のもので切る（第2オペランドは残り全部）。これは
    従来の正規表現の振る舞いをそのまま保った。
    """
    close = {"[": "]", "{": "}"}[opener]
    # **`\{A, B\}` is the same anticommutator.**  It is how the brace is written
    # in LaTeX — and it is what a math chip and the handwriting panel produce —
    # so the escaped spelling registered as an ordinary EXPRESSION and nothing
    # was computed (measured 2026-08-22).  The backslash before a brace is
    # syntax, not content; removing it changes no operand.
    if "\\{" in text or "\\}" in text:
        text = text.replace("\\{", "{").replace("\\}", "}")
    # 先頭の計算動詞を飛ばす（`compute [A, B]`）。同じ問いを2度書かないため
    # `skip_leading_compute_verb` に訊く。
    _skipped = skip_leading_compute_verb(text)
    if _skipped is not None:
        text = _skipped
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != opener:
        return None
    start = i
    depth, comma = 0, -1
    for j in range(i, len(text)):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                if comma < 0 or ch != close:
                    return None
                return (text[start + 1:comma], text[comma + 1:j], text[start:j + 1])
        elif ch == "," and depth == 1 and comma < 0:
            comma = j
    return None


#: Which operation each spelling of a BINARY OPERATION asks for — brackets and
#: the two vector products alike (renamed from ``BINARY_KIND_OPS`` when the
#: products joined it: a table named for brackets that holds a dot product is a
#: name that lies, and a lying name invites the next copy).  A commutator has
#: two candidates for the same reason `_try_bracket_parse` does: the Poisson
#: bracket is its classical analogue, they share the `a`/`b` fields, and
#: they are mutually exclusive by backend.
#: 括弧の**種類の語** → 操作の候補（バックエンドが選ぶ）。
#:
#: **語彙を足しても、この表が知らなければ読みは黙って降りる**——2026-08-27 の
#: ∇族の註と同じ形で、実際に踏んだ（2026-09-06）: `binary_bracket_phrase` の
#: `kind` に「ポアソン括弧」を入れてもここに無く、`A と B のポアソン括弧をとって`
#: は `nlp` に当たりながら `None` を返していた。**表と語彙は同じ問いに答えて
#: いる**ので、綴りの違い（`ポアソン`/`ポワソン`/`ポアソンの括弧`）は語彙側
#: （`ja_binary_operation_kind`）に、対応はここに、と分けたうえで
#: `_normalise_binary_kind` が両者を繋ぐ。
BINARY_KIND_OPS: "dict[str, tuple[str, ...]]" = {
    "交換子":     ("commutator", "poisson_bracket"),
    "交換関係":   ("commutator", "poisson_bracket"),
    "commutator": ("commutator", "poisson_bracket"),
    "反交換子":   ("anticommutator",),
    "反交換関係": ("anticommutator",),
    "anticommutator": ("anticommutator",),
    "ポアソン括弧": ("poisson_bracket",),
    "poisson bracket": ("poisson_bracket",),
    # 内積・外積も**同じ文の形・同じ `a`/`b` 欄**である（`A と B の内積` /
    # `the dot product of A and B`）。この表に無かったので位置で欄を埋める経路に
    # 落ち、**空白1つで1オペランド**に切られていた——`\nabla \times \vec{A} と
    # \vec{B} の内積` が `a="\nabla"`, `b="\times"` になり `nabla_x*times_x+…` を
    # `completed` で返していた（実測 2026-09-17、VEC-BINARY-OPERAND-WHITESPACE）。
    # 交換子が 2026-08-22 に踏んだ欠陥そのもので、答えはここに既に在った。
    "内積": ("dot_product",),
    "dot product": ("dot_product",),
    "外積": ("cross_product",),
    "cross product": ("cross_product",),
}

#: 綴りの揺れを表の鍵に寄せる。`ポワソン` / `ポアソンの括弧` は同じものである。
_BINARY_KIND_ALIASES = (
    ("ポワソン", "ポアソン"),
    ("ポアソンの括弧", "ポアソン括弧"),
)


def _normalise_binary_kind(kind: str) -> str:
    out = " ".join((kind or "").strip().lower().split())
    for src, dst in _BINARY_KIND_ALIASES:
        out = out.replace(src, dst)
    return out

def read_binary_operation_phrase(text: str, lang: str, supports) -> "tuple[str, str, str] | None":
    r"""Read `A と B の交換子` / `commutator of A and B` with SPAN operands.

    The operand boundary is the particle (と / の) or the keyword
    (of / and) — what the researcher actually wrote — not whitespace.
    Falling through to the positional field mapper meant one whitespace-
    separated TOKEN per operand, so a PRODUCT of operators was silently
    truncated (measured 2026-08-22):

        c^\dagger_{1\uparrow} c_{1\uparrow} と c^\dagger_{1\downarrow} c_{1\downarrow} の交換子
            →  commutator(a="c^\dagger_{1\uparrow}", b="c_{1\uparrow}")

    — a DIFFERENT pair of operators, answered as a success.  A product of
    creation and annihilation operators is the most ordinary operand there
    is in this algebra; without it the operator side cannot be driven from
    natural language at all.

    *supports* answers "is this operation available on this backend?"  That is
    the caller's knowledge (which backend the session runs on), so it is passed
    in rather than looked up here.

    Runs after `extra_pre_phase1` and the E-id-only interceptors, so every
    existing, more specific reading keeps its priority.
    """
    from notebook.runtime.nl_patterns import nlp, fragment
    m = nlp("binary_operation_phrase", lang).search(text)
    if not m:
        return None
    kind = _normalise_binary_kind(m.group("kind") or "")
    a, b = m.group("a").strip(), m.group("b").strip()
    if not a or not b:
        return None
    for operation in BINARY_KIND_OPS.get(kind, ()):
        if supports(operation):
            if operation in _VECTOR_PRODUCT_OPS:
                from notebook.runtime.vector_operand import vector_operation_operand
                a, b = vector_operation_operand(a), vector_operation_operand(b)
            return (operation, a, b)
    return None


#: The products whose operands are VECTORS, so an operand written as a ∇ operation
#: (`\nabla \times \vec{A}`) is that operation's result — `vector_operand`.
#: A commutator's operand is never read this way.
_VECTOR_PRODUCT_OPS = frozenset({"dot_product", "cross_product"})


# ---------------------------------------------------------------------------
# 先頭の式は「操作の対象」か、それとも「名前で渡すオペランド」か
# ---------------------------------------------------------------------------
#
# これも「オペランドの境界」の一問である——どこで**終わる**かではなく、切り出した
# ものが**何になる**か。日本語と英語が別々に答えていて、答えが違っていた
# （実測 2026-08-23、指示コーパス）:
#
#     [ja] a(1) の Heisenberg 方程式 E1        → E2='a(1)' と E3=結果 を登録
#     [en] Heisenberg equation of E1 for a(1) → E2=結果 だけを登録
#
# 日本語の `_FORMULA_OP_RE` は字種の変わり目で切って**操作を問わず登録**し、英語の
# `_en_verb_formula_re` は**要求フィールドが `expr` 1つの操作にしか**発火しない。
# 結果、同じ手順の日英シナリオで式番号がずれ、48本中3本では E 番号が丸ごと別物に
# なっていた（`bcs_gap_equation` は14個中12個）。英語のシナリオ文書が日本語と違う
# 番号で書かれていたのはそのためで、**言語をまたいで実例を読む研究者は別の式を指す。**
#
# 判別は**スキーマのデータ**である。`expr` は「この操作が作用する式」で、
# `operator` / `hamiltonian` / `a` / `b` は**それ自体が名前を持つもの**——研究者が
# 既に宣言した演算子に、もう一つ別の同一性（E番号）を与える必要はない。
# 括弧経路（`[A, B]`）が前からそう振る舞っていて、日英で番号が一致する唯一の
# 経路だった。
#
# **`EXPR_FIELDS` では分けられない**（実測）: `operator` も `hamiltonian` もその集合の
# 一員なので、`heisenberg_equation` は「式クラスのフィールドを2つ持つ」と答えて
# しまい、`differentiate`（`['expr','var']`）と区別が付かない。分けるのは
# **`expr` という名前そのもの**である。
#: 宣言だけで**それ自体が同一性を持つ**対象。`formula_is_an_operation_operand` の
#: 註が既に言っている原則——`operator` / `a` / `b` のような欄は「それ自体が名前を
#: 持つもの」を取るので E 番号を重ねる必要がない——の、**宣言側の一覧**である。
#:
#: **`kind` だけでは足りない。** 実測 2026-09-11、宣言はこう記録される:
#:
#:     行列       kind='scalar'        algebra='matrix'
#:     テンソル    kind='scalar'        algebra='tensor'
#:     ベクトル    kind='scalar'        algebra='vector'
#:     ベクトル場  kind='vector_field'  algebra='scalar'
#:     スカラー場  kind='scalar_field'  algebra='scalar'
#:     スカラー    kind='scalar'        algebra='scalar'   ← 対象ではない
#:
#: つまり**行列・テンソル・ベクトルは `kind` ではスカラーと区別できない**——
#: 判別材料は `math_type.algebra` の側にある。`kind` だけで書くと、研究者が
#: `G を行列とする` と宣言した G が漏れる。
_DECLARED_OBJECT_KINDS: frozenset[str] = frozenset({
    "vector_field", "scalar_field", "operator",
})

_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _is_declared_object(quantity) -> bool:
    """その量は「それ自体が同一性を持つ対象」か（変数ではなく）。"""
    if getattr(quantity, "kind", None) in _DECLARED_OBJECT_KINDS:
        return True
    algebra = getattr(getattr(quantity, "math_type", None), "algebra", None)
    return bool(algebra) and algebra != "scalar"


def declared_object_names(context) -> "set[str]":
    r"""このセッションが**宣言した対象**の名前（変数ではないもの）。

    「その名前はそれ自体が同一性を持つ対象か」を訊く場所が2つある——足場を
    止めるかの判定（`text_names_a_declared_object`）と、**欄を埋めるとき
    その名前を式として受けるか**（`token_field_mapping`）——ので、集合の作り方は
    ここに1つ置く。判別子は `_is_declared_object`（`kind` では足りない理由も
    そこに書いてある）。
    """
    if context is None:
        return set()
    return {name for name, prop in getattr(context, "registry", {}).items()
            if _is_declared_object(getattr(prop, "quantity", None))}


def text_names_a_declared_object(context, text: str, consumed=()) -> bool:
    r"""*text* が、このセッションが**宣言した対象**を名指しているか。

    足場（`interpret_pipeline` の step 2d）が直前の式を前置してよいかを決める。
    宣言には `defining_expr_id` が無いので 2a-sym は名前を式番号へ書き換えられず、
    その文は「式を名指していない」ように見える——そこで直前の式を前置すると、
    **研究者が名指した対象とは別の式の答えが `completed` で返る**。実測
    2026-09-11、`A のノルムを求めて` が直前に登録した `u` のノルム（5）を返し、
    直前が別のベクトルなら 10 を返した——**答えが「直前に何をしたか」で変わる**。

    **スカラーは数えない。** `x` は変数として文中に立つので、数えると
    `x で微分して` が「対象を名指した文」に見え、足場が止まって対象を失う。

    **実在の指示は1行も変わらない**——コーパス1102行を足場ありとなしで2回測って
    差分0（「文に書かれていない式番号がオペランドに入る」55行はすべて 2a-sym が
    名前を書き換えたもので、名前は式を持つので通り続ける）。

    **すでに別の役で読まれた名前は数えない**（*consumed*）。実測 2026-09-11、
    `S の各成分のハイゼンベルク方程式を求めて` の `S` は**対象ではなく、どの
    演算子の成分を発展させるか**であり、対象はハミルトニアンのほうである
    ——数えると、いちばん普通の綴りで workflow が黙って降りる。
    起票が「素朴に入れると壊れる」と書いていたのがこの形で、判別は
    「その名前が**オペランドの位置に**立っているか」——呼び出し側の文法は
    それを既に知っている（名前つきグループで読んでいる）ので、訊く。
    """
    names = declared_object_names(context) - {n for n in consumed if n}
    if not names:
        return False
    return any(tok in names for tok in _IDENT_RE.findall(text or ""))


#: 式参照の綴り——**一箇所に置く**。「直前の式に落ちてよいか」を訊く側が
#: それぞれ自前で綴っていた（`interpret_pipeline` に2回、`interpret_plan` に1回、
#: 行列単項はさらに緩い `E\d+`）。綴りが割れれば答えも割れる。
EXPR_ID_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9_])(E\d+)(?![A-Za-z0-9_])")


def may_fall_back_to_the_recent_expression(context, text: str, consumed=(),
                                           operand_names=()) -> bool:
    r"""「文が式を名指していないので**直前の式で代用する**」——してよいか。

    **呼ぶのは、文が名指したものを解決し終えた後**。してよいのは、文が本当に
    何も名指していないときだけである。

    この問いに、この系は**5箇所で別々に答えていた**（実測 2026-09-11）。
    どれも同じ形の誤りを返す——研究者が名指したものを、黙って別の式に
    すり替えて `completed` と言う:

    ① **宣言済みの対象を名指している**（`text_names_a_declared_object`）。
       宣言には `defining_expr_id` が無いので 2a-sym が式番号へ書き換えられず、
       その文は「式を名指していない」ように見える::

           G を行列とする（別に A = [2,1;1,2] が在る）
             G の行列式を求めて  →  completed = 3      ← **A の行列式**
             G のトレースを求めて →  completed          ← 直前の**結果**の跡

    ② **式番号を書いている**。ここへ来たということは、その番号が解決しな
       かったということなので、別の式で代用してはいけない::

           E99 の固有値と固有ベクトルを求めて  →  completed  ← **E1 の固有系**
           E99 を x で微分して（E1 を選択中）    →  completed = 2*x  ← **E1 の微分**

    **スカラーは数えない**（①の側の規則）。`x` は変数として文中に立つので、
    数えると `x で微分して` が「対象を名指した文」に見え、対象を失う。

    止めた先は操作の断り（「必要な引数が不足しています」）である。**能力には
    ならないが、沈黙した誤りは消える**——名前をオペランドとして通す半分は
    式クラスの欄に触るので別（open_items OPERAND-BOUNDARY 残①）。

    ③ **解決しない素の名前がオペランドの位置に立っている**（*operand_names*）。
       `zvar を簡約して` の `zvar` は宣言もされておらず値も持たないので、
       2a-sym は式番号へ書き換えられず、①と同じ形で足場が直前の式を前置する::

           y = x^2（直前の式）
             zvar を簡約して / simplify zvar   →  completed = x^2   ← **E1 の簡約**
             zvar を x で全微分して             →  differentiate(expr=E1, var='zvar')
                                                  →  completed = 0
                                                  ← 対象も変数も**両方**すり替わる

       実測 2026-09-18、3名前 × 6操作 × 日英の **36セル中30**。微分と積分では
       捨てられた名前が今度は**変数の欄**に入り、研究者が書いた `x` のほうが
       落ちる——研究者自身の記号で組み立てた、もっともらしい誤答である。

    **素の名前は「数えれば済む」ものではない**（①のスカラー除外と同じ罠の別の顔）。
    `x で微分して` の `x` も未定義の素の名前だが、そこでは**変数**として立って
    いるので、数えると対象を失う。英語ではさらに、操作の語そのもの（`simplify`）が
    識別子に見える。だから判別は**位置**で、位置を知っているのは呼び出し側の文法
    である——`operand_names` は「文法がオペランドの位置で読んだ名前」で、
    *consumed*（別の役で読んだ名前）のちょうど鏡である。

    *consumed* は、呼び出し側の文法が**すでに別の役で読んだ名前**（例:
    `S の各成分のハイゼンベルク方程式` の `S`）。数えると、対象を名指して
    いないのに名指したことになる。
    """
    if EXPR_ID_IN_TEXT_RE.search(text or ""):
        return False
    for name in operand_names or ():
        if name and name not in (consumed or ()) and not name_is_usable_as_an_operand(
                context, name):
            return False
    return not text_names_a_declared_object(context, text, consumed)


def name_is_usable_as_an_operand(context, name: str) -> bool:
    """その名前は、オペランドとして**渡せる**か。

    渡せるのは2つだけ:

    * **値を持つ名前**（H-1 の `defining_expr_id`）——2a-sym が式番号へ書き換える。
    * **宣言された対象**——式クラスの欄が名前のまま受ける（`declared_object_names`）。

    **「解決するか」では足りない**（実測 2026-09-18）。宣言済みの**スカラー**は
    解決するが、そのどちらでもない——欄はスカラーを受けず（`x` は変数として文中に
    立つので数えられない）、2a-sym も書き換えられない。だから `g をスカラーとする`
    のあとの `g を簡約して` は**直前の式の簡約**を `completed` で返していた
    （4操作 × 日英で **8/8**）。位置が分かっていれば種別で分ける必要はない——
    **オペランドの位置に立った名前は、渡せるか渡せないか**である。
    """
    if context is None:
        return False
    qty = context.get_quantity(name)
    if qty is None:
        return False
    if getattr(qty, "defining_expr_id", None):
        return True
    return _is_declared_object(qty)


def formula_is_an_operation_operand(rest_text: str, resolve_alias, max_alias_words,
                                    get_spec, supports, tokenize) -> bool:
    r"""残りの文が名指した操作は、先頭の式を `expr` として取らないか。

    True なら、先頭の式は**名前で渡すオペランド**であって式として登録しない。

    **操作が1つも名指されていなければ False**（＝いまの振る舞いのまま）。ここは
    「式＋認識できない指示」の場合で、登録をやめると素の式入力が書けなくなる。
    同じ理由で、名指された操作の**どれかが** `expr` を取るなら False にする——
    この規則が振る舞いを変えるのは「操作は名指されていて、そのどれも `expr` を
    取らない」ときだけである。

    引数で受けるのは、どれも呼び出し側の知識だから（どのバックエンドか、どの
    トークナイザか）——`read_binary_operation_phrase` の *supports* と同じ理由。
    """
    named = False
    for start, end, canonical in scan_or_empty(rest_text, resolve_alias,
                                               max_alias_words, tokenize):
        spec = get_spec(canonical)
        if spec is None or not supports(canonical, spec):
            continue
        named = True
        if "expr" in (spec.get("required_fields") or []):
            return False
    return named


def scan_or_empty(text: str, resolve_alias, max_alias_words, tokenize):
    """`scan_aliases` を呼ぶだけ——**呼び出し側と同じトークナイザで**。

    別のトークナイザで数えると、ここが「操作は名指されていない」と答え、供給側は
    名指されたものとして処理する、という食い違いが起こる。それが直そうとしている
    欠陥そのものの形なので、トークナイザは渡してもらう。
    """
    from notebook.runtime.token_field_mapping import scan_aliases
    return scan_aliases(tokenize(text), resolve_alias, max_alias_words)


#: 属格の頭——`<語> of <数式>`。英語だけの形である（日本語は `の` で切る）。
_GENITIVE_HEAD_RE = re.compile(r"(?i)^\s*(?P<noun>[A-Za-z][A-Za-z-]{2,})\s+of\s+\S")


def genitive_head_not_an_operation(formula: str, names_an_operation) -> "str | None":
    r"""`<語> of <数式>` の**語**——ただしその語が操作を名指していないときだけ。

    英語の「動詞＋式」の規則が切り出した「式」が、実は `<名詞> of <数式>` の
    ことがある。`of` は**2文字**なので `_nl_word_count`（3文字以上を数える）に
    見えず、`widget of x^2 - 1` は「英単語1語＋数式」として通っていた——実測
    2026-09-24、`expand the widget of x^2 - 1` が `of*widget*(x^2-1)` を
    **1つの式として登録**して `completed`（`2*into(a*x+b)` と同じ形）。
    語数で降りる綴り（`Expand the nearest-neighbor of x^2 - 1`）は別名走査に落ち、
    そちらは `expr='nearest-neighbor'` を作っていた——**根は2つ、形は1つ**。

    **操作の別名は除く。** `the curl of the curl of A` の内側や
    `expand the product of E1 and E2` は実在の綴りで、境界は**操作の別名そのもの**
    だという既存の規則がそのまま効く（コーパス611行で、この述語が発火するのは
    **0 行**——形に届く1行は `curl` なので除かれる、実測）。

    **等式の「辺」は除く。** `simplify the RHS of E1` の `RHS` は操作でも量でもなく、
    系が既に読んでいるもの（`en_side_phrase`——H-5/H-7 の族）である。除くのは語彙を
    手で並べたいからではなく**順序**が理由で、実測、この形に当たる読みのうち
    等式の族**だけ**がこの割り込み口より後ろで走る（変数変換・境界条件などは前で
    claim される）。**コーパスではなく綴り空間で測って出た**——`RHS` はテストの
    リテラルにしか無く、コーパス差分 0 は何も言っていなかった。

    **3文字以上**——`_en_verb_formula_re` が別名に課しているのと同じ理由で、短い綴りは
    英文の中で偶然当たる。加えて `f of x` の `f` は**研究者の関数**であって綴り間違いの
    操作名ではないので、「`f` という操作を知りません」は偽を述べることになる。変異試験
    では1文字まで許しても**どのセルも落ちない**（どちらも断りになるため）ので、これは
    測って残した決定であり、`TestAShortHeadIsASymbol` が固定する。
    """
    m = _GENITIVE_HEAD_RE.match(formula or "")
    if m is None:
        return None
    if re.match(fragment("en_side_phrase") + r"\b", formula or "", re.IGNORECASE):
        return None
    noun = m.group("noun")
    return None if names_an_operation(noun) else noun


def text_names_an_operation(text: str, resolve_alias, max_alias_words,
                            get_spec, tokenize) -> bool:
    r"""この文は**操作を名指している**か —— 「指示か式か」の2つめの証拠。

    英語側の答えは長く `instruction_marker[en]` の**手書きの動詞一覧**だけ
    だった。だから操作を名指した英文が、動詞がその一覧に無いというだけで
    **式として登録**されていた（実測 2026-09-10）:

        relabel k to -k in E1        → `relabel*k*to -k*in*(\sum_k …)` を登録
        the residue of E1 at z = 1   → `the*residue*of (1/(z-1))*at*z = 1` を登録
        wick contract E1 + E2        → `wick*contract (matrix(…)) + (matrix(…))`

    どれも `registered` が返る——研究者には成功に見える。答えはスキーマの
    データに在り、`formula_is_an_operation_operand` が同じ走査で既に
    「残りの文が名指した操作は何か」を答えている。

    **手書きの一覧は消さない。** 実測、非 CJK の実在入力480行のうち **37行**は
    一覧だけが「指示」と言っている（`Find …` / `Save the lattice figure` /
    `subtract E12 from E11`）——それらの動詞は操作の別名ではない。二つは違う
    問いに答えていて、どちらも「指示である」ことの証拠になる:

        スキーマの別名 … この文は操作を名指しているか
        動詞の一覧 ……… この文は命令形か

    **バックエンドが対応しているかは訊かない**（`formula_is_an_operation_operand`
    とはそこが違う）。「これは指示か」は**そのバックエンドが実行できるか**とは
    別の問いで、混ぜると Maxima セッションの `wick contract E1 + E2` が
    「指示ではない」と判定され、**研究者の言葉でできた式が登録される**
    ——実測 2026-09-10。持っていない操作を名指した文の正しい答えは、
    誤った登録ではなく**断り**である。

    引数で受けるのは、どれも呼び出し側の知識だから（どのトークナイザか）
    ——`formula_is_an_operation_operand` と同じ理由。
    """
    for start, end, canonical in scan_or_empty(text, resolve_alias,
                                               max_alias_words, tokenize):
        if get_spec(canonical) is not None:
            return True
    return False


def split_name_list(text: str, lang: str = "ja") -> "list[str]":
    r"""並んだ名前を切る —— **1つの答え**（`_fragments.list_sep_{ja,en}`）。

    この問いは 2026-09-05 に数えると **6箇所以上**にあって、どれも違う部分集合を
    知っていた: `decl_stmt[ja]` は `, 、 と`、`decl_stmt[en]` は `,` だけ、
    `solve_system_vars[ja]` は全角 `，` も知り、`solve_system_vars[en]` は
    `and` と Oxford comma を知り、`_solve_vars_from` は `and` を**別の
    `re.sub` で先に消して**から `[,、と\s]+` で切っていた。

    知らない区切りに当たったときの振る舞いが、どれも**成功を返す**:

        a、b および c をスカラーとする   → `c` **だけ**登録され confirmed
        Define a, b and c as scalars  → `c` だけ
        Define a, b, and c as scalars → `c` だけ
        … を x および y について解いて     → `vars: 'x, および, y'`（助詞が変数に）

    宣言はこの系のほとんどの読みの判別材料なので（`^T` が転置か、添字が index か）、
    黙って落ちた名前は以降ずっと未宣言の自由変数として扱われる。

    **括弧の中は切らない。** この族の第一の規則で（`_split_outside_brackets`）、
    `c(k,u), c'(-k,d)` は2つの名前である——`[,、と\\s]+` で切っていた旧実装は
    `['c(k', 'u)', ...]` を返していた（実測、逆変換のコーパス行がこの綴り）。
    """
    # どちらの言語の文にも相手の区切りが混じりうる（`E1 と E2 for x and y`）ので
    # 両方で切る——guard を言語で絞らない、と同じ理由。
    both = re.compile(f"(?:{fragment('list_sep_ja')}|{fragment('list_sep_en')}|\\s+)",
                      re.I)
    out, buf, depth = [], [], 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0:
            m = both.match(text, i)
            if m and m.end() > i:
                out.append("".join(buf)); buf = []
                i = m.end()
                continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [v.strip() for v in out if v and v.strip()]
