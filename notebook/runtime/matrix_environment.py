r"""行列の TeX 環境を読む —— **区切り記号が量を決める**。

`\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}` は、研究者が最初に書く行列の綴りで
あり、**このシステム自身が表示に使っている綴り**でもある
（`BackendHandler.matrix_literal_to_pmatrix`、公開実例に28箇所）。それを打ち直せ
なかった（実測 2026-09-10、TEX-MATRIX-INPUT: 9綴り中1、通るのは `[2, 1; 1, 2]`
だけ）。CLAUDE.md が複素共役 `^{\ast}` について記録している
「**システム自身の表示が返す形は、打ち直せなければならない**」と同じ形である。

**壊れ方は「読めない」ではなく「黙って崩れる」だった**（実測）:

    \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
        → \beginpmatrix 2 & 1 1 & 2 \endpmatrix     ← 行区切り `\\` が消え、
                                                       2行が1行に融合している
    \pmatrix{2 & 1 \\ 1 & 2}
        → +atrix2 & 1 1 & 2                          ← `\pm` の置換が
                                                       **前方一致**で `\pmatrix` を食う

崩れたあと `&` がパーサに届き、「文字 `&` は数式として読めません。ワープロや PDF
から貼り付けた記号が混ざっていないか確認してください」という、**研究者が書いた
ものと無関係な断り**が返っていた。

**区切り記号は飾りではない。**

| 環境 | 何を書いたか | ここでの読み |
|---|---|---|
| `matrix` / `pmatrix` / `bmatrix` / `Bmatrix` / `smallmatrix` / `array` | 行列 | `matrix([…], […])` |
| `vmatrix`（縦棒） | **行列式**（スカラー） | `determinant(matrix(…))` |
| `Vmatrix`（二重縦棒） | **ノルム**（スカラー） | 読まず、**ノルムの理由で断る**（`notation.pipe_norm`——二重縦棒と同じ文言） |

一律に行列と読むと、**スカラーの量が 2×2 の数表として登録される**。
`vmatrix` を行列式と読むのは推測ではない——縦棒で囲むのは行列式の万国共通の
記法で、競合する読みが無い（`\sin^{-1}` を逆関数と読んだのと同じ判断）。

**落とす先は `[a, b; c, d]` と同じ節点**（`matrix(...)`）にしてある。別の節点を
作ると `get_algebra_type` ほか、MATMUL-TEMPLATE-STAR で数えた12箇所の消費者を
二重に相手取ることになる。`;` 区切りのリテラルではなく `matrix(...)` を**直接**
書くのは、**1行の行列**が `[a, b]` になってしまい `normalize_matrix_literal`
（`;` を探す）が素通りして、行列がリスト／交換子として登録されるため（実測）。

**推測はしない。** 行ごとに列数が違う（ragged）ものは、書き換えずにそのまま返す
——不揃いな行を持つ `matrix(...)` は CAS が拒否するだけで、こちらで揃え方を
決める根拠がどこにも無い。
"""
from __future__ import annotations

import re

#: 行列そのものを表す環境。
MATRIX_ENVIRONMENTS = frozenset({
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "smallmatrix", "array",
})
#: 縦棒＝行列式。**行列ではなくスカラー。**
DETERMINANT_ENVIRONMENTS = frozenset({"vmatrix"})
#: 二重縦棒＝ノルム。**行列でもなく、まだ操作も無い**——読まずに、
#: ノルムの断り（`notation.pipe_norm`）へ回す。断りの文言は二重縦棒の
#: 綴りと共通で、そちらが「ノルムか」を訊く1箇所である。
NORM_ENVIRONMENTS = frozenset({"Vmatrix"})

#: `\begin{env} … \end{env}`。入れ子は内側から畳むので、本体に `\begin` /
#: `\end` を含めない（`to_canonical` は安定するまで最大6回まわす）。
_ENV_RE = re.compile(
    r"\\begin\{(?P<env>[A-Za-z]+)\}"
    r"(?P<body>(?:(?!\\begin\{)(?!\\end\{).)*?)"
    r"\\end\{(?P=env)\}",
    re.S,
)
#: `\begin{array}[t]{cc}` の列指定。**`array` のときだけ**外す——他の環境で
#: これを食べると、先頭の要素が波括弧で書かれている場合にそれを落とす。
_ARRAY_SPEC_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\}")
#: 行区切り。`\\[2pt]` のような行送り指定と、plain TeX の `\cr` も同じ区切り。
_ROW_SEP_RE = re.compile(r"\\\\(?:\s*\[[^\]]*\])?|\\cr(?![A-Za-z])")
#: 列区切り。`\&`（エスケープされたアンパサンド）は区切りではない。
_COL_SEP_RE = re.compile(r"(?<!\\)&")
#: 入れ子の深さの上限。行列の中の行列（ブロック行列）は2〜3段で足りる。
_MAX_NESTING = 8
#: plain TeX の `\pmatrix{…}`。
_PLAIN_PMATRIX_RE = re.compile(r"\\pmatrix\s*\{")


def _cells(body: str) -> "list[list[str]] | None":
    """本体を行×列に割る。読めなければ None（推測しない）。"""
    rows = []
    for raw_row in _ROW_SEP_RE.split(body):
        cells = [c.strip() for c in _COL_SEP_RE.split(raw_row)]
        if len(cells) == 1 and not cells[0]:
            continue                      # 末尾の `\\` が作る空行
        if any(not c for c in cells):
            return None                   # 空のセルは書き間違い——揃え方を決めない
        rows.append(cells)
    if not rows:
        return None
    if len({len(r) for r in rows}) != 1:
        return None                       # ragged —— 揃え方の根拠がどこにも無い
    return rows


def _matrix_call(rows: "list[list[str]]") -> str:
    return "matrix(" + ", ".join("[" + ", ".join(r) + "]" for r in rows) + ")"


def _render(env: str, body: str) -> "str | None":
    """1つの環境を正準形へ。読まない環境・読めない本体は None。"""
    if env in MATRIX_ENVIRONMENTS:
        if env == "array":
            body = _ARRAY_SPEC_RE.sub("", body, count=1)
    elif env not in DETERMINANT_ENVIRONMENTS:
        return None                       # `Vmatrix` ほか —— 行列ではない
    rows = _cells(body)
    if rows is None:
        return None
    call = _matrix_call(rows)
    return call if env in MATRIX_ENVIRONMENTS else f"determinant({call})"


def is_norm_environment_text(s: str) -> bool:
    r"""その文はノルムの環境（`\begin{Vmatrix}…`）を含むか。

    「これはノルムの記法か」は `ui/interpret_notation` が既に答えている問いで、
    そちらは**縦棒の文字**を見ている（`\|v\|` / `||x||`）。環境の綴りは文字を
    持たないので届かず、`\begin{Vmatrix}` だけが一般の「解釈できません」に
    落ちていた——**理由の出ない断り**で、それは TEX-MATRIX-INPUT が直そうと
    している当のものである。綴りの表はここ（環境を知っている側）に置き、
    断りの文言と判断はあちらに1つのまま置く。
    """
    return any(f"\\begin{{{env}}}" in (s or "") for env in NORM_ENVIRONMENTS)


def norm_environment_operand(s: str) -> "str | None":
    """`\\begin{Vmatrix} … \\end{Vmatrix}` の**中身**を行列の綴りにして返す。

    ノルムの環境は行列としては読まない（`NORM_ENVIRONMENTS` の註のとおり、
    二重縦棒はノルムであって行列ではない）。だが**ノルムを取る相手**はその中身の
    行列なので、記法をノルムの操作へ配線するときはここから取る——環境の綴りを
    知っているのはこのモジュールなので、読む側で `\\begin`/`\\end` を数え直さない。
    """
    m = _ENV_RE.search(s or "")
    if m is None or m.group("env") not in NORM_ENVIRONMENTS:
        return None
    rows = _cells(m.group("body"))
    if rows is None:
        return None
    return _matrix_call(rows)


def _braced_body(s: str, open_brace: int) -> "tuple[str, int] | None":
    """`{` の位置から、対応する `}` までの中身と、その次の位置。"""
    depth = 0
    for i in range(open_brace, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_brace + 1:i], i + 1
    return None


def rewrite(s: str) -> str:
    r"""TeX の行列環境を、`[a, b; c, d]` と同じ節点へ書き換える。

    変換器の**入口**で呼ぶ——`_subscript_separator` / `_conjugate_mark` と同じ
    移動で、下流には綴りが1つだけ届く。ここを通さないと `\\` は消え `&` は
    パーサまで生き残る。
    """
    if not s or "matrix" not in s and "array" not in s:
        return s

    def _env(m: re.Match) -> str:
        out = _render(m.group("env"), m.group("body"))
        return m.group(0) if out is None else out

    # 内側から畳む（本体に `\begin` を含めない正規表現なので、1回の sub で
    # いちばん内側だけが置き換わる）。入れ子の外側は次の周回で当たる。
    for _ in range(_MAX_NESTING):
        nxt = _ENV_RE.sub(_env, s)
        if nxt == s:
            break
        s = nxt

    # plain TeX の `\pmatrix{…}`。波括弧の対応を数えて取る——正規表現では
    # 本体に現れる波括弧（`\frac{a}{b}` など）で切れてしまう。
    while True:
        m = _PLAIN_PMATRIX_RE.search(s)
        if not m:
            break
        got = _braced_body(s, m.end() - 1)
        if got is None:
            break
        body, after = got
        out = _render("pmatrix", body)
        if out is None:
            break
        s = s[:m.start()] + out + s[after:]
    return s
