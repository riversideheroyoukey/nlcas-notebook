r"""上付きが**ラベル**か指数かの読み分け（GF-NOTATION、2026-08-20）。

物理は摂動次数を上付きで書く —— `G^0`（0次のグリーン関数）、`\Sigma^{(2)}`（2次の
自己エネルギー）。これを**指数**として読むと、実測でこうなる:

    G = G^0 + G^0 \Sigma G   →  Maxima: G*Sigma+1     ← G^0 が 1
    \Sigma^{(1)}             →  Sigma                 ← 1次の自己エネルギーが自分自身
    \Sigma^{(2)}(k,\omega)   →  Sigma^2 と (k,omega) の積
    f^{(3)}(x)               →  f^3 * x               ← 3階微分のつもりが3乗

いずれもエラーを出さずに別の式になる。読み分けの根拠は**推測ではない**:

1. **括弧付き `^{(n)}` は指数ではない。** 指数に括弧を付ける習慣が無く、しかも
   指数として読むと `X^{(0)}`=1・`X^{(1)}`=X と退化する。書く人がいない読みなので
   競合しない。（ラベルか **n 階微分**かの区別は別問題として残る —— どちらにせよ
   「指数ではない」ので、ここで指数と読むことだけは確実に誤り。）
2. **素の `^0` / `^1` もラベル。** 指数としてはやはり退化する（1 と X）ので、
   その意味で書く人はいない。
3. **それ以外の素の上付き（`^2`・`^R`）は指数のまま。** `x^2` は本物の2乗であり、
   ここを推測で倒すと日常的な式を壊す。ラベルとして使いたい場合は定義が要る
   （`G^R = …`）—— 研究者が書いたものが判断材料になる、という既存の順序
   （`subscript_policy` の「宣言 > ドメイン > 記憶 > 畳む」）と同じ立場。

**綴り**: ラベルは名前の一部になるので、CAS で有効な識別子に落とす ——
`G^0` → ``G_sup0``、`\Sigma^{(2)}` → ``Sigma_psup2``。`p` は括弧付きの印で、
表示を元の綴りに戻すために要る（`G^{2}` と `G^{(2)}` は再入力すると別の読みに
なるので、往復には区別が要る）。キャレットをエスケープして `G\^0` と綴る案は
実測で棄却した —— Maxima は原子として受けるが `tex1` が `{\it G^}_{0}` を返し、
**表示が壊れる**（プライム `x\'` が `{\it x'}` を返して往復したのとは違う）。

**順序非依存**: `c^\dagger_k` と `c_k^\dagger` は同じ演算子であり、ラベルは
スロットではないので、下付きより先に名前へ畳む。テンソルの上下添字は
**スロット順が意味を持つ**（`T^\mu{}_\nu`）ので、この規則には含めない
（open_items MIXED-SCRIPT-NESTING）。
"""
from __future__ import annotations

import re

from notebook.runtime.latex_typography import braced_group

#: 名前トークン: `G`、`\Sigma`、`\varphi` など。
_NAME = r"\\?[A-Za-z][A-Za-z0-9]*"

#: 束縛演算子。ここでの上付きは**範囲の上限**であってラベルではない
#: (`\int_0^1`・`\sum_{i=1}^{N}`)。除外を忘れると `\int_0^1 x dx` の `^1` が
#: 退化ラベルと読まれ、積分そのものが消える（実測 2026-08-20、フルスイートで
#: 25件が落ちた）。`\lim` も上付きを取らないが、同じ理由でここに置く。
_BINDERS = frozenset({
    "sum", "prod", "coprod", "int", "iint", "iiint", "oint",
    "bigcup", "bigcap", "bigoplus", "bigotimes", "bigwedge",
    "lim", "limsup", "liminf", "max", "min", "sup", "inf", "argmax", "argmin",
})

#: 素の上付きでラベルと決まる値（指数として退化するもの）。
_DEGENERATE = ("0", "1")

_SUFFIX_RE = re.compile(r"_(p?)sup([A-Za-z0-9]+)$")


def label_suffix(value: str, parenthesised: bool) -> str:
    """ラベルの canonical 接尾辞。``0`` → ``_sup0``、``(2)`` → ``_psup2``。"""
    return f"_{'p' if parenthesised else ''}sup{value}"


def split_label(name: str) -> "tuple[str, str, bool] | None":
    """``G_sup0`` → ``("G", "0", False)``、``Sigma_psup2`` → ``("Sigma", "2", True)``。

    ラベルを持たない名前には None。
    """
    m = _SUFFIX_RE.search(name)
    if not m:
        return None
    base = name[: m.start()]
    if not base:
        return None
    return base, m.group(2), bool(m.group(1))


def is_label_name(token: str) -> bool:
    """下付きトークンが上付きラベルの綴りか（``sup0``・``psup2``）。"""
    return bool(re.fullmatch(r"p?sup[A-Za-z0-9]+", token))


def _read_script(text: str, i: int) -> "tuple[str, str, bool, int] | None":
    """*i* にある ``^…`` / ``_…`` を1つ読む。

    ``(kind, inner, braced, next_index)`` を返す。``kind`` は ``"^"`` / ``"_"``。
    """
    if i >= len(text) or text[i] not in "^_":
        return None
    k = i + 1
    group = braced_group(text, k)
    if group is not None:
        return text[i], group[0].strip(), True, group[1]
    m = re.match(r"\\?[A-Za-z0-9]+|.", text[k:])
    if not m:
        return None
    return text[i], m.group(0), False, k + m.end()


def _as_label(inner: str, braced: bool) -> "tuple[str, bool] | None":
    """上付きの中身がラベルなら ``(値, 括弧付きか)``、でなければ None。"""
    stripped = inner.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        value = stripped[1:-1].strip()
        if re.fullmatch(r"[A-Za-z0-9]+", value):
            return value, True
        return None
    if stripped in _DEGENERATE:
        return stripped, False
    return None


def label_script_at(text: str, i: int) -> bool:
    r"""*i* から**上付きラベル**が始まるか（``^{(HF)}``・``^0``）。指数なら False。

    名前の後ろに続く綴りが「別の名前を作るのか、その名前の冪なのか」を訊く口。
    ここに口が要るのは、その問いに**別の場所が独自に答えて外していた**ため:
    2a-sym（``supervisor_alias_rewrite``）は「識別子の直後が ``_`` なら別の名前」
    と正しく除外しながら ``^`` を見ておらず、``G`` が定義済みのセッションで
    ``G^{(HF)}_{n\sigma}(t,t')`` が ``E22^{(HF)}_{n\sigma}(t,t')`` に化けた
    （実測 2026-08-21）。``^`` を一律に外すのは誤りで、``G^2`` の ``G`` は
    まさに定義された量である——だから判断はこの規則の持ち主に訊く。
    """
    read = _read_script(text, i)
    if read is None or read[0] != "^":
        return False
    return _as_label(read[1], read[2]) is not None


def rewrite(s: str) -> str:
    r"""ラベルの上付きを名前の接尾辞に畳む（順序非依存）。

    ``G^0`` → ``G_sup0``、``\Sigma^{(2)}`` → ``\Sigma_psup2``、
    ``G^0_\sigma`` と ``G_\sigma^0`` → どちらも ``G_sup0_\sigma``。
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        m = re.compile(_NAME).match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue
        name = m.group(0)
        if name.lstrip("\\") in _BINDERS:
            out.append(name)
            i = m.end()
            continue
        j = m.end()
        scripts: list[tuple[str, str, bool]] = []
        while True:
            read = _read_script(s, j)
            if read is None:
                break
            kind, inner, braced, nxt = read
            scripts.append((kind, inner, braced))
            j = nxt
            if len(scripts) == 2:
                break
        label = None
        rest: list[tuple[str, str, bool]] = []
        for kind, inner, braced in scripts:
            got = _as_label(inner, braced) if kind == "^" and label is None else None
            if got is not None:
                label = got
            else:
                rest.append((kind, inner, braced))
        if label is None:
            out.append(name)
            i = m.end()
            continue
        out.append(name + label_suffix(*label))
        for kind, inner, braced in rest:
            out.append(f"{kind}{{{inner}}}" if braced or len(inner) > 1
                       else f"{kind}{inner}")
        i = j
    return "".join(out)
