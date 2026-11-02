"""昇降演算子の呼び名——**どの語がどちら向きか**を1箇所で答える。

第二量子化では `消滅`/`生成`、調和振動子や角運動量の文脈では `下降`/`上昇`
（`lowering`/`raising`）と呼ぶ。**同じ対象の別の呼び名**であって、研究者は
自分の分野の語で書く。

直す前は `消滅|生成` / `creation|annihilation` しか知らず、`下降演算子` /
`lowering operator` は**型ラベルの既定に落ちて `define_scalar_parameter` を
`confirmed` で登録していた**（実測 2026-09-08、日英）——CLAUDE.md が
「演算子と書いた文がスカラーを登録する」として記録している当の欠陥が、
別の綴りで残っていた。

**語彙を広げるだけでは足りない。** 向きは下流で

    op_type_key = "annihilation" if (kind == "消滅" or "annihil" in kind) else "creation"

と**値で引かれて**おり、`else` が `creation` を既定にしている。`下降` を
パターンに足すだけだと、下降演算子が**生成演算子として宣言される**——
`multiply_expressions` のテンプレートが `*` を直書きしていたのと同じ形
（MATMUL-TEMPLATE-STAR）。だから語彙と対応付けをここに並べて置く。

`昇降` / `ladder` は**族の名前で、向きを言っていない**ので `LADDER_FAMILY` に
分ける。「昇降演算子の宣言か」を訊く guard はこれも受けてよいが、向きを要求する
読みは受けてはならない——受ければ既定で片方に倒れる。
"""
from __future__ import annotations

#: 向きを名指す語 → 向き。**既定を置かない**（書かれていない向きは埋めない）。
LADDER_DIRECTION: dict[str, str] = {
    "消滅": "annihilation", "下降": "annihilation",
    "annihilation": "annihilation", "lowering": "annihilation",
    "生成": "creation", "上昇": "creation",
    "creation": "creation", "raising": "creation",
}

#: 族の名前——向きを言っていない。
LADDER_FAMILY: tuple[str, ...] = ("昇降", "ladder")


def ladder_kind(word: str) -> "str | None":
    """`消滅` → `"annihilation"`、`raising` → `"creation"`、読めなければ `None`。

    **`None` は「向きが書かれていない」であって「生成」ではない。** 呼び出し側は
    `None` を既定で埋めず、書かれていないものとして扱うこと。
    """
    w = (word or "").strip().lower()
    return LADDER_DIRECTION.get(w) or LADDER_DIRECTION.get((word or "").strip())


def alternation(kind: str, lang: str) -> str:
    """`nl_patterns` の fragment が**この表と一致しているか**を測るための綴り。

    パターンは JSON なのでここを呼べない。だから fragment には綴りを書くが、
    それが**この表と同じであること**は `tests/test_ladder_operator_naming.py`
    が機械的に確かめる——書き写しを許すなら、割れていないことを検査が担保する。
    """
    def _is_ja(w: str) -> bool:
        return any(ord(c) > 0x2E80 for c in w)
    want_ja = lang == "ja"
    words = [w for w, k in LADDER_DIRECTION.items()
             if k == kind and _is_ja(w) == want_ja]
    return "|".join(words)
