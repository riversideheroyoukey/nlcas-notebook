r"""研究者が書いた次数を、その枝が本当に返せるか。

**書かれた次数が結果を変えないなら、黙らない。** 実測 2026-09-06、F-1 の3経路が
次数を読み取ったうえで捨てるか、範囲へ黙ってクランプしていた——どれも
`completed` で、`warnings` は `None` だった:

===========================  ===================================  ==============
打った文                      返っていたもの                        本当の上限
===========================  ===================================  ==============
``…S^x の8次までのモーメント``  ``M_0, M_2``（1〜8 どれでも同じ）      2次
``…S^x の5段の連分数``          ``b_1^2``（1〜5 どれでも同じ）         1段
``…高温展開を6次まで``          ``⟨H²⟩⟨H³⟩⟨H⁴⟩``（`max(2,min(4,n))`）  2〜4次
===========================  ===================================  ==============

**なぜ断るのか。** 返っていたのは**別の問いへの正しい答え**である。研究者は
4次モーメントを見たくて打ち、2次の式を受け取って先へ進む——このプロジェクトが
いちばん警戒している形そのもの。断り文は**行ける道を名指す**（一般式は記号 z の
低次だけだが、格子を具体的に書けば次数は効く——実測: 3サイト鎖で 2/4/6/8 が
そのとおり返る）。

**なぜ警告ではなく断りか。** 警告は「答えは合っているが注意」の合図である。ここは
答えが要求と違うので、`operand_guards` と同じ「使う地点で理由付きで断る」に倣う。

**上限は測って書く。** 一般モーメントが2次までなのは `general_low_moments`（名前が
low）、連分数が1段までなのは `general_b1_squared`、高温展開が4次までなのは
4次に要る `graph_four_cycle_count` はあるが5-サイクルの数え上げが無いため。
"""
from __future__ import annotations

from notebook.runtime.i18n import t

#: 一般式（配位数 z による記号形）で分かっている最高のモーメント次数。
GENERAL_MOMENT_MAX = 2
#: 同じく、連分数の段数。
GENERAL_CONTFRAC_MAX = 1
#: 高温展開が返せる次数の範囲。下限は ⟨H⟩ が無限温度で 0 になるため。
ENERGY_MOMENT_RANGE = (2, 4)


#: `次`（次数）で書かれたモーメントを、recursion の段数に直す割り算。
#: モーメントは偶数次しか無い（無限温度では奇数次が恒等的に 0）ので、
#: 段数 k は次数 2k である。
_MOMENT_ORDER_PER_STEP = 2


def recursion_units(mode: str, unit: "str | None", n: "int | None",
                    default: int) -> "tuple[int, int | None]":
    r"""研究者が書いた数を `(段数, 研究者が求めた量)` に直す。

    **`次` と `段` は別の量である。** モーメントは $M_0, M_2, M_4, \dots$ と
    偶数次しか無いので、`2次まで` は $M_0, M_2$ を指し、recursion を **2段**
    回すと $M_4$ まで出る——同じ「2」が2つの量になる。

    ====================  ==========  ===============  ==================
    打った文               unit        段数（計算に渡す）  求めた量（断りが比べる）
    ====================  ==========  ===============  ==================
    ``2次までのモーメント``   ``order``   1                2次
    ``2段までのモーメント``   ``depth``   2                4次
    ``2次までの連分数``      ``order``   2                2段
    ``2段までの連分数``      ``depth``   2                2段
    ====================  ==========  ===============  ==================

    **連分数だけ `次` と `段` が同じ数になるのは、対象が違うからである。**
    連分数の「n次近似」は係数 $b_1^2 \dots b_n^2$ を持つ段のことで、
    モーメントの「n次」は $M_n$ そのものを指す。言語が不整合なのではなく、
    数えている物が違う。

    実測 2026-09-10（起票 MOMENT-ORDER-MEANING）: 具体格子の枝が書かれた数を
    **段数として**使っていたので `2次までのモーメント` が $M_0, M_2, M_4$ を
    返していた。一般式の枝・断り文（上限 2）・**表示のラベル `M_{2k}` 自身**は
    次数として扱っており、断り文が名指す回避策（格子を具体的に書いて `4次まで`）
    に従うと $M_8$ まで返る、という食い違いになっていた。

    奇数次は無限温度で恒等的に 0 なので、`3次まで` は `2次まで` と同じ集合を
    求めている——返る $M_0, M_2$ がその答えである（ラベルが次数を言うので、
    何が返ったかは画面で分かる）。
    """
    if n is None:
        return default, None
    if mode == "contfrac":
        # 連分数は段そのものを数える。`次`＝n 次近似＝n 段。
        return n, n
    if unit == "depth":
        return n, n * _MOMENT_ORDER_PER_STEP
    return n // _MOMENT_ORDER_PER_STEP, n


def general_spin_moment_refusal(mode: str, requested: int, lang: str) -> "str | None":
    """一般式のモーメント／連分数で、書かれた次数が届かないなら理由。"""
    cap = GENERAL_CONTFRAC_MAX if mode == "contfrac" else GENERAL_MOMENT_MAX
    if requested <= cap:
        return None
    key = ("order.general_contfrac_beyond" if mode == "contfrac"
           else "order.general_moment_beyond")
    return t(key, lang, requested=requested, cap=cap)


def energy_moment_refusal(requested: int, lang: str) -> "str | None":
    """高温展開で、書かれた次数が範囲の外なら理由。"""
    lo, hi = ENERGY_MOMENT_RANGE
    if lo <= requested <= hi:
        return None
    return t("order.energy_moment_out_of_range", lang,
             requested=requested, lower=lo, upper=hi)
