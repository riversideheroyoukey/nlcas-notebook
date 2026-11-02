"""研究者が**語で書いた数**を読む —— 言語の数詞の表、1つ。

この表は `ui/interpret_common.py` に在った。**降りてきたのは、`notebook/runtime`
の2箇所が同じ問いを訊いたから**である（`_looks_like_type_label`：`ゼロ` は型の名詞
ではない／`to_canonical`：値として書かれた `zero` は記号ではない）。`ui` から
`notebook.runtime` へは import できないので、置いたままにすると表が2枚になる——
そして表が2枚あると、片方だけに綴りが足されて「当たるのに読めない」綴りができる
（`three times` がまさにそれで、序数の表だけを持っていたため**1階**を返していた）。
増え方は操作の集合ではなく言語の数詞に従うので、ここが置き場所である。

**数え方と値は別の語彙である。** `third`（序数）や `twice`（倍数副詞）は
「何階の微分か」を数えるが、**値ではない**——`set the right-hand side to twice`
は文ですらない。だから読みは2つに分かれる:

| 関数 | 読む語 | 用途 |
|---|---|---|
| `number_word` | 基数・序数・倍数副詞・数字 | 数える（階数・次数・回数） |
| `number_literal` | **基数と数字だけ** | 値として書かれた数 |
"""
from __future__ import annotations

import re

#: 全角数字 → ASCII。日本語で打つと IME が気づかないうちに出す。
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

#: 基数。**値になれるのはこの群だけ。**
#: 0 が無かったのは偶然ではない——この表は「何階微分か」「何回か」を数えるために
#: 育っており、そこに 0 は書かれない（`0階微分` とは誰も書かない）。値の欄が
#: 初めて要った 2026-09-10 に、**いちばん普通の値**が表に無いと分かった:
#: `set the right-hand side of E1 to zero` は `a*x + b = zero` を **completed**
#: で登録し（`zero` が Maxima の記号になる）、`x をゼロとする` は
#: `define_scalar_parameter` を **confirmed** で返していた（実測）。
_CARDINAL_WORDS: dict[str, int] = {
    '零': 0, '〇': 0, 'ゼロ': 0, 'zero': 0,
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}

#: 序数。**数えるだけ**で、値にはならない。
_ORDINAL_WORDS: dict[str, int] = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
}

#: 倍数副詞。「三回微分して」の英語は `three times`（基数）だが、`once`〜`thrice`
#: という1語の綴りもある。これも**数えるだけ**である。
_TIMES_WORDS: dict[str, int] = {'once': 1, 'twice': 2, 'thrice': 3}

#: 数える側が読む語の全部。**部分集合の関係をデータで書く**——表を3枚に分けて
#: 合成しているので、基数に綴りを足せば数える側にも自動で届く。
_NUMBER_WORDS: dict[str, int] = {**_CARDINAL_WORDS, **_ORDINAL_WORDS, **_TIMES_WORDS}

#: `3rd` / `2nd` / `4th`。語の序数（`third`）は `_ORDINAL_WORDS` が読む。
_ORDINAL_SUFFIX_RE = re.compile(r"(?i)^(\d+)(?:st|nd|rd|th)$")


def number_word(text: "str | None") -> "int | None":
    """`3` / `３` / `三` / `third` を 3 に。読めなければ None（推測しない）。"""
    if text is None:
        return None
    s = str(text).strip().translate(FULLWIDTH_DIGITS)
    if s in _NUMBER_WORDS:
        return _NUMBER_WORDS[s]
    low = s.lower()
    if low in _NUMBER_WORDS:
        return _NUMBER_WORDS[low]
    # `3rd` / `2nd` / `4th` —— 数字に序数の接尾辞が付いた形。**序数がどう書かれるか
    # の答えはこの関数1つ**であるべきで、直す前は5つの鍵がそれぞれ
    # `(?:st|nd|rd|th)?` を自前で綴り、変換は `int()` だった。だから語の序数
    # （`third`）はどこにも届かず、`take the third order derivative` が**1階**を
    # 返していた（実測 2026-09-06、`completed`）。
    m = _ORDINAL_SUFFIX_RE.match(s)
    if m:
        return int(m.group(1))
    try:
        return int(s)
    except ValueError:
        return None


def number_literal(text: "str | None") -> "str | None":
    """**文全体が数**なら、その数字の綴り。そうでなければ None（推測しない）。

    `number_word` との違いは2つあり、どちらも意図的である:

    - **読む語彙が基数と数字だけ。** 序数と倍数副詞は数え方であって値ではない
      ので、`to_canonical("third")` が `3` を返すようなことは起こらない。
    - **返すのは綴り（`str`）。** 値は CAS に渡す文字列であって、int にすると
      `-3` や全角の往復で綴りを作り直すことになる。

    **文の一部は決して書き換えない。** 式の中に立っている `one` は名前であって
    値ではなく、部分置換にすると研究者の記号を黙って数に変えてしまう。ここが
    「全体が数か」しか訊かない理由である。
    """
    if text is None:
        return None
    s = str(text).strip().translate(FULLWIDTH_DIGITS)
    if not s:
        return None
    if s in _CARDINAL_WORDS:
        return str(_CARDINAL_WORDS[s])
    low = s.lower()
    if low in _CARDINAL_WORDS:
        return str(_CARDINAL_WORDS[low])
    if re.fullmatch(r"[+-]?\d+", s):
        return s
    return None
