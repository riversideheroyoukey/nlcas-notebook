r"""NL-GRAMMAR ① 五十五問目「和の名詞は日英で同じ表か」（2026-09-24）。

綴り空間の ratchet は「誰も打っていない綴り 0」まで来ているので、選んだのは
もう一つの読み ——「同じ問いを両言語が**別の名前**で訊いている組」
（`scripts/nl_spelling_coverage.py --pairs`）。残っていた3組を測ると:

| 組 | 実測 | |
|---|---|---|
| `ja_side_word` ↔ `en_side_phrase` | 16セル | **16/16 正しい**（名前だけの違い） |
| **`nn_sum_kw_ja` ↔ `nn_kw_en`** | **30セル** | **当たり** |

**名前が違ったのは、中身が違う問いに答えていたからだった。** 日本語側は「和の
名詞」（最近接相互作用和・最近接和・隣接和・格子和）、英語側は
`nearest neighbou?r` という**形容詞だけ**で、`sum` と `lattice sum` は鍵の中に
直書きされていた。だから **隣接和の鏡（`neighbour sum`）がどこにも無い**:

| 打った文 | 返っていたもの |
|---|---|
| `Expand the neighbor sum of Sz_i Sz_j over 4 sites …`（1D） | 落ちる（大きな声） |
| **`Expand the neighbor sum of Sz_i Sz_j over a 3x3 honeycomb lattice`** | **`3 Sz(8) Sz(9) Expand a neighbor of over sum the …` を `completed`** |
| **`… nearest-neighbor sum of Sz_i Sz_j in a 3x3 honeycomb lattice`** | **`… a honeycomb in lattice x_3 …` を `completed`** |

**英単語が被和関数の因子になって成功が返る。** 日本語は同じ軸で 10/10——表に
無い `近接和` でさえ 2D では正しい。境界が**文法**（字種と助詞、`_ja_body_end`）
だからで、英語の境界は**語彙**だから、語彙に穴があると本体ごと落ちる。
ENGLISH-INLINE-INTEGRAL と同じ理由（全 ASCII の文では字種の変わり目が原理的に
発火しない）。

**増幅していたのは2D の枝の取りこぼし方**である。英語の本体が読めなかったとき、
`_parse_neighbor_sum` は**日本語の位置規則**（「指定の手前を全部取る」）に落ちて
いた。1D の枝は同じ問いに**既に正しく答えていた**（読めなければ `return None`）
——一つの問いに二つの答え、しかも片方が正しい、いつもの形。

**直したもの**: ①`nn_sum_kw_en` が「和の名詞」を答える（`sum` と `lattice sum`
が鍵から消え、`neighbour sum` が入る）。②本体の終端の前置詞に `in` を足す。
③日本語の位置規則は日本語の文にだけ当てる。

**実測 30セル、獲得5・沈黙→大声1・損失0**（予測どおり）。表に無い
`adjacent sum` は 2D で**もっともらしい成功から大声の失敗へ**変わる。
コーパス差分 0——`f1_ns_*` の綴りはシナリオ2本にしか無く、どちらも
`nearest-neighbor sum of … over …` である（CORPUS-WORKING-ONLY）。

**2026-09-08 の兄弟課題はこの半分を畳んで満点を返していた**（綴り空間 ja 10/10・
en 9/9）。あの読みは「誰かが打っているか」しか訊かないので、**相手の言語が同じ
語彙を持っているか**は見えない。測定器を替えると同じ場所からもう1件出る。
"""
import pytest

from notebook.runtime.nl_patterns import nlp
from ui.interpret_f1 import _parse_neighbor_sum

BODY = "S^z_i S^z_j"

#: 日本語側 fragment `nn_sum_kw_ja` の全綴り。
JA_SUM_NOUNS = ["最近接相互作用和", "最近接和", "隣接和", "格子和"]
#: 英語側 `nn_sum_kw_en` の全綴り。`neighbour sum` が 2026-09-24 に入った。
EN_SUM_NOUNS = ["nearest-neighbor sum", "nearest neighbor sum",
                "nearest neighbour sum", "nearest-neighbour sum",
                "neighbor sum", "neighbour sum", "lattice sum"]


def _ja_2d(kw):
    return f"{BODY} のハニカム格子 3x3 の{kw}を展開して"


def _ja_1d(kw):
    return f"{BODY} の{kw}を 4 サイト 周期境界で展開して"


def _en_2d(kw):
    return f"expand the {kw} of {BODY} over a 3x3 honeycomb lattice"


def _en_1d(kw):
    return f"expand the {kw} of {BODY} over a 4-site periodic chain"


class TestTheSumNounIsOneTableInBothLanguages:
    """**和の名詞**（形容詞ではない）が両言語の表であること。"""

    @pytest.mark.parametrize("kw", JA_SUM_NOUNS)
    @pytest.mark.parametrize("form", [_ja_2d, _ja_1d])
    def test_japanese(self, kw, form):
        parsed = _parse_neighbor_sum(form(kw), "ja")
        assert parsed is not None, form(kw)
        assert parsed[1]["body"] == BODY, (form(kw), parsed[1]["body"])

    @pytest.mark.parametrize("kw", EN_SUM_NOUNS)
    @pytest.mark.parametrize("form", [_en_2d, _en_1d])
    def test_english(self, kw, form):
        """`neighbour sum` は 2D で**英文まるごと**が本体になっていた綴り。"""
        parsed = _parse_neighbor_sum(form(kw), "en")
        assert parsed is not None, form(kw)
        assert parsed[1]["body"] == BODY, (form(kw), parsed[1]["body"])

    @pytest.mark.parametrize("key", ["f1_ns_trigger", "f1_ns_body"])
    def test_both_keys_read_the_same_english_table(self, key):
        """引き金と本体が**同じ表**を読む（片方だけ綴りを足せない）。"""
        for kw in EN_SUM_NOUNS:
            probe = (f"{kw} of X over a 3-site chain" if key == "f1_ns_body"
                     else kw)
            assert nlp(key, "en").search(probe), (key, kw)


class TestTheEnglishBodyEndsAtItsPreposition:
    r"""英語の本体の終わりも**語彙**である。`in` が抜けていた。

    `… of S^z_i S^z_j in a 3x3 honeycomb lattice` は本体が
    `S^z_i S^z_j in a 3x3 honeycomb lattice` になり、`a honeycomb in lattice`
    が因子として答えに入って `completed` が返っていた。
    """

    @pytest.mark.parametrize("prep", ["over", "on", "in", "for"])
    def test_every_preposition_ends_the_body(self, prep):
        text = f"expand the nearest-neighbor sum of {BODY} {prep} a 3x3 honeycomb lattice"
        parsed = _parse_neighbor_sum(text, "en")
        assert parsed is not None, text
        assert parsed[1]["body"] == BODY, (text, parsed[1]["body"])
        assert parsed[1]["pair_constraint"] == "honeycomb_2d", text


class TestAJapanesePositionalRuleDoesNotAnswerAnEnglishSentence:
    r"""**一つの問いに二つの答え、しかも片方が正しい。**

    1D の枝は「英語の本体が読めなければ `return None`」と答えており、2D の枝
    だけが「読めなかったら指定の手前を全部取る」——日本語の位置規則——に落ちて
    いた。全 ASCII の文でそれをやると英単語が数式になる。
    """

    def test_an_unreadable_english_sum_word_is_declined_not_invented(self):
        """`adjacent sum` は表に無い。**断念**であって、英文を本体にしない。"""
        text = _en_2d("adjacent sum")
        assert _parse_neighbor_sum(text, "en") is None, text

    def test_the_same_word_is_declined_in_the_1d_form_too(self):
        """1D は直す前からこう答えていた（写したのはこちらの答え）。"""
        assert _parse_neighbor_sum(_en_1d("adjacent sum"), "en") is None

    def test_japanese_keeps_its_positional_rule(self):
        """日本語は**和の語を知らなくても** 2D なら本体を正しく切る。

        引き金は格子名で、境界は助詞と字種——だから表に無い `近接和` でも通る。
        英語側を直すために日本語のこの強さを削ってはならない。
        """
        parsed = _parse_neighbor_sum(_ja_2d("近接和"), "ja")
        assert parsed is not None
        assert parsed[1]["body"] == BODY, parsed[1]["body"]

    def test_the_trigger_requires_the_word_sum(self):
        """`sum` 無しの引き金は**どこにも行かない綴り**だったので消した。

        本体は `… sum of …` を要求するので、`Expand the nearest-neighbor of X
        over a 3x3 honeycomb lattice` は直す前も後も F-1 が答えていない
        （実測、`completed` の中身も不変）。**通らない綴りを文法に置かない。**
        """
        text = f"expand the nearest-neighbor of {BODY} over a 3x3 honeycomb lattice"
        assert _parse_neighbor_sum(text, "en") is None, text
        assert not nlp("f1_ns_trigger", "en").search("nearest-neighbor of X")


class TestTheAllPairsFamilyIsUntouched:
    """畳んだのは近接和の語彙だけ。全ペア和（Σ_{i<j}）は両言語とも不変。"""

    def test_japanese(self):
        parsed = _parse_neighbor_sum(f"{BODY} の全ペア和を 4 サイトで展開して", "ja")
        assert parsed and parsed[1]["pair_constraint"] == "all_pairs"

    def test_english(self):
        parsed = _parse_neighbor_sum(
            f"expand the all-pairs sum of {BODY} over a 4-site chain", "en")
        assert parsed and parsed[1]["pair_constraint"] == "all_pairs"
