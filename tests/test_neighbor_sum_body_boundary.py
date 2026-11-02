r"""近接和の本体はどこで終わるか——**位置ではなく、指定が始まる手前**。

実測 2026-09-08（worklist の `site` / `lattice`。同じ主題なので一緒に測った
——鍵の集合が `f1_ns_*` で重なっており、別々に畳むと二度目に「一覧に寄せる」を
やり直すことになる）。

日本語には語順が2通りあり、**どちらもごく普通の書き方なのに片方しか通らなかった**:

| 打った文 | 直す前の body |
|---|---|
| `Sz_i Sz_j の正方格子 2x2 の最近接和を展開して` | `Sz_i Sz_j` ✔ |
| **`S^z_i S^z_j の最近接和を 4 サイト 正方格子で展開して`** | **`S^z_i S^z_j の最近接和を 4 サイト`** |
| `S^z_i S^z_j の最近接和を 4 サイト 周期境界で展開して`（格子名なし） | `S^z_i S^z_j` ✔ |

本体に**和の語とサイト数が混入**する。混入した `の` はパーサに届き、

    文字 `の` は数式として読めません。ワープロや PDF から貼り付けた記号
    （ダッシュ `–`、プライム `′`、アクセント付き文字など）が混ざっていないか
    確認してください。

——研究者が書いたものと**無関係な断り**が返っていた（「拒否の理由が入力に当たって
いるか測る」の型）。英語は語順が逆（`sum of BODY over LATTICE`）なので
`f1_ns_body[en]` が読み、この枝には来ない——**日英の割れ**だった。

境界は位置ではなく**文法**: 格子名・和の語・サイト数のうち、最初に現れるもの
（`_ja_body_end`）。

**併せて2つ。** ①`nearest neighbour`（英式綴り）がどこにも無かった
——`rationalize|rationalise` を両方受けるこの系の方針と食い違っていた。
②`f1_ns_trigger` と `f1_ns_body` が**同じ4語を二重に綴って**おり、英語側は
一致していなかった（body だけが `all-pairs sum` を持つ）。fragment
`nn_sum_kw_ja` / `nn_kw_en` に寄せた。**ただし畳み切れていなかった**——
英語側は形容詞だけを持ち、`sum` と `lattice sum` は鍵の中に残っていた
（2026-09-24 の五十五問目で `nn_sum_kw_en` に揃えた。下の
`TestTheTriggerVocabularyLivesInOnePlace` を見よ）。

**私の走査の訂正。** 最初 `site` を「英語側9・日本語側6で非対称」と報告したが
**誤り**——群名 `(?P<site>…)` を綴りとして数えていた（worklist の道具が直したのと
同じ誤り）。数え直すと日英とも6鍵で**対称**、`格子`/`lattice` も 7/7。

綴り空間 ja **5/10 → 10/10**、en **8/9 → 9/9**。
"""
import pathlib

import pytest

from notebook.runtime.nl_patterns import nlp
from ui.interpret_f1 import _parse_neighbor_sum

BODY = "S^z_i S^z_j"

JA_SPELLINGS = [
    ("S^z_i S^z_j の最近接和を 4 サイト 周期境界で展開して", None, "periodic"),
    ("S^z_i S^z_j の最近接相互作用和を 4 サイト 周期境界で展開して", None, "periodic"),
    ("S^z_i S^z_j の隣接和を 4 サイト 周期境界で展開して", None, "periodic"),
    ("S^z_i S^z_j の格子和を 4 サイト 周期境界で展開して", None, "periodic"),
    ("S^z_i S^z_j の最近接和を４サイト周期境界で展開して", None, "periodic"),
    # 格子名が **和の語より後ろ** に来る語順——直す前はここで body が壊れていた
    ("S^z_i S^z_j の最近接和を 4 サイト 正方格子で展開して", "square_2d", "open"),
    ("S^z_i S^z_j の最近接和を 4 サイト 三角格子で展開して", "triangular_2d", "open"),
    ("S^z_i S^z_j の最近接和を 4 サイト ハニカムで展開して", "honeycomb_2d", "open"),
    ("S^z_i S^z_j の最近接和を 4 サイト 蜂の巣格子で展開して", "honeycomb_2d", "open"),
]

EN_SPELLINGS = [
    ("expand the nearest neighbor sum of S^z_i S^z_j over a 4-site periodic chain",
     None, "periodic"),
    ("expand the nearest-neighbor sum of S^z_i S^z_j over a 4-site periodic chain",
     None, "periodic"),
    # 英式綴り——直す前はどこにも無かった
    ("expand the nearest neighbour sum of S^z_i S^z_j over a 4-site periodic chain",
     None, "periodic"),
    ("expand the lattice sum of S^z_i S^z_j over a 4-site periodic chain",
     None, "periodic"),
    ("expand the all-pairs sum of S^z_i S^z_j over a 4-site chain", "all_pairs", "open"),
    ("expand the nearest neighbor sum of S^z_i S^z_j over a 4-site square lattice",
     "square_2d", "open"),
    ("expand the nearest neighbor sum of S^z_i S^z_j over a 4-site triangular lattice",
     "triangular_2d", "open"),
    ("expand the nearest neighbor sum of S^z_i S^z_j over a 4-site honeycomb",
     "honeycomb_2d", "open"),
]


class TestTheBodyEndsWhereTheSpecificationBegins:
    @pytest.mark.parametrize("text,pair,boundary", JA_SPELLINGS)
    def test_japanese(self, text, pair, boundary):
        r = _parse_neighbor_sum(text, "ja")
        assert r is not None, text
        _, args = r
        assert args["body"] == BODY, (text, args["body"])
        assert args.get("pair_constraint") == pair, (text, args.get("pair_constraint"))
        assert args.get("boundary") == boundary, text

    @pytest.mark.parametrize("text,pair,boundary", EN_SPELLINGS)
    def test_english(self, text, pair, boundary):
        r = _parse_neighbor_sum(text, "en")
        assert r is not None, text
        _, args = r
        assert args["body"] == BODY, (text, args["body"])
        assert args.get("pair_constraint") == pair, (text, args.get("pair_constraint"))

    @pytest.mark.parametrize("text", [
        "Sz_i Sz_j の正方格子 2x2 の最近接和を展開して",
        "Sz_i Sz_j の三角格子 2x3 の最近接和を展開して",
    ])
    def test_the_other_japanese_word_order_still_works(self, text):
        """**格子名が和の語より前**に来る語順。直す前はこちらだけが通っていた。"""
        r = _parse_neighbor_sum(text, "ja")
        assert r is not None and r[1]["body"] == "Sz_i Sz_j", (text, r)

    @pytest.mark.parametrize("text,pair", [
        # 本体 → **サイト数** → 格子名 → 和の語。境界はサイト数——変異 M3
        # （境界からサイト数を外す）が最初**生き残った**のは、この語順を
        # 測っていなかったからで、冗長ではなく**検査の穴**だった。
        ("Sz_i Sz_j の 4 サイト正方格子の最近接和を展開して", "square_2d"),
        ("Sz_i Sz_j の4サイト三角格子の最近接和を展開して", "triangular_2d"),
    ])
    def test_the_site_count_can_be_the_boundary(self, text, pair):
        r = _parse_neighbor_sum(text, "ja")
        assert r is not None, text
        assert r[1]["body"] == "Sz_i Sz_j", (text, r[1]["body"])
        assert r[1]["pair_constraint"] == pair and r[1]["n_sites"] == 4, r[1]

    def test_a_trailing_particle_is_not_part_of_the_body(self):
        """`Sz_i Sz_j を 4 サイトの…` は `Sz_i Sz_j を` を残していた。
        数式が助詞で終わることは無い。"""
        r = _parse_neighbor_sum("Sz_i Sz_j を 4 サイトの正方格子で最近接和に展開して", "ja")
        assert r is not None and r[1]["body"] == "Sz_i Sz_j", r

    def test_both_word_orders_give_the_same_body(self):
        a = _parse_neighbor_sum("Sz_i Sz_j の正方格子 2x2 の最近接和を展開して", "ja")
        b = _parse_neighbor_sum("Sz_i Sz_j の最近接和を 4 サイト 正方格子で展開して", "ja")
        assert a[1]["body"] == b[1]["body"] == "Sz_i Sz_j", (a[1]["body"], b[1]["body"])
        assert a[1]["pair_constraint"] == b[1]["pair_constraint"] == "square_2d"


class TestTheBritishSpellingIsRead:
    @pytest.mark.parametrize("key", ["f1_ns_trigger", "f1_ns_body"])
    @pytest.mark.parametrize("word", ["nearest neighbor", "nearest neighbour",
                                      "nearest-neighbor", "nearest-neighbour"])
    def test_both_spellings(self, key, word):
        probe = (f"{word} sum of X over a 3-site chain" if key == "f1_ns_body"
                 else f"{word} sum")
        assert nlp(key, "en").search(probe), (key, word)


class TestTheTriggerVocabularyLivesInOnePlace:
    """`f1_ns_trigger` と `f1_ns_body` は同じ語彙を二重に綴っていた。"""

    _JSON = None

    @classmethod
    def _d(cls):
        import json
        if cls._JSON is None:
            cls._JSON = json.loads(
                (pathlib.Path(__file__).resolve().parent.parent / "i18n"
                 / "nl_patterns.json").read_text(encoding="utf-8"))
        return cls._JSON

    @pytest.mark.parametrize("key", ["f1_ns_trigger", "f1_ns_body"])
    def test_japanese_splices_the_fragment(self, key):
        assert "{{nn_sum_kw_ja}}" in self._d()[key]["ja"]
        assert "最近接和" not in self._d()[key]["ja"], "また自前で綴っている"

    @pytest.mark.parametrize("key", ["f1_ns_trigger", "f1_ns_body"])
    def test_english_splices_the_fragment(self, key):
        """**畳んだ相手が違う問いに答えていた**（2026-09-24、五十五問目）。

        日本語の `nn_sum_kw_ja` は「和の名詞」（最近接和・隣接和・格子和…）、
        英語の相手は `nearest neighbou?r` という**形容詞だけ**で、`sum` も
        `lattice sum` も鍵の中に直書きされていた。だから **隣接和の鏡
        （`neighbour sum`）がどこにも無かった**。名前も `nn_sum_kw_en` に揃えた
        ので、鍵が自前で綴ってよい語はもう無い（`all-pairs sum` だけは
        `f1_ns_allpairs_kw` に英語の body が無いので body に残る）。
        """
        assert "{{nn_sum_kw_en}}" in self._d()[key]["en"]
        for spelled in ("neighbor", "neighbou", "lattice"):
            assert spelled not in self._d()[key]["en"], (key, spelled, "また自前で綴っている")

    def test_the_two_keys_agree_on_the_keyword(self):
        """直す前は body だけが `all-pairs sum` を持つなど食い違っていた。
        引き金の語そのものは同じ fragment から来る。"""
        import re
        for lang, probe in (("ja", "X の最近接和を 3 サイトで展開して"),
                            ("en", "expand the nearest neighbour sum of X over a 3-site chain")):
            assert nlp("f1_ns_trigger", lang).search(probe), lang
            assert nlp("f1_ns_body", lang).search(probe), lang


class TestAnUnknownLatticeIsRefusedByName:
    """設計どおり——**黙って1D鎖に落とさず**、既知の格子名を挙げて断る。"""

    @pytest.mark.parametrize("lang,text,name", [
        ("ja", "S^z_i S^z_j の最近接和を 4 サイト カゴメ格子で展開して", "カゴメ"),
        ("en", "expand the nearest neighbor sum of S^z_i S^z_j over a 4-site kagome lattice",
         "kagome"),
    ])
    def test_forced_through_as_the_literal(self, lang, text, name):
        r = _parse_neighbor_sum(text, lang)
        assert r is not None and r[1]["pair_constraint"] == name, (text, r)
        assert r[1]["body"] == BODY, r[1]["body"]      # 本体は壊さない
