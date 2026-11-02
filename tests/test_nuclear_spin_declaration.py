r"""核スピンの宣言 —— NMR で書かれるとおりに読み、読めたものを記録する。

利用者の問い「NMR での核スピンの取り扱いをどうする？ $^{1}$H や $^{13}$C は
既存の粒子表に収まるか」を実測したところ、答えの**手前に2件の silent-wrong**
があった（2026-08-22）:

| 綴り | 実測 |
|---|---|
| Julia `S をスピン演算子として定義して` | `define_spin_operator` ✓ |
| Julia `I を核スピン演算子として定義して` | **`define_scalar_parameter`**——スカラーを登録して**成功** |
| Maxima（magnetic_resonance）どの綴りでも | **`define_scalar_parameter`** |

①主語の境界を「`を` ＋その語」で近似していたので、修飾が1つ挟まる
（「**核**スピン演算子」——NMR がそう書く）だけで宣言ぜんぶを失い、既定の
スカラー宣言に落ちていた。②読み取りが Julia ハンドラにしか無く、
`define_spin_operator` は julia_full 限定だった。

**核種は粒子表とは別の軸**である: 表の同一性は「粒子の種類」、核種は同位体で、
$^{13}$C のスピン 1/2 は「炭素」からは出ない（$^{12}$C は 0、$^{14}$C も 0）。
"""
import pytest

from notebook.runtime.backend_handler import read_spin_operator_declaration
from notebook.runtime.spin_value_set import (find_nuclide, format_numerator,
                                             ladder_numerators, load_nuclides,
                                             nuclide_spellings,
                                             projections_for_nuclide)


class TestTheSubjectBoundaryIsTheParticle:
    r"""修飾が挟まっても宣言を失わないこと。"""

    @pytest.mark.parametrize("text, name, desc", [
        ("S をスピン演算子として定義して", "S", ""),
        ("I を核スピン演算子として定義して", "I", "核"),
        (r"I を ^{13}C の核スピン演算子として定義して", "I", r"^{13}C の核"),
        ("S を全スピン演算子とする", "S", "全"),
        ("K を電子スピン演算子として定義して", "K", "電子"),
        ("define I as the nuclear spin operator of 13C", "I", "nuclear"),
        ("define S as a spin operator", "S", ""),
    ])
    def test_read(self, text, name, desc):
        got = read_spin_operator_declaration(text)
        assert got is not None, f"{text!r} が読めない"
        assert got == (name, desc)

    @pytest.mark.parametrize("text", [
        "A をスカラーパラメータとして定義して",
        "c を電子の消滅演算子として定義して",
        "define A as a scalar parameter",
        "define c as an electron annihilation operator",
        "E1 を展開して",
    ])
    def test_other_declarations_are_untouched(self, text):
        assert read_spin_operator_declaration(text) is None


class TestANuclideIsNotAParticleKind:
    r"""核種の同一性は同位体であって元素ではない。"""

    def test_carbon_isotopes_differ(self):
        table = {n["key"]: n for n in load_nuclides()}
        assert table["12C"]["two_s"] == 0      # NMR-silent
        assert table["13C"]["two_s"] == 1      # I = 1/2
        assert "14C" not in table or table["14C"]["two_s"] == 0

    def test_statistics_is_derivable_and_not_stored(self):
        """質量数の偶奇から厳密に決まるので保存しない（食い違う余地を作らない）。"""
        for n in load_nuclides():
            assert "statistics" not in n
            assert (n["mass_number"] % 2 == 1) == (n["two_s"] % 2 == 1)

    @pytest.mark.parametrize("spelling", [
        r"^{13}C", "^13C", "¹³C", "C-13", "13C",
    ])
    def test_every_spelling_finds_it(self, spelling):
        assert find_nuclide(spelling)["key"] == "13C"

    def test_a_longer_nuclide_is_not_read_as_a_shorter_one(self):
        r"""`13C` は `113Cd` の部分文字列——素朴な `in` では読み違える。"""
        assert find_nuclide("113Cd")["key"] == "113Cd"
        assert find_nuclide(r"^{113}Cd")["key"] == "113Cd"

    @pytest.mark.parametrize("text", ["nuclear", "カーボン", "core spin", ""])
    def test_nothing_is_invented(self, text):
        assert find_nuclide(text) is None
        assert projections_for_nuclide(text) is None


class TestTheProjectionsFollowTheNuclide:
    @pytest.mark.parametrize("text, spin, count", [
        (r"^{13}C", "1/2", 2),
        ("1H", "1/2", 2),
        ("2H", "1", 3),          # 四極子核
        ("14N", "1", 3),
        ("23Na", "3/2", 4),
        ("27Al", "5/2", 6),
        ("12C", "0", 1),         # NMR-silent
    ])
    def test_ladder(self, text, spin, count):
        nuc = find_nuclide(text)
        nums = ladder_numerators(int(nuc["two_s"]))
        assert format_numerator(int(nuc["two_s"])) == spin
        assert len(nums) == count
        assert projections_for_nuclide(text) == nums

    def test_the_numerators_stay_integers(self):
        """半整数を float にしない（F-9 の Surd 規則）。"""
        for n in load_nuclides():
            for m in ladder_numerators(int(n["two_s"])):
                assert isinstance(m, int)

    def test_spellings_are_generated_not_listed(self):
        """綴りは (A, 元素記号) から作る——列挙すると片方だけ足し忘れる。"""
        nuc = {n["key"]: n for n in load_nuclides()}["27Al"]
        assert set(nuclide_spellings(nuc)) >= {"27Al", "Al-27", r"^{27}Al", "²⁷Al"}


class TestOnlyNuclidesWithAOneCharacterSpellingExpand:
    r"""展開できるのは1文字の綴りがある値集合だけ（OPERAND-BOUNDARY ③ の線）。

    **どの集合に綴りがあるかは 2026-09-12 に広がった**（研究者の選択: 種別ごとに
    物理の慣用綴りを置く）。整数スピンは $\pi^+\pi^0\pi^-$ と同じ `⁺ ⁰ ⁻` に
    なったので、**スピン1の核種（²H・¹⁴N）は展開される**。3/2 と 5/2 には
    1文字の慣用綴りが無いので、そこは形式和のまま——推測しない。
    """

    @pytest.mark.parametrize("text,want", [
        ("13C", ["↓", "↑"]), ("1H", ["↓", "↑"]),          # 1/2 → 矢印
        ("2H", ["⁻", "⁰", "⁺"]), ("14N", ["⁻", "⁰", "⁺"]),  # 1   → 上付き符号
    ])
    def test_a_set_with_a_convention_gets_one(self, text, want):
        from notebook.runtime.spin_value_set import index_labels
        assert index_labels(projections_for_nuclide(text)) == want

    @pytest.mark.parametrize("text", ["23Na", "27Al", "17O"])
    def test_three_halves_and_above_have_no_one_character_spelling(self, text):
        r"""$3/2$・$5/2$ に1文字の慣用は無い（重い/軽い正孔の `⇑↑↓⇓` は
        半導体の話で、**粒子表にも核種表にも $2S=3$ の使い道が無い**——
        通らない綴りは足さない）."""
        from notebook.runtime.spin_value_set import index_labels
        assert index_labels(projections_for_nuclide(text)) is None


class TestBothLanguagesRecordTheNuclide:
    r"""日英で**核種の位置が逆**でも記録されること。

    日本語は前（`^{13}C の核スピン演算子`）、英語は後ろ（`nuclear spin operator
    of 13C`）に置く。捕獲した説明部だけを見ていたとき、英語は何も記録しなかった
    （実測 2026-08-22）——だから宣言文ぜんぶを見る。核種の綴りは必ず質量数を
    伴うので、演算子名の裸の `I` が誤って当たることはない。
    """

    @pytest.mark.parametrize("text, key", [
        (r"I を ^{13}C の核スピン演算子として定義して", "13C"),
        ("K を 27Al の核スピン演算子として定義して", "27Al"),
        ("define I as the nuclear spin operator of 13C", "13C"),
        ("define K as the nuclear spin operator of 27Al", "27Al"),
    ])
    def test_the_nuclide_is_found_in_the_whole_sentence(self, text, key):
        assert read_spin_operator_declaration(text) is not None
        assert find_nuclide(text)["key"] == key

    @pytest.mark.parametrize("text", [
        "S をスピン演算子として定義して",
        "define S as a spin operator",
        "I を核スピン演算子として定義して",          # 核種を書いていない
    ])
    def test_saying_nothing_records_nothing(self, text):
        assert read_spin_operator_declaration(text) is not None
        assert find_nuclide(text) is None

    def test_the_operator_name_is_not_read_as_a_nuclide(self):
        """`I` はヨウ素の元素記号だが、綴りは必ず質量数を伴う。"""
        assert find_nuclide("I をスピン演算子として定義して") is None
        assert find_nuclide("define I as a spin operator") is None


class TestSiteSpinsMayBeWrittenAsANuclide:
    r"""サイトのスピンを**核種**で書ける——大きさを言い直させない。

    核種を書けば $I$ は決まる。`^{13}C` と言ったあとに `1/2` と書かせるのは
    同じことを二度言わせることで、二度目のほうが間違えうる。

    **核種の読みが先**である: 数値パターンは `13C` の `13` を拾ってスピン 13 を
    記録する（実測 2026-08-22、最初の版がそうなった）。質量数と元素記号が
    くっついたものは大きさではないので、区別できるほうが先に走る。
    """

    @staticmethod
    def _pairs(text, lang):
        from ui.interpret_spin_species import _pairs
        return _pairs(text, lang)

    @pytest.mark.parametrize("lang, text, want", [
        ("ja", "サイト1のスピンは 13C、サイト2のスピンは 27Al とする",
         {"1": "1/2", "2": "5/2"}),
        ("ja", "サイト1のスピンを1/2、サイト2のスピンを1とする",
         {"1": "1/2", "2": "1"}),
        ("en", "site 1 has spin 13C and site 2 has spin 27Al",
         {"1": "1/2", "2": "5/2"}),
        ("en", "site 1 has spin 1/2 and site 2 has spin 1",
         {"1": "1/2", "2": "1"}),
        ("ja", r"サイト1のスピンは ^{13}C とする", {"1": "1/2"}),
        ("ja", "サイト1のスピンは C-13 とする", {"1": "1/2"}),
    ])
    def test_pairs(self, lang, text, want):
        assert self._pairs(text, lang) == want

    def test_the_mass_number_is_never_read_as_a_magnitude(self):
        """`13C` → 1/2 であって 13 ではない。"""
        assert self._pairs("サイト1のスピンは 13C とする", "ja") == {"1": "1/2"}
        assert self._pairs("site 1 has spin 13C", "en") == {"1": "1/2"}

    def test_an_unknown_nuclide_falls_back_to_the_number_reading(self):
        """表に無い綴りは核種として読まない——推測しない。"""
        got = self._pairs("サイト1のスピンは 99Zz とする", "ja")
        assert got.get("1") != "1/2"
