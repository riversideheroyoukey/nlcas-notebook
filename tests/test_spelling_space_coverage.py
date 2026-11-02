r"""**文法が受けると言った綴りは、どこかで打たれていること** —— CORPUS-WORKING-ONLY。

指示コーパス（`tests/corpus/instruction_corpus.json`、523行）は差分検証の主力だが、
`docs/scenarios/*.md` から収穫されている。シナリオは**通る指示**で書かれているので、
このコーパスは動かない組み合わせを構造的に含まない。だから
`scripts/measure_instruction_corpus.py` の差分 0 は、しばしば「安全」ではなく
「**その綴りは1行も無い**」であった（四十一・四十二問目で2問連続、2026-09-07）。

ここが押さえる不変条件は1つ:

    `i18n/nl_patterns.json` が受けると言っている綴りは、
    **コーパス・シナリオ・テストのどれかに、入力として書かれている。**

綴りを発明しないのが要点である（[[feedback-the-corpus-input-not-the-imagined-one]]）
——列挙元は**系が自分で書いた表**で、そこに載っているのに誰も打っていないものだけを
数える。2026-09-10 の初回測定（直す前の文法）で 770 件中 **61 件**がそれで、
`tests/test_never_typed_spellings.py` が全部打った結果 **0** になった。
そのうち**2件**は実際に壊れていた（`thrice` と `振っ`。四十八問目）。
3件目（`十次`）は下の**非対称の測定**が見つけた。

**この検査が落ちたときの直し方は「一覧を伸ばす」ではない。** 綴りを1つ足したなら、
それを**打って、何になるべきかを主張する**テストを1つ足す。書くだけなら数は下がるが
何も測っていない——この検査が見ていないものは
`scripts/nl_spelling_coverage.py` の docstring に4つ挙げてある。
"""
import pytest

from scripts.nl_spelling_coverage import (
    coverage, is_language_neutral, language_pair_report,
    neutral_fragment_asymmetry, spelling_space,
)


class TestTheInstrumentIsMeasuringSomething:
    """道具が黙って空になっていたら、以下は全部 vacuous に緑になる。"""

    def test_the_spelling_space_is_populated(self):
        space = spelling_space()
        assert len(space) > 500, f"綴りが {len(space)} 件しか取れていない"

    def test_every_source_is_readable_and_non_empty(self):
        cov = coverage()
        for name in ("corpus", "scenarios", "tests"):
            hit = sum(1 for where in cov.values() if where[name])
            assert hit > 100, f"{name} で当たった綴りが {hit} 件——読めていない疑い"

    def test_a_word_the_grammar_does_not_have_is_not_counted(self):
        """当たり判定が緩すぎないこと（何にでも当たるなら 0 件は無意味）。"""
        assert "zzzznotaspelling" not in spelling_space()


class TestEverySpellingIsTypedSomewhere:
    def test_no_spelling_is_left_untyped(self):
        gaps = sorted(lit for lit, where in coverage().items()
                      if not any(where.values()))
        assert gaps == [], (
            "文法が受けると言っているのに、入力として一度も書かれていない綴り: "
            f"{gaps}\n"
            "綴りを足したなら、それを打って何になるべきかを主張するテストを"
            "足すこと（tests/test_never_typed_spellings.py）。"
            "一覧に載せるだけでは、差分検証はそこを見ない。")


#: 日英で**言語中立な fragment**の引き方が違って**よい**もの。理由が要る。
#: 非対称そのものは欠陥ではない（† は生成側だけ、`neighbour` は英語だけ、
#: `実効磁場` は日本語だけ——4問で3回出た）。残すなら理由を書く、が契約である。
#: **書かれた理由が正しいとは限らない。** `heisenberg_operator_nl` はここに
#: 「en 側は前後の語が境界になるので綴りの表を引く必要がない」という理由つきで
#: 載っていた。実測すると偽で、`\\S+?` は**空白を跨げず** `c(1, u)`——系自身の
#: `tex1` が印刷する形——がどの語順でも当たらなかった（2026-09-11、72通り中18が
#: 壊れていた）。**もっともらしい理由を着けた欠陥**がこの一覧に住めるので、
#: 残すときは理由を書くだけでなく、**その理由が主張していることを測る**。
KNOWN_ASYMMETRY = {
    "diff_rule": (
        "ja だけが `{{ordinal_token}}` を引く。**理由は語順である**（2026-09-24 に"
        "81セルで測り直した）。階数そのものは両言語とも兄弟鍵（`order_clause[ja]` / "
        "`diff_order_ordinal[en]`）が読み、そちらは同じ `ordinal_token` を引く。"
        "違うのは**序数が鍵の一致スパンの内側に立つか外側に立つか**だった:"
        " 英語は `take the third derivative of E1 with respect to x` で"
        "`diff_rule[en]` のスパンが ` derivative of E1 …` から始まる——序数は"
        "**スパンの手前**なので枠が要らない。日本語は `E1 を x で2階微分` が"
        "`E1` から `微分` まで**連続した1つのスパン**なので、序数がその中に立つ。"
        "枠を外すと鍵は一致しなくなり（実測）、その文は別の規則が答えることになる。"
        " **答えは変わらない**——序数の綴り（`2`/`２`/`二`/`三`、`階`/`次`/`回`）× "
        "2つの語順 × 変数の綴り（`x`/`x_1`/`\\bar{y}`/`\\theta`）と、英語の3枠 × "
        "序数・基数・回数副詞・`-order` を並べた81セルで、枠の有無による差は 0。"
        "同じことは兄弟の語順が言っている: 枠を持たない `E1 の x について2階微分して` は"
        "`diff_rule[ja]` に**一致しない**のに階数2で正しく答える。"
        " **残す理由は「specific な鍵が答えるほうが安全」**であって「冗長でない」では"
        "ない。消すなら、落ちた先の規則が同じ答えを出し続けることを固定してから。"
        " **2026-09-13 にここへ書いてあった理由の測定には穴があった。**「測ってある」"
        "（日英18綴り）とあったが、その18綴りは**序数と種別語を掛けていなかった**。"
        "掛けると英語は64綴り中36が**黙って1階**を返していた——真の割れは"
        "`diff_rule` ではなく**先読みの側**で、日本語は `ja_deriv_word`（種別を"
        "合成した1つの表）を読み、英語は素の名詞だけを見ていた。鏡の断片"
        "`en_deriv_word` を作って畳んだ。**理由が名指す測定は、その理由が主張する"
        "軸を全部振ってあること。**"
        " **81セルの測定はもう1件を出した**——変数の綴りを振ると、プライム付きの"
        "`y'` だけが「偏微分か全微分か判別できません」と訊き返す（記号が1つしかない"
        "式に対して、日英とも）。`canonical_variables` の識別子走査に `'` が無く、"
        "`y'^3` の変数が `{y}` と読まれるためで、序数とは無関係の別欠陥"
        "（起票 PRIMED-VAR-NOT-A-VARIABLE）。"),
}


class TestNeutralFragmentsAreAskedByBothLanguages:
    """言語中立な fragment（`ordinal_token`・`symbol` …）は、両方の言語が引く。

    これが 2026-09-10 の四十八問目を見つけた測定である。`order_clause` は
    日本語だけが `{{ordinal_token}}` を引き、英語は `twice|once` と `\\d+` を
    自前で綴っていた。同じ表に載っている `thrice` と「語の数＋times」は
    **どこにも届かず、1階を completed で返して**いた。
    """

    def test_only_the_known_asymmetries_remain(self):
        found = {key for key, _, _ in neutral_fragment_asymmetry()}
        unexpected = found - set(KNOWN_ASYMMETRY)
        assert not unexpected, (
            f"言語中立な fragment を片側しか引いていない鍵: {sorted(unexpected)}。"
            "両方に引かせるか、KNOWN_ASYMMETRY に**理由つきで**足すこと。")

    def test_the_known_list_has_no_stale_entries(self):
        """直したものが一覧に残らないこと（残った理由は次の人を迷わせる）。"""
        found = {key for key, _, _ in neutral_fragment_asymmetry()}
        stale = set(KNOWN_ASYMMETRY) - found
        assert not stale, f"もう非対称でない鍵が一覧に残っている: {sorted(stale)}"

    @pytest.mark.parametrize("name,neutral", [
        ("ordinal_token", True), ("symbol", True), ("op_expr", True),
        ("ja_deriv_word", False), ("compute_verb_en", False),
        ("simplify_verb_ja", False), ("nn_sum_kw_en", False),
    ])
    def test_language_neutrality_is_read_from_the_name(self, name, neutral):
        """「言語中立か」は名前で決まる —— `_ja` / `_en` が付いていないもの。

        判定を表で持つと、fragment を足した人がそこを更新し忘れる。
        """
        assert is_language_neutral(name) is neutral, name


class TestTheThirdReadingIsCalibrated:
    r"""**片方の言語にだけ在る fragment** ——「その問いは片方でだけ畳まれている」の印。

    2026-09-12 に scratch でこれを回して3問を選んだ（宣言の繋辞・代入の動詞・
    解く文の語順）。**盲点は、語幹の綴りが日英で違う鏡を片側扱いすること**で、
    そのままだと上位に偽陽性が並び、次に選ぶ人が空振りする。判別は名前ではなく
    **読む鍵の集合が完全に一致するか**。

    「0 という結果は何も見ていない測定器でも出る」ので、**拾うべきものと拾っては
    ならないものを両方**固定する。
    """

    def test_the_known_mirrors_are_detected(self):
        mirrors, _ = language_pair_report()
        found = {(ja, en) for ja, en, _ in mirrors}
        # **この道具が選んだ問いが、名前の側にも残る。** `ja_solve_var_particle` の
        # 相手は 2026-09-19 まで共有の `en_op_var_head` だった——そして英語の3鍵の
        # うち `solve_single` だけが `in` を**直書き**していて、兄弟2鍵では
        # `compute the solution of E1 in x` が落ち、`solve E1 and E2 in x and y` は
        # **別名走査が答えて `x = 3-y` を completed** で返していた（五十四問目）。
        # solve 一族だけの表 `en_solve_var_head` に畳んだので、鏡の相手が変わる。
        # **`nn_sum_kw_ja` ↔ `nn_kw_en` はもう「名前の違う鏡」ではない。**
        # 2026-09-24（五十五問目）に測ると、名前が違ったのは**中身が違う問いに
        # 答えていた**からだった——ja は「和の名詞」、en は `nearest neighbou?r`
        # という形容詞だけで、`sum` と `lattice sum` は鍵の中に直書き。だから
        # 隣接和の鏡 `neighbour sum` がどこにも無く、2D の文では**英文まるごとが
        # 被和関数**になって `completed` が返っていた。英語側を「和の名詞」に
        # して `nn_sum_kw_en` と名乗らせたので、同じ語幹の**対**になり、この
        # 報告からは正しく消える（消えたことを下の件数が固定する）。
        for pair in [("ja_solve_var_particle", "en_solve_var_head"),
                     ("ja_side_word", "en_side_phrase")]:
            assert pair in found, (pair, sorted(found))
        assert not any(en == "nn_kw_en" for _, en, _ in mirrors), sorted(found)

    def test_the_variable_preposition_pair_is_a_mirror_plus_one(self):
        r"""`ja_op_var_particle` ↔ `en_op_var_head_wide` は 2026-09-18 まで**厳密な鏡**
        （同じ3鍵）だった。英語側を読む鍵が1つ増えて外れている——
        `integrate_inline_formula`、**英語にしか要らない枠**である。

        **測定器は正しく鳴った。** 日本語は字種の変わり目（`_FORMULA_OP_RE`）が式を
        切り出すので、同じ問いに鍵を持つ必要が無い。持たせれば同じ問いに2つ目の答えを
        作ることになる——英語に枠が要るのは、全 ASCII の文でその境界が**原理的に
        発火しない**からで、`_DIFF_INLINE_FORMULA_EN_RE` が在るのと同じ理由である
        （起票 ENGLISH-INLINE-INTEGRAL）。

        **厳密な一致でなくなった以上、差そのものを固定する**——英語側が **2つ**多い。
        2つとも「英語にしか要らない枠」である: `integrate_inline_formula` と、
        2026-09-18 に手書きの部分集合から**この表へ移した** `diff_inline_formula`
        （移す前は `differentiate a x^2 in x` / `for x` / `over x` が枠に当たらず、
        `differentiate(expr='x^2', var='a')` → `0` を completed で返していた——
        6前置詞中3）。**増えたのは修理であって非対称の悪化ではない**が、数は数える。
        """
        _, rest = language_pair_report()
        counts = {name: keys for name, _side, keys in rest}
        assert counts.get("en_op_var_head_wide") == counts.get("ja_op_var_particle", 0) + 2, counts

    def test_a_many_to_one_family_is_not_claimed_as_a_mirror(self):
        r"""`ja_decl_suffix` ＋ `ja_decl_suffix_named` ↔ `en_decl_copula` は 2↔1。
        読む鍵が違うので鏡とは言えず、**理由を書く側の仕事**として残る。"""
        mirrors, rest = language_pair_report()
        assert not any(en == "en_decl_copula" for _, en, _ in mirrors)
        assert "en_decl_copula" in {n for n, _, _ in rest}

    def test_the_inflection_tables_stay_one_sided(self):
        """その言語の形態論そのものなので、対になる断片は在りようがない。"""
        _, rest = language_pair_report()
        names = {n for n, _, _ in rest}
        assert {"ja_suru_infl", "ja_oku_infl"} <= names, sorted(names)

    def test_the_report_is_not_empty(self):
        """**2 は 2026-09-24 に 3 から下がった**（その前は 2026-09-18 に 4 から 3
        ——`ja_op_var_particle` ↔ `en_op_var_head_wide` が「鏡＋1」になったため）。

        下げた理由は**修理**である: `nn_sum_kw_ja` ↔ `nn_kw_en` は名前が違って
        いたのではなく**中身が違う問いに答えていて**、揃えたら同じ語幹の対に
        なった（五十五問目）。**この道具は残り2組**——`ja_side_word` ↔
        `en_side_phrase`（実測 16/16、両辺・両言語とも正しく、左辺への代入は
        両言語で理由つきの断り。名前だけの違い）と `ja_solve_var_particle` ↔
        `en_solve_var_head`（五十四問目で相手が変わった）。
        """
        mirrors, rest = language_pair_report()
        assert len(mirrors) >= 2 and len(rest) >= 10, (len(mirrors), len(rest))
