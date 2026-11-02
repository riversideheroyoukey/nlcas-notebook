"""「何階の微分か」を読む語彙は1つ —— NL-GRAMMAR ① の一問目。

**なぜこの一問から始めたか。** 189キー351パターンから機械的に「同じ literal が
3つ以上のキーに現れる」ものを数え、コーパスの行数で並べたところ、頭2つが
``微分``（50行・4キー）と ``derivative``（47行・3キー）だった。523行のうち
``differentiate`` は50行で、``(expression)`` を除けば**最大の操作**である。

**測ったら、補い合いの穴が1つだけあった。** 実パイプラインで33綴りを走らせると
31が正しく、落ちたのは ``二階全微分`` と ``x について 二階全微分`` の2つだけ。原因は
2つの答えの継ぎ目である:

* ``order_clause`` は**漢数字を知っていて**、先読みが ``偏微分|微分`` しか列挙して
  いないので ``全微分`` の前で発火しなかった。
* ``diff_rule`` の予備の順序グループは ``全`` を知っていて、**ASCII 数字しか**読めない。

どちらか一方で足りる綴りは全部通り、**両方が要る漢数字×全微分だけがどこにも
読まれずに落ちていた**。語彙を ``ordinal_token``（当時 ``ja_digit``）/ ``ja_deriv_word`` の2 fragment に
まとめると、予測どおり ``order_clause`` が15綴り・``diff_rule`` が18綴りを獲得し、
**失うものは0**（実測、48綴りの空間を全数比較）。

この検査が押さえるのは結果ではなく**性質**: 順序の語彙を読む2つのキーが、
同じ綴りの集合に同じ答えを返すこと。片方だけを広げれば、また継ぎ目ができる。
"""
import itertools
import re

import pytest

from notebook.runtime.nl_patterns import fragment, nlp

#: The spelling space the two keys must agree on: a digit (ASCII or kanji), a
#: counter word, and the derivative word — every combination a researcher can
#: write, not a list of the ones someone happened to think of.
_DIGITS = ["", "2 ", "2", "二", "3 ", "三", "10 "]
_COUNTERS = ["", "階", "次", "回"]
_DERIVS = ["微分", "偏微分", "全微分"]
SPACE = sorted({f"{d}{c}{k}"
                for d, c, k in itertools.product(_DIGITS, _COUNTERS, _DERIVS)
                if bool(d.strip()) == bool(c)})


def _diff_rule_order_re():
    """``diff_rule``'s own order-and-derivative segment, built from the same
    fragments — this is the half that used to know ``全`` but not kanji."""
    return re.compile(rf'(?:(?:{fragment("ordinal_token")})\s*[階次回]\s*)?'
                      rf'{fragment("ja_deriv_word")}')


class TestTheTwoReadingsAgreeOverTheWholeSpace:
    @pytest.mark.parametrize("spelling", SPACE)
    def test_a_written_order_is_read_by_both(self, spelling):
        """If a digit+counter is written, ``order_clause`` reads it AND
        ``diff_rule``'s segment accepts the phrase.  Before the fold this failed
        for every kanji×全微分 combination."""
        has_order = bool(re.match(r'\s*(?:[0-9]+|[一二三四五六七八九])\s*[階次回]', spelling))
        m = nlp("order_clause", "ja").search(spelling)
        assert bool(m) == has_order, spelling
        assert _diff_rule_order_re().fullmatch(spelling), spelling

    @pytest.mark.parametrize("spelling", SPACE)
    def test_the_derivative_word_never_decides_whether_an_order_is_seen(self, spelling):
        """The bug in one sentence: 全微分 made the order invisible."""
        stem = spelling.replace("全微分", "").replace("偏微分", "").replace("微分", "")
        seen = {k: bool(nlp("order_clause", "ja").search(stem + k))
                for k in _DERIVS}
        assert len(set(seen.values())) == 1, (spelling, seen)


class TestTheSpellingsThatUsedToFall:
    """The two the live pipeline dropped, and their neighbours that did not —
    kept as literals so a regression names the researcher's own spelling."""

    @pytest.mark.parametrize("spelling,order", [
        ("二階全微分", "二"), ("二階偏微分", "二"), ("二階微分", "二"),
        ("三次全微分", "三"), ("2 回全微分", "2"), ("10 階全微分", "10"),
    ])
    def test_order_clause_reads_it(self, spelling, order):
        m = nlp("order_clause", "ja").search(spelling)
        assert m and m.group("digit") == order

    @pytest.mark.parametrize("spelling", ["微分", "偏微分", "全微分"])
    def test_no_order_is_invented_when_none_is_written(self, spelling):
        assert nlp("order_clause", "ja").search(spelling) is None


class TestTheFullInstructionStillReadsVarAndOrder:
    """``diff_rule`` end-to-end on the instruction shape, both particles."""

    @pytest.mark.parametrize("text,var,order", [
        ("E1 を x で 二階全微分して",      "x", "二"),
        ("E1 を x について 二階全微分して",  "x", "二"),
        ("E1 を x で 2 回偏微分して",      "x", "2"),
        ("E1 を x で 三次微分して",        "x", "三"),
    ])
    def test_var_from_diff_rule_and_order_from_order_clause(self, text, var, order):
        m = nlp("diff_rule", "ja").search(text)
        assert m and (m.group("var1") or m.group("var2")) == var
        mo = nlp("order_clause", "ja").search(text)
        assert mo and mo.group("digit") == order

    def test_the_genitive_form_still_reads(self):
        m = nlp("diff_rule", "ja").search("E1 の x に関する全微分")
        assert m and m.group("var2") == "x"


class TestEnglishIsOrderedNotParticled:
    """English states the order in words or with ``N times``, so it is a
    different spelling — the ORDER is stated in words or with ``N times``."""

    @pytest.mark.parametrize("text,group,value", [
        ("Differentiate E1 twice with respect to x", "word", "twice"),
        ("Differentiate E1 with respect to x 3 times", "ntimes", "3"),
    ])
    def test_order_clause_en(self, text, group, value):
        m = nlp("order_clause", "en").search(text)
        assert m and m.group(group) == value

    @pytest.mark.parametrize("text,var", [
        ("Differentiate E1 with respect to x", "x"),
        ("Differentiate E1 w.r.t. x", "x"),
        ("Differentiate E1 by x", "x"),
        ("Take the partial derivative of E1 with respect to t", "t"),
    ])
    def test_diff_rule_en_reads_the_variable(self, text, var):
        m = nlp("diff_rule", "en").search(text)
        assert m and m.group("var1") == var
