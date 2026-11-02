r"""序数の綴りは1つの読み —— 答えは `number_word` に既にあったのに、誰も訊かなかった。

実測 2026-09-06。`order` は6つの鍵に現れ、それぞれ**別の量**（微分の階数・級数の
次数・N重交換子・モーメント・展開の深さ）を指す。ところが「序数はどう書かれるか」
という**同じ問い**に、鍵ごとに違う文字クラスで答えていた:

| 鍵 | 日本語の数字 | 英語の序数語 |
|---|---|---|
| `diff_order_ordinal` | `[0-9]+` **半角のみ** | 数字のみ |
| `f1_order_n` / `f1_spec_depth` / `spin_expectation` | `[0-9０-９]+`（全角可・漢数字不可） | 数字のみ |
| `taylor_expand_nl` / `series_expand_no_var` | `{{ja_digit}}`（全角・漢数字可） | `first…tenth` 可 |

**答えは `ui/interpret_common.number_word` に既にあった**——`3`/`３`/`三`/`third` を
読み、読めなければ None で推測しない。訊いていたのは `order_clause` だけで、
他は自前の文字クラス＋`int()` だった。だから語の序数はどこにも届かず、
**4綴りが silent-wrong**（すべて `completed`）:

| 打った文 | 直す前 | 正しい |
|---|---|---|
| `take the third order derivative of E1 with respect to x` | `df/dx = 5x^4`（**1階**） | `d³f/dx³ = 60x²` |
| `Find the second order commutator of S^x_2 …` | `[H, S^x_2]`（**1重**） | `[H, [H, S^x_2]]` |
| `…高温展開を一般式で四次まで求めて` | 既定の3次（⟨H⁴⟩ が無い） | ⟨H⁴⟩ まで |
| `…up to fourth order` | 同上 | 同上 |

畳み方は2段:（1）綴りは共有 fragment `ordinal_token`、（2）**変換は全消費者が
`number_word`**（`int()` を消した）。`3rd` / `2nd` / `4th` の形も `number_word` が
読むようにしたので、鍵の側で `(?:st|nd|rd|th)?` を綴る必要も無くなった。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_common import number_word
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

CM = ["classical_mechanics"]
CD = ["condensed_matter"]

#: 4族 × 5綴り（半角・全角・漢数字・数字序数・語序数）。
DERIV = [("ja", "E1 を x で3階微分して"), ("ja", "E1 を x で３階微分して"),
         ("ja", "E1 を x で三階微分して"),
         ("en", "take the 3rd order derivative of E1 with respect to x"),
         ("en", "take the third order derivative of E1 with respect to x")]
NFOLD = [("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の2重交換子を求めて"),
         ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の２重交換子を求めて"),
         ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の二重交換子を求めて"),
         ("en", "Find the 2nd order commutator of S^x_2 for the Heisenberg model on a 3-site chain"),
         ("en", "Find the second order commutator of S^x_2 for the Heisenberg model on a 3-site chain")]
HIGHT = [("ja", "ハイゼンベルク模型の高温展開を一般式で4次まで求めて"),
         ("ja", "ハイゼンベルク模型の高温展開を一般式で４次まで求めて"),
         ("ja", "ハイゼンベルク模型の高温展開を一般式で四次まで求めて"),
         ("en", "Find the high-temperature expansion of the Heisenberg model in general form up to 4th order"),
         ("en", "Find the high-temperature expansion of the Heisenberg model in general form up to fourth order")]


def _ask(dom, lang, prep, text):
    nb.start_engine(title="or", mode="standard", language=lang, domains=dom)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=dom, primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in prep:
            run(p)
        r = run(text)
        regs = r.get("registered_expressions") or []
        return r, (r.get("latex") or (regs[-1].get("latex") if regs else "") or "")
    finally:
        nb.stop_engine()


class TestEverySpellingOfAnOrderIsRead:
    @pytest.mark.parametrize("lang,text", DERIV)
    def test_the_derivative_order(self, lang, text):
        r, body = _ask(CM, lang, ["f = x^5"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "60" in body, (text, body)          # 1階なら 5 x^4

    @pytest.mark.parametrize("lang,text", NFOLD)
    def test_the_nfold_commutator(self, lang, text):
        r, body = _ask(CD, lang, [], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert body.count("[H,") >= 2, (text, body)   # 2重なら [H, [H, …]]

    @pytest.mark.parametrize("lang,text", HIGHT)
    def test_the_high_temperature_order(self, lang, text):
        r, body = _ask(CD, lang, [], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "H^4" in body, (text, body)


#: 具体格子のモーメント（`_parse_spin_spectral`）。一般式は上限2次で
#: 断られるので、次数が効くのは具体格子の枝である（三十二問目）。
#: **数は 8 にしてある**——`次` は次数なので `4次` は既定（2段＝$M_4$ まで）と
#: 同じ答えになり、綴りが届いたかを区別できない（2026-09-10、五十問目）。
#: `8次` なら 4段＝$M_8$ まで出るので、既定と区別がつく。
CONCRETE_MOMENTS = [
    ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の8次までのモーメントを求めて"),
    ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の８次までのモーメントを求めて"),
    ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の八次までのモーメントを求めて"),
    ("en", "Find the moments up to 8th order of S^x_2 for the Heisenberg model on a 3-site chain"),
    ("en", "Find the moments up to eighth order of S^x_2 for the Heisenberg model on a 3-site chain"),
]


#: 「何**回**微分するか」——2026-09-10（四十八問目）に測った綴り空間。
#: 日本語は `一回`〜`十回` が全部通っていたのに、英語は `once` / `twice` と
#: **半角数字＋times** だけで、`thrice`（同じ表に載っている）と語の基数が
#: どこにも届かず **1階を `completed`** で返していた。
TIMES = [
    ("ja", "E1 を x で 3 回微分して"), ("ja", "E1 を x で 三回微分して"),
    ("ja", "E1 を x で ３回微分して"),
    ("en", "differentiate E1 with respect to x 3 times"),
    ("en", "differentiate E1 with respect to x thrice"),
    ("en", "differentiate E1 with respect to x three times"),
]


class TestHowManyTimesIsOneReadingToo:
    r"""倍数副詞と「数＋回 / N times」も、同じ1つの表から読む。"""

    @pytest.mark.parametrize("lang,text", TIMES)
    def test_three_times_is_the_third_derivative(self, lang, text):
        r, body = _ask(CM, lang, ["f = x^5"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "60" in body, (text, body)          # 1階なら 5 x^4

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を x で 十回微分して"),
        ("en", "differentiate E1 with respect to x ten times"),
    ])
    def test_ten_times_exhausts_a_fifth_power(self, lang, text):
        r, body = _ask(CM, lang, ["f = x^5"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert body.strip().endswith("0"), (text, body)


#: 級数展開の次数。ここは逆向きの片側欠落で、**日本語だけ**が `十` を持たない
#: 表（`ja_digit`、`一`〜`九`）を引いていた。`十階微分` は通るのに
#: `十次まで展開` は「解釈できません」だった（実測 2026-09-10）。
EXPANSION_ORDER = [
    ("ja", "E1 を x について 3 次まで展開して", "3"),
    ("ja", "E1 を x について 十次まで展開して", "10"),
    ("ja", "E1 を 十次まで展開して", "10"),
    ("en", "expand E1 with respect to x up to order ten", "10"),
    ("en", "expand E1 up to order tenth", "10"),
]


class TestTheExpansionOrderReadsTheSameTable:
    @pytest.mark.parametrize("lang,text,want", EXPANSION_ORDER)
    def test_the_order_reaches_the_operation(self, lang, text, want):
        r, _ = _ask(CM, lang, ["f = exp(x)"], text)
        assert r.get("status") != "failed", (text, r.get("error"))
        assert (r.get("args") or {}).get("order") == want, (text, r.get("args"))


class TestTheConcreteLatticeDepthReadsThemToo:
    r"""`_parse_spin_spectral` の次数 —— ここも同じ問いに答えている。

    **この検査が見ているのは「数の綴りが届くか」で、単位ではない。**
    2026-09-10（五十問目）に `次` が次数だと決まったので、区別できる数
    （8次＝4段＝$M_8$）に置き換えてある。単位そのものは
    `tests/test_requested_order_is_honoured.py` が押さえる。
    """

    @pytest.mark.parametrize("lang,text", CONCRETE_MOMENTS)
    def test_the_moment_depth(self, lang, text):
        r, body = _ask(CD, lang, [], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        # 8次 → 4段 → M_8 まで（既定の2段なら M_4 で止まる）。
        assert "M_{8}" in body, (text, body)


class TestTheAnswerLivesInOneFunction:
    @pytest.mark.parametrize("token,want", [
        ("3", 3), ("３", 3), ("三", 3), ("third", 3), ("THIRD", 3), ("3rd", 3),
        ("2nd", 2), ("4th", 4), ("fourth", 4), ("十", 10), ("tenth", 10),
        ("twice", 2), ("once", 1),
        # 2026-09-10（四十八問目）: 倍数副詞の3語目と、英語の基数。
        # `three times` は日本語の `三回` と同じ普通の綴りで、
        # 直す前は **1階** を `completed` で返していた。
        ("thrice", 3), ("three", 3), ("ten", 10), ("one", 1),
    ])
    def test_number_word_reads_every_spelling(self, token, want):
        assert number_word(token) == want, token

    @pytest.mark.parametrize("token", ["x", "", "th", "thir", "0x3"])
    def test_it_refuses_rather_than_guesses(self, token):
        """読めなければ None——推測しない。"""
        assert number_word(token) is None, token

    @pytest.mark.parametrize("token", ["3", "３", "三", "third", "3rd", "十",
                                       "thrice", "three", "ten"])
    def test_the_shared_fragment_matches_what_number_word_reads(self, token):
        """**綴りの表と変換の表が食い違わないこと。** 片方だけ広げると、
        当たるのに読めない（あるいはその逆の）綴りができる。"""
        import re
        rx = re.compile("^(?:" + fragment("ordinal_token") + ")$", re.I)
        assert rx.match(token), token
        assert number_word(token) is not None, token

    def test_every_word_in_the_table_is_read(self):
        """**枝を数え上げる** —— 語を1つ足したときに、片方だけ広がるのを止める。

        上の parametrize は手で書いた代表例なので、語を足しても行が増えない
        （[[feedback-verify-the-guard-catches-its-own-defect]] ⑥）。ここは
        fragment の alternation を**全部**取り出して、`number_word` が
        1つ残らず読めることを見る。表は2つ（正規表現と dict）あってよいが、
        **食い違ってはいけない**——`thrice` はまさにそこから漏れた。
        """
        import re
        # 末尾の `(?:st|nd|rd|th)?` は**数字に付く接尾辞**で、それ自体は数では
        # ない（`3rd` を読むのは `_ORDINAL_SUFFIX_RE` の側）。語を数える前に外す。
        src = re.sub(r"\(\?:st\|nd\|rd\|th\)\?$", "", fragment("ordinal_token"))
        assert "st|nd" not in src, "接尾辞が外れていない"
        words = sorted(set(re.findall(r"[A-Za-z]+", src)))
        assert len(words) >= 20, f"枝が {len(words)} 語しか取れていない: {words}"
        unreadable = [w for w in words if number_word(w) is None]
        assert unreadable == [], unreadable

    #: 手で並べた一覧が、まさにこの欠陥の温床だった（2026-09-10）。上の表は
    #: 9行あって `("order_clause", "ja")` は在り **`("order_clause", "en")` が
    #: 無かった**——そして英語側だけが `twice|once` を自前で綴り、`thrice` と
    #: 「語の数＋times」が届かず **1階を `completed`** で返していた。
    #: 一覧ではなく**性質**で書く: 片方の言語が表を引いているなら、もう片方も引く
    #: （例外は `tests/test_spelling_space_coverage.py` に理由つきで置く）。
    def test_no_key_reads_the_shared_fragment_in_only_one_language(self):
        from scripts.nl_spelling_coverage import neutral_fragment_asymmetry
        offenders = [(key, only_ja, only_en)
                     for key, only_ja, only_en in neutral_fragment_asymmetry()
                     if "ordinal_token" in only_ja + only_en]
        assert offenders == [("diff_rule", ["ordinal_token"], [])], offenders

    @pytest.mark.parametrize("key,lang", [
        ("diff_order_ordinal", "ja"), ("diff_order_ordinal", "en"),
        ("f1_order_n", "ja"), ("f1_order_n", "en"),
        ("f1_spec_depth", "ja"), ("f1_spec_depth", "en"),
        ("spin_expectation", "ja"), ("spin_expectation", "en"),
        ("order_clause", "ja"), ("order_clause", "en"),
        ("taylor_expand_nl", "ja"), ("taylor_expand_nl", "en"),
        ("series_expand_no_var", "ja"), ("series_expand_no_var", "en"),
    ])
    def test_every_key_reads_the_shared_fragment(self, key, lang):
        import json
        import pathlib
        src = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())[key][lang]
        assert "{{ordinal_token}}" in src, (key, lang, src)

    @pytest.mark.parametrize("key,lang,text,want", [
        ("diff_order_ordinal", "en", "third order", 3),
        ("diff_order_ordinal", "ja", "三階", 3),
        ("f1_order_n", "en", "second order commutator", 2),
        ("f1_order_n", "ja", "二重交換子", 2),
        ("f1_spec_depth", "en", "up to fourth order", 4),
        ("f1_spec_depth", "ja", "四次", 4),
    ])
    def test_the_match_hands_a_token_number_word_can_read(self, key, lang, text, want):
        m = nlp(key, lang).search(text)
        assert m is not None, (key, text)
        tok = next((g for g in m.groups() if g), None)
        assert number_word(tok) == want, (key, text, tok)
