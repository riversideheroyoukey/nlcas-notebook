"""テイラー展開の言い方と、展開中心について何を伝えるか。

十二問目（`tests/test_which_variable_particle.py`）の測定で出た族。実測
2026-09-05、``E1 = f = exp(x)``——**多項式で測ってはいけない**: 代数 ``expand``
は多項式を入力のまま返すので、間違った操作が「もっともらしく正しく」見える。

    expand E1 in x to order 3              → **expand {expr: E1}** → `%e^x` を completed
    expand E1 in x around 0 to order 3     → 同上
    series expand E1 in x to order 3       → 同上
    expand E1 to third order in x          → 同上
    taylor expand E1 … around x=0 …        → point='x=0' → **`Maxima evaluation error`**
    E1 を x に関して 0 のまわりで 3 次まで…      → **order='0', point='3'**（入れ替わる）
    E1 を x について 3次でテーラー展開して        → どのパターンにも当たらない

英語で4綴りが落ちていたのは、``taylor_expand_nl[en]`` が**文頭に `taylor` /
`maclaurin` を要求**していたため。物理で普通に書く `expand … to order 3` は
別名索引の `expand`（代数展開）に落ち、次数も変数も捨てられて成功が返る。
判別子は**次数の指定**——代数展開に次数は無い。

**そして「展開中心が書かれていないとき何を伝えるか」に答えが2つあった。**
パイプラインの 3c-pre / 3c（supervisor 経路）は「マクローリン展開とします。
実行しますか？」と訊き、``interpret_integral`` のインターセプタは
``point = (m.group("point") or "0")`` と**黙って 0 を入れて**いた。args は
両者同一で、「中心が書かれていない」という事実は読んだ層に依らない。
判断は `interpret_common.taylor_center_note` に1つだけ置いた。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_common import number_word, pick_group, taylor_center_note
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


#: (言語, 綴り, 中心が書かれているか)。日英そろえてある。
SPELLINGS = [
    ("ja", "E1 を x について 0 のまわりで 3 次までテイラー展開して", True),
    ("ja", "E1 を x について 0 のまわりで 3 次までマクローリン展開して", True),
    ("ja", "E1 を x に関して 0 のまわりで 3 次までテイラー展開して", True),
    ("ja", "E1 を x について 3 次までテイラー展開して", False),
    ("ja", "E1 を x について 3次でテーラー展開して", False),
    ("ja", "E1 を x について 3 次のテイラー展開をして", False),
    ("ja", "E1 を x のまわりで 3 次までテイラー展開して", False),
    ("en", "taylor expand E1 in x around 0 to order 3", True),
    ("en", "taylor expand E1 with respect to x around x=0 to order 3", True),
    ("en", "expand E1 in x around 0 to order 3", True),
    ("en", "taylor expand E1 with respect to x to order 3", False),
    ("en", "expand E1 in x to order 3", False),
    ("en", "series expand E1 in x to order 3", False),
    ("en", "expand E1 to third order in x", False),
]

WANT_ARGS = {"expr": "E1", "var": "x", "point": "0", "order": "3"}


class TestThePatternReadsAllFourFields:
    @pytest.mark.parametrize("lang,text,has_center", SPELLINGS)
    def test_expr_var_point_order(self, lang, text, has_center):
        m = nlp("taylor_expand_nl", lang).match(text)
        assert m is not None, f"[{lang}] どのパターンにも当たらない: {text!r}"
        assert pick_group(m, "expr") == "E1", (text, m.groupdict())
        assert pick_group(m, "var") == "x", (text, m.groupdict())
        assert number_word(pick_group(m, "order")) == 3, (text, m.groupdict())
        if has_center:
            assert pick_group(m, "point") is not None, (text, "中心を読み落とした")

    @pytest.mark.parametrize("text", [
        "expand E1",
        "expand E1 into components",
        "E1 を展開して",
    ])
    def test_a_plain_expand_is_not_a_series(self, text):
        """次数が書かれていなければ代数展開のまま —— 判別子は次数の指定。"""
        for lang in ("ja", "en"):
            assert nlp("taylor_expand_nl", lang).match(text) is None, (lang, text)


class TestTheCenterIsOneAnswer:
    """`x=0` は `0` である。書かれていなければそう伝える。"""

    @pytest.mark.parametrize("written,point,has_note", [
        (None, "0", True),
        ("", "0", True),
        ("0", "0", False),
        ("x=0", "0", False),
        ("x = 0", "0", False),
        ("5", "5", True),
        ("-1", "-1", True),
        ("%pi/2", "%pi/2", True),
    ])
    def test_the_reading_and_the_note(self, written, point, has_note):
        got_point, note = taylor_center_note(written, "ja")
        assert got_point == point, (written, got_point)
        assert bool(note) is has_note, (written, note)

    def test_both_languages_have_the_words(self):
        for lang in ("ja", "en"):
            assert taylor_center_note(None, lang)[1]
            assert taylor_center_note("5", lang)[1]


class TestANumberWordIsReadInOnePlace:
    """漢数字の表は `interpret_pipeline` と `interpret_equation_intercepts` に
    **別々の写しとして**あり、英語の序数語で3つめが要った。"""

    @pytest.mark.parametrize("word,want", [
        ("3", 3), ("３", 3), ("三", 3), ("third", 3), ("Third", 3),
        ("twice", 2), ("十", 10), ("tenth", 10), ("1", 1),
    ])
    def test_it_reads(self, word, want):
        assert number_word(word) == want

    @pytest.mark.parametrize("word", ["x", "", None, "order"])
    def test_it_does_not_guess(self, word):
        assert number_word(word) is None


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(scope="module", params=["ja", "en"])
def session(request):
    """``f = exp(x)`` —— 級数展開でないと違う答えになる式で測る。"""
    lang = request.param
    nb.start_engine(title="taylor_family", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "f = exp(x)")
    yield lang, eng, sup
    nb.stop_engine()


def test_every_spelling_reaches_taylor_expand(session):
    """中心を書けば計算まで、書かなければ**どちらの経路でも**同じ確認が返る。"""
    lang, eng, sup = session
    for spell_lang, text, has_center in SPELLINGS:
        if spell_lang != lang:
            continue
        resp = _run(eng, sup, text)
        if has_center:
            assert resp.get("status") == "completed", (text, resp.get("status"),
                                                       resp.get("error"))
            got = (resp.get("canonical") or "").replace(" ", "")
            assert "x^3" in got and "x^2" in got, (text, got)
        else:
            assert resp.get("status") == "needs_confirmation", (text, resp.get("status"))
            assert resp.get("operation") == "taylor_expand", (text, resp.get("operation"))
            assert resp.get("args") == WANT_ARGS, (text, resp.get("args"))
            assert resp.get("confirmation_prompt"), (text, "何を仮定したか言っていない")
