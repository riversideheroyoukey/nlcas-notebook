r"""EN-BARE-NOUN-AS-FORMULA（2026-09-24）——`<操作> the <名詞> of <式>` の名詞。

操作の別名でない名詞が属格の頭に立つと、**書かれた式が消える**か、**英単語が
被和関数の因子になる**。どちらも `completed`（実測、HEAD）:

| 打った文 | 返っていたもの |
|---|---|
| `Expand the nearest-neighbor of x^2 - 1` | `expand(expr='nearest-neighbor')` ——研究者の `x^2 - 1` が消える |
| `expand the widget of x^2 - 1` | `widget of x^2 - 1` を1つの式として登録 → **`of*widget*(x^2-1)`** |
| `simplify the foo of …` / `factor the gizmo of …` | 同上 |
| `widget of x^2 - 1 を展開して`（日本語の文） | 同上——字種の変わり目が切る「式」の頭に属格が立つ |

**根は2つ、形は1つ。** `of` は**2文字**なので `_nl_word_count`（3文字以上を数える）
に見えず、`widget of x^2 - 1` は「英単語1語＋数式」として英語の「動詞＋式」の規則を
通る。語数で降りる綴り（`nearest-neighbor of …`）は別名走査に落ち、そちらが
`expr='nearest-neighbor'` を作る。だから**断りは1つ**で両方を覆う。

**判別は操作の別名そのもの**——`the curl of the curl of A` の内側や
`expand the product of E1 and E2` は実在の綴りで、そのまま通る。

**着手条件（起票時に自分で書いた）を先に実行した**: コーパス611行に対して、この
述語が発火するのは **0 行**。形に届く行は1つ（`the curl of the curl of A`）で、
`curl` は別名なので除かれる。CORPUS-WORKING-ONLY なので 0 は安全の結果にすぎない。

**案内する2つの道は実測した**: `expand x^2 - 1` は `x^2-1` を計算し、
`widget = x^2 - 1` のあとの `expand widget` も計算する。

日本語の `x^2 - 1 の widget を展開して` は**式が正しく切れており**、名詞が黙って
捨てられる——落ちているのは別名走査が置き場のないトークンを捨てる一般則で、
ここでは断らない（別起票 LEFTOVER-NOUN-DROPPED）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_boundary import genitive_head_not_an_operation
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import _names_an_operation
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _run(lang, setup, line, domain="classical_mechanics"):
    async def go():
        nb.start_engine(title="gen", mode="standard", language=lang, domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for text in list(setup) + [line]:
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
            return resp, [(k, v.canonical) for k, v in engine.expressions.items()]
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheHeadIsReadFromTheAliasTable:
    """判別は「その語が操作を名指しているか」——語彙の手書き一覧ではない。"""

    @pytest.mark.parametrize("formula,want", [
        ("widget of x^2 - 1",           "widget"),
        ("nearest-neighbor of x^2 - 1", "nearest-neighbor"),
        ("square of u",                 "square"),
        # 別名なら属格の頭でも通す——実在の綴り
        ("curl of A",                   None),
        ("product of E1 and E2",        None),
        # 属格ではないもの
        (r"\vec{A}",                    None),
        ("a x^2 + b x + c",             None),
        ("of x",                        None),
    ])
    def test_head(self, formula, want):
        p = _names_an_operation(NotebookEngine())
        assert genitive_head_not_an_operation(formula, p) == want, formula


class TestTheNounIsNamedInTheRefusal:
    @pytest.mark.parametrize("line,noun", [
        ("Expand the nearest-neighbor of x^2 - 1", "nearest-neighbor"),
        ("expand the widget of x^2 - 1",           "widget"),
        ("simplify the foo of x^2 + 1",            "foo"),
        ("factor the gizmo of x^2 - 1",            "gizmo"),
        ("expand the widget of A",                 "widget"),
    ])
    def test_english(self, line, noun):
        resp, exprs = _run("en", [], line)
        assert resp.get("status") == "failed", (line, resp)
        assert noun in (resp.get("error") or ""), resp.get("error")
        # **式は登録されない**——英単語が因子になっていた場所である
        assert exprs == [], exprs

    def test_the_japanese_sentence_with_an_english_genitive(self):
        r"""`widget of x^2 - 1 を展開して` ——字種の変わり目が切る「式」の頭。"""
        resp, exprs = _run("ja", [], "widget of x^2 - 1 を展開して")
        assert resp.get("status") == "failed", resp
        assert "widget" in (resp.get("error") or ""), resp.get("error")
        assert exprs == [], exprs


class TestAShortHeadIsASymbol:
    r"""`f of x` の `f` は研究者の**関数**であって、綴り間違いの操作名ではない。

    変異試験で最短長を1文字まで下げても、当時のどのセルも落ちなかった——どちらも
    断りになるからである。measured: 今は「必要な引数が不足」で落ち、下げるとこの
    断りが「`f` という操作を知りません」と**偽を述べる**。だから3文字は決定であり、
    `_en_verb_formula_re` が別名に課している `len >= 3` と同じ理由に立つ。
    """

    @pytest.mark.parametrize("line,noun", [("simplify f of x", "f"),
                                           ("simplify ab of x", "ab")])
    def test_a_short_head_is_not_called_an_unknown_operation(self, line, noun):
        resp, _ = _run("en", [], line)
        assert resp.get("status") == "failed", resp
        assert "no operation called" not in (resp.get("error") or ""), resp.get("error")

    def test_the_head_reader_declines_a_short_head(self):
        p = _names_an_operation(NotebookEngine())
        assert genitive_head_not_an_operation("f of x", p) is None
        assert genitive_head_not_an_operation("ab of x", p) is None


class TestTheEquationSideIsReadElsewhere:
    r"""`simplify the RHS of E1` の `RHS` は操作でも量でもなく**等式の辺**。

    **綴り空間で測って出た**——`RHS` はテストのリテラルにしか無く、コーパス差分 0 は
    何も言っていなかった（フルスイートが2件で止めた）。除く理由は語彙ではなく
    **順序**で、この形に当たる読みのうち等式の族だけがこの割り込み口より**後ろ**で
    走る（変数変換・境界条件などは前で claim される、実測）。
    """

    @pytest.mark.parametrize("line,want", [
        ("simplify the RHS of E1", "x^2-1"),
        ("expand the LHS of E1",   None),
        ("simplify the right-hand side of E1", "x^2-1"),
    ])
    def test_the_side_family_still_answers(self, line, want):
        resp, exprs = _run("en", ["(a+b)^2 = x^2 - 1"], line)
        assert resp.get("status") == "completed", (line, resp.get("error"))
        assert "no operation called" not in (resp.get("error") or "")
        if want:
            assert exprs[-1][1] == want, exprs

    def test_the_head_reader_declines_a_side(self):
        p = _names_an_operation(NotebookEngine())
        for f in ("RHS of E1", "LHS of E1", "right-hand side of E1", "left side of E1"):
            assert genitive_head_not_an_operation(f, p) is None, f


class TestTheRoadsInTheRefusalWork:
    """断り文が名指す道は実測したものであること。"""

    def test_dropping_the_genitive(self):
        resp, exprs = _run("en", [], "expand x^2 - 1")
        assert resp.get("status") == "completed", resp
        assert exprs[-1][1] == "x^2-1", exprs

    def test_writing_the_value_first(self):
        resp, exprs = _run("en", ["widget = x^2 - 1"], "expand widget")
        assert resp.get("status") == "completed", resp
        assert exprs[-1][1].endswith("x^2-1"), exprs


class TestTheSpellingsThatWorkedStillWork:
    """別名が属格の頭に立つ実在の綴りと、属格でない綴り。"""

    @pytest.mark.parametrize("lang,setup,line,domain,want", [
        ("en", [], "expand (x+1)^2", "classical_mechanics", "x^2+2*x+1"),
        ("en", [], "simplify a x^2 + b x + c", "classical_mechanics", None),
        ("ja", [], "x^2 - 1 を展開して", "classical_mechanics", "x^2-1"),
    ])
    def test_plain(self, lang, setup, line, domain, want):
        resp, exprs = _run(lang, setup, line, domain)
        assert resp.get("status") == "completed", (line, resp.get("error"))
        if want is not None:
            assert exprs[-1][1] == want, exprs

    @pytest.mark.parametrize("lang,line", [
        ("en", r"the curl of \vec{A}"),
        ("en", "the norm of A"),
        ("ja", "A の回転を求めて"),
    ])
    def test_a_declared_field(self, lang, line):
        resp, _ = _run(lang, ["A をベクトル場とする" if lang == "ja"
                              else "declare A as a vector field"],
                       line, "electromagnetism")
        assert resp.get("status") == "completed", (line, resp.get("error"))
