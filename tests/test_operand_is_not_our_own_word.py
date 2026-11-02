"""②の共通の仕組み —— **オペランドは、システム自身の語ではない**。

`docs/open_items.md` の②の本文は「操作を名指した文が、別の操作で答えられては
ならない」だった。2026-09-05 に実在1046綴りで測ると**そのままでは不変条件に
ならない**:

- 素朴に書くと **88/381 が誤検知**。`Define X as a scalar` は `define` を名指して
  `define_scalar_parameter` が答えるが、これは違反ではなく**特殊化**である。
- しかも同じ日に見つけた欠陥5件のうち、**捕まえるのは1件だけ**だった。

5件に共通していたのは別のことで、**研究者が書いた語がオペランドの欄に入って
いた**——そしてその語はいずれも**この系自身の語彙**だった:

    integrate E1 over x         → var='over'          `over` は文法語
    derive the inverse of E1    → var='inverse'       `inverse` は操作の別名
    … を x および y について解いて   → vars='x, および, y'   `および` は並びの区切り
    substitute x = 2 into E1    → `into` が CAS の関数に   `into` は文法語

**測って捨てた候補が2つある**（どちらも雑音が多すぎた）——「答えた操作 ≠ 名指された
操作」（88/381 の誤検知）と「オペランドはセッションが既知の記号である」（実セッション
の再生で34件。大文字1文字の変数 `Q` `S` `T` `X` や `3x3` に鳴る）。

**この判定が書けるのは、①で語彙を1箇所ずつに畳んだからである。** 同じ日の朝には
文法語は2箇所、区切りは6箇所、計算動詞は ja 9・en 8箇所に散っていて、**1つの検査が
それらに訊くことができなかった**。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_guards import word_operand_warnings
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

V = OperationValidator()


def _warn(op, args, lang="ja"):
    return word_operand_warnings(op, args, validator=V, lang=lang)


#: このセッションで直した欠陥の「直す前」の args。どれも成功を返していた。
DEFECTS = [
    ("十二問目", "integrate_indefinite", {"expr": "E1", "var": "over"}, "over"),
    ("十四問目", "solve_system", {"equations": "E1, E2", "vars": "x, および, y"}, "および"),
    ("十五問目", "differentiate", {"expr": "E1", "var": "inverse"}, "inverse"),
]

#: 正しいオペランド。**大文字1文字の変数と格子の形は鳴らしてはいけない**
#: （「セッションが既知の記号か」で測ったとき、ここが34件鳴って捨てた）。
GOOD = [
    ("integrate_indefinite", {"expr": "E1", "var": "x"}),
    ("differentiate", {"expr": "E1", "var": "Q"}),
    ("differentiate", {"expr": "E1", "var": "S"}),
    ("solve_equation", {"equation": "E1", "var": "X"}),
    ("taylor_expand", {"expr": "E1", "var": "lambda", "point": "0", "order": "3"}),
    ("define_scalar_parameter", {"name": "theta", "domain": "real", "sign": "positive"}),
    ("multiply_both_sides", {"expr": "E1", "factor": "2", "side": "left"}),
    ("expand_neighbor_sum", {"expr": "E1", "boundary": "periodic", "lattice_shape": "3x3"}),
    ("substitute", {"expr": "E1", "lhs": "x", "rhs": "a*x+b"}),
    ("substitute", {"expr": "E1", "lhs": "T", "rhs": "88"}),
    ("substitute", {"expr": "E1", "lhs": "x", "rhs": "foo"}),
    ("divergence", {"expr": "dBdt"}),
]


class TestItCatchesTheMeasuredDefects:
    @pytest.mark.parametrize("label,op,args,word", DEFECTS)
    def test_the_defect_shape_warns(self, label, op, args, word):
        w = _warn(op, args)
        assert w, (label, op, args)
        assert word in w[0], (label, w)

    @pytest.mark.parametrize("label,op,args,word", DEFECTS)
    def test_both_languages_say_it(self, label, op, args, word):
        for lang in ("ja", "en"):
            assert _warn(op, args, lang), (label, lang)

    def test_it_names_which_kind_of_word(self):
        """どの語彙のものか言う —— 「未知の記号」より研究者に多くを伝える。"""
        assert "文法" in _warn("integrate_indefinite", {"var": "over"})[0]
        assert "区切り" in _warn("solve_system", {"vars": "x, および, y"})[0]
        assert "操作の名前" in _warn("differentiate", {"var": "inverse"})[0]


class TestItIsSilentOnRealOperands:
    @pytest.mark.parametrize("op,args", GOOD)
    def test_no_warning(self, op, args):
        assert _warn(op, args) == [], (op, args)

    def test_an_undeclared_multiletter_name_is_not_our_word(self):
        """`foo` は未宣言だが**この系の語ではない** —— ここは鳴らさない。

        「未知の記号」で鳴らす案は実セッションの再生で34件鳴って捨てた。既存の
        未定義記号検出が式について同じ判定をしており、そちらの答えを操作の応答に
        配線するほうが筋（open_items の残件）。
        """
        assert _warn("substitute", {"expr": "E1", "lhs": "x", "rhs": "foo"}) == []


class TestTheFourVocabulariesAreTheOnesTheSystemUses:
    """判定は①で1箇所に畳んだ語彙に訊く —— 語の一覧をここに写さない。"""

    #: `and` は**文法語でもあり区切りでもある**ので、種類は先に当たったほうになる
    #: （どちらでも研究者に伝わることは同じ）。種類を1つに決められる語だけ固定する。
    @pytest.mark.parametrize("word,kind", [
        ("over", "文法"), ("into", "文法"), ("compute", "文法"),
        ("および", "区切り"), ("求めて", "動詞"),
        ("inverse", "操作の名前"), ("eigenvalues", "操作の名前"),
    ])
    def test_a_word_of_ours_is_recognised(self, word, kind):
        w = _warn("substitute", {"expr": "E1", "lhs": "x", "rhs": word})
        assert w, word
        assert kind in w[0], (word, w[0])

    @pytest.mark.parametrize("word", ["and", "&", "、", "w.r.t."])
    def test_a_word_in_two_of_our_vocabularies_still_warns(self, word):
        assert _warn("substitute", {"expr": "E1", "lhs": "x", "rhs": word}), word


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_the_warning_reaches_the_response(lang):
    """画面まで —— **操作の結果を返す応答18箇所のうち17箇所が `warnings` を
    渡していなかった**（実測 2026-09-05）。答えは `run_interceptors` に1つ置いた
    （七問目が `operation_label` で採ったのと同じ場所・同じ理由）。
    """
    nb.start_engine(title="own_word", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        _run(eng, sup, "a をスカラーとする" if lang == "ja" else "Define a as a scalar")
        _run(eng, sup, "f = x^2 + a*x")
        # `over` を変数として渡す（十二問目の直す前の形）
        step, _val = eng.run_operation("integrate_indefinite",
                                       {"expr": "E1", "var": "over"})
        out = eng.expressions.get(step.output_refs[0])
        assert out.warnings, "結果の式に警告が付いていない"

        # **インターセプタの実経路を通す。** `_carry_result_warnings` を直に呼ぶと
        # 呼び出し側の配線が外れても落ちない（変異で確認した本物の弱さ）。
        from ui.interpret_registry import (run_interceptors, InterceptorContext,
                                            Interceptor)

        async def _fake(ctx):
            return {"status": "completed", "operation": "integrate_indefinite",
                    "expr_id": out.display_id, "registered_expressions": []}

        ctx = InterceptorContext(engine=eng, loop=asyncio.get_event_loop(),
                                 lang=lang, text="dummy", registered=[])
        resp = asyncio.get_event_loop().run_until_complete(
            run_interceptors(ctx, [Interceptor(name="fake_route", run=_fake)]))
        assert resp is not None
        assert resp.get("warnings") == out.warnings, resp
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_a_correct_operation_carries_no_warning(lang):
    nb.start_engine(title="own_word_ok", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        _run(eng, sup, "a をスカラーとする" if lang == "ja" else "Define a as a scalar")
        _run(eng, sup, "f = x^2 + a*x")
        r = _run(eng, sup, "E1 を x で微分して" if lang == "ja"
                 else "differentiate E1 with respect to x")
        assert r.get("status") == "completed", r
        assert not r.get("warnings"), r.get("warnings")
    finally:
        nb.stop_engine()
