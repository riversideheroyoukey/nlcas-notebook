r"""欄の値でも、**語で書かれた数は数である** —— 起票 FIELD-VALUE-OR-NAME。

四十九問目で「文全体が語で書かれた数なら、それは数である」を LaTeX→正準形の扉
（`to_canonical`）に置いた。値の欄はほとんどそこを通るので大半が同時に直ったが、
**1綴りだけ残った**（実測 2026-09-10）:

    substitute x = zero into E1   →  a*zero+b   completed   ← 記号 `zero` を代入
    substitute zero for x in E1   →  b          ✔（同じ操作の別の綴り）
    evaluate E1 at x = zero       →  b          ✔
    E1 の x をゼロに置換して        →  b          ✔

インターセプタが読む綴りは値を**書かれたテキスト**として扱ってから扉に渡すが、
別名走査に落ちる綴りは `map_tokens_to_fields` が欄を**位置で**埋め、そのまま IR を
通って `zero` が正当な `Symbol` になる。**この経路にだけ答えが無かった。**

**起票の見立ては外れていた。** 「スキーマに欄の種別（値／名前／式）を持たせる」と
書いたが、**要る半分は既に在った**——`IDENTIFIER_KEYS` が「識別子の名前を運ぶ欄」
（数式として解析しない欄）を名指しており、`name` はそこに在る。だから
`declare zero as a scalar` は動かない。欄の種別を新しく作る必要は無かった。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.expr_ref_fields import IDENTIFIER_KEYS
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["differential_equations"]


def _run(lang, setup, text):
    nb.start_engine(title="fvn", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")
        loop = asyncio.get_event_loop()
        for line in setup:
            loop.run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=line)]), eng, sup))
        r = loop.run_until_complete(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))
        last = list(eng.expressions.values())[-1]
        return r, (last.canonical or "")
    finally:
        nb.stop_engine()


class TestEverySpellingOfOneOperationAgrees:
    r"""**同じ操作の綴りが、同じ答えに届くこと。** ここが割れていた。"""

    @pytest.mark.parametrize("lang,text", [
        ("en", "substitute x = zero into E1"),
        ("en", "substitute zero for x in E1"),
        ("en", "In E1, substitute zero for x"),
        ("en", "replace x with zero in E1"),
        ("en", "evaluate E1 at x = zero"),
        ("en", "set x = zero in E1"),
        ("ja", "E1 の x をゼロに置換して"),
        ("ja", "E1 で x = ゼロ として"),
    ])
    def test_a_written_zero_substitutes_the_number(self, lang, text):
        r, canon = _run(lang, ["f = a x + b"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == "b", (text, canon)

    @pytest.mark.parametrize("lang,text", [
        ("en", "substitute x = one into E1"),
        ("en", "substitute one for x in E1"),
    ])
    def test_a_written_one_substitutes_the_number(self, lang, text):
        r, canon = _run(lang, ["f = a x + b"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") in ("b+a", "a+b"), (text, canon)

    def test_a_digit_still_works(self):
        r, canon = _run("en", ["f = a x + b"], "substitute x = 2 into E1")
        assert r.get("status") == "completed", r.get("error")
        assert canon.replace(" ", "") == "b+2*a", canon


class TestANameFieldIsNotTouched:
    r"""**どの欄を触ってはいけないかは、既に名指されている。**

    四十九問目③で踏んだ「名前が数に化ける」（`declare zero as a scalar` が
    `declare` を宣言）を、今度は欄の側で作らないための境界。
    """

    def test_name_is_an_identifier_field(self):
        assert "name" in IDENTIFIER_KEYS

    @pytest.mark.parametrize("lang,text,want", [
        ("en", "declare zero as a scalar", "zero"),
        ("en", "declare one as a scalar", "one"),
        ("ja", "one をスカラーとする", "one"),
        # **この2つが guard の効いている場所である**（変異試験で分かった、
        # 2026-09-10）。宣言の多くは `extra_pre_phase1` が `name` を自分で埋める
        # ので `map_tokens_to_fields` を通らないが、演算子の宣言は別名走査に
        # 落ちて欄が位置で埋まる——guard を外すと `define spin operator one` が
        # **名前 `1` の記号**を宣言する。
        ("ja", "one のスピン演算子を定義して", "one"),
        ("en", "define spin operator one", "one"),
    ])
    def test_a_symbol_named_like_a_number_keeps_its_name(self, lang, text, want):
        nb.start_engine(title="fvn", mode="standard", language=lang, domains=DOM)
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=DOM, primary_backend="maxima")
            r = asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))
            names = list(eng.symbol_context.registry)
        finally:
            nb.stop_engine()
        assert r.get("status") == "confirmed", (text, r.get("error"))
        assert want in names, (text, names)
        assert "declare" not in names and "define" not in names, names


class TestTheReadingIsTheSameOneAsTheExpressionSide:
    r"""欄の値と式で**別の答え**を持たない——`number_literal` 1つである。"""

    def test_the_field_reading_and_the_expression_reading_agree(self):
        from notebook.runtime.maxima_latex_converter import latex_to_canonical
        from notebook.runtime.number_words import number_literal
        for written in ("zero", "one", "ten", "ゼロ", "零", "3", "x", "a + b"):
            as_field = number_literal(written)
            as_expr = latex_to_canonical(written)
            if as_field is None:
                assert as_expr != "0" or written == "0", written
            else:
                assert as_expr == as_field, (written, as_expr, as_field)

    def test_ordinals_are_not_values_here_either(self):
        """序数と倍数副詞は数え方であって値ではない（`number_literal` の語彙）。"""
        from notebook.runtime.number_words import number_literal
        for written in ("third", "twice", "thrice", "first"):
            assert number_literal(written) is None, written


class TestTheOtherFieldsAreUnchanged:
    @pytest.mark.parametrize("lang,text", [
        ("en", "differentiate E1 with respect to x"),
        ("en", "expand E1 to third order in x"),
        ("en", "integrate E1 over x from 0 to 1"),
        ("en", "solve E1 for x"),
        ("ja", "E1 を x で微分して"),
        ("ja", "E1 の x を 2 に置換して"),
    ])
    def test_ordinary_instructions_still_run(self, lang, text):
        """**「動く」に確認ダイアログを含める。** `expand E1 to third order in x` は
        展開中心を訊くので `needs_confirmation` を返す——これは読めている印であって
        失敗ではない（変更の前後で同じ、実測）。**塞がりに合わせたのではなく、
        こちらの期待の書き方が狭かった**ので直した。"""
        r, _ = _run(lang, ["g = x^2 + 1"], text)
        assert r.get("status") in ("completed", "registered", "needs_confirmation"), \
            (text, r.get("status"), r.get("error"))
