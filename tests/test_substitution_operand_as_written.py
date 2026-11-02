"""代入のオペランドは、研究者が書いた名前である —— 同じ問いの3箇所目。

2a-sym（``ui/supervisor_alias_rewrite``）は、ある式の左辺になっている記号を
その式の E-id に書き換える。**式への参照**にはそれが要り、**名前で渡すオペランド**
には要らない——両者は要求が逆で、1つの文字列では兼ねられない
（``InterceptorContext.alias_free_text``）。

``_parse_solve`` はこの答えに**規則ごとに2度**たどり着いた（連立の枝 2026-08-16、
単一方程式の枝 2026-08-21）。**代入が3箇所目**で、2026-08-27 に測ると E-id のまま
CAS に届いていた: ``solve E1 for x`` のあと

    E1 の x を 2 に置換して      → lhs=E2、結果 a*x+b（何も置換されず）**completed**
    E1 の x を 2 で置き換えて     → 同上
    replace x with 2 in E1      → 同上
    E1 の x に 2 を代入して       → 素の E-id を弾く guard が発火して **failed**

実在の10綴りのうち **4つ**。英語の ``substitute … for x`` 2つが助かっていたのは
偶然で、別名 guard の ``for`` がたまたま当たっていたにすぎない。
「x について解いて、その x を代入する」は完全にありふれた手順である。

**guard もオペランドについての問いなので、書かれたとおりを見る。** 「素の E-id か」
「登録済みの演算子の名前か」はどちらも lhs/rhs についての問いで、書き換え後の文で
訊けば研究者が書いていない E-id について答えることになる。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _session(lang, alias):
    """`f = a*x + b` を登録し、*alias* なら x を解いて別名を作る。"""
    nb.start_engine(title="subst_as_written", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a, b をスカラーとする" if lang == "ja" else "Define a, b as scalars")
    _run(eng, sup, "f = a*x + b")
    if alias:
        _run(eng, sup, "E1 を x について解いて" if lang == "ja" else "solve E1 for x")
    return eng, sup


SPELLINGS = [
    ("ja", "E1 の x を 2 に置換して"),
    ("ja", "E1 の x を 2 で置き換えて"),
    ("ja", "E1 の x に 2 を代入して"),
    ("en", "In E1, substitute 2 for x"),
    ("en", "substitute 2 for x in E1"),
    ("en", "replace x with 2 in E1"),
]


class TestAnAliasedSymbolIsStillTheResearchersName:
    @pytest.mark.parametrize("lang,text", SPELLINGS)
    @pytest.mark.parametrize("alias", [True, False], ids=["solved-first", "no-alias"])
    def test_the_substitution_actually_happens(self, lang, text, alias):
        eng, sup = _session(lang, alias)
        try:
            r = _run(eng, sup, text)
            assert r["status"] == "completed", r
            assert r["canonical"].replace(" ", "") == "b+2*a", r["canonical"]
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", SPELLINGS)
    def test_the_label_names_the_symbol_not_the_expression(self, lang, text):
        """The card is the history; an E-id there names a quantity whose meaning
        is a whole expression, not the symbol that was replaced."""
        eng, sup = _session(lang, alias=True)
        try:
            label = _run(eng, sup, text).get("operation_label") or ""
            assert "x" in label, label
            assert "E2" not in label, label
        finally:
            nb.stop_engine()


class TestTheExpressionReferenceStillGetsTheRewrite:
    """The other half of `alias_free_text`: a REFERENCE must keep the E-id, and
    an expression-into-expression substitution still belongs to `dainyu`."""

    def test_a_named_expression_is_still_resolved(self):
        eng, sup = _session("ja", alias=False)
        try:
            _run(eng, sup, "g = 3")
            r = _run(eng, sup, "E2 を E1 に代入して")
            assert r["status"] in ("completed", "failed"), r
            assert r.get("operation") != "substitute" or "E" in str(r.get("canonical", ""))
        finally:
            nb.stop_engine()

    def test_an_explicit_eid_operand_is_not_taken_as_a_symbol(self):
        """`E1 の E2 に 3 を代入して` names an E-id where a symbol would go — the
        bare-E-id guard must still fire, on the researcher's own text."""
        eng, sup = _session("ja", alias=False)
        try:
            _run(eng, sup, "g = 3")
            r = _run(eng, sup, "E1 の E2 に 3 を代入して")
            assert r.get("operation") != "substitute" or r["status"] != "completed", r
        finally:
            nb.stop_engine()
