r"""ハイゼンベルク族は1つの形 —— 方程式と演算子で語順の読みが割れていた。

実測 2026-09-06（`second_quantization` / julia_full。`a` を光子の消滅演算子、
`w` をスカラー、`E1 = w a'(1) a(1)`）。`heisenberg_equation` と
`heisenberg_operator` は**同じ2欄**（演算子・ハミルトニアン）を取り、日本語では
同じ助詞（`のもと` / `を…で`）、英語では同じ前置詞（`of`/`for` … `under`）が
どちらがどちらかを名指す。にもかかわらず読みが4通りに割れていた。

| # | 打った文 | 直す前 | 原因 |
|---|---|---|---|
| 1 | `E1 と a(1) のハイゼンベルク方程式` | **`dH/dt = +i w a₁`** | 専用パターンが `Heisenberg`（ラテン文字）だけを綴っていて、カタカナは別名走査に落ちる。そこは**欄を位置で埋める**（BP-6）ので演算子とハミルトニアンが入れ替わり、符号まで逆になった。`dH/dt` は保存量なので本来 0 である |
| 2 | `E1 のもとでの a(1) のハイゼンベルク方程式を求めて` | 同上 | ハミルトニアンが先に来る語順を誰も読んでいなかった |
| 3 | `the Heisenberg operator of a(1) under E1` | `a_{1} \vee 1 \mathrel{\ldotp\mkern-2.5mu\ldotp} - w` | 読みが `evolve … under` の1綴りだけ。残りは別名走査 →`run_operation` →**内部プロトコル文字列**（`基底;…|行;…`）が式として公開される |
| 4 | `E1 のもとでの a(1) の Heisenberg 演算子` | `dH/dt = +i w a₁` | **演算子**を訊いたのに別名 `heisenberg` が**方程式**を釣り上げ、さらに欄が位置埋め |

**`ui/interpret_pair_rep.py` の冒頭は最初から**「`run_operation` の一般経路には
乗せない——返るのが CAS の値ではなく構造だからである」と書いていた。書かれて
いなかったのは `run_operation` の側で、そこに辿り着く道（別名走査・LLM）は開いた
ままだった。読みを広げてもガードは残す（`\max_i` のときと同じ）。

**「ハイゼンベルク模型」は横取りしない。** コーパスの `ハイゼンベルク` 行は
4行すべて模型（Heisenberg MODEL）で、演算子の読みは頭の名詞（`演算子` /
`時間発展` / `operator` / `time evolution`）を要求するので触れない。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.operand_guards import STRUCTURE_ONLY_OPERATIONS
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

PREP = {"ja": ["a を光子の消滅演算子とする", "w をスカラーとする",
               "H = w a'(1) a(1)"],
        "en": ["define a as a photon annihilation operator",
               "define w as a scalar", "H = w a'(1) a(1)"]}

#: 方程式を訊く綴り。**どれも同じ2欄**——演算子 `a(1)`、ハミルトニアン `E1`。
EQUATION = [
    ("ja", "E1 と a(1) の Heisenberg 方程式"),
    ("ja", "E1 と a(1) のハイゼンベルク方程式"),
    ("ja", "E1 と a(1) のハイゼンベルグ方程式"),
    ("ja", "a(1) の Heisenberg 方程式 E1"),
    ("ja", "a(1) のハイゼンベルク方程式 E1"),
    ("ja", "a(1) のハイゼンベルク運動方程式を E1 で求めて"),
    ("ja", "E1 のもとでの a(1) のハイゼンベルク方程式を求めて"),
    ("ja", "E1 の下での a(1) の Heisenberg 方程式"),
    ("en", "Heisenberg equation of a(1) for E1"),
    ("en", "the Heisenberg equation of a(1) for E1"),
    ("en", "Heisenberg equation for a(1) under E1"),
    ("en", "the Heisenberg equation of a(1) under E1"),
    ("en", "compute the Heisenberg equation of a(1) for E1"),
    ("en", "find the Heisenberg equation for a(1) under E1"),
]

#: 演算子（ハイゼンベルク描像／時間発展）を訊く綴り。同じ2欄。
OPERATOR = [
    ("ja", "a(1) を E1 のもとで時間発展させて"),
    ("ja", "a(1) を E1 の下で時間発展させて"),
    ("ja", "E1 のもとで a(1) を時間発展させて"),
    ("ja", "E1 のもとでの a(1) のハイゼンベルク演算子"),
    ("ja", "E1 のもとでの a(1) の Heisenberg 演算子"),
    ("ja", "E1 の下での a(1) のハイゼンベルク描像の演算子"),
    ("ja", "a(1) の E1 のもとでのハイゼンベルク演算子を求めて"),
    ("ja", "a(1) のハイゼンベルク描像での演算子を E1 のもとで求めて"),
    ("ja", "a(1) の E1 のもとでの時間発展"),
    ("en", "evolve a(1) under E1"),
    ("en", "evolve a(1) in time t under E1"),
    ("en", "the Heisenberg operator of a(1) under E1"),
    ("en", "Heisenberg operator of a(1) under E1"),
    ("en", "the Heisenberg picture operator of a(1) under E1"),
    ("en", "the Heisenberg operator for a(1) under E1"),
    ("en", "compute the Heisenberg operator of a(1) under E1"),
    ("en", "the time evolution of a(1) under E1"),
]

#: 演算子の読みが**取ってはいけない**もの。模型は計算対象、方程式は別の操作。
NOT_THE_OPERATOR = [
    ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 のスペクトル関数を求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で4次まで求めて"),
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の2次モーメントを求めて"),
    ("ja", "E1 と a(1) のハイゼンベルク方程式"),
    ("ja", "E1 のもとでの a(1) のハイゼンベルク方程式を求めて"),
    ("en", "Find the spectral function of S^x_2 for the Heisenberg model"),
    ("en", "Heisenberg equation of E1 for c(k,u)"),
    ("en", "the Heisenberg equation of a(1) under E1"),
]


def _ask(lang, text):
    nb.start_engine(title="heis", mode="second_quantization", language=lang,
                    domains=["condensed_matter"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["condensed_matter"],
                            primary_backend="julia_full")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in PREP[lang]:
            run(p)
        return run(text)
    finally:
        nb.stop_engine()


class TestTheEquationReadsTheSameTwoFields:
    """どの語順・どの表記でも、演算子は `a(1)`、ハミルトニアンは `E1`。"""

    @pytest.mark.parametrize("lang,text", EQUATION)
    def test_computes_da_dt(self, lang, text):
        resp = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("error"))
        latex = resp.get("latex") or ""
        # da(1)/dt = -i w a_1。**入れ替わると `dH/dt` になり符号も逆になる**。
        assert "a(1)" in latex and "d t" in latex, (text, latex)
        assert "- w" in latex, (text, latex)


class TestTheOperatorReadsTheSameTwoFields:
    """`e^{-i w t} a_1`——閉包が1×1なので厳密。"""

    @pytest.mark.parametrize("lang,text", OPERATOR)
    def test_computes_the_phase(self, lang, text):
        resp = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("error"))
        assert resp.get("operation") == "heisenberg_operator", (text, resp)
        assert resp.get("latex") == "e^{- i t w} a_{1}", (text, resp.get("latex"))


class TestTheReadingIsOnePattern:
    """CAS を起動せずに、綴り空間そのものを固定する。"""

    @pytest.mark.parametrize("lang,text", OPERATOR)
    def test_every_operator_spelling_names_both_operands(self, lang, text):
        m = nlp("heisenberg_operator_nl", lang).search(text)
        assert m is not None, text
        assert pick_group(m, "op") == "a(1)", (text, m.groupdict())
        assert pick_group(m, "eid") == "E1", (text, m.groupdict())

    @pytest.mark.parametrize("lang,text", NOT_THE_OPERATOR)
    def test_the_operator_reading_claims_nothing_else(self, lang, text):
        assert nlp("heisenberg_operator_nl", lang).search(text) is None, text

    @pytest.mark.parametrize("lang,text,op,ham", [
        ("ja", "E1 のもとでの a(1) のハイゼンベルク方程式を求めて", "a(1)", "E1"),
        ("ja", "E1 の下での c'(m) のハイゼンベルク運動方程式", "c'(m)", "E1"),
        ("en", "the Heisenberg equation of a(1) under E1", "a(1)", "E1"),
        ("en", "compute the Heisenberg equation for c(m) under H", "c(m)", "H"),
    ])
    def test_the_under_word_order_names_which_is_which(self, lang, text, op, ham):
        """**位置ではなく助詞／前置詞が決める。** 名前付きグループなので、
        枝を足しても呼ぶ側は変わらない。"""
        m = nlp("heisenberg_eom_under_first", lang).search(text)
        assert m is not None, text
        assert (m.group("op"), m.group("ham")) == (op, ham), m.groupdict()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_time_variable_survives_the_branches(self, lang):
        text = {"ja": "c'(m) を E3 で時刻 u まで時間発展させて",
                "en": "evolve c'(m) in time u under E3"}[lang]
        m = nlp("heisenberg_operator_nl", lang).search(text)
        assert m is not None and pick_group(m, "time") == "u", text

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_default_time_variable_is_not_invented(self, lang):
        """時刻が書かれていない枝は `time` を埋めない——既定の `t` は読みの側で
        推測せず、呼ぶ側が1箇所で決める。"""
        text = {"ja": "E1 のもとでの a(1) のハイゼンベルク演算子",
                "en": "the Heisenberg operator of a(1) under E1"}[lang]
        m = nlp("heisenberg_operator_nl", lang).search(text)
        assert m is not None and pick_group(m, "time") is None, text


class TestAStructureIsNotAGenericOperation:
    """`heisenberg_operator` の結果は値ではないので、汎用経路では走らせない。"""

    def test_the_operation_is_listed(self):
        assert "heisenberg_operator" in STRUCTURE_ONLY_OPERATIONS

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_run_operation_refuses_with_a_reason(self, lang):
        nb.start_engine(title="heis_guard", mode="second_quantization",
                        language=lang, domains=["condensed_matter"])
        try:
            eng = nb.get_engine()
            eng.run_operation("define_boson_operator", {"name": "a"})
            eng.run_operation("define_scalar_parameter", {"name": "w"})
            eng.register_expression(r"w a'(1) a(1)")
            with pytest.raises(ValueError) as err:
                eng.run_operation("heisenberg_operator",
                                  {"operator": "a(1)", "hamiltonian": "E1"})
            msg = str(err.value)
            # **理由が研究者の言葉であること**——プロトコル文字列でも
            # 生の CAS 文言でもない。
            assert msg == t("pair_rep.not_a_generic_operation", lang,
                            operation="heisenberg_operator")
            assert "|" not in msg and ";" not in msg
        finally:
            nb.stop_engine()
