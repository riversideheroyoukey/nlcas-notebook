r"""OPERAND-BOUNDARY 残①（「直前の式に落ちる」族, 2026-09-11）。

**研究者が名指したものを、別の式で代用してはならない。**

「文が式を名指していないので直前の式で代用する」——この問いに、この系は
**5箇所で別々に答えていた**。足場（step 2d）だけが 2026-09-11 に塞がれ、
残りは開いたままで、どれも同じ形の誤りを返していた（実測）:

| 打った文 | 走った操作 | 返った答え |
|---|---|---|
| `G の行列式を求めて`（G は宣言済みの行列） | `determinant(expr=E1)` | **completed = 3**（別の行列 A の行列式） |
| `G のトレースを求めて` | `trace(expr=E3)` | **completed**（直前の**結果**の跡） |
| `G の固有値と固有ベクトルを求めて` | `execute_plan(target=E1)` | **completed**（A の固有系。日英とも） |
| `E99 の固有値と固有ベクトルを求めて` | `execute_plan(target=E1)` | **completed**（書いた番号は無いのに） |
| `E99 を x で微分して`（E1 を選択中） | `differentiate(expr=E1)` | **completed = 2*x** |

②の行——**書いた式番号が解決しないとき**——はこの起票が名指していなかった
半分である。`E99` と書いて別の式の答えが返るのは、宣言済みの名前の場合と
同じ形の silent-wrong であり、同じ1行が返していた。

直した先は**操作の断り**であって能力ではない（名前を式クラスの欄に通す半分は
open_items OPERAND-BOUNDARY 残①）。**沈黙した誤りが、声を出す失敗に変わる。**

**`interpret_rules` の `_dr_active[-1]` は到達不能だった**——`selected_expr_id` は
pydantic の `str` で既定が `""` なので、直上の `is None` は真にならない。測って
分かったので、「起票が 未 と書いていた3箇所」のうち1つは欠陥ではなく地雷だった。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import MathType, Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_boundary import may_fall_back_to_the_recent_expression
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ctx(**decls) -> SymbolContext:
    ctx = SymbolContext()
    for name, decl in decls.items():
        kind, algebra = decl if isinstance(decl, tuple) else (decl, "scalar")
        ctx.register(name, quantity=Quantity(name=name, kind=kind,
                                             math_type=MathType(algebra=algebra)))
    return ctx


class TestTheQuestionItself:
    """落ちてよいのは、文が**何も名指していない**ときだけ."""

    @pytest.mark.parametrize("text", ["簡約して", "simplify", "行列式を求めて"])
    def test_a_sentence_naming_nothing_may_fall_back(self, text):
        assert may_fall_back_to_the_recent_expression(_ctx(G=("scalar", "matrix")), text)

    @pytest.mark.parametrize("text", ["G の行列式を求めて", "the determinant of G"])
    def test_a_declared_object_stops_it(self, text):
        assert not may_fall_back_to_the_recent_expression(
            _ctx(G=("scalar", "matrix")), text)

    @pytest.mark.parametrize("text", [
        "E99 の行列式を求めて",
        "the determinant of E99",
        "E1 を x で微分して",
    ])
    def test_a_written_expression_id_stops_it(self, text):
        r"""ここへ来たのは番号が解決しなかったからである——代用してはいけない."""
        assert not may_fall_back_to_the_recent_expression(_ctx(), text)

    @pytest.mark.parametrize("text", ["x で微分して", "differentiate with respect to x"])
    def test_a_scalar_is_a_variable_not_an_object(self, text):
        assert may_fall_back_to_the_recent_expression(_ctx(x="scalar"), text)

    def test_a_substring_is_not_an_expression_id(self):
        r"""`E1` の綴りは境界つき——`XE1x` は式参照ではない."""
        assert may_fall_back_to_the_recent_expression(_ctx(), "XE1x を簡約して")


# --------------------------------------------------------------------------
# 生きた経路。**「答えが返らない」ではなく「別の式の答えが返らない」を測る。**
# --------------------------------------------------------------------------

def _run(lang, domain, lines, selected="", mode="standard"):
    async def go():
        nb.start_engine(title="recent-substitute", mode=mode,
                        language=lang, domains=[domain])
        out = []
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for i, line in enumerate(lines):
                req = InterpretRequest(
                    tokens=[InputToken(type="text", value=line)],
                    selected_expr_id=(selected if i == len(lines) - 1 else ""))
                out.append(await run_interpret_pipeline(req, engine, sup))
        finally:
            nb.stop_engine()
        return out
    return asyncio.run(go())


_MATRIX = {
    "ja": ["A = \\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}", "G を行列とする"],
    "en": ["A = \\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}",
           "declare G as a matrix"],
}


class TestAMatrixUnaryDoesNotAnswerAboutAnotherMatrix:
    """`interpret_pipeline` の行列単項（`_recent()`）."""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "G の行列式を求めて"),
        ("ja", "G のトレースを求めて"),
        ("en", "find the determinant of G"),
    ])
    def test_a_declared_matrix_is_not_replaced(self, lang, text):
        last = _run(lang, "physics", _MATRIX[lang] + [text])[-1]
        assert last.get("status") != "completed", last


class TestThePlanTargetIsWhatTheSentenceNames:
    """`interpret_plan._find_target`——日英とも同じ1行が答えていた."""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "G の固有値と固有ベクトルを求めて"),
        ("en", "eigenvalues and eigenvectors of G"),
    ])
    def test_a_declared_matrix_is_not_replaced(self, lang, text):
        last = _run(lang, "physics", _MATRIX[lang] + [text])[-1]
        assert last.get("status") != "completed", last

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E99 の固有値と固有ベクトルを求めて"),
        ("en", "eigenvalues and eigenvectors of E99"),
    ])
    def test_an_expression_id_that_is_not_there_is_not_replaced(self, lang, text):
        last = _run(lang, "physics", [_MATRIX[lang][0], text])[-1]
        assert last.get("status") != "completed", last

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の固有値と固有ベクトルを求めて"),
        ("en", "eigenvalues and eigenvectors of E1"),
    ])
    def test_the_expression_the_sentence_names_still_runs(self, lang, text):
        """止めるのは名指したものが解決しないときだけ——通る道は通る."""
        last = _run(lang, "physics", [_MATRIX[lang][0], text])[-1]
        assert last.get("status") == "completed", last


class TestTheSelectedExpressionIsNotASubstituteEither:
    r"""`interpret_rules` の微分規則。**この規則は式番号を必ず要求する**ので、
    ここへ来たら書いた番号が解決しなかったということである."""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E99 を x で微分して"),
        ("en", "differentiate E99 with respect to x"),
    ])
    def test_a_written_id_that_is_not_there_is_not_replaced(self, lang, text):
        last = _run(lang, "physics", ["y = x^2", "z = x^3", text],
                    selected="E1")[-1]
        assert last.get("status") != "completed", last
        assert "E99" in (last.get("error") or ""), last

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を x で微分して"),
        ("en", "differentiate E1 with respect to x"),
    ])
    def test_the_expression_the_sentence_names_still_runs(self, lang, text):
        last = _run(lang, "physics", ["y = x^2", "z = x^3", text])[-1]
        assert last.get("status") == "completed", last
        assert "2*x" in (last.get("canonical") or ""), last


_LAGRANGIAN = {
    "ja": ["x を時間依存関数として定義する", "m をスカラーとする", "k をスカラーとする",
           r"L = (1/2) m \dot{x}^2 - (1/2) k x^2"],
    "en": ["define x as a time-dependent function", "let m be a scalar",
           "let k be a scalar", r"L = (1/2) m \dot{x}^2 - (1/2) k x^2"],
}


class TestTheVelocityRuleAsksTheSameQuestion:
    r"""`interpret_rules` の q̇ 規則（`_recent_ids[-1]`）——起票が数えていなかった
    5つ目。ここの E-id は**任意**なので落ちること自体は能力だが、**書いた番号が
    無いとき**に落ちれば同じ silent-wrong になる。実測 2026-09-11、guard を外すと
    日英とも `completed = m`（直前の**結果** `m*x_dot` を ẋ で微分したもの）."""

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"E99 を \dot{x} で偏微分して"),
        ("en", r"the derivative of E99 with respect to \dot{x}"),
    ])
    def test_a_written_id_that_is_not_there_is_not_replaced(self, lang, text):
        lines = _LAGRANGIAN[lang] + [
            {"ja": r"E1 を \dot{x} で偏微分して",
             "en": r"the derivative of E1 with respect to \dot{x}"}[lang], text]
        out = _run(lang, "analytical_mechanics", lines)
        assert out[-2].get("status") == "completed", out[-2]   # 通る道は通る
        assert out[-1].get("status") != "completed", out[-1]

    @pytest.mark.parametrize("lang,text", [
        pytest.param("ja", r"\dot{x} で偏微分して", marks=pytest.mark.xfail(
            strict=True,
            reason="open_items OPERAND-BOUNDARY 残④ —— 日本語側は 2a-sym が "
                   r"`\dot{x}` を式番号へ書き換えてしまい `E2 で偏微分して` に "
                   "なる。HEAD で同じなので、この guard とは無関係の日英割れ")),
        ("en", r"differentiate with respect to \dot{x}"),
    ])
    def test_a_sentence_naming_nothing_still_falls_back(self, lang, text):
        """落ちること自体は能力——止めるのは名指したときだけ."""
        out = _run(lang, "analytical_mechanics", _LAGRANGIAN[lang] + [text])
        assert out[-1].get("status") == "completed", out[-1]


class TestANameTheGrammarAlreadyReadIsNotTheOperand:
    r"""**起票が「素朴に入れると壊れる」と書いていた形**——`S の各成分のハイゼンベルク
    方程式を求めて` の `S` は宣言済みの演算子だが、**対象ではない**。どの演算子の
    成分を発展させるかであって、対象はハミルトニアンのほうである。

    実測 2026-09-11、guard を入れて初めて出た: `_find_target` が `None` を返し、
    いちばん普通の綴りで workflow が黙って降りた。判別は「その名前が**オペランドの
    位置に**立っているか」——呼び出し側の文法は名前つきグループでそれを既に
    知っているので、読んだ名前を渡してもらう。
    """

    @pytest.mark.parametrize("lang,text", [
        ("ja", "S の各成分のハイゼンベルク方程式を求めて"),
        ("en", "Heisenberg equations for all components of S"),
    ])
    def test_the_base_operator_is_not_the_target(self, lang, text):
        setup = {
            "ja": ["S をスピン演算子として定義して", "gamma をスカラーとする",
                   "B をスカラーとする", "H = -gamma B Sz_1"],
            "en": ["define S as a spin operator", "let gamma be a scalar",
                   "let B be a scalar", "H = -gamma B Sz_1"],
        }[lang]
        out = _run(lang, "condensed_matter", setup + [text], mode="second_quantization")
        assert out[-1].get("status") == "completed", out[-1]
        assert len(out[-1].get("plan_step_results") or []) == 3, out[-1]
