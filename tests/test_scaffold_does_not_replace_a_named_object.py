r"""OPERAND-BOUNDARY 残①（安全側の半分, 2026-09-11）。

**足場は、名前を持つ対象を黙って別の式に差し替えてはならない。**

宣言だけされた対象（ベクトル場・行列・演算子…）には `defining_expr_id` が無いので
2a-sym が名前を式番号へ書き換えられず、その文は「式を名指していない」ように見える。
そこで step 2d が**直前の式**を前置すると、研究者が名指した対象について訊いたのに
別の式の答えが `completed` で返る。実測 2026-09-11::

    A をベクトル場とする / u = (3,4)  のセッションで
      A のノルムを求めて → norm(expr=E1) → completed = 5   ← u のノルム
      直前が w = (6,8) なら                 completed = 10

**答えが「直前に何をしたか」で変わる。**

名指しているなら前置しない。そうすると操作は必要な引数が足りずに断るので、
**誤答が失敗に変わる**——能力にはならないが、沈黙した誤りは消える。
通る道は在って実測してある: `\|A\|` は `sqrt(A_z^2+A_y^2+A_x^2)`、
`\nabla \times A` は3成分を返す。

**実在の指示は1行も変わらない**——コーパス1102行を足場ありとなしで2回測って差分0。

**スカラーは数えない。** `x` は変数として文中に立つので、数えると `x で微分して` が
「対象を名指した文」に見え、足場が止まって対象を失う。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_boundary import text_names_a_declared_object
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ctx(**decls) -> SymbolContext:
    """`name="kind"` か `name=("kind", "algebra")` で宣言を作る。"""
    from notebook.models.quantity import MathType
    ctx = SymbolContext()
    for name, decl in decls.items():
        kind, algebra = decl if isinstance(decl, tuple) else (decl, "scalar")
        ctx.register(name, quantity=Quantity(name=name, kind=kind,
                                            math_type=MathType(algebra=algebra)))
    return ctx


class TestWhichNamesCount:
    """**`kind` だけでは足りない**——実測（2026-09-11）で、行列・テンソル・
    ベクトルは `kind='scalar'` として記録され、判別材料は `math_type.algebra` の
    側にある。`kind` だけで書くと `G を行列とする` の G が漏れる。"""

    @pytest.mark.parametrize("decl", [
        "vector_field",            # A をベクトル場とする
        "scalar_field",            # phi をスカラー場とする
        "operator",                # c を電子の消滅演算子とする
        ("scalar", "matrix"),      # G を行列とする      ← kind は scalar
        ("scalar", "tensor"),      # T をテンソルとする   ← 同上
        ("scalar", "vector"),      # v をベクトルとする   ← 同上
    ])
    def test_an_object_with_its_own_identity_counts(self, decl):
        assert text_names_a_declared_object(_ctx(A=decl), "A のノルムを求めて")

    @pytest.mark.parametrize("decl", [
        ("scalar", "scalar"),      # x をスカラーとする
        ("anonymous", "scalar"),   # r を時間依存関数として定義する
    ])
    def test_a_variable_does_not_count(self, decl):
        assert not text_names_a_declared_object(_ctx(A=decl), "A のノルムを求めて")

    @pytest.mark.parametrize("text", ["x で微分して", "differentiate with respect to x"])
    def test_a_scalar_does_not_count(self, text):
        r"""`x` は変数として文中に立つ。数えると足場が止まり、対象を失う."""
        assert not text_names_a_declared_object(_ctx(x="scalar"), text)

    def test_a_name_that_is_not_written_does_not_count(self):
        assert not text_names_a_declared_object(_ctx(A="vector_field"), "簡約して")

    def test_a_substring_is_not_a_name(self):
        assert not text_names_a_declared_object(_ctx(A="vector_field"),
                                                "Ab のノルムを求めて")

    def test_no_context(self):
        assert not text_names_a_declared_object(None, "A のノルムを求めて")


def _supervisor_text(lang, domain, setup, text) -> "list[str]":
    """足場が前置されたかは、**足場の直接の消費者**に当てて測る。"""
    import ui.interpret_pipeline as P
    seen: list = []
    original = P._try_rule_short_circuits

    async def spy(req, engine, loop, lang_, supervisor_text, *a, **k):
        seen.append(supervisor_text)
        return await original(req, engine, loop, lang_, supervisor_text, *a, **k)

    async def go():
        nb.start_engine(title="scaffold-named", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            seen.clear()
            P._try_rule_short_circuits = spy
            try:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
            finally:
                P._try_rule_short_circuits = original
        finally:
            nb.stop_engine()
        return seen

    asyncio.run(go())
    return seen


_SETUP = {
    "ja": ["A をベクトル場とする", "u = [1, 2, 3]"],
    "en": ["define A as a vector field", "u = [1, 2, 3]"],
}


class TestTheScaffoldStopsAtANamedObject:
    @pytest.mark.parametrize("lang,text", [
        ("ja", "A のノルムを求めて"),
        ("en", "the norm of A"),
    ])
    def test_no_expression_is_prepended(self, lang, text):
        given = _supervisor_text(lang, "physics", _SETUP[lang], text)
        assert given, "supervisor に届いた文字列が取れていない"
        assert all("E1" not in g for g in given), given

    @pytest.mark.parametrize("lang,text,marker", [
        ("ja", "簡約して", "に対して"),
        ("en", "simplify",  "E1"),
    ])
    def test_a_sentence_naming_nothing_still_gets_it(self, lang, text, marker):
        """止めるのは名指したときだけ——足場そのものは要る."""
        given = _supervisor_text(lang, "physics", _SETUP[lang], text)
        assert given, "supervisor に届いた文字列が取れていない"
        assert any(marker in g for g in given), given
