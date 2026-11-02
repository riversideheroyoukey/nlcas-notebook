r"""DOT-DISPLAY-IS-AN-ATOM（2026-09-25）——`∂/∂q̇` が 0 を返していた。

起票は「表示 `\dot{r}` を打ち直すと独立な原子になる」という**設計の問い**として
立てたが、着手条件（どちらの読みが実在の指示で要るかを数える）を実行すると、
中に**生きた silent-wrong** が在った。同じ文・同じ式で、**宣言の有無だけ**で答えが
変わる:

    Define m, k as scalars
    L_z = m r^2 \dot{theta}
    differentiate E1 with respect to \dot{theta} partially   →  m*r^2   ✔ 正準運動量

    Define theta as a time-dependent function      ← この1行を足すだけで
    Define m, k as scalars
    L_z = m r^2 \dot{theta}
    differentiate E1 with respect to \dot{theta} partially   →  0       ✘ completed

**綴りが3つあり、変数だけが第3のものを使っていた。** 時間依存量の q̇ は、式の中では
研究者が書いた原子 `q_dot`（`_prepare_expr_lookup` の表が `'diff(q,t,1)` へ写す）か、
CAS が返した `'diff(q,t,1)` である。ところが変数は `'diff(q(t),t,1)` と綴られていた
——`depends([q],[t])` は**式に `q(t)` と書かずに** q を t の関数にする仕組みなので、
その綴りはどちらとも一致しない。実測（生きた Maxima）::

    diff(m*r^2*'diff(theta,t,1), 'diff(theta,t,1), 1)     →  m*r^2
    diff(m*r^2*'diff(theta,t,1), 'diff(theta(t),t,1), 1)  →  0
    diff(m*r^2*theta_dot,        theta_dot, 1)            →  m*r^2

だから綴りは**1つのヘルパ**（`time_dep_derivative`）が答え、式の中を書き換える表と
変数を綴る側が同じものを読む。**どちらの綴りかは式が決める**——CAS の文字列を直接
打った研究者の式は `q(t)` を持つので、そのときは関数形のまま（既存の
`tests/test_engine.py::TestLagrangianIndependentAtoms` がその道を固定している）。

`substitute` は両方の道で最初から正しかった——**同じ問いに2つの答え**があり、片方が
既に正しい、いつもの形である。
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.maxima_backend_handler import time_dep_derivative


def _run(setup, probe, domain="classical_mechanics"):
    import asyncio

    import ui.notebook_bridge as nb
    from notebook.runtime.llm_supervisor import LLMSupervisor
    from notebook.runtime.operation_validator import OperationValidator
    from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                       run_interpret_pipeline)

    async def go():
        nb.start_engine(title="qdot", mode="standard", language="en", domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for text in list(setup) + [probe]:
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
            return resp, list(engine.expressions.values())[-1].canonical
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_SCALARS = "Define m, k as scalars"
_LZ = r"L_z = m r^2 \dot{theta}"
_TIME_DEP = "Define theta as a time-dependent function"


class TestTheAnswerDoesNotDependOnTheDeclaration:
    r"""同じ文・同じ式は、宣言の有無で違う答えを返さない。"""

    @pytest.mark.parametrize("setup", [
        [_SCALARS, _LZ],                 # 宣言なし——ここは昔から正しい
        [_TIME_DEP, _SCALARS, _LZ],      # 宣言あり——ここが 0 を返していた
    ])
    def test_the_canonical_momentum(self, setup):
        resp, value = _run(setup, r"differentiate E1 with respect to \dot{theta} partially")
        assert resp.get("status") == "completed", resp.get("error")
        assert value == "m*r^2", value

    @pytest.mark.parametrize("setup", [
        [_SCALARS, _LZ],
        [_TIME_DEP, _SCALARS, _LZ],
    ])
    def test_substitution_was_always_right(self, setup):
        """片方が既に正しい——`substitute` は両方の道で通っていた。"""
        resp, value = _run(setup, r"In E1, substitute 5 for \dot{theta}")
        assert resp.get("status") == "completed", resp.get("error")
        assert value == "5*m*r^2", value


class TestTheCasProducedFormIsReachedToo:
    r"""CAS が返した `'diff(theta,t,1)` を持つ式にも、同じ変数が届く。"""

    def test_partial_by_qdot_of_a_cas_result(self):
        resp, value = _run([_TIME_DEP, "Define r as a time-dependent function",
                            _SCALARS, _LZ,
                            "Take the total derivative of E1 with respect to t"],
                           r"differentiate E2 with respect to \dot{theta} partially")
        assert resp.get("status") == "completed", resp.get("error")
        assert value == "2*m*r*'diff(r,t,1)", value


class TestOneHelperSpellsIt:
    r"""式の中を書き換える表と、変数を綴る側が、同じヘルパを読む。"""

    def test_the_spelling_has_no_function_argument(self):
        """`depends` は式に `q(t)` と書かずに依存を作る——だから名前は素で立つ。"""
        assert time_dep_derivative("theta", "t", 1) == "'diff(theta,t,1)"
        assert time_dep_derivative("theta", "t", 2) == "'diff(theta,t,2)"

    def test_the_variable_follows_the_expression(self):
        h = NotebookEngine(title="sp")._handler
        # 研究者の綴り（原子／CAS の noun）——表が写す素の形
        assert h.wrt_qdot_canonical("q", "t", "m*q_dot^2") == "'diff(q,t,1)"
        assert h.wrt_qdot_canonical("q", "t", "m*'diff(q,t,1)") == "'diff(q,t,1)"
        # CAS の文字列を直接打った式——関数形のまま
        assert h.wrt_qdot_canonical("q", "t", "(1/2)*m*diff(q(t),t)^2") \
            == "'diff(q(t),t,1)"
        # 部分一致で騙されない
        assert h.wrt_qdot_canonical("q", "t", "m*qq(t)^2") == "'diff(q,t,1)"
