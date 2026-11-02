r"""**系が印刷した射影を、そのまま打ち返せる**（OPERAND-BOUNDARY ③ 残、2026-09-12）。

4つ目の供給源——「系が印刷した綴りを打ち直す」——で、この供給源は過去に
TEX-MATRIX-INPUT を出している。整数スピンの射影は 2026-09-12 に**表示の綴り**
（`⁺ ⁰ ⁻` → 画面 `+ 0 -`）を得たが、**入力の綴りが無かった**::

    系が印刷        {a}_{1{+}}^{\dagger}          （画面では a^†_{1+}）
    打ち返して使う   → 生の Julia `ParseError:`   ← 研究者の顔に CAS の文言（原則#1）

**m=0 に至っては印刷そのものが曖昧だった**——`{b}_{1{0}}` は画面で `b_{10}` で、
**サイト10と見分けが付かない**。系が読み戻せない形を印刷していたことになる。

矢印はなぜ通っていたか: `\uparrow` は**コマンドで、それ自体が添字の境界**である
（CLAUDE.md の `g_{\mu\nu}` 規則）。素の符号は境界を持たない。だから

- **表示**は区切りを入れる（`a^\dagger_{1,+}`、`b_{1,0}`）——系自身の多添字表示
  （`A_{i,j}`）と同じ形。**矢印には入れない**: コマンドが境界なので要らず、
  `c_{1\uparrow}` は凝縮系のふつうの綴りである。**区切りは、入力側が境界を
  見つけられないときにだけ印刷する。**
- **入力**は符号を名前に落とす（`+` → `mplus` → QA の `⁺`）。矢印と同じ3段で、
  3つの表が合成して往復が閉じる::

      QA `⁺` → 画面 `+`       spin_value_set.PROJECTION_LATEX
      入力 `+` → 名前 `mplus`  spin_label_canon.PROJECTION_SIGN_INPUT
      名前 `mplus` → QA `⁺`    spin_label_canon.SPIN_LABEL_CANON

**位置は判別子にならない。** `\omega_0` は「数字だけの添字は名前の一部」、
`Y_{1,0}` は球面調和関数の成分、`S_+` は昇降演算子——どれも符号や 0 が単独で
添字に立つ。判別子は**宣言**で、しかも `quantum_operator_names` より1段狭い
**整数スピン（＝ボソン）**である: 電子の `c_{1,+}` に当てると、その粒子が持たない
射影を黙って添字にしてしまう。

**限界**: 宣言に残っているのは統計までで**粒子そのものは残っていない**ので、
「その粒子の値集合に本当にこの射影が在るか」は確かめられない（スピン0のボソンに
`a_{1,+}` と書けば意味の無い添字が付く）。値集合まで見るには宣言側に粒子を残す
必要がある。
"""
import asyncio
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.spin_label_canon import (PROJECTION_SIGN_INPUT,
                                               SPIN_LABEL_CANON)
from notebook.runtime.spin_value_set import PROJECTION_LATEX
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_DOM = ["condensed_matter"]


def _session(particle, op, lines):
    """宣言＋Σ の下拵えのあと *lines* を順に打ち、最後の応答と式の一覧を返す。"""
    async def go():
        nb.start_engine(title="proj", mode="standard", language="ja", domains=_DOM)
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=_DOM,
                                primary_backend=engine.session.primary_backend)
            for line in [f"{op} を{particle}の消滅演算子とする",
                         f"P = \\sum_\\sigma {op}^\\dagger_{{1\\sigma}} {op}_{{2\\sigma}}",
                         f"Q = \\sum_\\sigma {op}^\\dagger_{{1\\sigma}} {op}_{{1\\sigma}}"]:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            resp = None
            for line in lines:
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            return resp, engine
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheThreeTablesCompose:
    r"""往復が閉じることを、表の合成として固定する（実行を伴わない）。"""

    @pytest.mark.parametrize("glyph", ["⁺", "⁰", "⁻"])
    def test_the_loop_closes_for_the_signs(self, glyph):
        shown = PROJECTION_LATEX[glyph]           # QA → 画面
        name = PROJECTION_SIGN_INPUT[shown]       # 入力 → 名前
        assert SPIN_LABEL_CANON[name] == glyph    # 名前 → QA

    @pytest.mark.parametrize("glyph", ["↑", "↓"])
    def test_the_loop_closes_for_the_arrows(self, glyph):
        cmd = PROJECTION_LATEX[glyph]             # `\uparrow`
        assert SPIN_LABEL_CANON[cmd.lstrip("\\")] == glyph

    def test_the_glyph_is_accepted_as_input_too(self):
        r"""画面には出さないが、貼り付けられたら読む。"""
        assert PROJECTION_SIGN_INPUT["⁺"] == "mplus"


class TestTheSystemPrintsWhatCanBeTypedBack:

    @pytest.mark.parametrize("particle,op,want", [
        ("光子", "a", ["1{,+}", "1{,-}"]),
        ("マグノン", "b", ["1{,0}", "1{,+}", "1{,-}"]),
    ])
    def test_the_printed_projection_carries_a_separator(self, particle, op, want):
        r"""区切りが無いと m=0 は `b_{10}` と印刷され、**サイト10と曖昧**になる。"""
        resp, _ = _session(particle, op, ["P と Q の交換子を求めて"])
        printed = resp.get("latex") or ""
        for w in want:
            assert w in printed, (w, printed)
        assert "{b}_{10}" not in printed and "{a}_{1+}" not in printed, printed

    @pytest.mark.parametrize("typed", ["a_{1,+}", "a^\\dagger_{1,+}", "a_{1+}"])
    def test_the_printed_spelling_registers_as_the_operator(self, typed):
        resp, engine = _session("光子", "a", [typed])
        eid = max(engine.expressions, key=lambda k: int(re.sub(r"\D", "", k) or 0))
        assert "mplus" in (engine.expressions[eid].canonical or ""), \
            engine.expressions[eid].canonical


class TestTheProjectionActuallySifts:
    r"""登録できることと、**射影として効く**ことは別——矢印は `δ(↑,↓)=0` が落ちる。"""

    @pytest.mark.parametrize("a,b,want", [
        ("a_{1,+}", "a^\\dagger_{1,+}", "1"),
        ("a_{1,+}", "a^\\dagger_{1,-}", "0"),
        ("a_{1+}",  "a^\\dagger_{1+}",  "1"),
        ("a_{1+}",  "a^\\dagger_{1-}",  "0"),
    ])
    def test_the_delta_falls(self, a, b, want):
        resp, _ = _session("光子", "a", [f"{a} と {b} の交換子を求めて"])
        assert resp.get("latex") == want, (a, b, resp)


class TestWhatMustNotBeTouched:
    r"""**位置は判別子にならない。** 宣言（しかも整数スピン）だけが当たる。"""

    @pytest.mark.parametrize("text", [
        r"\omega_0",        # 数字だけの添字は名前の一部（添字分類表）
        r"Y_{1,0}",         # 球面調和関数の成分
        r"T_{1,0}",         # テンソル成分
        r"S_+",             # 昇降演算子
        r"a_{k+q}",         # 代数的な添字（SymPy 経路）
        r"a_{k-q}",
        r"a_{+}",           # 単独——区切る相手がいない
        r"c_{1,+}",         # **フェルミオン**——電子に `+` の射影は無い
        r"c_{1+}",
    ])
    def test_it_is_left_alone(self, text):
        async def go():
            nb.start_engine(title="guard", mode="standard", language="ja", domains=_DOM)
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=_DOM,
                                    primary_backend=engine.session.primary_backend)
                for line in ["a を光子の消滅演算子とする", "c を電子の消滅演算子とする"]:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return engine._handler._projection_signs_to_names(text)
            finally:
                nb.stop_engine()
        assert asyncio.run(go()) == text, text

    def test_the_discriminator_is_integer_spin_not_merely_operator(self):
        r"""`quantum_operator_names` より1段狭い——電子は入らない。"""
        async def go():
            nb.start_engine(title="stat", mode="standard", language="ja", domains=_DOM)
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=_DOM,
                                    primary_backend=engine.session.primary_backend)
                for line in ["a を光子の消滅演算子とする", "c を電子の消滅演算子とする"]:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return engine._handler._integer_spin_operator_names()
            finally:
                nb.stop_engine()
        names = asyncio.run(go())
        assert "a" in names and "c" not in names, sorted(names)
