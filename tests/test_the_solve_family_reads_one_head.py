r"""NL-GRAMMAR ① 五十四問目「変数を導く語は solve 一族で1つの表か」（2026-09-19）。

日本語はずっと1つの表だった（`ja_solve_var_particle` ＝ `について|に関して`、3鍵が
読む）。英語は `in` を `solve_single` の中に**直書き**しており、兄弟2鍵
（`solve_solution_of_var` / `solve_system_vars`）は共有の `en_op_var_head`
（`for|with respect to|w.r.t.`）しか読まなかった。

**実測 2026-09-19、同じ前置詞で3鍵を並べる（英 12セル・日 6セル）:**

| 打った文 | 返っていたもの |
|---|---|
| `compute the solution of E1 in x` | 落ちる（大きな声） |
| `find the solution to E1 in x` | 落ちる |
| **`solve E1 and E2 in x and y`** | **`solve_equation(E1, x)` → `x = 3-y` を `completed`** |

3つめが重い——**連立の指示に、1本目だけを1変数について解いた答えが成功として返る**
（兄弟3綴りは全部 `x = 2, y = 1`）。どの鍵にも当たらないので**別名走査が答えていた**、
2026-09-12 の四十五問目と同じ形の別セル。日本語は 6/6 正しい。

**`in` を共有の `en_op_var_head` に入れるのは誤り**——あれは 2a-sym も読む共有語彙で、
`multiply both sides of E2 by A` の因子まで守って日英が割れる（実測済み、
`supervisor_alias_rewrite` の註）。だから **solve 一族だけの表**
（`en_solve_var_head`）にして、3鍵がそれを読む。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _run(lang, lines, domain="classical_mechanics"):
    async def go():
        nb.start_engine(title="svh", mode="standard", language=lang, domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            out = []
            for line in lines:
                out.append(await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup))
            ops = [(s.operation, getattr(s, "operation_args", None))
                   for s in engine.steps.values()
                   if s.kind == "cas_operation" and s.operation]
            exprs = {k: v.latex for k, v in engine.expressions.items()}
            return out[-1], (ops[-1] if ops else (None, None)), exprs
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_SYS = ["x + y = 3", "x - y = 1"]
_ONE = ["2 x - 4 = 0"]
#: 英語の「変数を導く語」——3鍵が同じ4つを受けなければならない。
_EN_HEADS = ["for", "with respect to", "w.r.t.", "in"]
#: 日本語の同じ表（ずっと1つだった側）。
_JA_PARTICLES = ["について", "に関して"]


class TestTheThreeKeysReadTheSameHeads:
    @pytest.mark.parametrize("head", _EN_HEADS)
    @pytest.mark.parametrize("setup,tmpl,want", [
        (_ONE, "solve E1 {h} x", "solve_equation"),
        (_ONE, "compute the solution of E1 {h} x", "solve_equation"),
        (_SYS, "solve E1 and E2 {h} x and y", "solve_system"),
    ])
    def test_english(self, setup, tmpl, want, head):
        r, (op, _), _ = _run("en", setup + [tmpl.format(h=head)])
        assert r.get("status") == "completed", (tmpl.format(h=head), r)
        assert op == want, (tmpl.format(h=head), op)

    @pytest.mark.parametrize("particle", _JA_PARTICLES)
    @pytest.mark.parametrize("setup,tmpl,want", [
        (_ONE, "E1 を x {p} 解いて", "solve_equation"),
        (_ONE, "E1 の解を x {p} 求めて", "solve_equation"),
        (_SYS, "E1 と E2 を x と y {p} 解いて", "solve_system"),
    ])
    def test_japanese_is_unchanged(self, setup, tmpl, want, particle):
        r, (op, _), _ = _run("ja", setup + [tmpl.format(p=particle)])
        assert r.get("status") == "completed", (tmpl.format(p=particle), r)
        assert op == want, (tmpl.format(p=particle), op)


class TestTheSystemIsNotAnsweredByOneEquation:
    r"""**値まで見る。** `solve E1 and E2 in x and y` は `x = 3-y` を `completed` で
    返していた——連立の指示に、1本目だけを解いた答えが成功として返る形である。"""

    @pytest.mark.parametrize("head", _EN_HEADS)
    def test_both_variables_are_solved(self, head):
        _, _, exprs = _run("en", _SYS + [f"solve E1 and E2 {head} x and y"])
        got = {(v or "").replace(" ", "") for v in exprs.values()}
        assert "x=2" in got and "y=1" in got, (head, exprs)

    @pytest.mark.parametrize("particle", _JA_PARTICLES)
    def test_the_japanese_sibling_too(self, particle):
        _, _, exprs = _run("ja", _SYS + [f"E1 と E2 を x と y {particle} 解いて"])
        got = {(v or "").replace(" ", "") for v in exprs.values()}
        assert "x=2" in got and "y=1" in got, (particle, exprs)


class TestTheOtherWordOrderIsUntouched:
    r"""`solve for x in E1` では `in` が**式のほう**を導く。広げた表がこれを
    壊していないこと（四十五問目が作った枝）。"""

    @pytest.mark.parametrize("lines,want", [
        (_ONE + ["solve for x in E1"], "solve_equation"),
        (_SYS + ["solve for x and y in E1 and E2"], "solve_system"),
        (_ONE + ["find x from E1"], "solve_equation"),
    ])
    def test_it_still_reaches_the_same_operation(self, lines, want):
        r, (op, _), _ = _run("en", lines)
        assert r.get("status") == "completed", r
        assert op == want, (lines[-1], op)


class TestTheHeadIsTheSolveFamilysOwnTable:
    r"""**共有の `en_op_var_head` に `in` を入れてはならない**——2a-sym も読むので
    `multiply both sides of E2 by A` の因子まで守り、日英が割れる（実測済み）。"""

    def test_the_shared_table_does_not_spell_in(self):
        from notebook.runtime.nl_patterns import fragment
        assert r"\bin" not in fragment("en_op_var_head")
        assert r"\bin" in fragment("en_solve_var_head")

    def test_all_three_keys_read_it(self):
        import json
        import pathlib
        pats = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
        for key in ("solve_single", "solve_solution_of_var", "solve_system_vars"):
            assert "{{en_solve_var_head}}" in pats[key]["en"], key
            assert "{{en_op_var_head}}" not in pats[key]["en"], key
