r"""DEFINED-NAME-VAR-REWRITTEN (2026-09-19) — 読みが受ける綴りは guard も知る。

2a-sym（`ui/supervisor_alias_rewrite.py`）は研究者の記号名を式番号へ書き換える。
その記号が**操作の変数**として立っているときは書き換えてはならず、guard が
それを見ている——が、その guard は**枠の手書きの一覧**で、grammar の枠より狭かった。

**症状は「書いてある欄について『引数が不足』と言われる」。** 実測 2026-09-19、
`2 y = 8` を `y` について解いたあと（`y` は別名表に載り、CAS には束縛されない）:

| 打った文 | 返っていたもの |
|---|---|
| `E1 を y で積分して` / `integrate E1 over y` | 「必要な引数が不足しています: ['var']」 |
| `E1 の y に関する極限を 0 で求めて` | 同上 |
| `E1 を y について部分分数分解して` / `collect E1 in y` | 同上 |
| `E1 の y = 1 における留数を求めて` / `the residue of E1 at y = 1` | 同上 |
| `E1 の y を u に変数変換して` | 同上 |
| `substitute y = 3 into E1` | 同上 |

**同じ文が `y` を解く前には通る**——それがこの検査の形である（FREE と SOLVED の
2列を比べる）。定積分とテイラー展開は無傷だった: 読みがインターセプタで**生の
テキスト**を見るからで、2a-sym の出力を読む綴りだけが落ちていた。

**この穴は3度目である。** `_ja_op_var_tail_note` が微分（二十二問目）と solve
（2026-09-06）について同じことを記録している。だから直し方は枠を1つ足すことでは
なく、**枠の表**（`_NAME_SLOT_FRAMES`）にして「読みを1つ足したら枠を1つ足す」を
1箇所で言えるようにすることである。

**出口は足すたびに数え直す。** 枠を広げた瞬間 `E1 の y を u に変数変換して` が
CAS まで届くようになり、**生の `Maxima evaluation error`** が出た（`old_var` は
前日の `VARIABLE_FIELDS` に入っていなかった）。
"""
import asyncio
import types

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)
from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids


def _run(lang, lines, domain="classical_mechanics"):
    async def go():
        nb.start_engine(title="vr", mode="standard", language=lang, domains=[domain])
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
            return out[-1]
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


def _msg(r):
    return str(r.get("error") or r.get("message") or "")


#: `y` はどこにも登録されていない（対照）。
_FREE = ["z^2 + y"]
#: `y` は**解の結果**——別名表に載り、CAS には束縛されない。2a-sym が働く状態で
#: あって、操作そのものは通らなければならない。**枠は族の言語で書く**（英語
#: セッションに日本語の setup を混ぜると `y` が登録されず、測定が嘘をつく）。
_SOLVED = {"ja": ["z^2 + y", "2 y = 8", "E2 を y について解いて"],
           "en": ["z^2 + y", "2 y = 8", "solve E2 for y"]}

#: 実測で割れていた綴りと、その正しい値。
_BROKEN = [
    ("ja", "E1 を y で積分して", "y*z^2+y^2/2+C"),
    ("ja", "E1 の y に関する極限を 0 で求めて", "z^2"),
    ("ja", "E1 を y について部分分数分解して", "z^2+y"),
    ("ja", "E1 の y = 1 における留数を求めて", "0"),
    ("en", "integrate E1 over y", "y*z^2+y^2/2+C"),
    ("en", "collect E1 in y", "z^2+y"),
    ("en", "the residue of E1 at y = 1", "0"),
    ("en", "substitute y = 3 into E1", "z^2+3"),
]


class TestSolvingForAVariableDoesNotBreakTheSentence:
    @pytest.mark.parametrize("lang,text,want", _BROKEN)
    def test_it_still_reaches_the_operation(self, lang, text, want):
        r = _run(lang, _SOLVED[lang] + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r

    @pytest.mark.parametrize("lang,text,want", _BROKEN)
    def test_the_control_column_is_unchanged(self, lang, text, want):
        """`y` を解く前——ここが変わってはならない（比べる相手である）。"""
        r = _run(lang, _FREE + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheChangeOfVariablesFrameNeedsAnIntegralOperand:
    r"""`E1 の y を u に変数変換して` もこの guard が守る枠の1つだが、**被演算子は
    積分でなければならない**（CHANGE-VARIABLE-NOOP、同日に直した）。

    この行はもともと `_BROKEN` に `z^2+y` を期待値として入っていた——それは
    `changevar` が**何もしないで返した**値であり、**silent-wrong を正解として
    固定していた**（自分が前日に書いた）。実装から書いた期待値は、実装の誤りを
    正解と宣言する。
    """

    @pytest.mark.parametrize("setup", [
        ["z^2 + y", "2 y = 8", "E2 を y について解いて"],   # 解いたあと
        ["z^2 + y"],                                      # 対照
    ])
    def test_the_name_is_not_rewritten(self, setup):
        """2a-sym が `y` を式番号へ書き換えれば、断り文が別のことを言う。"""
        r = _run("ja", setup + ["E1 の y を u に変数変換して"])
        assert r.get("status") == "failed", r
        assert "積分" in _msg(r), r        # 「積分が書かれていない」＝欄は届いている
        assert "引数が不足" not in _msg(r), r

    def test_with_an_integral_it_computes_after_a_solve(self):
        r = _run("ja", [r"\int y^2 \, dy", "2 y = 8", "E2 を y について解いて",
                        "E1 の y を u に変数変換して"])
        assert r.get("status") == "completed", r
        assert _value(r) == "'integrate(u^2,u)", r


class TestTheGuardIsATableOfFrames:
    """単体で——名前が式番号へ化けないこと。枠ごとに1件ずつ。"""

    _ENGINE = types.SimpleNamespace(lhs_alias_map={"y": "E9"})

    @pytest.mark.parametrize("text", [
        # 日本語: 助詞 + 操作の語
        "E1 を y で積分して",
        "E1 を y で定積分して",
        "E1 の y に関する極限を 0 で求めて",
        "E1 を y について部分分数分解して",  # 日本語側は読める（英語の同義文は読めない）
        "E1 を y についてまとめて",
        # 日本語: 記号 = 値 … 留数／代入
        "E1 の y = 1 における留数を求めて",
        # 日本語: 変数変換（旧・新の両方の位置）
        "E1 の y を u に変数変換して",
        "E1 の x を y に変数変換して",
        # 英語: 操作の節 + 前置詞
        "integrate E1 over y",
        "collect E1 in y",
        "expand E1 in y around 0 to order 3",
        "differentiate E1 by y totally",
        # 英語: 代入／留数の節 + 記号 =
        "substitute y = 3 into E1",
        "the residue of E1 at y = 1",
        # 既存の枠（退行させない）
        "differentiate E1 with respect to y totally",
        "solve E1 for y",
        "solve for y in E1",
        "find y from E1",
        "E1 を y について解いて",
        "E1 から y を求めて",
    ])
    def test_the_name_survives(self, text):
        assert "E9" not in rewrite_symbols_to_eids(text, self._ENGINE), text

    @pytest.mark.parametrize("text", [
        # 名前が**対象**として立っている文——ここは書き換わらなければならない
        # （2a-sym の存在理由そのもの）。
        "E1 と y の交換子を求めて",
        "y を簡約して",
        "y の発散を求めて",
        "simplify y",
        "the divergence of y",
        # 両辺に掛ける**因子**は変数ではない（`en_op_var_head` を広げてはならない
        # と実測が言っている当の形）。
        "multiply both sides of E1 by y",
        "E1 の両辺を y で割って",
    ])
    def test_a_name_in_another_role_is_still_rewritten(self, text):
        assert "E9" in rewrite_symbols_to_eids(text, self._ENGINE), text

    def test_every_frame_is_used(self):
        """表が飾りでないこと——枠はすべて上のどれかの文に当たる。"""
        from ui.supervisor_alias_rewrite import _NAME_SLOT_FRAMES
        assert len(_NAME_SLOT_FRAMES) >= 10


class TestAWiderGuardOpensANewExit:
    r"""枠を広げると、これまで手前で止まっていた文が CAS まで届く。

    実測 2026-09-19: `E1 の y を u に変数変換して` は `y = x^3` のあと**生の
    `Maxima evaluation error`** になった——`old_var` は前日の `VARIABLE_FIELDS`
    に入っていなかった。出口は足すたびに数え直す。
    """

    def test_the_change_of_variables_refusal_is_translated(self):
        # **被演算子は積分でなければならない**（CHANGE-VARIABLE-NOOP、同日）。
        # 素の式は「積分が書かれていない」という**より手前の**理由で断られる
        # ようになったので、CAS まで届く形——積分が在る形——で測る。
        r = _run("ja", [r"\int y^2 \, dy", "y = x^3", "E1 の y を u に変数変換して"])
        assert r.get("status") == "failed", r
        assert "Maxima" not in _msg(r), r
        assert "y" in _msg(r) and "x^3" in _msg(r) and "E2" in _msg(r), r

    def test_the_new_variable_is_not_in_the_set_because_the_cas_does_not_refuse(self):
        """`new_var` を入れなかった理由——測ると CAS は断らない（変換して返す）。"""
        r = _run("ja", [r"\int x^2 \, dx", "y = x^3", "E1 の x を y に変数変換して"])
        assert r.get("status") == "completed", r
