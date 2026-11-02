r"""ENGLISH-INLINE-INTEGRAL (2026-09-18) — 微分には枠が在り、積分には無かった。

英語は全 ASCII なので、日本語の「字種の変わり目」（`_FORMULA_OP_RE`）が原理的に
発火しない。だから英語には**操作ごとの枠**が要る——`_DIFF_INLINE_FORMULA_EN_RE`
（`differentiate <式> with respect to <変数>`）はそのために書かれた（2026-08-22、
`simplify a x^2 + b x + c` が `expr='x^2'` を `completed` で返していた、72綴り中63）。
積分側の枠は無く、被積分関数が式として登録されないまま**欄が位置で埋まり、変数と
入れ替わって**いた（実測 2026-09-18、`y = x^2` が直前）:

| 打った文 | 実際に走った操作 | 返った値 |
|---|---|---|
| `integrate a x^2 over x` | `integrate(expr='x^2', var=**'a'**)` | **`a*x^2+C`** |
| `integrate b over x` | `integrate(expr=**'E1'**, var='b')` | **`b*x^2+C`** |
| `integrate x^2 + 1 over x` | `integrate(expr='x^2', var='1')` | 失敗 |

英語3綴り × 被積分関数5つの **15セル中 正しいのは3**（原子1つの `x^3` だけ）、
**6が黙って誤答**、6が失敗。**日本語は同じ族で正しい。**

**前置詞は共有の表を読む**（`en_op_var_head_wide`）。`def_integral[en]`——**境界つきの
同じ文**——が読んでいるのと同じ問いで、微分の枠だけが手書きの部分集合を持っていた。
境界つきと境界なしで綴りが割れれば、`from 0 to 1` を足しただけで読める前置詞が
変わることになる（このファイルがそれを固定する）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   _integrate_inline_formula, run_interpret_pipeline)


def _run(lang, lines, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="inline-int", mode="standard", language=lang,
                        domains=[domain])
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
            ops = [(getattr(s, "operation", None), getattr(s, "operation_args", None))
                   for s in engine.steps.values()
                   if s.kind == "cas_operation" and getattr(s, "operation", None)]
            return out[-1], (ops[-1] if ops else (None, None))
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


#: 直前の式。すり替わると答えはこれに由来する。
_SETUP = ["y = x^2"]

#: 被積分関数 → `x` で積分した値（CAS の綴りそのまま）。
_INTEGRANDS = {
    "x^3":     "x^4/4+C",
    "a x^2":   "(a*x^3)/3+C",
    "x^2 + 1": "x^3/3+x+C",
    "2 x":     "x^2+C",
}
_EN_SPELLINGS = ["integrate {0} over x", "integrate {0} with respect to x",
                 "the integral of {0} over x"]


class TestTheIntegrandIsWhatWasWritten:
    @pytest.mark.parametrize("spelling", _EN_SPELLINGS)
    @pytest.mark.parametrize("integrand,want", list(_INTEGRANDS.items()))
    def test_english(self, spelling, integrand, want):
        r, (op, args) = _run("en", _SETUP + [spelling.format(integrand)])
        assert r.get("status") == "completed", r
        assert _value(r) == want, (r, args)
        assert (args or {}).get("var") == "x", args      # 変数は研究者が書いたほう

    @pytest.mark.parametrize("integrand,want", list(_INTEGRANDS.items()))
    def test_japanese_is_unchanged(self, integrand, want):
        r, (op, args) = _run("ja", _SETUP + [f"{integrand} を x で積分して"])
        assert r.get("status") == "completed", r
        assert _value(r) == want, (r, args)


class TestABareUndefinedNameIsStillRefused:
    """前周の決定（OPERAND-BOUNDARY 残⑤）——枠が入っても名前は名前である。"""

    @pytest.mark.parametrize("lang,text,name", [
        ("en", "integrate b over x", "b"),
        ("en", "integrate zvar over x", "zvar"),
        ("ja", "b を x で積分して", "b"),
    ])
    def test_it_names_the_name(self, lang, text, name):
        r, _ = _run(lang, _SETUP + [text])
        assert r.get("status") == "failed", r
        assert name in (r.get("error") or ""), r


class TestTheBoundedSpellingsKeepTheirOwnReading:
    r"""より特定の読みに道を譲る——`from … to …` と `over the region …` は専用の
    インターセプタが持っていて、**実測で正しく動く**（E-id の被積分関数も受ける）。"""

    @pytest.mark.parametrize("lines,op,want", [
        (["integrate x^2 over x from 0 to 1"], "integrate_definite", "1/3"),
        (["integrate a x^2 over x from 0 to 1"], "integrate_definite", "a/3"),
        (["y = x^2", "integrate E1 over x from 0 to 1"], "integrate_definite", "1/3"),
    ])
    def test_the_definite_integral(self, lines, op, want):
        r, (got_op, _) = _run("en", lines)
        assert r.get("status") == "completed", r
        assert got_op == op, (got_op, r)
        assert _value(r) == want, r

    def test_the_region_integral_still_registers_display_only(self):
        r, (op, _) = _run("en", ["integrate x^2 + y^2 over the region x^2 + y^2 <= 1"])
        assert r.get("status") == "registered", r
        assert op is None, op

    @pytest.mark.parametrize("text", [
        "integrate x^2 over x from 0 to 1",
        "integrate x^2 + y^2 over the region x^2 + y^2 <= 1",
    ])
    def test_the_frame_itself_declines_them(self, text):
        assert _integrate_inline_formula(text) is None, text


class TestOnePrepositionTableForBothReadings:
    r"""境界つきと境界なしは**同じ文**である。綴りが割れれば、`from 0 to 1` を足した
    だけで読める前置詞が変わる。"""

    @pytest.mark.parametrize("prep", ["over", "with respect to", "w.r.t.", "by", "in", "for"])
    def test_bounded_and_unbounded_agree(self, prep):
        bounded, (op_b, _) = _run("en", [f"integrate a x^2 {prep} x from 0 to 1"])
        unbounded, (op_u, _) = _run("en", [f"integrate a x^2 {prep} x"])
        assert bounded.get("status") == "completed", bounded
        assert unbounded.get("status") == "completed", unbounded
        assert op_b == "integrate_definite" and op_u == "integrate_indefinite"
        assert _value(unbounded) == "(a*x^3)/3+C", unbounded


class TestTheDerivativeFrameIsUntouched:
    @pytest.mark.parametrize("text,want", [
        ("differentiate a x^2 with respect to x totally", "2*a*x"),
        ("differentiate x^3 with respect to x totally", "3*x^2"),
    ])
    def test_it_still_reads_its_own_sentence(self, text, want):
        r, _ = _run("en", _SETUP + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheFrameCutsWhereTheGrammarSays:
    @pytest.mark.parametrize("text,want", [
        ("integrate a x^2 over x", ("integrate", "a x^2", "over x")),
        ("the integral of b over x", ("the integral of", "b", "over x")),
        ("integrate x^2 + 1 with respect to x", ("integrate", "x^2 + 1", "with respect to x")),
        ("definite integrate f over x", ("definite integrate", "f", "over x")),
    ])
    def test_leadin_formula_trailing(self, text, want):
        m = _integrate_inline_formula(text)
        assert m is not None, text
        assert m.groups() == want, (text, m.groups())

    @pytest.mark.parametrize("text", ["simplify a x^2", "differentiate a x^2 with respect to x",
                                      "integrate", "integrate x^2"])
    def test_what_is_not_this_frame(self, text):
        assert _integrate_inline_formula(text) is None, text
