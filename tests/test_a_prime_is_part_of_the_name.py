r"""PRIMED-VAR-NOT-A-VARIABLE（2026-09-24）——プライムは名前の一部である。

`--asymmetry` の残り1件（`diff_rule`）を81セルで測り直している途中、**変数の綴りを
振ったら出た**。`canonical_variables` の識別子走査が `[A-Za-z_]\w*` で `'` を含まず、
`y'^3` の変数が `{y}` と読まれる。すると `differentiate_is_ambiguous` は
「微分変数 `y'` ではない自由な記号 `y` が残っている」と判定し——

| 打った文（`y'^3` を登録したあと） | 返っていたもの |
|---|---|
| `E1 を y' で微分して` / `differentiate E1 with respect to y'` | **「偏微分か全微分か判別できません」** |
| `E1 を y' で偏微分して` | `3*y'^2`（種別を書けば通る） |
| `x^3` / `x_1^3` / `\bar{y}^3` / `\theta^3` の同じ文 | どれも訊き返さない |

**記号が1つしかない式に対して、在りえない区別を訊き返していた**（日英とも、
classical_mechanics でも electromagnetism でも）。どちらの読みでも「曖昧ではない」が
正しい——プライムが**別の変数**の印なら（この系自身の規則。`t'`・`\sigma'` は
グリーン関数の普通の綴り）式の唯一の記号は `var` そのもの。プライムが `dy/dx` の
意味なら、式に `y` は単独で現れていない。

**答えは系の別の層が既に持っていた**: パーサは `'` を識別子の一部として扱う
（`f'(x)`、CLAUDE.md「Math Input Conventions」）。走査だけが知らなかった。

**表示側も同じ原因で壊れていた。** 生成された `\frac{\partial}{\partial y'}` が
読み戻せず、「d と2文字以上の変数名の間に区切りがありません（「d x」のように空白を
入れてください）」——**研究者が破っていない規則**の警告が、この系自身が書いた表示に
ついて出ていた（BP-9）。

**広げたら出口が1つ増えた。** プライム付きの変数が `differentiate` に届いた途端、
同じモジュールが `\frac{d}{dy'}`（区切り無し）を生成しはじめ、今度はそちらを
読み戻せなくなった。`dy'` は `d` + `y'` であり、`dy` が `d` + `y` である理由
（1文字＋装飾）がそのまま当てはまる。

**Julia 側は逆で、変えていない**——あちらのプライムは随伴で、`a'(k)` は演算子 `a`
である（`_JULIA_OP_RE` が明示的に剥がしている）。同じ綴りが別の問いに答えている。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_derivative_latex import (
    REASON_NO_SEPARATOR, restore_derivative_display_latex)
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.lhs_formatting import _format_provenance_lhs
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

DOMAIN = "classical_mechanics"


def _handler():
    return NotebookEngine()._handler


def _run(lang, setup, line):
    async def go():
        nb.start_engine(title="prime", mode="standard", language=lang, domains=[DOMAIN])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[DOMAIN],
                                primary_backend=engine.session.primary_backend)
            for text in (setup, line):
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
            values = [e.canonical for e in engine.expressions.values()]
            return resp, (values[-1] if values else "")
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheScanSeesThePrime:
    """`canonical_variables` は「この式に出てくる名前は何か」に答える。"""

    @pytest.mark.parametrize("canonical,want", [
        ("y'^3",      {"y'"}),
        # Maxima escapes the quote in a name it RETURNS.  Without removing it
        # the scan sees `y` again and the SECOND derivative asks the question
        # all over again (measured 2026-09-24 — a widening opens a new exit).
        (r"3*y\'^2",  {"y'"}),
        (r"2*t\'*t",  {"t", "t'"}),
        ("y'^2 + y",  {"y'", "y"}),
        ("t'*t",      {"t'", "t"}),
        ("x^3",       {"x"}),
        ("sigma'^3",  {"sigma'"}),
    ])
    def test_maxima(self, canonical, want):
        assert _handler().canonical_variables(canonical) == want, canonical

    def test_julia_strips_the_prime_on_purpose(self):
        """あちらでは `'` は随伴の印で、`a'(k)` は演算子 `a` である。

        同じ綴りに**別の問い**が乗っているので、片方に合わせてはならない。
        """
        from notebook.runtime.julia_handler_common import _JULIA_OP_RE
        assert [m.group(1) for m in _JULIA_OP_RE.finditer("a'(k) * a(k)")] == ["a", "a"]


class TestOneSymbolIsNotAmbiguous:
    r"""記号が1つしかない式に「偏微分か全微分か」は訊けない。"""

    @pytest.mark.parametrize("setup,var", [
        ("y'^3",       "y'"),
        ("t'^3",       "t'"),
        (r"\sigma'^3", r"\sigma'"),
    ])
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_primed_variable_computes(self, setup, var, lang):
        line = (f"E1 を {var} で微分して" if lang == "ja"
                else f"differentiate E1 with respect to {var}")
        resp, value = _run(lang, setup, line)
        assert resp.get("status") == "completed", (line, resp.get("error"))
        assert value.startswith("3*"), (line, value)

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_second_derivative_of_the_result_computes(self, lang):
        """1階の結果をもう一度微分する——CAS が返した綴りで同じ問いが立つ。"""
        line = ("E2 を y' で微分して" if lang == "ja"
                else "differentiate E2 with respect to y'")
        async def go():
            nb.start_engine(title="p2", mode="standard", language=lang, domains=[DOMAIN])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=[DOMAIN],
                                    primary_backend=engine.session.primary_backend)
                first = ("E1 を y' で微分して" if lang == "ja"
                         else "differentiate E1 with respect to y'")
                for text in ("y'^3", first, line):
                    resp = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        engine, sup)
                return resp, list(engine.expressions.values())[-1].canonical
            finally:
                nb.stop_engine()
        resp, value = asyncio.run(go())
        assert resp.get("status") == "completed", (line, resp.get("error"))
        assert value.replace("\\", "") == "6*y'", value

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_real_ambiguity_is_still_asked(self, lang):
        """`y'^2 + y` を `y'` で微分するなら `y` が残る——訊くのが正しい。"""
        line = ("E1 を y' で微分して" if lang == "ja"
                else "differentiate E1 with respect to y'")
        resp, _ = _run(lang, "y'^2 + y", line)
        assert resp.get("status") == "failed", line

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_base_name_still_matches_the_variable(self, lang):
        """`y'` を `y` で微分するのは**今までどおり**曖昧ではない。

        `_d+ot$` を剥がして `q_dot` を `q` と同一視する行がすぐ隣に在り、
        プライムはこの物理では同じ約束（q と q̇ は独立原子）。剥がさないと、
        プライムが走査に見えるようになった途端この答えが変わってしまう。
        """
        line = ("E1 を y で微分して" if lang == "ja"
                else "differentiate E1 with respect to y")
        resp, value = _run(lang, "y'^2 + y", line)
        assert resp.get("status") == "completed", (line, resp.get("error"))
        assert value == "1", value


class TestADefinedNameIsNotRewrittenInsideAPrimedName:
    r"""2a-sym は `y` を式番号に書き換える——`y'` の中の `y` は別の名前である。

    **回帰シナリオの replay が捕まえた**（起票の時点では見えていなかった）。
    このファイルは前半で `y = x^3` を書くので `y` が別名表に載り、そのあとの
    `E78 を y' で微分して` が変数 `E18'` に書き換わって、研究者が書いていない
    鍵を名指す断りが返っていた。**英語だけ無事だった**のは、英語の枠が `\b` で
    終わるので変数スロットの guard が先に効いたから——日本語の枠は記号の直後に
    助詞を求め、プライムがその間に立つ。

    直したのは枠ではなく**境界**である。上付きラベル（`G^{(HF)}`）と下付きに
    ついて「別の名前になる」と同じ行が既に言っており、プライムも同じ。
    """

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_prime_survives_the_alias_rewrite(self, lang):
        line = ("E2 を y' で微分して" if lang == "ja"
                else "differentiate E2 with respect to y'")
        async def go():
            nb.start_engine(title="rw", mode="standard", language=lang, domains=[DOMAIN])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=[DOMAIN],
                                    primary_backend=engine.session.primary_backend)
                for text in ("y = x^3", "y'^3", line):
                    resp = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        engine, sup)
                return resp, list(engine.expressions.values())[-1].canonical
            finally:
                nb.stop_engine()
        resp, value = asyncio.run(go())
        assert resp.get("status") == "completed", (line, resp.get("error"))
        assert value.replace("\\", "") == "3*y'^2", value

    def test_the_bare_name_is_still_rewritten(self):
        """狭めたのは**プライムが続く場合だけ**——素の `y` は今までどおり書き換わる。"""
        from types import SimpleNamespace

        from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids
        eng = SimpleNamespace(lhs_alias_map={"y": "E1"})
        assert rewrite_symbols_to_eids("y を簡約して", eng) == "E1 を簡約して"
        assert rewrite_symbols_to_eids("y' を簡約して", eng) == "y' を簡約して"


class TestTheGeneratedDisplayReadsBack:
    r"""この系が書いた表示は、この系が読み戻せること（BP-9 の往復）。"""

    def _restore(self, s, refusals=None):
        on_refused = (lambda frac, reason: refusals.append((frac, reason))
                      ) if refusals is not None else None
        return restore_derivative_display_latex(s, lambda x: x, on_refused=on_refused)

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{\partial}{\partial y'}\left(f\right)", "'diff(f,y',1)"),
        (r"\frac{d}{dy'}\left(f\right)",                "'diff(f,y',1)"),
        (r"\frac{d}{d y'}\left(f\right)",               "'diff(f,y',1)"),
        (r"\frac{d^{2}}{dy'^{2}}\left(f\right)",        "'diff(f,y',2)"),
    ])
    def test_primed_forms(self, latex, canonical):
        assert self._restore(latex) == canonical, latex

    def test_against_the_real_generator(self):
        """表示の書式が変わってもここで落ちるよう、**生成器に訊く**。"""
        h = _handler()
        latex = _format_provenance_lhs("f", ["y'"], False, qty=None,
                                       parse_ir=h.parse_ir)
        assert h.latex_to_canonical(latex) == "'diff(f,y',1)", latex

    def test_a_multiletter_unseparated_base_is_still_refused(self):
        r"""`dxy'` は `d`+`xy'` とも名前 `dxy'` とも読める——**決めない**。

        プライムを許したのは「1文字＋装飾」の枠内だけである。
        """
        src = r"\frac{d}{dxy'}\left(f\right)"
        refusals = []
        assert self._restore(src, refusals) == src
        assert [r for _, r in refusals] == [REASON_NO_SEPARATOR]

    def test_a_greek_variable_reads_back(self):
        """DERIV-DISPLAY-CANONICAL-NAME はこの翌に解決した（strict xfail が XPASS
        してスイートを落とし、それが合図になった）。族の回帰は
        `tests/test_a_derivative_label_spells_the_researchers_name.py`。"""
        h = _handler()
        latex = _format_provenance_lhs("f", ["theta"], False, qty=None,
                                       parse_ir=h.parse_ir)
        assert h.latex_to_canonical(latex) == "'diff(f,theta,1)", latex
