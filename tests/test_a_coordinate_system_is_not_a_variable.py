r"""INTEGRATE-IN-COORDINATES (2026-09-18) — `in polar coordinates` は副詞句である。

共有の前置詞表は `in` を変数の導入語として持つ（`integrate f in x` は読めるべき
綴り）。そのため `in polar coordinates` の `polar` が**変数の欄**に入り、研究者が
変数として書いていない記号が答えに立っていた——実測 2026-09-18、`completed`:

    integrate x^2 in polar coordinates       →  polar*x^2+C
    integrate x^2 in spherical coordinates   →  spherical*x^2+C
    integrate x^2 in cylindrical coordinates →  cylindrical*x^2+C

**日本語は同じ綴りが声を上げて失敗していた**（`x^2 を極座標で積分して`）ので、
黙っていたのは英語だけである。

**判別は語彙ではなく性質**——`coordinates` / `座標` が後ろに付く語は座標系であって
変数ではない。語彙の表にすると、研究者が `define_coordinate_system` で定義した
名前が漏れる。

**そして起票で私が案内した道は、測ると別のものだった。** `convert E1 to polar
coordinates` は `to_polar`——**複素数の極形式**（`polarform(...)`）で、座標変換では
ない。断り文が名指すのは、シナリオ `scenario_coordinate_jacobian` が実際に打って
いる2段の綴りで、日英とも `%pi/2` に落ちることを測ってある。

**同じ表の割れが微分側にも生きていた。** `_DIFF_INLINE_FORMULA_EN_RE` は前置詞を
手書きで持っており（`with respect to|w.r.t.|by`）、`in` / `for` / `over` は枠に
当たらず、欄が位置で埋まって `differentiate(expr='x^2', var='a')` → `0` を
`completed` で返していた（6前置詞中3）。共有表 `en_op_var_head_wide` へ移して 6/6。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   _diff_inline_frame, run_interpret_pipeline)


def _run(lang, lines, domain="applied_mathematics"):
    async def go():
        nb.start_engine(title="coords", mode="standard", language=lang, domains=[domain])
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


class TestACoordinateSystemDoesNotFillTheVariable:
    @pytest.mark.parametrize("system", ["polar", "spherical", "cylindrical", "elliptic"])
    @pytest.mark.parametrize("spelling", [
        "integrate x^2 in {0} coordinates",
        "integrate x^2 in {0} coordinates from 0 to 1",
        "the integral of x^2 in {0} coordinates",
    ])
    def test_english_refuses_with_the_road(self, system, spelling):
        r, (op, args) = _run("en", [spelling.format(system)])
        assert r.get("status") == "failed", r
        assert op is None, (op, args)
        assert system in (r.get("error") or ""), r
        # **実測で通る道**（シナリオ `scenario_coordinate_jacobian` の綴り）
        assert "Change variables" in (r.get("error") or ""), r

    @pytest.mark.parametrize("text", ["x^2 を極座標で積分して", "x^2 + y^2 を球座標で定積分して",
                                      "x^2 を極座標系で積分して"])
    def test_japanese_gains_a_reason(self, text):
        """日本語は元から失敗していたが、理由が無かった。"""
        r, _ = _run("ja", [text])
        assert r.get("status") == "failed", r
        assert "変数変換" in (r.get("error") or ""), r


class TestTheRoadTheRefusalNamesIsMeasured:
    """案内した回避策が効くかは、断りの一部である。"""

    @pytest.mark.parametrize("lang,lines", [
        ("en", ["definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1",
                "Change variables in E1 from x,y to r,theta via "
                "x=r*cos(theta), y=r*sin(theta)",
                "evaluate this integral"]),
        ("ja", ["x^2 + y^2 <= 1 の領域で x^2 + y^2 を定積分して",
                "E1 を x,y から r,theta へ x=r*cos(theta), y=r*sin(theta) で変数変換して",
                "この定積分を評価して"]),
    ])
    def test_it_lands_on_the_value(self, lang, lines):
        r, _ = _run(lang, lines)
        assert r.get("status") == "completed", r
        assert _value(r) == "%pi/2", r


class TestAVariableInThatPositionStillReads:
    """`in` は変数の導入語である——座標系だけを外す。"""

    @pytest.mark.parametrize("lang,text,want", [
        ("en", "integrate a x^2 in x", "(a*x^3)/3+C"),
        ("en", "integrate a x^2 over x", "(a*x^3)/3+C"),
        ("ja", "a x^2 を x で積分して", "(a*x^3)/3+C"),
    ])
    def test_it_computes(self, lang, text, want):
        r, (_, args) = _run(lang, [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r
        assert (args or {}).get("var") == "x", args


class TestOnePrepositionTableForTheDerivativeToo:
    r"""枠が手書きの部分集合を持っていた代償——`in` / `for` / `over` が
    `differentiate(expr='x^2', var='a')` → `0` を `completed` で返していた。"""

    @pytest.mark.parametrize("prep", ["with respect to", "w.r.t.", "by", "in", "for", "over"])
    def test_every_preposition_reads_the_same_sentence(self, prep):
        r, (_, args) = _run("en", ["y = x^2", f"differentiate a x^2 {prep} x totally"])
        assert r.get("status") == "completed", r
        assert _value(r) == "2*a*x", (r, args)
        assert (args or {}).get("var") == "x", args

    def test_the_frame_reads_the_shared_table(self):
        """手書きに戻すと、この4つのうち3つが枠に当たらなくなる。"""
        rx = _diff_inline_frame()
        for prep in ("with respect to", "in", "for", "over"):
            m = rx.match(f"differentiate a x^2 {prep} x")
            assert m is not None and m.group("formula") == "a x^2", (prep, m)


class TestWhatTheCoordinatePatternClaims:
    @pytest.mark.parametrize("lang,text,want", [
        ("en", "integrate x^2 in polar coordinates", ("x^2", "polar")),
        ("en", "the integral of x^2 in cylindrical coordinates", ("x^2", "cylindrical")),
        ("ja", "x^2 + y^2 を球座標で定積分して", ("x^2 + y^2", "球")),
    ])
    def test_it_reads_integrand_and_system(self, lang, text, want):
        m = nlp("integral_in_coordinates", lang).match(text)
        assert m is not None, text
        assert (m.group("integrand"), m.group("system")) == want, text

    @pytest.mark.parametrize("lang,text", [
        ("en", "integrate a x^2 in x"),
        ("en", "integrate x^2 over x"),
        ("en", "definite-integrate x^2+y^2 over the region x^2+y^2<=1"),
        ("en", "Change variables in E1 from x,y to r,theta via x=r*cos(theta), y=r*sin(theta)"),
        ("ja", "x^2 を x で積分して"),
        ("ja", "x^2 + y^2 <= 1 の領域で x^2 + y^2 を定積分して"),
    ])
    def test_what_it_must_not_claim(self, lang, text):
        assert nlp("integral_in_coordinates", lang).match(text) is None, text
