r"""OPERAND-BOUNDARY 残⑤ の族 (2026-09-18) — 解決しない名前も「文が名指したもの」。

2026-09-11 の「名指したものを黙ってすり替えない」guard は、**書かれた式番号**と
**宣言済みの対象**を数える。数えていなかったのは、**ただ未定義なだけの名前**で、
そこだけ同じ silent-wrong が生き残っていた（実測 2026-09-18、`y = x^2` が直前）:

| 打った文 | 実際に走った操作 | 返った値 |
|---|---|---|
| `zvar を簡約して` / `simplify zvar` | `simplify(expr=**E1**)` | `x^2` |
| `zvar を x で全微分して` | `differentiate(expr=**E1**, var=**'zvar'**)` | **`0`** |
| `zvar を x で積分して` | `integrate(expr=**E1**, var=**'zvar'**)` | `x^2*zvar+C` |

3名前 × 6操作 × 日英の **36セル中30**。微分・積分では捨てられた名前が今度は
**変数の欄**に入り、研究者が書いた `x` のほうが落ちる——研究者自身の記号で
組み立てた、もっともらしい誤答である。

**素の名前は「数えれば済む」ものではない。** `x で微分して` の `x` も未定義の
素の名前だが、そこでは**変数**として立っているので、数えると対象を失う（CLAUDE.md
がスカラー除外の理由として記録している罠の、別の顔）。英語ではさらに操作の語
（`simplify`）が識別子に見える。判別は**位置**で、位置を知っているのは文法である
——だから読み方はこの層が既に持っている3つ（字種の変わり目・操作の別名・
英語の微分の枠）をそのまま使い、guard へ「オペランドの位置で読んだ名前」を渡す。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.i18n import t
from notebook.runtime.operand_boundary import (
    may_fall_back_to_the_recent_expression, name_is_usable_as_an_operand)
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   _operand_name_in_text, run_interpret_pipeline)


def _run(lang, lines, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="unresolved", mode="standard", language=lang,
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
            args = [getattr(s, "operation_args", None) for s in engine.steps.values()
                    if s.kind == "cas_operation" and getattr(s, "operation", None)]
            return out[-1], (args[-1] if args else None)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


#: `y = x^2` が直前の式。すり替わると答えはこれに由来する（`x^2` / `2*x` / `0`）。
_SETUP = ["y = x^2"]

_SPELLING = {
    "ja": {"simplify": "{0} を簡約して", "expand": "{0} を展開して",
           "factor": "{0} を因数分解して", "diff": "{0} を x で全微分して",
           "integrate": "{0} を x で積分して", "norm": "{0} のノルムを求めて"},
    "en": {"simplify": "simplify {0}", "expand": "expand {0}",
           "factor": "factor {0}", "diff": "differentiate {0} with respect to x totally",
           "integrate": "integrate {0} over x", "norm": "the norm of {0}"},
}
#: 名前の長さも綴りも変える——1文字（自由変数に見える）・ギリシャ（⚠ の対象外）・普通。
_NAMES = ["zvar", "xi", "q"]

#: 英語の不定積分は 2026-09-18 に枠が入って閉じた（ENGLISH-INLINE-INTEGRAL）。
#: それまでここは strict xfail で留めてあり、**枠が入った日にスイートを落とした**
#: ——「あとで確かめる」を機構に預けた形が設計どおり働いた。
_CELLS = [(lang, op, name)
          for lang in ("ja", "en") for op in _SPELLING["ja"] for name in _NAMES]


class TestAnUnresolvedNameIsNotReplaced:
    @pytest.mark.parametrize("lang,op,name", _CELLS)
    def test_the_recent_expression_does_not_answer(self, lang, op, name):
        r, args = _run(lang, _SETUP + [_SPELLING[lang][op].format(name)])
        assert (args or {}).get("expr") != "E1", (r, args)


class TestTheVariablePositionStillFallsBack:
    """`x で微分して` の `x` も未定義の素の名前——**そこでは変数**である。"""

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "x で微分して", "2*x"),
        ("ja", "x で全微分して", "2*x"),
        ("ja", "簡約して", "x^2"),
        ("en", "differentiate with respect to x totally", "2*x"),
        ("en", "simplify", "x^2"),
        ("en", "expand", "x^2"),
    ])
    def test_it_still_answers_about_the_recent_expression(self, lang, text, want):
        r, _ = _run(lang, _SETUP + [text])
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == want, r


class TestANameThatResolvesIsUntouched:
    @pytest.mark.parametrize("lang,setup,text,want", [
        ("ja", ["v = [y, -x, 0]"], "v の回転を求めて", "[0,0,-2]"),
        ("ja", ["A をベクトル場とする"], "A のノルムを求めて",
         "sqrt(A_z^2+A_y^2+A_x^2)"),
        ("en", ["define A as a vector field"], "the norm of A",
         "sqrt(A_z^2+A_y^2+A_x^2)"),
        ("ja", ["y = x^2"], "E1 を簡約して", "x^2"),
        ("en", ["y = x^2"], "simplify E1", "x^2"),
    ])
    def test_h1_declared_and_expression_ids_all_keep_working(self, lang, setup, text, want):
        r, _ = _run(lang, setup + [text])
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == want, r


class TestThePositionIsReadByTheGrammarThatKnowsIt:
    @pytest.mark.parametrize("text,want", [
        # 日本語: 字種の変わり目、続きが `を` / `の` のときだけ
        ("zvar を簡約して", "zvar"),
        ("zvar の回転を求めて", "zvar"),
        ("zvar を x で全微分して", "zvar"),
        ("x で微分して", None),                     # `で` は**変数**の位置
        ("簡約して", None),
        # 英語: 操作の別名が境界
        ("simplify zvar", "zvar"),
        ("expand zvar", "zvar"),
        ("the norm of zvar", "zvar"),
        ("differentiate zvar with respect to x totally", "zvar"),
        ("differentiate with respect to x totally", None),
        # 式番号と数学はここへ来ない（手前の gate が式として登録する）
        ("E1 を簡約して", None),
        ("simplify E1", None),
        ("2 v を簡約して", None),
        ("x^2 + y^2 を簡約して", None),
        # **素の既知関数名は名前ではない**——ここを数えると、断り文が
        # 「`sin` に値を与えてください」と誘うことになる。素の関数名には
        # それ自身の断りが在る（CLAUDE.md「A WRITTEN FUNCTION IS APPLIED」）。
        ("sin を簡約して", None),
        ("simplify sin", None),
        ("log を簡約して", None),
    ])
    def test_operand_name_in_text(self, text, want):
        assert _operand_name_in_text(text) == want, text


class TestTheGuardAsksWhetherTheNameResolves:
    def _ctx(self):
        ctx = SymbolContext()
        ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
        ctx.register("A", Quantity(name="A", kind="vector_field",
                                   components=["A_x", "A_y", "A_z"]))
        return ctx

    def test_an_unresolved_operand_name_stops_the_fallback(self):
        assert not may_fall_back_to_the_recent_expression(
            self._ctx(), "zvar を簡約して", operand_names=("zvar",))

    def test_a_name_that_resolves_does_not(self):
        """H-1 の名前は 2a-sym が、宣言済みの対象は①が面倒を見る。"""
        ctx = self._ctx()
        assert may_fall_back_to_the_recent_expression(ctx, "v を簡約して",
                                                      operand_names=("v",))

    def test_a_name_the_grammar_already_consumed_does_not_count(self):
        """`consumed` のちょうど鏡——別の役で読んだ名前は対象ではない。"""
        assert may_fall_back_to_the_recent_expression(
            self._ctx(), "zvar で微分して", consumed=("zvar",), operand_names=("zvar",))

    def test_with_no_operand_name_the_old_answer_stands(self):
        assert may_fall_back_to_the_recent_expression(self._ctx(), "簡約して")
        assert not may_fall_back_to_the_recent_expression(self._ctx(), "A のノルムを求めて")


#: 宣言済みの**スカラー**は「解決する」が、オペランドとしては渡せない——欄は
#: スカラーを受けず（`x` は変数として文中に立つ）、2a-sym も書き換え先を持たない。
_SCALAR_DECL = {"ja": "g をスカラーとする", "en": "let g be a scalar"}
_SCALAR_SPELL = {
    "ja": ["g を簡約して", "g を展開して", "g を因数分解して", "g を x で全微分して"],
    "en": ["simplify g", "expand g", "factor g",
           "differentiate g with respect to x totally"],
}


class TestADeclaredScalarIsNotUsableAsAnOperandEither:
    """`g をスカラーとする` のあと `g を簡約して` は直前の式の簡約を返していた（8/8）。"""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("i", range(4))
    def test_the_recent_expression_does_not_answer(self, lang, i):
        r, args = _run(lang, _SETUP + [_SCALAR_DECL[lang], _SCALAR_SPELL[lang][i]])
        assert (args or {}).get("expr") != "E1", (r, args)


class TestTheRefusalNamesWhatTheResearcherWrote:
    r"""断り文は `expr` ではなく、研究者が書いた名前について言う。"""

    @pytest.mark.parametrize("lang,setup,text,name", [
        ("ja", [], "zvar を簡約して", "zvar"),
        ("en", [], "simplify zvar", "zvar"),
        ("ja", ["g をスカラーとする"], "g を簡約して", "g"),
        ("en", ["let g be a scalar"], "simplify g", "g"),
    ])
    def test_the_name_and_the_banner_are_both_there(self, lang, setup, text, name):
        r, _ = _run(lang, _SETUP + setup + [text])
        assert r.get("status") == "failed", r
        assert name in (r.get("error") or ""), r
        assert "expr" not in (r.get("error") or ""), r
        # ⚠ の未定義記号バナーは、描画側がこの欄を読む（`notebook_main.js` のエラー枝）
        assert r.get("undefined_symbols") == [name], r

    def test_the_road_it_names_is_measured_to_work(self):
        """案内した回避策が効くかは、断りの一部である。"""
        r, _ = _run("ja", _SETUP + ["zvar = a x + b", "zvar を簡約して"])
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == "a*x+b", r

    def test_a_scalar_declaration_is_deliberately_not_named_as_a_road(self):
        """実測で通らないので案内しない——宣言しても欄はスカラーを受けない。"""
        msg = t("operand.name_has_no_value", "ja", name="zvar")
        assert "スカラー" not in msg, msg


class TestTheBannerRidesOnASuccessToo:
    r"""`E_{total} を簡約して` は `E(total)` を返す（自由変数として正しい）が、
    研究者にはそれが未定義だと伝わっていなかった——応答にキーごと不在だった。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"E_{total} を簡約して"), ("en", r"simplify E_{total}"),
    ])
    def test_the_value_is_computed_and_the_name_is_reported(self, lang, text):
        r, _ = _run(lang, _SETUP + [text])
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == "E(total)", r
        assert r.get("undefined_symbols") == ["E_total"], r

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を簡約して"), ("en", "simplify E1"),
    ])
    def test_a_resolvable_operand_carries_no_banner(self, lang, text):
        r, _ = _run(lang, _SETUP + [text])
        assert r.get("status") == "completed", r
        assert not r.get("undefined_symbols"), r


class TestTheLookupUsesTheCanonicalSpelling:
    def test_a_subscripted_name_is_looked_up_as_it_is_registered(self):
        r"""研究者は `E_{total}` と書き、量は `E_total` で登録される。"""
        from ui.interpret_pipeline import _operand_key
        import ui.notebook_bridge as _nb

        async def go():
            _nb.start_engine(title="key", mode="standard", language="ja",
                             domains=["electromagnetism"])
            try:
                eng = _nb.get_engine()
                return (_operand_key(eng, r"E_{total}"), _operand_key(eng, "zvar"),
                        _operand_key(eng, "one"))
            finally:
                _nb.stop_engine()
        sub, bare, word = asyncio.run(go())
        assert sub == "E_total"
        assert bare == "zvar"
        # 名前の扉は名前を返す——数値語の表は `one` を `1` にするが、鍵は名前である
        assert word == "one"


class TestUsableAsAnOperand:
    def test_the_three_answers(self):
        ctx = SymbolContext()
        ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
        ctx.register("A", Quantity(name="A", kind="vector_field",
                                   components=["A_x", "A_y", "A_z"]))
        ctx.register("g", Quantity(name="g", kind="scalar"))
        assert name_is_usable_as_an_operand(ctx, "v")        # 値がある
        assert name_is_usable_as_an_operand(ctx, "A")        # 宣言された対象
        assert not name_is_usable_as_an_operand(ctx, "g")    # 宣言済みスカラー
        assert not name_is_usable_as_an_operand(ctx, "zvar")  # 何も無い
