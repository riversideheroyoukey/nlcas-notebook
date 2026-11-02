r"""CROSS-PRODUCT-IN-GENERIC-OP (2026-09-19) — `~` のままでは CAS が評価できない。

生成器は `CrossProduct` を `((左) ~ (右))` と書き、**Maxima はそれを評価できない**
（実測: リストでも原子でも生の `Maxima evaluation error`）。だから外積を含む式に
一般の操作をかけると、研究者には CAS の言葉がそのまま返っていた（Core Principle #1）
——実測 2026-09-18、日英とも、**H-1 で値を与えた名前でも宣言された場でも**:

| 入力 | 返っていたもの |
|---|---|
| `F = 2 (v ~ w)` のあと `F を簡約して` / `展開` / `因数分解` | 生の `Maxima evaluation error` |
| `F を x で全微分して` / `x で積分して` / `x に 1 を代入して` | 同上 |
| `F のノルム` / `発散` / `回転` | **通っていた**——成分の読みを経由するから |

**2つそろって初めて正しい**: 構文木から `nb_cross` へ書き換え、そして**ベクトルを
成分のリストへ開く**。開かずに `nb_cross(A, B)` を宣言された場に当てると
`A[2]*B[3]-…`——研究者が書いていない名前——が返り、生のエラーを silent-wrong と
取り替えることになる（実測）。

**型は利用者書式から読む。** 差し替えた側は研究者の綴りを `__user__<eid>` に残し、
成分の読みとノルムはそちらを見る（`user_form`、`lower_expr_lookup` と同じ約束）。
1箇所に置いていなかったあいだ、**ノルムだけが**「ベクトルにも行列にも見えません」で
断った——実装中に実際に踏んだ。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _run(lang, lines, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="xgen", mode="standard", language=lang, domains=[domain])
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
            cmds = [getattr(s, "rendered_command", None) for s in engine.steps.values()]
            return out[-1], [c for c in cmds if c]
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


#: `v ~ w = [-x^2, -x y, 0]`、`F = 2 (v ~ w) = [-2x^2, -2x y, 0]`（手計算）。
_H1 = {"ja": ["v = [y, -x, 0]", "w = [0, 0, x]", "F = 2 (v ~ w)"],
       "en": ["v = [y, -x, 0]", "w = [0, 0, x]", "F = 2 (v ~ w)"]}
_DECL = {"ja": ["A をベクトル場とする", "B をベクトル場とする", "G = 2 (A ~ B)"],
         "en": ["define A as a vector field", "define B as a vector field",
                "G = 2 (A ~ B)"]}

_GENERIC = [
    ("ja", "F を簡約して", "[-(2*x^2),-(2*x*y),0]"),
    ("ja", "F を展開して", "[-(2*x^2),-(2*x*y),0]"),
    ("ja", "F を x で全微分して", "[-(4*x),-(2*y),0]"),
    ("ja", "F の x に 1 を代入して", "[-2,-(2*y),0]"),
    ("en", "simplify F", "[-(2*x^2),-(2*x*y),0]"),
    ("en", "expand F", "[-(2*x^2),-(2*x*y),0]"),
    ("en", "differentiate F with respect to x totally", "[-(4*x),-(2*y),0]"),
]


class TestTheGenericOperationsCompute:
    @pytest.mark.parametrize("lang,text,want", _GENERIC)
    def test_no_raw_cas_text_reaches_the_researcher(self, lang, text, want):
        r, _ = _run(lang, _H1[lang] + [text])
        assert r.get("status") == "completed", r
        assert "Maxima" not in (r.get("error") or ""), r
        assert _value(r) == want, r


class TestADeclaredFieldKeepsItsOwnComponentNames:
    r"""`nb_cross(A, B)` を**原子**に当てると `A[2]*B[3]` を返す（実測）。
    リストに開いてから渡すので、答えは研究者が宣言した成分名で書かれている。"""

    @pytest.mark.parametrize("lang,text", [("ja", "G を簡約して"), ("en", "simplify G")])
    def test_the_answer_names_the_declared_components(self, lang, text):
        r, cmds = _run(lang, _DECL[lang] + [text])
        assert r.get("status") == "completed", r
        val = _value(r)
        assert "A_y*B_z" in val and "A_z*B_y" in val, r
        assert "A[2]" not in val and "B[3]" not in val, r
        flat = ["".join(c.split()) for c in cmds]
        assert any("nb_cross([A_x,A_y,A_z],[B_x,B_y,B_z])" in c for c in flat), cmds[-1]


class TestTheComponentReadingStillWorks:
    """∇ の3操作は元から通っていた——差し替えで壊してはならない。"""

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "F の発散を求めて", "-(6*x)"),
        ("ja", "F の回転を求めて", "[0,0,-(2*y)]"),
        ("ja", "F のノルムを求めて", "sqrt(4*x^2*y^2+4*x^4)"),
        ("en", "the divergence of F", "-(6*x)"),
        ("en", "the norm of F", "sqrt(4*x^2*y^2+4*x^4)"),
    ])
    def test_it_is_unchanged(self, lang, text, want):
        r, _ = _run(lang, _H1[lang] + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheResearchersSpellingIsKept:
    r"""差し替えるのは CAS へ渡す綴りだけ——**canonical は研究者の書いたもの**。"""

    def test_the_registered_expression_still_carries_the_cross(self):
        async def go():
            nb.start_engine(title="keep", mode="standard", language="ja",
                            domains=["electromagnetism"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["electromagnetism"],
                                    primary_backend=engine.session.primary_backend)
                for line in _H1["ja"] + ["F を簡約して"]:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return {k: v.canonical for k, v in engine.expressions.items()}
            finally:
                nb.stop_engine()
        exprs = asyncio.run(go())
        assert any("~" in c for c in exprs.values()), exprs
        assert not any("nb_cross" in c for c in exprs.values()), exprs


class TestTheViewIsTheBackendsAnswer:
    def test_maxima_opens_the_vectors_and_calls_nb_cross(self):
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.vector_op_types import value_lookup

        values = {"E1": "[y, -x, 0]", "E2": "[0, 0, x]"}
        ctx = SymbolContext()
        ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
        ctx.register("w", Quantity(name="w", kind="anonymous", defining_expr_id="E2"))
        h = MaximaBackendHandler(None)
        h._context = ctx
        vo = value_lookup(ctx, values.get)
        got = h.vector_cas_view("2*((v) ~ (w))", vo)
        assert got is not None and "nb_cross" in got and "~" not in got, got
        assert "[y, -x, 0]" in got and "[0, 0, x]" in got, got

    def test_a_cross_product_the_backend_cannot_spell_gets_no_view(self):
        r"""綴りへ落とせなければ研究者の書いたままにする——`\sin^2 x` は「書かれた関数は
        適用されるか、量ではない」で解決が断るので `parse_ir` が通らない。落とせなかった
        ものを渡すと、`~` を含んだ綴りがそのまま CAS へ行く（そして開く側が None で落ちる）。
        """
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.vector_op_types import value_lookup

        values = {"E1": "[y, -x, 0]", "E2": "[0, 0, x]"}
        ctx = SymbolContext()
        ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
        ctx.register("w", Quantity(name="w", kind="anonymous", defining_expr_id="E2"))
        h = MaximaBackendHandler(None)
        h._context = ctx
        assert h.vector_cas_view(r"\sin^2 x * ((v) ~ (w))",
                                 value_lookup(ctx, values.get)) is None

    def test_a_text_without_a_cross_product_gets_no_view(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        assert MaximaBackendHandler(None).vector_cas_view("2*v", lambda _n: None) is None

    def test_a_backend_with_no_answer_says_so(self):
        """共有層は訊くだけ——持たないバックエンドは None を返す（既定）。"""
        from notebook.runtime.backend_handler import BackendHandler
        assert BackendHandler.vector_cas_view(object(), "2*((v) ~ (w))", lambda _n: None) is None
