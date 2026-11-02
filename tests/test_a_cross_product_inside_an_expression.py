r"""CROSS-PRODUCT-INSIDE-OPERAND (2026-09-18) — 掛かり方は文字列では決まらない。

外積だけのオペランド（`v ~ w の回転を求めて`）は通っていた。**式の中に埋まった**
外積は、`nb_cross` がどこを置き換えるかを「いちばん外側の `~` で2つに割る」という
文字列の近似で決めていたので、どれも読めなかった。回転・発散・ノルムの3操作 ×
10綴り × 日英で測ると、HEAD で通ったのは **60 セル中 24**:

| 綴り | HEAD |
|---|---|
| `v ~ w` ・ `(v ~ w)` ・ `\vec{v} \times \vec{w}` ・ `2 v ~ w` | 24 / 24 |
| `2 (v ~ w)` ・ `x (v ~ w)` ・ `(v ~ w) + r` ・ `v ~ w + r` ・ `(v ~ w) ~ r` ・ `2 (\vec v \times \vec w)` | **0 / 36** |

回転と発散は `vector.cross_product_inside` で断り、**ノルムだけは別の理由**
（「丸括弧の組はベクトルとして読みません」）で断っていた——同じ問いに2つの答え。

掛かり方は構文の問題なので、答えを持っているのはパーサである。書き換えを
**構文木の上で**やる（`vector_operand.cross_products_as_calls`）と 60/60 になり、
`v ~ w + r` が `nb_cross(v, w) + r` であって `nb_cross(v, w + r)` ではないことも
そこから出る（文字列で割ると後者になる）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models import ir
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.vector_operand import cross_products_as_calls
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _run(lang, lines, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="cross", mode="standard", language=lang, domains=[domain])
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
            return out, [getattr(s, "rendered_command", None) for s in engine.steps.values()]
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _ask(lang, setup, text, domain="electromagnetism"):
    return _run(lang, list(setup) + [text], domain)[0][-1]


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


#: `v ~ w = [-x^2, -x y, 0]` —— 以下の期待値はすべてこれを手で微分したもの。
_SETUP = ["v = [y, -x, 0]", "w = [0, 0, x]", "r = [x, y, z]"]

_PHRASE = {("ja", "curl"): "{0} の回転を求めて",
           ("ja", "div"): "{0} の発散を求めて",
           ("ja", "norm"): "{0} のノルムを求めて",
           ("en", "curl"): "the curl of {0}",
           ("en", "div"): "the divergence of {0}",
           ("en", "norm"): "the norm of {0}"}

#: 綴り → (回転, 発散, ノルム)。**綴りが違っても同じ量**である組は同じ値が入る。
_SPELLINGS = {
    "v ~ w":                          ("[0,0,-y]", "-(3*x)", "sqrt(x^2*y^2+x^4)"),
    "(v ~ w)":                        ("[0,0,-y]", "-(3*x)", "sqrt(x^2*y^2+x^4)"),
    r"\vec{v} \times \vec{w}":        ("[0,0,-y]", "-(3*x)", "sqrt(x^2*y^2+x^4)"),
    "2 (v ~ w)":                      ("[0,0,-(2*y)]", "-(6*x)", "sqrt(4*x^2*y^2+4*x^4)"),
    "2 v ~ w":                        ("[0,0,-(2*y)]", "-(6*x)", "sqrt(4*x^2*y^2+4*x^4)"),
    r"2 (\vec{v} \times \vec{w})":    ("[0,0,-(2*y)]", "-(6*x)", "sqrt(4*x^2*y^2+4*x^4)"),
    "x (v ~ w)":                      ("[0,0,-(2*x*y)]", "-(4*x^2)", "sqrt(x^4*y^2+x^6)"),
    "(v ~ w) + r":                    ("[0,0,-y]", "3-3*x", "sqrt(z^2+(y-x*y)^2+(x-x^2)^2)"),
    "v ~ w + r":                      ("[0,0,-y]", "3-3*x", "sqrt(z^2+(y-x*y)^2+(x-x^2)^2)"),
    "(v ~ w) ~ r":                    ("[-x^2,-(x*y),3*x*z]", "-(y*z)",
                                       "sqrt(x^2*y^2*z^2+x^4*z^2)"),
}
_CASES = [(lang, op, spelling, want)
          for lang in ("ja", "en")
          for spelling, wants in _SPELLINGS.items()
          for op, want in zip(("curl", "div", "norm"), wants)]


class TestEverySpellingComputes:
    @pytest.mark.parametrize("lang,op,spelling,want", _CASES)
    def test_the_family(self, lang, op, spelling, want):
        r = _ask(lang, _SETUP, _PHRASE[(lang, op)].format(spelling))
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheParserDecidesWhatTheCrossProductBinds:
    """`v ~ w + r` は `(v ~ w) + r`。**文字列で割ると `v ~ (w + r)` になる**。"""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_two_groupings_are_different_and_the_written_one_wins(self, lang):
        loose = _ask(lang, _SETUP, _PHRASE[(lang, "div")].format("v ~ w + r"))
        left = _ask(lang, _SETUP, _PHRASE[(lang, "div")].format("(v ~ w) + r"))
        right = _ask(lang, _SETUP, _PHRASE[(lang, "div")].format("v ~ (w + r)"))
        assert _value(left) == _value(loose) == "3-3*x", (left, loose)
        assert _value(right) == "-(2*z)-3*x", right     # 手計算: v ~ [x, y, x+z]


class TestItAgreesWithTheTwoStepRoad:
    """起票が回避策として名指していた道。**同じ量なら同じ答え**でなければならない。"""

    @pytest.mark.parametrize("lang,first", [
        ("ja", "v と w の外積を求めて"), ("en", "the cross product of v and w"),
    ])
    def test_one_step_equals_two(self, lang, first):
        # **式番号は数えず、外積が返したものを使う。** 数えると設定の行を1本足した
        # だけで別の式を指し、ここは `2 r` の回転 `[0,0,0]` を「一致」と読みかけた。
        async def go():
            nb.start_engine(title="cross", mode="standard", language=lang,
                            domains=["electromagnetism"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["electromagnetism"],
                                    primary_backend=engine.session.primary_backend)
                r = None
                for line in _SETUP + [first]:
                    r = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                eid = r.get("expr_id")
                return eid, await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(
                        type="text", value=_PHRASE[(lang, "curl")].format(f"2 {eid}"))]),
                    engine, sup)
            finally:
                nb.stop_engine()
        eid, two = asyncio.run(go())
        one = _ask(lang, _SETUP, _PHRASE[(lang, "curl")].format("2 (v ~ w)"))
        assert _value(two) == _value(one) == "[0,0,-(2*y)]", (eid, two, one)


class TestADeclaredFieldKeepsItsOwnComponentNames:
    r"""`nb_cross(A, B)` は A が**原子**だと `A[2]*B[3]-…` を返す（実測）。

    研究者が書いていない名前なので、成分のリストに開いてから渡す（`_listify`）。
    生成器そのものが `~` を `nb_cross` に書き換えないのはこれが理由である。
    """

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", ["A をベクトル場とする", "B をベクトル場とする"], "2 (A ~ B) の発散を求めて"),
        ("en", ["define A as a vector field", "define B as a vector field"],
         "the divergence of 2 (A ~ B)"),
    ])
    def test_the_command_names_the_declared_components(self, lang, setup, text):
        rs, cmds = _run(lang, setup + [text])
        assert rs[-1].get("status") == "completed", rs[-1]
        cmd = [c for c in cmds if c and "nb_div" in c][-1]
        assert "[A_x,A_y,A_z]" in cmd and "[B_x,B_y,B_z]" in cmd, cmd
        assert "A[2]" not in cmd and "B[3]" not in cmd, cmd


class TestTheCasSpellingIsNeverShown:
    """Core Principle #1 —— `nb_cross` は系の内部の名前である。"""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_no_nb_cross_on_screen(self, lang):
        r = _ask(lang, _SETUP, _PHRASE[(lang, "curl")].format("2 (v ~ w)"))
        for field in ("latex", "canonical", "message"):
            assert "nb_cross" not in str(r.get(field) or ""), (field, r)


class TestWhatStillRefuses:
    @pytest.mark.parametrize("lang,text", [("ja", "v w の回転を求めて"),
                                           ("en", "the curl of v w")])
    def test_a_product_of_two_vectors_still_names_the_missing_word(self, lang, text):
        """内積か外積かは研究者が決める——読めるようになったのは**書かれた** `~` だけ。"""
        r = _ask(lang, _SETUP, text)
        assert r.get("status") == "failed", r
        assert "Maxima" not in (r.get("error") or ""), r

    @pytest.mark.parametrize("lang,text", [("ja", "2 (u ~ u) の回転を求めて"),
                                           ("en", "the curl of 2 (u ~ u)")])
    def test_a_parenthesised_tuple_is_still_not_a_vector(self, lang, text):
        r = _ask(lang, ["u = (1, 2, 3)"], text)
        assert r.get("status") == "failed", r
        assert "Maxima" not in (r.get("error") or ""), r


class TestTheRewriteWalksTheTree:
    """節点の種類を数え上げず、欄をたどる——外積を包める節点が増えても直し忘れない。"""

    @pytest.mark.parametrize("text,want", [
        ("v ~ w", "nb_cross(v, w)"),
        ("2*(v ~ w)", "2 * nb_cross(v, w)"),
        ("x*(v ~ w)", "x * nb_cross(v, w)"),
        ("(v ~ w) + r", "nb_cross(v, w) + r"),
        ("v ~ w + r", "nb_cross(v, w) + r"),          # 文字列で割ると `nb_cross(v, w + r)`
        ("(v ~ w) ~ r", "nb_cross(nb_cross(v, w), r)"),
        ("2*([y,-x,0] ~ [0,0,x])", "2 * nb_cross([y, -x, 0], [0, 0, x])"),
        ("2*v", "2 * v"),                              # 外積が無ければ何も変わらない
    ])
    def test_the_maxima_spelling(self, text, want):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        assert MaximaBackendHandler(None).vector_expression_code(text) == want

    def test_a_cross_product_inside_a_power_is_reached(self):
        """`Pow` は左右の欄を持たない節点——数え上げ方式ならここで落ちる。"""
        node = ir.Pow(base=ir.CrossProduct(left=ir.Symbol(name="v"),
                                           right=ir.Symbol(name="w")),
                      exponent=ir.Number(value=2))
        out = cross_products_as_calls(node, "nb_cross")
        assert isinstance(out.base, ir.FunctionCall)
        assert out.base.func.name == "nb_cross"
        assert out.exponent.value == 2

    def test_a_tree_without_a_cross_product_is_returned_as_is(self):
        node = ir.Add(args=[ir.Symbol(name="a"), ir.Symbol(name="b")])
        assert cross_products_as_calls(node, "nb_cross") == node

    def test_a_cross_product_the_backend_cannot_spell_is_not_read(self):
        r"""綴りへ落とせなかったら読まない——**残った `~` は生の CAS エラーになる**。

        到達するのは、型は読めるのに**解決が断る**綴り。`\sin^2 x` は
        「書かれた関数は適用されるか、量ではない」で断られるので `parse_ir` が
        通らず、`infer_algebra` は（解決しない表層 AST を読むので）`vector` と
        答える。この guard が無いと、生の LaTeX と評価できない `~` を含んだ
        `(\sin^2 x * ((([y, -x, 0])) ~ (([0, 0, x]))))[1]` が CAS へ渡る（実測）。
        """
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.vector_op_types import infer_algebra, value_lookup
        from notebook.runtime.vector_operand import expression_components

        values = {"E1": "[y, -x, 0]", "E2": "[0, 0, x]"}
        ctx = SymbolContext()
        ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
        ctx.register("w", Quantity(name="w", kind="anonymous", defining_expr_id="E2"))
        vo = value_lookup(ctx, values.get)
        handler = MaximaBackendHandler(None)
        text = r"\sin^2 x * ((v) ~ (w))"
        assert infer_algebra(text, ctx, vo) == "vector"          # 型は読める
        assert handler.vector_expression_code(text) is None      # 綴りは落とせない
        assert expression_components(text, ctx, vo,
                                     to_cas=handler.vector_expression_code) is None
