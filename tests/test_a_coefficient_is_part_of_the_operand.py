r"""OPERAND-COEFFICIENT-DROPPED / H1-VECTOR-ALGEBRA-DEFAULT (2026-09-17).

`2 v の回転を求めて` は `curl(v)` を計算し、係数 2 を黙って捨てて `completed` を返していた。
測ると**回転・発散・勾配・ラプラシアン・ノルム・行列式の6操作 × 日英3綴りのすべて**で同じ
（18 セル、HEAD）。

**根は「このオペランドは名前1つか、それ以上の数学か」の判定**（`_operand_is_mathematics`）。
演算子の記号・関数名・LaTeX コマンドだけを数学と数えていたので、**空白の並置——この系の
乗算——が名前1つとして扱われ**、式として登録されずに 2a-sym が `v` だけを式番号へ書き換えた。

並置を数学に数えると、スカラーの係数（勾配・ラプラシアン・行列式）はそのまま正しく計算され、
ベクトルの側は「E6 はスカラーです」という**誤った理由の断り**に変わった。そこから先が
H1-VECTOR-ALGEBRA-DEFAULT で、2つは切り離せなかった:

* **型**: H-1 で値を与えた名前は `algebra='scalar'` と記録され、型の推論がそれを宣言として
  読んでいた。しかも名前を**数える**方式では `v + w`（和＝ベクトル）と `v . w`（内積＝
  スカラー）を区別できない。値から読み、式の**構造**で読む形にした。
* **成分**: CAS はリストの演算を最初から持っていた——`(2*v)[1] = 2*y`、
  `curl(x v) = [0,0,-3x]`（積の微分まで込み、実測）。宣言された場は原子なので成分の
  リストへ、H-1 のベクトル名はその値へ置き換えてから `(式)[i]` を成分にする。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import MathType, Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ask(lang, setup, text):
    async def go():
        nb.start_engine(title="coefficient", mode="standard", language=lang,
                        domains=["electromagnetism"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["electromagnetism"],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]), engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]), engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


_SETUP = {"ja": ["A をベクトル場とする", "v = [y, -x, 0]", "w = [0, 0, x]", "r = [x, y, z]",
                 "u = x^2 + y^2", "M = [1, 2; 3, 4]"],
          "en": ["define A as a vector field", "v = [y, -x, 0]", "w = [0, 0, x]",
                 "r = [x, y, z]", "u = x^2 + y^2", "M = [1, 2; 3, 4]"]}


def _spellings(ja_noun, en_noun, operand):
    return [("ja", f"{operand} の{ja_noun}を求めて"),
            ("en", f"the {en_noun} of {operand}"),
            ("en", f"compute the {en_noun} of {operand}")]


_COEFFICIENT = [
    ("回転", "curl", "2 v", "[0,0,-4]"),
    ("発散", "divergence", "2 r", "6"),
    ("勾配", "gradient", "2 u", "[4*x,4*y,0]"),
    ("ラプラシアン", "Laplacian", "2 u", "8"),
    ("ノルム", "norm", "2 r", "sqrt(4*z^2+4*y^2+4*x^2)"),
    ("行列式", "determinant", "2 M", "-8"),
]


class TestTheCoefficientIsKept:
    @pytest.mark.parametrize("ja_noun,en_noun,operand,want", _COEFFICIENT)
    def test_all_three_spellings(self, ja_noun, en_noun, operand, want):
        for lang, text in _spellings(ja_noun, en_noun, operand):
            r = _ask(lang, _SETUP[lang], text)
            assert r.get("status") == "completed", (text, r)
            assert _value(r) == want, (text, r)


class TestAVectorExpressionIsAVector:
    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "v + w の回転を求めて", "[0,-1,-2]"),
        ("en", "the curl of v + w", "[0,-1,-2]"),
        ("ja", "(2 v) の回転を求めて", "[0,0,-4]"),
        ("ja", "v - 2 w の回転を求めて", "[0,2,-2]"),
        ("en", "the curl of v - 2 w", "[0,2,-2]"),
        ("ja", "-v の回転を求めて", "[0,0,2]"),
        ("ja", r"\frac{1}{2} v の回転を求めて", "[0,0,-1]"),
        ("en", "the curl of a v", "[0,0,-(2*a)]"),
        # the product rule comes out of the CAS: curl(x v) = x curl v + ∇x × v
        ("ja", "x v の回転を求めて", "[0,0,-(3*x)]"),
        ("en", "the curl of x v", "[0,0,-(3*x)]"),
        # a DECLARED field is an atom: it reaches the CAS as its component list
        ("ja", "2 A の発散を求めて", "2*'diff(A_z,z,1)+2*'diff(A_y,y,1)+2*'diff(A_x,x,1)"),
        ("en", "the divergence of 2 A", "2*'diff(A_z,z,1)+2*'diff(A_y,y,1)+2*'diff(A_x,x,1)"),
        ("ja", r"\vec{A} + v の発散を求めて", "'diff(A_z,z,1)+'diff(A_y,y,1)+'diff(A_x,x,1)"),
    ])
    def test_it_computes(self, lang, text, want):
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) == want, r

    @pytest.mark.parametrize("lang,text", [("ja", "2 B2 の発散を求めて"),
                                           ("en", "the divergence of 2 B2")])
    def test_a_named_vector_whose_value_mentions_a_declared_field(self, lang, text):
        """`B2 = 2 A` は CAS では原子の積に束縛される——値を成分のリストへ開いてから渡す。"""
        r = _ask(lang, _SETUP[lang] + ["B2 = 2 A"], text)
        assert r.get("status") == "completed", r
        assert _value(r) == "4*'diff(A_z,z,1)+4*'diff(A_y,y,1)+4*'diff(A_x,x,1)", r


class TestTheValueIsSplicedAsStored:
    @pytest.mark.parametrize("lang,text,want", [("ja", "2 v の回転を求めて", "[0,0,4*x]"),
                                                ("en", "the curl of 2 v", "[0,0,4*x]")])
    def test_an_indexed_name_in_the_value_is_not_lowered(self, lang, text, want):
        """降ろすと `y_1` が `y(1)`（`y` の関数）になり、書いていない微分項が混ざった。"""
        r = _ask(lang, ["v = [x y_1, x^2, 0]"], text)
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestACoefficientOnAnExpressionId:
    # E1 is `v` (the field declaration registers no expression)
    @pytest.mark.parametrize("lang,text", [("ja", "2 E1 の回転を求めて"), ("en", "the curl of 2 E1"),
                                           ("en", "compute the curl of 2 E1")])
    def test_both_languages_keep_it(self, lang, text):
        """英語は `_formula_runs_to_the_end` が式番号を含む式を「参照の並び」として断り、
        係数が落ちていた——日本語は同じ直しで正しかった（日英の割れ）。"""
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) == "[0,0,-4]", r


class TestDotAndCrossInsideAnOperand:
    @pytest.mark.parametrize("lang,text,want", [
        # HEAD: all four dropped everything after the first name and answered `completed`
        ("ja", "v ~ w の回転を求めて", "[0,0,-y]"),
        ("ja", r"\vec{v} \times \vec{w} の回転を求めて", "[0,0,-y]"),
        ("en", r"the curl of \vec{v} \times \vec{w}", "[0,0,-y]"),
        ("ja", "v ~ w の発散を求めて", "-(3*x)"),
    ])
    def test_a_cross_product_operand_computes(self, lang, text, want):
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) == want, r

    def test_it_agrees_with_computing_the_cross_product_first(self):
        async def go():
            nb.start_engine(title="coefficient", mode="standard", language="ja",
                            domains=["electromagnetism"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["electromagnetism"],
                                    primary_backend=engine.session.primary_backend)
                r = None
                for line in ["v = [y, -x, 0]", "w = [0, 0, x]", "v と w の外積を求めて",
                             "E3 の回転を求めて"]:
                    r = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]), engine, sup)
                return r
            finally:
                nb.stop_engine()
        assert _value(asyncio.run(go())) == "[0,0,-y]"

    @pytest.mark.parametrize("lang,text", [("ja", "u.u の勾配を求めて"), ("ja", "u . u の勾配を求めて")])
    def test_a_dot_is_not_dropped(self, lang, text):
        """HEAD は `u.u` を1つの未宣言の名前と読み、勾配 `[0,0,0]` を返していた。"""
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) != "[0,0,0]", r

    @pytest.mark.parametrize("lang,text", [("ja", "2 (v ~ w) の回転を求めて"),
                                           ("en", "the curl of 2 (v ~ w)")])
    def test_a_buried_cross_product_computes(self, lang, text):
        """この断りは**塞がり**で、素のアサーションで固定してしまっていた。

        CLAUDE.md の規則は「未解決の問題に合わせた検査は strict xfail」——素で
        書くと塞がりが取れたことを誰も確かめない。ここは起票（CROSS-PRODUCT-
        INSIDE-OPERAND）の翌日に解けたので気づいたが、気づいたのは**次の問いを
        受け取ったから**であって、検査が言ったからではない。
        """
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) == "[0,0,-(2*y)]", r


class TestAScalarOperandIsUnchanged:
    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "u の勾配を求めて", "[2*x,2*y,0]"),
        ("ja", "u + x の勾配を求めて", "[2*x+1,2*y,0]"),
        ("en", "the gradient of u + x", "[2*x+1,2*y,0]"),
        ("ja", "x^2 y の勾配を求めて", "[2*x*y,x^2,0]"),
        ("ja", "2 u を x で偏微分して", "4*x"),
        ("en", "the partial derivative of 2 u with respect to x", "4*x"),
    ])
    def test_it_computes(self, lang, text, want):
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheRefusalFitsTheOperand:
    @pytest.mark.parametrize("lang,text", [("ja", "v w の回転を求めて"), ("en", "the curl of v w")])
    def test_a_product_of_two_vectors_names_the_missing_kind(self, lang, text):
        """以前は「丸括弧の組はベクトルとして読みません」——別の問いへの答えだった。"""
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "failed", r
        assert r.get("error") == t("vector.product_of_vectors", lang, operand="v * w"), r

    @pytest.mark.parametrize("lang,text", [("ja", "v . w の回転を求めて"), ("en", "the curl of v . w")])
    def test_a_dot_product_is_a_scalar(self, lang, text):
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "failed", r
        assert t("declare.algebra_scalar", lang) in (r.get("error") or ""), r

    @pytest.mark.parametrize("lang,text", [("ja", "u の回転を求めて"), ("en", "the curl of u")])
    def test_a_defined_scalar_is_refused_as_a_scalar(self, lang, text):
        """型の検査が値を引けて初めて「スカラーです」と言える（引けなければ判定しない）。"""
        r = _ask(lang, _SETUP[lang], text)
        assert r.get("status") == "failed", r
        assert t("declare.algebra_scalar", lang) in (r.get("error") or ""), r

    def test_the_road_the_product_refusal_names_works(self):
        r = _ask("ja", _SETUP["ja"], "v と w の外積を求めて")
        assert r.get("status") == "completed", r


class TestTheOperandGate:
    @pytest.mark.parametrize("text,want", [
        ("2 v", True), ("x v", True), ("a b", True), ("v + w", True), ("x^2", True),
        ("v . w", True), ("v.w", True), ("v ~ w", True), ("2 E1", True),
        ("v", False), ("A", False),
        # measured by the corpus differential: the first, wider version stole these two
        ("E1 twice", False), ("both sides of E30", False),
        ("E1 in x", False),               # a two-letter function word is not an atom
    ])
    def test_a_juxtaposition_is_mathematics(self, text, want):
        from ui.interpret_pipeline import _operand_is_mathematics
        assert _operand_is_mathematics(text) is want

    @pytest.mark.parametrize("following,want", [("の回転を求めて", True), ("を展開して", True),
                                                ("について解いて", False), ("で微分して", False)])
    def test_japanese_counts_it_only_before_an_object_particle(self, following, want):
        """`x y について` は変数の並びであって積ではない。"""
        from ui.interpret_pipeline import _operand_is_mathematics
        assert _operand_is_mathematics("x y", following) is want

    def test_the_english_word_rule_keeps_a_real_instruction(self):
        """コーパスの行。並置の最初の版は `E1 twice` を積として登録し、階数を失っていた。"""
        r = _ask("en", ["y = k^3"], "Differentiate E1 twice with respect to k")
        assert r.get("status") == "completed", r
        assert _value(r) == "6*k", r


class TestTheVectorMarkWrapsOneName:
    @pytest.mark.parametrize("text,want", [
        ("vec(A)", "A"), ("vec(A(t))", "A(t)"), (" vec(v) ", "v"),
        ("vec(v) ~ vec(w)", None),         # two wrappers — the greedy pattern read `v) ~ vec(w`
        ("2*vec(v)", None),
    ])
    def test_unwrap(self, text, want):
        from notebook.runtime.vector_operand import unwrap_vector_mark
        assert unwrap_vector_mark(text) == want


def _ctx():
    ctx = SymbolContext()
    ctx.register("a", Quantity(name="a", kind="vector_field", components=["a_x", "a_y", "a_z"]))
    ctx.register("m", Quantity(name="m", kind="scalar"))
    ctx.register("vv", Quantity(name="vv", kind="scalar", math_type=MathType(algebra="vector")))
    ctx.register("v", Quantity(name="v", kind="anonymous", defining_expr_id="E1"))
    ctx.register("w", Quantity(name="w", kind="anonymous", defining_expr_id="E2"))
    ctx.register("u", Quantity(name="u", kind="anonymous", defining_expr_id="E3"))
    return ctx


_VALUES = {"E1": "[y, -x, 0]", "E2": "[0, 0, x]", "E3": "x^2 + y^2"}


class TestTheTypeIsReadFromTheStructure:
    @pytest.mark.parametrize("text,want", [
        ("2 v", "vector"), ("v + w", "vector"), ("x v", "vector"), ("(2 v)", "vector"),
        ("v - 2 w", "vector"), ("v . w", "scalar"), ("v w", None), ("v * w", None),
        ("v + u", None),                 # a vector plus a scalar has no type
        ("u + x", "scalar"), ("2 u", "scalar"), ("m * a", "vector"), ("a * vv", None),
    ])
    def test_with_values(self, text, want):
        from notebook.runtime.vector_op_types import infer_algebra, value_lookup
        ctx = _ctx()
        assert infer_algebra(text, ctx, value_lookup(ctx, _VALUES.get)) == want

    def test_without_values_an_h1_name_is_not_judged(self):
        """記録された `algebra='scalar'` は登録側の既定値——値が引けなければ判定しない。"""
        from notebook.runtime.vector_op_types import infer_algebra
        assert infer_algebra("2 v", _ctx()) is None


class TestComponentsOfAVectorExpression:
    def test_names_are_opened_into_lists(self):
        from notebook.runtime.vector_op_types import value_lookup
        from notebook.runtime.vector_operand import expression_components
        ctx = _ctx()
        vo = value_lookup(ctx, _VALUES.get)
        assert expression_components("2*v", ctx, vo) == tuple(
            f"(2*([y, -x, 0]))[{i}]" for i in (1, 2, 3))
        assert expression_components("2*a", ctx, vo) == tuple(
            f"(2*[a_x,a_y,a_z])[{i}]" for i in (1, 2, 3))
        assert expression_components("v . w", ctx, vo) is None
        assert expression_components("2*vv", ctx, vo) is None   # a vector with no component names
