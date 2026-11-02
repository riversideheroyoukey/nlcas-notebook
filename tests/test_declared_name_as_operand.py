r"""OPERAND-BOUNDARY 残①（能力の半分, 2026-09-11）。

**宣言された対象の名前は、それ自体が式である——渡せない操作には、満たされて
いない条件を述べて断る。**

H-1 で定義した名前（`u = …`）は 2a-sym が式番号へ書き換えるので通っていたが、
**宣言だけの対象**（ベクトル場・行列…）には `defining_expr_id` が無いため
書き換え先が無く、素の ASCII として `param_tokens` へ落ちて `expr` が誰にも
埋められなかった。日英とも「必要な引数が不足しています: ['expr']」——研究者は
オペランドを書いているのに、である。

**広げるだけでは駄目だった。** 開いた面を全数で測ると、`expr` を取る 47 操作の
うち宣言を読むのはベクトル場で3つだけ（`curl`/`divergence`/`norm`）、9つが断り、
**35 が宣言を無視して素の記号として CAS に渡していた**——`differentiate`→`0`、
`integrate`→`A*x`、`summation`→`2*A`。系自身の `divergence` が
`'diff(A_x,x,1)` を返すことと矛盾する。失敗が**もっともらしい誤答**に変わる
取引なので、読めない操作は断る（`operand_guards.reject_declared_object_without_value`）。

**断り文は条件を述べる**（依頼者の指定）: 何として宣言されているか、**値が
与えられていないこと**、そして**測って通ることを確かめた道**——値を与える／
宣言だけで計算できる操作を使う。行列には後者が1つも無いので、その場合は
前者だけを案内する。

**免除は「操作と種別の組」で決まる。** `gradient`/`scalar_laplacian` はスカラー場
を読むが、行列・テンソルには `[0,0,0]`/`0` を返す。操作名だけで免除すると、
塞いだはずの穴をその2つが開けたままにする（実測）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_DOM = ["physics"]


def _ask(lang, setup, text):
    async def go():
        nb.start_engine(title="declared-operand", mode="standard", language=lang,
                        domains=_DOM)
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=_DOM,
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_VEC = {"ja": ["A をベクトル場とする"], "en": ["define A as a vector field"]}
_MAT = {"ja": ["G を行列とする"], "en": ["declare G as a matrix"]}
_SCA = {"ja": ["f をスカラー場とする"], "en": ["define f as a scalar field"]}


class TestTheNameReachesTheOperation:
    """**読める操作には通る**——これが起票の言う「能力の半分」."""

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "A のノルムを求めて", "sqrt(A_z^2+A_y^2+A_x^2)"),
        ("en", "the norm of A", "sqrt(A_z^2+A_y^2+A_x^2)"),
        ("en", "find the norm of A", "sqrt(A_z^2+A_y^2+A_x^2)"),
    ])
    def test_the_norm_of_a_declared_field(self, lang, text, want):
        r = _ask(lang, _VEC[lang], text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").replace(" ", "") == want, r

    @pytest.mark.parametrize("lang,text", [
        ("ja", "A の発散を求めて"), ("en", "the divergence of A"),
    ])
    def test_the_divergence_of_a_declared_field(self, lang, text):
        r = _ask(lang, _VEC[lang], text)
        assert r.get("status") == "completed", r
        assert "diff(A_x,x,1)" in (r.get("canonical") or "").replace(" ", ""), r

    @pytest.mark.parametrize("lang,text", [
        ("ja", "f を x で微分して"), ("en", "differentiate f with respect to x"),
    ])
    def test_a_scalar_field_still_differentiates(self, lang, text):
        r"""スカラー場は `depends` つきで宣言されるので `'diff(f,x,1)` は正直——
        **免除表に載せてあるのは、そう測れたからである**。

        日本語の `偏微分して` は素の名前では届かない（`f を x で偏微分して` は
        どの規則にも当たらない。E 番号なら通る）——名前が流れるようになって
        初めて見えた綴りの穴で、この変更とは別（open_items OPERAND-BOUNDARY 残）。
        """
        r = _ask(lang, _SCA[lang], text)
        assert r.get("status") == "completed", r
        assert "diff(f,x,1)" in (r.get("canonical") or "").replace(" ", ""), r


class TestTheRefusalStatesTheConditionThatIsNotMet:
    """「できません」ではなく、**何が足りないか**と**満たす道**を言う."""

    @pytest.mark.parametrize("lang,setup,text,name,kind_word", [
        ("ja", "ja", "A を x で微分して", "A", "ベクトル場"),
        ("en", "en", "differentiate A with respect to x", "A", "a vector field"),
    ])
    def test_a_vector_field_names_both_roads(self, lang, setup, text, name, kind_word):
        r = _ask(lang, _VEC[setup], text)
        assert r.get("status") == "failed", r
        msg = r.get("error") or ""
        assert name in msg and kind_word in msg, msg          # 何として宣言したか
        assert ("値が与えられていません" in msg
                or "no value has been given" in msg), msg      # 満たされていない条件
        assert "pmatrix" in msg, msg                           # ① 値を与える例
        assert ("発散を計算する" in msg or "divergence" in msg), msg   # ② 読める操作
        # **案内に免除の一覧をそのまま使わない**——`\nabla f` はベクトル場に対して
        # 型エラーになるので、勧めてはいけない（`declaration_computing_ops`）。
        assert "勾配" not in msg and "gradient" not in msg, msg

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "G を x で微分して"),
        ("en", "en", "differentiate G with respect to x"),
    ])
    def test_a_matrix_is_offered_only_the_road_that_works(self, lang, setup, text):
        r"""**案内した回避策は効かねばならない。** 行列の宣言だけから計算できる
        操作は1つも無い（実測）ので、そこを案内してはいけない."""
        r = _ask(lang, _MAT[setup], text)
        assert r.get("status") == "failed", r
        msg = r.get("error") or ""
        assert ("いまのところありません" in msg
                or "No operation computes" in msg), msg
        assert "&" in msg, msg        # 行列の例（2×2）であって列ベクトルではない
        assert "発散" not in msg and "divergence" not in msg, msg

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "G の勾配を求めて"),
        ("en", "en", "the gradient of G"),
    ])
    def test_gradient_does_not_slip_through_on_a_matrix(self, lang, setup, text):
        r"""免除を**操作名だけ**で書くと、`gradient` が行列に `[0,0,0]` を返して
        穴が開いたままになる（実測 2026-09-11）."""
        r = _ask(lang, _MAT[setup], text)
        assert r.get("status") != "completed", r
        assert "[0,0,0]" not in (r.get("canonical") or ""), r


class TestTheRoadsInTheMessageActuallyWork:
    """断り文が案内する道を、**そのまま打って**確かめる."""

    @pytest.mark.parametrize("lang,setup,value,text,want", [
        ("ja", "ja", r"A = \begin{pmatrix} x^2 \\ 0 \\ 0 \end{pmatrix}",
         "A を x で偏微分して", "matrix([2*x],[0],[0])"),
        ("en", "en", r"A = \begin{pmatrix} x^2 \\ 0 \\ 0 \end{pmatrix}",
         "take the partial derivative of A with respect to x",
         "matrix([2*x],[0],[0])"),
    ])
    def test_giving_a_value_satisfies_the_condition(self, lang, setup, value,
                                                    text, want):
        r = _ask(lang, _VEC[setup] + [value], text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").replace(" ", "") == want, r


class TestWhatMustNotChange:

    @pytest.mark.parametrize("lang,text", [
        ("ja", "u のノルムを求めて"), ("en", "the norm of u"),
    ])
    def test_a_name_that_has_a_value_is_untouched(self, lang, text):
        r"""H-1 の名前は `defining_expr_id` を持つので、guard は見ない."""
        setup = [r"u = \begin{pmatrix} 3 \\ 4 \end{pmatrix}"]
        r = _ask(lang, setup, text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").strip() == "5", r

    @pytest.mark.parametrize("lang,text", [
        ("ja", "x をスカラーとする"), ("en", "let x be a scalar"),
    ])
    def test_a_scalar_is_not_an_object(self, lang, text):
        r"""スカラーは変数として文中に立つ——対象として数えない."""
        r = _ask(lang, [], text)
        assert r.get("status") in ("confirmed", "declared"), r


class TestAScalarFieldIsGuardedToo:
    r"""スカラー場は `differentiate` だけが正直で、**積分も簡約も宣言を忘れる**
    （`\int f\,dx` → `f*x`、実測）——種別ごとの免除表がそこを分けている."""

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "f を x で不定積分して"),
        ("en", "en", "integrate f with respect to x"),
        ("ja", "ja", "f を簡約して"),
        ("en", "en", "simplify f"),
    ])
    def test_an_operation_that_forgets_the_declaration_is_refused(
            self, lang, setup, text):
        r = _ask(lang, _SCA[setup], text)
        assert r.get("status") == "failed", r
        msg = r.get("error") or ""
        assert ("値が与えられていません" in msg
                or "no value has been given" in msg), msg

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "f を簡約して"), ("en", "en", "simplify f"),
    ])
    def test_the_example_actually_shows_a_value(self, lang, setup, text):
        r"""「例: `f`」は**値の与え方の例になっていない**——右辺が要る
        （表にスカラー場の欄が無く、自分で踏んだ）."""
        msg = _ask(lang, _SCA[setup], text).get("error") or ""
        assert "`f = " in msg, msg


class TestABetterReasonIsNotReplaced:
    r"""**手前の guard が、より正確な理由を消してはいけない**（全スイートが
    落ちて気づいた、実測 2026-09-11）。

    `curl` にスカラー場を渡すと「'f' はスカラー場ですが、ベクトル場が必要です」
    という**型の理由**が返る。値を与えても `curl` はスカラー場に対して正しく
    ないので、**満たされていない条件は値ではない**——だから型エラーを出す
    (操作, 種別) の組も免除表に入れる。
    """

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "f の回転を求めて"), ("en", "en", "the curl of f"),
    ])
    def test_a_scalar_field_given_to_curl_keeps_the_type_error(
            self, lang, setup, text):
        r = _ask(lang, _SCA[setup], text)
        assert r.get("status") == "failed", r
        msg = r.get("error") or ""
        # 型の断り（2026-09-17 に日英同時の手書き文面から i18n の鍵へ移した）
        from notebook.runtime.i18n import t as _t
        assert msg == _t("vector.operand_is_scalar", lang, operand="f"), msg
        assert "値が与えられていません" not in msg, msg
        assert "no value has been given" not in msg, msg

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "ja", "A の勾配を求めて"), ("en", "en", "the gradient of A"),
    ])
    def test_a_vector_field_given_to_gradient_keeps_the_type_error(
            self, lang, setup, text):
        r = _ask(lang, _VEC[setup], text)
        assert r.get("status") == "failed", r
        msg = r.get("error") or ""
        from notebook.runtime.i18n import t as _t
        assert msg == _t("vector.operand_is_vector", lang, operand="A"), msg


class TestTheRoadsItNamesAreMeasured:
    r"""**案内した回避策は効かねばならない**（この系の既存の教訓）。

    免除の一覧（`declaration_reading_ops`）は**型エラーを出す組も含む**ので、
    そのまま案内すると `\nabla f` をベクトル場に勧めることになる。案内は
    `declaration_computing_ops` から取り、ここでその全要素を実際に走らせる。
    """

    @pytest.mark.parametrize("decl,kind", [
        ("define_vector_field", "vector_field"),
        ("define_scalar_field", "scalar_field"),
    ])
    def test_every_suggested_operation_computes(self, decl, kind):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="suggested") as engine:
            suggested = engine._handler.declaration_computing_ops(kind, "scalar")
            assert suggested, kind
        for op in suggested:
            with NotebookEngine(title="suggested") as engine:
                engine.run_operation(decl, {"name": "A"})
                if op in ("dot_product", "cross_product"):
                    engine.run_operation(decl, {"name": "B"})
                    args = {"a": "A", "b": "B"}
                elif op == "differentiate":
                    args = {"expr": "A", "var": "x"}
                else:
                    args = {"expr": "A"}
                _step, value = engine.run_operation(op, args)   # 断れば例外で落ちる
                assert str(value).strip(), (op, kind, value)

    @pytest.mark.parametrize("decl,kind", [
        ("define_matrix", "matrix"), ("define_tensor", "tensor"),
    ])
    def test_a_kind_with_no_road_names_none(self, decl, kind):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="none") as engine:
            assert engine._handler.declaration_computing_ops(kind, kind) == ()
