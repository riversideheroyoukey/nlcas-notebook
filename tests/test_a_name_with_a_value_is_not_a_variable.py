r"""DEFINED-NAME-AS-VARIABLE (2026-09-19) — 値を持つ名前は変数ではない。

H-1 の P2 は定義を CAS 側へ束縛する（`y : x^3`）。その名前が操作の**変数の欄**に
立つと Maxima は断り、研究者に返るのは生の `Maxima evaluation error` ——原因も次の
一手も言っていない文字列である（Core Principle #1）。実測 2026-09-19、`z^2 + y` を
登録して `y = x^3` と書いたあと、日英の差は無く6綴り:

| 打った文 | 返っていたもの |
|---|---|
| `E1 を y で偏微分して` / 全微分 / 定積分 / テイラー展開 | 生の `Maxima evaluation error` |
| `E1 を y で積分して` / 極限 / `integrate E1 over y` / `substitute y = 3 into E1` | 「必要な引数が不足」（2a-sym が名前を式番号へ書き換える別の根、起票 DEFINED-NAME-VAR-REWRITTEN） |
| `E1 の y に 3 を代入して` | **正しい**——`block([y], …)` で影に隠しているから |

**∇ と同じ直し方（`block([…])` の影）は実測で却下されている**——P2 は代入時に
評価するので `f = 1/r` は `1/sqrt(x^2+y^2+z^2)` として格納され、`r` という記号は
式の中にもう無い。隠すと `diff(f,r)` は `-1/r^2` ではなく `0` を返す。

**境界は「欄」ではなく CAS にある**（実測、束縛された名前を変数の欄に置いて15操作）。
断るのは `diff` / 定積分 / `taylor` / `limit` / `residue` だけで、`solve` / `sum` /
`product` / `ratsimp` / `partfrac` / `charpoly` / `coeff` は**答える。そしてそれが
正しい**——`solve` は「その名前が指すもの」について解くので、`scenario_coulomb_gauge`
は `Az_z = \partial A_z/\partial z` と名前をつけてからゲージ条件を `Az_z` について
解いている。だから**断ったあとに読み替える**（事前に欄で断ると、その本筋が止まる
——コーパス差分が捕まえた）。

**決めるのは CAS で、帳簿ではない。** `defining_expr_id` は上位集合である。
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
        nb.start_engine(title="dnv", mode="standard", language=lang, domains=[domain])
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
            exprs = {k: v.latex for k, v in engine.expressions.items()}
            return out[-1], exprs
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


def _msg(r):
    return str(r.get("error") or r.get("message") or "")


#: `y` に値を与えたあと。E1 は `z^2 + y`、E2 が `y = x^3`。
_BOUND = ["z^2 + y", "y = x^3"]
#: 同じ文を、値を与えずに打った列（**対照**——ここが変わってはならない）。
_FREE = ["z^2 + y"]

#: 生の CAS 文言が返っていた綴り（Core Principle #1）。
_WAS_RAW_CAS = [
    ("ja", "E1 を y で偏微分して", "1"),
    ("ja", "E1 を y で全微分して", "1"),
    ("ja", "E1 を y で 0 から 1 まで定積分して", "(2*z^2+1)/2"),
    ("ja", "E1 を y について 0 のまわりで 3 次まで展開して", "z^2+y"),
    ("en", "differentiate E1 with respect to y partially", "1"),
    ("en", "integrate E1 over y from 0 to 1", "(2*z^2+1)/2"),
    ("en", "expand E1 in y around 0 to order 3", "z^2+y"),
]


class TestNoRawCasTextReachesTheResearcher:
    @pytest.mark.parametrize("lang,text,_want_free", _WAS_RAW_CAS)
    def test_it_is_refused_with_a_reason(self, lang, text, _want_free):
        r, _ = _run(lang, _BOUND + [text])
        assert r.get("status") == "failed", r
        assert "Maxima" not in _msg(r), r
        assert "y" in _msg(r) and "x^3" in _msg(r) and "E2" in _msg(r), r

    @pytest.mark.parametrize("lang,text,want_free", _WAS_RAW_CAS)
    def test_the_same_sentence_without_a_value_is_unchanged(self, lang, text, want_free):
        """対照。**断るのは値が在るときだけ**——普通の変数は通り続ける。"""
        r, _ = _run(lang, _FREE + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want_free, r


class TestTheBoundaryIsTheCasNotTheField:
    r"""`solve` は同じ形で**正しく答える**——欄で先回りして断ってはならない。

    実測 2026-09-19: `Az_z = \partial A_z/\partial z` と名前をつけてからゲージ条件を
    `Az_z` について解くのは `scenario_coulomb_gauge` の本筋であり、Maxima は
    `'diff(A_z,z,1) = -'diff(A_y,y,1)-'diff(A_x,x,1)` と答える。欄（`var`）だけで
    断った最初の実装はこれを止め、**コーパス差分が捕まえた**。
    """

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を g について解いて"), ("en", "solve E1 for g"),
    ])
    def test_solving_for_a_named_subexpression_still_works(self, lang, text):
        r, exprs = _run(lang, ["a + b + c = 0", "g = c", text])
        assert r.get("status") == "completed", r
        assert any((v or "").replace(" ", "") == "c=-b-a" for v in exprs.values()), exprs

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を y について解いて"), ("en", "solve E1 for y"),
    ])
    def test_solving_for_a_defined_name_answers_about_what_it_names(self, lang, text):
        """`y` が `x^3` なら `y = z^2` は `x^3 = z^2` である——同じ読みである。"""
        r, exprs = _run(lang, ["y - z^2", "y = x^3", text])
        assert r.get("status") == "completed", r
        assert any((v or "").replace(" ", "") == "x^3=z^2" for v in exprs.values()), exprs


class TestTheMessageNamesTheConditionAndTwoRoads:
    r"""断り文は「できない」ではなく**満たされていない条件**を述べる（依頼者の指定）。

    案内する道は実測してある（`y = x^3` のあと）:

        E1 を u で偏微分して  →  0        ← 値を持たない名前
        E1 を x で偏微分して  →  3*x^2    ← 値が書かれている変数
    """

    @pytest.mark.parametrize("lang,text,frag", [
        ("ja", "E1 を y で偏微分して", "値を持つ名前は変数ではない"),
        ("en", "differentiate E1 with respect to y partially",
         "A name that holds a value is not a variable"),
    ])
    def test_the_researchers_language(self, lang, text, frag):
        r, _ = _run(lang, _BOUND + [text])
        assert frag in _msg(r), r

    def test_it_carries_the_value_as_the_researcher_wrote_it(self):
        r"""`\sqrt{x^2+y^2+z^2}` であって内部の正準形ではない（起票の例）。"""
        r, _ = _run("ja", [r"r = \sqrt{x^2+y^2+z^2}", "f = 1/r", "E2 を r で偏微分して"])
        assert r.get("status") == "failed", r
        assert r"\sqrt{x^2+y^2+z^2}" in _msg(r), r
        assert "E1" in _msg(r), r

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を y で偏微分して"),
        ("en", "differentiate E1 with respect to y partially"),
    ])
    def test_it_shows_the_value_and_not_the_whole_equation(self, lang, text):
        """与えられているのは値であって等式ではない——`y = x^3` ではなく `x^3`。"""
        r, _ = _run(lang, _BOUND + [text])
        assert "`x^3`" in _msg(r), r
        assert "y = x^3" not in _msg(r), r

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 を u で偏微分して", "0"),
        ("ja", "E1 を x で偏微分して", "3*x^2"),
        ("en", "differentiate E1 with respect to u partially", "0"),
        ("en", "differentiate E1 with respect to x partially", "3*x^2"),
    ])
    def test_the_roads_it_names_are_measured(self, lang, text, want):
        r, _ = _run(lang, _BOUND + [text])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheLedgerIsNotTheDecider:
    r"""起票の反例——事前に帳簿で断ると、**通る文**を断ってしまう。

    実測 2026-09-19: `defining_expr_id` は「CAS が値を持つ」の上位集合である。

    | その名前 | `defining_expr_id` | CAS が値を持つか |
    |---|---|---|
    | `y = x^3`（H-1） | `E2` | **true** |
    | `2x = 4` を解いた `x` | `E2` | **false** |
    | 連立を解いた `x`, `y` | `E3` / `E4` | **false** |
    """

    @pytest.mark.parametrize("lang,lines", [
        ("ja", ["2*x = 4", "E1 を x について解いて", "E2 を x で偏微分して"]),
        ("en", ["2*x = 4", "solve E1 for x", "differentiate E2 with respect to x partially"]),
    ])
    def test_a_solved_variable_is_still_a_variable(self, lang, lines):
        r, _ = _run(lang, lines)
        assert r.get("status") == "completed", r

    def test_the_guard_asks_the_cas_not_the_ledger(self):
        """単体で——帳簿が言っても CAS が持たなければ断らない。"""
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.operand_guards import translate_value_bearing_variable_failure

        ctx = SymbolContext()
        ctx.register("x", Quantity(name="x", kind="anonymous", defining_expr_id="E2"))
        args = {"expr": "E1", "var": "x"}
        kw = dict(symbol_context=ctx, value_of_name=lambda _e: "2",
                  operation_label="differentiate", lang="en")
        translate_value_bearing_variable_failure("differentiate", args,
                                      cas_holds_value=lambda _n: False, **kw)
        with pytest.raises(ValueError):
            translate_value_bearing_variable_failure("differentiate", args,
                                          cas_holds_value=lambda _n: True, **kw)

    def test_only_the_variable_field_is_read(self):
        r"""断り文は「**{操作}の変数にはできません**」と言う——だから読む欄は
        変数を名指す欄だけである。`substitute` の `lhs` も束縛された名前を運ぶが、
        あちらは `block([y], …)` で影に隠すので失敗せず、広げても**振る舞いは
        変わらない**（変異試験で実測）。変わるのは文面が真かどうかで、そこは
        単体で固定する。"""
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.operand_guards import \
            translate_value_bearing_variable_failure as _translate

        ctx = SymbolContext()
        ctx.register("y", Quantity(name="y", kind="anonymous", defining_expr_id="E2"))
        _translate("substitute", {"expr": "E1", "lhs": "y", "rhs": "3"},
                   symbol_context=ctx, cas_holds_value=lambda _n: True,
                   value_of_name=lambda _e: "x^3", operation_label="substitute",
                   lang="en")

    def test_a_name_the_notebook_never_defined_is_not_asked_about(self):
        """帳簿は**安いふるい**——普通の変数では CAS に訊きに行かない。"""
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.operand_guards import translate_value_bearing_variable_failure

        asked = []

        def _ask(name):
            asked.append(name)
            return True

        translate_value_bearing_variable_failure(
            "differentiate", {"expr": "E1", "var": "t"},
            symbol_context=SymbolContext(), cas_holds_value=_ask,
            value_of_name=lambda _e: "", operation_label="differentiate", lang="en")
        assert asked == []


class TestWhatIsDeliberatelyNotTouched:
    def test_substitute_still_computes(self):
        r"""`lhs` は変数の欄ではなく、その builder は `block([y], …)` で影に隠す
        ——実測、束縛が在っても答えは `z^2+3` で正しい。"""
        for lang, text in (("ja", "E1 の y に 3 を代入して"),
                           ("en", "substitute 3 for y in E1")):
            r, _ = _run(lang, _BOUND + [text])
            assert r.get("status") == "completed", (lang, r)
            assert _value(r) == "z^2+3", (lang, r)

    def test_solve_system_shadows_its_own_variables(self):
        r"""`vars` を欄の集合に入れなかった理由（実測 2026-09-19）——
        `_build_solve_system` は `block([x, y], …)` を出すので、束縛が在っても
        無くても**同じ命令**になる。"""
        r, _ = _run("ja", ["x + y = 3", "x - y = 1", "y = 5",
                           "E1 と E2 を x と y について解いて"])
        assert r.get("status") == "completed", r


class TestTheBackendAnswersWhetherItHoldsAValue:
    def test_maxima_asks_its_own_list_of_bound_names(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler

        handler = MaximaBackendHandler(None)
        with MaximaAdapter() as adapter:
            assert handler.cas_holds_value(adapter, "u") is False
            adapter.execute("y : x^3")
            assert handler.cas_holds_value(adapter, "y") is True
            assert handler.cas_holds_value(adapter, "u") is False

    def test_a_name_that_is_not_an_identifier_is_never_sent(self):
        """CAS へ送る前に形で落とす——`values` に式を投げても意味が無い。

        `E1` は**落とさない**——Maxima の識別子として正しく、`values` に無いので
        `false` が返るだけである。式番号を変数の欄から外すのは帳簿側の仕事で、
        そこには `E1` という名前の量が無い。
        """
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler

        class _Boom:
            def execute(self, _cmd):
                raise AssertionError("CAS に訊いてはいけない")

        handler = MaximaBackendHandler(None)
        for bad in ("x + 1", "", "\\alpha", "y(1)", "a b"):
            assert handler.cas_holds_value(_Boom(), bad) is False

    def test_a_backend_with_no_answer_says_no(self):
        from notebook.runtime.backend_handler import BackendHandler
        assert BackendHandler.cas_holds_value(object(), object(), "y") is False
