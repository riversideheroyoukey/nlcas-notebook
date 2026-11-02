"""H-6: a vector operation must not accept an operand of the wrong type.

`div` is defined on a vector and `grad` on a scalar; the other combinations have
no answer.  Measured 2026-08-06 they came back as SUCCESSES:

    divergence of the scalar `x^2+y^2`   → `y^2+x^2`  (unchanged)
    curl of the same scalar              → `y^2+x^2`
    gradient of a vector field           → `[0=0, 0=0, 0=0]`

No error anywhere, so the researcher moves on believing a divergence was taken.
That is the condition H-6 named as its own trigger ("when operations that use
the type information exist"), and it is the same rule
`_reject_display_only_operands` already applies to display-only objects: say no
where it is used, with a reason.

**The guard must not guess** — but "do not guess" and "have not measured" are
different things, and 2026-09-17 they turned out to be the same line here. The
guard returned "unknown" whenever an UNDECLARED name was present, and the first
example in this docstring (`x^2+y^2`) is made of undeclared names: measured, its
divergence returned a raw `Maxima evaluation error`, and `H = p + q` — a case
this file listed as one the guard "has to keep letting through" — answered
**0** for the divergence and **[0 = 0, 0 = 0, 0 = 0]** for the gradient. Both
are the shapes above, surviving on the undeclared side.

A vector here is one of three things — a COMPONENT LIST, a DECLARATION, or the
system's own `vec(...)` wrapper — and an expression with none of them is a
scalar (an undeclared symbol is a free variable, a CAS atom). What the guard
still must not decide is a SHAPE with two vectors in it (dot? cross?).
"""
import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _session(lang="ja"):
    nb.start_engine(title="h6", mode="standard", language=lang,
                    domains=["electromagnetism"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["electromagnetism"], primary_backend="maxima")
    return nb, eng, sup


def _run(eng, sup, text):
    import asyncio
    tok = InputToken(type="text", value=text)
    r = asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))
    if r["status"] == "needs_confirmation" and r.get("candidates"):
        c = r["candidates"][-1]
        try:
            eng.run_operation(c["operation"], c["args"])
            return {"status": "confirmed"}
        except Exception as exc:                      # noqa: BLE001 - reported
            return {"status": "failed", "error": str(exc)}
    return r


class TestTheUndefinedCombinationsAreRefused:
    @pytest.mark.parametrize("setup,ask,word", [
        (["x, y をスカラーパラメータとして定義して", "S = x^2 + y^2"],
         "E1 の発散を求めて", "divergence"),
        (["x, y をスカラーパラメータとして定義して", "S = x^2 + y^2"],
         "E1 の回転を求めて", "curl"),
        (["a をベクトル場として定義して", "F = a"],
         "E1 の勾配を求めて", "gradient"),
    ])
    def test_refused_with_a_reason(self, setup, ask, word):
        """**断り文は研究者の語で操作を呼ぶ**（2026-09-17 に変更）。

        以前ここは `divergence` / `curl` / `gradient` という**内部名**が
        メッセージに出ることを固定していた。日本語では「E1 は scalar なので、
        divergence の対象にできません」と読め、Core Principle #1 に反する。
        語は既に在る（`_operation_label`）ので、そちらを固定する。
        """
        nb_, eng, sup = _session()
        try:
            for c in setup:
                _run(eng, sup, c)
            n_before = len(eng.expressions)
            r = _run(eng, sup, ask)
            assert r["status"] == "failed", r
            label = eng._operation_label(word)
            assert label in r["error"], (label, r["error"])
            assert word not in r["error"], f"内部名が日本語文に: {r['error']!r}"
            # the message must say what to do, not only "no"
            assert "定義" in r["error"] or "declare" in r["error"]
            assert len(eng.expressions) == n_before, "nothing may be registered"
        finally:
            nb_.stop_engine()

    def test_the_english_message_is_english(self):
        nb_, eng, sup = _session("en")
        try:
            _run(eng, sup, "Define x, y as scalars")
            _run(eng, sup, "S = x^2 + y^2")
            r = _run(eng, sup, "divergence of E1")
            assert r["status"] == "failed", r
            assert "divergence" in r["error"] and "scalar" in r["error"]
            assert not any("぀" <= ch <= "ヿ" for ch in r["error"]), r["error"]
        finally:
            nb_.stop_engine()


class TestWhatItMustStillAllow:
    """Refusing too much is the worse failure: a correct calculation stopped for
    no reason.  These pin the cases the guard has to stay out of."""

    @pytest.mark.parametrize("setup,ask", [
        (["a をベクトル場として定義して", "F = a"], "E1 の発散を求めて"),
        (["phi をスカラー場として定義して", "G = phi"], "E1 の勾配を求めて"),
        # `H = p + q`（未宣言）はここから**外した**——下の
        # `TestAnUndeclaredExpressionIsAScalar` を参照（2026-09-17、実測）。
        (["u = [1, 2, 3]"], "E1 の発散を求めて"),      # 成分リストはベクトル
        ([r"w = \vec{B}"], "E1 の発散を求めて"),       # 包みもベクトル
    ])
    def test_allowed(self, setup, ask):
        nb_, eng, sup = _session()
        try:
            for c in setup:
                _run(eng, sup, c)
            r = _run(eng, sup, ask)
            assert r["status"] == "completed", r
        finally:
            nb_.stop_engine()

    def test_a_vector_written_by_its_components_is_a_vector(self):
        r"""∇·(A×B).  The cross product's canonical is
        `[A_y*B_z-A_z*B_y, …]` and every component name is a declared SCALAR
        function — counting the names says "scalar" about the very thing that is
        a vector.  Measured when this test's older sibling in test_engine.py
        started failing on the first version of the guard."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="cross", mode=SessionMode.standard_symbolic,
                            coordinate_system="cartesian") as e:
            e.start()
            for n in ("A", "B"):
                e.run_operation("define_vector_field",
                                {"name": n, "coordinate_system": "cartesian"})
            step, _ = e.run_operation("cross_product", {"a": "A", "b": "B"})
            cid = step.output_refs[0]
            e.run_operation("divergence", {"expr": cid})   # must not raise


class TestAnUndeclaredExpressionIsAScalar:
    r"""未宣言の記号でできた式は**スカラー**である（2026-09-17）。

    このファイルの冒頭は「**guard は推測してはいけない**——未知の型は通す」と
    書き、`H = p + q` を「通さねばならない例」に置いていた。**実測するとそれは
    正しい計算ではなかった**（HEAD）:

    | 打った文 | 返っていたもの |
    |---|---|
    | `H = p + q` の**発散** | `completed` で **0** |
    | `H = p + q` の**勾配** | `completed` で **[0 = 0, 0 = 0, 0 = 0]** |

    どちらもこのファイルの docstring が「2026-08-06 に直した」と書いた形そのもの
    で、**宣言されていない側で生き残っていた**。「通さねばならない例」は、実装が
    そう振る舞うという事実を正解として写したものだった。

    ベクトルだと言えるのは**成分リスト・宣言・この系の包み `vec(...)`** の3つだけ。
    """

    @pytest.mark.parametrize("setup,ask", [
        (["H = p + q"], "E1 の発散を求めて"),
        (["H = p + q"], "E1 の回転を求めて"),
        (["f = x^2 + y^2"], "E1 の発散を求めて"),
    ])
    def test_it_is_refused_with_a_reason(self, setup, ask):
        nb_, eng, sup = _session()
        try:
            for c in setup:
                _run(eng, sup, c)
            r = _run(eng, sup, ask)
            assert r["status"] == "failed", r
            assert "Maxima evaluation error" not in r["error"], r["error"]
            assert "スカラー" in r["error"], r["error"]
        finally:
            nb_.stop_engine()

    def test_the_gradient_of_such_an_expression_still_works(self):
        """勾配はスカラーに対して定義されるので、断ってはいけない."""
        nb_, eng, sup = _session()
        try:
            _run(eng, sup, "H = p + q")
            r = _run(eng, sup, "E1 の勾配を求めて")
            assert r["status"] == "completed", r
        finally:
            nb_.stop_engine()


class TestTheInferenceItself:
    def _ctx(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity, MathType
        ctx = SymbolContext()
        ctx.register("a", Quantity(name="a", kind="vector_field"))
        ctx.register("b", Quantity(name="b", kind="vector_field"))
        ctx.register("m", Quantity(name="m", kind="scalar"))
        ctx.register("v", Quantity(name="v", kind="scalar",
                                   math_type=MathType(algebra="vector")))
        return ctx

    @pytest.mark.parametrize("text,expected", [
        ("m * a", "vector"),          # scalar × vector — unambiguous
        ("m + 1", "scalar"),
        ("a", "vector"),
        ("v", "vector"),              # the other place a type can live
        ("a * b", None),              # two vectors: dot? cross? — do not guess
        # `m * p`（p は未宣言）は **2026-09-17 に `scalar` へ変わった**。
        # 以前の `None`（判定しない）は「推測しない」ではなく**測っていない**
        # だった——ベクトルだと言うものが何も無い式は、この系ではスカラーである
        # （未宣言の記号は自由変数＝CAS の記号原子）。言わなかった代償は、
        # `H = p + q` の発散が **0** を、勾配が **[0=0, 0=0, 0=0]** を
        # `completed` で返すことだった（実測 2026-09-17、HEAD）。
        ("m * p", "scalar"),
        ("[1, 2, 3]", "vector"),      # component list
        ("vec(B)", "vector"),         # この系自身の包み（宣言が無くてもベクトル）
    ])
    def test_infer(self, text, expected):
        from notebook.runtime.vector_op_types import infer_algebra
        assert infer_algebra(text, self._ctx()) == expected
