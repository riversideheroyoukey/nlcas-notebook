r"""式どうしの積が、行列に対して要素積になっていた——**値が誤って `completed`**。

実測 2026-09-07（NL-GRAMMAR ① 四十三問目で `matrix` の読みを測っている途中に踏み、
MATMUL-TEMPLATE-STAR として起票 → 本問で根治）。
`[2,1;1,2] · [1,0;0,3]` の正解は `[2,3;1,6]`、要素積なら `[2,0;0,6]`:

| 打った文 | 走った操作 | 直す前の値 |
|---|---|---|
| `find the matrix product of E1 and E2` | `matrix_multiply` | `matrix([2,3],[1,6])` ✔ |
| `multiply the matrices E1 and E2` | `multiply_expressions` | **`matrix([2,0],[0,6])`** |
| `E1 と E2 を掛け合わせて` | `combine_expressions` | **`matrix([2,0],[0,6])`** |

**規則も仕組みも既にあった。** CLAUDE.md の AUDIT-MATMUL-STAR がこの罠を記録し、
`MaximaCodeGenerator._mul_operator` が「オペランドのどれかが matrix/vector/tensor
なら `.`」を実装している。届いていなかったのは**テンプレート経路**——
`backends/maxima.full.json` が `ratsimp(({a}) * ({b}))` と `*` を直書きしており、
そこは IR の `Mul` を通らないので生成器の判断が及ばない。

**直しは2箇所だけ。** ①判断を `product_operator` として1つに出し、生成器と
テンプレート経路の**両方が同じものを読む**。②`get_algebra_type` が**行列リテラル**を
読めるようにする——`[a,b;c,d]` は `SharedParser` が `FunctionCall(matrix, rows)` に
するが、`get_algebra_type` の汎用 `args` 走査は行の中の**数**を見て「scalar」と
答えていた。宣言済みの行列は `Symbol` 枝が既に答えていたので、**欠陥はリテラルに
だけ出ていた**（`A * B` は最初から `.` になる）。

`matrix` という名前は **IR のもの**で Maxima のものではない（`SharedParser` は
意味を持たない層で、両バックエンドが同じ節点を見る）ので、共有の
`get_algebra_type` が読んでよい。

**同じ根から、2つ目の silent-wrong も直った。** 冪の生成器は
`matrix([1,2],[3,4])^2` が要素ごとの平方で `^^2` が行列の平方だと**コメントに
書いたうえで guard を持っていた**——が、その guard も `get_algebra_type` に訊くので、
リテラルに対しては**一度も発火していなかった**。実測: `[2,1;1,2]^2` の簡約が
`matrix([4,1],[1,4])`（要素ごと）を `completed` で返していた。正しくは
`matrix([5,4],[4,5])`。事前点検の「その値を他に誰が読むか数える」で出た
（`get_algebra_type` の消費者は12箇所）。

**`/` は触っていない。** 行列の除算に当たる演算子が無く、`A . B^^-1` に書き換える
のは「B が正則である」という、記法が言っていないことを仮定する。`+` / `-` は
行列でも要素ごとで正しいので、`op == "*"` のときだけ判断する。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_code_generator import product_operator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.semantic_resolver import get_algebra_type
from notebook.runtime.shared_parser import SharedParser
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

#: `[2,1;1,2] . [1,0;0,3] = [2,3;1,6]`。要素積なら `[2,0;0,6]` になる——
#: **対角でない数表を選んである**（対角どうしだと両者が一致して何も測れない）。
MATS = ["[2, 1; 1, 2]", "[1, 0; 0, 3]"]
PRODUCT = "matrix([2,3],[1,6])"
HADAMARD = "matrix([2,0],[0,6])"


def _ask(lang, text, prep):
    nb.start_engine(title="mts", mode="standard", language=lang, domains=[])
    calls: list = []
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        orig = eng.run_operation

        def spy(op, args):
            calls.append(op)
            return orig(op, args)
        eng.run_operation = spy

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(InterpretRequest(
                    tokens=[InputToken(type="text", value=x)]), eng, sup))
        for p in prep:
            run(p)
        calls.clear()
        r = run(text)
        return calls, r.get("status"), (r.get("canonical") or "").replace(" ", "")
    finally:
        nb.stop_engine()


class TestTheProductOfTwoMatricesIsTheMatrixProduct:
    """**値で測る。** どの操作に届いたかでは足りない——`multiply_expressions` に
    届くのは正しく、返っていた値が誤りだった。"""

    @pytest.mark.parametrize("lang,text", [
        ("en", "find the matrix product of E1 and E2"),
        ("en", "multiply the matrices E1 and E2"),
        ("ja", "E1 と E2 の行列積を求めて"),
        ("ja", "E1 と E2 を掛け合わせて"),
        ("ja", "E1 と E2 を掛ける"),
    ])
    def test_value(self, lang, text):
        calls, st, val = _ask(lang, text, MATS)
        assert st == "completed", (text, st, calls)
        assert val == PRODUCT, (text, calls, val)
        assert val != HADAMARD, text

    def test_declared_symbolic_matrices(self):
        """宣言済みの記号行列でも、**2式を繋ぐ積**が `*` のままだった
        （内側の積は生成器が `.` にしていたので `A . B*B . A` と混在していた）。"""
        calls, st, val = _ask("ja", "E1 と E2 を掛け合わせて",
                              ["A を行列とする", "B を行列とする", "A * B", "B * A"])
        assert st == "completed", (st, calls)
        assert "*" not in val, val          # `A . B^^2 . A`
        assert val.startswith("A."), val


class TestTheSameRootAlsoFixedTheMatrixPower:
    """生成器の guard は最初から在り、**リテラルが scalar と答えるので発火して
    いなかった**だけ。`get_algebra_type` を直すと同時に効く。"""

    def test_a_matrix_literal_squared_is_the_matrix_square(self):
        calls, st, val = _ask("ja", "E1 を簡約して", ["[2, 1; 1, 2]^2"])
        assert st == "completed", (st, calls)
        assert val == "matrix([5,4],[4,5])", val      # 直す前は matrix([4,1],[1,4])

    def test_a_scalar_power_is_untouched(self):
        _, st, val = _ask("ja", "E1 を簡約して", ["(x + 1)^2"])
        assert st == "completed" and "x^2" in val, val


class TestWhatMustNotChange:
    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 と E2 を掛け合わせて"), ("en", "multiply E1 and E2"),
    ])
    def test_scalars_are_untouched(self, lang, text):
        calls, st, val = _ask(lang, text, ["f = x + 1", "g = x - 1"])
        assert st == "completed" and val.startswith("x^2-1"), (text, st, val)

    @pytest.mark.parametrize("text,want", [
        ("E1 と E2 を足して", "matrix([3,1],[1,5])"),
        ("E1 から E2 を引いて", "matrix([1,1],[1,-1])"),
    ])
    def test_plus_and_minus_stay_elementwise(self, text, want):
        """`+` / `-` は行列でも要素ごとが正しい。判断するのは `*` のときだけ。"""
        _, st, val = _ask("ja", text, MATS)
        assert st == "completed" and val == want, (text, st, val)


class TestTheDecisionLivesInOnePlace:
    @staticmethod
    def _ir(text, ctx=None):
        ctx = ctx or SymbolContext()
        return MaximaSemanticResolver().resolve(SharedParser().parse(text), ctx), ctx

    @pytest.mark.parametrize("text,want", [
        ("[2, 1; 1, 2]", "matrix"),      # 直す前は "scalar"——行の中の数しか見ていなかった
        ("[1, 0; 0, 3]", "matrix"),
        ("x^2 + 1", "scalar"),
        ("[a, b]", "scalar"),            # これは交換子であって行列ではない
    ])
    def test_get_algebra_type_reads_a_matrix_literal(self, text, want):
        ir, ctx = self._ir(text)
        assert get_algebra_type(ir, ctx) == want, (text, ir)

    def test_product_operator(self):
        lit, ctx = self._ir("[2, 1; 1, 2]")
        sca, _ = self._ir("x + 1")
        assert product_operator([lit, sca], ctx).strip() == "."
        assert product_operator([sca, sca], ctx).strip() == "*"
        assert product_operator([sca], None).strip() == "*"     # 文脈が無ければ従来どおり

    def test_the_generator_asks_the_same_helper(self):
        """`_mul_operator` を別に綴らない——**同じ問いは1箇所**。"""
        import inspect
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        src = inspect.getsource(MaximaCodeGenerator._mul_operator)
        assert "product_operator(" in src, src
        assert "get_algebra_type" not in src, src

    def test_the_template_has_an_operator_slot(self):
        import json
        import pathlib
        d = json.loads((pathlib.Path(__file__).resolve().parent.parent / "backends"
                        / "maxima.full.json").read_text(encoding="utf-8"))
        ops = d.get("operations", d)
        assert "{op}" in ops["multiply_expressions"]["template"]
        assert "{op}" in ops["combine_expressions"]["template"]
