r"""REFUSAL-NOT-REACHED ③（2026-09-17）。

**添字つきの微分形式が測度として読まれず、積分が黙って積になっていた。**

    \int f dx     → f*x                    ✔
    \int x^2 dx   → x^3/3                  ✔
    \int f(x) dx  → 'integrate(f(x),x)     ✔
    \int f dx_1   → **dx(1)*f*int**        ← `completed`

`\int` が記号 `int` のまま残り、微分形式が因数になった**積**が成功として返る。
原因は測度の読み手で、`dx` は `Symbol("dx")` として届くが `dx_1` は
`Subscript(base=Symbol("dx"), sub=Number(1))` として届く——添字がトークンを割るので、
記号の判定が当たらなかった。**添字の綴りだけ**が落ちていた。

**最初の直しは、見える積を見えない積に変えただけだった。** `dx_1` を
`Symbol("x_1")` に畳んで測度にすると:

    \int x_1 dx_1    → **x(1)*x_1**        （正しくは x_1^2/2）
    \int x_1^2 dx_1  → **x(1)^2*x_1**

被積分関数の `x_1` は未宣言なので添字として `x(1)` に降り、測度だけが原子 `x_1` に
なる——**別の CAS 物**なので、CAS は被積分関数を定数と見る。前より悪い（前の形には
少なくとも `int` という見える異物があった）。**測ってから気づいた**。

**直しは「同じ扉を通す」。** 測度の変数を、被積分関数と同じ解決器に通す
（`resolve_node_fn`）。素の `dx` では恒等で、添字つきでだけ効く。そのうえで
未宣言の添字は `IndexedSymbol` になり、`Integral.var` は `Symbol` を要求するので
——**IR 自身が「積分変数は記号である」と言っている**——生の pydantic
ValidationError になる手前で、**既に在る断り**に繋ぐ（`integral.var_is_indexed`。
文の綴り「… を x_1 で 0 から 1 まで定積分して」が返すのと同じ理由・同じ2つの出口）。

**案内した回避策が効くことも測った**: `x_1` を宣言すれば `\int x_1 dx_1` は
`Integral(var=Symbol("x_1"))` になり、被積分関数と一致する。
"""
import pytest

from notebook.models.ir import Integral, Number, Subscript, Symbol
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError
from notebook.runtime.sum_integral_prefix import _differential_var


def _resolve(canonical, ctx=None):
    return MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                            ctx or SymbolContext())


class TestTheDifferentialIsRead:

    @pytest.mark.parametrize("canonical", [
        r"\int*f*dx_1",
        r"\int_0^1*f*dx_1",
    ])
    def test_an_indexed_differential_is_not_a_factor(self, canonical):
        """読まれないと `Mul`（積）になる——それが 2026-09-17 の欠陥だった."""
        with pytest.raises(UnknownNotationError) as ei:
            _resolve(canonical)
        assert ei.value.i18n_key == "integral.var_is_indexed"

    @pytest.mark.parametrize("latex", [
        r"\int f dx_1", r"\int f dx_{1}", r"\int f \, dx_1", r"\int_0^1 f dx_1",
    ])
    def test_the_researchers_spellings_all_reach_it(self, latex):
        """**系が作る正準形で測る。** 波括弧つきは変換器が `dx_1` に均すので、
        `dx_{1}` という正準形は存在しない——直接与えると、誰も打たない綴りを
        測ることになる（最初に書いたテストがそれだった）。"""
        from notebook.runtime.backend_factory import make_backend
        _, _, h = make_backend("maxima")
        h._context = SymbolContext()
        canonical = h.latex_to_canonical(latex)
        assert "dx_1" in canonical, canonical
        with pytest.raises(UnknownNotationError) as ei:
            _resolve(canonical)
        assert ei.value.i18n_key == "integral.var_is_indexed"

    def test_the_variable_node_is_handed_over_unfolded(self):
        """名前に畳まず、書かれた部分木のまま返す（解決は呼び出し側の仕事）."""
        node = Subscript(base=Symbol(name="dx"), sub=Number(value=1))
        got = _differential_var(node)
        assert isinstance(got, Subscript), got
        assert got.base.name == "x"

    @pytest.mark.parametrize("canonical,var", [
        (r"\int*f*dx", "x"), (r"\int*x^2*dx", "x"), (r"\int*f(x)*dx", "x"),
    ])
    def test_the_plain_spelling_is_unchanged(self, canonical, var):
        ir = _resolve(canonical)
        assert isinstance(ir, Integral) and ir.var.name == var


class TestTheMeasureGoesThroughTheSameDoorAsTheIntegrand:

    def test_declaring_the_name_makes_them_match(self):
        """案内している回避策が**効く**こと（断りの2つ目の出口）."""
        ctx = SymbolContext()
        ctx.register_quantity("x_1", kind="scalar")
        ir = _resolve(r"\int*x_1*dx_1", ctx)
        assert isinstance(ir, Integral)
        assert isinstance(ir.var, Symbol) and ir.var.name == "x_1", ir.var
        # 被積分関数の側も同じ綴り——ここがずれると CAS は定数と見る
        assert getattr(ir.expr, "name", None) == "x_1", ir.expr

    def test_the_refusal_names_the_spelling_the_researcher_wrote(self):
        with pytest.raises(UnknownNotationError) as ei:
            _resolve(r"\int*x_1*dx_1")
        kw = ei.value.fmt_kwargs
        assert kw.get("var") == "x_1", kw
        assert kw.get("base") == "x", kw
        assert "(" in str(kw.get("lowered")), kw


class TestTheValueIsRightWhereItComputes:
    """CAS まで通る綴りは値が変わっていないこと（実 CAS は使わず IR で見る）."""

    @pytest.mark.parametrize("canonical", [
        r"\int*f*dx", r"\int*x*dx", r"\int*x^2*dx",
        r"\int_0^1*x*dx", r"\int*f*dx*dy",
    ])
    def test_it_is_still_an_integral(self, canonical):
        assert isinstance(_resolve(canonical), Integral)
