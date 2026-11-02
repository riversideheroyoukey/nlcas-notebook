r"""適用形の添字は、**個数によらず**同じ理由で断る（JULIA-APPLIED-INDEXED-COUNT）。

`docs/refusals.md` は `a_i(k)` と `D_{i,j}(x)` を同じ拒否と書き、生成器のコメントは
「**添字の個数は無関係**——単一添字も同じように失敗する」と書いていた。
2026-08-18 に refusals.md の全行を実測する監査で、**そうなっていなかった**:

    D_{i,j}(x)  → REFUSED (julia_applied_indexed)
    a_i(k)      → OK: a_i(k)          ← 拒否に到達しない

コメントは**拒否の分岐**については正しく、**振る舞い**については誤りだった——単一添字は
`subscript_policy` が名前に畳むので、その分岐まで来ない。

生の `a_i(k)` を Julia で実測（2026-08-18）:

    a(QuIndex("i"), QuIndex("k"))  →  a(ik)                        ← 動く
    a_i(1)                         →  Sym a_i is not callable      ← 生の CAS 文字列
    a(QuIndex("i"))(1)             →  MethodError: QuExpr not callable

QuantumAlgebra の README は「これらの関数は**任意個の添字を引数に取る**」と言うので、
既存の断り文の助言（「引数も添字として書く」）は正しく、**両方の個数がそこへ到達すべき**
だった。宣言済み QA 演算子を「添字を取る量」として扱うことで揃えた。
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError


def _jcas(latex: str, setup=None) -> str:
    ctx = SymbolContext()
    if setup:
        setup(ctx)
    canonical = JuliaBackendHandler(None).latex_to_canonical(latex)
    ir = JuliaSemanticResolver().resolve(SharedParser().parse(canonical), ctx)
    return JuliaCodeGenerator(context=ctx).generate(ir)


def _op(name):
    return lambda ctx: ctx.register_quantity(name, kind="operator")


class TestTheCountIsIrrelevant:
    @pytest.mark.parametrize("latex,setup", [
        (r"a_i(k)", _op("a")),               # 単一添字 — これが漏れていた
        (r"a_{i,\sigma}(k)", _op("a")),      # 多添字
        (r"D_{i,j}(x)", None),               # 宣言なしの多添字
    ])
    def test_refused_by_the_same_name(self, latex, setup):
        with pytest.raises(UnknownNotationError) as exc:
            _jcas(latex, setup)
        assert exc.value.args[0] == "parser.unsupported_notation.julia_applied_indexed"

    def test_the_reason_names_the_fix_the_manual_supports(self):
        """助言「引数も添字として書く」は QA の README どおり——
        実測で `a(QuIndex("i"), QuIndex("k"))` は `a(ik)` になる。"""
        from notebook.runtime.i18n import t
        for lang in ("ja", "en"):
            msg = t("parser.unsupported_notation.julia_applied_indexed", lang,
                    name="a_i(k)", base="a", index="i")
            assert msg != "parser.unsupported_notation.julia_applied_indexed"
            assert "a" in msg


class TestWhatMustStayFolded:
    r"""畳む既定は、宣言の無い基底のために残っている（CLAUDE.md）。"""

    def test_an_undeclared_base_still_folds(self):
        """`p_x(x)` は確率分布の周辺化で、添字は**名前の一部**。
        添字として読むと `p_x(x)` と `p_y(y)` が同じ `p[1](1)` に潰れる。"""
        assert _jcas(r"p_x(x)") == "p_x(x)"

    def test_a_declared_SCALAR_base_still_folds(self):
        """演算子だけが「添字を取る量」——スカラー宣言は答えにならない。"""
        assert _jcas(r"p_x(x)", lambda c: c.register_quantity("p", kind="scalar")) \
            == "p_x(x)"


class TestMaximaIsUnaffected:
    """判定はバックエンドの知識として Julia 側に置いた。Maxima の読みは不変。"""

    @staticmethod
    def _mcas(latex, setup=None):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        ctx = SymbolContext()
        if setup:
            setup(ctx)
        canonical = MaximaBackendHandler(None).latex_to_canonical(latex)
        ir = MaximaSemanticResolver().resolve(SharedParser().parse(canonical), ctx)
        return MaximaCodeGenerator(context=ctx).generate(ir)

    @pytest.mark.parametrize("latex,setup,expect", [
        (r"a_i(k)", _op("a"), "a_i(k)"),     # Maxima では従来どおり名前に畳む
        (r"p_x(x)", None, "p_x(x)"),
        (r"D_{i,j}(x)", None, "D[i, j](x)"),
    ])
    def test_unchanged(self, latex, setup, expect):
        assert self._mcas(latex, setup) == expect
