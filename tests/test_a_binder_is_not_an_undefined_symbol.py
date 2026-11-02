r"""REFUSAL-NOT-REACHED ①（2026-09-17）。

**設計した断りの手前で、別の断りが出ていた。**

Julia セッションで `\lim_{x \to 0} \frac{\sin x}{x}` を打って使うと:

    registered（正準形 `\lim_{x,to,0} ((sin(x))/(x))`——ここまでは Maxima と同一）
    E1 を簡約して → **「未定義の記号: lim, to」**

`lim` も `to` も**研究者は書いていない**。極限の正準綴り（`limit_subscript_latex`）を
ばらした断片である。そして生成器の `julia_limit`——このコードベースが
「**書けて表示はできる、断るのは使用地点**」という方針で書いた断り——は、この guard の
**後ろ**にあるので一度も到達していなかった。

**原因は表現の取り違え。** `_pre_cas_undefined_guard` は `Expression.ir_ast` を読むが、
Julia の `parse_ir` は `None` を返す（正準形はふつう `_nb_E3` というハンドルなので）。
そこで engine は canonical を `SharedParser` で**再解析**していた——**意味解決を通さずに**。
束縛子は解決前には記号と見分けが付かない:

    surface: Subscript(base=Symbol('lim'), sub=FunctionCall([Symbol('x'), Symbol('to'), 0]))
    IR:      Limit(var=x, point=0, expr=…)

**同じ問い（この式の未定義記号は何か）を、他の全員が読むのと違う表現に対して
答えていた**（docs/developer_manuals/35 の型）。

**直しは「解決器に訊く」。** 実測すると**両方のバックエンドの解決器**がこの正準形を
`Limit` に解決できる——能力は先に在って、guard がそれを使っていなかっただけである。
`lim`/`to` を `_KNOWN_SYMBOL_NAMES` に足す案は採らない: 内部の綴りを
「研究者が使ってよい名前の表」に入れることになり、`to` を変数として書けてしまう。

**どの AST を読むかはバックエンドの知識**なので、共有層はフック
（`BackendHandler.parse_for_undefined_scan`）に委譲する（Backend Encapsulation Rule）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.ir import Limit
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_CANONICAL = r"\lim_{x,to,0} ((sin(x))/(x))"


class TestTheResolverAlreadyKnew:
    """能力は先に在った——guard がそれを使っていなかった."""

    @pytest.mark.parametrize("mod,cls", [
        ("notebook.runtime.maxima_semantic_resolver", "MaximaSemanticResolver"),
        ("notebook.runtime.julia_semantic_resolver", "JuliaSemanticResolver"),
    ])
    def test_both_resolvers_read_the_canonical_as_a_limit(self, mod, cls):
        from notebook.runtime.shared_parser import SharedParser
        R = getattr(__import__(mod, fromlist=[cls]), cls)
        ir = R().resolve(SharedParser().parse(_CANONICAL), SymbolContext())
        assert isinstance(ir, Limit), f"{cls} → {type(ir).__name__}"

    def test_the_surface_ast_looks_like_symbols(self):
        """解決前は `lim` と `to` が記号に見える——これが誤報の出どころ."""
        from notebook.models.ir import Subscript, Symbol
        from notebook.runtime.shared_parser import SharedParser
        surf = SharedParser().parse(_CANONICAL)
        assert isinstance(surf, Subscript)
        assert isinstance(surf.base, Symbol) and surf.base.name == "lim"


class TestTheHookIsWhereTheBackendAnswers:

    def test_the_default_parses_only(self):
        """共有層の既定は、engine が直書きしていたものと同じ（surface）."""
        from notebook.models.ir import Subscript
        from notebook.runtime.backend_factory import make_backend
        _, _, h = make_backend("maxima")
        assert isinstance(h.parse_for_undefined_scan(_CANONICAL), Subscript)

    def test_julia_resolves(self):
        from notebook.runtime.backend_factory import make_backend
        _, _, h = make_backend("julia_full")
        h._context = SymbolContext()
        assert isinstance(h.parse_for_undefined_scan(_CANONICAL), Limit)

    def test_an_unreadable_canonical_falls_back_rather_than_raising(self):
        """読めない綴りで例外を上げると、断りが解決器の内部エラーに化ける."""
        from notebook.runtime.backend_factory import make_backend
        _, _, h = make_backend("julia_full")
        h._context = SymbolContext()
        h.parse_for_undefined_scan("a'(k) a(k)")   # 例外を上げないこと


# ---------------------------------------------------------------------------
# 生きた経路（日英）
# ---------------------------------------------------------------------------
def _ask(lang, domain, lines):
    async def go():
        nb.start_engine(title="limit-guard", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            last = None
            for text in lines:
                last = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
            return last
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_LIM = r"\lim_{x \to 0} \frac{\sin x}{x}"


class TestTheRefusalTheCodebaseWroteIsTheOneThatArrives:

    @pytest.mark.parametrize("lang,use", [("ja", "E1 を簡約して"),
                                          ("en", "simplify E1")])
    def test_julia_says_why_a_limit_cannot_be_computed(self, lang, use):
        r = _ask(lang, "condensed_matter", [_LIM, use])
        assert r.get("status") != "completed", r
        err = str(r.get("error") or "")
        assert "lim, to" not in err, f"研究者が書いていない綴りを名指している: {err!r}"
        assert ("Symbolics" in err or "QuantumAlgebra" in err), err

    @pytest.mark.parametrize("lang,use", [("ja", "E1 を簡約して"),
                                          ("en", "simplify E1")])
    def test_maxima_still_computes_it(self, lang, use):
        """対照: 極限を持つバックエンドでは値が出るまま."""
        r = _ask(lang, "classical_mechanics", [_LIM, use])
        assert r.get("status") == "completed", r


class TestTheGuardStillBlocksWhatItIsFor:
    """校正: 未宣言の物理パラメータは今も止まる（guard を壊していない）."""

    @pytest.mark.parametrize("lang,use", [("ja", "E1 を正規順序化して"),
                                          ("en", "normal order E1")])
    def test_an_undeclared_parameter_is_still_named(self, lang, use):
        r = _ask(lang, "condensed_matter", ["xi c'(1) c(1)", use])
        assert r.get("status") != "completed", r
        assert "xi" in str(r.get("error") or ""), r
