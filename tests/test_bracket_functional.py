"""R-9: display-only bracket functionals — E[X] (expectation) and declarable
F[X]/Var[X].

E[X] resolves to a functional (CAS: expectation(X)) automatically in
statistics-family domains (notation profile E_bracket), while the DISPLAY keeps
E[X].  Any other symbol becomes a bracket functional by declaration
("F を期待値汎函数とする" / "declare F as an expectation functional"); the symbol
is NOT registered as an operator, so it gains no operator algebra.  A[i,j]
(matrix element) and [A,B] (commutator) are unaffected.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _cas(text, ctx):
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(text), ctx)
    return MaximaCodeGenerator(ctx).generate(ir)


class TestResolution:
    def test_expectation_in_statistics_domain(self):
        ctx = SymbolContext(); ctx.domains = ["econometrics"]
        assert _cas("E[X]", ctx) == "expectation(X)"
        assert _cas("E[X,Y]", ctx) == "expectation(X, Y)"

    def test_conditional_expectation_subscript(self):
        ctx = SymbolContext(); ctx.domains = ["probability_statistics"]
        assert _cas("E_1[X]", ctx) == "expectation(X)"

    def test_declared_functional(self):
        ctx = SymbolContext()
        ctx.bracket_functionals = {"F": "expectation", "Var": "variance"}
        assert _cas("F[X]", ctx) == "expectation(X)"
        assert _cas("Var[X]", ctx) == "variance(X)"

    def test_commutator_unaffected(self):
        # [A,B] is still a commutator, never a functional (no head symbol)
        assert _cas("[A,B]", SymbolContext()) == "((A)*(B) - (B)*(A))"

    def test_no_domain_no_declaration_is_not_functional(self):
        # E[X] without a statistics domain / declaration stays the R-3 error path
        import pytest
        from notebook.runtime.shared_parser import UnknownNotationError
        from notebook.runtime.expr_generator import ExprGenerator
        ir = MaximaSemanticResolver().resolve(SharedParser().parse("E[X]"), SymbolContext())
        with pytest.raises(UnknownNotationError):
            ExprGenerator().generate(ir)


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, text, ttype="text", value=None):
    tok = (InputToken(type="math", value=value, canonical=value)
           if ttype == "math" else InputToken(type="text", value=text))
    return asyncio.run(run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestDeclarationAndDisplay:
    def _engine(self, lang="ja"):
        nb.start_engine(title="bf", mode="standard", language=lang)
        return nb.get_engine(), _sup()

    def test_declare_ja_and_display_preserved(self):
        eng, sup = self._engine("ja")
        try:
            r = _run(eng, sup, "F を期待値汎函数とする")
            assert r["status"] == "completed"
            assert eng.symbol_context.bracket_functionals == {"F": "expectation"}
            r2 = _run(eng, sup, None, "math", "F[X]")
            expr = eng.expressions[r2["registered_expressions"][-1]["expr_id"]]
            assert expr.latex == "F[X]"           # display keeps the brackets
            assert "F" not in eng.symbol_context.registry  # NOT an operator symbol
        finally:
            nb.stop_engine()

    def test_declare_en(self):
        eng, sup = self._engine("en")
        try:
            r = _run(eng, sup, "declare G as an expectation functional")
            assert r["status"] == "completed"
            assert eng.symbol_context.bracket_functionals.get("G") == "expectation"
        finally:
            nb.stop_engine()

    def test_declare_generic_functional_uses_symbol_name(self):
        # "汎関数" without 期待値 / "as a functional" without expectation → the
        # CAS name is the symbol itself (H[X] → H(X)), not expectation.
        eng, sup = self._engine("en")
        try:
            r = _run(eng, sup, "declare H as a functional")
            assert r["status"] == "completed"
            assert eng.symbol_context.bracket_functionals.get("H") == "H"
        finally:
            nb.stop_engine()

    def test_declare_ja_kanji_variant(self):
        # 汎関数 (modern 関) and 汎函数 (traditional 函) both accepted
        eng, sup = self._engine("ja")
        try:
            r = _run(eng, sup, "K を期待値汎関数とする")
            assert r["status"] == "completed"
            assert eng.symbol_context.bracket_functionals.get("K") == "expectation"
        finally:
            nb.stop_engine()

    def test_expectation_by_domain_no_declaration(self):
        nb.start_engine(title="bf2", mode="standard", language="en",
                        domains=["econometrics"])
        eng = nb.get_engine()
        try:
            r = _run(eng, _sup(), None, "math", "E[X]")
            expr = eng.expressions[r["registered_expressions"][-1]["expr_id"]]
            assert expr.latex == "E[X]"
            assert "E" not in eng.symbol_context.registry
        finally:
            nb.stop_engine()
