"""R-3 (SR-1/SR-2): unsupported notation must fail loudly, never silently.

Before 2026-07-03:
  - the tokenizer silently SKIPPED unknown characters: |A| parsed as A,
    dropping the absolute value / determinant / norm without any warning;
  - ExprGenerator fell back to str(node) for unsupported IR nodes, leaking
    the pydantic repr ("simplification_log=[] a=Symbol(...)") into canonicals
    and CAS commands (A[i,j] / E[X] inputs).

Both now raise UnknownNotationError, translated to a user-readable
ValueError at the engine boundary (same pattern as PrimeDerivativeError).
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError
from notebook.runtime.expr_generator import ExprGenerator
from notebook.runtime.i18n import t


class TestParserPipeRejection:
    def test_abs_bars_raise(self):
        with pytest.raises(UnknownNotationError):
            SharedParser().parse("|A|")

    def test_abs_bars_in_larger_expression_raise(self):
        with pytest.raises(UnknownNotationError):
            SharedParser().parse("|x+y| * 2")

    def test_abs_function_form_still_works(self):
        from notebook.models.ir import FunctionCall
        ast = SharedParser().parse("abs(A)")
        assert isinstance(ast, FunctionCall)

    def test_backslash_tex_commands_still_parse(self):
        """The tokenizer legitimately ignores backslashes from TeX commands —
        the sum/delta machinery depends on it and must not be affected."""
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        ast = SharedParser().parse(r"\sum_j*delta_ij*f_j")
        res = MaximaSemanticResolver().resolve(ast, SymbolContext())
        # Σ_j δ_ij f_j statically contracts to f(i)
        assert MaximaCodeGenerator(SymbolContext()).generate(res) == "f(i)"

    def test_error_carries_i18n_key(self):
        with pytest.raises(UnknownNotationError) as ei:
            SharedParser().parse("|A|")
        assert ei.value.i18n_key == "parser.unsupported_notation.pipe"
        for lang in ("ja", "en"):
            msg = t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)
            assert "abs" in msg  # points the user at the supported spelling


class TestGeneratorNoReprLeak:
    def _resolved(self, text, backend="maxima"):
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        ast = SharedParser().parse(text)
        return MaximaSemanticResolver().resolve(ast, SymbolContext())

    def test_bracket_after_symbol_raises_not_leaks(self):
        """A[i,j] resolves to Mul(A, Commutator) — the base ExprGenerator has
        no rendering for it and must raise, not emit the pydantic repr."""
        res = self._resolved("A[i,j]")
        with pytest.raises(UnknownNotationError):
            ExprGenerator().generate(res)

    def test_expectation_bracket_raises_not_leaks(self):
        res = self._resolved("E[X]")
        with pytest.raises(UnknownNotationError):
            ExprGenerator().generate(res)

    def test_error_message_translates_with_node_name(self):
        res = self._resolved("E[X]")
        with pytest.raises(UnknownNotationError) as ei:
            ExprGenerator().generate(res)
        for lang in ("ja", "en"):
            msg = t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)
            assert "simplification_log" not in msg

    def test_generated_output_never_contains_repr(self):
        """Property check on supported nodes: no pydantic repr fragments."""
        for text in ("a x + b", "f(x)", "x^2", "[A,B]", "sin(x) * cos(y)"):
            from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
            out = MaximaCodeGenerator(SymbolContext()).generate(self._resolved(text))
            assert "simplification_log" not in out
            assert "Symbol(" not in out


class TestEngineBoundaryTranslation:
    def test_register_pipe_expression_fails_with_readable_error_ja(self):
        from notebook.engine import NotebookEngine
        e = NotebookEngine(title="pipe ja")
        with pytest.raises(ValueError) as ei:
            e.register_expression("|A|")
        assert "abs" in str(ei.value)
        assert "parser.unsupported_notation" not in str(ei.value)  # translated

    def test_register_pipe_expression_fails_with_readable_error_en(self):
        from notebook.engine import NotebookEngine
        e = NotebookEngine(title="pipe en")
        e.session.language = ["en", "ja"]
        with pytest.raises(ValueError) as ei:
            e.register_expression("|A|")
        assert "not supported" in str(ei.value)

    def test_no_corrupted_expression_is_stored(self):
        from notebook.engine import NotebookEngine
        e = NotebookEngine(title="pipe store")
        with pytest.raises(ValueError):
            e.register_expression("|A|")
        assert all("|" not in ex.canonical for ex in e.expressions.values())
