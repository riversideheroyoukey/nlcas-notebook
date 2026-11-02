import pytest
import re
from notebook.models.symbol_context import SymbolContext
from notebook.models.ir import (
    Symbol, IndexSymbol, IndexedSymbol, KroneckerDelta, IndexSum, Mul, NonCommutativeMul,
    Add, Number, FunctionCall, Derivative, PrimeExpr, Pow, BracketExpr
)
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.models.quantity import Quantity

# -------------------------------------------------------------
# A2: Display Bridge Test (Kronecker Delta, x(i) -> x_i)
# -------------------------------------------------------------
def test_display_bridge_restoration():
    from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    
    ctx = SymbolContext()
    ctx.register_quantity("x", kind="scalar")
    ctx.register_quantity("A", kind="scalar")
    
    # 1. Kronecker Delta restoration test
    maxima_handler = MaximaBackendHandler(None)
    julia_handler = JuliaBackendHandler(None)
    
    raw_maxima_latex = "kron_delta(i, j)"
    restored_maxima = maxima_handler.postprocess_result_latex("kron_delta(i, j)", raw_maxima_latex, ctx)
    assert "\\delta_{i, j}" in restored_maxima
    
    raw_julia_latex = "kron_delta(i, j)"
    restored_julia = julia_handler.postprocess_result_latex("kron_delta(i, j)", raw_julia_latex, ctx)
    assert "\\delta_{i, j}" in restored_julia

    # 2. x(i) -> x_i and A[i,j] -> A_{i,j} restoration test
    raw_sum_latex = r"\mathrm{sum}\left( a(i), i \right)"
    restored_sum = maxima_handler.postprocess_result_latex("sum(a(i), i)", raw_sum_latex, ctx)
    assert "\\sum_{i} a(i)" in restored_sum
    
    # 3. Matrix operations restoration test
    raw_adjoint = r"\mathrm{ctranspose}\left( M \right)"
    assert "{M}^{\\dagger}" in maxima_handler.postprocess_result_latex("ctranspose(M)", raw_adjoint, ctx).replace(" ", "")
    
    raw_conj = r"\mathrm{conjugate}\left( M \right)"
    assert "{M}^{\\ast}" in maxima_handler.postprocess_result_latex("conjugate(M)", raw_conj, ctx).replace(" ", "")
    
    raw_trans = r"\mathrm{transpose}\left( M \right)"
    assert "{M}^{T}" in maxima_handler.postprocess_result_latex("transpose(M)", raw_trans, ctx).replace(" ", "")
    
    # Julia matrix operations restoration test
    raw_jl_adjoint = r"\mathrm{adjoint}\left( M \right)"
    assert "{M}^{\\dagger}" in julia_handler.postprocess_result_latex("adjoint(M)", raw_jl_adjoint, ctx).replace(" ", "")

    maxima_latex_x = "x\\left(i\\right)"
    restored_x = maxima_handler.postprocess_result_latex("x(i)", maxima_latex_x, ctx)
    assert "x_{i}" in restored_x

    julia_latex_A = "A\\left[i, j\\right]"
    restored_A = julia_handler.postprocess_result_latex("A[i, j]", julia_latex_A, ctx)
    assert "A_{i,j}" in restored_A

# -------------------------------------------------------------
# A3: Non-commutative lock (A B -> NonCommutativeMul)
# -------------------------------------------------------------
def test_non_commutative_mul_lock():
    parser = SharedParser()
    ctx = SymbolContext()
    
    # Register A and B as noncommutative
    ctx.register("A", quantity=Quantity(name="A", kind="operator"), statistics_kind="noncommutative")
    ctx.register("B", quantity=Quantity(name="B", kind="operator"), statistics_kind="noncommutative")
    
    resolver = MaximaSemanticResolver()
    
    # Implicit multiplication A B
    ast = parser.parse("A B")
    resolved = resolver.resolve(ast, ctx)
    
    # Should resolve to NonCommutativeMul, not standard Mul
    assert isinstance(resolved, NonCommutativeMul)
    assert len(resolved.args) == 2
    assert resolved.args[0].name == "A"
    assert resolved.args[1].name == "B"
    
    # Normal variables should still be commutative Mul
    ctx.register_quantity("x", kind="scalar")
    ctx.register_quantity("y", kind="scalar")
    ast_comm = parser.parse("x y")
    resolved_comm = resolver.resolve(ast_comm, ctx)
    assert not isinstance(resolved_comm, NonCommutativeMul)

# -------------------------------------------------------------
# A4: Prime dynamic resolution (calculus f'(x) -> Derivative, quantum a'(i) -> a')
# -------------------------------------------------------------
def test_prime_dynamic_resolution():
    parser = SharedParser()
    
    # 1. Calculus Domain (default or calculus domains)
    ctx_calc = SymbolContext()
    ctx_calc.domains = ["calculus"]
    ctx_calc.session_mode = "standard_symbolic"
    resolver_calc = MaximaSemanticResolver()
    
    ast_calc = parser.parse("f'(x)")
    resolved_calc = resolver_calc.resolve(ast_calc, ctx_calc)
    
    # Should resolve to Derivative(f(x), [(x, 1)])
    assert isinstance(resolved_calc, Derivative)
    assert resolved_calc.expr.func.name == "f"
    assert resolved_calc.vars[0][0].name == "x"
    assert resolved_calc.vars[0][1] == 1

    # 2. Quantum / Second Quantization Domain
    ctx_quantum = SymbolContext()
    ctx_quantum.domains = ["second_quantization"]
    ctx_quantum.session_mode = "second_quantization"
    ctx_quantum.register("a", quantity=Quantity(name="a", kind="operator"), statistics_kind="fermi")
    
    resolver_quantum = MaximaSemanticResolver()
    ast_quantum = parser.parse("a'(i)")
    resolved_quantum = resolver_quantum.resolve(ast_quantum, ctx_quantum)
    
    # Should resolve to FunctionCall(Symbol("a'"), [IndexSymbol("i")])
    assert isinstance(resolved_quantum, FunctionCall)
    assert resolved_quantum.func.name == "a'"
    assert resolved_quantum.args[0].name == "i"

# -------------------------------------------------------------
# A5: Imaginary i vs Index i Collision Avoidance
# -------------------------------------------------------------
def test_imaginary_vs_index_i():
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()
    
    # 1. Subscript i: x_{i} -> IndexSymbol("i")
    ast_sub = parser.parse("x_{i}")
    resolved_sub = resolver.resolve(ast_sub, ctx)
    assert isinstance(resolved_sub, IndexedSymbol)
    assert isinstance(resolved_sub.indices[0], IndexSymbol)
    assert resolved_sub.indices[0].name == "i"
    
    # 2. Standalone explicit \mathrm{i} -> imaginary unit (%i in Maxima)
    ast_explicit = parser.parse("\\mathrm{i}")
    resolved_explicit = resolver.resolve(ast_explicit, ctx)
    assert isinstance(resolved_explicit, Symbol)
    assert resolved_explicit.name == "%i"
    
    # 3. Inside exponent e^{i x} -> imaginary unit
    ast_exp = parser.parse("e^{i x}")
    resolved_exp = resolver.resolve(ast_exp, ctx)
    assert isinstance(resolved_exp, Pow)
    exponent = resolved_exp.exponent
    assert isinstance(exponent, BracketExpr)
    inner_mul = exponent.args[0]
    assert isinstance(inner_mul, Mul)
    assert any(isinstance(arg, Symbol) and arg.name == "%i" for arg in inner_mul.args)
    
    # 4. Registered "i" (e.g. current) should remain "i"
    ctx_curr = SymbolContext()
    ctx_curr.register_quantity("i", kind="scalar")
    ast_curr = parser.parse("e^{i x}")
    resolved_curr = resolver.resolve(ast_curr, ctx_curr)
    exponent_curr = resolved_curr.exponent
    assert isinstance(exponent_curr, BracketExpr)
    inner_curr_mul = exponent_curr.args[0]
    assert isinstance(inner_curr_mul, Mul)
    assert any(isinstance(arg, Symbol) and arg.name == "i" for arg in inner_curr_mul.args)

# -------------------------------------------------------------
# A1: Save/Load Type Preservation
# -------------------------------------------------------------
def test_save_load_type_preservation(tmp_path):
    from notebook.engine import NotebookEngine
    
    engine = NotebookEngine("maxima")
    engine.session.domains = ["second_quantization"]
    engine.symbol_context.domains = ["second_quantization"]
    engine.symbol_context.session_mode = "second_quantization"
    engine.symbol_context.register("a", quantity=Quantity(name="a", kind="operator"), statistics_kind="fermi")
    
    expr = engine.register_expression("a'(i)")
    
    session_file = tmp_path / "test_session.json"
    engine.save(session_file)
    
    loaded_engine = NotebookEngine.load(session_file)
    
    assert loaded_engine.session.domains == ["second_quantization"]
    assert loaded_engine.symbol_context.domains == ["second_quantization"]
    
    loaded_expr = loaded_engine.expressions[expr.display_id]
    
    resolved_ast = loaded_engine._handler.parse_ir(loaded_expr.canonical)
    
    assert isinstance(resolved_ast, FunctionCall)
    assert resolved_ast.func.name == "a'"

def test_matrix_operations_disambiguation():
    """B4: Test Adjoint, Conjugate, and Transpose disambiguation based on MathType."""
    from notebook.models.ir import Pow, Adjoint, Conjugate, Transpose
    from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
    from notebook.runtime.shared_parser import SharedParser
    from notebook.models.symbol_context import SymbolContext
    from notebook.models.quantity import Quantity, MathType

    ctx = SymbolContext()
    ctx.register_quantity("M", kind="anonymous", math_type=MathType(algebra="matrix"))
    ctx.register_quantity("O", kind="operator", math_type=MathType(algebra="operator"))
    ctx.register_quantity("s", kind="scalar", math_type=MathType(algebra="scalar"))

    parser = SharedParser()
    resolver = MaximaSemanticResolver()

    # Transpose: M^T vs s^T
    ast_m_t = parser.parse("M^T")
    res_m_t = resolver.resolve(ast_m_t, ctx)
    assert isinstance(res_m_t, Transpose)
    assert res_m_t.expr.name == "M"

    ast_s_t = parser.parse("s^T")
    res_s_t = resolver.resolve(ast_s_t, ctx)
    assert isinstance(res_s_t, Pow) # Scalar defaults to Pow
    assert res_s_t.exponent.name == "T"

    # Conjugate: s^* and M^*
    ast_s_star = parser.parse("s^*") # Parser replaces ^* with ^{\ast}
    res_s_star = resolver.resolve(ast_s_star, ctx)
    assert isinstance(res_s_star, Conjugate)
    
    # Adjoint: O^\dagger
    ast_o_dag = parser.parse("O^\\dagger")
    res_o_dag = resolver.resolve(ast_o_dag, ctx)
    assert isinstance(res_o_dag, Adjoint)



def test_greek_parameter_names_converted_to_latex():
    r"""Regression: QuantumAlgebra outputs 'Delta' and 'xi' without backslashes.

    postprocess_result_latex must convert registered scalar parameters whose
    names are Greek letters to proper LaTeX commands (\Delta, \xi, ...) so that
    KaTeX can rasterize them in the notebook UI.

    Bug: [H, c_k] where H contains Delta and xi_k parameters showed
    'Deltac†_K' and 'xi_j c_k' as plain text instead of rendered math.
    """
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext
    from notebook.models.quantity import Quantity

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()
    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
    ctx.register("xi",    Quantity(name="xi",    kind="scalar"))

    # QuantumAlgebra-style output: parameter names without backslashes
    raw = (r"Delta c^{\dagger}_{K} - Delta \delta_{Kk} c^{\dagger}_{K}"
           r" - xi_{j} c_{k} - xi_{k} c_{k}")
    out = handler.postprocess_result_latex("val", raw, ctx)

    assert r"\Delta" in out, f"Delta must become \\Delta; got: {out}"
    assert r"\xi"   in out, f"xi must become \\xi; got: {out}"
    # Original occurrences without backslash must be gone
    import re
    assert not re.search(r'(?<![A-Za-z\\])Delta(?![A-Za-z])', out), \
        f"Bare 'Delta' must not remain; got: {out}"
    assert not re.search(r'(?<![A-Za-z\\])xi(?![A-Za-z])', out), \
        f"Bare 'xi' must not remain; got: {out}"


def test_greek_conversion_does_not_double_escape():
    """Already-escaped \\Delta must not become \\\\Delta."""
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext
    from notebook.models.quantity import Quantity

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()
    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

    already_escaped = r"\Delta c^{\dagger}_{k} + \Delta^{2}"
    out = handler.postprocess_result_latex("val", already_escaped, ctx)
    # Must be unchanged — no double-escaping
    assert out == already_escaped, f"Double-escape occurred; got: {out}"


def test_greek_conversion_ignores_compound_identifiers():
    """DeltaE and xi_star must not be modified (compound identifiers)."""
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext
    from notebook.models.quantity import Quantity

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()
    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
    ctx.register("xi",    Quantity(name="xi",    kind="scalar"))

    raw = r"DeltaE + xistar + OmegaDelta"
    out = handler.postprocess_result_latex("val", raw, ctx)
    assert out == raw, f"Compound identifiers must not be modified; got: {out}"


def test_unregistered_greek_name_not_converted():
    """Omega that is NOT registered in context must not be converted."""
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()  # empty context — Omega not registered

    raw = r"Omega_{n} c_{k}"
    out = handler.postprocess_result_latex("val", raw, ctx)
    assert "Omega" in out, f"Unregistered 'Omega' must not be converted; got: {out}"


# -------------------------------------------------------------
# QuantumAlgebra uppercase index → negated lowercase (step 4)
# -------------------------------------------------------------

def test_uppercase_index_converted_to_negated_lowercase():
    r"""K in subscript → -k when lowercase k is also present.

    QuantumAlgebra uses uppercase K to represent the negated momentum -k.
    E.g. c^{\dagger}_{K} in the output of [H_BCS, c_k] should become
    c^{\dagger}_{-k} so that the result reads Δc†_{-k} instead of Δc†_{K}.

    Also: Kronecker delta δ_{Kk} → δ_{-k,k} (two-letter subscript case).
    """
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()

    # K + k both present → K becomes -k (single-letter case)
    raw = r"\Delta c^{\dagger}_{K} - \xi_{k} c_{k}"
    out = handler.postprocess_result_latex("val", raw, ctx)
    assert r"_{-k}" in out, f"K must become -k; got: {out}"
    assert r"_{K}" not in out, f"No bare _{{K}} should remain; got: {out}"

    # Two-letter subscript in Kronecker delta: δ_{Kk} → δ_{-k,k}
    raw2 = r"\Delta\delta_{Kk}c^{\dagger}_{-k} - \xi_{k} c_{k}"
    out2 = handler.postprocess_result_latex("val", raw2, ctx)
    assert r"_{-k,k}" in out2, f"delta_{{Kk}} must become delta_{{-k,k}}; got: {out2}"
    assert r"_{Kk}" not in out2, f"No bare _{{Kk}} should remain; got: {out2}"


def test_uppercase_index_not_converted_without_lowercase():
    r"""K in subscript is left alone when lowercase k is absent.

    If k does not appear as a subscript in the same expression, K is treated
    as a standalone label (e.g. a K-point in the Brillouin zone) and must
    not be rewritten.
    """
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.models.symbol_context import SymbolContext

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = SymbolContext()

    raw = r"E_{K} + V_{K}"   # no lowercase k anywhere
    out = handler.postprocess_result_latex("val", raw, ctx)
    assert r"_{K}" in out, f"Standalone K must be preserved; got: {out}"
    assert r"_{-k}" not in out, f"No -k conversion should happen; got: {out}"


def test_define_scalar_parameter_latex_postprocessed():
    r"""define_scalar_parameter result is postprocessed so xi → \xi.

    The engine calls postprocess_result_latex even for side-effect (void)
    operations so that the UI receives \xi instead of plain xi for the
    display of parameter-definition CAS blocks.
    """
    from notebook.engine import NotebookEngine
    from notebook.models.quantity import Quantity

    engine = NotebookEngine("julia")
    engine.symbol_context.session_mode = "second_quantization"
    engine.symbol_context.domains = ["condensed_matter"]

    # Pre-register xi as scalar (simulating LLM having parsed the request)
    engine.symbol_context.register("xi", Quantity(name="xi", kind="scalar"))
    engine.symbol_context.register("Delta", Quantity(name="Delta", kind="scalar"))

    # Directly test the postprocess path that the engine now uses for void ops
    from unittest.mock import MagicMock
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler

    handler = JuliaBackendHandler(builder=MagicMock())
    ctx = engine.symbol_context

    assert r"\xi" in handler.postprocess_result_latex("xi", "xi", ctx), \
        "define_scalar_parameter('xi') display must render as \\xi"
    assert r"\Delta" in handler.postprocess_result_latex("Delta", "Delta", ctx), \
        "define_scalar_parameter('Delta') display must render as \\Delta"


# ---------------------------------------------------------------------------
# SymPy operator LaTeX normalization (step 1.9 in postprocess_result_latex)
# ---------------------------------------------------------------------------

class TestSymPyOperatorLatexNormalization:
    """postprocess_result_latex must convert SymPy function-call LaTeX
    (\\operatorname{cd}{\\left(k\\right)}, c{\\left(k\\right)}) to subscript
    notation (c^{\\dagger}_{k}, c_{k}) for registered quantum operators."""

    def _handler_and_ctx(self):
        from unittest.mock import MagicMock
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity

        handler = JuliaBackendHandler(builder=MagicMock())
        ctx = SymbolContext()
        ctx.session_mode = "second_quantization"
        ctx.domains = ["condensed_matter"]
        ctx.register("c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        ctx.register("c'", Quantity(name="c'", kind="operator"), statistics_kind="fermi")
        return handler, ctx

    def test_creation_operator_converted(self):
        """\\operatorname{cd}{\\left(-k\\right)} → c^{\\dagger}_{-k}"""
        handler, ctx = self._handler_and_ctx()
        raw = r"\operatorname{cd}{\left(- k \right)}"
        out = handler.postprocess_result_latex("val", raw, ctx)
        assert r"c^{\dagger}_{-k}" in out, f"Expected c^{{\\dagger}}_{{-k}}; got: {out}"
        assert r"\operatorname{cd}" not in out, f"operatorname should be removed; got: {out}"

    def test_annihilation_operator_converted(self):
        """c{{\\left(k\\right)}} → c_{k}"""
        handler, ctx = self._handler_and_ctx()
        raw = r"c{\left(k \right)}"
        out = handler.postprocess_result_latex("val", raw, ctx)
        assert r"c_{k}" in out, f"Expected c_{{k}}; got: {out}"
        assert r"\left(" not in out, f"\\left( should be removed; got: {out}"

    def test_bcs_commutator_result_display(self):
        """Full BCS result: \\Delta c^{\\dagger}_{-k} - \\xi_{k} c_{k}"""
        handler, ctx = self._handler_and_ctx()
        from notebook.models.quantity import Quantity
        ctx.register("xi", Quantity(name="xi", kind="scalar"))
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

        # Typical SymPy LaTeX output (SymPy 1.13+ with nonzero-k simplification)
        raw = r"\Delta \operatorname{cd}{\left(- k \right)} - \xi_{k} c{\left(k \right)}"
        out = handler.postprocess_result_latex("val", raw, ctx)
        assert r"c^{\dagger}_{-k}" in out, f"Creation op not converted; got: {out}"
        assert r"c_{k}" in out, f"Annihilation op not converted; got: {out}"
        assert r"\operatorname" not in out, f"operatorname artifact; got: {out}"
        assert r"\Delta" in out, f"Delta not converted; got: {out}"

    def test_double_negative_cleaned(self):
        """'- -1*\\Delta c^{...}' after sign expansion → '\\Delta c^{...}'"""
        handler, ctx = self._handler_and_ctx()
        from notebook.models.quantity import Quantity
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

        raw = r"- -1*Delta \operatorname{cd}{\left(- k \right)} - \xi_{k} c{\left(k \right)}"
        out = handler.postprocess_result_latex("val", raw, ctx)
        assert "- -1" not in out, f"Double-negative not removed; got: {out}"
        assert r"c^{\dagger}_{-k}" in out, f"Creation op not converted; got: {out}"

    def test_boson_operator_converted(self):
        """\\operatorname{ad}{\\left(k\\right)} → a^{\\dagger}_{k}"""
        from unittest.mock import MagicMock
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity

        handler = JuliaBackendHandler(builder=MagicMock())
        ctx = SymbolContext()
        ctx.session_mode = "second_quantization"
        ctx.domains = ["condensed_matter"]
        ctx.register("a", Quantity(name="a", kind="operator"), statistics_kind="bose")
        ctx.register("a'", Quantity(name="a'", kind="operator"), statistics_kind="bose")

        raw = r"\operatorname{ad}{\left(k \right)} - a{\left(- k \right)}"
        out = handler.postprocess_result_latex("val", raw, ctx)
        assert r"a^{\dagger}_{k}" in out, f"Boson creation op not converted; got: {out}"
        assert r"a_{-k}" in out, f"Boson annihilation op not converted; got: {out}"
