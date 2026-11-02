import pytest
from notebook.models.session import SessionMode
from notebook.models.ir import (
    KroneckerDelta, DiracDelta, IndexSum, Integral, Symbol, Number, FunctionCall
)
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.models.symbol_context import SymbolContext
from notebook.engine import NotebookEngine

def test_ast_compilation_pipeline():
    """Verify that continuous/discrete delta and sum expressions compile to the correct Julia commands."""
    parser = SharedParser()
    resolver = JuliaSemanticResolver()
    generator = JuliaCodeGenerator()
    ctx = SymbolContext()

    # 1. delta(i, j) -> KroneckerDelta -> QuantumAlgebra.δ(i, j)
    ast = parser.parse("delta(i, j)")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, KroneckerDelta)
    cmd = generator.generate(resolved)
    assert "_QA.δ" in cmd

    # 2. DiracDelta(x, y) -> DiracDelta -> SymPy.sympy.DiracDelta(x - y)
    ast = parser.parse("DiracDelta(x, y)")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, DiracDelta)
    cmd = generator.generate(resolved)
    assert "SymPy.sympy.DiracDelta" in cmd

    # 3. IndexSum(f(j), j) -> IndexSum -> QuantumAlgebra's own sum
    ast = parser.parse("IndexSum(f(j), j)")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, IndexSum)
    cmd = generator.generate(resolved)
    # 2026-08-07: was `"_nb_eval_index_sum" in cmd`, which pinned the string-eval
    # MECHANISM.  QuantumAlgebra has a sum of its own (`∑`), and using it is what
    # makes the RESULT a sum — the old helper only substituted a sumindex for the
    # dummy, so `normal_form` lost the Σ and the latex had none.
    assert "_QA.∑(:j," in cmd

    # 4. integrate(DiracDelta(x, y) * f(y), y, -Inf, Inf) -> Integral -> _nb_eval_continuous_integral
    ast = parser.parse("integrate(DiracDelta(x, y) * f(y), y, -Inf, Inf)")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, Integral)
    cmd = generator.generate(resolved)
    assert "_nb_eval_continuous_integral" in cmd
    assert "DiracDelta" in cmd

def test_julia_live_integration():
    """Verify that live Julia execution successfully evaluates and contracts deltas/sums/integrals."""
    with NotebookEngine(title="delta_integration_test", mode=SessionMode.second_quantization) as e:
        e.start()
        
        # Define boson operator
        e.run_operation("define_boson_operator", {"name": "a"})
        
        # A1. Commutator: [a_i, a'_j] = delta_ij
        expr_comm = e.register_expression("[a_i, a'_j]")
        step, _ = e.run_operation("normal_order", {"expr": expr_comm.display_id})
        print("RENDERED COMMAND:", step.rendered_command)
        res = e.expressions[step.output_refs[0]]
        assert "delta" in res.cas_repr.lower() or "δ" in res.cas_repr
        assert "delta" in res.latex.lower() or "δ" in res.latex
        
        # A2. Numeric evaluations
        expr_num1 = e.register_expression("[a(1), a'(1)]")
        step_num1, _ = e.run_operation("normal_order", {"expr": expr_num1.display_id})
        res_num1 = e.expressions[step_num1.output_refs[0]]
        assert res_num1.cas_repr.strip() == "1"
        
        # C1. Index sum contraction: Σ_j δ_ij f_j via SymPy / QuantumAlgebra helper
        adapter = e._adapter
        
        # Pre-register f(j) parameter
        adapter.execute("param(:f, :j)")
        
        # Evaluate index sum: Σ_j δ_ij f_j
        ans_sum = adapter.execute('_nb_eval_index_sum("_QA.δ(_QA.QuIndex(\\"i\\"), :j) * param(:f, :j)", "j")')
        assert "f(i)" in ans_sum.value
        
        # E2. Continuous integration with Dirac delta (SymPy)
        # ∫ dy delta(x-y) * f(y) over (-Inf, Inf)
        ans_integ = adapter.execute('_nb_eval_continuous_integral("SymPy.sympy.DiracDelta(x - y) * f(y)", "y", "-Inf", "Inf")')
        assert "f(x)" in ans_integ.value

def test_latex_dummy_index_postprocessing():
    r"""Dummy index strings like \#_{1} are named for display.

    2026-08-07: the names were `["j","k","l","m","n"]` indexed MODULO 5, so a
    sixth dummy came back as `j` again — two different summation indices under
    one name.

    2026-08-08, rewritten to pin the PROPERTIES rather than the table it used to
    assert (`k_{1}, k_{2}, …`), because both halves of that table changed for
    reasons the old assertion could not express:

    * indices are SEPARATED.  `\delta_{ik_{1}}` is, by this project's own rule,
      one index named `ik_{1}` (AUDIT-SUBSCRIPT-SPLIT) — the display did not
      round-trip.
    * the name SPENDS NO LETTER the researcher could still use.  A bound index
      lives inside one sum; taking `k`, then `l`, then `m` for it shrinks the
      alphabet left for actual quantities (raised by the researcher).  Extra
      indices are built by subscripting, and every letter already in play — here
      the free `i` and `j` — is skipped.
    """
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    from notebook.runtime.julia_command_builder import JuliaCommandBuilder
    from notebook.models.symbol_context import SymbolContext

    handler = JuliaBackendHandler(JuliaCommandBuilder())
    ctx = SymbolContext()

    raw = r"\sum_{\#_{1}} \delta_{i\#_{1}} \delta_{\#_{1}j}"
    processed = handler.postprocess_result_latex("dummy_canonical", raw, ctx)

    # ONE name throughout, and it is neither of the free indices standing beside it
    name = processed.split(r"\sum_{", 1)[1].split("}", 1)[0] + "}"
    assert name not in ("i", "j"), processed
    assert processed.count(name) == 3, processed
    # separated from its neighbours, so the display can be read back
    assert f"\\delta_{{i,{name}}}" in processed, processed
    assert f"\\delta_{{{name},j}}" in processed, processed

    # No modulo: a sixth dummy is its own name, not the first one again.
    sixth = handler.postprocess_result_latex("c", r"\#_{6}", ctx)
    first = handler.postprocess_result_latex("c", r"\#_{1}", ctx)
    assert sixth != first, (first, sixth)
    # …and no single letter is consumed for any of them.
    for n in range(1, 8):
        got = handler.postprocess_result_latex("c", r"\#_{%d}" % n, ctx)
        assert "_" in got, (n, got)

def test_character_indices_lowering():
    """Verify parsing, resolution, and lowering of character indices for both Julia and Maxima backends."""
    from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
    from notebook.runtime.julia_code_generator import JuliaCodeGenerator
    from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
    from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
    from notebook.models.ir import IndexedSymbol, IndexSymbol, IndexSum
    
    parser = SharedParser()
    ctx = SymbolContext()
    
    # Register boson operator "a"
    ctx.register_quantity("a", kind="operator")
    
    # 1. Julia Lowering tests
    julia_res = JuliaSemanticResolver()
    julia_gen = JuliaCodeGenerator(ctx)
    
    # Scalar index x_i
    ast_x = parser.parse("x_i")
    res_x = julia_res.resolve(ast_x, ctx)
    assert isinstance(res_x, IndexedSymbol)
    assert res_x.base.name == "x"
    assert len(res_x.indices) == 1
    assert isinstance(res_x.indices[0], IndexSymbol)
    assert res_x.indices[0].name == "i"
    # The lowering used to be asserted here as "x(i)".  Measured 2026-07-30, that
    # string was never valid Julia — a bare `x` is not callable, so every such
    # command came back as `Sym x is not callable. Use @syms x(...)`, raw CAS
    # syntax in the researcher's face.  The generator now refuses it with a
    # reason instead (open_items JULIA-INDEXED-SCALAR); the IR above is
    # unchanged, which is what this test is really about.
    with pytest.raises(UnknownNotationError):
        julia_gen.generate(res_x)

    # Matrix index A_{i,j} — indices need a comma or a space between them
    # (AUDIT-SUBSCRIPT-SPLIT, 2026-07-30).  This used to be written A_{ij} and
    # rely on the per-character split that also turned E_{rot} into E(r, o, t).
    ast_A = parser.parse("A_{i,j}")
    res_A = julia_res.resolve(ast_A, ctx)
    assert isinstance(res_A, IndexedSymbol)
    assert res_A.base.name == "A"
    assert len(res_A.indices) == 2
    assert res_A.indices[0].name == "i"
    assert res_A.indices[1].name == "j"
    with pytest.raises(UnknownNotationError):      # same reason as x_i above
        julia_gen.generate(res_A)

    # Unseparated: ONE index named "ij".
    res_A1 = julia_res.resolve(parser.parse("A_{ij}"), ctx)
    assert [ix.name for ix in res_A1.indices] == ["ij"]
    
    # Operator index a_i (free index)
    ast_a = parser.parse("a_i")
    res_a = julia_res.resolve(ast_a, ctx)
    assert isinstance(res_a, IndexedSymbol)
    # Unbound operator index uses QuIndex so QuantumAlgebra operators receive the
    # correct type (QuIndex(::Num) is rejected at runtime when a Symbolics variable
    # is passed to a QuantumAlgebra operator).
    # Note: The code generator uses the _QA alias rather than the full module name.
    # a_i is an operator → emitted as a(_QA.QuIndex("i"))
    assert julia_gen.generate(res_a) == 'a(_QA.QuIndex("i"))'

    # Operator index inside IndexSum (bound index)
    ast_sum = parser.parse("IndexSum(a_j, j)")
    res_sum = julia_res.resolve(ast_sum, ctx)
    assert isinstance(res_sum, IndexSum)
    # 2026-08-07: the dummy is now bound by QuantumAlgebra's own `∑`, so it is
    # written as the plain symbol rather than substituted for a sumindex.
    assert "_QA.∑(:j," in julia_gen.generate(res_sum)
    
    # 2. Maxima Lowering tests
    maxima_res = MaximaSemanticResolver()
    maxima_gen = MaximaCodeGenerator(ctx)
    
    # Scalar index x_i -> x(i) (defaults to function_call in Maxima now!)
    res_x_max = maxima_res.resolve(ast_x, ctx)
    assert isinstance(res_x_max, IndexedSymbol)
    assert maxima_gen.generate(res_x_max) == "x(i)"
    
    # Matrix index A_{ij} -> A(i, j) (defaults to function_call when not registered!)
    res_A_max = maxima_res.resolve(ast_A, ctx)
    assert isinstance(res_A_max, IndexedSymbol)
    assert maxima_gen.generate(res_A_max) == "A(i, j)"
    
    # Operator index a_i -> a(i)
    res_a_max = maxima_res.resolve(ast_a, ctx)
    assert isinstance(res_a_max, IndexedSymbol)
    assert maxima_gen.generate(res_a_max) == "a(i)"

    # 3. Explicit math_type and style overrides tests
    # Register A_matrix as a matrix component -> array_access
    ctx.register_quantity("A_matrix", kind="scalar", math_type={"algebra": "matrix"})
    ast_Am = parser.parse("A_matrix_{i,j}")
    res_Am_max = maxima_res.resolve(ast_Am, ctx)
    assert maxima_gen.generate(res_Am_max) == "A_matrix[i, j]"

    # Register A_matrix_override with explicit function_call style override
    ctx.register_quantity("A_matrix_override", kind="scalar", math_type={"algebra": "matrix"}, index_lowering_style="function_call")
    ast_Amo = parser.parse("A_matrix_override_{i,j}")
    res_Amo_max = maxima_res.resolve(ast_Amo, ctx)
    assert maxima_gen.generate(res_Amo_max) == "A_matrix_override(i, j)"

    # Register x_array with explicit array_access override
    ctx.register_quantity("x_array", kind="scalar", index_lowering_style="array_access")
    ast_xa = parser.parse("x_array_i")
    res_xa_max = maxima_res.resolve(ast_xa, ctx)
    assert maxima_gen.generate(res_xa_max) == "x_array[i]"

    # Register x_atom with explicit atom override
    ctx.register_quantity("x_atom", kind="scalar", index_lowering_style="atom")
    ast_xatom = parser.parse("x_atom_i")
    res_xatom_max = maxima_res.resolve(ast_xatom, ctx)
    assert maxima_gen.generate(res_xatom_max) == "x_atom_i"

    # Julia overrides:
    # array_access override
    res_xa_julia = julia_res.resolve(ast_xa, ctx)
    assert julia_gen.generate(res_xa_julia) == "x_array[i]"

    # atom override
    res_xatom_julia = julia_res.resolve(ast_xatom, ctx)
    assert julia_gen.generate(res_xatom_julia) == "x_atom_i"

