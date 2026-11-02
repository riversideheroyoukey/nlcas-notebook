import pytest
from notebook.models.symbol_context import SymbolContext
from notebook.models.ir import (
    Symbol, IndexSymbol, IndexedSymbol, KroneckerDelta, IndexSum, Mul, Add, Number, FunctionCall
)
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.index_contractor import contract_indices, alphatize_indices

def test_basic_contraction():
    """Verify that a discrete sum containing a Kronecker delta contracts correctly."""
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()
    
    # \sum_j \delta_{ij} f(j) -> f(i)
    # The parser parses delta(i, j) as KroneckerDelta and IndexSum as IndexSum
    ast = parser.parse("IndexSum(delta(i, j) * f(j), j)")
    resolved = resolver.resolve(ast, ctx)
    
    # Contracted should be FunctionCall(Symbol(f), [Symbol(i)])
    assert isinstance(resolved, FunctionCall) or (hasattr(resolved, 'func') and resolved.func.name == 'f')
    assert resolved.func.name == "f"
    assert len(resolved.args) == 1
    assert isinstance(resolved.args[0], Symbol)
    assert resolved.args[0].name == "i"


def test_chained_contraction():
    """Verify that multiple nested Kronecker deltas are chained and contracted correctly."""
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()
    
    # \sum_j \sum_k \delta_{ij} \delta_{jk} g(k) -> g(i)
    ast = parser.parse("IndexSum(IndexSum(delta(i, j) * delta(j, k) * g(k), j), k)")
    resolved = resolver.resolve(ast, ctx)
    
    assert isinstance(resolved, FunctionCall) or (hasattr(resolved, 'func') and resolved.func.name == 'f')
    assert resolved.func.name == "g"
    assert len(resolved.args) == 1
    assert resolved.args[0].name == "i"


def test_diagonal_delta():
    """Verify that a Kronecker delta of identical symbols simplifies to 1."""
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()
    
    ast = parser.parse("delta(i, i)")
    resolved = resolver.resolve(ast, ctx)
    
    assert isinstance(resolved, Number)
    assert resolved.value == 1


def test_alphatize_nested_collisions():
    """Verify that nested discrete sums sharing the same dummy index name are automatically renamed."""
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()
    
    # \sum_i \sum_i f(i) -> outer sum has dummy i, inner sum has dummy i which gets renamed to j
    ast = parser.parse("IndexSum(IndexSum(f(i), i), i)")
    resolved = resolver.resolve(ast, ctx)
    
    assert isinstance(resolved, IndexSum)
    assert resolved.dummy_indices[0].name == "i"
    
    inner = resolved.body
    assert isinstance(inner, IndexSum)
    assert inner.dummy_indices[0].name == "j" # Renamed to next candidate
    assert isinstance(inner.body, FunctionCall)
    assert inner.body.args[0].name == "j"


def test_alphatize_leaves_parallel_sums_alone():
    r"""PARALLEL sums keep the name as written — renaming them was gratuitous.

    Reversed 2026-08-17 (EXPR-REF-PROVENANCE).  This test previously required
    the second sum to become `\sum_j`.  Their scopes are disjoint, so no
    capture is possible, and the rename was not free: Maxima's noun forms do
    not alpha-convert, so `\sum_i x_i - \sum_i x_i` came back unsimplified
    instead of 0 (measured — `ratsimp('sum(x(i),i,a,b) - 'sum(x(j),j,a,b))`
    does not reduce, while the same-dummy version does).

    Measured for the product form specifically, which is what this case is:
    Maxima expands `'sum(f(i),i,1,3) * 'sum(g(i),i,1,3)` to
    `(f(3)+f(2)+f(1))*(g(3)+g(2)+g(1))` — correctly, and identically to the
    renamed spelling.  Nothing merged the two binders.

    NESTED sums are still renamed; see `test_alphatize_nested_collisions`."""
    parser = SharedParser()
    ctx = SymbolContext()
    resolver = MaximaSemanticResolver()

    # (\sum_i f(i)) * (\sum_i g(i))
    ast = parser.parse("IndexSum(f(i), i) * IndexSum(g(i), i)")
    resolved = resolver.resolve(ast, ctx)

    assert isinstance(resolved, Mul)
    assert len(resolved.args) == 2
    sum1, sum2 = resolved.args[0], resolved.args[1]

    assert isinstance(sum1, IndexSum)
    assert isinstance(sum2, IndexSum)
    assert sum1.dummy_indices[0].name == "i"
    assert sum2.dummy_indices[0].name == "i"
    assert sum2.body.args[0].name == "i"


def test_domain_compatibility():
    """Verify that contracting indices with mismatching domains raises a compatibility error, but propagating domain works."""
    ctx = SymbolContext()
    
    # Create two IndexSymbols with compatible/incompatible domains
    idx_spin1 = IndexSymbol(name="s1", domain="spin")
    idx_spin2 = IndexSymbol(name="s2", domain="spin")
    idx_space = IndexSymbol(name="x1", domain="spatial")
    idx_none = IndexSymbol(name="n1")
    
    # 1. Matching domains is fine
    delta_ok = KroneckerDelta(left=idx_spin1, right=idx_spin2)
    expr_ok = IndexSum(body=Mul(args=[delta_ok, Symbol(name="f")]), dummy_indices=[idx_spin2])
    # Resolving contraction should succeed
    contracted = contract_indices(expr_ok, ctx)
    assert isinstance(contracted, Symbol)
    
    # 2. Mismatching domains raises ValueError
    delta_bad = KroneckerDelta(left=idx_spin1, right=idx_space)
    expr_bad = IndexSum(body=Mul(args=[delta_bad, Symbol(name="f")]), dummy_indices=[idx_space])
    with pytest.raises(ValueError, match="Domain mismatch"):
        contract_indices(expr_bad, ctx)
        
    # 3. Propagating domain works
    assert idx_none.domain is None
    delta_prop = KroneckerDelta(left=idx_spin1, right=idx_none)
    expr_prop = IndexSum(body=Mul(args=[delta_prop, Symbol(name="f")]), dummy_indices=[idx_none])
    contract_indices(expr_prop, ctx)
    # The domain "spin" of idx_spin1 should have propagated to idx_none!
    assert idx_none.domain == "spin"


def test_non_commutative_order_preservation():
    """Verify that contraction and alphatizing strictly preserve the order of non-commutative products."""
    ctx = SymbolContext()
    # Register A and B as non-commutative operators
    ctx.register_quantity("A", kind="operator", statistics_kind="noncommutative")
    ctx.register_quantity("B", kind="operator", statistics_kind="noncommutative")
    
    # 1. Mul containing delta(i, j) * A_j * B_k (order: delta, A, B)
    # After contraction of j -> i, result must be A_i * B_k
    dummy_j = IndexSymbol(name="j")
    idx_i = IndexSymbol(name="i")
    idx_k = IndexSymbol(name="k")
    
    delta = KroneckerDelta(left=idx_i, right=dummy_j)
    A_j = IndexedSymbol(base=Symbol(name="A"), indices=[dummy_j])
    B_k = IndexedSymbol(base=Symbol(name="B"), indices=[idx_k])
    
    body = Mul(args=[delta, A_j, B_k])
    expr = IndexSum(body=body, dummy_indices=[dummy_j])
    
    contracted = contract_indices(expr, ctx)
    assert isinstance(contracted, Mul)
    assert len(contracted.args) == 2
    # Check that A_i is first and B_k is second (order is preserved!)
    assert contracted.args[0].base.name == "A"
    assert contracted.args[0].indices[0].name == "i"
    assert contracted.args[1].base.name == "B"
    assert contracted.args[1].indices[0].name == "k"
    
    # 2. Mul containing delta(i, j) * B_k * A_j (order: delta, B, A)
    # After contraction of j -> i, result must be B_k * A_i
    body2 = Mul(args=[delta, B_k, A_j])
    expr2 = IndexSum(body=body2, dummy_indices=[dummy_j])
    
    contracted2 = contract_indices(expr2, ctx)
    assert isinstance(contracted2, Mul)
    assert len(contracted2.args) == 2
    # Check that B_k is first and A_i is second (order is preserved!)
    assert contracted2.args[0].base.name == "B"
    assert contracted2.args[0].indices[0].name == "k"
    assert contracted2.args[1].base.name == "A"
    assert contracted2.args[1].indices[0].name == "i"


def test_save_load_indexed_vs_function_distinction(tmp_path):
    """Verify that the distinction between IndexedSymbol (e.g., x_i) and FunctionCall (e.g., x(i)) is perfectly preserved across session save and load."""
    from notebook.engine import NotebookEngine
    
    path = tmp_path / "index_vs_func.json"
    
    # 1. Register both and save
    with NotebookEngine(title="index_vs_func_test") as e:
        e.start()
        # E1 is parsed from LaTeX "x_i" -> resolving to IndexedSymbol(x, [i])
        e.register_expression("x_i")
        # E2 is parsed from LaTeX "x(i)" -> resolving to FunctionCall(x, [i])
        e.register_expression("x(i)")
        e.save(path)
        
    # 2. Load and verify distinction
    loaded = NotebookEngine.load(path)
    # Ensure they both exist
    assert "E1" in loaded.expressions
    assert "E2" in loaded.expressions
    
    # Start loaded engine explicitly to trigger resolve/ir_ast reconstruction
    loaded.start()
    
    e1_expr = loaded.expressions["E1"]
    e2_expr = loaded.expressions["E2"]
    
    # Verify their canonical representations in the JSON
    assert e1_expr.canonical == "x_i"
    assert e2_expr.canonical == "x(i)"
    
    # Verify that reconstruction via engine._handler.parse_ir restores the distinction properly
    e1_ast = loaded._handler.parse_ir(e1_expr.canonical)
    e2_ast = loaded._handler.parse_ir(e2_expr.canonical)
    assert isinstance(e1_ast, IndexedSymbol)
    assert isinstance(e2_ast, FunctionCall)



def test_alphatize_composite_index_renamed():
    """When _alphatize renames dummy k→j in a parallel sum, composite indices like -k must also be renamed to -j.

    This is the BCS ξ_j problem root cause: before the fix,
    IndexSum([k], c'_k * c'_{-k}) in a parallel Add would rename c'_k → c'_j
    but leave c'_{-k} unchanged, producing c'_j * c'_{-k} (inconsistent indices).
    """
    from notebook.runtime.index_contractor import substitute_symbol_or_index

    # Simulate: IndexSymbol("-k") when k→j substitution is applied
    neg_k = IndexSymbol(name="-k")
    result = substitute_symbol_or_index(neg_k, "k", Symbol(name="j"))
    assert result.name == "-j", f"Expected '-j', got '{result.name}'"

    # Simulate: IndexedSymbol(c', [IndexSymbol("-k")]) — the creation operator
    c_prime = Symbol(name="c'")
    c_neg_k = IndexedSymbol(base=c_prime, indices=[IndexSymbol(name="-k")])
    result2 = substitute_symbol_or_index(c_neg_k, "k", Symbol(name="j"))
    assert isinstance(result2, IndexedSymbol)
    assert result2.indices[0].name == "-j", f"Expected index '-j', got '{result2.indices[0].name}'"

    # Verify non-component names are NOT changed: "ky" should stay "ky"
    ky = IndexSymbol(name="ky")
    result3 = substitute_symbol_or_index(ky, "k", Symbol(name="j"))
    assert result3.name == "ky", f"Expected 'ky' unchanged, got '{result3.name}'"

    # Verify composite sum index: "k+q" → "j+q"
    kplusq = IndexSymbol(name="k+q")
    result4 = substitute_symbol_or_index(kplusq, "k", Symbol(name="j"))
    assert result4.name == "j+q", f"Expected 'j+q', got '{result4.name}'"


def test_alphatize_keeps_one_momentum_across_parallel_sums():
    r"""The BCS Hamiltonian: `Σ_k ξ_k c_k + Σ_k Δ c'_k c'_{-k}` keeps ONE `k`.

    Reversed 2026-08-17 (EXPR-REF-PROVENANCE).  This test used to require the
    pairing term to become `Σ_j … c'_{-j}`.  Both spellings are correct
    mathematics, but the Julia builder had already decided against the rename
    for this exact Hamiltonian — `julia_build_mixin` passes
    `skip_alphatize=True` at every one of its seven call sites, with the note
    that "parallel IndexSum nodes should keep the same dummy index k … so that
    normal_form() treats them as the same physical momentum".

    So the same question had two answers: the shared alphatiser renamed, and
    the one consumer that cared opted out.  The default now agrees with the
    consumer.  (`skip_alphatize` still differs for NESTED sums and stays.)
    """
    # Build: xi_k * c_k (kinetic) and Delta * c'_k * c'_{-k} (pairing)
    k = IndexSymbol(name="k")
    xi_k = IndexedSymbol(base=Symbol(name="xi"), indices=[k])
    c_k  = IndexedSymbol(base=Symbol(name="c"),  indices=[k])
    cd_k = IndexedSymbol(base=Symbol(name="c'"), indices=[k])
    cd_neg_k = IndexedSymbol(base=Symbol(name="c'"), indices=[IndexSymbol(name="-k")])
    Delta = Symbol(name="Delta")

    kinetic = IndexSum(body=Mul(args=[xi_k, c_k]),  dummy_indices=[IndexSymbol(name="k")])
    pairing = IndexSum(body=Mul(args=[Delta, cd_k, cd_neg_k]), dummy_indices=[IndexSymbol(name="k")])
    hamiltonian = Add(args=[kinetic, pairing])

    result = alphatize_indices(hamiltonian)

    # First term keeps dummy k
    assert isinstance(result, Add)
    sum1, sum2 = result.args
    assert sum1.dummy_indices[0].name == "k"

    # Second term keeps k as written
    assert sum2.dummy_indices[0].name == "k"

    # …and so does its body, including the composite index -k
    pairing_body = sum2.body  # Mul([Delta, cd_k, cd_{-k}])
    assert isinstance(pairing_body, Mul)
    ops = [a for a in pairing_body.args if isinstance(a, IndexedSymbol)]
    assert sorted(idx.name for op in ops for idx in op.indices) == ["-k", "k"]
