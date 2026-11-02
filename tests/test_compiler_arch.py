import pytest
from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import Quantity
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.ir import (
    Juxtaposition, BracketExpr, Assignment, Mul, Commutator, Add, Symbol, Number, Equation
)

def test_shared_parser_juxtaposition():
    """Verify that SharedParser outputs pure syntax Juxtaposition node for implicit mul."""
    parser = SharedParser()
    ast = parser.parse("a b")
    assert isinstance(ast, Juxtaposition)
    assert isinstance(ast.left, Symbol) and ast.left.name == "a"
    assert isinstance(ast.right, Symbol) and ast.right.name == "b"

def test_shared_parser_brackets():
    """Verify that SharedParser outputs generic BracketExpr nodes."""
    parser = SharedParser()
    
    ast_square = parser.parse("[x, y]")
    assert isinstance(ast_square, BracketExpr)
    assert ast_square.kind == "square"
    assert len(ast_square.args) == 2
    
    ast_curly = parser.parse("{x, y}")
    assert isinstance(ast_curly, BracketExpr)
    assert ast_curly.kind == "curly"
    assert len(ast_curly.args) == 2

def test_shared_parser_assignment():
    """Verify that SharedParser outputs Assignment nodes."""
    parser = SharedParser()
    ast = parser.parse("c := a + b")
    assert isinstance(ast, Assignment)
    assert isinstance(ast.lhs, Symbol) and ast.lhs.name == "c"

def test_julia_compilation_pipeline():
    """Verify Julia-specific Semantic Resolver and Code Generator pipeline."""
    parser = SharedParser()
    resolver = JuliaSemanticResolver()
    generator = JuliaCodeGenerator()
    ctx = SymbolContext()
    
    # 1. Bracket [A, B] -> comm(A, B)
    ast = parser.parse("[A, B]")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, Commutator)
    cmd = generator.generate(resolved)
    assert cmd == "comm(A, B)"
    
    # 2. Anticommutator {A, B} -> ((A)*(B) + (B)*(A))
    ast = parser.parse("{A, B}")
    resolved = resolver.resolve(ast, ctx)
    from notebook.models.ir import Anticommutator
    assert isinstance(resolved, Anticommutator)
    cmd = generator.generate(resolved)
    assert cmd == "((A)*(B) + (B)*(A))"
    
    # 3. Juxtaposition -> Mul
    ast = parser.parse("a'(1) a(1)")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, Mul)
    cmd = generator.generate(resolved)
    assert cmd.replace(" ", "") == "a'(1)*a(1)"

def test_maxima_compilation_pipeline():
    """Verify Maxima-specific Semantic Resolver and Code Generator pipeline."""
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    generator = MaximaCodeGenerator()
    ctx = SymbolContext()
    
    # 1. Bracket [A, B] -> Commutator in Maxima too
    ast = parser.parse("[A, B]")
    resolved = resolver.resolve(ast, ctx)
    cmd = generator.generate(resolved)
    assert cmd == "((A)*(B) - (B)*(A))"
    
    # 2. Juxtaposition -> Mul
    ast = parser.parse("a b")
    resolved = resolver.resolve(ast, ctx)
    assert isinstance(resolved, Mul)
    cmd = generator.generate(resolved)
    assert cmd.replace(" ", "") == "a*b"


def test_julia_negated_index_in_index_sum():
    """c_{-k} inside an IndexSum must generate c(sumindex(-1)), not QuIndex(Symbol('-k')).

    This is the BCS Hamiltonian case: ∑_k ... c(-k) c'(-k) ...
    QuantumAlgebra rejects multi-char Symbol names, so '-k' must never reach QuIndex.
    -sumindex(1) would raise MethodError: -(::QuIndex), so we use sumindex(-1) instead.
    We build the IR directly to avoid the parser treating 'sum_k' as a subscript.
    """
    from notebook.models.ir import IndexSum, IndexSymbol, IndexedSymbol, Symbol
    generator = JuliaCodeGenerator()

    # IndexSum over k, body = c(-k)
    dummy_k = IndexSymbol(name="k")
    minus_k  = IndexSymbol(name="-k")   # produced by _extract_indices for -k subscript
    c_minus_k = IndexedSymbol(base=Symbol(name="c"), indices=[minus_k])
    ir = IndexSum(body=c_minus_k, dummy_indices=[dummy_k])

    cmd = generator.generate(ir)

    assert 'Symbol("-k")' not in cmd, (
        f'QuIndex(Symbol("-k")) must not appear; got: {cmd}'
    )
    assert "sumindex(-1)" in cmd, (
        f"Expected sumindex(-1) in generated code (not -sumindex(1)); got: {cmd}"
    )


def test_julia_negated_index_unbound():
    """c_{-k} outside an IndexSum must generate c(QuantumAlgebra.QuIndex(Symbol("K"))).

    QuantumAlgebra requires single-char Symbol names for QuIndex, so Symbol("-k") is
    illegal.  -QuIndex(:k) also fails (MethodError: -(::QuIndex)).  Symbolics bare
    variables are rejected by QuIndex(::Num) when passed to a QuantumAlgebra operator.
    We use the uppercase-letter convention: -k → QuIndex(:K), -i → QuIndex(:I), etc.
    Delta contractions for BCS-type calculations cannot be evaluated automatically
    within QuantumAlgebra and must be interpreted manually by the user.
    """
    from notebook.models.ir import IndexSymbol, IndexedSymbol, Symbol
    generator = JuliaCodeGenerator()

    minus_k   = IndexSymbol(name="-k")
    c_minus_k = IndexedSymbol(base=Symbol(name="c"), indices=[minus_k])

    cmd = generator.generate(c_minus_k)

    assert '_QA.QuIndex("K")' in cmd, (
        f'Expected _QA.QuIndex("K") (uppercase convention for -k); got: {cmd}'
    )
    assert 'Symbol("-k")' not in cmd, (
        f'Symbol("-k") must never appear (QuantumAlgebra single-char constraint); got: {cmd}'
    )
