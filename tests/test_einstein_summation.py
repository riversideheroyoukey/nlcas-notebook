import pytest
from notebook.models.ir import Symbol, IndexedSymbol, IndexSymbol, IndexSum, Subscript, Superscript, Pow, Mul
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.auto_einstein import auto_einstein_sum

def test_ast_order_preservation():
    parser = SharedParser()
    
    # A^\mu_\nu (Upper then lower)
    ast1 = parser.parse("A^mu_nu")
    assert isinstance(ast1, Subscript)
    assert isinstance(ast1.base, Superscript)
    assert ast1.base.script.name == "mu"
    assert ast1.sub.name == "nu"

    # A_\nu^\mu (Lower then upper)
    ast2 = parser.parse("A_nu^mu")
    assert isinstance(ast2, Superscript)
    assert isinstance(ast2.base, Subscript)
    assert ast2.base.sub.name == "nu"
    assert ast2.script.name == "mu"

def test_semantic_resolution():
    context = SymbolContext()
    context.einstein_summation = True
    resolver = MaximaSemanticResolver()
    
    parser = SharedParser()
    ast = parser.parse("A^mu_nu")
    resolved = resolver.resolve(ast, context)
    
    assert isinstance(resolved, IndexedSymbol)
    assert resolved.base.name == "A"
    assert len(resolved.indices) == 2
    
    assert resolved.indices[0].name == "mu"
    assert resolved.indices[0].is_up == True
    
    assert resolved.indices[1].name == "nu"
    assert resolved.indices[1].is_up == False

def test_einstein_summation_auto_contract():
    context = SymbolContext()
    context.einstein_summation = True
    resolver = MaximaSemanticResolver()
    parser = SharedParser()
    
    ast = parser.parse("A^mu * B_mu")
    resolved = resolver.resolve(ast, context)
    
    assert isinstance(resolved, IndexSum)
    assert len(resolved.dummy_indices) == 1
    assert resolved.dummy_indices[0].name == "mu"
    
    body = resolved.body
    assert isinstance(body, Mul)

def test_einstein_summation_disabled():
    context = SymbolContext()
    context.einstein_summation = False
    resolver = MaximaSemanticResolver()
    parser = SharedParser()
    
    # A^mu is just Pow when einstein_summation is false
    ast = parser.parse("A^mu * B_mu")
    resolved = resolver.resolve(ast, context)
    
    assert not isinstance(resolved, IndexSum)
    assert isinstance(resolved, Mul)
    # A^mu becomes Pow(A, mu)
    # B_mu becomes IndexedSymbol(B, [mu])
    
    has_pow = False
    for arg in resolved.args:
        if isinstance(arg, Pow):
            has_pow = True
            assert arg.base.name == "A"
            assert arg.exponent.name == "mu"
    assert has_pow

if __name__ == "__main__":
    pytest.main(["-v", __file__])
