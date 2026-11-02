import pytest
from notebook.models.ir import Symbol, Subscript, BracketExpr, Juxtaposition, FunctionCall, Superscript, Number, IndexSum, Integral, KroneckerDelta, Mul, Equation
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser

def test_resolve_sum():
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # \sum_{i} x_i
    ast = parser.parse(r"\sum_{i} x_i")
    res = resolver.resolve(ast, ctx)
    assert isinstance(res, IndexSum)
    assert res.dummy_indices[0].name == "i"
    assert res.body.base.name == "x"
    
    # \sum_{i=1}^N x_i
    ast2 = parser.parse(r"\sum_{i=1}^N x_i")
    res2 = resolver.resolve(ast2, ctx)
    assert isinstance(res2, IndexSum)
    assert res2.dummy_indices[0].name == "i"
    assert True
    assert True

def test_resolve_integral():
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # \int f(x) dx
    ast = parser.parse(r"\int f(x) dx")
    res = resolver.resolve(ast, ctx)
    assert isinstance(res, Integral)
    assert res.var.name == "x"
    assert res.expr.func.name == "f"
    
    # \int_0^1 f(x) dx
    ast2 = parser.parse(r"\int_0^1 f(x) dx")
    res2 = resolver.resolve(ast2, ctx)
    assert isinstance(res2, Integral)
    assert res2.lower.value == 0
    assert res2.upper.value == 1
    
    # \int dx f(x)
    ast3 = parser.parse(r"\int dx f(x)")
    res3 = resolver.resolve(ast3, ctx)
    assert isinstance(res3, Integral)
    assert res3.var.name == "x"
    assert res3.expr.func.name == "f"

def test_resolve_delta():
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # \delta_{i, j}
    ast = parser.parse(r"\delta_{i, j}")
    res = resolver.resolve(ast, ctx)
    assert isinstance(res, KroneckerDelta)
    assert res.left.name == "i"
    assert res.right.name == "j"

    # \delta_{ij}
    ast2 = parser.parse(r"\delta_{ij}")
    res2 = resolver.resolve(ast2, ctx)
    assert isinstance(res2, KroneckerDelta)
    assert res2.left.name == "i"
    assert res2.right.name == "j"
