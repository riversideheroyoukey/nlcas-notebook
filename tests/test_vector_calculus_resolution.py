import pytest
from notebook.models.ir import Symbol, FunctionCall, DotProduct, CrossProduct, Gradient, Divergence, Curl, Laplacian, Mul, Number, Pow
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver

def test_vector_calculus_function_call():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # div(F) -> Divergence(F)
    f_call = FunctionCall(func=Symbol(name="div"), args=[Symbol(name="F")])
    res = resolver._resolve_node(f_call, ctx)
    assert isinstance(res, Divergence)
    assert res.expr.name == "F"

    # grad(phi) -> Gradient(phi)
    g_call = FunctionCall(func=Symbol(name="grad"), args=[Symbol(name="phi")])
    res = resolver._resolve_node(g_call, ctx)
    assert isinstance(res, Gradient)
    assert res.expr.name == "phi"

    # rot(A) -> Curl(A)
    c_call = FunctionCall(func=Symbol(name="rot"), args=[Symbol(name="A")])
    res = resolver._resolve_node(c_call, ctx)
    assert isinstance(res, Curl)
    assert res.expr.name == "A"

def test_vector_calculus_omitted_parens():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()

    # div F (implicit mul) -> Divergence(F)
    mul_div = Mul(args=[Symbol(name="div"), Symbol(name="F")])
    res = resolver._resolve_node(mul_div, ctx)
    assert isinstance(res, Divergence)
    assert res.expr.name == "F"

    # grad phi
    mul_grad = Mul(args=[Symbol(name="grad"), Symbol(name="phi")])
    res = resolver._resolve_node(mul_grad, ctx)
    assert isinstance(res, Gradient)
    assert res.expr.name == "phi"

def test_vector_calculus_nabla():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()

    # nabla . F
    dot = DotProduct(left=Symbol(name="nabla"), right=Symbol(name="F"))
    res = resolver._resolve_node(dot, ctx)
    assert isinstance(res, Divergence)
    assert res.expr.name == "F"

    # nabla ~ A
    cross = CrossProduct(left=Symbol(name="nabla"), right=Symbol(name="A"))
    res = resolver._resolve_node(cross, ctx)
    assert isinstance(res, Curl)
    assert res.expr.name == "A"

    # nabla phi
    mul = Mul(args=[Symbol(name="nabla"), Symbol(name="phi")])
    res = resolver._resolve_node(mul, ctx)
    assert isinstance(res, Gradient)
    assert res.expr.name == "phi"

    # nabla^2 phi
    lap = Mul(args=[Pow(base=Symbol(name="nabla"), exponent=Number(value=2)), Symbol(name="phi")])
    res = resolver._resolve_node(lap, ctx)
    assert isinstance(res, Laplacian)
    assert res.expr.name == "phi"
