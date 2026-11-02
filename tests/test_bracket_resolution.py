import pytest
from notebook.models.ir import Symbol, BracketExpr, Commutator, Anticommutator
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver

def test_commutator_resolution():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # [A, B] -> Commutator(A, B)
    bracket = BracketExpr(args=[Symbol(name="A"), Symbol(name="B")], kind="square")
    res = resolver._resolve_node(bracket, ctx)
    assert isinstance(res, Commutator)
    assert res.a.name == "A"
    assert res.b.name == "B"

def test_anticommutator_resolution():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # {A, B} -> Anticommutator(A, B)
    bracket = BracketExpr(args=[Symbol(name="A"), Symbol(name="B")], kind="curly")
    res = resolver._resolve_node(bracket, ctx)
    
    assert isinstance(res, Anticommutator)
    assert res.a.name == "A"
    assert res.b.name == "B"

