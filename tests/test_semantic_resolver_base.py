import pytest
from notebook.models.ir import Symbol, PrimeExpr, IndexedSymbol, Add, Mul, Pow, FunctionCall, Bra, Ket, Commutator, NonCommutativeMul, IndexSymbol
from notebook.models.symbol_context import SymbolContext, SymbolStatistics
from notebook.models.quantity import Quantity, MathType
from notebook.runtime.semantic_resolver import get_algebra_type, is_noncommutative

def test_get_algebra_type():
    ctx = SymbolContext()
    
    # Register a matrix
    ctx.register("A", Quantity(name="A", kind="anonymous", math_type=MathType(algebra="matrix")), "bose")
    # Register an operator
    ctx.register("Op", Quantity(name="Op", kind="operator", math_type=MathType(algebra="operator")), "noncommutative")
    
    # 1. Symbol no prop -> "scalar"
    assert get_algebra_type(Symbol(name="x"), ctx) == "scalar"
    
    # 2. PrimeExpr -> forwards to inner
    assert get_algebra_type(PrimeExpr(expr=Symbol(name="A")), ctx) == "matrix"
    
    # 3. IndexedSymbol -> forwards to base
    assert get_algebra_type(IndexedSymbol(base=Symbol(name="Op"), indices=[IndexSymbol(name="i")]), ctx) == "operator"
    
    # 4. Add / Mul -> looks for non-scalar
    assert get_algebra_type(Add(args=[Symbol(name="x"), Symbol(name="A")]), ctx) == "matrix"
    assert get_algebra_type(Mul(args=[Symbol(name="x"), Symbol(name="y")]), ctx) == "scalar"
    
    # 5. Pow -> forwards to base
    assert get_algebra_type(Pow(base=Symbol(name="A"), exponent=Symbol(name="2")), ctx) == "matrix"
    
    # 6. FunctionCall -> forwards to func
    assert get_algebra_type(FunctionCall(func=Symbol(name="Op"), args=[]), ctx) == "operator"
    
    # 7. Fallback
    class DummyExpr:
        pass
    assert get_algebra_type(DummyExpr(), ctx) == "scalar"

def test_is_noncommutative():
    ctx = SymbolContext()
    ctx.register("A", Quantity(name="A", kind="operator", math_type=MathType(algebra="operator")), "noncommutative")
    ctx.register("F", Quantity(name="F", kind="operator", math_type=MathType(algebra="operator")), "fermi")
    ctx.register("x", Quantity(name="x", kind="scalar", math_type=MathType(algebra="scalar")), "bose")
    
    # Symbol
    assert is_noncommutative(Symbol(name="A"), ctx) is True
    assert is_noncommutative(Symbol(name="F"), ctx) is True
    assert is_noncommutative(Symbol(name="x"), ctx) is False
    assert is_noncommutative(Symbol(name="unknown"), ctx) is False
    
    # Bra, Ket, Commutator, NonCommutativeMul
    assert is_noncommutative(Bra(label=Symbol(name="n")), ctx) is True
    assert is_noncommutative(Ket(label=Symbol(name="n")), ctx) is True
    assert is_noncommutative(Commutator(a=Symbol(name="x"), b=Symbol(name="y")), ctx) is True
    assert is_noncommutative(NonCommutativeMul(args=[Symbol(name="x"), Symbol(name="y")]), ctx) is True
    
    # IndexedSymbol
    assert is_noncommutative(IndexedSymbol(base=Symbol(name="A"), indices=[IndexSymbol(name="i")]), ctx) is True
    assert is_noncommutative(IndexedSymbol(base=Symbol(name="x"), indices=[IndexSymbol(name="i")]), ctx) is False
    
    # PrimeExpr
    assert is_noncommutative(PrimeExpr(expr=Symbol(name="F")), ctx) is True
    
    # Mul
    assert is_noncommutative(Mul(args=[Symbol(name="x"), Symbol(name="A")]), ctx) is True
    assert is_noncommutative(Mul(args=[Symbol(name="x"), Symbol(name="y")]), ctx) is False
    
    # Add
    assert is_noncommutative(Add(args=[Symbol(name="x"), Symbol(name="F")]), ctx) is True
    assert is_noncommutative(Add(args=[Symbol(name="x"), Symbol(name="y")]), ctx) is False
    
    # Pow
    assert is_noncommutative(Pow(base=Symbol(name="A"), exponent=Symbol(name="2")), ctx) is True
    
    # FunctionCall
    assert is_noncommutative(FunctionCall(func=Symbol(name="A"), args=[]), ctx) is True
    assert is_noncommutative(FunctionCall(func=Symbol(name="unknown"), args=[]), ctx) is False
    
    # FunctionCall where the func is a Symbol but not resolved recursively in a weird way
    # Just checking it picks up "F" as a func
    assert is_noncommutative(FunctionCall(func=Symbol(name="F"), args=[Symbol(name="x")]), ctx) is True

    # Fallback
    class DummyExpr:
        pass
    assert is_noncommutative(DummyExpr(), ctx) is False
