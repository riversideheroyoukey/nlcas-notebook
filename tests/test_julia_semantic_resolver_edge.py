import pytest
from notebook.models.ir import Symbol, BracketExpr, FunctionCall, CasOperation, IndexSum, Integral, PrimeExpr, Juxtaposition, IndexSymbol
from notebook.models.symbol_context import SymbolContext, SymbolStatistics
from notebook.models.quantity import Quantity, MathType
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.models.ir import Commutator, Add, Mul, NonCommutativeMul, Pow, IndexedSymbol, Adjoint, Transpose

def test_bracket_expr_fallback():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    # Bracket with wrong number of args returns BracketExpr
    be_wrong_args = BracketExpr(kind="square", args=[Symbol(name="A")])
    res_wrong = resolver.resolve(be_wrong_args, ctx)
    assert isinstance(res_wrong, BracketExpr)
    assert len(res_wrong.args) == 1
        
    # Unknown bracket kind returns BracketExpr
    be_unknown = BracketExpr(kind="round", args=[Symbol(name="A"), Symbol(name="B")])
    res_unknown = resolver.resolve(be_unknown, ctx)
    assert isinstance(res_unknown, BracketExpr)

def test_bracket_expr_commutators():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    # Square bracket -> Commutator
    be_square = BracketExpr(kind="square", args=[Symbol(name="A"), Symbol(name="B")])
    res_sq = resolver.resolve(be_square, ctx)
    assert isinstance(res_sq, Commutator)
    assert res_sq.a.name == "A"
    
    # Curly bracket -> AntiCommutator
    be_curly = BracketExpr(kind="curly", args=[Symbol(name="A"), Symbol(name="B")])
    res_curly = resolver.resolve(be_curly, ctx)
    from notebook.models.ir import Anticommutator
    assert isinstance(res_curly, Anticommutator)
    assert res_curly.a.name == "A"

def test_pow_dagger_transpose_fallback():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    ctx.register("A", Quantity(name="A", kind="anonymous", math_type=MathType(algebra="matrix")), "bose")
    ctx.register("x", Quantity(name="x", kind="scalar", math_type=MathType(algebra="scalar")), "bose")
    
    # Adjoint on Matrix -> FunctionCall(adjoint)
    p_adj = Pow(base=Symbol(name="A"), exponent=Symbol(name="dagger"))
    res_adj = resolver.resolve(p_adj, ctx)
    assert isinstance(res_adj, Adjoint)
    assert res_adj.expr.name == "A"
    
    # Adjoint on Scalar -> Adjoint(x)
    p_adj_s = Pow(base=Symbol(name="x"), exponent=Symbol(name="dagger"))
    res_adj_s = resolver.resolve(p_adj_s, ctx)
    assert isinstance(res_adj_s, Adjoint)
    assert res_adj_s.expr.name == "x"

    # Transpose on Matrix -> FunctionCall(transpose)
    p_t = Pow(base=Symbol(name="A"), exponent=Symbol(name="T"))
    res_t = resolver.resolve(p_t, ctx)
    assert isinstance(res_t, Transpose)
    assert res_t.expr.name == "A"
    
    # Transpose on Scalar -> Pow(x, T)
    p_t_s = Pow(base=Symbol(name="x"), exponent=Symbol(name="T"))
    res_t_s = resolver.resolve(p_t_s, ctx)
    assert isinstance(res_t_s, Pow)

def test_cas_operation_and_prime():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    # CasOperation
    op = CasOperation(
        operation="simplify",
        args={"expr": Symbol(name="x"), "list_arg": [Symbol(name="y")]}
    )
    res = resolver.resolve(op, ctx)
    assert isinstance(res.args["expr"], Symbol)
    assert isinstance(res.args["list_arg"][0], Symbol)
    
    # PrimeExpr
    pe = PrimeExpr(expr=Symbol(name="x"))
    res_pe = resolver.resolve(pe, ctx)
    assert isinstance(res_pe, Symbol)
    assert res_pe.name == "x'"

def test_juxtaposition_curly_invalid_index():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    jux = Juxtaposition(
        left=Symbol(name="A"),
        right=BracketExpr(kind="curly", args=[FunctionCall(func=Symbol(name="f"), args=[])])
    )
    with pytest.raises(ValueError, match="IndexedSymbol index must resolve to a Symbol or IndexSymbol"):
        resolver.resolve(jux, ctx)

def test_integral_and_index_sum_exceptions():
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    
    fc_int = FunctionCall(func=Symbol(name="Integral"), args=[Symbol(name="x")])
    with pytest.raises(ValueError, match="Integral requires 2 to 4 arguments"):
        resolver.resolve(fc_int, ctx)
        
    fc_int_var = FunctionCall(func=Symbol(name="Integral"), args=[Symbol(name="x"), FunctionCall(func=Symbol(name="f"), args=[])])
    with pytest.raises(ValueError, match="Integral variable must be a symbol"):
        resolver.resolve(fc_int_var, ctx)
        
    fc_sum = FunctionCall(func=Symbol(name="IndexSum"), args=[Symbol(name="x")])
    with pytest.raises(ValueError, match="IndexSum requires exactly 2 arguments"):
        resolver.resolve(fc_sum, ctx)


# ---------------------------------------------------------------------------
# Regression: Delta must not be mistaken for Laplacian in quantum/BCS sessions
# ---------------------------------------------------------------------------
# See: https://github.com/.../issues/... (ハミルトニアン保存後 [E1, c_k] → laplacian error)

def test_delta_as_laplacian_without_context():
    """Delta f with no context → Laplacian (math convention preserved)."""
    from notebook.models.ir import Laplacian
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    # Mul([Symbol("Delta"), Symbol("f")]) without context → Laplacian
    mul = Mul(args=[Symbol(name="Delta"), Symbol(name="f")])
    res = resolver._resolve_node(mul, ctx)
    assert isinstance(res, Laplacian), f"Expected Laplacian, got {type(res)}"
    assert isinstance(res.expr, Symbol) and res.expr.name == "f"


def test_delta_as_symbol_when_registered():
    """Delta followed by operator when Delta is registered → multiplication, NOT Laplacian.

    This is the BCS regression: Δ c†_k c†_{-k} must not become Laplacian(c†_k c†_{-k}).
    """
    from notebook.models.ir import Laplacian
    from notebook.models.quantity import Quantity
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    # Register Delta as a scalar parameter (BCS order parameter)
    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

    # Mul([Symbol("Delta"), Symbol("psi")]) with Delta in context → NOT Laplacian
    mul = Mul(args=[Symbol(name="Delta"), Symbol(name="psi")])
    res = resolver._resolve_node(mul, ctx)
    assert not isinstance(res, Laplacian), (
        f"Delta registered in context must NOT be treated as Laplacian; got {type(res)}"
    )


def test_delta_juxtaposition_without_context():
    """Juxtaposition(Delta, f) without context → Laplacian."""
    from notebook.models.ir import Laplacian
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    jux = Juxtaposition(left=Symbol(name="Delta"), right=Symbol(name="f"))
    res = resolver._resolve_node(jux, ctx)
    assert isinstance(res, Laplacian), f"Expected Laplacian, got {type(res)}"


def test_delta_juxtaposition_registered_is_not_laplacian():
    """Juxtaposition(Delta, operator_expr) when Delta registered → multiplication.

    Regression test: [H, c_k] where H contains Delta c†_k c†_{-k} caused
    'Sym laplacian is not callable' because Delta was mistaken for ∇².
    """
    from notebook.models.ir import Laplacian
    from notebook.models.quantity import Quantity
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

    # Simulate: Delta followed by an operator product
    jux = Juxtaposition(
        left=Symbol(name="Delta"),
        right=Symbol(name="c"),
    )
    res = resolver._resolve_node(jux, ctx)
    assert not isinstance(res, Laplacian), (
        f"Delta registered in context must NOT become Laplacian in Juxtaposition; got {type(res)}"
    )


def test_laplacian_explicit_function_always_works():
    """laplacian(f) as explicit function call → always Laplacian regardless of context."""
    from notebook.models.ir import Laplacian
    from notebook.models.quantity import Quantity
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    # Even if Delta is registered, explicit "laplacian" function → Laplacian
    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

    fc = FunctionCall(func=Symbol(name="laplacian"), args=[Symbol(name="f")])
    res = resolver._resolve_node(fc, ctx)
    assert isinstance(res, Laplacian), f"Explicit laplacian() must always → Laplacian; got {type(res)}"


def test_nabla_squared_always_laplacian():
    """∇² f → Laplacian regardless of Delta registration."""
    from notebook.models.ir import Laplacian, Pow, Number
    from notebook.models.quantity import Quantity
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()

    ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

    lap = Mul(args=[
        Pow(base=Symbol(name="nabla"), exponent=Number(value=2)),
        Symbol(name="phi"),
    ])
    res = resolver._resolve_node(lap, ctx)
    assert isinstance(res, Laplacian), f"∇² phi must always → Laplacian; got {type(res)}"
