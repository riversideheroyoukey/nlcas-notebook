import pytest
from notebook.models.ir import Symbol, Add, Mul, BracketExpr, FunctionCall, KroneckerDelta, IndexSymbol, Integral, IndexSum, CasOperation, PrimeExpr, IndexedSymbol, Derivative
from notebook.models.symbol_context import SymbolContext, SymbolStatistics
from notebook.models.quantity import Quantity, MathType
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver, PrimeDerivativeError

def test_index_sum_exceptions():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # dummy_node is a square BracketExpr (e.g. [i, j])
    # Func = IndexSum(expr, [i, j])
    fc = FunctionCall(
        func=Symbol(name="IndexSum"),
        args=[
            Symbol(name="x"),
            BracketExpr(kind="square", args=[Symbol(name="i"), IndexSymbol(name="j")])
        ]
    )
    res = resolver.resolve(fc, ctx)
    assert isinstance(res, IndexSum)
    assert len(res.dummy_indices) == 2
    assert res.dummy_indices[0].name == "i"
    assert res.dummy_indices[1].name == "j"

    # dummy_node is an IndexSymbol directly
    fc2 = FunctionCall(
        func=Symbol(name="IndexSum"),
        args=[
            Symbol(name="x"),
            IndexSymbol(name="k")
        ]
    )
    res2 = resolver.resolve(fc2, ctx)
    assert isinstance(res2, IndexSum)
    assert res2.dummy_indices[0].name == "k"

    # dummy_node is something invalid
    fc_invalid = FunctionCall(
        func=Symbol(name="IndexSum"),
        args=[
            Symbol(name="x"),
            CasOperation(operation="invalid")
        ]
    )
    with pytest.raises(ValueError, match="IndexSum summation variables must be symbols"):
        resolver.resolve(fc_invalid, ctx)

def test_integral_exceptions():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()

    # Wrong number of args
    fc_wrong_args = FunctionCall(func=Symbol(name="Integral"), args=[Symbol(name="x")])
    with pytest.raises(ValueError, match="Integral requires 2 to 4 arguments, got 1"):
        resolver.resolve(fc_wrong_args, ctx)
        
    # Var node is not a Symbol
    fc_bad_var = FunctionCall(
        func=Symbol(name="Integral"),
        args=[Symbol(name="x"), FunctionCall(func=Symbol(name="f"), args=[])]
    )
    with pytest.raises(ValueError, match="Integral variable must be a symbol"):
        resolver.resolve(fc_bad_var, ctx)

def test_kronecker_delta_conversion():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # operands are Symbol -> should become IndexSymbol
    kd = KroneckerDelta(left=Symbol(name="i"), right=Symbol(name="j"))
    res = resolver.resolve(kd, ctx)
    assert isinstance(res.left, IndexSymbol)
    assert isinstance(res.right, IndexSymbol)
    assert res.left.name == "i"
    assert res.right.name == "j"

def test_cas_operation_args():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # CasOperation with mixed args
    op = CasOperation(
        operation="simplify",
        args={
            "expr": Symbol(name="x"),
            "options": [Symbol(name="y"), "string_val"],
            "flag": True
        }
    )
    res = resolver.resolve(op, ctx)
    assert isinstance(res.args["expr"], Symbol)
    assert isinstance(res.args["options"][0], Symbol)
    assert res.args["options"][1] == "string_val"
    assert res.args["flag"] == True

def test_prime_expr():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    ctx.register("a", Quantity(name="a", kind="operator", math_type=MathType(algebra="operator")), "noncommutative")
    
    # Double prime
    pe = PrimeExpr(expr=PrimeExpr(expr=Symbol(name="a")))
    res = resolver.resolve(pe, ctx)
    assert isinstance(res, Symbol)
    assert res.name == "a''"
    
    # Verify it registered the new symbol with the same properties
    prop = ctx.get("a''")
    assert prop is not None
    assert prop.statistics.kind == "noncommutative"
    
    # Prime on an INDEX.  This used to drop the prime — the default for any
    # non-Symbol did — and that is how `v(\mathbf r - \mathbf r\')` reached
    # Maxima as `v(0)` (measured 2026-08-20).  An index is a name, so it takes
    # the prime: `c_{k\'\sigma}` really is a second momentum.
    pe2 = PrimeExpr(expr=IndexSymbol(name="i"))
    res2 = resolver.resolve(pe2, ctx)
    assert isinstance(res2, IndexSymbol)
    assert res2.name == "i'"

def test_indexed_symbol_mathrm_i():
    resolver = MaximaSemanticResolver()
    ctx = SymbolContext()
    
    # \mathrm{i} translated by tokenizer
    ie = IndexedSymbol(base=Symbol(name="mathrm"), indices=[IndexSymbol(name="i")])
    res = resolver.resolve(ie, ctx)
    assert isinstance(res, Symbol)
    assert res.name == "%i"
    
    ie2 = IndexedSymbol(base=Symbol(name="text"), indices=[IndexSymbol(name="i")])
    res2 = resolver.resolve(ie2, ctx)
    assert isinstance(res2, Symbol)
    assert res2.name == "%i"


# ---------------------------------------------------------------------------
# B-3: prime-as-derivative in Maxima context
# ---------------------------------------------------------------------------

def _make_prime_call(func_name: str, order: int, arg_expr):
    """Build FunctionCall(PrimeExpr^order(func_name), [arg_expr])."""
    inner = Symbol(name=func_name)
    wrapped = inner
    for _ in range(order):
        wrapped = PrimeExpr(expr=wrapped)
    return FunctionCall(func=wrapped, args=[arg_expr])


class TestPrimeAsDerivative:
    """In non-quantum Maxima sessions, f'(x) must resolve to Derivative(f(x), x)."""

    def setup_method(self):
        self.resolver = MaximaSemanticResolver()
        self.ctx = SymbolContext()

    def test_simple_variable_first_order(self):
        """f'(x) → diff(f(x), x)."""
        node = _make_prime_call("f", 1, Symbol(name="x"))
        result = self.resolver.resolve(node, self.ctx)
        assert isinstance(result, Derivative)
        assert result.vars[0][1] == 1
        assert result.vars[0][0].name == "x"

    def test_simple_variable_second_order(self):
        """f''(x) → diff(f(x), x, 2)."""
        node = _make_prime_call("f", 2, Symbol(name="x"))
        result = self.resolver.resolve(node, self.ctx)
        assert isinstance(result, Derivative)
        assert result.vars[0][1] == 2

    def test_complex_arg_scalar_excluded(self):
        """f'(x + a) where a is declared scalar → diff(f(x+a), x)."""
        from notebook.models.quantity import Quantity
        self.ctx.register("a", quantity=Quantity(name="a", kind="scalar"))
        arg = Add(args=[Symbol(name="x"), Symbol(name="a")])
        node = _make_prime_call("f", 1, arg)
        result = self.resolver.resolve(node, self.ctx)
        assert isinstance(result, Derivative)
        assert result.vars[0][0].name == "x"

    def test_complex_arg_two_free_vars_raises(self):
        """f'(x + y) with both x and y undeclared → PrimeDerivativeError."""
        arg = Add(args=[Symbol(name="x"), Symbol(name="y")])
        node = _make_prime_call("f", 1, arg)
        with pytest.raises(PrimeDerivativeError):
            self.resolver.resolve(node, self.ctx)

    def test_multi_arg_raises(self):
        """g'(x, y) → PrimeDerivativeError (partial variable ambiguous)."""
        node = FunctionCall(
            func=PrimeExpr(expr=Symbol(name="g")),
            args=[Symbol(name="x"), Symbol(name="y")],
        )
        with pytest.raises(PrimeDerivativeError):
            self.resolver.resolve(node, self.ctx)

    def test_numeric_arg_only(self):
        """f'(2*t) where t is the only free variable → diff(f(2t), t)."""
        from notebook.models.ir import Number
        arg = Mul(args=[Number(value=2), Symbol(name="t")])
        node = _make_prime_call("f", 1, arg)
        result = self.resolver.resolve(node, self.ctx)
        assert isinstance(result, Derivative)
        assert result.vars[0][0].name == "t"


class TestSpaceBeforeParenDisambiguation:
    """`f (x)` (space before paren) is a product f·(x) unless `f` is a known or
    declared function; `f(x)` (no space) is always a call.  The parser records
    the space on FunctionCall.spaced and the resolver decides using context."""

    def _gen(self, s, ctx):
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        return MaximaCodeGenerator().generate(
            MaximaSemanticResolver().resolve(SharedParser().parse(s), ctx))

    def test_scalar_space_paren_is_product(self):
        ctx = SymbolContext()
        ctx.register("a", Quantity(name="a", kind="scalar"))
        assert self._gen("a (x+1)", ctx) == "a * (x + 1)"

    def test_undeclared_space_paren_is_product(self):
        # No declaration → cannot be a call → multiplication.
        assert self._gen("g (y)", SymbolContext()) == "g * y"

    def test_number_space_paren_is_product(self):
        assert self._gen("2 (x+1)", SymbolContext()) == "2 * (x + 1)"

    def test_declared_function_space_paren_stays_call(self):
        ctx = SymbolContext()
        ctx.register("f", Quantity(name="f", kind="function"))
        assert self._gen("f (x)", ctx) == "f(x)"

    def test_builtin_function_space_paren_stays_call(self):
        assert self._gen("sin (x)", SymbolContext()) == "sin(x)"

    def test_no_space_is_always_call(self):
        # "a(x+1)" with no space is function-call syntax even for a scalar name.
        ctx = SymbolContext()
        ctx.register("a", Quantity(name="a", kind="scalar"))
        assert self._gen("a(x+1)", ctx) == "a(x + 1)"

    def test_parser_records_spaced_flag(self):
        from notebook.runtime.shared_parser import SharedParser
        spaced = SharedParser().parse("f (x)")
        plain = SharedParser().parse("f(x)")
        assert isinstance(spaced, FunctionCall) and spaced.spaced is True
        assert isinstance(plain, FunctionCall) and plain.spaced is False

    def test_julia_resolver_same_disambiguation(self):
        # The rule is shared (BaseSemanticResolver.callee_is_function), so the
        # Julia backend resolves space-before-paren identically to Maxima.
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        def jgen(s, ctx):
            return JuliaCodeGenerator().generate(
                JuliaSemanticResolver().resolve(SharedParser().parse(s), ctx))

        ctx = SymbolContext()
        ctx.register("a", Quantity(name="a", kind="scalar"))
        ctx.register("f", Quantity(name="f", kind="function"))
        assert jgen("a (x+1)", ctx) == "a * (x + 1)"   # scalar + space → product
        assert jgen("f (x)", ctx) == "f(x)"            # declared function → call
        assert jgen("sin (x)", ctx) == "sin(x)"        # builtin → call
        assert jgen("2 (x+1)", ctx) == "2 * (x + 1)"   # number → product

    def test_builtin_function_table_is_backend_specific(self):
        # Maxima and Julia have DIFFERENT builtin spellings, so the known-function
        # table for the space-disambiguation is per-backend (Backend Encapsulation
        # Rule): Maxima knows "conjugate", Julia knows "conj".  The OTHER backend's
        # spelling is unknown → treated as a product when a space precedes "(".
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator

        def mx(s):
            return MaximaCodeGenerator().generate(
                MaximaSemanticResolver().resolve(SharedParser().parse(s), SymbolContext()))

        def jl(s):
            return JuliaCodeGenerator().generate(
                JuliaSemanticResolver().resolve(SharedParser().parse(s), SymbolContext()))

        # Maxima spelling "conjugate" is a call in Maxima, a product in Julia.
        assert mx("conjugate (x)") == "conjugate(x)"
        assert jl("conjugate (x)") == "conjugate * x"
        # Julia spelling "conj" is a call in Julia, a product in Maxima.
        assert jl("conj (x)") == "conj(x)"
        assert mx("conj (x)") == "conj * x"


# ---------------------------------------------------------------------------
# R-5 (SR-3): declared Delta must not be hijacked into Laplacian — the same
# guard JuliaSemanticResolver has had (test_julia_semantic_resolver_edge.py);
# before 2026-07-03 the Maxima resolver ignored the declaration.
# ---------------------------------------------------------------------------

class TestDeltaLaplacianGuard:
    def test_delta_as_laplacian_without_context(self):
        """Delta f with no declaration → Laplacian (math convention preserved)."""
        from notebook.models.ir import Laplacian, Mul, Symbol
        resolver = MaximaSemanticResolver()
        ctx = SymbolContext()
        mul = Mul(args=[Symbol(name="Delta"), Symbol(name="f")])
        res = resolver._resolve_node(mul, ctx)
        assert isinstance(res, Laplacian)
        assert isinstance(res.expr, Symbol) and res.expr.name == "f"

    def test_delta_as_symbol_when_registered_mul(self):
        """Declared Delta (gap function / finite difference) → multiplication."""
        from notebook.models.ir import Laplacian, Mul, Symbol
        from notebook.models.quantity import Quantity
        resolver = MaximaSemanticResolver()
        ctx = SymbolContext()
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
        mul = Mul(args=[Symbol(name="Delta"), Symbol(name="psi")])
        res = resolver._resolve_node(mul, ctx)
        assert not isinstance(res, Laplacian), (
            f"declared Delta must NOT become Laplacian; got {type(res)}"
        )

    def test_delta_juxtaposition_respects_declaration(self):
        """'Delta f' via SharedParser (Juxtaposition path) honours the declaration."""
        from notebook.models.ir import Laplacian, Mul
        from notebook.models.quantity import Quantity
        from notebook.runtime.shared_parser import SharedParser
        parser, resolver = SharedParser(), MaximaSemanticResolver()

        undeclared = resolver.resolve(parser.parse("Delta f"), SymbolContext())
        assert isinstance(undeclared, Laplacian)

        ctx = SymbolContext()
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
        declared = resolver.resolve(parser.parse("Delta f"), ctx)
        assert isinstance(declared, Mul)

    def test_explicit_laplacian_name_unaffected_by_declaration(self):
        """laplacian(f) stays Laplacian even if a symbol 'Delta' is declared."""
        from notebook.models.ir import Laplacian
        from notebook.models.quantity import Quantity
        from notebook.runtime.shared_parser import SharedParser
        parser, resolver = SharedParser(), MaximaSemanticResolver()
        ctx = SymbolContext()
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
        res = resolver.resolve(parser.parse("laplacian(f)"), ctx)
        assert isinstance(res, Laplacian)


# ---------------------------------------------------------------------------
# A function name with a single-char subscript immediately followed by a
# call, e.g. "bessel_j(0,x)": SharedParser is semantics-free (CLAUDE.md), so
# it reads the trailing "(0,x)" as attached to the LAST identifier before it
# ("j"), producing Subscript(base=bessel, sub=FunctionCall(j, [0,x])) rather
# than a plain index.  Before the fix that FunctionCall flowed into
# _extract_indices (which only accepts IndexSymbol-shaped nodes) and crashed
# with a raw pydantic ValidationError instead of the clean
# numeric.unsupported_function error the numpy-whitelist gate is supposed to
# raise (R-3 fail-fast).  Found via the numeric-layer S-ac practice loop
# (2026-07-10): a real coder LLM reply that used bessel_j(0,x) crashed engine
# registration outright, never reaching the "unsupported function" message a
# researcher could act on.
# ---------------------------------------------------------------------------

class TestSubscriptFunctionCallReconstruction:
    def _resolve(self, s, ctx=None):
        from notebook.runtime.shared_parser import SharedParser
        return MaximaSemanticResolver().resolve(
            SharedParser().parse(s), ctx or SymbolContext())

    def test_single_char_subscript_call_becomes_function_call(self):
        res = self._resolve("bessel_j(0,x)")
        assert isinstance(res, FunctionCall)
        assert isinstance(res.func, Symbol) and res.func.name == "bessel_j"
        assert len(res.args) == 2

    def test_reconstructed_call_reports_clean_unsupported_function_error(self):
        # the actual failure mode found in the eval: this must reach the
        # numpy-whitelist gate's clean error, not crash registration.
        from notebook.numeric.expr_to_numpy import canonical_to_numpy
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError) as ei:
            canonical_to_numpy("bessel_j(0,x)", SymbolContext())
        assert ei.value.i18n_key == "numeric.unsupported_function"

    def test_registered_combined_name_still_wins_over_reconstruction(self):
        # a symbol actually named "a_f" (registered) must still resolve as
        # that symbol when bare — reconstruction only applies when the
        # subscript's sub is ITSELF a call, which a bare registered name never is.
        ctx = SymbolContext()
        ctx.register("a_f", Quantity(name="a_f", kind="scalar"))
        res = self._resolve("a_f", ctx)
        assert isinstance(res, Symbol) and res.name == "a_f"

    def test_bare_multichar_subscript_unaffected(self):
        # base_sub reconstruction only fires when the subscript's sub is
        # ITSELF a FunctionCall (i.e. immediately followed by a call) — a
        # bare "H_BCS" (no call) never takes that shape, so this path is not
        # reached and the pre-existing, unrelated behavior stands unchanged:
        # an unregistered multi-char subscript in an operation resolves to
        # IndexedSymbol (CLAUDE.md "undeclared subscript in an operation →
        # index"), not a plain Symbol.
        from notebook.models.ir import IndexedSymbol
        res = self._resolve("H_BCS")
        assert isinstance(res, IndexedSymbol)
        assert res.base.name == "H" and res.indices[0].name == "BCS"

    def test_registration_does_not_change_the_reconstructed_outcome(self):
        # "my_func(x)" hits the SAME subscript+call surface shape as
        # "bessel_j(0,x)" (SharedParser has no notion of function names) —
        # reconstruction fires regardless of whether "my_func" happens to be
        # registered, and correctly either way: a called base_sub shape is
        # unambiguously that function, registered or not.
        ctx = SymbolContext()
        ctx.register("my_func", Quantity(name="my_func", kind="function"))
        res = self._resolve("my_func(x)", ctx)
        assert isinstance(res, FunctionCall)
        assert isinstance(res.func, Symbol) and res.func.name == "my_func"
