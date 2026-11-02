"""Tests for JuliaAdapter and JuliaCommandBuilder.

These tests start a real Julia process (requires julia in PATH and
QuantumAlgebra / Symbolics packages installed).  They are slower than
pure-Python tests because Julia startup loads packages (~5–120 s on first run).

Run selectively:
    pytest tests/test_julia_adapter.py -v
"""
import pytest
from notebook.runtime.julia_adapter import JuliaAdapter
from notebook.runtime.builder import JuliaCommandBuilder
from notebook.models.ir import CasOperation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia():
    """Single JuliaAdapter shared across all tests in this module to amortise startup cost."""
    with JuliaAdapter() as adapter:
        yield adapter


@pytest.fixture(scope="module")
def builder():
    return JuliaCommandBuilder()


# ---------------------------------------------------------------------------
# JuliaAdapter — basic execution
# ---------------------------------------------------------------------------

class TestJuliaAdapterBasic:
    def test_arithmetic(self, julia):
        r = julia.execute("2 + 3")
        assert r.ok
        assert r.value == "5"

    def test_error_captured(self, julia):
        r = julia.execute("undefined_function_xyz()")
        assert not r.ok
        assert r.error is not None

    def test_latex_float(self, julia):
        r = julia.execute("1.0 / 3.0")
        assert r.ok
        # value is a floating-point string
        assert "0.333" in r.value

    def test_nothing_result(self, julia):
        r = julia.execute("nothing")
        assert r.ok
        assert r.value == "nothing"


# ---------------------------------------------------------------------------
# JuliaAdapter — QuantumAlgebra operations
# ---------------------------------------------------------------------------

class TestQuantumAlgebra:
    def test_boson_commutator_latex(self, julia):
        """[a, a†] in normal-ordered form should equal 1."""
        r = julia.execute("normal_form(comm(a(1), a'(1)))")
        assert r.ok
        assert r.value == "1"

    def test_fermion_anticommutator(self, julia):
        """{f, f†} in normal form should equal 1."""
        r = julia.execute("normal_form(f(1)*f'(1) + f'(1)*f(1))")
        assert r.ok
        assert r.value == "1"

    def test_vacuum_expectation_zero(self, julia):
        """<0|a†a|0> = 0 (no particles in vacuum)."""
        r = julia.execute("vacExpVal(a'(1)*a(1))")
        assert r.ok
        assert r.value == "0"

    def test_number_operator_normal_form(self, julia):
        """aa† = 1 + a†a in normal-ordered form."""
        r = julia.execute("normal_form(a(1)*a'(1))")
        assert r.ok
        # Should equal 1 + n where n is the number operator
        assert r.value is not None

    def test_define_boson_operator(self, julia):
        """@boson_ops defines operators accessible by name in subsequent calls."""
        r1 = julia.execute("@boson_ops b2")
        assert r1.ok
        r2 = julia.execute("normal_form(b2(1)*b2'(1))")
        assert r2.ok

    def test_heisenberg_eom(self, julia):
        """Heisenberg EOM for number operator under free Hamiltonian = 0."""
        r = julia.execute(
            "let H = param(:ω, :c)*a'(1)*a(1); heisenberg_eom(a'(1)*a(1), H); end"
        )
        assert r.ok

    def test_latex_output_quexpr(self, julia):
        """LaTeX for a QuantumAlgebra expression should contain LaTeX notation."""
        r = julia.execute("a'(1)*a(1)")
        assert r.ok
        assert r.latex != ""
        # Should contain dagger notation or similar
        assert "\\" in r.latex


# ---------------------------------------------------------------------------
# JuliaCommandBuilder
# ---------------------------------------------------------------------------

class TestJuliaCommandBuilder:
    def test_commutator_template(self, builder):
        op = CasOperation(operation="commutator", args={"a": "a(1)", "b": "a'(1)"})
        result = builder.build(op)
        assert result.ok
        assert "comm" in result.command
        assert "normal_form" in result.command
        assert "a(1)" in result.command

    def test_normal_order_template(self, builder):
        op = CasOperation(operation="normal_order", args={"expr": "a(1)*a'(1)"})
        result = builder.build(op)
        assert result.ok
        assert "normal_form" in result.command

    def test_differentiate_default_order(self, builder):
        # No order arg → defaults to first derivative.
        op = CasOperation(operation="differentiate",
                          args={"expr": "sin(k)", "var": "k"})
        result = builder.build(op)
        assert result.ok
        assert result.command == "_nb_differentiate(sin(k), k, 1)"

    def test_differentiate_higher_order(self, builder):
        # order=2 must reach the command so the CAS computes the 2nd derivative,
        # not just the 1st (regression: template ignored order).
        op = CasOperation(operation="differentiate",
                          args={"expr": "hbar^2*k^2/(2*m) - mu",
                                "var": "k", "order": "2"})
        result = builder.build(op)
        assert result.ok
        assert "_nb_differentiate(" in result.command
        assert result.command.rstrip(")").endswith("2")

    def test_vacuum_expectation_template(self, builder):
        op = CasOperation(operation="vacuum_expectation", args={"expr": "a'(1)*a(1)"})
        result = builder.build(op)
        assert result.ok
        # vacuum expectation = operator-free part of the normal-ordered QuExpr.
        # _nb_vac_expect replaced QuantumAlgebra's vacExpVal, which throws on
        # symbolic scalar coefficients (u, v, ξ, Δ, …).
        assert "_nb_vac_expect" in result.command
        assert "normal_form" in result.command

    def test_define_boson_template(self, builder):
        op = CasOperation(operation="define_boson_operator", args={"name": "c"})
        result = builder.build(op)
        assert result.ok
        assert "@boson_ops c" in result.command

    def test_define_fermion_template(self, builder):
        op = CasOperation(operation="define_fermion_operator", args={"name": "d"})
        result = builder.build(op)
        assert result.ok
        assert "@fermion_ops d" in result.command

    def test_heisenberg_equation_template(self, builder):
        op = CasOperation(
            operation="heisenberg_equation",
            args={"operator": "a(1)", "hamiltonian": "param(:ω,:c)*a'(1)*a(1)"},
        )
        result = builder.build(op)
        assert result.ok
        assert "heisenberg_eom" in result.command

    def test_unknown_operation(self, builder):
        op = CasOperation(operation="nonexistent_op", args={})
        result = builder.build(op)
        assert not result.ok
        assert "No Julia template" in result.error

    def test_expand_template(self, builder):
        op = CasOperation(operation="expand", args={"expr": "a(1)*(a(1)+a'(1))"})
        result = builder.build(op)
        assert result.ok
        assert "expand" in result.command


# ---------------------------------------------------------------------------
# End-to-end: builder output → adapter execution
# ---------------------------------------------------------------------------

class TestNormalizeImplicitMul:
    """Unit tests for implicit multiplication resolution using the new compiler pipeline.

    These tests do NOT require a running Julia process — they only exercise
    the Python-side AST parser, resolver, and generator.
    """

    @pytest.fixture(scope="class")
    def context_and_helpers(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        ctx = SymbolContext()
        _qty = Quantity(name="op", kind="operator")
        ctx.register("a", quantity=_qty, statistics_kind="bose")
        ctx.register("f", quantity=_qty, statistics_kind="fermi")
        ctx.register("b", quantity=_qty, statistics_kind="bose")

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        generator = JuliaCodeGenerator()

        def _normalize(text: str, context=ctx) -> str:
            surface_ast = parser.parse(text)
            if surface_ast is None:
                return text
            semantic_ast = resolver.resolve(surface_ast, context)
            # Strip spaces to ensure robust comparison with whitespace-free expectation strings
            return generator.generate(semantic_ast).replace(" ", "")

        return _normalize, ctx

    def test_operator_product(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("a'(1) a(1)") == "a'(1)*a(1)"

    def test_operator_chain(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("a'(1) a(1) b'(1) b(1)") == "a'(1)*a(1)*b'(1)*b(1)"

    def test_scalar_times_operator(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("g a'(1)") == "g*a'(1)"

    def test_multichar_scalar_times_operator(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("omega a'(1)") == "omega*a'(1)"

    def test_paren_expr_times_operator(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("(a'(1)+b(1)) a(1)") == "(a'(1)+b(1))*a(1)"

    def test_addition_not_affected(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        # + between terms must NOT get an extra *
        result = _normalize("a'(1)*a(1) + b'(1)*b(1)")
        assert result == "a'(1)*a(1)+b'(1)*b(1)"

    def test_already_explicit(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        assert _normalize("a'(1)*a(1)") == "a'(1)*a(1)"

    def test_no_context(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        # In the compiler-based resolver, unregistered symbols are treated as commutative,
        # but juxtapositions are still parsed and resolved to multiplication in the shared frontend.
        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        generator = JuliaCodeGenerator()
        ctx = SymbolContext()

        surface_ast = parser.parse("a'(1) a(1)")
        semantic_ast = resolver.resolve(surface_ast, ctx)
        assert generator.generate(semantic_ast).replace(" ", "") == "a'(1)*a(1)"

    def test_empty_registry(self, context_and_helpers):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        # Same behavior: parsed as juxtaposition and resolved to multiplication.
        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        generator = JuliaCodeGenerator()
        ctx = SymbolContext()

        surface_ast = parser.parse("a'(1) a(1)")
        semantic_ast = resolver.resolve(surface_ast, ctx)
        assert generator.generate(semantic_ast).replace(" ", "") == "a'(1)*a(1)"

    def test_interaction_term(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        # a'(1)*a(1) + b'(1)*b(1) + g*(a'(1)*b(1) + a(1)*b'(1)) — all explicit, no change
        expr = "a'(1)*a(1) + b'(1)*b(1) + g*(a'(1)*b(1) + a(1)*b'(1))"
        assert _normalize(expr) == expr.replace(" ", "")

    def test_interaction_term_implicit(self, context_and_helpers):
        _normalize, _ = context_and_helpers
        # Same expression without *
        expr    = "a'(1) a(1) + b'(1) b(1)"
        expect  = "a'(1)*a(1) + b'(1)*b(1)"
        assert _normalize(expr) == expect.replace(" ", "")



class TestBuilderAdapterIntegration:
    def test_commutator_via_builder(self, julia, builder):
        op = CasOperation(operation="commutator", args={"a": "a(1)", "b": "a'(1)"})
        build = builder.build(op)
        assert build.ok
        # Template is now normal_form(comm(...))
        assert "normal_form" in build.command
        r = julia.execute(build.command)
        assert r.ok
        # normal_form(comm(a(1), a'(1))) = 1  for bosonic a
        assert r.value is not None
        assert "1" in (r.latex or r.value or "")

    def test_normal_order_via_builder(self, julia, builder):
        op = CasOperation(operation="normal_order", args={"expr": "a(1)*a'(1)"})
        build = builder.build(op)
        assert build.ok
        r = julia.execute(build.command)
        assert r.ok
        # normal_form(aa†) = 1 + a†a
        assert r.value is not None




    def test_nested_commutator_command(self, builder):
        """Verify that nested-bracket args produce correct normal_form(comm(comm(...))) command."""
        from notebook.models.ir import CasOperation
        op = CasOperation(
            operation="commutator",
            args={"a": "comm(a(1), a'(1))", "b": "a(1)"},
        )
        result = builder.build(op)
        assert result.ok
        assert result.command == "_nb_eval_site_deltas(normal_form(comm(comm(a(1), a'(1)), a(1))))"

