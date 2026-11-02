"""Tests for MaximaIRCompiler.parse() and ExprGenerator.generate()."""
import pytest
from notebook.models.ir import (
    Add, Decorated, Derivative, Equation, FunctionCall, Mul, Number, Superscript, Pow, Symbol,
)
from notebook.runtime.maxima_ir_compiler import MaximaIRCompiler
from notebook.runtime.expr_generator import ExprGenerator


@pytest.fixture
def c():
    return MaximaIRCompiler()


@pytest.fixture
def g():
    return ExprGenerator()


# ---------------------------------------------------------------------------
# parse() — atoms
# ---------------------------------------------------------------------------

class TestParseAtoms:
    def test_bare_symbol(self, c):
        ir = c.parse("x")
        assert isinstance(ir, Symbol)
        assert ir.name == "x"

    def test_integer(self, c):
        ir = c.parse("42")
        assert isinstance(ir, Number)
        assert ir.value == 42

    def test_float(self, c):
        ir = c.parse("3.14")
        assert isinstance(ir, Number)
        assert abs(ir.value - 3.14) < 1e-9

    def test_maxima_pi(self, c):
        ir = c.parse("%pi")
        assert isinstance(ir, Symbol)
        assert ir.name == "%pi"

    def test_maxima_e(self, c):
        ir = c.parse("%e")
        assert isinstance(ir, Symbol)
        assert ir.name == "%e"

    def test_maxima_inf(self, c):
        ir = c.parse("inf")
        assert isinstance(ir, Symbol)
        assert ir.name == "inf"

    def test_empty_returns_none(self, c):
        assert c.parse("") is None
        assert c.parse("   ") is None


# ---------------------------------------------------------------------------
# parse() — function calls
# ---------------------------------------------------------------------------

class TestParseFunctionCall:
    def test_single_arg(self, c):
        ir = c.parse("sin(x)")
        assert isinstance(ir, FunctionCall)
        assert isinstance(ir.func, Symbol) and ir.func.name == "sin"
        assert len(ir.args) == 1
        assert isinstance(ir.args[0], Symbol) and ir.args[0].name == "x"

    def test_two_args(self, c):
        ir = c.parse("f(x, t)")
        assert isinstance(ir, FunctionCall)
        assert ir.func.name == "f"
        assert len(ir.args) == 2

    def test_nested_call(self, c):
        ir = c.parse("f(g(x))")
        assert isinstance(ir, FunctionCall)
        inner = ir.args[0]
        assert isinstance(inner, FunctionCall)
        assert inner.func.name == "g"


# ---------------------------------------------------------------------------
# parse() — equations
# ---------------------------------------------------------------------------

class TestParseEquation:
    def test_simple_eq(self, c):
        ir = c.parse("y = x^2")
        assert isinstance(ir, Equation)
        assert isinstance(ir.lhs, Symbol) and ir.lhs.name == "y"
        assert isinstance(ir.rhs, (Pow, Superscript))

    def test_function_eq(self, c):
        ir = c.parse("f(x) = sin(x)")
        assert isinstance(ir, Equation)
        assert isinstance(ir.lhs, FunctionCall)
        assert ir.lhs.func.name == "f"

    def test_define_not_parsed_as_equation(self, c):
        # := should NOT produce Equation
        ir = c.parse("f(x) := sin(x)")
        # Falls back to Symbol or something; not Equation
        assert not isinstance(ir, Equation)


# ---------------------------------------------------------------------------
# parse() — inequalities (relations with op != "=")
# ---------------------------------------------------------------------------

class TestParseInequality:
    def test_le(self, c):
        ir = c.parse("2*x_A + x_B <= 100")
        assert isinstance(ir, Equation)
        assert ir.op == "<="
        assert isinstance(ir.rhs, Number) and ir.rhs.value == 100

    def test_ge(self, c):
        ir = c.parse("2*x_A + x_B >= 100")
        assert isinstance(ir, Equation)
        assert ir.op == ">="

    def test_lt(self, c):
        ir = c.parse("x < 5")
        assert isinstance(ir, Equation)
        assert ir.op == "<"

    def test_gt(self, c):
        ir = c.parse("x > 5")
        assert isinstance(ir, Equation)
        assert ir.op == ">"

    def test_plain_equation_op_defaults_to_eq(self, c):
        ir = c.parse("y = x^2")
        assert isinstance(ir, Equation)
        assert ir.op == "="

    def test_chained_relation_rejected(self, c):
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError):
            c.parse("0 <= x <= 100")

    def test_stray_relational_char_rejected(self, c):
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError):
            c.parse("f(x<y) <= 100")


# ---------------------------------------------------------------------------
# parse() — compound expressions
# ---------------------------------------------------------------------------

class TestParseCompound:
    def test_add_two_symbols(self, c):
        ir = c.parse("x + y")
        assert isinstance(ir, Add)
        assert len(ir.args) == 2
        assert all(isinstance(a, Symbol) for a in ir.args)

    def test_add_three(self, c):
        ir = c.parse("a + b + c")
        assert isinstance(ir, Add)
        assert len(ir.args) == 3

    def test_sub_becomes_add_neg(self, c):
        ir = c.parse("x - y")
        assert isinstance(ir, Add)
        assert len(ir.args) == 2
        # second term should be Mul([-1, y])
        neg_term = ir.args[1]
        assert isinstance(neg_term, Mul)
        assert isinstance(neg_term.args[0], Number) and neg_term.args[0].value == -1

    def test_mul_two(self, c):
        ir = c.parse("x * y")
        assert isinstance(ir, Mul)
        assert len(ir.args) == 2

    def test_pow(self, c):
        ir = c.parse("x^2")
        assert isinstance(ir, (Pow, Superscript))
        assert isinstance(ir.base, Symbol) and ir.base.name == "x"
        assert isinstance(ir.script, Number) and ir.script.value == 2

    def test_pow_starstar(self, c):
        ir = c.parse("x**2")
        assert isinstance(ir, (Pow, Superscript))

    def test_pow_right_assoc(self, c):
        """a^b^c = a^(b^c), as in Maxima, Julia and ordinary notation.

        This test used to pin the LEFT-associative reading — recorded when the
        parser happened to do that (commit e07478a, Notation Bridge P1), with no
        rationale anywhere.  It was wrong: the IR said (a^b)^c while every CAS
        says a^(b^c).  It went unnoticed because the code generator re-emitted
        `a^b^c` unparenthesised, so the CAS re-read it right-associatively and
        produced the right answer from a wrong AST.  Parenthesising a Pow base —
        required so that `k/r^2` emits `k*(r^2)^-1` and not `k*r^2^-1`
        (= k*sqrt(r)) — removed that cover.  See
        tests/test_power_division_codegen.py and open_items AUDIT-POW-DENOM.
        """
        ir = c.parse("a^b^c")
        assert isinstance(ir, (Pow, Superscript))
        assert isinstance(ir.script, (Pow, Superscript)), "read as (a^b)^c"
        assert isinstance(ir.base, Symbol) and ir.base.name == "a"

    def test_unary_minus_number(self, c):
        ir = c.parse("-3")
        assert isinstance(ir, Number)
        assert ir.value == -3

    def test_unary_minus_symbol(self, c):
        ir = c.parse("-x")
        assert isinstance(ir, Mul)
        assert isinstance(ir.args[0], Number) and ir.args[0].value == -1
        assert isinstance(ir.args[1], Symbol) and ir.args[1].name == "x"

    def test_division(self, c):
        ir = c.parse("a / b")
        assert isinstance(ir, Mul)
        # a * b^-1
        assert isinstance(ir.args[1], (Pow, Superscript))
        assert isinstance(ir.args[1].exponent, Number)
        assert ir.args[1].exponent.value == -1

    def test_parentheses(self, c):
        ir = c.parse("(a + b) * c")
        assert isinstance(ir, Mul)
        assert isinstance(ir.args[0], Add)

    def test_function_in_add(self, c):
        ir = c.parse("sin(x) + cos(y)")
        assert isinstance(ir, Add)
        assert isinstance(ir.args[0], FunctionCall)
        assert isinstance(ir.args[1], FunctionCall)

    def test_polynomial(self, c):
        ir = c.parse("2*x^2 + 3*x + 1")
        assert isinstance(ir, Add)
        assert len(ir.args) == 3

    def test_complex_nested(self, c):
        ir = c.parse("a * (b + c)^2")
        assert isinstance(ir, Mul)
        pow_node = ir.args[1]
        assert isinstance(pow_node, (Pow, Superscript))
        assert isinstance(pow_node.base, Add)


# ---------------------------------------------------------------------------
# parse() — unevaluated derivative
# ---------------------------------------------------------------------------

class TestParseUneval:
    def test_diff_simple(self, c):
        ir = c.parse("'diff(f(t), t, 1)")
        assert isinstance(ir, Derivative)
        assert isinstance(ir.expr, FunctionCall)
        assert len(ir.vars) == 1
        var, order = ir.vars[0]
        assert var.name == "t"
        assert order == 1

    def test_diff_order2(self, c):
        ir = c.parse("'diff(q(t), t, 2)")
        assert isinstance(ir, Derivative)
        _, order = ir.vars[0]
        assert order == 2


# ---------------------------------------------------------------------------
# generate() — round-trip and formatting
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_symbol(self, g):
        assert g.generate(Symbol(name="x")) == "x"

    def test_number_int(self, g):
        assert g.generate(Number(value=3)) == "3"

    def test_number_float_whole(self, g):
        # 2.0 → "2" not "2.0"
        assert g.generate(Number(value=2.0)) == "2"

    def test_add(self, g):
        ir = Add(args=[Symbol(name="a"), Symbol(name="b")])
        assert g.generate(ir) == "a + b"

    def test_sub_via_neg(self, g):
        ir = Add(args=[Symbol(name="a"), Mul(args=[Number(value=-1), Symbol(name="b")])])
        assert g.generate(ir) == "a - b"

    def test_mul(self, g):
        ir = Mul(args=[Symbol(name="a"), Symbol(name="b")])
        assert g.generate(ir) == "a * b"

    def test_neg(self, g):
        ir = Mul(args=[Number(value=-1), Symbol(name="x")])
        assert g.generate(ir) == "-x"

    def test_pow(self, g):
        ir = Pow(base=Symbol(name="x"), exponent=Number(value=2))
        assert g.generate(ir) == "x^2"

    def test_func_call(self, g):
        ir = FunctionCall(func=Symbol(name="sin"), args=[Symbol(name="x")])
        assert g.generate(ir) == "sin(x)"

    def test_equation(self, g):
        ir = Equation(lhs=Symbol(name="y"), rhs=Symbol(name="x"))
        assert g.generate(ir) == "y=x"

    def test_inequality_le(self, g):
        ir = Equation(lhs=Symbol(name="y"), rhs=Symbol(name="x"), op="<=")
        assert g.generate(ir) == "y<=x"

    def test_inequality_roundtrip(self, c, g):
        ir = c.parse("2*x_A + x_B <= 100")
        assert g.generate(ir) == "2 * x_A + x_B<=100"

    def test_add_parenthesised_in_mul(self, g):
        ir = Mul(args=[
            Symbol(name="a"),
            Add(args=[Symbol(name="b"), Symbol(name="c")]),
        ])
        assert g.generate(ir) == "a * (b + c)"


# ---------------------------------------------------------------------------
# Round-trip: parse → generate → parse
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("expr_str", [
        "x",
        "sin(x)",
        "x + y",
        "x * y",
        "x^2",
        "sin(x) + cos(y)",
        "2*x^2 + 3*x + 1",
        "(a + b) * c",
    ])
    def test_parse_generate_yields_equivalent(self, expr_str):
        c = MaximaIRCompiler()
        ir = c.parse(expr_str)
        assert ir is not None
        # The generated string should re-parse to the same structure
        out = c.generate(ir)
        ir2 = c.parse(out)
        assert ir2 is not None
        # Both should have the same type at the top level
        assert type(ir) == type(ir2)


# ---------------------------------------------------------------------------
# B-6: Decorated IR node (SharedParser + ExprGenerator round-trip)
# ---------------------------------------------------------------------------

class TestDecorated:
    def test_hat_parses_to_decorated(self, c):
        ir = c.parse("H_hat")
        assert isinstance(ir, Decorated)
        assert isinstance(ir.base, Symbol)
        assert ir.base.name == "H"
        assert ir.decoration == "hat"

    def test_tilde_parses_to_decorated(self, c):
        ir = c.parse("omega_tilde")
        assert isinstance(ir, Decorated)
        assert ir.base.name == "omega"
        assert ir.decoration == "tilde"

    def test_bar_parses_to_decorated(self, c):
        ir = c.parse("x_bar")
        assert isinstance(ir, Decorated)
        assert ir.base.name == "x"
        assert ir.decoration == "bar"

    def test_dot_parses_to_decorated(self, c):
        ir = c.parse("x_dot")
        assert isinstance(ir, Decorated)
        assert ir.decoration == "dot"

    def test_ddot_parses_to_decorated(self, c):
        ir = c.parse("q_ddot")
        assert isinstance(ir, Decorated)
        assert ir.decoration == "ddot"

    def test_subscript_base_hat(self, c):
        # H_0_hat → Decorated(Subscript(H, 0), "hat")
        # base is Subscript, not a flat Symbol named "H_0"
        ir = c.parse("H_0_hat")
        assert isinstance(ir, Decorated)
        assert ir.decoration == "hat"
        from notebook.models.ir import Subscript
        assert isinstance(ir.base, Subscript)

    def test_generate_roundtrips_to_canonical(self, c, g):
        # Decorated → generate → same canonical suffix string
        ir = c.parse("H_hat")
        assert g.generate(ir) == "H_hat"

    def test_generate_subscript_base_roundtrip(self, c, g):
        # Decorated(Subscript(H, 0), "hat") → "H_0_hat"
        ir = c.parse("H_0_hat")
        result = g.generate(ir)
        assert result == "H_0_hat"

    def test_plain_identifier_not_decorated(self, c):
        # H_0 → Subscript(H, 0), not Decorated
        ir = c.parse("H_0")
        assert not isinstance(ir, Decorated)

    def test_decorated_in_expression(self, c, g):
        ir = c.parse("H_hat + V_hat")
        assert isinstance(ir, Add)
        assert isinstance(ir.args[0], Decorated)
        assert isinstance(ir.args[1], Decorated)
        assert g.generate(ir) == "H_hat + V_hat"
