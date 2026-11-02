from notebook.models.ir import CasOperation
"""Tests for MaximaCommandBuilder.

Builder is purely functional: same inputs → same command string.
No Maxima process is required.
"""
import pytest
from notebook.runtime.builder import MaximaCommandBuilder, BuildResult


@pytest.fixture(scope="module")
def b():
    return MaximaCommandBuilder()


EXPR = {"E1": "x^2-1", "E2": "x-1", "E3": "(x^2-1)/(x-1)"}


# ---------------------------------------------------------------------------
# Supported operations
# ---------------------------------------------------------------------------

class TestSupportedOperations:
    def test_operations_loaded(self, b):
        assert len(b.supported_operations) > 0

    def test_mvp_operations_present(self, b):
        for op in ["expand", "factor", "simplify", "rational_simplify",
                   "substitute", "differentiate", "integrate_indefinite",
                   "solve_equation", "taylor_expand", "to_latex"]:
            assert op in b.supported_operations, f"'{op}' missing"


# ---------------------------------------------------------------------------
# Algebraic rewrite commands
# ---------------------------------------------------------------------------

class TestAlgebraicRewrite:
    def test_expand_with_expr_id(self, b):
        r = b.build(CasOperation(operation="expand", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "expand(x^2-1)"

    def test_expand_inline(self, b):
        r = b.build(CasOperation(operation="expand", args={"expr": "(x+1)^2"}))
        assert r.ok
        assert r.command == "expand((x+1)^2)"

    def test_factor(self, b):
        r = b.build(CasOperation(operation="factor", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "factor(x^2-1)"

    def test_simplify(self, b):
        # General simplify = rational + trig (SIMP-UNIFY): applies sin²+cos²=1 so
        # the `簡約` keyword is backend-agnostic.  trigsimp is a no-op on trig-free
        # input, so this still reduces (x^2-1)/(x-1) → x+1 in practice.
        r = b.build(CasOperation(operation="simplify", args={"expr": "(x^2-1)/(x-1)"}))
        assert r.ok
        assert r.command == "trigsimp(ratsimp((x^2-1)/(x-1)))"

    def test_rational_simplify_is_ratsimp_only(self, b):
        # rational_simplify stays ratsimp-only (no trig) — the escape hatch for
        # users who do NOT want trig rewriting.  Distinct from simplify now.
        r = b.build(CasOperation(operation="rational_simplify", args={"expr": "(x^2-1)/(x-1)"}))
        assert r.ok
        assert r.command == "ratsimp((x^2-1)/(x-1))"

    def test_substitute(self, b):
        r = b.build(CasOperation(operation="substitute", args={"expr": "x^2-1", "lhs": "x", "rhs": "2"}))
        assert r.ok
        assert r.command == "subst(2,x,x^2-1)"

    def test_substitute_expr_id_in_lhs(self, b):
        r = b.build(CasOperation(operation="substitute", args={"expr": "x^2-1", "lhs": "x-1", "rhs": "0"}))
        assert r.ok
        assert r.command == "subst(0,x-1,x^2-1)"

    def test_collect(self, b):
        r = b.build(CasOperation(operation="collect", args={"expr": "x^2-1", "var": "x"}))
        assert r.ok
        assert r.command == "collectterms(x^2-1,x)"

    def test_trig_simplify(self, b):
        r = b.build(CasOperation(operation="trig_simplify", args={"expr": "sin(x)^2+cos(x)^2"}))
        assert r.ok
        assert r.command == "trigsimp(sin(x)^2+cos(x)^2)"


# ---------------------------------------------------------------------------
# Calculus commands
# ---------------------------------------------------------------------------

class TestCalculus:
    def test_differentiate(self, b):
        r = b.build(CasOperation(operation="differentiate", args={"expr": "x^2-1", "var": "x", "order": "1"}))
        assert r.ok
        assert r.command == "diff(x^2-1,x,1)"

    def test_integrate_indefinite(self, b):
        r = b.build(CasOperation(operation="integrate_indefinite", args={"expr": "x^2-1", "var": "x"}))
        assert r.ok
        assert r.command == "integrate(x^2-1,x)"

    def test_integrate_definite(self, b):
        r = b.build(CasOperation(operation="integrate_definite", args={"expr": "x^2", "var": "x", "lower": "0", "upper": "1"}))
        assert r.ok
        assert r.command == "integrate(x^2,x,0,1)"

    def test_limit(self, b):
        r = b.build(CasOperation(operation="limit", args={"expr": "sin(x)/x", "var": "x", "point": "0"}))
        assert r.ok
        assert r.command == "limit(sin(x)/x,x,0)"

    def test_taylor_expand(self, b):
        r = b.build(CasOperation(operation="taylor_expand", args={"expr": "exp(x)", "var": "x", "point": "0", "order": "3"}))
        assert r.ok
        assert r.command == "ratdisrep(taylor(exp(x),x,0,3))"

    def test_taylor_expand_sin(self, b):
        # series_expand was merged into taylor_expand (point is now optional)
        r = b.build(CasOperation(operation="taylor_expand", args={"expr": "sin(x)", "var": "x", "point": "0", "order": "5"}))
        assert r.ok
        assert "sin(x)" in r.command and "5" in r.command


# ---------------------------------------------------------------------------
# Equation solving commands
# ---------------------------------------------------------------------------

class TestSolving:
    def test_solve_equation(self, b):
        r = b.build(CasOperation(operation="solve_equation", args={"equation": "x^2-4=0", "var": "x"}))
        assert r.ok
        assert r.command == "solve(x^2-4=0,x)"

    def test_solve_equation_with_expr_id(self, b):
        lookup = {"E5": "x^2-4=0"}
        r = b.build(CasOperation(operation="solve_equation", args={"equation": "x^2-4=0", "var": "x"}))
        assert r.ok
        assert r.command == "solve(x^2-4=0,x)"

    def test_solve_system(self, b):
        r = b.build(CasOperation(operation="solve_system", args={"equations": ["2*x+y=3", "x-y=1"], "vars": ["x", "y"]}))
        assert r.ok
        assert r.command == "solve([2*x+y=3, x-y=1],[x, y])"

    def test_solve_system_with_expr_ids(self, b):
        lookup = {"E1": "2*x+y=3", "E2": "x-y=1"}
        r = b.build(CasOperation(operation="solve_system", args={"equations": ["2*x+y=3", "x-y=1"], "vars": ["x", "y"]}))
        assert r.ok
        assert r.command == "solve([2*x+y=3, x-y=1],[x, y])"


# ---------------------------------------------------------------------------
# Linear algebra commands
# ---------------------------------------------------------------------------

class TestLinearAlgebra:
    def test_determinant(self, b):
        r = b.build(CasOperation(operation="determinant", args={"expr": "matrix([1,2],[3,4])"}))
        assert r.ok
        assert r.command == "determinant(matrix([1,2],[3,4]))"

    def test_inverse_matrix(self, b):
        r = b.build(CasOperation(operation="inverse_matrix", args={"expr": "matrix([1,2],[3,4])"}))
        assert r.ok
        assert r.command == "invert(matrix([1,2],[3,4]))"

    def test_eigenvalues(self, b):
        r = b.build(CasOperation(operation="eigenvalues", args={"expr": "matrix([1,2],[3,4])"}))
        assert r.ok
        assert r.command == "eigenvalues(matrix([1,2],[3,4]))"


# ---------------------------------------------------------------------------
# Output commands
# ---------------------------------------------------------------------------

class TestOutput:
    def test_to_latex(self, b):
        r = b.build(CasOperation(operation="to_latex", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "tex1(x^2-1)"

    def test_to_plain_text(self, b):
        r = b.build(CasOperation(operation="to_plain_text", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "string(x^2-1)"


# ---------------------------------------------------------------------------
# Expression ID resolution
# ---------------------------------------------------------------------------

class TestExprResolution:
    def test_unresolved_id_used_verbatim(self, b):
        # E99 not in lookup → used as-is
        r = b.build(CasOperation(operation="expand", args={"expr": "E99"}))
        assert r.ok
        assert r.command == "expand(E99)"

    def test_no_lookup_provided(self, b):
        r = b.build(CasOperation(operation="expand", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "expand(x^2-1)"

    def test_non_expr_id_not_substituted(self, b):
        r = b.build(CasOperation(operation="differentiate", args={"expr": "sin(x)", "var": "x", "order": "1"}))
        assert r.ok
        assert r.command == "diff(sin(x),x,1)"

    def test_integer_arg_converted_to_str(self, b):
        r = b.build(CasOperation(operation="taylor_expand", args={"expr": "exp(x)", "var": "x", "point": 0, "order": 3}))
        assert r.ok
        assert r.command == "ratdisrep(taylor(exp(x),x,0,3))"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_operation(self, b):
        r = b.build(CasOperation(operation="frobnicate", args={"expr": "x"}))
        assert not r.ok
        assert "frobnicate" in r.error
        assert r.command == ""

    def test_missing_template_placeholder(self, b):
        # differentiate needs {var} but it's absent
        r = b.build(CasOperation(operation="differentiate", args={"expr": "x^2"}))
        assert not r.ok
        assert "var" in r.error

    def test_result_is_dataclass(self, b):
        r = b.build(CasOperation(operation="expand", args={"expr": "x"}))
        assert isinstance(r, BuildResult)


# ---------------------------------------------------------------------------
# Round-trip: build → adapter-ready string
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_expand_round_trip(self, b):
        lookup = {"E3": "(x^2-1)/(x-1)"}
        r = b.build(CasOperation(operation="rational_simplify", args={"expr": "(x^2-1)/(x-1)"}))
        assert r.ok
        # This string can be passed directly to MaximaAdapter.execute()
        assert r.command == "ratsimp((x^2-1)/(x-1))"

    def test_full_physics_pipeline(self, b):
        # Simulate: differentiate E(k) = -2*t*cos(k) with respect to k
        lookup = {"E12": "-2*t*cos(k)"}
        r = b.build(CasOperation(operation="differentiate", args={"expr": "-2*t*cos(k)", "var": "k", "order": "1"}))
        assert r.ok
        assert r.command == "diff(-2*t*cos(k),k,1)"
