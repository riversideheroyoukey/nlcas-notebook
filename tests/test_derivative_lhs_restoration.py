"""Tests for D-7 (open_items.md): an H-1 LHS containing literal-division
derivative notation (``dy/dx``) is a compound expression, not the single
simple atom engine._build_expr_lookup's H-1 equation-restoration guard
(``_SIMPLE_LHS_RE``) accepts — so it was silently skipped and operations
like solve_ode/solve_ode_general only ever saw the bare RHS canonical
(e.g. "x" for a registered "dy/dx + y = x"), returning `false` instead of
the real solution.
"""
from notebook.engine import NotebookEngine
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler


def _maxima() -> MaximaBackendHandler:
    return MaximaBackendHandler(None)


class TestLiteralDerivativeCanonical:

    def test_maxima_first_derivative(self):
        assert _maxima().literal_derivative_canonical("y", "x") == "'diff(y,x)"

    def test_julia_unsupported_by_default(self):
        assert JuliaBackendHandler(None).literal_derivative_canonical("y", "x") is None


class TestRestoreDerivativeLhs:

    def test_derivative_plus_atom(self):
        assert _maxima().restore_derivative_lhs("dy/dx+y") == "'diff(y,x)+y"

    def test_atom_plus_derivative_with_spaces(self):
        assert _maxima().restore_derivative_lhs("y + dy/dx") == "y+'diff(y,x)"

    def test_bare_derivative(self):
        assert _maxima().restore_derivative_lhs("dy/dx") == "'diff(y,x)"

    def test_different_variable_pair(self):
        assert _maxima().restore_derivative_lhs("dx/dt+x") == "'diff(x,t)+x"

    def test_no_derivative_present_returns_none(self):
        """A pure simple-atom LHS is already handled by the existing
        _SIMPLE_LHS_RE fast path in engine.py — this method has nothing to
        add and must not shadow it."""
        assert _maxima().restore_derivative_lhs("H") is None

    def test_unsupported_term_aborts(self):
        """A term that is neither a literal derivative nor a simple atom
        (here: a LaTeX fraction) must abort the whole restoration rather
        than guess — matches the original _SIMPLE_LHS_RE philosophy of
        never feeding CAS syntax it can't vouch for."""
        assert _maxima().restore_derivative_lhs(r"dy/dx+\frac{a}{b}") is None

    def test_julia_backend_never_restores(self):
        """literal_derivative_canonical returns None by default, so the
        first derivative term aborts the whole restoration immediately —
        a safe no-op on backends that don't override it."""
        assert JuliaBackendHandler(None).restore_derivative_lhs("dy/dx+y") is None


class TestEngineIntegration:
    """End-to-end through run_operation — the same $target expr-id
    resolution path (engine._build_expr_lookup) a plan step uses."""

    def test_solve_ode_via_h1_registered_expression(self):
        with NotebookEngine(title="d7 solve_ode") as e:
            ex = e.register_expression("x", latex="dy/dx + y = x")
            assert ex.canonical == "x"  # sanity: H-1 RHS-only canonical
            step, val = e.run_operation(
                "solve_ode", {"equation": ex.display_id, "y": "y", "x": "x"})
            assert step.status.value == "success"
            assert val != "false"
            assert "x-1" in val.replace(" ", "")

    def test_solve_ode_general_via_h1_registered_expression(self):
        with NotebookEngine(title="d7 solve_ode_general") as e:
            ex = e.register_expression("x", latex="dy/dx + y = x")
            step, val = e.run_operation(
                "solve_ode_general",
                {"equation": ex.display_id, "depvar": "y", "indepvar": "x"})
            assert step.status.value == "success"
            assert val != "false"

    def test_simple_atom_lhs_regression_unaffected(self):
        """The pre-existing _SIMPLE_LHS_RE path (e.g. "H = ...") must keep
        working unchanged — restore_derivative_lhs only fires for keys the
        fast path already rejected."""
        with NotebookEngine(title="d7 regression") as e:
            ex = e.register_expression("m*a", latex="F = m*a")
            step, val = e.run_operation("solve_equation", {
                "equation": ex.display_id, "var": "a"})
            assert step.status.value == "success"
