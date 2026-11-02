"""Tests for collapsing repeated same-variable differentiation in the LHS label.

When a derivative result (e.g. velocity d/dt(x)) is differentiated again w.r.t.
the same variable, the LHS must collapse to a higher-order derivative
(d²/dt²(x)) rather than nesting (d/dt(d/dt(x))).

Covers the physics use case: position → velocity → acceleration.
"""
from notebook.engine import _bump_same_var_derivative, _format_derivative_lhs


class TestDerivativeBump:
    def _first(self, sym: str, var: str, is_partial: bool = False) -> str:
        return _format_derivative_lhs(sym, var, 1, is_partial=is_partial, parse_ir=None)

    def test_ordinary_second_order(self):
        v = self._first(r"\xi_k", "k")
        assert v == r"\frac{d}{dk}\left(\xi_k\right)"
        a = _bump_same_var_derivative(v, "k", 1, parse_ir=None)
        assert a == r"\frac{d^{2}}{dk^{2}}\left(\xi_k\right)"

    def test_third_order_chain(self):
        v = self._first("x", "t")
        a = _bump_same_var_derivative(v, "t", 1, parse_ir=None)
        j = _bump_same_var_derivative(a, "t", 1, parse_ir=None)
        assert j == r"\frac{d^{3}}{dt^{3}}\left(x\right)"

    def test_position_velocity_acceleration(self):
        # d/dt(x) → d²/dt²(x)
        vel = self._first("x", "t")
        acc = _bump_same_var_derivative(vel, "t", 1, parse_ir=None)
        assert acc == r"\frac{d^{2}}{dt^{2}}\left(x\right)"

    def test_partial_second_order(self):
        p1 = self._first(r"\psi", "x", is_partial=True)
        p2 = _bump_same_var_derivative(p1, "x", 1, parse_ir=None)
        assert p2 == r"\frac{\partial^{2}}{\partial x^{2}}\left(\psi\right)"

    def test_add_order_two_in_one_step(self):
        v = self._first(r"\xi_k", "k")
        # differentiating a 1st derivative by order=2 → 3rd derivative
        r = _bump_same_var_derivative(v, "k", 2, parse_ir=None)
        assert r == r"\frac{d^{3}}{dk^{3}}\left(\xi_k\right)"

    def test_different_variable_does_not_bump(self):
        v = self._first(r"\xi_k", "k")
        # differentiating d/dk(...) by t is a genuine mixed derivative — do not collapse
        assert _bump_same_var_derivative(v, "t", 1, parse_ir=None) is None

    def test_non_derivative_source_returns_none(self):
        assert _bump_same_var_derivative(r"\xi_k", "k", 1, parse_ir=None) is None

    def test_mixed_partial_denominator_not_collapsed(self):
        # \frac{\partial^{2}}{\partial x \partial y}(f) is a mixed partial; the
        # denominator has two distinct vars and must not be treated as same-var.
        mixed = r"\frac{\partial^{2}}{\partial x\partial y}\left(f\right)"
        assert _bump_same_var_derivative(mixed, "x", 1, parse_ir=None) is None
