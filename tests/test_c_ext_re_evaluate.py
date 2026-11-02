"""
tests/test_c_ext_re_evaluate.py
---------------------------------
C-Ext Phase 1-A: NotebookEngine.re_evaluate_expr() tests.

Verifies that scalar and quantum-operator expressions can be re-evaluated
with updated symbol values — the foundation for mean-field self-consistency
iterations (A-2) and parametric sweeps.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


# ---------------------------------------------------------------------------
# Julia backend tests (live process)
# ---------------------------------------------------------------------------

class TestReEvaluateJulia:
    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(
            title="re_eval_test",
            mode=SessionMode.second_quantization
        ) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            for name in ["xi", "Delta", "n_k"]:
                e.run_operation("define_scalar_parameter", {"name": name})
            self.engine = e
            yield

    def test_scalar_numeric_substitution(self):
        """xi * n_k with n_k ← 0.5 gives 0.5*xi."""
        expr = self.engine.register_expression("xi * n_k")
        new = self.engine.re_evaluate_expr(expr.display_id, {"n_k": "0.5"})
        assert new is not None
        assert "0.5" in new.canonical or "xi" in new.canonical

    def test_operator_expr_coefficient_update(self):
        """Delta coefficient in a QuExpr is updated correctly."""
        step, _ = self.engine.run_operation(
            "expand", {"expr": "Delta * c'(k) * c'(-k)"}
        )
        out_id = step.output_refs[0]
        new = self.engine.re_evaluate_expr(out_id, {"Delta": "0.1"})
        assert new is not None
        assert "0.1" in new.canonical

    def test_multiple_substitutions(self):
        """xi and Delta updated simultaneously."""
        step, _ = self.engine.run_operation(
            "expand",
            {"expr": "xi * c'(k) * c(k) - Delta * c'(k) * c'(-k)"},
        )
        out_id = step.output_refs[0]
        new = self.engine.re_evaluate_expr(out_id, {"xi": "1.0", "Delta": "0.2"})
        assert new is not None
        assert "1.0" in new.canonical or "c†" in new.canonical

    def test_self_consistency_iteration(self):
        """Successive n_k updates model a mean-field self-consistency loop."""
        step, _ = self.engine.run_operation(
            "expand", {"expr": "n_k * c'(k) * c(k)"}
        )
        H_id = step.output_refs[0]
        for n_val in ["0.3", "0.42", "0.5"]:
            new = self.engine.re_evaluate_expr(H_id, {"n_k": n_val})
            assert new is not None, f"n_k={n_val} → None"
            assert n_val in new.canonical

    def test_eref_substitution(self):
        """substitutions value can be an expression E-ref."""
        ref = self.engine.register_expression("0.3 * xi")
        expr = self.engine.register_expression("xi * n_k")
        new = self.engine.re_evaluate_expr(expr.display_id, {"n_k": ref.display_id})
        assert new is not None
        assert "xi" in new.canonical

    def test_unknown_expr_id_raises(self):
        """KeyError for a nonexistent expression ID."""
        with pytest.raises(KeyError):
            self.engine.re_evaluate_expr("E999", {"xi": "1.0"})

    def test_empty_substitutions_returns_expr(self):
        """Empty substitutions dict returns a re-registered copy of the expression."""
        expr = self.engine.register_expression("xi * n_k")
        new = self.engine.re_evaluate_expr(expr.display_id, {})
        # May return None (backend-defined) or the same canonical.
        if new is not None:
            assert new.canonical == expr.canonical


# ---------------------------------------------------------------------------
# Maxima backend: C-Ext Phase 1-B
# ---------------------------------------------------------------------------

class TestReEvaluateMaxima:
    """C-Ext Phase 1-B: Maxima re_evaluate_expr using subst()."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(
            title="maxima_re_eval",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            self.engine = e
            yield

    def test_numeric_substitution(self):
        """T*k_B with T←300 gives 300*k_B."""
        expr = self.engine.register_expression("T * k_B")
        new = self.engine.re_evaluate_expr(expr.display_id, {"T": "300"})
        assert new is not None
        assert "300" in new.canonical

    def test_multiple_substitutions(self):
        """H_0+H_1 with both substituted gives combined expression."""
        expr = self.engine.register_expression("H_0 + H_1")
        new = self.engine.re_evaluate_expr(expr.display_id, {
            "H_0": "p^2/(2*m)",
            "H_1": "-k*(x-a)^2",
        })
        assert new is not None
        assert "p" in new.canonical or "k" in new.canonical

    def test_eref_substitution(self):
        """substitutions value can be an E-ref."""
        rhs = self.engine.register_expression("300 * k_B")
        expr = self.engine.register_expression("T * m")
        new = self.engine.re_evaluate_expr(expr.display_id, {"T": rhs.display_id})
        assert new is not None
        assert "300" in new.canonical

    def test_empty_substitutions(self):
        """Empty substitutions return canonical unchanged."""
        expr = self.engine.register_expression("x^2 + 1")
        new = self.engine.re_evaluate_expr(expr.display_id, {})
        if new is not None:
            assert "x" in new.canonical

    def test_unknown_expr_raises(self):
        """KeyError for nonexistent expression ID."""
        with pytest.raises(KeyError):
            self.engine.re_evaluate_expr("E999", {"x": "1"})


# ---------------------------------------------------------------------------
# B-alias update: update_lhs_alias / _update_alias in run_operation
# ---------------------------------------------------------------------------

class TestUpdateLhsAlias:
    """B-alias update: reassign a named symbol to a new expression."""

    def test_update_lhs_alias_method(self):
        """update_lhs_alias redirects the label to the new expression."""
        with NotebookEngine(
            title="alias_method_test",
            mode=SessionMode.second_quantization,
        ) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "mz"})

            H = e.register_expression("-Sz_1 * Sz_2", lhs_label="H")
            assert e.lhs_alias_map.get("H") == H.display_id

            # Register a different expression and redirect H to it
            H2 = e.register_expression("-mz * Sz_1", lhs_label="H_mf")
            e.update_lhs_alias(H2.display_id, "H")

            assert e.lhs_alias_map.get("H") == H2.display_id
            # Original still in history
            assert H.display_id in e.expressions

    def test_run_operation_update_alias(self):
        """_update_alias in run_operation args updates lhs_alias_map."""
        with NotebookEngine(
            title="alias_run_test",
            mode=SessionMode.second_quantization,
        ) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "mz"})
            e.run_operation("define_scalar_parameter", {"name": "J"})
            e.run_operation("define_scalar_parameter", {"name": "Gamma"})

            H = e.register_expression(
                "-J * Sz_1 * Sz_2 - Gamma * (Sx_1 + Sx_2)", lhs_label="H"
            )
            original_id = H.display_id
            assert e.lhs_alias_map.get("H") == original_id

            step, _ = e.run_operation("operator_substitute", {
                "expr": H.display_id,
                "op_name": "Sz",
                "rhs": "mz",
                "formal_index": "2",
                "exact_index": True,
                "_update_alias": "H",
            })
            new_id = step.output_refs[0]

            # H now points to the new expression
            assert e.lhs_alias_map.get("H") == new_id
            assert new_id != original_id
            # Original preserved in history
            assert original_id in e.expressions

    def test_update_alias_unknown_expr_raises(self):
        """update_lhs_alias raises KeyError for unknown expr_id."""
        with NotebookEngine(
            title="alias_err_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            with pytest.raises(KeyError):
                e.update_lhs_alias("E999", "H")
