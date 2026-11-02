"""End-to-end scenario: OLS estimator β̂ = (X'X)⁻¹ X'y (AUDIT-XTX validation).

Exercises the statistics prime→transpose notation (AUDIT-XTX) inside a real
econometrics derivation: declare the design matrix X and response y, then let
``X'X`` and ``X'y`` (the prime = transpose) feed the matrix inverse / product
operations to compute a concrete β̂.

Design: X = [[1,1],[1,2],[1,3]] (intercept + one regressor), y = [1,2,2].
The OLS fit is ŷ = 2/3 + (1/2)x, so β̂ = [2/3, 1/2].
"""
from notebook.engine import NotebookEngine


def _register_and_simplify(engine, text):
    expr = engine.register_expression(text)
    _, value = engine.run_operation("simplify", {"expr": expr.display_id})
    return expr, value


def test_ols_beta_hat_via_prime_transpose():
    with NotebookEngine(title="OLS β̂") as e:
        e.start()
        # Econometrics session → prime means transpose (statistics notation profile).
        e.session.domains = ["econometrics"]
        e.symbol_context.domains = ["econometrics"]

        # 1. Declare the design matrix X and response y, then bind their values.
        e.run_operation("define_matrix", {"name": "X"})
        e.run_operation("define_matrix", {"name": "y"})
        e.register_expression("X = [1,1;1,2;1,3]")
        e.register_expression("y = [1;2;2]")

        # 2. X'X — the prime is transpose here (AUDIT-XTX); the CAS renders it as
        #    the matrix product transpose(X) . X and evaluates it.
        xtx, xtx_val = _register_and_simplify(e, "X'X")
        assert xtx_val.replace(" ", "") == "matrix([3,6],[6,14])"

        # 3. (X'X)⁻¹
        inv_step, inv_val = e.run_operation("inverse_matrix", {"expr": xtx.display_id})
        inv = e.expressions[inv_step.output_refs[0]]
        assert inv_val.replace(" ", "") == "matrix([7/3,-1],[-1,1/2])"

        # 4. X'y (prime = transpose again)
        xty, xty_val = _register_and_simplify(e, "X'y")
        assert xty_val.replace(" ", "") == "matrix([5],[11])"

        # 5. β̂ = (X'X)⁻¹ X'y
        beta_step, beta_val = e.run_operation(
            "matrix_multiply", {"a": inv.display_id, "b": xty.display_id})
        assert beta_val.replace(" ", "") == "matrix([2/3],[1/2])"
