"""Free parameters are NAMES the CAS holds, so they must be declared (docs/32 §2.7).

``ode2`` emits ``%k1``/``%k2``; the engine renames them to ``k_1``/``k_2`` INSIDE
Maxima, so those names exist on the CAS side.  Nothing declared them on the
Python side, so ``k_1`` read as an undeclared subscripted symbol and lowered to
``k(1)`` — and ``substitute k_1 = 0``, which is the entire point of FREE-PARAM
(imposing an initial condition on a general solution), returned success while
changing nothing.

These tests assert the VALUE, not just the status: the bug returned success.
"""
import json

import pytest

from notebook.engine import NotebookEngine


def _general_solution(engine):
    engine.declare_time_dependent("q", "t")
    eq = engine.register_expression("q_ddot + w^2*q = 0")
    return engine.solve_ode_general(eq.display_id)


class TestFreshSession:

    def test_ode_constants_are_declared(self):
        with NotebookEngine(title="fp decl") as e:
            sol = _general_solution(e)
            assert sol.free_parameters == ["k_1", "k_2"]
            assert "k_1" in e.symbol_context.registry
            assert "k_2" in e.symbol_context.registry

    def test_declared_constants_lower_to_themselves(self):
        """Undeclared, ``k_1`` lowers to ``k(1)`` and no substitution can match."""
        with NotebookEngine(title="fp lower") as e:
            assert e._handler.to_backend_code("k_1").code == "k(1)"
            _general_solution(e)
            assert e._handler.to_backend_code("k_1").code == "k_1"

    def test_initial_condition_actually_applies(self):
        with NotebookEngine(title="fp subst") as e:
            sol = _general_solution(e)
            st, val = e.run_operation(
                "substitute", {"expr": sol.display_id, "lhs": "k_1", "rhs": "0"})
            assert st.status.value == "success"
            # The sin term carried k_1 and must be gone; k_2's cos term remains.
            assert "sin" not in val
            assert "k_2" in val and "cos" in val

    def test_unused_candidate_name_is_not_reserved(self):
        """solve_ode_general mints k_1/k_2/C up front but a 2nd-order ODE uses
        only two of them; declaring the third would reserve a name for nothing."""
        with NotebookEngine(title="fp unused") as e:
            _general_solution(e)
            assert "C" not in e.symbol_context.registry


class TestLoadedSession:

    def test_legacy_file_without_the_declaration_still_works(self, tmp_path):
        """A file written before this rule records free_parameters on the
        expression but has no registry entry.  Loading must restore it, or the
        continuation after a reload behaves differently from the live session
        (the very hole tests/test_roundtrip_persistence.py exists to close)."""
        path = tmp_path / "legacy.json"
        with NotebookEngine(title="fp legacy") as e:
            sol = _general_solution(e)
            sol_id = sol.display_id
            e.save(path)

        # Simulate the legacy file: drop the free parameters from the registry.
        data = json.loads(path.read_text())
        for name in ("k_1", "k_2"):
            data["quantity_registry"].pop(name, None)
        assert data["expressions"][sol_id]["free_parameters"] == ["k_1", "k_2"]
        path.write_text(json.dumps(data))

        loaded = NotebookEngine.load(path)
        loaded.start()
        try:
            assert "k_1" in loaded.symbol_context.registry
            st, val = loaded.run_operation(
                "substitute", {"expr": sol_id, "lhs": "k_1", "rhs": "0"})
            assert st.status.value == "success"
            assert "sin" not in val
        finally:
            loaded.close()
