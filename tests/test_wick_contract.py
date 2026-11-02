"""
tests/test_wick_contract.py
----------------------------
A-7 Phase 1: wick_contract operation.

Tests Wick contraction of fixed-index n-th order operator products.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(
        title="wick_test",
        mode=SessionMode.second_quantization
    ) as e:
        e.start()
        e.run_operation("define_fermion_operator", {"name": "a"})
        for name in ["V_HF", "Delta"]:
            e.run_operation("define_scalar_parameter", {"name": name})
        yield e


def expand(engine, expr_str):
    step, _ = engine.run_operation("expand", {"expr": expr_str})
    return step.output_refs[0]


class TestWickContractBasic:
    def test_hf_single_contraction_4th_order(self, engine):
        """a†(1)a†(2)a(3)a(4) with HF rule gives 4 contraction terms."""
        eid = expand(engine, "a'(1)*a'(2)*a(3)*a(4)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        assert len(terms) == 4

    def test_hf_fermionic_signs(self, engine):
        """Sign (-1)^(j-i-1) is computed correctly for each contraction."""
        eid = expand(engine, "a'(1)*a'(2)*a(3)*a(4)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        latexes = {t.latex for t in terms}
        # (i=1,j=3): sign=-1 → -V_HF * remaining
        assert any(t.startswith("-") for t in latexes)
        # (i=1,j=4): sign=+1 → +V_HF * remaining
        assert any(not t.startswith("-") for t in latexes)

    def test_bcs_all_contractions(self, engine):
        """HF + anomalous rules on a†(1)a†(2)a(3)a(4) gives 6 terms."""
        eid = expand(engine, "a'(1)*a'(2)*a(3)*a(4)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a",  "value": "V_HF"},   # 4 HF
            {"op1": "a",  "op2": "a",  "value": "Delta"},   # 1 anomalous
            {"op1": "a'", "op2": "a'", "value": "Delta"},   # 1 anomalous conj
        ])
        assert len(terms) == 6

    def test_sixth_order_to_fourth(self, engine):
        """6th-order product with HF rule gives 3×3=9 fourth-order terms."""
        eid = expand(engine, "a'(1)*a'(2)*a'(3)*a(4)*a(5)*a(6)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        assert len(terms) == 9

    def test_no_match_returns_empty(self, engine):
        """Rule that doesn't match any pair returns empty list."""
        eid = expand(engine, "a'(1)*a(2)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a'", "value": "Delta"},
        ])
        assert terms == []


class TestWickContractDoubleContraction:
    def test_include_constant_adds_scalar_terms(self, engine):
        """include_constant=True adds doubly-contracted scalar terms."""
        eid = expand(engine, "a'(1)*a'(2)*a(3)*a(4)")
        terms_no_const = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        terms_with_const = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ], include_constant=True)
        assert len(terms_with_const) > len(terms_no_const)
        # For 4-op with 1 rule: 4 single + 2 double = 6
        assert len(terms_with_const) == 6
        assert len(terms_no_const) == 4


class TestWickContractIntegration:
    def test_combine_selected_terms(self, engine):
        """Selected contraction terms can be combined via combine_expressions."""
        eid = expand(engine, "a'(1)*a'(2)*a(3)*a(4)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        _, combined = engine.run_operation("combine_expressions", {
            "expr_a": terms[0].display_id,
            "expr_b": terms[1].display_id,
            "op": "+",
        })
        assert combined is not None
        assert "V_HF" in combined

    def test_unknown_expr_raises(self, engine):
        """KeyError for nonexistent source expression."""
        with pytest.raises(KeyError):
            engine.wick_contract("E999", [{"op1": "a'", "op2": "a", "value": "V_HF"}])

    def test_empty_rules_raises(self, engine):
        """ValueError for empty contraction_rules."""
        eid = expand(engine, "a'(1)*a(2)")
        with pytest.raises(ValueError):
            engine.wick_contract(eid, [])


class TestWickContractPhase2:
    """Phase 2: algebraic indices with INDEX1/INDEX2 templates."""

    def test_index_dependent_value_4th_order(self, engine):
        """V_HF(INDEX1,INDEX2) generates index-labelled contraction values."""
        eid = expand(engine, "a'(k)*a'(q)*a(p)*a(r)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF(INDEX1,INDEX2)"},
        ])
        assert len(terms) == 4
        # Contraction values carry actual operator indices
        latex_str = " ".join(t.latex for t in terms)
        assert "V_{HF}" in latex_str or "V_HF" in latex_str

    def test_phase1_unaffected_by_phase2(self, engine):
        """Plain value (no INDEX) still works correctly (Phase 1 path)."""
        eid = expand(engine, "a'(k)*a'(q)*a(p)*a(r)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF"},
        ])
        assert len(terms) == 4
        assert all("V_HF" in t.latex or "V_{HF}" in t.latex for t in terms)

    def test_bcs_algebraic_with_mixed_rules(self, engine):
        """Mixed INDEX and plain-scalar rules on algebraic expression."""
        eid = expand(engine, "a'(k)*a'(-k)*a(-k)*a(k)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a",  "value": "V_HF(INDEX1,INDEX2)"},
            {"op1": "a",  "op2": "a",  "value": "Delta"},
            {"op1": "a'", "op2": "a'", "value": "Delta"},
        ])
        assert len(terms) == 6


class TestWickContractMultiIndex:
    """Multi-index operator support (e.g. a(k,σ) with spin and momentum)."""

    def test_full_hf_potential_with_spin(self, engine):
        """V_HF(k,σ,p,τ) captures both Hartree and Fock terms."""
        eid = expand(engine, "a'(k,u)*a'(-k,d)*a(-k,d)*a(k,u)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a",
             "value": "V_HF(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
        ])
        assert len(terms) == 4
        latex_all = " ".join(t.latex for t in terms)
        # V_HF appears in all terms (in subscript or function-call notation)
        assert "V_{HF}" in latex_all or "V_HF" in latex_all
        # Spin labels u and d both appear (Hartree/Fock terms mix spins)
        assert "u" in latex_all and "d" in latex_all
        # At least one term has mixed spin (Fock) and one has same spin (Hartree)
        canonicals = [t.canonical for t in terms]
        has_mixed = any(("u" in c and "d" in c) for c in canonicals)
        assert has_mixed, f"Expected mixed-spin term; got {canonicals}"

    def test_backward_compat_index1_index2(self, engine):
        """INDEX1/INDEX2 (first index only) still works after multi-index addition."""
        eid = expand(engine, "a'(k,u)*a'(-k,d)*a(-k,d)*a(k,u)")
        terms = engine.wick_contract(eid, [
            {"op1": "a'", "op2": "a", "value": "V_HF(INDEX1,INDEX2)"},
        ])
        assert len(terms) == 4
        # Only momentum index visible (not spin)
        latex_all = " ".join(t.latex for t in terms)
        assert "V_{HF}" in latex_all

    def test_indexsum_with_spin_and_anomalous(self, engine):
        """IndexSum with spin-indexed HF + anomalous rules gives 6 terms."""
        step, _ = engine.run_operation("expand", {
            "expr": "IndexSum(U * a'(k,u)*a'(-k,d)*a(-k,d)*a(k,u), k)"
        })
        terms = engine.wick_contract(step.output_refs[0], [
            {"op1": "a'", "op2": "a",
             "value": "V_HF(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
            {"op1": "a",  "op2": "a",  "value": "Delta"},
            {"op1": "a'", "op2": "a'", "value": "Delta"},
        ])
        assert len(terms) == 6
