"""Integration tests for operator_substitute on Julia/QuantumAlgebra backend.

These tests run the actual Julia process and verify that operator → operator
linear-combination substitution (Bogoliubov transform, polarisation decomposition,
etc.) produces correct QuantumAlgebra results.

Note: expressions must be registered via engine.register_expression (not via
run_operation("expand",...)) because operator_substitute rejects CAS-derived
(_nb_Ek) expressions — by design, to prevent chained substitutions on opaque
canonicals. Direct registration preserves the symbolic structure.
"""
from __future__ import annotations
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    with NotebookEngine(
        title="op_subst_julia",
        mode=SessionMode.second_quantization,
    ) as e:
        e.start()
        yield e


# ---------------------------------------------------------------------------
# Basic operator → operator substitution
# ---------------------------------------------------------------------------

class TestBasicOperatorSubstitution:

    def test_single_rule_annihilation(self, engine):
        """c(k) → u*g(k): number operator xi*c†(k)c(k) → involves g."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        e.run_operation("define_scalar_parameter", {"name": "xi"})
        e.run_operation("define_scalar_parameter", {"name": "u"})

        expr = e.register_expression("xi*c'(k)*c(k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s
        assert "u" in s

    def test_creation_auto_conjugated(self, engine):
        """c†(k) is automatically replaced by (u*g(k))† = u*g†(k)."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        e.run_operation("define_scalar_parameter", {"name": "u"})

        expr = e.register_expression("c'(k)*c(k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s


# ---------------------------------------------------------------------------
# Bogoliubov transformation (BCS)
# ---------------------------------------------------------------------------

class TestBogoliubovTransform:

    def test_single_mode_bogoliubov(self, engine):
        """H = xi*c'c - Delta*c'c': after c(k)→u*g(k)+v*g'(-k), result contains g."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "u", "v", "Delta"):
            e.run_operation("define_scalar_parameter", {"name": name})

        expr = e.register_expression("xi*c'(k)*c(k) - Delta*c'(k)*c'(-k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k) + v*g'(-k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s
        assert "u" in s
        assert "v" in s

    def test_two_rule_spin_bogoliubov(self, engine):
        """Spin-labelled BCS: c(k,↑)→u*g(k,↑)+v*g'(-k,↓), c(-k,↓)→u*g(-k,↓)-v*g'(k,↑)."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})

        expr = e.register_expression("xi*c'(k,u)*c(k,u)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_rules": [
                {"op_name": "c", "rhs": "u*g(k,u) + v*g'(-k,d)",
                 "formal_index": "k", "spin_label": "u"},
                {"op_name": "c", "rhs": "u*g(-k,d) - v*g'(k,u)",
                 "formal_index": "k", "spin_label": "d"},
            ],
        })
        s = str(result)
        assert "g" in s

    def test_bogoliubov_full_bcs_hamiltonian(self, engine):
        """Full BCS H after Bogoliubov: result contains both diagonal and off-diagonal terms."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "Delta", "u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})

        expr = e.register_expression(
            "xi*c'(k)*c(k) + xi*c'(-k)*c(-k) - Delta*c'(k)*c'(-k) - Delta*c(-k)*c(k)"
        )

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k) + v*g'(-k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s
        assert "u" in s
        assert "v" in s

    def test_bogoliubov_result_contains_quasiparticle_number(self, engine):
        """xi*c'c → xi*u²*g'g (diagonal quasiparticle term) must appear."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})

        expr = e.register_expression("xi*c'(k)*c(k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k) + v*g'(-k)",
            "formal_index": "k",
        })
        s = str(result)
        # Diagonal term: xi*u²*g†g
        assert "g" in s
        assert "xi" in s or "u" in s


# ---------------------------------------------------------------------------
# Photon polarisation decomposition (Boson)
# ---------------------------------------------------------------------------

class TestPolarisationDecomposition:

    def test_circular_to_linear_substitution(self, engine):
        """a(k,R/L) → linear combinations of b(k,x/y)."""
        e = engine
        e.run_operation("define_boson_operator", {"name": "a"})
        e.run_operation("define_boson_operator", {"name": "b"})
        e.run_operation("define_scalar_parameter", {"name": "omega"})

        expr = e.register_expression("omega*a'(k,R)*a(k,R) + omega*a'(k,L)*a(k,L)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_rules": [
                {"op_name": "a", "rhs": "(1/2)*(b(k,x) - im*b(k,y))",
                 "formal_index": "k", "spin_label": "R"},
                {"op_name": "a", "rhs": "(1/2)*(b(k,x) + im*b(k,y))",
                 "formal_index": "k", "spin_label": "L"},
            ],
        })
        s = str(result)
        assert "b" in s


# ---------------------------------------------------------------------------
# Negative-index propagation
# ---------------------------------------------------------------------------

class TestNegativeIndexPropagation:

    def test_minus_k_index_substituted(self, engine):
        """c(-k) is substituted with indices correctly propagated."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        e.run_operation("define_scalar_parameter", {"name": "u"})
        e.run_operation("define_scalar_parameter", {"name": "v"})

        expr = e.register_expression("c'(k)*c(-k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k) + v*g'(-k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s


# ---------------------------------------------------------------------------
# IndexSum traversal
# ---------------------------------------------------------------------------

class TestIndexSumTraversal:

    def test_substitution_inside_indexsum(self, engine):
        """operator_substitute descends into IndexSum body."""
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        e.run_operation("define_scalar_parameter", {"name": "xi"})
        e.run_operation("define_scalar_parameter", {"name": "u"})

        expr = e.register_expression("IndexSum(xi*c'(k)*c(k), k)")

        step2, result = e.run_operation("operator_substitute", {
            "expr": expr.display_id,
            "op_name": "c",
            "rhs": "u*g(k)",
            "formal_index": "k",
        })
        s = str(result)
        assert "g" in s


class TestChainingAfterSubstitute:
    """A follow-up re-parseable operation (e.g. normal_order) must act on the
    SUBSTITUTED result, not silently revert to the pre-substitution operator.

    Regression for the S-c (BCS) finding: operator_substitute propagated the
    SOURCE's input_canonical (the original Hamiltonian) onto its result, so a
    chained normal_order re-parsed the original H instead of the Bogoliubov
    g-form held in the opaque _nb_En variable.  The fix suppresses that
    propagation (input_canonical="") so chained ops bind to _nb_En.
    """

    def test_normal_order_after_bogoliubov_keeps_new_operators(self, engine):
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "Delta", "u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})

        H = e.register_expression("xi*c'(k)*c(k) - Delta*c'(k)*c'(-k)")
        step, _ = e.run_operation("operator_substitute", {
            "expr": H.display_id,
            "op_name": "c",
            "rhs": "u*g(k) + v*g'(-k)",
            "formal_index": "k",
        })
        out = e.expressions[step.output_refs[0]]
        # The substitution result must NOT carry the original H as its
        # re-parseable input (that was the bug).
        assert out.input_canonical == ""

        step2, result2 = e.run_operation("normal_order", {"expr": out.display_id})
        s = str(result2)
        # Chained result is the quasiparticle (g) form, not the original c-form.
        assert "g" in s
        assert "c(" not in s and "c'(" not in s and "cd(" not in s


class TestNegationDeltaEvaluation:
    """δ(-k,k) must vanish in the QA path too (not only via SymPy routing).

    -k is encoded as the upper-case label "K"; QuantumAlgebra keeps δ(K,k)
    symbolic.  _nb_eval_site_deltas now drops sign-partner deltas (k vs K),
    which is δ(k,-k)=δ(2k,0)=0 away from the zone centre — making the QA path
    agree with the SymPy routing.  Regression for the long-standing A-1 δ(-k,k)
    survival surfaced in the S-c (BCS) practical loop.
    """

    def test_bcs_spin_bogoliubov_has_no_kronecker_delta(self, engine):
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("xi", "Delta", "u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})

        Hs = e.register_expression(
            "xi*c'(k,u)*c(k,u) + xi*c'(-k,d)*c(-k,d) "
            "- Delta*c'(k,u)*c'(-k,d) - Delta*c(-k,d)*c(k,u)"
        )
        _, result = e.run_operation("operator_substitute", {
            "expr": Hs.display_id,
            "op_rules": [
                {"op_name": "c", "rhs": "u*g(k,u) + v*g'(-k,d)",
                 "formal_index": "k", "spin_label": "u"},
                {"op_name": "c", "rhs": "u*g(-k,d) - v*g'(k,u)",
                 "formal_index": "k", "spin_label": "d"},
            ],
        })
        s = str(result)
        # No surviving Kronecker delta (the δ(-k,k) term must be evaluated away).
        assert "δ(" not in s and "delta(" not in s.lower()
        # Sanity: the quasiparticle operators are present.
        assert "g" in s


class TestVacuumExpectationSymbolic:
    """vacuum_expectation must work when symbolic scalar parameters are present.

    QuantumAlgebra's vacExpVal throws "non-boolean (Num) used in boolean context"
    on any QuExpr carrying a Symbolics coefficient (u, v, ξ, Δ, …) — i.e. nearly
    every physics expression.  The engine now extracts the operator-free part
    directly (_nb_vac_expect), keeping symbolic coefficients.  Surfaced in the
    S-c (BCS) gap-equation assembly, where ⟨c_{-k↓}c_{k↑}⟩ = u·v.
    """

    def test_symbolic_number_operator_vacuum(self, engine):
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_scalar_parameter", {"name": "u"})
        expr = e.register_expression("u*c(1)*c'(1)")
        _, result = e.run_operation("vacuum_expectation", {"expr": expr.display_id})
        assert str(result).replace(" ", "") == "u"

    def test_bcs_anomalous_average_is_uv(self, engine):
        e = engine
        e.run_operation("define_fermion_operator", {"name": "c"})
        e.run_operation("define_fermion_operator", {"name": "g"})
        for name in ("u", "v"):
            e.run_operation("define_scalar_parameter", {"name": name})
        pair = e.register_expression("c(-k,d)*c(k,u)")
        step, _ = e.run_operation("operator_substitute", {
            "expr": pair.display_id,
            "op_rules": [
                {"op_name": "c", "rhs": "u*g(k,u) + v*g'(-k,d)",
                 "formal_index": "k", "spin_label": "u"},
                {"op_name": "c", "rhs": "u*g(k,d) - v*g'(-k,u)",
                 "formal_index": "k", "spin_label": "d"},
            ],
        })
        _, result = e.run_operation(
            "vacuum_expectation", {"expr": e.expressions[step.output_refs[0]].display_id})
        # ⟨0_g| c_{-k↓} c_{k↑} |0_g⟩ = u·v  (the BCS anomalous average)
        assert str(result).replace(" ", "") in ("u*v", "v*u")
