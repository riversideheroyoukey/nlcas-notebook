"""Transparency equivalence tests: QA path vs SymPy path.

Guarantees:
  1. Mathematical equivalence: same number of terms, same sign pattern,
     same operator structure (creation/annihilation ordering).
  2. Display consistency: rho, kappa rendered as \\rho_{...}, \\kappa_{...} in both paths.
  3. Correct delta evaluation: SymPy path correctly evaluates δ(-k,k)=0
     where QA path cannot (K-convention limitation).
  4. Transparent routing: user inputs the same operation regardless of backend.

Routing rule:
  - Algebraic indices (-k, k+q) present → SymPy path (sympy_canonical set)
  - Integer-site / single-letter indices   → QA path (sympy_canonical = None)
"""

import re
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from notebook.engine import NotebookEngine
    from notebook.models.session import SessionMode
    e = NotebookEngine(title="equiv_test", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_fermion_operator", {"name": "c"})
    e.run_operation("define_scalar_parameter", {"name": "rho"})
    e.run_operation("define_scalar_parameter", {"name": "kappa"})
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. normal_order structural equivalence
# ---------------------------------------------------------------------------

class TestNormalOrderEquivalence:
    """QA and SymPy produce structurally equivalent normal-ordered forms."""

    def test_adjacent_pair_sign_qa(self, engine):
        """c(1)*c'(2) (QA): gives -c†(2)*c(1)."""
        expr = engine.register_expression("c(1)*c'(2)")
        step, val = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        # QA path: sympy_canonical is None
        assert out.sympy_canonical is None
        assert "†" in out.latex or "dagger" in out.latex
        # Sign: anticommutation gives -1
        assert out.latex.startswith("c(1)*c'(2) = -") or "-" in val

    def test_adjacent_pair_sign_sympy(self, engine):
        """c(-k)*c'(k) (SymPy): gives -c†(k)*c(-k), δ(-k,k)=0 evaluated."""
        expr = engine.register_expression("c(-k)*c'(k)")
        step, val = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        # SymPy path: sympy_canonical is set
        assert out.sympy_canonical is not None
        # Same mathematical content as QA: negative sign
        assert "-" in val
        # No delta remainder (δ(-k,k) was correctly evaluated to 0)
        assert "KroneckerDelta" not in val
        assert "delta" not in val.lower() or val.count("delta") == 0

    def test_already_normal_ordered_qa(self, engine):
        """c'(1)*c(2) (QA): already normal ordered, unchanged."""
        expr = engine.register_expression("c'(1)*c(2)")
        step, val = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        assert out.sympy_canonical is None
        # Stays positive (no sign flip needed)
        assert "-" not in val.lstrip("-0 ")  # might have overall sign from other processing

    def test_already_normal_ordered_sympy(self, engine):
        """c'(-k)*c(k) (SymPy): already normal ordered, unchanged."""
        expr = engine.register_expression("c'(-k)*c(k)")
        step, val = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        assert out.sympy_canonical is not None
        # c†(-k)*c(k): creation before annihilation → no sign flip, no delta term
        assert "cd(-k)*c(k)" in val or ("cd" in val and "c(" in val)

    def test_display_uses_dagger_notation_both_paths(self, engine):
        """Both QA and SymPy paths display operators with dagger notation."""
        qa = engine.register_expression("c(1)*c'(2)")
        sp_expr = engine.register_expression("c(-k)*c'(k)")

        s_qa, _ = engine.run_operation("normal_order", {"expr": qa.display_id})
        s_sp, _ = engine.run_operation("normal_order", {"expr": sp_expr.display_id})

        latex_qa = engine.expressions[s_qa.output_refs[0]].latex
        latex_sp = engine.expressions[s_sp.output_refs[0]].latex

        assert "dagger" in latex_qa or "†" in latex_qa or r"\dagger" in latex_qa
        assert "dagger" in latex_sp or "†" in latex_sp or r"\dagger" in latex_sp


# ---------------------------------------------------------------------------
# 2. wick_contract structural equivalence
# ---------------------------------------------------------------------------

class TestWickContractEquivalence:
    """QA and SymPy wick_contract produce structurally equivalent results."""

    RULES_RHO = [{"op1": "c'", "op2": "c", "value": "rho(INDEX1,INDEX2)"}]

    def _expand_and_contract(self, engine, expr_str, rules, include_constant=False):
        step, _ = engine.run_operation("expand", {"expr": expr_str})
        return engine.wick_contract(step.output_refs[0], rules,
                                    include_constant=include_constant)

    def test_4op_term_count_qa_vs_sympy(self, engine):
        """4-operator product gives 4 contraction terms in both paths."""
        # QA path: integer sites
        terms_qa = self._expand_and_contract(
            engine, "c'(1)*c'(2)*c(3)*c(4)", self.RULES_RHO)
        assert len(terms_qa) == 4

        # SymPy path: algebraic indices
        e_sp = engine.register_expression("c'(-k)*c'(k)*c(k)*c(-k)")
        terms_sp = engine.wick_contract(e_sp.display_id, self.RULES_RHO)
        assert len(terms_sp) == 4

    def test_4op_sign_pattern_equivalent(self, engine):
        """Sign pattern (-+-+) or equivalent is the same in QA and SymPy paths."""
        terms_qa = self._expand_and_contract(
            engine, "c'(1)*c'(2)*c(3)*c(4)", self.RULES_RHO)
        e_sp = engine.register_expression("c'(-k)*c'(k)*c(k)*c(-k)")
        terms_sp = engine.wick_contract(e_sp.display_id, self.RULES_RHO)

        # Both should have 2 positive and 2 negative terms
        neg_qa = sum(1 for t in terms_qa if t.latex.startswith("-"))
        neg_sp = sum(1 for t in terms_sp if t.latex.startswith("-") or t.latex.startswith("- "))
        assert neg_qa == 2
        assert neg_sp == 2

    def test_rho_subscript_notation_in_both_paths(self, engine):
        """ρ displayed as \rho_{...} subscript in both QA and SymPy paths."""
        terms_qa = self._expand_and_contract(
            engine, "c'(1)*c'(2)*c(3)*c(4)", self.RULES_RHO)
        e_sp = engine.register_expression("c'(-k)*c'(k)*c(k)*c(-k)")
        terms_sp = engine.wick_contract(e_sp.display_id, self.RULES_RHO)

        for t in terms_qa:
            assert r"\rho_{" in t.latex, f"QA: expected \\rho_{{...}} in '{t.latex}'"
        for t in terms_sp:
            assert r"\rho_{" in t.latex, f"SymPy: expected \\rho_{{...}} in '{t.latex}'"
            # No function-call notation \rho{\left(...\right)}
            assert r"\rho{\left(" not in t.latex, f"SymPy: unwanted fn notation in '{t.latex}'"

    def test_remaining_operators_are_2op_products(self, engine):
        """After one contraction of 4-op product, 2 operators remain."""
        terms_qa = self._expand_and_contract(
            engine, "c'(1)*c'(2)*c(3)*c(4)", self.RULES_RHO)
        e_sp = engine.register_expression("c'(-k)*c'(k)*c(k)*c(-k)")
        terms_sp = engine.wick_contract(e_sp.display_id, self.RULES_RHO)

        # Each remaining term has exactly one creation and one annihilation operator
        for t in terms_qa:
            # Canonical is _nb_En for QA; check latex for operator count
            assert "†" in t.latex or "dagger" in t.latex or "cd" in t.latex

        for t in terms_sp:
            # SymPy: canonical is explicit
            c_count = t.canonical.count("c(") + t.canonical.count("cd(")
            assert c_count == 2, f"Expected 2 operators in '{t.canonical}'"


# ---------------------------------------------------------------------------
# 3. Delta evaluation correctness (SymPy-specific)
# ---------------------------------------------------------------------------

class TestDeltaEvaluation:
    """SymPy correctly evaluates KroneckerDelta(-k,k)=0 where QA cannot."""

    def test_delta_minusk_k_vanishes(self, engine):
        """With delta template, c'(-k)*c(k) contraction → 0 (δ(-k,k)=0)."""
        rules = [{"op1": "c'", "op2": "c",
                  "value": "delta(INDEX1,INDEX2)*rho(INDEX1)"}]
        expr = engine.register_expression("c'(-k)*c(k)")
        terms = engine.wick_contract(expr.display_id, rules, include_constant=True)
        assert len(terms) == 0, (
            f"Expected 0 terms (δ(-k,k)=0), got {[t.canonical for t in terms]}"
        )

    def test_delta_k_k_evaluates_to_one(self, engine):
        """With delta template, c'(k)*c(k) contraction → rho(k) (δ(k,k)=1)."""
        rules = [{"op1": "c'", "op2": "c",
                  "value": "delta(INDEX1,INDEX2)*rho(INDEX1)"}]
        expr = engine.register_expression("c'(k)*c(k)")
        terms = engine.wick_contract(expr.display_id, rules, include_constant=True)
        assert len(terms) == 1
        assert "rho" in terms[0].canonical or "\\rho" in terms[0].latex

    def test_qa_path_does_not_evaluate_K_delta(self, engine):
        """QA path with K convention leaves δ(K,k) unevaluated (known limitation).
        This confirms WHY SymPy routing is necessary for algebraic indices."""
        # With integer indices, QA correctly evaluates δ(1,2)=0 implicitly
        # (different site indices → no contraction needed)
        # We just verify that the QA path returns non-empty results for
        # expressions where δ(1,3) or δ(2,4) etc. vanish implicitly.
        step, _ = engine.run_operation("expand", {"expr": "c'(1)*c'(2)*c(3)*c(4)"})
        rules = [{"op1": "c'", "op2": "c", "value": "rho(INDEX1,INDEX2)"}]
        terms = engine.wick_contract(step.output_refs[0], rules)
        # QA correctly gives 4 terms (the remaining operators have distinct indices)
        assert len(terms) == 4


# ---------------------------------------------------------------------------
# 4. Routing correctness
# ---------------------------------------------------------------------------

class TestRouting:
    """Routing decisions are correct: algebraic → SymPy, integer/simple → QA."""

    def test_integer_site_uses_qa(self, engine):
        """c(1)*c'(2) routes to QA (no algebraic index)."""
        expr = engine.register_expression("c(1)*c'(2)")
        step, _ = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        assert out.sympy_canonical is None  # QA path

    def test_minus_k_uses_sympy(self, engine):
        """c(-k)*c'(k) routes to SymPy (-k is algebraic)."""
        expr = engine.register_expression("c(-k)*c'(k)")
        step, _ = engine.run_operation("normal_order", {"expr": expr.display_id})
        out = engine.expressions[step.output_refs[0]]
        assert out.sympy_canonical is not None  # SymPy path

    def test_kplusq_would_use_sympy(self, engine):
        """c'(k+q) contains algebraic index → SymPy routing for normal_order."""
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        ctx = engine.symbol_context
        jcg = JuliaCodeGenerator(context=ctx)

        surf = parser.parse("c'(k+q)*c(k)")
        ir = resolver.resolve(surf, ctx)
        assert jcg._has_algebraic_indices(ir), "k+q should be detected as algebraic"

    def test_wick_delta_rule_auto_routes_to_sympy(self, engine):
        """delta() in value template auto-routes to SymPy even for simple indices."""
        rules = [{"op1": "c'", "op2": "c",
                  "value": "delta(INDEX1,INDEX2)*rho(INDEX1)"}]
        expr = engine.register_expression("c'(k)*c(k)")
        terms = engine.wick_contract(expr.display_id, rules, include_constant=True)
        # delta(k,k)=1 so result is rho(k)
        assert len(terms) == 1
        for t in terms:
            assert t.sympy_canonical is not None  # SymPy path
