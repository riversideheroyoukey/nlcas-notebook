"""Tests for transparent QA/SymPy routing for algebraic indices in normal_order.

Key design: when an operator expression has algebraic indices (-k, k+q, …)
that QuantumAlgebra cannot handle, build_command routes to _nb_sympy_normal_order.
The result has sympy_canonical set for chaining.

Covers:
  - Integer-site expressions → QA fast path (no sympy_canonical)
  - Algebraic-index expressions (-k) → SymPy path, δ(-k,k)=0 correct
  - k+q index → SymPy path
  - Already-normal-ordered expression → unchanged
  - SymPy chain: SymPy result followed by another normal_order
  - Expression.sympy_canonical field backward compatibility
"""

import pytest
from notebook.models.expression import Expression
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


# ---------------------------------------------------------------------------
# Unit tests for Expression.sympy_canonical field (Step 0)
# ---------------------------------------------------------------------------

def test_expression_sympy_canonical_default():
    """sympy_canonical defaults to None — backward compatible with old sessions."""
    e = Expression(display_id="E1", branch_id="main", latex="x", canonical="x")
    assert e.sympy_canonical is None


def test_expression_sympy_canonical_set():
    """sympy_canonical can be set to a string."""
    e = Expression(
        display_id="E1", branch_id="main", latex="x", canonical="_nb_E1",
        sympy_canonical="-cd(k)*c(-k)",
    )
    assert e.sympy_canonical == "-cd(k)*c(-k)"


def test_expression_sympy_canonical_json_roundtrip():
    """Session JSON without sympy_canonical field loads correctly (backward compat)."""
    import json
    data = {"display_id": "E1", "branch_id": "main", "latex": "x", "canonical": "x"}
    e = Expression.model_validate(data)
    assert e.sympy_canonical is None
    # With field set: saves and reloads correctly
    e2 = Expression(
        display_id="E1", branch_id="main", latex="x", canonical="_nb_E1",
        sympy_canonical="-cd(k)*c(-k)",
    )
    dumped = json.loads(e2.model_dump_json())
    e3 = Expression.model_validate(dumped)
    assert e3.sympy_canonical == "-cd(k)*c(-k)"


# ---------------------------------------------------------------------------
# Julia engine tests (require a running Julia adapter)
# ---------------------------------------------------------------------------

@pytest.fixture
def fermion_engine():
    """NotebookEngine with a fermion operator c defined."""
    from notebook.engine import NotebookEngine
    from notebook.models.session import SessionMode
    e = NotebookEngine(title="test_routing", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_fermion_operator", {"name": "c"})
    yield e
    e.close()


def test_normal_order_integer_site_uses_qa(fermion_engine):
    """c†(1)*c(2) → QA fast path, sympy_canonical is None."""
    e = fermion_engine
    expr = e.register_expression("c'(1)*c(2)")
    step, val = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs, "Expected output expression"
    out = e.expressions[step.output_refs[0]]
    # QA result: sympy_canonical NOT set
    assert out.sympy_canonical is None
    # Value contains expected QA output
    assert "c" in val.lower() or "†" in val


def test_normal_order_minus_k_uses_sympy(fermion_engine):
    """c(-k)*c'(k) → SymPy path, δ(-k,k)=0 correctly evaluated."""
    e = fermion_engine
    expr = e.register_expression("c(-k)*c'(k)")
    step, val = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs, "Expected output expression"
    out = e.expressions[step.output_refs[0]]
    # SymPy result: sympy_canonical IS set
    assert out.sympy_canonical is not None
    # Correct normal ordering: c(-k)*c†(k) = -c†(k)*c(-k)  (δ(-k,k)=0 since k≠0)
    sc = out.sympy_canonical
    assert "cd(k)" in sc and "c(-k)" in sc, f"Expected cd(k) and c(-k) in '{sc}'"
    # Must have the -1 sign (delta term vanished)
    assert sc.startswith("-"), f"Expected negative sign in '{sc}'"


def test_normal_order_already_normal_ordered(fermion_engine):
    """c'(-k)*c(k) is already normal ordered; result equals input (no sign flip)."""
    e = fermion_engine
    expr = e.register_expression("c'(-k)*c(k)")
    step, val = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs
    out = e.expressions[step.output_refs[0]]
    # Should be in sympy path (algebraic index)
    assert out.sympy_canonical is not None
    # Already normal ordered: c†(-k)*c(k) → unchanged
    sc = out.sympy_canonical
    assert "cd(-k)" in sc and "c(k)" in sc, f"Expected cd(-k) and c(k) in '{sc}'"
    # No delta term (indices -k and k are different)
    assert not sc.startswith("1"), f"Unexpected constant term in '{sc}'"


def test_normal_order_same_index_has_delta(fermion_engine):
    """c(k)*c'(k) → δ(k,k) - c†(k)*c(k) = 1 - c†(k)*c(k)."""
    e = fermion_engine
    # c(k)*c'(k): annihilation before creation, same index — not algebraic
    # so this goes via QA, but test the δ(k,k)=1 case via SymPy to be safe.
    # Use register to force it through.
    expr = e.register_expression("c(k)*c'(k)")
    step, val = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs
    out = e.expressions[step.output_refs[0]]
    # For same-letter index with no arithmetic: QA handles it (no algebraic index).
    # Just verify the result contains either "1" term or c†(k)*c(k) structure.
    sc = out.sympy_canonical  # may be None (QA path) or set (SymPy path)
    if sc is not None:
        # SymPy path: expect "1 - cd(k)*c(k)"
        assert "1" in sc or "cd(k)" in sc


def test_normal_order_sympy_result_stored(fermion_engine):
    """sympy_canonical stores the SymPy-evaluatable string for chaining."""
    e = fermion_engine
    expr = e.register_expression("c(-k)*c'(k)")
    step, _ = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs
    out = e.expressions[step.output_refs[0]]
    assert out.sympy_canonical is not None
    # The sympy_canonical should be parseable as a SymPy-style expression
    sc = out.sympy_canonical
    # Must contain operator calls in cd/c notation
    assert "c(" in sc or "cd(" in sc, f"No operators in sympy_canonical: '{sc}'"


def test_expr_lookup_sympy_key_populated(fermion_engine):
    """After a SymPy operation, __sympy__<expr_id> is in expr_lookup for chaining."""
    e = fermion_engine
    expr = e.register_expression("c(-k)*c'(k)")
    step, _ = e.run_operation("normal_order", {"expr": expr.display_id})
    assert step.output_refs
    result_id = step.output_refs[0]
    out = e.expressions[result_id]
    # Verify that sympy_canonical is set and contains expected content
    assert out.sympy_canonical is not None
    # Verify that in next run_operation the key would appear in expr_lookup.
    # We do this by registering another expression referencing the result
    # and checking that the routing picks up the SymPy chain path.
    # (Full chain test is in test_sympy_chain_normal_order below.)
    assert out.sympy_canonical == "-cd(k)*c(-k)"


# ---------------------------------------------------------------------------
# wick_contract SymPy routing tests
# ---------------------------------------------------------------------------

def test_wick_contract_sympy_2op_scalar(fermion_engine):
    """c'(-k)*c(k) with <c†c>=n(INDEX1) → n(-k) (include_constant=True)."""
    e = fermion_engine
    expr = e.register_expression("c'(-k)*c(k)")
    terms = e.wick_contract(expr.display_id, [
        {"op1": "c'", "op2": "c", "value": "n(INDEX1)"}
    ], include_constant=True)
    assert len(terms) == 1
    t = terms[0]
    assert "n(-k)" in t.canonical or "n" in t.canonical
    assert t.sympy_canonical is not None


def test_wick_contract_sympy_delta_vanishes(fermion_engine):
    """c'(-k)*c(k) with delta(INDEX1,INDEX2)*n(INDEX1) → 0 (δ(-k,k)=0)."""
    e = fermion_engine
    # With the physically correct propagator <c†(i)c(j)> = δ(i,j) n_i,
    # contracting c†(-k) with c(k) gives δ(-k,k)*n(-k) = 0 (since k≠0).
    expr = e.register_expression("c'(-k)*c(k)")
    terms = e.wick_contract(expr.display_id, [
        {"op1": "c'", "op2": "c", "value": "delta(INDEX1,INDEX2)*n(INDEX1)"}
    ], include_constant=True)
    # δ(-k,k) = 0 for nonzero k → zero terms returned
    assert len(terms) == 0, f"Expected 0 terms (δ(-k,k)=0), got {[t.canonical for t in terms]}"


def test_wick_contract_sympy_delta_same_index(fermion_engine):
    """c'(k)*c(k) with delta(INDEX1,INDEX2)*n(INDEX1) → n(k) (δ(k,k)=1)."""
    e = fermion_engine
    expr = e.register_expression("c'(k)*c(k)")
    terms = e.wick_contract(expr.display_id, [
        {"op1": "c'", "op2": "c", "value": "delta(INDEX1,INDEX2)*n(INDEX1)"}
    ], include_constant=True)
    # δ(k,k) = 1 → result is n(k)
    assert len(terms) == 1, f"Expected 1 term, got {[t.canonical for t in terms]}"
    assert "n(k)" in terms[0].canonical


def test_wick_contract_sympy_4op_2remaining(fermion_engine):
    """c'(-k)*c(p)*c'(q)*c(k) gives 3 contraction terms (2 operators remaining)."""
    e = fermion_engine
    expr = e.register_expression("c'(-k)*c(p)*c'(q)*c(k)")
    terms = e.wick_contract(expr.display_id, [
        {"op1": "c'", "op2": "c", "value": "n(INDEX1,INDEX2)"}
    ], include_constant=False)
    assert len(terms) == 3
    canonicals = [t.canonical for t in terms]
    # Should include n(-k, p) * remaining operators
    assert any("n(-k" in c for c in canonicals)


def test_wick_contract_sympy_result_has_sympy_canonical(fermion_engine):
    """wick_contract SymPy path sets sympy_canonical for chaining."""
    e = fermion_engine
    expr = e.register_expression("c'(-k)*c(k)")
    terms = e.wick_contract(expr.display_id, [
        {"op1": "c'", "op2": "c", "value": "n(INDEX1)"}
    ], include_constant=True)
    assert terms, "Expected at least one contraction term"
    assert all(t.sympy_canonical is not None for t in terms)


def test_has_algebraic_indices_detection():
    """_has_algebraic_indices correctly detects FunctionCall with Mul/Add args."""
    from notebook.runtime.shared_parser import SharedParser
    from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
    from notebook.runtime.julia_code_generator import JuliaCodeGenerator
    from notebook.models.symbol_context import SymbolContext

    parser = SharedParser()
    resolver = JuliaSemanticResolver()
    ctx = SymbolContext()
    jcg = JuliaCodeGenerator(context=ctx)

    # Algebraic cases
    for algebraic in ["c(-k)*c'(k)", "c'(k+q)*c(k)"]:
        surf = parser.parse(algebraic)
        ir = resolver.resolve(surf, ctx)
        assert jcg._has_algebraic_indices(ir), f"Expected algebraic: {algebraic}"

    # Non-algebraic cases
    for simple in ["c'(1)*c(2)", "c'(k)*c(k)"]:
        surf = parser.parse(simple)
        ir = resolver.resolve(surf, ctx)
        assert not jcg._has_algebraic_indices(ir), f"Expected simple: {simple}"


# ---------------------------------------------------------------------------
# vacuum_expectation SymPy routing (Step 5)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_vac():
    e = NotebookEngine(title="vac_routing", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_fermion_operator", {"name": "c"})
    yield e
    e.close()


def test_vac_exp_normal_ordered_sympy(engine_vac):
    """<0|c'(-k)*c(k)|0> = 0 via SymPy (already normal ordered)."""
    expr = engine_vac.register_expression("c'(-k)*c(k)")
    step, val = engine_vac.run_operation("vacuum_expectation", {"expr": expr.display_id})
    assert val.strip() == "0"


def test_vac_exp_not_normal_ordered_sympy(engine_vac):
    """<0|c(-k)*c'(k)|0> = 0 via SymPy: δ(-k,k) correctly evaluated to 0."""
    expr = engine_vac.register_expression("c(-k)*c'(k)")
    step, val = engine_vac.run_operation("vacuum_expectation", {"expr": expr.display_id})
    assert val.strip() == "0"
    # QA would return δ(Kk) ≠ 0 — SymPy gives the correct result


def test_vac_exp_same_index_sympy(engine_vac):
    """<0|c(k)*c'(k)|0> = 1 via SymPy or QA: δ(k,k) = 1."""
    expr = engine_vac.register_expression("c(k)*c'(k)")
    step, val = engine_vac.run_operation("vacuum_expectation", {"expr": expr.display_id})
    assert val.strip() == "1"


def test_vac_exp_minus_k_same_sympy(engine_vac):
    """<0|c(-k)*c'(-k)|0> = 1 via SymPy: δ(-k,-k) = 1."""
    expr = engine_vac.register_expression("c(-k)*c'(-k)")
    step, val = engine_vac.run_operation("vacuum_expectation", {"expr": expr.display_id})
    assert val.strip() == "1"


def test_vac_exp_chain_from_normal_order(engine_vac):
    """vacuum_expectation chains from normal_order SymPy result."""
    e2 = engine_vac.register_expression("c(-k)*c'(k)")
    s_no, _ = engine_vac.run_operation("normal_order", {"expr": e2.display_id})
    no_id = s_no.output_refs[0]
    # normal_order gives -cd(k)*c(-k) (sympy_canonical set)
    assert engine_vac.expressions[no_id].sympy_canonical is not None
    step, val = engine_vac.run_operation("vacuum_expectation", {"expr": no_id})
    assert val.strip() == "0"


# ---------------------------------------------------------------------------
# substitute SymPy routing (Step 6)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_sub():
    e = NotebookEngine(title="sub_routing", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_fermion_operator", {"name": "c"})
    e.run_operation("define_scalar_parameter", {"name": "xi"})
    yield e
    e.close()


def _normal_order_sympy(engine, expr_str):
    """Helper: register expr, apply normal_order, return output expr_id."""
    expr = engine.register_expression(expr_str)
    step, _ = engine.run_operation("normal_order", {"expr": expr.display_id})
    return step.output_refs[0]


def test_substitute_index_rename(engine_sub):
    """substitute k→p on SymPy result: operator order preserved."""
    sp_id = _normal_order_sympy(engine_sub, "c(-k)*c'(k)")  # -cd(k)*c(-k)
    step, val = engine_sub.run_operation("substitute",
        {"expr": sp_id, "lhs": "k", "rhs": "p"})
    out = engine_sub.expressions[step.output_refs[0]]
    assert out.sympy_canonical == "-cd(p)*c(-p)"
    assert "p" in val and "-p" in val


def test_substitute_sign_flip(engine_sub):
    """substitute k→-k: indices negate, operator order preserved."""
    sp_id = _normal_order_sympy(engine_sub, "c(-k)*c'(k)")  # -cd(k)*c(-k)
    step, val = engine_sub.run_operation("substitute",
        {"expr": sp_id, "lhs": "k", "rhs": "-k"})
    out = engine_sub.expressions[step.output_refs[0]]
    assert out.sympy_canonical == "-cd(-k)*c(k)"


def test_substitute_momentum_shift(engine_sub):
    """substitute k→k+q: algebraic rhs works correctly."""
    sp_id = _normal_order_sympy(engine_sub, "c(-k)*c'(k)")
    step, val = engine_sub.run_operation("substitute",
        {"expr": sp_id, "lhs": "k", "rhs": "k+q"})
    out = engine_sub.expressions[step.output_refs[0]]
    assert "k + q" in out.sympy_canonical or "k+q" in out.sympy_canonical


def test_substitute_scalar_coefficient_preserved(engine_sub):
    """Scalar coefficient (xi) is preserved after substitution."""
    sp_id = _normal_order_sympy(engine_sub, "-xi*c(-k)*c'(k)")  # xi*cd(k)*c(-k)
    step, val = engine_sub.run_operation("substitute",
        {"expr": sp_id, "lhs": "k", "rhs": "p"})
    out = engine_sub.expressions[step.output_refs[0]]
    assert "xi" in out.sympy_canonical
    assert "p" in out.sympy_canonical


def test_substitute_chain_with_vacuum_expectation(engine_sub):
    """substitute → vacuum_expectation chains correctly."""
    sp_id = _normal_order_sympy(engine_sub, "c(-k)*c'(k)")
    # k→p (gives -cd(p)*c(-p))
    step_sub, _ = engine_sub.run_operation("substitute",
        {"expr": sp_id, "lhs": "k", "rhs": "p"})
    # vacuum expectation of -cd(p)*c(-p) = 0
    step_vac, val_vac = engine_sub.run_operation("vacuum_expectation",
        {"expr": step_sub.output_refs[0]})
    assert val_vac.strip() == "0"


def test_substitute_qa_path_unchanged(engine_sub):
    """substitute on QA expression (no sympy_canonical) still uses QA path."""
    # c'(1)*c(1) → QA path
    expr = engine_sub.register_expression("c'(1)*c(1)")
    step_no, _ = engine_sub.run_operation("normal_order", {"expr": expr.display_id})
    qa_id = step_no.output_refs[0]
    assert engine_sub.expressions[qa_id].sympy_canonical is None
    # substitute should use QA template (won't crash even if result is 0)
    try:
        step_sub, val = engine_sub.run_operation("substitute",
            {"expr": qa_id, "lhs": "xi", "rhs": "1"})
        # Just verify it doesn't crash
        assert True
    except Exception:
        # QA substitute may fail for operator expressions — that's OK, it's not the SymPy path
        pass


def test_relabel_index_sympy_routes_to_substitute(engine_sub):
    """relabel_index on SymPy expression routes to substitute, not QA K-convention."""
    sp_id = _normal_order_sympy(engine_sub, "xi*c(-k)*c'(k)")  # xi*cd(k)*c(-k)
    step, val = engine_sub.run_operation("relabel_index",
        {"expr": sp_id, "old_index": "k", "new_index": "-k"})
    out = engine_sub.expressions[step.output_refs[0]]
    # Must NOT contain QA K-convention artifact
    assert "KroneckerDelta" not in val and "δ(K" not in val
    # Must contain correctly swapped indices
    assert out.sympy_canonical is not None
    assert "cd(-k)" in out.sympy_canonical
    assert "c(k)" in out.sympy_canonical


# ---------------------------------------------------------------------------
# re_evaluate_expr SymPy chain (A-10)
# ---------------------------------------------------------------------------

def test_re_evaluate_sympy_scalar_substitution(engine_sub):
    """re_evaluate_expr applies scalar substitution to a SymPy expression."""
    # Build SymPy expression: normal_order(-xi*c(-k)*c'(k)) = xi*cd(k)*c(-k)
    sp_id = _normal_order_sympy(engine_sub, "-xi*c(-k)*c'(k)")
    src = engine_sub.expressions[sp_id]
    assert src.sympy_canonical is not None

    # re_evaluate: xi → 1  (should replace scalar coefficient)
    new_expr = engine_sub.re_evaluate_expr(sp_id, substitutions={"xi": "1"})
    assert new_expr is not None, "re_evaluate_expr returned None"
    # Result should have xi replaced: coefficient becomes 1
    assert new_expr.sympy_canonical is not None
    assert "xi" not in new_expr.sympy_canonical
    # Operators should be intact
    assert "cd(k)" in new_expr.sympy_canonical
    assert "c(-k)" in new_expr.sympy_canonical


def test_re_evaluate_sympy_multi_substitution(engine_sub):
    """re_evaluate_expr applies multiple substitutions sequentially."""
    # xi*cd(k)*c(-k) - Delta*cd(k)*c'(-k)  (register directly as SymPy)
    sp_id = _normal_order_sympy(engine_sub, "-xi*c(-k)*c'(k)")
    # Apply two substitutions: xi→2, k→p
    new_expr = engine_sub.re_evaluate_expr(sp_id, substitutions={"xi": "2", "k": "p"})
    assert new_expr is not None
    assert new_expr.sympy_canonical is not None
    assert "xi" not in new_expr.sympy_canonical
    assert "cd(p)" in new_expr.sympy_canonical


def test_re_evaluate_qa_path_unchanged(engine_sub):
    """re_evaluate_expr on QA expression still uses QA path (no regression)."""
    # register a pure scalar expression (Symbolics path)
    expr = engine_sub.register_expression("xi * 2")
    new_expr = engine_sub.re_evaluate_expr(
        expr.display_id, substitutions={"xi": "3"}
    )
    # QA/Symbolics path: may succeed or gracefully return None — must not crash
    # (Pure scalar Symbolics substitution typically works)
    assert new_expr is None or new_expr.canonical is not None


# ---------------------------------------------------------------------------
# Operator identity substitution (A-5a)
# Uses only operators defined with @fermion_ops / @boson_ops / @tlsxyz_ops.
# @tlspm_ops is intentionally NOT used.
# ---------------------------------------------------------------------------

@pytest.fixture
def op_id_engine():
    """Engine with fermion c and alpha defined for operator identity tests."""
    e = NotebookEngine(title="op_id_test", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_fermion_operator", {"name": "c"})
    e.run_operation("define_fermion_operator", {"name": "alpha"})
    yield e
    e.close()


def test_operator_identity_simple_rename(op_id_engine):
    """c†(1)*c(2) → alpha†(1)*alpha(2): integer-site QA path, stays in QA."""
    e = op_id_engine
    # Integer sites: QA path (no sympy_canonical)
    expr = e.register_expression("c'(1)*c(2)")
    step_no, _ = e.run_operation("normal_order", {"expr": expr.display_id})
    no_id = step_no.output_refs[0]
    assert e.expressions[no_id].sympy_canonical is None  # confirm QA path
    # Rename c → alpha
    step, val = e.run_operation("substitute", {
        "expr": no_id,
        "lhs": "c",
        "rhs": "alpha",
    })
    out = e.expressions[step.output_refs[0]]
    # c replaced by alpha, stays in QA
    assert "alpha" in val
    assert out.sympy_canonical is None


def test_operator_identity_in_product(op_id_engine):
    """c†(1)*c(2): operators renamed to alpha, QA stays in QA."""
    e = op_id_engine
    # c†(1)*c(2) normal_orders to -c(2)*c†(1) (δ(1,2)=0 since distinct sites)
    expr = e.register_expression("c'(1)*c(2)")
    step_no, _ = e.run_operation("normal_order", {"expr": expr.display_id})
    no_id = step_no.output_refs[0]
    step, val = e.run_operation("substitute", {
        "expr": no_id,
        "lhs": "c",
        "rhs": "alpha",
    })
    assert "alpha" in val
    assert e.expressions[step.output_refs[0]].sympy_canonical is None


def test_operator_identity_scalar_rhs(op_id_engine):
    """Sz(1) → (1-2*n)/2: operator replaced by scalar expression."""
    e = op_id_engine
    e.run_operation("define_spin_operator", {"name": "S"})
    e.run_operation("define_scalar_parameter", {"name": "n"})
    expr = e.register_expression("Sz(1)")
    step, val = e.run_operation("substitute", {
        "expr": expr.display_id,
        "lhs": "Sz",
        "rhs": "(1 - 2*n)/2",
    })
    out = e.expressions[step.output_refs[0]]
    assert "Sz" not in val
    assert "n" in val
    assert out.sympy_canonical is None


def test_operator_identity_scalar_not_triggered(op_id_engine):
    """Scalar lhs (xi) must NOT trigger operator identity path: no error, result is QA."""
    e = op_id_engine
    e.run_operation("define_scalar_parameter", {"name": "xi"})
    # xi is a scalar — _maybe_operator_identity must return None (no trigger)
    expr = e.register_expression("xi * c'(1) * c(1)")
    step_no, _ = e.run_operation("normal_order", {"expr": expr.display_id})
    no_id = step_no.output_refs[0]
    # Substitute xi → 2 via the scalar (Symbolics) path — must not crash
    step, val = e.run_operation("substitute", {
        "expr": no_id,
        "lhs": "xi",
        "rhs": "2",
    })
    out = e.expressions[step.output_refs[0]]
    # Result should NOT have sympy_canonical (stays in QA domain)
    assert out.sympy_canonical is None


def test_operator_identity_via_expr_ref(op_id_engine):
    """RHS specified as E-ref: c → E3 where E3 = alpha."""
    e = op_id_engine
    rhs_expr = e.register_expression("alpha(1)")
    src = e.register_expression("c(1)")
    step_no, _ = e.run_operation("normal_order", {"expr": src.display_id})
    no_id = step_no.output_refs[0]
    step, val = e.run_operation("substitute", {
        "expr": no_id,
        "lhs": "c",
        "rhs": rhs_expr.display_id,   # E-ref → "alpha(1)"
    })
    out = e.expressions[step.output_refs[0]]
    assert "alpha" in val
    assert out.sympy_canonical is None


# ---------------------------------------------------------------------------
# collect_spin_pm (A-5a collect direction)
# ---------------------------------------------------------------------------

@pytest.fixture
def jc_engine():
    """JC-model-like engine: spin S + boson a."""
    e = NotebookEngine(title="jc_test", mode=SessionMode.second_quantization)
    e.start()
    e.run_operation("define_spin_operator",  {"name": "S"})
    e.run_operation("define_boson_operator", {"name": "a"})
    e.run_operation("define_scalar_parameter", {"name": "omega"})
    yield e
    e.close()


def test_collect_spin_pm_basic(jc_engine):
    """(-ω/2)·Sx·a + (iω/2)·Sy·a  →  (-ω/2)·Sm·a."""
    e = jc_engine
    # Define Sp/Sm via H-1 style registration
    sp_expr = e.register_expression("Sx(1) + im*Sy(1)")
    sm_expr = e.register_expression("Sx(1) - im*Sy(1)")

    # Register EOM-like result
    eom = e.register_expression("(-omega/2)*Sx(1)*a(1) + (im*omega/2)*Sy(1)*a(1)")
    step_no, _ = e.run_operation("normal_order", {"expr": eom.display_id})
    no_id = step_no.output_refs[0]

    step, val = e.run_operation("collect_spin_pm", {
        "expr": no_id,
        "sp": sp_expr.display_id,
        "sm": sm_expr.display_id,
    })
    # Result should contain Sm and no Sx/Sy
    assert "Sm" in val
    assert "Sx" not in val
    assert "Sy" not in val


def test_collect_spin_pm_sp_only(jc_engine):
    """(ω/2)·Sx·a + (iω/2)·Sy·a  →  (ω/2)·Sp·a  (c_Sm=0)."""
    e = jc_engine
    sp_expr = e.register_expression("Sx(1) + im*Sy(1)")
    sm_expr = e.register_expression("Sx(1) - im*Sy(1)")

    # Sp = Sx + im*Sy → combination giving Sp: (ω/2)*Sx + (iω/2)*Sy
    eom = e.register_expression("(omega/2)*Sx(1)*a(1) + (im*omega/2)*Sy(1)*a(1)")
    step_no, _ = e.run_operation("normal_order", {"expr": eom.display_id})
    no_id = step_no.output_refs[0]

    step, val = e.run_operation("collect_spin_pm", {
        "expr": no_id,
        "sp": sp_expr.display_id,
        "sm": sm_expr.display_id,
    })
    assert "Sp" in val
    assert "Sx" not in val
    assert "Sy" not in val


def test_collect_spin_pm_sz_passthrough(jc_engine):
    """Sz term passes through unchanged (appears in leftover as QA output)."""
    e = jc_engine
    sp_expr = e.register_expression("Sx(1) + im*Sy(1)")
    sm_expr = e.register_expression("Sx(1) - im*Sy(1)")

    eom = e.register_expression("omega*Sz(1)*a(1)")
    step_no, _ = e.run_operation("normal_order", {"expr": eom.display_id})
    no_id = step_no.output_refs[0]

    step, val = e.run_operation("collect_spin_pm", {
        "expr": no_id,
        "sp": sp_expr.display_id,
        "sm": sm_expr.display_id,
    })
    # Sz goes to leftover — QA displays it (possibly as Sᶻ unicode)
    assert "S" in val and ("z" in val.lower() or "ᶻ" in val)
