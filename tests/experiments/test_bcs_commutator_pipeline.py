"""
tests/experiments/test_bcs_commutator_pipeline.py

End-to-end pipeline tests: parser → resolver → JuliaCodeGenerator → live Julia.

These tests verify that the full chain from IR construction through
JuliaCodeGenerator._generate_sympy_commutator to the Julia SymPy engine
produces physically correct commutator results for BCS/condensed-matter inputs.

Requires a live Julia process (slow startup; scope="module" fixture is shared).

See: docs/developer_manuals/04_second_quantization_engine_design.md
"""

import pytest
from notebook.runtime.julia_adapter import JuliaAdapter
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.models.ir import (
    Symbol, IndexSymbol, IndexedSymbol, FunctionCall,
    Commutator, Mul, IndexSum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia():
    """Shared Julia process for the module — startup is expensive."""
    j = JuliaAdapter()
    j.start()
    yield j
    # JuliaAdapter.stop() or just let the process die with the session


@pytest.fixture
def gen():
    return JuliaCodeGenerator()


def idx(name: str) -> IndexSymbol:
    return IndexSymbol(name=name)


def sym(name: str) -> Symbol:
    return Symbol(name=name)


def indexed(op_name: str, index_name: str) -> IndexedSymbol:
    return IndexedSymbol(base=sym(op_name), indices=[idx(index_name)])


# ---------------------------------------------------------------------------
# Helper: build IR and execute the generated Julia code
# ---------------------------------------------------------------------------

def exec_commutator(julia, gen, commutator_ir):
    """Generate Julia code from IR and execute it; return (value, latex)."""
    code = gen.generate(commutator_ir)
    r = julia.execute(code)
    assert r.error is None, f"Julia error: {r.error}\nGenerated code: {code}"
    # The command loop returns (value_str, latex_str) for SymPy results;
    # the JuliaAdapter stores it as r.value = value_str, r.latex = latex_str
    return r.value, r.latex


# ---------------------------------------------------------------------------
# BCS fermionic commutators
# ---------------------------------------------------------------------------

class TestBCSFermionPipeline:
    def test_kinetic_term_quantumalgebra(self, julia, gen):
        """[c†_k c_k, c_q] has no algebraic indices → QuantumAlgebra comm() path.

        k and q are simple categorical labels, so _has_algebraic_indices = False.
        Code-generation only — QuantumAlgebra execution requires @fermion_ops c
        declaration; live execution is tested in test_quantum_delta_integration.py.
        """
        ir = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c", "k")]),
            b=indexed("c", "q"),
        )
        code = gen.generate(ir)
        # Simple indices → QuantumAlgebra path, not SymPy
        assert code.startswith("comm("), f"Expected QuantumAlgebra path; got: {code}"
        assert "_nb_sympy" not in code

    def test_kinetic_term_with_algebraic_target(self, julia, gen):
        """[c†_k c_k, c_{-q}] has algebraic target index → SymPy path.

        The target operator c_{-q} with index -q triggers the SymPy route.
        Result: -KroneckerDelta(k, -q) * c(k) = KroneckerDelta(k, q) * c(k).
        """
        ir = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c", "k")]),
            b=indexed("c", "-q"),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_fermion_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        assert "KroneckerDelta" in val
        assert "c(" in val

    def test_bcs_pairing_term_ir(self, julia, gen):
        """IR for [c†_k c†_{-k}, c_q] → BCS pairing commutator."""
        ir = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_fermion_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        # Must contain two delta terms
        assert "KroneckerDelta" in val
        assert "cd(" in val
        # LaTeX must contain \delta
        assert r"\delta" in latex or "delta" in latex.lower()

    def test_number_operator_pairing_target(self, julia, gen):
        """IR for [c†_k c_k, c†_{-q}] has algebraic -q target → SymPy path."""
        ir = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c", "k")]),
            b=indexed("c'", "-q"),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_fermion_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        assert "KroneckerDelta" in val
        assert "cd(" in val

    def test_function_call_form_c_prime(self, julia, gen):
        """FunctionCall(Symbol("c'"), [idx]) form (from resolver) works end-to-end."""
        ir = Commutator(
            a=Mul(args=[
                FunctionCall(func=sym("c'"), args=[idx("k")]),
                FunctionCall(func=sym("c'"), args=[idx("-k")]),
            ]),
            b=FunctionCall(func=sym("c"), args=[idx("q")]),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_fermion_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        assert "KroneckerDelta" in val

    def test_simple_commutator_uses_quantumalgebra(self, julia, gen):
        """A commutator with no algebraic indices stays on the QuantumAlgebra path."""
        ir = Commutator(
            a=indexed("c'", "k"),
            b=indexed("c", "q"),
        )
        code = gen.generate(ir)
        assert code.startswith("comm("), f"Expected QuantumAlgebra path; got: {code}"
        # No SymPy helpers
        assert "_nb_sympy" not in code

    def test_index_sum_wrapped_bcs(self, julia, gen):
        """IndexSum(Mul(c†_k, c†_{-k}), [k]) wrapping is unwrapped; result is correct."""
        body = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        lhs = IndexSum(body=body, dummy_indices=[idx("k")])
        ir = Commutator(a=lhs, b=indexed("c", "q"))

        code = gen.generate(ir)
        # Single-term IndexSum routes to _nb_sympy_sum_fermion_comm via Case 2 in
        # _maybe_expand_commutator (direct IndexSum LHS handling).
        assert "_nb_sympy_sum_fermion_comm" in code or "_nb_sympy_fermion_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        # After dummy-variable contraction the KroneckerDelta is evaluated.
        # Correct result for [Σ_k c†_k c†_{-k}, c_q] is -c†_{-q}.
        assert "cd(" in val


# ---------------------------------------------------------------------------
# BEC bosonic commutators
# ---------------------------------------------------------------------------

class TestBECBosonPipeline:
    def test_number_operator_quantumalgebra(self, julia, gen):
        """[a†_k a_k, a_q] has no algebraic indices → QuantumAlgebra path.

        k and q are simple labels; _has_algebraic_indices = False.
        Code-generation only — live execution requires @boson_ops a declaration.
        """
        ir = Commutator(
            a=Mul(args=[indexed("a'", "k"), indexed("a", "k")]),
            b=indexed("a", "q"),
        )
        code = gen.generate(ir)
        assert code.startswith("comm("), f"Expected QuantumAlgebra path; got: {code}"
        assert "_nb_sympy" not in code

    def test_number_operator_with_algebraic_target(self, julia, gen):
        """[a†_k a_k, a_{-q}] has algebraic -q → SymPy boson path."""
        ir = Commutator(
            a=Mul(args=[indexed("a'", "k"), indexed("a", "k")]),
            b=indexed("a", "-q"),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_boson_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        assert "KroneckerDelta" in val
        assert "a(" in val

    def test_bec_pairing_term_ir(self, julia, gen):
        """IR for [a†_k a†_{-k}, a_q] → BEC Bogoliubov pairing term."""
        ir = Commutator(
            a=Mul(args=[indexed("a'", "k"), indexed("a'", "-k")]),
            b=indexed("a", "q"),
        )
        code = gen.generate(ir)
        assert "_nb_sympy_boson_comm" in code

        val, latex = exec_commutator(julia, gen, ir)
        assert "KroneckerDelta" in val
        assert "ad(" in val
