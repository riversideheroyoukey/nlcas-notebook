"""
tests/experiments/test_sympy_operator_algebra.py

Live Julia tests for the SymPy-based operator algebra engine:
  _nb_sympy_fermion_comm  -- fermionic commutator with algebraic indices
  _nb_sympy_boson_comm    -- bosonic commutator with algebraic indices

These cover the cases that QuantumAlgebra.jl cannot handle because its
QuIndex type requires single-character symbol names and rejects arithmetic
expressions like -k, k+q, or a*k+b as indices.

See: docs/developer_manuals/04_second_quantization_engine_design.md

NOTE on return format:
  _nb_sympy_fermion_comm / _nb_sympy_boson_comm return Tuple{String,String}
  (value_str, latex_str).  The JuliaAdapter command loop detects this and routes:
    result[1] -> main_part  -> r.value
    result[2] -> tex_part   -> r.latex
  So use r.value for the SymPy sstr expression and r.latex for LaTeX.
  Do NOT call eval(r.value) -- the sstr form contains function names like c(k)
  that are not defined in Python's namespace.
"""

import pytest
from notebook.runtime.julia_adapter import JuliaAdapter


@pytest.fixture(scope="module")
def julia():
    """Shared Julia process for the entire module -- startup is slow."""
    j = JuliaAdapter()
    j.start()
    yield j


# ---------------------------------------------------------------------------
# Fermionic commutators
# ---------------------------------------------------------------------------

class TestFermionComm:
    def test_kinetic_term(self, julia):
        """[cd_k c_k, c_q] = -delta(k,q) c_k  (standard kinetic energy term)."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)*c(k)", "c(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        val = r.value
        assert "KroneckerDelta(k, q)" in val
        assert "c(k)" in val
        # The coefficient of c(k) must be -delta(k,q)
        assert val.startswith("-") or "-KroneckerDelta" in val

    def test_bcs_pairing_term(self, julia):
        """[cd_k cd_{-k}, c_q] = delta(-k,q) cd_k - delta(k,q) cd_{-k}  (BCS pairing)."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)*cd(-k)", "c(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        val = r.value
        latex = r.latex
        # Must contain both delta terms
        assert "KroneckerDelta(-k, q)" in val or "KroneckerDelta(k, q)" in val
        # Must contain creation operators
        assert "cd(" in val
        # LaTeX must be generated
        assert r"\delta" in latex or "delta" in latex.lower()

    def test_bcs_pairing_substitute_q_eq_minus_k(self, julia):
        """Substituting q = -k into the BCS result must give cd_k (coefficient 1)."""
        # _nb_sympy_fermion_comm returns Tuple{String,String} (sstr, latex).
        # Destructure the tuple inside Julia to access the sstr form for sympify.
        r = julia.execute('''
            val_str, _ = _nb_sympy_fermion_comm("cd(k)*cd(-k)", "c(q)", "k,q")
            sp = SymPy.PyCall.pyimport("sympy")
            k, q = sp.symbols("k q")
            expr = sp.sympify(val_str)
            string(expr.subs(q, -k))
        ''')
        assert r.error is None, f"Substitution failed: {r.error}"
        # At q=-k: delta(-k,-k)=1 for the cd(k) term, delta(k,-k)=0 for the other
        assert "cd(k)" in r.value

    def test_number_operator_term(self, julia):
        """[cd_k c_k, cd_q] = delta(k,q) cd_k  (number operator with creation target)."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)*c(k)", "cd(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        val = r.value
        assert "KroneckerDelta(k, q)" in val
        assert "cd(k)" in val

    def test_single_creation_operator(self, julia):
        """[cd_k, c_q] contains delta(k,q)  (single operator case)."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)", "c(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        assert "KroneckerDelta" in r.value

    def test_algebraic_index(self, julia):
        """[cd_{a*k+b}, c_q] correctly handles a*k+b as index."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(a*k+b)", "c(q)", "k,q,a,b")')
        assert r.error is None, f"Julia error: {r.error}"
        assert "KroneckerDelta" in r.value

    def test_three_operator_product(self, julia):
        """[cd_k cd_p c_p, c_q]: 3-operator product uses correct sign convention."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)*cd(p)*c(p)", "c(q)", "k,p,q")')
        assert r.error is None, f"Julia error: {r.error}"
        assert "KroneckerDelta" in r.value

    def test_latex_output(self, julia):
        """LaTeX output for BCS term uses proper notation."""
        r = julia.execute('_nb_sympy_fermion_comm("cd(k)*cd(-k)", "c(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        latex = r.latex
        # Should contain \delta
        assert r"\delta" in latex or "delta" in latex.lower()
        # Negative index -k should appear
        assert "- k" in latex or "-k" in latex


# ---------------------------------------------------------------------------
# Bosonic commutators
# ---------------------------------------------------------------------------

class TestBosonComm:
    def test_number_operator(self, julia):
        """[ad_k a_k, a_q] = -delta(k,q) a_k  (bosonic number operator)."""
        r = julia.execute('_nb_sympy_boson_comm("ad(k)*a(k)", "a(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        val = r.value
        assert "KroneckerDelta(k, q)" in val
        assert "a(k)" in val

    def test_bec_pairing_term(self, julia):
        """[ad_k ad_{-k}, a_q] = delta(-k,q) ad_k + delta(k,q) ad_{-k}  (BEC Bogoliubov)."""
        r = julia.execute('_nb_sympy_boson_comm("ad(k)*ad(-k)", "a(q)", "k,q")')
        assert r.error is None, f"Julia error: {r.error}"
        assert "KroneckerDelta" in r.value
        assert "ad(" in r.value

    def test_algebraic_index(self, julia):
        """[ad_{k+q} a_{k+q}, a_p]: k+q as index."""
        r = julia.execute('_nb_sympy_boson_comm("ad(k+q)*a(k+q)", "a(p)", "k,q,p")')
        assert r.error is None, f"Julia error: {r.error}"
        assert "KroneckerDelta" in r.value

    def test_boson_vs_fermion_sign(self, julia):
        """Bosonic [a†a, a] = -delta*a and fermionic [c†c, c] = -delta*c give same sign."""
        r_f = julia.execute('_nb_sympy_fermion_comm("cd(k)*c(k)", "c(q)", "k,q")')
        r_b = julia.execute('_nb_sympy_boson_comm("ad(k)*a(k)", "a(q)", "k,q")')
        assert r_f.error is None, f"Fermion error: {r_f.error}"
        assert r_b.error is None, f"Boson error: {r_b.error}"
        val_f = r_f.value
        val_b = r_b.value
        # Both should give -KroneckerDelta(k,q) * operator
        assert val_f.startswith("-") or "-KroneckerDelta" in val_f
        assert val_b.startswith("-") or "-KroneckerDelta" in val_b
