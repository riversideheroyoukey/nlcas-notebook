"""
tests/test_julia_sympy_routing.py

Unit tests for the SymPy routing helpers in JuliaCodeGenerator.

These tests exercise the pure-Python logic (no Julia process required):
  _has_algebraic_indices  — detects -k / k+q / a*k+b
  _collect_index_vars     — gathers alphabetic tokens from indices
  _to_sympy_op_name       — maps c/c'/a/a' → cd/c/ad/a
  _serialize_single_op_to_sympy — single operator → "cd(k)" etc.
  _serialize_lhs_to_sympy — product / IndexSum → prod_str
  _detect_statistics      — fermion vs boson heuristic
  _generate_sympy_commutator — full call-string generation

See: docs/developer_manuals/04_second_quantization_engine_design.md, notebook/runtime/julia_code_generator.py
"""

import pytest
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.models.ir import (
    Symbol, IndexSymbol, IndexedSymbol, FunctionCall, Adjoint,
    Commutator, Mul, NonCommutativeMul, Add, IndexSum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    return JuliaCodeGenerator()


def idx(name: str) -> IndexSymbol:
    return IndexSymbol(name=name)


def sym(name: str) -> Symbol:
    return Symbol(name=name)


def indexed(op_name: str, index_name: str) -> IndexedSymbol:
    return IndexedSymbol(base=sym(op_name), indices=[idx(index_name)])


def fcall(func_name: str, index_name: str) -> FunctionCall:
    return FunctionCall(func=sym(func_name), args=[idx(index_name)])


# ---------------------------------------------------------------------------
# _has_algebraic_indices
# ---------------------------------------------------------------------------

class TestHasAlgebraicIndices:
    def test_simple_index_is_not_algebraic(self, gen):
        node = idx("k")
        assert gen._has_algebraic_indices(node) is False

    def test_negated_index_is_algebraic(self, gen):
        node = idx("-k")
        assert gen._has_algebraic_indices(node) is True

    def test_sum_index_is_algebraic(self, gen):
        node = idx("k+q")
        assert gen._has_algebraic_indices(node) is True

    def test_product_index_is_algebraic(self, gen):
        node = idx("a*k+b")
        assert gen._has_algebraic_indices(node) is True

    def test_indexed_symbol_with_negated_index(self, gen):
        node = indexed("c'", "-k")
        assert gen._has_algebraic_indices(node) is True

    def test_indexed_symbol_with_simple_index(self, gen):
        node = indexed("c'", "k")
        assert gen._has_algebraic_indices(node) is False

    def test_mul_contains_algebraic(self, gen):
        node = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        assert gen._has_algebraic_indices(node) is True

    def test_mul_all_simple(self, gen):
        node = Mul(args=[indexed("c'", "k"), indexed("c", "k")])
        assert gen._has_algebraic_indices(node) is False

    def test_commutator_with_algebraic_lhs(self, gen):
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        assert gen._has_algebraic_indices(node) is True

    def test_commutator_with_algebraic_rhs(self, gen):
        node = Commutator(
            a=indexed("c'", "k"),
            b=indexed("c", "-k"),
        )
        assert gen._has_algebraic_indices(node) is True

    def test_commutator_all_simple(self, gen):
        node = Commutator(
            a=indexed("c'", "k"),
            b=indexed("c", "q"),
        )
        assert gen._has_algebraic_indices(node) is False

    def test_index_sum_dummy_index_negated(self, gen):
        """IndexSum with body containing -k index."""
        body = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        node = IndexSum(body=body, dummy_indices=[idx("k")])
        assert gen._has_algebraic_indices(node) is True

    def test_function_call_algebraic(self, gen):
        node = fcall("c'", "k+q")
        assert gen._has_algebraic_indices(node) is True


# ---------------------------------------------------------------------------
# _collect_index_vars
# ---------------------------------------------------------------------------

class TestCollectIndexVars:
    def test_simple_index(self, gen):
        node = idx("k")
        assert gen._collect_index_vars(node) == {"k"}

    def test_negated_index(self, gen):
        node = idx("-k")
        assert gen._collect_index_vars(node) == {"k"}

    def test_sum_index(self, gen):
        node = idx("k+q")
        assert gen._collect_index_vars(node) == {"k", "q"}

    def test_product_index(self, gen):
        node = idx("a*k+b")
        # "a", "k", "b" are all alpha tokens
        assert gen._collect_index_vars(node) >= {"k"}

    def test_commutator_collects_all(self, gen):
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        vars_ = gen._collect_index_vars(node)
        assert "k" in vars_
        assert "q" in vars_

    def test_index_sum_collects_dummy(self, gen):
        body = indexed("c'", "k")
        node = IndexSum(body=body, dummy_indices=[idx("k")])
        vars_ = gen._collect_index_vars(node)
        assert "k" in vars_


# ---------------------------------------------------------------------------
# _to_sympy_op_name
# ---------------------------------------------------------------------------

class TestToSympyOpName:
    def test_annihilation_c(self, gen):
        assert gen._to_sympy_op_name("c") == "c"

    def test_creation_c_prime(self, gen):
        assert gen._to_sympy_op_name("c'") == "cd"

    def test_creation_c_dagger(self, gen):
        assert gen._to_sympy_op_name("c†") == "cd"

    def test_boson_annihilation(self, gen):
        assert gen._to_sympy_op_name("a") == "a"

    def test_boson_creation_prime(self, gen):
        assert gen._to_sympy_op_name("a'") == "ad"

    def test_fermion_d_family(self, gen):
        assert gen._to_sympy_op_name("d") == "c"
        assert gen._to_sympy_op_name("d'") == "cd"

    def test_boson_b_family(self, gen):
        assert gen._to_sympy_op_name("b") == "a"
        assert gen._to_sympy_op_name("b'") == "ad"

    def test_force_creation(self, gen):
        assert gen._to_sympy_op_name("c", force_creation=True) == "cd"
        assert gen._to_sympy_op_name("a", force_creation=True) == "ad"


# ---------------------------------------------------------------------------
# _serialize_single_op_to_sympy
# ---------------------------------------------------------------------------

class TestSerializeSingleOp:
    def test_indexed_creation(self, gen):
        node = indexed("c'", "k")
        assert gen._serialize_single_op_to_sympy(node) == "cd(k)"

    def test_indexed_annihilation(self, gen):
        node = indexed("c", "q")
        assert gen._serialize_single_op_to_sympy(node) == "c(q)"

    def test_indexed_negated(self, gen):
        node = indexed("c'", "-k")
        assert gen._serialize_single_op_to_sympy(node) == "cd(-k)"

    def test_indexed_sum_index(self, gen):
        node = indexed("c'", "k+q")
        assert gen._serialize_single_op_to_sympy(node) == "cd(k+q)"

    def test_function_call_creation(self, gen):
        node = fcall("c'", "k")
        assert gen._serialize_single_op_to_sympy(node) == "cd(k)"

    def test_function_call_annihilation(self, gen):
        node = fcall("c", "q")
        assert gen._serialize_single_op_to_sympy(node) == "c(q)"

    def test_boson_creation(self, gen):
        node = indexed("a'", "k")
        assert gen._serialize_single_op_to_sympy(node) == "ad(k)"

    def test_boson_annihilation(self, gen):
        node = indexed("a", "q")
        assert gen._serialize_single_op_to_sympy(node) == "a(q)"

    def test_adjoint_function_call(self, gen):
        inner = FunctionCall(func=sym("c"), args=[idx("k")])
        node = Adjoint(expr=inner)
        result = gen._serialize_single_op_to_sympy(node)
        assert result == "cd(k)"

    def test_adjoint_indexed_symbol(self, gen):
        inner = IndexedSymbol(base=sym("c"), indices=[idx("k")])
        node = Adjoint(expr=inner)
        result = gen._serialize_single_op_to_sympy(node)
        assert result == "cd(k)"


# ---------------------------------------------------------------------------
# _serialize_lhs_to_sympy
# ---------------------------------------------------------------------------

class TestSerializeLhs:
    def test_single_operator(self, gen):
        node = indexed("c'", "k")
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)"
        assert extra == set()

    def test_mul_two_operators(self, gen):
        node = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*cd(-k)"

    def test_non_commutative_mul(self, gen):
        node = NonCommutativeMul(args=[indexed("c'", "k"), indexed("c", "k")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*c(k)"

    def test_index_sum_unwrapped(self, gen):
        """IndexSum wrapping: body is extracted, dummy vars added to extra_vars."""
        body = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        node = IndexSum(body=body, dummy_indices=[idx("k")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*cd(-k)"
        assert "k" in extra

    def test_three_operator_product(self, gen):
        node = Mul(args=[indexed("c'", "k"), indexed("c'", "p"), indexed("c", "p")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*cd(p)*c(p)"

    def test_scalar_coefficient_stripped(self, gen):
        """BCS regression: Mul([Symbol("Delta"), c'(k), c'(-k)]) → 'cd(k)*cd(-k)'.

        Symbol("Delta") is a scalar parameter; it must be stripped from the
        prod_str passed to the SymPy helper so it is not mistaken for an operator
        (i.e., must NOT become 'c(Delta)*cd(k)*cd(-k)').
        """
        from notebook.models.ir import Number
        node = Mul(args=[sym("Delta"), indexed("c'", "k"), indexed("c'", "-k")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*cd(-k)", (
            f"Scalar coefficient Delta must be stripped; got: {result}"
        )

    def test_number_coefficient_stripped(self, gen):
        """Numeric factor (e.g. 2) is stripped from the operator product."""
        from notebook.models.ir import Number
        node = Mul(args=[Number(value=2), indexed("c'", "k"), indexed("c", "k")])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == "cd(k)*c(k)"

    def test_scalar_only_mul_returns_empty(self, gen):
        """A Mul with only scalar factors returns empty string (no operators)."""
        from notebook.models.ir import Number
        node = Mul(args=[sym("Delta"), Number(value=2)])
        extra: set = set()
        result = gen._serialize_lhs_to_sympy(node, extra)
        assert result == ""


# ---------------------------------------------------------------------------
# _detect_statistics
# ---------------------------------------------------------------------------

class TestDetectStatistics:
    def test_fermion_by_name_c(self, gen):
        lhs = indexed("c'", "k")
        assert gen._detect_statistics(lhs) == "fermion"

    def test_fermion_by_name_c_prime(self, gen):
        lhs = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        assert gen._detect_statistics(lhs) == "fermion"

    def test_boson_by_name_a(self, gen):
        lhs = indexed("a'", "k")
        assert gen._detect_statistics(lhs) == "boson"

    def test_fermion_d_family(self, gen):
        lhs = indexed("d'", "k")
        assert gen._detect_statistics(lhs) == "fermion"

    def test_boson_b_family(self, gen):
        lhs = indexed("b'", "k")
        assert gen._detect_statistics(lhs) == "boson"

    def test_default_is_fermion(self, gen):
        """Unknown operator name → fallback to fermion."""
        lhs = indexed("z'", "k")
        # 'z' is not in any known family → default "fermion"
        assert gen._detect_statistics(lhs) == "fermion"


# ---------------------------------------------------------------------------
# _generate_sympy_commutator  (full call-string, no Julia execution)
# ---------------------------------------------------------------------------

class TestGenerateSympyCommutator:
    def test_bcs_pairing_term(self, gen):
        """[c†_k c†_{-k}, c_q] → _nb_sympy_fermion_comm("cd(k)*cd(-k)", "c(q)", "k,q")"""
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        result = gen._generate_sympy_commutator(node)
        assert '_nb_sympy_fermion_comm(' in result
        assert '"cd(k)*cd(-k)"' in result
        assert '"c(q)"' in result
        assert "k" in result and "q" in result

    def test_kinetic_term(self, gen):
        """[c†_k c_k, c_q] → _nb_sympy_fermion_comm("cd(k)*c(k)", "c(q)", "k,q")"""
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c", "k")]),
            b=indexed("c", "q"),
        )
        result = gen._generate_sympy_commutator(node)
        assert '_nb_sympy_fermion_comm(' in result
        assert '"cd(k)*c(k)"' in result
        assert '"c(q)"' in result

    def test_boson_bec_term(self, gen):
        """[a†_k a†_{-k}, a_q] → _nb_sympy_boson_comm"""
        node = Commutator(
            a=Mul(args=[indexed("a'", "k"), indexed("a'", "-k")]),
            b=indexed("a", "q"),
        )
        result = gen._generate_sympy_commutator(node)
        assert '_nb_sympy_boson_comm(' in result
        assert '"ad(k)*ad(-k)"' in result
        assert '"a(q)"' in result

    def test_generate_dispatches_for_algebraic(self, gen):
        """generate(Commutator) with algebraic index → SymPy path."""
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        result = gen.generate(node)
        assert '_nb_sympy_fermion_comm(' in result

    def test_generate_uses_quantumalgebra_for_simple(self, gen):
        """generate(Commutator) with simple indices → comm() QuantumAlgebra path."""
        node = Commutator(
            a=indexed("c'", "k"),
            b=indexed("c", "q"),
        )
        result = gen.generate(node)
        assert result.startswith("comm(")
        assert "_nb_sympy" not in result

    def test_vars_str_sorted(self, gen):
        """vars_str must be sorted so output is deterministic."""
        node = Commutator(
            a=Mul(args=[indexed("c'", "k"), indexed("c'", "-k")]),
            b=indexed("c", "q"),
        )
        result = gen._generate_sympy_commutator(node)
        # Extract the third argument (vars_str)
        import re
        m = re.search(r'"([^"]*)"[^"]*$', result)
        assert m is not None
        vars_str = m.group(1)
        parts = vars_str.split(",")
        assert parts == sorted(parts), f"vars_str not sorted: {vars_str}"

    def test_function_call_form(self, gen):
        """Operators as FunctionCall(Symbol('c''), ...) also serialize correctly."""
        node = Commutator(
            a=Mul(args=[fcall("c'", "k"), fcall("c'", "-k")]),
            b=fcall("c", "q"),
        )
        result = gen._generate_sympy_commutator(node)
        assert '_nb_sympy_fermion_comm(' in result
        assert '"cd(k)*cd(-k)"' in result
        assert '"c(q)"' in result

    def test_index_sum_wrapping(self, gen):
        """IndexSum wrapping the product is unwrapped; dummy var added to vars_str."""
        body = Mul(args=[indexed("c'", "k"), indexed("c'", "-k")])
        lhs = IndexSum(body=body, dummy_indices=[idx("k")])
        node = Commutator(a=lhs, b=indexed("c", "q"))
        result = gen._generate_sympy_commutator(node)
        assert '_nb_sympy_fermion_comm(' in result
        assert '"cd(k)*cd(-k)"' in result
        assert '"c(q)"' in result


# ---------------------------------------------------------------------------
# _maybe_expand_commutator — Hamiltonian distribution tests
# ---------------------------------------------------------------------------

class TestMaybeExpandCommutator:
    """Tests for _maybe_expand_commutator with multi-term Hamiltonians."""

    def _mk_kinetic(self) -> "NonCommutativeMul":
        """ξ_k c†_k c_k — kinetic term."""
        return NonCommutativeMul(args=[
            indexed("xi", "k"),
            indexed("c'", "k"),
            indexed("c", "k"),
        ])

    def _mk_kinetic_minus_k(self) -> "NonCommutativeMul":
        """ξ_k c†_{-k} c_{-k} — kinetic term for -k mode."""
        return NonCommutativeMul(args=[
            indexed("xi", "k"),
            indexed("c'", "-k"),
            indexed("c", "-k"),
        ])

    def _mk_pairing(self) -> "NonCommutativeMul":
        """Δ c†_k c†_{-k} — BCS pairing term."""
        return NonCommutativeMul(args=[
            sym("Delta"),
            indexed("c'", "k"),
            indexed("c'", "-k"),
        ])

    def test_lhs_add_with_algebraic_indices_uses_sum_comm(self, gen):
        """[A + B, c_q] where A or B has algebraic index → _nb_sympy_sum_*_comm."""
        kinetic = self._mk_kinetic()
        kinetic_mk = self._mk_kinetic_minus_k()
        hamiltonian = Add(args=[kinetic, kinetic_mk])
        comm_node = Commutator(a=hamiltonian, b=indexed("c", "q"))
        result = gen.generate(comm_node)
        assert "_nb_sympy_sum_fermion_comm(" in result, result

    def test_lhs_indexsum_add_uses_sum_comm(self, gen):
        """[IndexSum(A + B), c_q] expands to per-term SymPy calls."""
        kinetic = self._mk_kinetic()
        kinetic_mk = self._mk_kinetic_minus_k()
        body = Add(args=[kinetic, kinetic_mk])
        lhs = IndexSum(body=body, dummy_indices=[idx("k")])
        comm_node = Commutator(a=lhs, b=indexed("c", "q"))
        result = gen.generate(comm_node)
        assert "_nb_sympy_sum_fermion_comm(" in result, result

    def test_scalar_distributed_over_add(self, gen):
        """ξ_k*(c†_k c_k + c†_{-k} c_{-k}) → two separate (coeff, op) pairs."""
        inner_add = Add(args=[
            NonCommutativeMul(args=[indexed("c'", "k"), indexed("c", "k")]),
            NonCommutativeMul(args=[indexed("c'", "-k"), indexed("c", "-k")]),
        ])
        term = NonCommutativeMul(args=[indexed("xi", "k"), inner_add])
        terms = gen._collect_hamiltonian_terms(term)
        # Should produce 2 pairs (one per addend), not 1 pair with Add as op
        assert len(terms) == 2, f"expected 2 terms, got {len(terms)}: {terms}"
        coeffs = [c for c, _ in terms]
        assert all("xi" in c for c in coeffs), coeffs
        # Each operator part should be a simple product, not an Add
        for _, op in terms:
            assert not isinstance(op, Add), f"op should not be Add: {op}"

    def test_lhs_simple_no_algebraic_does_not_expand(self, gen):
        """[c†_k c_k, c_q] — no algebraic index → fall through to QuantumAlgebra."""
        comm_node = Commutator(
            a=NonCommutativeMul(args=[indexed("c'", "k"), indexed("c", "k")]),
            b=indexed("c", "q"),
        )
        result = gen.generate(comm_node)
        assert result.startswith("comm("), result
        assert "_nb_sympy" not in result

    def test_bcs_hamiltonian_sum_structure(self, gen):
        """Full BCS kinetic + pairing structure → sum_comm with all terms."""
        kinetic1 = NonCommutativeMul(args=[indexed("xi","k"), indexed("c'","k"), indexed("c","k")])
        kinetic2 = NonCommutativeMul(args=[indexed("xi","k"), indexed("c'","-k"), indexed("c","-k")])
        pairing  = NonCommutativeMul(args=[sym("Delta"), indexed("c'","k"), indexed("c'","-k")])
        hamiltonian = Add(args=[kinetic1, kinetic2, pairing])
        lhs_sum = IndexSum(body=hamiltonian, dummy_indices=[idx("k")])
        comm_node = Commutator(a=lhs_sum, b=indexed("c", "q"))
        result = gen.generate(comm_node)
        assert "_nb_sympy_sum_fermion_comm(" in result, result
        # All three terms should appear (pairing has Delta coeff, kinetic has xi coeff)
        assert "Delta" in result or "cd(k)*cd(-k)" in result, result
        assert "xi" in result or "cd(k)*c(k)" in result, result


# ---------------------------------------------------------------------------
# Equation-unwrapping fix: "H = rhs" in input_lookup → expand RHS, not H=rhs
# ---------------------------------------------------------------------------

class TestEquationUnwrapping:
    """_maybe_expand_commutator Case 1 must unwrap 'H = Σ_k...' equations.

    Before the fix, the full Equation IR was passed to _collect_hamiltonian_terms
    which returned a single garbage term (the equation node as an operator), causing
    no SymPy routing.  After the fix, the RHS is extracted and terms are collected
    correctly, producing the expected _nb_sympy_sum_fermion_comm call.
    """

    def _gen_with_bcs_context(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity
        ctx = SymbolContext()
        ctx.session_mode = "second_quantization"
        ctx.domains = ["condensed_matter"]
        ctx.register("c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        ctx.register("c'", Quantity(name="c'", kind="operator"), statistics_kind="fermi")
        ctx.register("xi", Quantity(name="xi", kind="scalar"))
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))
        gen = JuliaCodeGenerator(context=ctx)
        return gen

    def test_equation_form_unwrapped(self):
        """'H = \\sum_k...' canonical in input_lookup → RHS extracted → SymPy routing."""
        from notebook.models.ir import Commutator, IndexedSymbol, Symbol, IndexSymbol

        gen = self._gen_with_bcs_context()
        # Actual session input_canonical format
        bcs_canon = (
            r"H = \sum_k*xi_k*(c'_k*c_k + c'_-k*c_-k)"
            r" - \sum_k*(Delta*c'_k*c'_-k + Delta^(\ast)*c_-k*c_k)"
        )
        gen.input_lookup = {"E1": bcs_canon}

        target = IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="k")])
        comm = Commutator(a=Symbol(name="_nb_E1"), b=target)
        code = gen.generate(comm)

        # Must route to SymPy sum commutator (not QuantumAlgebra comm())
        assert "_nb_sympy_sum_fermion_comm(" in code, (
            f"Expected SymPy sum routing for equation-form Hamiltonian; got: {code}"
        )

    def test_equation_form_correct_terms(self):
        """4 correct terms (kinetic×2, pairing, reverse) are collected from 'H = ...'."""
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.models.ir import Equation, Assignment

        gen = self._gen_with_bcs_context()
        bcs_canon = (
            r"H = \sum_k*xi_k*(c'_k*c_k + c'_-k*c_-k)"
            r" - \sum_k*(Delta*c'_k*c'_-k + Delta^(\ast)*c_-k*c_k)"
        )

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        ir = resolver.resolve(parser.parse(bcs_canon), gen.context)

        if isinstance(ir, (Equation, Assignment)):
            ir = ir.rhs

        terms = gen._collect_hamiltonian_terms(ir)
        assert len(terms) == 4, f"Expected 4 BCS terms; got {len(terms)}"

        extra = set()
        ops_strs = [gen._serialize_lhs_to_sympy(op, extra) for _, op in terms]
        assert "cd(k)*c(k)" in ops_strs, f"Missing kinetic term; got: {ops_strs}"
        assert "cd(-k)*c(-k)" in ops_strs, f"Missing mirror kinetic; got: {ops_strs}"
        # After alphatize correctly renames the second sum's dummy k→j, the
        # pairing terms use j instead of k.  Accept either index name.
        pairing = [s for s in ops_strs if "cd" in s and "c(" not in s.replace("cd", "")]
        assert pairing, f"Missing pairing term; got: {ops_strs}"

    def test_no_xi_j_in_generated_code(self):
        """Generated code must not contain xi_j (the core ξ_j bug)."""
        from notebook.models.ir import Commutator, IndexedSymbol, Symbol, IndexSymbol

        gen = self._gen_with_bcs_context()
        bcs_canon = (
            r"H = \sum_k*xi_k*(c'_k*c_k + c'_-k*c_-k)"
            r" - \sum_k*(Delta*c'_k*c'_-k + Delta^(\ast)*c_-k*c_k)"
        )
        gen.input_lookup = {"E1": bcs_canon}

        target = IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="k")])
        comm = Commutator(a=Symbol(name="_nb_E1"), b=target)
        code = gen.generate(comm)

        assert "xi_j" not in code, (
            f"ξ_j must not appear in generated code; got: {code}"
        )
        assert "xi_k" in code, f"xi_k must appear in kinetic term; got: {code}"


# ---------------------------------------------------------------------------
# Test: build_command commutator intercept (production code path)
# ---------------------------------------------------------------------------

class TestBuildCommandCommutatorIntercept:
    """The real production code path: build_command intercepts commutator and
    routes to _nb_sympy_sum_fermion_comm instead of the QuantumAlgebra template.

    Previously _maybe_expand_commutator was only reachable via Commutator IR
    nodes, which the template-based system never creates (it generates
    'normal_form(comm({a},{b}))' directly).  The fix adds an intercept in
    build_command that creates a temporary Commutator IR node and calls
    _maybe_expand_commutator before falling through to the template.
    """

    def _make_handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.julia_command_builder import JuliaCommandBuilder
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity

        ctx = SymbolContext()
        ctx.session_mode = "second_quantization"
        ctx.domains = ["condensed_matter"]
        ctx.register("c_prime", Quantity(name="c'", kind="operator"), statistics_kind="fermi")
        ctx.register("c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        ctx.register("xi", Quantity(name="xi", kind="scalar"))
        ctx.register("Delta", Quantity(name="Delta", kind="scalar"))

        handler = JuliaBackendHandler(JuliaCommandBuilder())
        handler.set_symbol_context(ctx)
        return handler

    def _bcs_expr_lookup(self):
        canon = (
            r"H = \sum_k*xi_k*(c'_k*c_k + c'_-k*c_-k)"
            r" - \sum_k*(Delta*c'_k*c'_-k + Delta^(\ast)*c_-k*c_k)"
        )
        return {"E1": canon, "__input__E1": canon}

    def test_commutator_routes_to_sympy_not_qalg(self):
        """build_command('commutator', {'a':'E1','b':'c_k'}) → SymPy helper, not comm()."""
        handler = self._make_handler()
        result = handler.build_command(
            "commutator", {"a": "E1", "b": "c_k"}, self._bcs_expr_lookup()
        )
        assert result.ok, f"build_command failed: {result.error}"
        assert "_nb_sympy_sum_fermion_comm(" in result.command, (
            f"Expected SymPy routing; got: {result.command[:200]}"
        )
        assert "normal_form(comm(" not in result.command, (
            f"Must not fall through to QuantumAlgebra; got: {result.command[:200]}"
        )

    def test_commutator_has_no_xi_j(self):
        """ξ_j must not appear in the build_command output."""
        handler = self._make_handler()
        result = handler.build_command(
            "commutator", {"a": "E1", "b": "c_k"}, self._bcs_expr_lookup()
        )
        assert result.ok
        assert "xi_j" not in result.command, (
            f"xi_j (ξ_j bug) must not appear; got: {result.command}"
        )
        assert "xi_k" in result.command

    @staticmethod
    def _parse_terms_from_cmd(cmd: str):
        """Extract and parse the JSON terms array from a _nb_sympy_sum_*_comm call."""
        import json
        # Command format: _nb_sympy_sum_fermion_comm("[{\"coeff\":...}]", ...)
        # The JSON is the first string arg; internal quotes are escaped as \"
        # Find the content between the outer quotes: "[...  ]"
        start = cmd.index('"[') + 1      # skip the opening "
        end = cmd.rindex(']"') + 1       # include up to ]
        raw = cmd[start:end]             # raw = '[{\"coeff\":...}]'
        # Unescape \" → " so json.loads can parse it
        unescaped = raw.replace('\\"', '"')
        return json.loads(unescaped)

    def test_commutator_four_bcs_terms(self):
        """Generated JSON must contain exactly 4 BCS terms."""
        handler = self._make_handler()
        result = handler.build_command(
            "commutator", {"a": "E1", "b": "c_k"}, self._bcs_expr_lookup()
        )
        assert result.ok
        terms = self._parse_terms_from_cmd(result.command)
        assert len(terms) == 4, f"Expected 4 BCS terms; got {len(terms)}: {terms}"

    def test_compound_coeff_in_terms(self):
        """Pairing term must have coeff='-1*Delta', not a garbled sp.Symbol name."""
        handler = self._make_handler()
        result = handler.build_command(
            "commutator", {"a": "E1", "b": "c_k"}, self._bcs_expr_lookup()
        )
        assert result.ok
        terms = self._parse_terms_from_cmd(result.command)
        coeffs = [t["coeff"] for t in terms]
        assert "-1*Delta" in coeffs, f"Pairing coeff must be '-1*Delta'; got: {coeffs}"
        assert "-1*conjugate_Delta" in coeffs, (
            f"Reverse pairing coeff must be '-1*conjugate_Delta'; got: {coeffs}"
        )

    def _make_spin_handler(self):
        """JuliaBackendHandler with S, Sx, Sy, Sz registered."""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        for name in ("S", "Sx", "Sy", "Sz"):
            ctx.register_quantity(name, kind="operator")
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = ctx
        return h, ctx

    def test_spin_component_subscript_not_flagged_indexed_symbol(self):
        """IndexedSymbol path: S_x where Sx is registered must not flag x."""
        from notebook.models.ir import IndexedSymbol, Symbol, IndexSymbol
        h, ctx = self._make_spin_handler()
        ast = IndexedSymbol(base=Symbol(name="S"), indices=[IndexSymbol(name="x")])
        assert h.find_undefined_symbols(ast, ctx) == [], \
            "S_x (IndexedSymbol) should not flag x when Sx is registered"

    def test_spin_component_subscript_not_flagged_surface_ast(self):
        """SharedParser path: [H, S_x] → Subscript(S, x) — x must not be flagged."""
        from notebook.runtime.shared_parser import SharedParser
        h, ctx = self._make_spin_handler()
        ast = SharedParser().parse("[H, S_x]")
        result = h.find_undefined_symbols(ast, ctx)
        assert "x" not in result, f"x should not be flagged; got: {result}"

    def test_underscore_fullname_subscript_not_flagged(self):
        """L-1x: a forward-transform LHS like alpha_mkd is registered as a whole,
        so the Subscript(alpha, mkd) must not flag/block its multi-char subscript."""
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        for name in ("alpha_k", "alpha_mkd", "u", "v"):
            ctx.register_quantity(name, kind="scalar")
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = ctx
        ast = SharedParser().parse("alpha_k*u + alpha_mkd*v")
        # the subscripts must NOT be flagged as independent symbols …
        result = h.find_undefined_symbols(ast, ctx)
        assert "mkd" not in result and "k" not in result, result
        # … and therefore must not block CAS execution (the L-1x bug)
        assert "mkd" not in h.blocks_on_undefined_symbols(ast, ctx)

    def test_spin_site_index_still_flagged(self):
        """S_k (site index k) IS flagged because Sk is not a registered component."""
        from notebook.models.ir import IndexedSymbol, Symbol, IndexSymbol
        h, ctx = self._make_spin_handler()
        ast = IndexedSymbol(base=Symbol(name="S"), indices=[IndexSymbol(name="k")])
        result = h.find_undefined_symbols(ast, ctx)
        assert result != [], "S_k (site index) should still be flagged"

    def test_simple_commutator_falls_through_to_qalg(self):
        """Simple [c(k), c(k)] with no Hamiltonian structure → QuantumAlgebra template."""
        handler = self._make_handler()
        # No E1 in expr_lookup; direct simple operators
        result = handler.build_command(
            "commutator", {"a": "c_k", "b": "c_k"}, {}
        )
        assert result.ok
        # Should use QuantumAlgebra (no algebraic indices in c_k*c_k)
        assert "normal_form(comm(" in result.command, (
            f"Simple commutator should use QuantumAlgebra; got: {result.command}"
        )
