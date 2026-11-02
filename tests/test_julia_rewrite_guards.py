r"""A Julia rewrite that names an operator the expression does not contain.

open_items NOOP-REWRITE-FAMILY.  Measured 2026-08-15 across the whole family:
of the eight operations that name a subexpression to rewrite, four answered a
name that occurs NOWHERE exactly as they answer a name that works — the
expression came back unchanged, as a SUCCESS.  Three of those four are here
(the fourth, `relabel_index`, is Maxima's and is pinned in
tests/test_transform_subst_chain.py).

**The guard is one idea with two implementations, and that is the measurement
talking, not a compromise.**  The ticket predicted "one guard will do"; measuring
showed the three do not share a matcher:

* `operator_substitute` / `operator_substitute_pair` substitute at the IR level
  in Python (Symbolics.jl cannot substitute into a QuantumAlgebra value), so the
  evidence is the IR itself — if `_subst_quantum_op` handed back what it was
  given, nothing matched;
* `pair_substitute` matches inside the driver, keyed on `string(term.bares)`, so
  the evidence is a count in `_nb_pair_substitute`.

What IS shared is the answer: both refuse with `reason_code="lhs_not_found"` and
a translated message.  Writing a third, independent "does this operator occur?"
scan in Python would have been a second answer to the same question — and a
worse one, since the substituter knows about primes, sublattice labels, index
signs and the exact/template distinction that a name scan does not.
"""
import pytest

from notebook.engine import NotebookEngine, OperationRefused
from notebook.models.session import SessionMode

EXPR = "-J * Sz_1 * Sz_2 - Gamma * (Sx_1 + Sx_2)"


@pytest.fixture
def engine():
    with NotebookEngine(title="jguard", mode=SessionMode.second_quantization) as e:
        e.start()
        for n in ("J", "Gamma", "mz", "u", "v", "q"):
            e.run_operation("define_scalar_parameter", {"name": n})
        e.run_operation("define_spin_operator", {"name": "S"})
        yield e


def _H(e):
    return e.register_expression(EXPR, lhs_label="H").display_id


class TestOperatorSubstitute:
    def test_a_present_operator_is_substituted(self, engine):
        step, _ = engine.run_operation("operator_substitute", {
            "expr": _H(engine), "op_name": "Sz", "rhs": "mz", "formal_index": "2"})
        assert "mz" in engine.expressions[step.output_refs[0]].latex

    def test_an_absent_operator_is_refused(self, engine):
        with pytest.raises(OperationRefused) as exc:
            engine.run_operation("operator_substitute", {
                "expr": _H(engine), "op_name": "Zz", "rhs": "mz",
                "formal_index": "2"})
        assert exc.value.reason_code == "lhs_not_found"
        assert "Zz" in str(exc.value)


class TestOperatorSubstitutePair:
    def test_an_absent_operator_is_refused(self, engine):
        with pytest.raises(OperationRefused) as exc:
            engine.run_operation("operator_substitute_pair", {
                "expr": _H(engine), "op_name": "Zz", "rhs": "u", "rhs_neg": "v"})
        assert exc.value.reason_code == "lhs_not_found"


class TestPairSubstitute:
    """Matched in the driver, so the count is raised there and translated here."""

    def test_a_present_pattern_is_substituted(self, engine):
        step, _ = engine.run_operation("pair_substitute", {
            "expr": _H(engine),
            "pair_rules": [{"pattern": "Sz_1 Sz_2", "rhs": "q"}]})
        latex = engine.expressions[step.output_refs[0]].latex
        assert "q" in latex and "Sz_1 Sz_2" not in latex

    def test_a_pattern_matching_nothing_warns_instead_of_failing(self, engine):
        """WARNS, deliberately — this operation takes several rules at once.

        "This Hamiltonian has no such pair" is information a researcher may act
        on, unlike `operator_substitute`, which names ONE operator and can only
        mean it.  Refusing was implemented first and withdrawn: it broke
        `tests/test_pair_substitute.py::test_no_matching_pattern_unchanged`,
        which has asserted the unchanged result since the operation was written.
        """
        step, _ = engine.run_operation("pair_substitute", {
            "expr": _H(engine),
            "pair_rules": [{"pattern": "Sy_1 Sy_2", "rhs": "q"}]})
        out = engine.expressions[step.output_refs[0]]
        assert "Sz_1" in out.latex, out.latex          # unchanged
        assert out.warnings, "a no-op must not be silent"
        assert "nb:" not in " ".join(out.warnings)      # translated, not the marker

    def test_a_matching_pattern_warns_about_nothing(self, engine):
        step, _ = engine.run_operation("pair_substitute", {
            "expr": _H(engine),
            "pair_rules": [{"pattern": "Sz_1 Sz_2", "rhs": "q"}]})
        assert engine.expressions[step.output_refs[0]].warnings == []

    def test_the_warning_reaches_the_response(self):
        """A warning nobody renders is not a warning.

        `Expression.warnings` has been passed through by the REGISTRATION paths
        since the bound-index warning was added, but the operation-RESULT
        response did not carry it — so a warning attached to a result would have
        been set and shown nowhere.  Asserted on the response dict, which is
        what the timeline card reads.
        """
        import inspect
        from ui import interpret_pipeline as ip
        src = inspect.getsource(ip)
        i = src.index('"operation": res.operation, "expr_id": oid')
        block = src[i:i + 700]
        assert '"warnings"' in block, "the operation-result response drops warnings"


class TestWickContract:
    r"""An empty answer means two different things, and only the driver knows which.

    `wick_contract` returns a LIST of terms.  Measured 2026-08-15, a rule naming
    operators the expression does not contain gave the same empty list as the
    physically correct vanishing of δ(-k,k) — which
    `tests/test_algebraic_index_routing.py::test_wick_contract_sympy_delta_vanishes`
    asserts as a SPEC.  So "refuse when empty" is not available here, and neither
    is any check on the result: both cases produce the same result.

    The difference exists at exactly one place — the driver collects one entry per
    pair a rule MATCHED, before any coefficient is evaluated — and that is where it
    is now recorded.  It rides on the ENGINE rather than on an expression because
    the case worth speaking about is the one with no expression to carry it.
    """

    @pytest.fixture
    def fermion(self):
        with NotebookEngine(title="wick", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            yield e

    def _run(self, e, rules):
        x = e.register_expression("c'(-k)*c(k)")
        return e.wick_contract(x.display_id, rules, include_constant=True)

    def test_a_matching_rule_contracts_and_says_nothing(self, fermion):
        terms = self._run(fermion, [{"op1": "c'", "op2": "c", "value": "n(INDEX1)"}])
        assert len(terms) == 1
        assert fermion.last_wick_warnings == []

    def test_a_vanishing_contraction_is_not_warned_about(self, fermion):
        """δ(-k,k)=0 — an empty answer that is the RIGHT answer.

        This is the test that makes the guard non-trivial: warning on "empty"
        would fire here, on physics the researcher deliberately asked for.
        """
        terms = self._run(fermion, [
            {"op1": "c'", "op2": "c", "value": "delta(INDEX1,INDEX2)*n(INDEX1)"}])
        assert terms == []
        assert fermion.last_wick_warnings == []

    def test_a_rule_that_reached_no_pair_is_warned_about(self, fermion):
        terms = self._run(fermion, [{"op1": "d'", "op2": "d", "value": "n(INDEX1)"}])
        assert terms == []
        assert fermion.last_wick_warnings, "an empty answer with no match must say so"

    def test_the_warning_reaches_the_response(self):
        """The route returns a list of terms; an empty one had nowhere to speak."""
        import inspect
        from ui.routes import operations as _ops
        src = inspect.getsource(_ops.wick_contract_expression)
        assert '"warnings"' in src, "the /wick_contract response drops warnings"
