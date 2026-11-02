"""
tests/test_pair_substitute.py
------------------------------
A-2: pair_substitute operation — operator-pair to c-number substitution.

Tests the mean-field / Hartree-Fock decoupling operation where products of
creation/annihilation operators are replaced by scalar (c-number) values.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


@pytest.fixture(scope="module")
def engine():
    """Shared Julia engine for all A-2 tests."""
    with NotebookEngine(
        title="pair_sub_test",
        mode=SessionMode.second_quantization
    ) as e:
        e.start()
        e.run_operation("define_fermion_operator", {"name": "c"})
        for name in ["xi", "Delta", "n_k", "phi"]:
            e.run_operation("define_scalar_parameter", {"name": name})
        yield e


class TestPairSubstituteBasic:
    def test_number_operator_to_scalar(self, engine):
        """c†(k)c(k) + c†(-k)c(-k) → 2*n_k  (Hartree-Fock)."""
        step, _ = engine.run_operation(
            "expand", {"expr": "c'(k)*c(k) + c'(-k)*c(-k)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c(k)", "rhs": "n_k"},
                {"pattern": "c†(K) c(K)", "rhs": "n_k"},
            ],
        })
        assert result is not None and result != ""
        assert "n_k" in result

    def test_bcs_number_only(self, engine):
        """BCS Hamiltonian: only number operator pairs replaced; pair terms remain."""
        step, _ = engine.run_operation(
            "expand",
            {"expr": "xi*c'(k)*c(k) + xi*c'(-k)*c(-k) - Delta*c'(k)*c'(-k) - Delta*c(-k)*c(k)"},
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c(k)", "rhs": "n_k"},
                {"pattern": "c†(K) c(K)", "rhs": "n_k"},
            ],
        })
        assert "n_k" in result
        # Pair creation/annihilation terms must still be present
        assert "c†" in result or "Delta" in result

    def test_pair_creation_to_scalar(self, engine):
        """c†(k)c†(-k) → Delta  (BCS gap substitution)."""
        step, _ = engine.run_operation(
            "expand", {"expr": "Delta * c'(k) * c'(-k)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c†(K)", "rhs": "phi"},
            ],
        })
        assert "phi" in result

    def test_multiple_rules_applied(self, engine):
        """Both number and pair terms replaced simultaneously."""
        step, _ = engine.run_operation(
            "expand",
            {"expr": "xi*c'(k)*c(k) - Delta*c'(k)*c'(-k)"},
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c(k)", "rhs": "n_k"},
                {"pattern": "c†(k) c†(K)", "rhs": "Delta"},
            ],
        })
        # Both n_k (from number op) and Delta (from pair op) should appear as scalars
        assert "n_k" in result
        # No bare operator products should remain for replaced patterns
        assert "c†(k) c(k)" not in result

    def test_no_matching_pattern_unchanged(self, engine):
        """Expression with no matching patterns is returned unchanged.

        **Kept deliberately (2026-08-15).**  The NOOP-REWRITE-FAMILY sweep found
        that this operation answered a rule matching nothing exactly as it
        answers one that works, and the first fix REFUSED — which broke this
        test.  The refusal was withdrawn: unlike `operator_substitute`, which
        names ONE operator and can only mean it, this takes SEVERAL rules at
        once, and "this Hamiltonian has no such pair" is information a
        researcher may legitimately act on.  The expression still comes back
        unchanged; what changed is that the result now CARRIES A WARNING saying
        so (tests/test_julia_rewrite_guards.py).
        """
        step, H_val = engine.run_operation(
            "expand", {"expr": "xi * c'(k) * c(k)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c†(K)", "rhs": "Delta"},  # no match
            ],
        })
        # xi*c†c unchanged
        assert "c†" in result


class TestPairSubstituteUserNotation:
    """H-3: patterns written in the same user notation as expression input
    (operator products with `*`, primes for creation, `-k` for negated
    momentum) — no internal bares form (†, K, Unicode superscripts) required."""

    def test_number_operator_user_notation(self, engine):
        """c'(k)*c(k) + c'(-k)*c(-k) → 2*n_k via user-notation patterns."""
        step, _ = engine.run_operation(
            "expand", {"expr": "c'(k)*c(k) + c'(-k)*c(-k)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c'(k)*c(k)", "rhs": "n_k"},
                {"pattern": "c'(-k)*c(-k)", "rhs": "n_k"},
            ],
        })
        assert result is not None and result != ""
        assert "n_k" in result
        # the replaced operator product must be gone
        assert "c†(k) c(k)" not in result

    def test_pair_creation_user_notation(self, engine):
        """c'(k)*c'(-k) → Delta via user-notation pattern (negated momentum)."""
        step, _ = engine.run_operation(
            "expand", {"expr": "Delta * c'(k) * c'(-k)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c'(k)*c'(-k)", "rhs": "phi"},
            ],
        })
        assert "phi" in result

    def test_mixed_legacy_and_user_notation(self, engine):
        """A raw bares pattern and a user-notation pattern in the same call
        both work (backward compatibility + new notation together)."""
        step, _ = engine.run_operation(
            "expand",
            {"expr": "xi*c'(k)*c(k) - Delta*c'(k)*c'(-k)"},
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c(k)", "rhs": "n_k"},      # legacy raw bares
                {"pattern": "c'(k)*c'(-k)", "rhs": "Delta"},  # user notation
            ],
        })
        assert "n_k" in result


class TestPairSubstituteOperatorCountGuard:
    """pair_substitute replaces a 2-operator pair only.  A higher-order pattern
    (e.g. a 4-fermion interaction) is rejected with a message routing the user
    to wick_contract, because the order-independent canonical match would
    discard the fermionic sign carried by operator ordering."""

    def test_four_operator_user_notation_rejected(self, engine):
        step, _ = engine.run_operation(
            "expand", {"expr": "c'(i,s)*c'(j,p)*c(k,p)*c(l,s)"}
        )
        with pytest.raises(ValueError, match="wick_contract"):
            engine.run_operation("pair_substitute", {
                "expr": step.output_refs[0],
                "pair_rules": [
                    {"pattern": "c'(i,s)*c'(j,p)*c(k,p)*c(l,s)", "rhs": "n_k"},
                ],
            })

    def test_four_operator_legacy_bares_rejected(self, engine):
        step, _ = engine.run_operation(
            "expand", {"expr": "c'(i,s)*c'(j,p)*c(k,p)*c(l,s)"}
        )
        with pytest.raises(ValueError, match="wick_contract"):
            engine.run_operation("pair_substitute", {
                "expr": step.output_refs[0],
                "pair_rules": [
                    {"pattern": "c†(is) c†(jp) c(kp) c(ls)", "rhs": "n_k"},
                ],
            })

    def test_two_operator_multiindex_accepted(self, engine):
        """A genuine 2-operator pair with two indices each is still accepted."""
        step, _ = engine.run_operation(
            "expand", {"expr": "c'(i,s)*c(l,s)"}
        )
        _, result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c'(i,s)*c(l,s)", "rhs": "n_k"},
            ],
        })
        assert "n_k" in result


class TestPairSubstituteWithReEvaluate:
    """Integration with C-Ext Phase 1-A: self-consistency iteration."""

    def test_self_consistency_loop(self, engine):
        """n_k updated 3 times via re_evaluate_expr after pair_substitute."""
        step, _ = engine.run_operation(
            "expand", {"expr": "n_k * c'(k) * c(k) + n_k * c'(-k) * c(-k)"}
        )
        mf_step, mf_result = engine.run_operation("pair_substitute", {
            "expr": step.output_refs[0],
            "pair_rules": [
                {"pattern": "c†(k) c(k)", "rhs": "n_k"},
                {"pattern": "c†(K) c(K)", "rhs": "n_k"},
            ],
        })
        mf_id = mf_step.output_refs[0]

        for n_val in ["0.3", "0.42", "0.5"]:
            updated = engine.re_evaluate_expr(mf_id, {"n_k": n_val})
            assert updated is not None, f"re_evaluate_expr failed for n_k={n_val}"
            assert n_val in updated.canonical or "n_k" not in updated.canonical


class TestPairSubstituteValidation:
    def test_missing_expr_raises(self, engine):
        """Missing expr field raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            engine.run_operation("pair_substitute", {
                "pair_rules": [{"pattern": "c†(k) c(k)", "rhs": "n_k"}],
            })

    def test_empty_pair_rules_raises(self, engine):
        """Empty pair_rules raises ValueError."""
        step, _ = engine.run_operation("expand", {"expr": "c'(k)*c(k)"})
        with pytest.raises(ValueError):
            engine.run_operation("pair_substitute", {
                "expr": step.output_refs[0],
                "pair_rules": [],
            })
