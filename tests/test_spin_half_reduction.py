"""A-5d: same-site products on spin-½ sites are reducible, and reducing them
removes the wall rather than moving it.

The blowup was real — a 4-site chain's repeated commutators went
2 → 12 → 76 → 474 → 2758 and then hit the term limit, which the researcher saw
as a 3-minute failure.  But the growth is intrinsic to the mathematics (nested
commutators, Dyson series, high-order perturbation all do it), so "reduce the
terms" only helps if it changes the growth LAW.

For spin-½ it does, because the operator algebra is finite-dimensional
(4^N Pauli strings): S^a S^a = 1/4 and S^a S^b = (i/2)ε_abc S^c close the
single-site products, and the term count saturates.  Measured:
2 → 6 → 14 → 21 → 24 → 28 → 30, i.e. no wall at all.

**Both identities are spin-½ ONLY** — for general S, (S^a)² is still an
operator.  So the reduction fires only for sites the researcher has DECLARED as
spin-½ (F-13's `site_spins`), and the general-spin representation is untouched.
"""
from fractions import Fraction as F

import pytest

import notebook.runtime.spin_algebra as spa


HALF4 = {i: F(1, 2) for i in (1, 2, 3, 4)}


def _chain(n):
    return spa.parse_spin_expression(" + ".join(
        f"S^{a}_{i} S^{a}_{i+1}" for i in range(1, n) for a in "xyz"))


class TestTheWallIsGone:
    def test_unreduced_still_hits_the_limit(self):
        """Without declared spins the behaviour is exactly as before — the
        general-spin representation must not change."""
        H, cur = _chain(4), spa.parse_spin_expression("S^x_1")
        counts = []
        with pytest.raises(spa.ComplexityLimitExceeded):
            for _ in range(8):
                cur = spa.commutator(H, cur)
                counts.append(len(cur))
        assert counts[:5] == [2, 12, 76, 474, 2758], counts

    def test_declared_spin_half_saturates(self):
        H, cur = _chain(4), spa.parse_spin_expression("S^x_1")
        counts = []
        for _ in range(8):
            cur = spa.commutator(H, cur, HALF4)
            counts.append(len(cur))
        assert max(counts) < 100, counts
        assert counts[-1] <= counts[4] * 2, counts       # saturating, not growing


class TestThePhysicsIsUnchanged:
    """The decisive check: the reduction must not change any answer."""

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_moments_agree_with_and_without_reduction(self, order):
        H = _chain(3)
        O = spa.spin_operator("x", 1)
        half = {i: F(1, 2) for i in (1, 2, 3)}
        spins = {1: "1/2", 2: "1/2", 3: "1/2"}
        a = spa.nested_commutator(H, O, order) if order else O
        b = spa.nested_commutator(H, O, order, half) if order else O
        va = spa.expectation_general(spa.multiply(spa.dagger(a), a), spins)
        vb = spa.expectation_general(spa.multiply(spa.dagger(b), b), spins)
        assert va == vb, (order, va.to_latex(), vb.to_latex())

    @pytest.mark.parametrize("expr,expected", [
        ("S^x_1 S^x_1", F(1, 4)),
        ("S^x_1 S^y_1", F(0)),
        ("S^x_1 S^x_1 S^x_1 S^x_1", F(1, 16)),
    ])
    def test_same_site_expectations_unchanged(self, expr, expected):
        """These were already correct without the reduction — the trace handled
        them.  Pinned so the reduction cannot quietly change them."""
        e = spa.parse_spin_expression(expr)
        got = spa.expectation_general(e, {1: "1/2"})
        assert got == spa.Coeff({(): (expected, F(0))}), got.to_latex()


class TestItOnlyFiresWhereDeclared:
    def test_a_general_spin_site_is_left_alone(self):
        """`(S^z_2)^2` must stay symbolic for a site that is not declared ½ —
        the identity is not true there, and the 2-site scenario displays that
        unreduced form deliberately."""
        e = spa.parse_spin_expression("S^z_2 S^z_2")
        out = spa._collect(e, spa._reducible_sites({1: F(1, 2)}))   # site 2 not ½
        assert out[0][1] == (("z", 2), ("z", 2)), out

    def test_a_mixed_expression_is_not_reduced(self):
        """A term touching a non-½ site keeps its factors: reducing only part of
        it would mix two representations in one expression."""
        e = spa.parse_spin_expression("S^x_1 S^x_1 S^z_2 S^z_2")
        out = spa._collect(e, spa._reducible_sites({1: F(1, 2), 2: F(1)}))
        assert len(out[0][1]) == 4, out

    def test_spin_one_is_not_reducible(self):
        assert spa._reducible_sites({1: F(1), 2: F(3, 2)}) == frozenset()
        assert spa._reducible_sites({1: F(1, 2)}) == frozenset({1})
        assert spa._reducible_sites(None) == frozenset()


class TestTheTwoPathsAssumeDifferentThings:
    """Why the reduction is gated on one path and not the other.

    Measured, and it corrected the first design: the recursion method's inner
    product is `inner` → `expectation`, whose docstring says spin-½ and which
    returns 1/4 for (S^x)²; `expectation_general` refuses a symbolic S outright.
    So `moments`/`continued_fraction` were ALREADY committed to spin-½, and
    gating their reduction behind a declaration would have asked the researcher
    to declare what the calculation assumes — on exactly the path that hits the
    wall.  What gets DISPLAYED is different: `[H,[H,S^x_1]] = J² S^x_1 (S^z_2)²`
    is a general-spin answer the 2-site scenario shows on purpose.
    """

    def test_evaluation_path_reduces_without_any_declaration(self):
        H, O = _chain(4), spa.spin_operator("x", 1)
        b2s = spa.continued_fraction(H, O, 8)          # no spins argument
        assert len(b2s) == 8, b2s

    def test_display_path_does_not_reduce_without_a_declaration(self):
        """The counter-case: same site, same algebra, left alone."""
        out = spa.nested_commutator(spa.parse_spin_expression("S^z_1 S^z_2"),
                                    spa.spin_operator("x", 1), 2)
        assert any(len(ops) > 2 for _, ops in out), out   # (S^z_2)^2 survives

    def test_the_expectation_underneath_really_is_spin_half(self):
        """The premise of the split, pinned so it cannot drift silently."""
        e = spa.parse_spin_expression("S^x_1 S^x_1")
        assert spa.expectation(e) == spa.Coeff({(): (F(1, 4), F(0))})
        with pytest.raises(ValueError):
            spa.expectation_general(e, {1: "S"})       # no symbolic spin here


class TestTheFoldRespectsOrder:
    """Same-site factors do NOT commute — the fold must not reorder them.

    open_items A-5d named this as the hard part of a correct implementation
    ("正しい演算子順序で実装し…"): the sort that groups a term by site has to be
    stable, or `S^x S^y` and `S^y S^x` collapse to the same thing and the sign is
    silently lost.  Different sites DO commute and may be reordered freely.
    """

    def test_swapping_two_same_site_factors_flips_the_sign(self):
        half = frozenset({1})
        a = spa._collect(spa.parse_spin_expression("S^x_1 S^y_1"), half)
        b = spa._collect(spa.parse_spin_expression("S^y_1 S^x_1"), half)
        assert spa.to_latex(a) == r"\frac{1}{2} i S^{z}_{1}"
        assert spa.to_latex(b) == r"-\frac{1}{2} i S^{z}_{1}"

    def test_different_sites_are_left_as_a_product(self):
        out = spa._collect(spa.parse_spin_expression("S^x_1 S^y_2"),
                           frozenset({1, 2}))
        assert spa.to_latex(out) == r"S^{x}_{1} S^{y}_{2}"
