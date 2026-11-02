"""F-9: angular-momentum coupling — exact arithmetic, Wigner symbols, wiring.

Verification strategy: the values are checked against **mathematical identities**
(orthogonality, symmetry, the 9j → 6j reduction, normalization of a coupled
state) rather than against a reference library, so the suite runs anywhere and
does not pin the project to a symbolic-algebra dependency it does not otherwise
need.  A layer of published table values is checked on top, because identities
alone cannot catch a globally wrong phase convention.

(During development the whole grid was additionally cross-checked against
``sympy.physics.wigner`` via the Julia-side Conda interpreter — 1642 3j, 2408 CG,
15625 6j and 245 non-zero 9j values, all matching.  That check is not part of
this suite: sympy is not an application dependency.)
"""
import asyncio
import itertools
from fractions import Fraction
from unittest.mock import MagicMock

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime import angular_momentum as am
from notebook.runtime import angular_momentum_latex as amx
from notebook.runtime.angular_momentum import (
    AngularMomentumError, Surd, cg_expansion, clebsch_gordan, coupled_j_values,
    dot_product_eigenvalue, lande_g_factor, parse_j, parse_term_symbol,
    wigner_3j, wigner_6j, wigner_9j, wigner_eckart_coefficient,
)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

F = Fraction
HALVES = [F(n, 2) for n in range(0, 7)]          # 0, 1/2, … 3


def _ms(j):
    m, out = -j, []
    while m <= j:
        out.append(m)
        m += 1
    return out


# ---------------------------------------------------------------------------
# Exact arithmetic
# ---------------------------------------------------------------------------

class TestSurd:
    def test_product_of_radicands_reduces_by_gcd(self):
        # √6·√10 = 2√15 — reduced without factoring anything.
        assert Surd.term(1, 6) * Surd.term(1, 10) == Surd.term(2, 15)

    def test_sum_keeps_distinct_radicands_apart(self):
        s = Surd.term(1, 2) + Surd.term(1, 3)
        assert s.terms == {2: F(1), 3: F(1)}
        assert float(s) == pytest.approx(2 ** 0.5 + 3 ** 0.5)

    def test_square_of_a_surd_is_rational(self):
        assert (Surd.term(F(1, 2), 2)).square() == Surd.rational(F(1, 2))

    def test_latex_prefers_the_unrationalized_shape(self):
        assert Surd.term(F(1, 2), 2).latex() == r"\frac{1}{\sqrt{2}}"
        assert Surd.term(F(1, 3), 6).latex() == r"\frac{\sqrt{6}}{3}"
        assert Surd.rational(F(-4, 3)).latex() == r"-\frac{4}{3}"

    def test_canonical_is_cas_syntax(self):
        assert Surd.term(F(1, 2), 2).canonical() == "sqrt(2)/2"
        assert Surd.term(F(-3, 7), 5).canonical() == "-3*sqrt(5)/7"
        assert Surd().canonical() == "0"

    def test_float_is_refused(self):
        # A float here means a phase or coefficient was computed inexactly; it
        # must fail loudly rather than become Fraction(0.333…).
        with pytest.raises(TypeError):
            Surd.term(0.5, 2)

    def test_phase_helper_handles_negative_exponents(self):
        # (-1)**-2 is 1.0 (a float) in plain Python — the bug this guards.
        assert am._phase(-2) == 1 and isinstance(am._phase(-2), int)
        assert am._phase(-3) == -1
        assert am._phase(0) == 1


class TestParseJ:
    @pytest.mark.parametrize("text,expected", [
        ("3/2", F(3, 2)), ("-1/2", F(-1, 2)), ("2", F(2)), (" 5 / 2 ", F(5, 2)),
        ("１/２", F(1, 2)),                      # full-width digits
        ("−1/2", F(-1, 2)),                      # unicode minus
    ])
    def test_accepts(self, text, expected):
        assert parse_j(text) == expected

    @pytest.mark.parametrize("text", ["1/3", "0.7", "j", "", "3/4"])
    def test_rejects_rather_than_rounds(self, text):
        with pytest.raises(AngularMomentumError):
            parse_j(text)


# ---------------------------------------------------------------------------
# Published table values (catch a wrong global convention)
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_two_spin_half_singlet_and_triplet(self):
        assert clebsch_gordan("1/2", "1/2", "1/2", "-1/2", 1, 0) == Surd.term(F(1, 2), 2)
        assert clebsch_gordan("1/2", "1/2", "1/2", "-1/2", 0, 0) == Surd.term(F(1, 2), 2)
        # Condon-Shortley: the singlet is antisymmetric.
        assert clebsch_gordan("1/2", "-1/2", "1/2", "1/2", 0, 0) == Surd.term(F(-1, 2), 2)
        assert clebsch_gordan("1/2", "1/2", "1/2", "1/2", 1, 1) == Surd.rational(1)

    @pytest.mark.parametrize("args,expected", [
        ((1, 0, 1, 0, 2, 0), Surd.term(F(1, 3), 6)),        # √(2/3)
        ((1, 1, 1, -1, 0, 0), Surd.term(F(1, 3), 3)),       # 1/√3
        ((1, 0, 1, 0, 1, 0), Surd()),                        # vanishes
    ])
    def test_spin_one_table(self, args, expected):
        assert clebsch_gordan(*args) == expected

    def test_3j_special_cases(self):
        # (j j 0 / m -m 0) = (-1)^(j-m)/√(2j+1)
        for j in (F(1, 2), F(1), F(3, 2)):
            for m in _ms(j):
                expected = Surd.term(F(am._phase(int(j - m)), int(2 * j + 1)),
                                     1) * Surd.term(1, 1)
                got = wigner_3j(j, j, 0, m, -m, 0)
                assert float(got) == pytest.approx(
                    am._phase(int(j - m)) / (2 * float(j) + 1) ** 0.5)
                assert not got.is_zero()
                del expected
        assert wigner_3j(1, 1, 0, 0, 0, 0) == Surd.term(F(-1, 3), 3)
        assert float(wigner_3j(2, 2, 2, 0, 0, 0)) == pytest.approx(-(2 / 35) ** 0.5)

    def test_6j_and_9j_values(self):
        assert wigner_6j(1, 1, 1, 1, 1, 1) == Surd.rational(F(1, 6))
        assert wigner_6j(1, 2, 3, 2, 1, 2) == Surd.term(F(1, 105), 21)
        # An all-equal 9j with odd j-sum must vanish by the odd-permutation phase.
        assert wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 1).is_zero()
        assert wigner_9j("1/2", "1/2", 1, "1/2", "1/2", 1, 1, 1, 0) == \
            Surd.rational(F(-1, 18))


# ---------------------------------------------------------------------------
# Identities — the real verification
# ---------------------------------------------------------------------------

class TestIdentities:
    def test_3j_orthogonality_over_m(self):
        # Σ_{m1 m2} (2j3+1) (j1 j2 j3 / m1 m2 m3)² = 1 at FIXED m3 (the relation
        # carries a δ_{m3 m3'}, so m3 must be held fixed while m1 runs).
        for j1, j2 in itertools.product([F(1, 2), F(1), F(3, 2)], repeat=2):
            for j3 in coupled_j_values(j1, j2):
                for m3 in _ms(j3):
                    total = Surd()
                    for m1 in _ms(j1):
                        m2 = -m3 - m1
                        if abs(m2) > j2:
                            continue
                        total = total + wigner_3j(j1, j2, j3, m1, m2, m3).square()
                    assert total * Fraction(int(2 * j3 + 1)) == Surd.rational(1)

    def test_3j_symmetry_under_permutation_and_sign_flip(self):
        for j1, j2 in itertools.product([F(1, 2), F(1), F(3, 2), F(2)], repeat=2):
            for j3 in coupled_j_values(j1, j2):
                for m1 in _ms(j1):
                    for m2 in _ms(j2):
                        m3 = -(m1 + m2)
                        if abs(m3) > j3:
                            continue
                        base = wigner_3j(j1, j2, j3, m1, m2, m3)
                        # cyclic permutation: invariant
                        assert wigner_3j(j2, j3, j1, m2, m3, m1) == base
                        # odd permutation: phase (-1)^(j1+j2+j3)
                        phase = am._phase(int(j1 + j2 + j3))
                        assert wigner_3j(j2, j1, j3, m2, m1, m3) == base * phase
                        # all m → -m: same phase
                        assert wigner_3j(j1, j2, j3, -m1, -m2, -m3) == base * phase

    def test_clebsch_gordan_columns_are_orthonormal(self):
        for j1, j2 in itertools.product([F(1, 2), F(1), F(3, 2)], repeat=2):
            for big_j in coupled_j_values(j1, j2):
                for big_m in _ms(big_j):
                    total = Surd()
                    for m1 in _ms(j1):
                        m2 = big_m - m1
                        if abs(m2) > j2:
                            continue
                        total = total + clebsch_gordan(j1, m1, j2, m2,
                                                       big_j, big_m).square()
                    assert total == Surd.rational(1)

    def test_6j_orthogonality(self):
        # Σ_x (2x+1) {j1 j2 x / j3 j4 j5}{j1 j2 x / j3 j4 j5'} = δ/(2j5+1)
        j1, j2, j3, j4 = F(1), F(3, 2), F(1), F(1, 2)
        for j5 in coupled_j_values(j1, j4):
            for j5p in coupled_j_values(j1, j4):
                total = Surd()
                for x in coupled_j_values(j1, j2):
                    a = wigner_6j(j1, j2, x, j3, j4, j5)
                    b = wigner_6j(j1, j2, x, j3, j4, j5p)
                    if a.is_zero() or b.is_zero():
                        continue
                    total = total + a * b * Fraction(int(2 * x + 1))
                expected = (Surd.rational(Fraction(1, int(2 * j5 + 1)))
                            if j5 == j5p else Surd())
                assert total == expected

    def test_9j_with_a_zero_argument_reduces_to_a_6j(self):
        # {a b c / d e f / g h 0} = δ_gh δ_cf (-1)^(b+c+d+g)/√((2c+1)(2g+1)) {a b c / e d g}
        for a, b in itertools.product([F(1, 2), F(1), F(3, 2)], repeat=2):
            for c in coupled_j_values(a, b):
                for d, g in itertools.product([F(1, 2), F(1)], repeat=2):
                    for e in [F(1, 2), F(1), F(3, 2)]:
                        got = wigner_9j(a, b, c, d, e, c, g, g, 0)
                        six = wigner_6j(a, b, c, e, d, g)
                        norm = int((2 * c + 1) * (2 * g + 1))
                        out, rad = am._sqrt_int(norm)
                        pref = Surd.term(
                            Fraction(am._phase(int(b + c + d + g)) * out, norm), rad)
                        assert got == six * pref

    def test_9j_transpose_symmetry(self):
        rows = (F(1), F(1, 2), F(3, 2), F(1), F(1), F(1), F(1, 2), F(1, 2), F(1))
        direct = wigner_9j(*rows)
        a, b, c, d, e, f, g, h, i = rows
        transposed = wigner_9j(a, d, g, b, e, h, c, f, i)
        assert transposed == direct * am._phase(int(sum(rows)))

    def test_symbols_vanish_when_a_triangle_fails(self):
        assert wigner_3j(1, 1, 5, 0, 0, 0).is_zero()
        assert wigner_6j(1, 1, 9, 1, 1, 1).is_zero()
        assert wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 9).is_zero()
        # and when the projections do not sum to zero
        assert wigner_3j(1, 1, 2, 1, 1, 1).is_zero()


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

class TestCoupling:
    def test_allowed_values_and_dimension_count(self):
        for j1, j2 in itertools.product(HALVES, repeat=2):
            values = coupled_j_values(j1, j2)
            assert values[0] == abs(j1 - j2) and values[-1] == j1 + j2
            # Σ (2J+1) = (2j1+1)(2j2+1): the coupled basis has the same size.
            assert sum(2 * v + 1 for v in values) == (2 * j1 + 1) * (2 * j2 + 1)

    def test_dot_product_eigenvalue_reproduces_the_singlet_triplet_split(self):
        # two spin-½: singlet -3/4, triplet +1/4 (in units of ħ²)
        assert dot_product_eigenvalue("1/2", "1/2", 0) == F(-3, 4)
        assert dot_product_eigenvalue("1/2", "1/2", 1) == F(1, 4)

    def test_dot_product_eigenvalue_matches_the_landé_interval_rule(self):
        # E(F) - E(F-1) = A·F for H = A I·J — the interval rule, from the identity.
        i, j = F(3, 2), F(1, 2)
        levels = coupled_j_values(i, j)
        for lo, hi in zip(levels, levels[1:]):
            gap = dot_product_eigenvalue(i, j, hi) - dot_product_eigenvalue(i, j, lo)
            assert gap == hi

    def test_total_outside_the_coupling_is_rejected(self):
        with pytest.raises(AngularMomentumError):
            dot_product_eigenvalue("1/2", "1/2", 2)

    def test_expansion_is_normalized_and_reproduces_the_singlet(self):
        terms = cg_expansion("1/2", "1/2", 0, 0)
        assert len(terms) == 2
        norm = Surd()
        for _, _, coeff in terms:
            norm = norm + coeff.square()
        assert norm == Surd.rational(1)
        # antisymmetric combination
        assert terms[0][2] == -terms[1][2]

    def test_expansion_of_a_stretched_state_is_a_single_term(self):
        terms = cg_expansion(1, "1/2", "3/2", "3/2")
        assert len(terms) == 1
        assert terms[0][2] == Surd.rational(1)


# ---------------------------------------------------------------------------
# Spectroscopy bridges
# ---------------------------------------------------------------------------

class TestTermSymbolsAndLandeG:
    @pytest.mark.parametrize("text,expected", [
        ("^2P_{3/2}", (F(1, 2), F(1), F(3, 2))),
        ("^{2}P_{3/2}", (F(1, 2), F(1), F(3, 2))),
        ("2P3/2", (F(1, 2), F(1), F(3, 2))),
        ("^3P_2", (F(1), F(1), F(2))),
        ("^1S_0", (F(0), F(0), F(0))),
    ])
    def test_parse_term_symbol(self, text, expected):
        assert parse_term_symbol(text) == expected

    def test_term_symbol_with_impossible_j_is_rejected(self):
        with pytest.raises(AngularMomentumError):
            parse_term_symbol("^2P_{7/2}")       # L=1, S=1/2 → J ∈ {1/2, 3/2}

    @pytest.mark.parametrize("term,expected", [
        ("^2S_{1/2}", F(2)),
        ("^2P_{1/2}", F(2, 3)),
        ("^2P_{3/2}", F(4, 3)),
        ("^2D_{5/2}", F(6, 5)),
        ("^3P_2", F(3, 2)),
    ])
    def test_lande_g_textbook_values(self, term, expected):
        spin, orbital, total = parse_term_symbol(term)
        assert lande_g_factor(orbital, spin, total) == expected

    def test_lande_g_reduces_to_g_l_and_g_s_in_the_pure_cases(self):
        assert lande_g_factor(2, 0, 2) == 1          # pure orbital (S=0) → g_L
        assert lande_g_factor(0, F(1, 2), F(1, 2)) == 2   # pure spin (L=0) → g_S

    def test_lande_g_undefined_for_j_zero(self):
        with pytest.raises(AngularMomentumError):
            lande_g_factor(1, 1, 0)


class TestWignerEckart:
    def test_dipole_selection_rules_fall_out_of_the_coefficient(self):
        # k=1: ΔJ ∈ {0, ±1} and Δm = q, else the geometric factor vanishes.
        for jp in [F(1, 2), F(3, 2), F(5, 2)]:
            allowed = abs(jp - F(1, 2)) <= 1 <= jp + F(1, 2)
            coeff = wigner_eckart_coefficient(F(1, 2), F(1, 2), 1, 0, jp, F(1, 2))
            assert coeff.is_zero() != allowed
        # Δm ≠ q vanishes
        assert wigner_eckart_coefficient(
            F(1, 2), F(1, 2), 1, 0, F(3, 2), F(3, 2)).is_zero()

    def test_zero_to_zero_transition_is_forbidden(self):
        assert wigner_eckart_coefficient(0, 0, 1, 0, 0, 0).is_zero()

    @pytest.mark.parametrize("j,k,jp", [
        (F(1, 2), F(1), F(3, 2)), (F(1), F(1), F(1)), (F(3, 2), F(2), F(1, 2)),
    ])
    def test_geometric_factors_square_to_one_over_the_multiplet(self, j, k, jp):
        # Σ_{m,q} |CG(j m k q|j' m+q)|² = 2j'+1 (each |j' M⟩ is normalized, and
        # there are 2j'+1 of them), so the 1/√(2j'+1) in the theorem makes the
        # geometric factors square to exactly 1.
        total = Surd()
        for m in _ms(j):
            for q in _ms(k):
                if abs(m + q) > jp:
                    continue
                total = total + wigner_eckart_coefficient(j, m, k, q, jp,
                                                          m + q).square()
        assert total == Surd.rational(1)


# ---------------------------------------------------------------------------
# Display forms
# ---------------------------------------------------------------------------

class TestLatex:
    def test_term_symbol_subscript_is_not_a_stacked_fraction(self):
        assert amx.term_symbol_latex(F(1, 2), F(1), F(3, 2)) == r"^{2}\!P_{3/2}"

    def test_wigner_symbol_brackets_by_kind(self):
        assert amx.wigner_symbol_latex("3j", [[F(1), F(2), F(3)],
                                              [F(0), F(0), F(0)]]).startswith(
            r"\begin{pmatrix}")
        assert amx.wigner_symbol_latex("6j", [[F(1)] * 3, [F(1)] * 3]).startswith(
            r"\begin{Bmatrix}")

    def test_dot_product_canonical_is_cas_parseable(self):
        assert amx.dot_product_canonical(F(3, 2), F(1, 2), "F") == \
            "(F*(F+1) - 15/4 - 3/4)/2"

    def test_expansion_latex_signs(self):
        terms = cg_expansion("1/2", "1/2", 0, 0)
        latex = amx.expansion_latex(F(1, 2), F(1, 2), F(0), F(0), terms)
        assert latex.count(" - ") == 1 and " + " not in latex

    def test_named_expansion_display_mirrors_its_canonical(self):
        # The display must not promise more than the canonical delivers: both
        # carry the same named basis functions.
        terms = cg_expansion("1/2", "1/2", 0, 0)
        latex = amx.expansion_latex(F(1, 2), F(1, 2), F(0), F(0), terms,
                                    basis=("chi", "phi"))
        canonical = amx.expansion_canonical(terms, ("chi", "phi"))
        assert r"\chi\!\left(\frac{1}{2}\right)" in latex
        assert r"\phi\!\left(-\frac{1}{2}\right)" in latex
        assert canonical == \
            "sqrt(2)/2*chi(1/2)*phi(-1/2) - sqrt(2)/2*chi(-1/2)*phi(1/2)"
        assert latex.count(" - ") == canonical.count(" - ") == 1


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

class TestEngine:
    def test_registers_exact_value_with_provenance(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            ex = e.clebsch_gordan_coefficient("1/2", "1/2", "1/2", "-1/2", 1, 0)
            assert ex.canonical == "sqrt(2)/2"
            assert ex.latex.endswith(r"= \frac{1}{\sqrt{2}}")
            assert ex.angular_descriptor["kind"] == "clebsch_gordan"
            assert ex.angular_descriptor["quantum_numbers"][0] == "1/2"

    def test_coupling_registers_both_the_levels_and_the_identity(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            allowed, identity = e.couple_angular_momenta("3/2", "1/2", "F", ("I", "S"))
            assert allowed.canonical == "[1, 2]"
            assert allowed.angular_descriptor["allowed"] == ["1", "2"]
            assert identity.canonical == "(F*(F+1) - 15/4 - 3/4)/2"
            assert r"\mathbf{I}\cdot\mathbf{S}" in identity.latex

    def test_the_coupling_identity_composes_with_ordinary_cas_operations(self):
        # The point of keeping J symbolic: the result is not a dead end.
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            _, identity = e.couple_angular_momenta("3/2", "1/2", "F")
            step, value = e.run_operation("substitute", {
                "expr": identity.display_id, "lhs": "F", "rhs": "2"})
            assert step.status.value == "success"
            # ⟨I·S⟩ at F=2 for I=3/2, S=1/2 — the upper hyperfine level.
            assert value.strip() == "3/4"

    def test_unnamed_expansion_gets_system_chosen_symbols_and_is_computable(self):
        # The researcher names the bases if they want to; if they don't, the
        # system picks free ones — so an expansion is never a dead end.
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            state = e.coupled_state_expansion("1/2", "1/2", 1, 0)
            assert state.angular_descriptor["basis"] == ["chi", "psi"]
            assert state.angular_descriptor["basis_auto"] is True
            assert state.canonical == \
                "sqrt(2)/2*chi(1/2)*psi(-1/2) + sqrt(2)/2*chi(-1/2)*psi(1/2)"
            assert r"\chi\!\left" in state.latex
            step, _ = e.run_operation("simplify", {"expr": state.display_id})
            assert step.status.value == "success"

    def test_generated_names_step_aside_when_taken(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            e.register_expression("x^2", lhs_label="chi")
            state = e.coupled_state_expansion("1/2", "1/2", 1, 0)
            assert "chi(" not in state.canonical
            assert state.angular_descriptor["basis"] == ["psi", "xi"]

    def test_declared_non_computable_operand_is_refused_at_point_of_use(self):
        # The cas_usable contract: no F-9 operation produces False today, but
        # the guard is what keeps that an invariant.
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            odd = e.register_direct(r"\left| 1,\,0 \right\rangle",
                                    angular_descriptor={"kind": "coupled_state",
                                                        "cas_usable": False})
            with pytest.raises(ValueError) as exc:
                e.run_operation("simplify", {"expr": odd.display_id})
            assert odd.display_id in str(exc.value)

    def test_named_expansion_is_a_real_expression(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            state = e.coupled_state_expansion("1/2", "1/2", 1, 0,
                                              basis=("chi", "phi"))
            assert state.canonical == \
                "sqrt(2)/2*chi(1/2)*phi(-1/2) + sqrt(2)/2*chi(-1/2)*phi(1/2)"
            assert state.angular_descriptor["cas_usable"] is True
            assert r"\chi\!\left(\frac{1}{2}\right)" in state.latex
            step, _ = e.run_operation("simplify", {"expr": state.display_id})
            assert step.status.value == "success"

    @pytest.mark.parametrize("definition", ["x^2", "(1+sqrt(5))/2"])
    def test_basis_name_already_defined_is_refused(self, definition):
        # H-1 binds "phi : x^2" in the CAS, so reusing the name as a basis made
        # simplify return SUCCESS with "(x^2)(1/2)" in it — plausible nonsense.
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            e.register_expression(definition, lhs_label="phi")
            with pytest.raises(AngularMomentumError) as exc:
                e.coupled_state_expansion("1/2", "1/2", 1, 0, basis=("chi", "phi"))
            assert exc.value.code == "basis_name_taken"
            assert exc.value.params["name"] == "phi"

    def test_basis_name_used_by_an_earlier_expansion_is_refused(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            e.coupled_state_expansion("1/2", "1/2", 1, 0, basis=("chi", "phi"))
            with pytest.raises(AngularMomentumError) as exc:
                e.coupled_state_expansion(1, "1/2", "3/2", "1/2",
                                          basis=("chi", "psi"))
            assert exc.value.code == "basis_name_taken"

    def test_the_two_basis_names_must_differ(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            with pytest.raises(AngularMomentumError) as exc:
                e.coupled_state_expansion("1/2", "1/2", 1, 0, basis=("u", "u"))
            assert exc.value.code == "basis_names_identical"

    def test_free_basis_names_still_work(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            e.register_expression("x^2", lhs_label="phi")
            state = e.coupled_state_expansion("1/2", "1/2", 1, 0,
                                              basis=("u", "v"))
            assert state.canonical.startswith("sqrt(2)/2*u(1/2)*v(-1/2)")

    def test_display_only_guard_leaves_ordinary_expressions_alone(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            plain = e.register_expression("(x+1)^2")
            step, _ = e.run_operation("expand", {"expr": plain.display_id})
            assert step.status.value == "success"

    def test_wigner_symbol_shape_is_validated(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            with pytest.raises(AngularMomentumError):
                e.wigner_symbol("6j", [[1, 2, 3]])
            with pytest.raises(AngularMomentumError):
                e.wigner_symbol("12j", [[1, 2, 3], [1, 2, 3]])

    def test_lande_g_from_term_symbol_or_quantum_numbers(self):
        with NotebookEngine(title="f9", mode=SessionMode.standard_symbolic) as e:
            assert e.lande_g_factor(term_symbol="^2P_{3/2}").canonical == "4/3"
            assert e.lande_g_factor(1, "1/2", "1/2").canonical == "2/3"


# ---------------------------------------------------------------------------
# Natural-language triggers — every trigger in BOTH languages
# ---------------------------------------------------------------------------

def _sup():
    s = MagicMock(spec=LLMSupervisor)
    s._v = OperationValidator()
    s.needs_expression_injection.return_value = False
    return s


def _run(engine, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, _sup()))


class TestNaturalLanguage:
    JA_EN = [
        ("clebsch_gordan",
         "j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0 のCG係数を計算して",
         "compute the Clebsch-Gordan coefficient for "
         "j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0"),
        ("wigner_3j",
         "3j記号 (1 2 3; 0 0 0) を計算して",
         "compute the 3j symbol (1 2 3; 0 0 0)"),
        ("wigner_6j",
         "6j記号 {1 2 3; 2 1 2} を求めて",
         "evaluate the 6j symbol {1 2 3; 2 1 2}"),
        ("wigner_9j",
         "9j記号 {1/2 1/2 1; 1/2 1/2 1; 1 1 0} を計算して",
         "compute the 9j symbol {1/2 1/2 1; 1/2 1/2 1; 1 1 0}"),
        ("couple_angular_momenta",
         "I=3/2 と S=1/2 を合成して",
         "couple angular momenta I=3/2 and S=1/2"),
        ("coupled_state_expansion",
         "j1=1/2, j2=1/2 の J=1, M=0 の状態を非結合基底で展開して",
         "expand the state with j1=1/2, j2=1/2, J=1, M=0 in the uncoupled basis"),
        ("lande_g_factor",
         "^2P_{3/2} のランデのg因子を求めて",
         "Lande g-factor of ^2P_{3/2}"),
        ("wigner_eckart",
         "j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2 についてWigner-Eckartの定理を書き下して",
         "Wigner-Eckart theorem for j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2"),
    ]

    @pytest.mark.parametrize("operation,ja,en", JA_EN)
    def test_trigger_fires_in_both_languages(self, operation, ja, en):
        for lang, text in (("ja", ja), ("en", en)):
            nb.start_engine(title="f9", mode="standard", language=lang)
            try:
                r = _run(nb.get_engine(), text)
                assert r["status"] == "completed", (lang, text, r.get("error"))
                assert r["operation"] == operation, (lang, text)
                assert r["registered_expressions"]
            finally:
                nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        ("ja", "j1=1/2, j2=1/2, J=1, M=0 を chi と phi の非結合基底で展開して"),
        ("en", "expand j1=1/2, j2=1/2, J=1, M=0 in the uncoupled basis chi and phi"),
    ])
    def test_named_basis_expansion_via_nl(self, lang, text):
        nb.start_engine(title="f9", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r["status"] == "completed"
            got = r["registered_expressions"][-1]
            assert got["canonical"] == \
                "sqrt(2)/2*chi(1/2)*phi(-1/2) + sqrt(2)/2*chi(-1/2)*phi(1/2)"
            assert "chi" in r["message"] and "phi" in r["message"]
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_unnamed_expansion_reports_the_symbols_it_chose(self, lang):
        text = ("j1=1/2, j2=1/2 の J=1, M=0 の状態を非結合基底で展開して" if lang == "ja"
                else "expand the state with j1=1/2, j2=1/2, J=1, M=0 "
                     "in the uncoupled basis")
        nb.start_engine(title="f9", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r["status"] == "completed"
            assert "chi" in r["message"] and "psi" in r["message"]
            assert r["registered_expressions"][-1]["canonical"] == \
                "sqrt(2)/2*chi(1/2)*psi(-1/2) + sqrt(2)/2*chi(-1/2)*psi(1/2)"
        finally:
            nb.stop_engine()

    def test_values_reach_the_notebook_intact(self):
        nb.start_engine(title="f9", mode="standard", language="ja")
        try:
            r = _run(nb.get_engine(), "I=3/2 と S=1/2 を合成して")
            got = {x["canonical"] for x in r["registered_expressions"]}
            assert "[1, 2]" in got
            assert "(F*(F+1) - 15/4 - 3/4)/2" in got
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        ("ja", "j1=1/2, m1=1/2, j2=1/2 のCG係数を計算して"),
        ("en", "compute the Clebsch-Gordan coefficient for j1=1/2, m1=1/2, j2=1/2"),
    ])
    def test_missing_quantum_numbers_fail_with_a_translated_message(self, lang, text):
        nb.start_engine(title="f9", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r["status"] == "failed"
            assert "j1, m1, j2, m2, J, M" in r["error"]
            # the message is i18n'd, not an English literal in JA
            assert ("量子数" in r["error"]) == (lang == "ja")
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        ("ja", "m が j を超えています: j1=1/2, m1=3/2, j2=1/2, m2=-1/2, J=1, M=1 のCG係数"),
        ("en", "Clebsch-Gordan coefficient for j1=1/2, m1=3/2, j2=1/2, m2=-1/2, J=1, M=1"),
    ])
    def test_impossible_quantum_numbers_are_reported_not_silently_zero(self, lang, text):
        nb.start_engine(title="f9", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r["status"] == "failed"
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        # "展開" / "expand" alone is the ordinary CAS operation and must stay so —
        # only the BASIS words hand it to F-9.
        ("ja", "E1 を展開して"),
        ("en", "expand E1"),
        ("ja", "E1 と E2 を足して"),
        ("en", "add E1 and E2"),
        ("ja", "j=3 の球面調和関数を求めて"),
        ("en", "compute the spherical harmonic for j=3"),
    ])
    def test_unrelated_instructions_fall_through(self, lang, text):
        from ui import interpret_angular as ia
        for handler in (ia._try_coupled_state, ia._try_clebsch_gordan,
                        ia._try_wigner_symbol, ia._try_couple,
                        ia._try_wigner_eckart, ia._try_lande_g):
            got = asyncio.run(handler(None, None, lang, text, []))
            assert got is None, (handler.__name__, lang, text)
