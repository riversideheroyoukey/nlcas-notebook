"""F-1 stage 3c: composable symbolic spin algebra and nested commutators.

The point is composition: a commutator result is structured data, so a *second*
commutator [H,[H,O]] can be taken on it — the equation-of-motion / moment
hierarchy that the engine's opaque-canonical result blocked (problem B).
"""
import asyncio
from unittest.mock import MagicMock

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.spin_algebra import (
    Coeff, build_hamiltonian, spin_operator, commutator, nested_commutator, to_latex,
    inner, moments, continued_fraction, general_low_moments, general_b1_squared,
    energy_moment, energy_moment_general_latex, energy_cumulants_general,
    specific_heat_general_latex, specific_heat_concrete_latex,
    greens_function, spectral_latex, spectral_peaks, terminator_spectral_latex,
    expectation, expectation_general, _site_trace_poly,
    general_energy_moment, general_spin_energy_moment_latex,
    parse_spin_expression, coeff_vector_norm)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline, _parse_spin_commutator,
    _parse_spin_spectral)


class TestParseSpinExpression:
    """Parser (string → term list) — the inverse of to_latex; lets a custom spin
    Hamiltonian be entered as a general expression and chained through the general
    spin primitives instead of a canned per-scenario helper."""

    def test_roundtrip_lab_hamiltonian(self):
        lab = parse_spin_expression(r"-\omega_0 S^z_1 - \omega_1 S^x_1")
        assert parse_spin_expression(to_latex(lab)) == lab

    def test_general_chain_reproduces_precession_eom(self):
        # H_lab + frame term ω S^z (plain addition) → general commutator == iω₁S^y,
        # i.e. the rotating-frame derivation runs on GENERAL primitives (no canned).
        hr = parse_spin_expression(r"-\omega_0 S^z_1 - \omega_1 S^x_1 + \omega S^z_1")
        assert to_latex(hr) == r"\left( \omega - \omega_0 \right) S^{z}_{1} - \omega_1 S^{x}_{1}"
        eom = commutator(hr, spin_operator("z", 1))
        assert to_latex(eom) == r"i \omega_1 S^{y}_{1}"

    def test_strips_lhs_and_handles_powers(self):
        h = parse_spin_expression(r"H = 2 S^z_1 S^z_2")
        assert to_latex(h) == r"2 S^{z}_{1} S^{z}_{2}"

    def test_rejects_non_spin(self):
        import pytest
        with pytest.raises(ValueError):
            parse_spin_expression(r"a x + b")

    def test_coeff_vector_norm(self):
        # √(Ω²+ω₁²) for the rotating-frame H_rot = −Ω S^z − ω₁ S^x
        h = parse_spin_expression(r"-\Omega S^z_1 - \omega_1 S^x_1")
        assert coeff_vector_norm(h) == r"\sqrt{ \Omega^2 + \omega_1^2 }"

    def test_coeff_vector_norm_polynomial(self):
        # before substituting Ω: coefficient (ω−ω₀) is parenthesized under the square
        hr = parse_spin_expression(r"-\omega_0 S^z_1 - \omega_1 S^x_1 + \omega S^z_1")
        assert coeff_vector_norm(hr) == r"\sqrt{ \left(\omega - \omega_0\right)^2 + \omega_1^2 }"


class TestCoeff:
    def test_i_squared_is_minus_one(self):
        assert (Coeff.imag() * Coeff.imag()) == Coeff.int(-1)

    def test_param_power_and_latex(self):
        assert (Coeff.param("J") * Coeff.param("J")).to_latex() == "J^{2}"
        assert (Coeff.imag() * Coeff.param("J")).to_latex() == "i J"

    def test_cancellation_to_zero(self):
        c = Coeff.param("J") + (Coeff.int(-1) * Coeff.param("J"))
        assert c.is_zero()


class TestCommutators:
    def test_ising_first_commutator(self):
        h = build_hamiltonian([(1, 2)], kind="ising")
        out = commutator(h, spin_operator("x", 1))
        assert to_latex(out) == r"i J S^{y}_{1} S^{z}_{2}"

    def test_ising_double_commutator_composes(self):
        # the key problem-B case: [H,[H,Sx_1]] = J² Sx_1 (Sz_2)²
        h = build_hamiltonian([(1, 2)], kind="ising")
        out = nested_commutator(h, spin_operator("x", 1), 2)
        assert to_latex(out) == r"J^{2} S^{x}_{1} \left(S^{z}_{2}\right)^{2}"

    def test_heisenberg_first_commutator(self):
        h = build_hamiltonian([(1, 2)], kind="heisenberg")
        out = commutator(h, spin_operator("x", 1))
        assert to_latex(out) == (r"-i J S^{z}_{1} S^{y}_{2} + i J S^{y}_{1} S^{z}_{2}")

    def test_chain_double_commutator_interference(self):
        # site 2 has two neighbours (1,3): the cross term 2J² Sz_1 Sx_2 Sz_3 is the
        # two-bond interference that the single-stage relabel could never produce
        h = build_hamiltonian([(1, 2), (2, 3)], kind="ising")
        out = nested_commutator(h, spin_operator("x", 2), 2)
        assert r"2 J^{2} S^{z}_{1} S^{x}_{2} S^{z}_{3}" in to_latex(out)

    def test_self_commutator_is_zero(self):
        o = spin_operator("x", 1)
        assert commutator(o, o) == []


class TestModelsAndField:
    def test_xxz_anisotropy(self):
        h = build_hamiltonian([(1, 2)], kind="xxz")
        out = to_latex(commutator(h, spin_operator("x", 1)))
        assert r"i J_z S^{y}_{1} S^{z}_{2}" in out      # z-bond carries J_z
        assert r"-i J S^{z}_{1} S^{y}_{2}" in out       # xy-bond carries J

    def test_transverse_field_precession(self):
        h = build_hamiltonian([(1, 2)], kind="ising", n_sites=2, field=("h", "x"))
        out = to_latex(commutator(h, spin_operator("z", 1)))
        assert out == r"i h S^{y}_{1}"                  # field term alone (Ising part 0)

    def test_xy_model(self):
        h = build_hamiltonian([(1, 2)], kind="xy")
        out = to_latex(commutator(h, spin_operator("z", 1)))
        assert r"i J S^{x}_{1} S^{y}_{2}" in out and r"-i J S^{y}_{1} S^{x}_{2}" in out

    def test_third_order_ising(self):
        h = build_hamiltonian([(1, 2)], kind="ising")
        out = nested_commutator(h, spin_operator("x", 1), 3)
        assert to_latex(out) == r"i J^{3} S^{y}_{1} \left(S^{z}_{2}\right)^{3}"


class TestSpectral:
    def test_infinite_temperature_inner_product(self):
        assert inner(spin_operator("x", 1), spin_operator("x", 1)).to_latex() == r"\frac{1}{4}"
        assert inner(spin_operator("x", 1), spin_operator("y", 1)).is_zero()

    def test_ising_moments(self):
        h = build_hamiltonian([(1, 2)], kind="ising")
        ms = moments(h, spin_operator("x", 1), 2)
        assert [m.to_latex() for m in ms] == [r"\frac{1}{4}", r"\frac{1}{16} J^{2}",
                                              r"\frac{1}{64} J^{4}"]

    def test_continued_fraction_terminates(self):
        # 2-site Ising: b_1²=J²/4, then the next Krylov vector is null → stop
        h = build_hamiltonian([(1, 2)], kind="ising")
        bs = continued_fraction(h, spin_operator("x", 1), 4)
        assert [b.to_latex() for b in bs] == [r"\frac{1}{4} J^{2}"]

    def test_continued_fraction_b1_is_second_moment_ratio(self):
        # Heisenberg chain, site 1 (one neighbour): b_1² = M_2/M_0 = J²/2
        h = build_hamiltonian([(1, 2), (2, 3), (3, 4)], kind="heisenberg")
        bs = continued_fraction(h, spin_operator("x", 1), 3)
        assert bs[0].to_latex() == r"\frac{1}{2} J^{2}"


class TestGeneralFormula:
    def test_ising_m2_in_z(self):
        m0, m2 = general_low_moments("ising", "x")
        assert m0.to_latex() == r"\frac{1}{4}"
        assert m2.to_latex() == r"\frac{1}{16} J^{2} z"

    def test_heisenberg_m2_and_b1(self):
        _, m2 = general_low_moments("heisenberg", "x")
        assert m2.to_latex() == r"\frac{1}{8} J^{2} z"
        assert general_b1_squared("heisenberg", "x").to_latex() == r"\frac{1}{2} J^{2} z"

    def test_xxz_anisotropy_in_general_moment(self):
        _, m2 = general_low_moments("xxz", "x")
        assert m2.to_latex() == r"\frac{1}{16} J^{2} z + \frac{1}{16} J_z^{2} z"

    def test_general_reproduces_concrete_at_fixed_z(self):
        # interior chain site (z=2) Heisenberg: concrete M_2 = J²/4 = (z J²/8)|_{z=2}
        h = build_hamiltonian([(1, 2), (2, 3)], kind="heisenberg")
        concrete_m2 = moments(h, spin_operator("x", 2), 1)[1]
        assert concrete_m2.to_latex() == r"\frac{1}{4} J^{2}"


class TestEnergyMomentsTriangle:
    def test_ising_h3_is_pure_triangle(self):
        # Ising ⟨H³⟩ has only the triangle term (3/32 N△ J³)
        assert energy_moment_general_latex("ising") == (
            r"\langle H^2 \rangle = \frac{1}{16} J^{2} |E|,\ "
            r"\langle H^3 \rangle = \frac{3}{32} J^{3} N_\triangle")

    def test_heisenberg_h3_has_edge_and_triangle(self):
        assert energy_moment_general_latex("heisenberg") == (
            r"\langle H^2 \rangle = \frac{3}{16} J^{2} |E|,\ "
            r"\langle H^3 \rangle = \frac{9}{32} J^{3} N_\triangle "
            r"- \frac{3}{32} J^{3} |E|")

    def test_no_triangle_no_h3_for_ising(self):
        # a triangle-free graph (square 2×2 = 4-cycle): Ising ⟨H³⟩ = 0
        assert energy_moment("ising", [(1, 2), (2, 3), (3, 4), (4, 1)], 3).is_zero()

    def test_triangle_lattice_concrete_matches_formula(self):
        # square 3×3 (|E|=12, 0 triangles) Heisenberg ⟨H³⟩ = -3/32·12 = -9/8
        sq = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9),
              (1, 4), (4, 7), (2, 5), (5, 8), (3, 6), (6, 9)]
        assert energy_moment("heisenberg", sq, 3).to_latex() == r"-\frac{9}{8} J^{3}"

    def test_h4_general_has_square_term(self):
        latex = energy_moment_general_latex("heisenberg", upto=4)
        assert r"\langle H^4 \rangle" in latex
        assert r"\frac{9}{32} J^{4} N_\square" in latex      # the 4-cycle term

    def test_kappa4_drops_disconnected_e2(self):
        # κ₄ is connected: the disconnected |E|² piece must cancel
        _, _, k4 = energy_cumulants_general("heisenberg")
        assert not any("|E|" == nm and ex == 2 for mono in k4.d for nm, ex in mono)

    def test_specific_heat_leading_term_is_edges(self):
        latex = specific_heat_general_latex("heisenberg")
        assert latex.startswith(r"C(\beta) = \frac{3}{16} J^{2} \beta^{2} |E|")
        assert r"N_\square" in latex          # 4-cycle enters at β⁴

    def test_specific_heat_concrete_square(self):
        # square 2×2 (|E|=4, N△=0, N□=1): C(β)=¾J²β²+⅜J³β³−3/32 J⁴β⁴
        latex = specific_heat_concrete_latex("heisenberg",
                                             [(1, 2), (2, 3), (3, 4), (4, 1)])
        assert latex == (r"C(\beta) = \frac{3}{4} J^{2} \beta^{2} "
                         r"+ \frac{3}{8} J^{3} \beta^{3} "
                         r"- \frac{3}{32} J^{4} \beta^{4} + O(\beta^5)")

    def test_h4_fit_reproduces_held_out_graphs(self):
        # the {|E|,|E|²,wedge,N△,N□} fit must reproduce graphs not used to fit it
        from notebook.runtime.spin_algebra import energy_moment4_coeffs
        from notebook.runtime.backend_handler import (
            build_adjacency_matrix, graph_edge_count, graph_triangle_count,
            graph_four_cycle_count, graph_degree_sequence)
        coeffs = energy_moment4_coeffs("heisenberg")
        for edges in ([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],   # K4
                      [(1, 2), (2, 3), (3, 1), (3, 4)]):                   # paw
            a, _ = build_adjacency_matrix("custom", edges=edges)
            e = graph_edge_count(a)
            w = sum(d * (d - 1) // 2 for d in graph_degree_sequence(a))
            row = [e, e * e, w, graph_triangle_count(a), graph_four_cycle_count(a)]
            pred = Coeff()
            for cf, v in zip(coeffs, row):
                pred = pred + cf * Coeff({(): (v, 0)})
            assert pred == energy_moment("heisenberg", edges, 4)


class TestGeneralSpinTrace:
    """F-1s: general-spin-S single-site trace as a polynomial in x = S(S+1)."""

    def test_known_two_and_three_point_traces(self):
        assert _site_trace_poly(["z", "z"]).to_latex() == r"\frac{1}{3} x"
        assert _site_trace_poly(["x", "x"]).to_latex() == r"\frac{1}{3} x"
        assert _site_trace_poly(["x", "y"]).is_zero()
        # ⟨S^x S^y S^z⟩ = (i/6) x
        assert _site_trace_poly(["x", "y", "z"]).to_latex() == r"\frac{1}{6} i x"

    def test_sz_fourth_moment(self):
        # ⟨(S^z)^4⟩ = (3x² − x)/15
        assert _site_trace_poly(["z", "z", "z", "z"]).to_latex() == (
            r"-\frac{1}{15} x + \frac{1}{5} x^{2}")

    def test_reduces_to_spin_half_exhaustively(self):
        # at x = S(S+1) = 3/4 the general trace must equal the spin-½ expectation
        from notebook.runtime.spin_algebra import _site_identity_coeff
        import itertools
        from fractions import Fraction
        def at_half(c):
            re, im = Fraction(0), Fraction(0)
            for mono, (r, i) in c.d.items():
                p = Fraction(1)
                for nm, e in mono:
                    p *= Fraction(3, 4) ** e  # only the lone x parameter appears
                re += r * p; im += i * p
            return (re, im)
        for k in range(1, 5):
            for comps in itertools.product("xyz", repeat=k):
                assert at_half(_site_trace_poly(list(comps))) == \
                    _site_identity_coeff(list(comps)), comps

    def test_general_spin_energy_moment_reduces_to_spin_half(self):
        # ⟨H²⟩, ⟨H³⟩ general formulas reduce to the spin-½ invariants at x=3/4
        latex = general_spin_energy_moment_latex("heisenberg", upto=3)
        assert latex == (
            r"\langle H^2 \rangle = \frac{1}{3} J^{2} x^{2} |E|,\ "
            r"\langle H^3 \rangle = \frac{2}{3} J^{3} N_\triangle x^{3} "
            r"- \frac{1}{6} J^{3} x^{2} |E|\quad (x = S(S+1))")

    def test_general_spin_heisenberg_bond_second_moment(self):
        # single Heisenberg bond ⟨H²⟩ = J² x²/3  (→ 3/16 J² at x=3/4)
        assert general_energy_moment("heisenberg", [(0, 1)], 2).to_latex() == \
            r"\frac{1}{3} J^{2} x^{2}"

    def test_pipeline_parses_general_spin_request(self):
        # "一般スピンで…の高温展開を求めて" routes to the general-S formula
        from ui.interpret_pipeline import _parse_energy_moment
        spec = _parse_energy_moment("ハイゼンベルク模型の一般スピンでの高温展開を求めて")
        assert spec is not None and spec["general_spin"] is True
        # a plain request must NOT flip the general-spin flag
        plain = _parse_energy_moment("ハイゼンベルク模型の高温展開を一般式で求めて")
        assert plain is not None and plain["general_spin"] is False


class TestSpectralFunction:
    def test_greens_function_rational(self):
        # 2-site Ising single mode: G(z) = z/(z² − J²/4)
        h = build_hamiltonian([(1, 2)], kind="ising")
        b2 = continued_fraction(h, spin_operator("x", 1), 8)
        num, den = greens_function(b2)
        assert num.to_latex() == "z"
        assert den.to_latex() == r"-\frac{1}{4} J^{2} + z^{2}"

    def test_single_mode_spectral_peaks(self):
        h = build_hamiltonian([(1, 2)], kind="ising")
        b2 = continued_fraction(h, spin_operator("x", 1), 8)
        latex = spectral_latex(b2)
        # exact spin precession at ω = J/2
        assert r"\delta(\omega - \frac{1}{2} J)" in latex
        assert r"\delta(\omega + \frac{1}{2} J)" in latex

    def test_two_mode_closed_form_peaks(self):
        # Heisenberg 3-chain (2 coeffs): central peak + ±√(b₁²+b₂²)=±3J/2
        h = build_hamiltonian([(1, 2), (2, 3)], kind="heisenberg")
        b2 = continued_fraction(h, spin_operator("x", 2), 8)
        peaks = spectral_peaks(b2)
        assert [(w.to_latex(), om.to_latex()) for w, om in peaks] == [
            (r"\frac{5}{9}", "0"),
            (r"\frac{2}{9}", r"\frac{3}{2} J"),
            (r"\frac{2}{9}", r"-\frac{3}{2} J")]

    def test_spectral_weights_sum_to_one(self):
        h = build_hamiltonian([(1, 2), (2, 3)], kind="heisenberg")
        peaks = spectral_peaks(continued_fraction(h, spin_operator("x", 2), 8))
        total = Coeff()
        for w, _ in peaks:
            total = total + w
        assert total == Coeff.int(1)            # sum rule

    def test_higher_mode_falls_back_to_rational(self):
        # ≥3 coefficients with no clean √ → rational G(z)=P/Q (poles = roots of Q).
        # Exercise spectral_latex's rational fallback directly with synthetic
        # coefficients instead of an order-8 continued_fraction on a 4-site
        # Heisenberg ring: that end-to-end expansion grows combinatorially (>2 GB),
        # while continued_fraction itself is covered by the order-2 tests above.
        b2 = [Coeff.int(1), Coeff.int(2), Coeff.int(3)]
        assert len(b2) >= 3
        assert not spectral_peaks(b2)              # no closed-form poles
        assert r"G(z) = \frac{" in spectral_latex(b2)   # rational form shown

    def test_terminator_semicircle_closed_form(self):
        # infinite-system continuous band: √-terminator + Wigner semicircle
        latex = terminator_spectral_latex()
        assert r"t(z) = \frac{z - \sqrt{z^2 - 4 b_\infty^2}}{2}" in latex
        assert r"S(\omega) = \frac{\sqrt{4 b_\infty^2 - \omega^2}}{2\pi b_\infty^2}" in latex
        assert r"|\omega| \le 2 b_\infty" in latex


class TestRotatingFrame:
    def test_frame_term_shifts_zeeman(self):
        # Adding the frame term +ω Sz to lab (-ω0 Sz - ω1 Sx) → (ω-ω0) Sz - ω1 Sx.
        # The rotating frame is just this addition (no dressed-up transform op).
        out = parse_spin_expression(r"-\omega_0 S^z_1 - \omega_1 S^x_1 + \omega S^z_1")
        assert to_latex(out) == r"\left( \omega - \omega_0 \right) S^{z}_{1} - \omega_1 S^{x}_{1}"

    def test_rotating_frame_precession_eom(self):
        # [H_rot, Sz] = i ω1 Sy (the RF nutates Sz); [H_rot, Sx] = -i Ω Sy.
        # H_rot is built from a general spin expression, not a canned helper.
        hr = parse_spin_expression(r"-\Omega S^z_1 - \omega_1 S^x_1")
        assert to_latex(commutator(hr, spin_operator("z", 1))) == r"i \omega_1 S^{y}_{1}"
        assert to_latex(commutator(hr, spin_operator("x", 1))) == r"-i \Omega S^{y}_{1}"


class TestParseSpinCommutator:
    def test_double_order_and_model(self):
        spec = _parse_spin_commutator("ハイゼンベルク模型 3サイト鎖で S^x_1 の二重交換子を求めて")
        assert spec["kind"] == "heisenberg" and spec["order"] == 2
        assert spec["comp"] == "x" and spec["site"] == 1
        assert spec["spec"]["n_sites"] == 3

    def test_ising_first_order(self):
        spec = _parse_spin_commutator("イジング模型 2サイトで S^z_2 の運動方程式を求めて")
        assert spec["kind"] == "ising" and spec["order"] == 1 and spec["comp"] == "z"

    def test_needs_commutator_keyword(self):
        assert _parse_spin_commutator("ハイゼンベルク模型 3サイト鎖の S^x_1") is None

    def test_xxz_order_three(self):
        spec = _parse_spin_commutator("XXZ模型 2サイトで S^x_1 の3段交換子を求めて")
        assert spec["kind"] == "xxz" and spec["order"] == 3

    def test_xxz_order_three_en(self):
        spec = _parse_spin_commutator(
            "XXZ model on a 2-site chain: the 3rd-order commutator of S^x_1", "en")
        assert spec["kind"] == "xxz" and spec["order"] == 3

    def test_double_order_and_model_en(self):
        spec = _parse_spin_commutator(
            "Heisenberg model on a 3-site chain: the double commutator of S^x_1", "en")
        assert spec["kind"] == "heisenberg" and spec["order"] == 2
        assert spec["comp"] == "x" and spec["site"] == 1
        assert spec["spec"]["n_sites"] == 3

    def test_order_n_en_fold_form(self):
        spec = _parse_spin_commutator(
            "Heisenberg model on a 3-site chain: the 4-fold commutator of S^x_1", "en")
        assert spec["order"] == 4

    def test_transverse_field(self):
        spec = _parse_spin_commutator("横磁場イジング模型 2サイトで S^z_1 の交換子を求めて")
        assert spec["kind"] == "ising" and spec["field"] == ("h", "x")

    def test_xy_model_parse(self):
        spec = _parse_spin_commutator("XY模型 2サイトで S^z_1 の交換子を求めて")
        assert spec["kind"] == "xy" and spec["field"] is None

    def test_moments_parse(self):
        """`2次まで` は $M_0, M_2$ ——recursion は **1段**（起票 MOMENT-ORDER-MEANING）。
        直す前はここが `depth == 2` を正しいとしており、`2次` と書いて $M_4$ が
        返ることを固定していた。"""
        spec = _parse_spin_spectral("ハイゼンベルク模型 3サイト鎖で S^x_2 の2次までのモーメントを求めて")
        assert spec["mode"] == "moments" and spec["depth"] == 1 and spec["kind"] == "heisenberg"
        assert spec["requested"] == 2, spec

    def test_continued_fraction_parse(self):
        spec = _parse_spin_spectral("イジング模型 2サイトで S^x_1 の連分数を求めて")
        assert spec["mode"] == "contfrac"

    def test_commutator_not_spectral(self):
        assert _parse_spin_spectral("イジング模型 2サイトで S^x_1 の交換子を求めて") is None

    def test_general_mode_no_lattice(self):
        spec = _parse_spin_spectral("ハイゼンベルク模型で一般の格子の S^x の2次モーメントを求めて")
        assert spec["general"] is True and spec["spec"] is None
        assert spec["kind"] == "heisenberg" and spec["comp"] == "x"


def _sup():
    s = MagicMock(spec=LLMSupervisor)
    s._v = OperationValidator()
    s.needs_expression_injection.return_value = False
    return s


def _run(engine, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, _sup()))


class TestSpinPipeline:
    def test_first_commutator_command(self):
        nb.start_engine(title="S1", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型 2サイトで S^x_1 の交換子を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spin_commutator"
            latex = eng.expressions[r["expr_id"]].latex
            assert latex == r"[H, S^{x}_{1}] = i J S^{y}_{1} S^{z}_{2}"
        finally:
            nb.stop_engine()

    def test_double_commutator_command(self):
        nb.start_engine(title="S2", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型 2サイトで S^x_1 の二重交換子を求めて")
            assert r["status"] == "completed"
            latex = eng.expressions[r["expr_id"]].latex
            assert latex == (r"[H, [H, S^{x}_{1}]] = "
                             r"J^{2} S^{x}_{1} \left(S^{z}_{2}\right)^{2}")
        finally:
            nb.stop_engine()

    def test_xxz_command(self):
        nb.start_engine(title="S3", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "XXZ模型 2サイトで S^x_1 の交換子を求めて")
            assert r["status"] == "completed"
            assert r"J_z" in eng.expressions[r["expr_id"]].latex
        finally:
            nb.stop_engine()

    def test_transverse_field_command(self):
        nb.start_engine(title="S4", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "横磁場イジング模型 2サイトで S^z_1 の交換子を求めて")
            assert r["status"] == "completed"
            assert eng.expressions[r["expr_id"]].latex == r"[H, S^{z}_{1}] = i h S^{y}_{1}"
        finally:
            nb.stop_engine()

    def test_moments_command(self):
        nb.start_engine(title="S5", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型 2サイトで S^x_1 の2次までのモーメントを求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spin_moments"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"M_{0} = \frac{1}{4}" in latex
            assert r"M_{2} = \frac{1}{16} J^{2}" in latex
        finally:
            nb.stop_engine()

    def test_continued_fraction_command(self):
        nb.start_engine(title="S6", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型 2サイトで S^x_1 の連分数を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spin_contfrac"
            assert eng.expressions[r["expr_id"]].latex == r"b_{1}^2 = \frac{1}{4} J^{2}"
        finally:
            nb.stop_engine()

    def test_general_moment_command(self):
        nb.start_engine(title="S7", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型で一般の格子の S^x の2次モーメントを求めて")
            assert r["status"] == "completed"
            latex = eng.expressions[r["expr_id"]].latex
            assert latex == r"M_0 = \frac{1}{4},\ M_2 = \frac{1}{8} J^{2} z"
        finally:
            nb.stop_engine()

    def test_general_continued_fraction_command(self):
        nb.start_engine(title="S8", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型で配位数 z の S^x の連分数を求めて")
            assert r["status"] == "completed"
            assert eng.expressions[r["expr_id"]].latex == r"b_1^2 = \frac{1}{4} J^{2} z"
        finally:
            nb.stop_engine()

    def test_energy_moment_general_command(self):
        nb.start_engine(title="S9", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型の高温展開を一般式で求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "energy_moment"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"N_\triangle" in latex and r"|E|" in latex
        finally:
            nb.stop_engine()

    def test_energy_moment_concrete_command(self):
        nb.start_engine(title="S10", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型 正方格子 3x3 の高温展開を求めて")
            assert r["status"] == "completed"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"N_\triangle = 0" in latex and "|E| = 12" in latex
        finally:
            nb.stop_engine()

    def test_energy_moment_h4_general_command(self):
        nb.start_engine(title="S11", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型の高温展開を一般式で4次まで求めて")
            assert r["status"] == "completed"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"\langle H^4 \rangle" in latex and r"N_\square" in latex
        finally:
            nb.stop_engine()

    def test_energy_moment_h4_concrete_shows_square_count(self):
        nb.start_engine(title="S12", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            # square 2×2 = one 4-cycle (N□=1, N△=0)
            r = _run(eng, "ハイゼンベルク模型 正方格子 2x2 の高温展開を4次まで求めて")
            assert r["status"] == "completed"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"\langle H^4 \rangle" in latex and r"N_\square = 1" in latex
        finally:
            nb.stop_engine()

    def test_specific_heat_general_command(self):
        nb.start_engine(title="S13", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型の比熱の高温展開を一般式で求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "specific_heat"
            latex = eng.expressions[r["expr_id"]].latex
            assert latex.startswith(r"C(\beta) =") and r"\beta^{2} |E|" in latex
        finally:
            nb.stop_engine()

    def test_specific_heat_concrete_command(self):
        nb.start_engine(title="S14", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型 正方格子 2x2 の比熱の高温展開を求めて")
            assert r["status"] == "completed"
            assert r"\frac{3}{4} J^{2} \beta^{2}" in eng.expressions[r["expr_id"]].latex
        finally:
            nb.stop_engine()

    def test_spectral_function_command(self):
        nb.start_engine(title="S15", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "イジング模型 2サイトで S^x_1 のスペクトル関数を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spectral_function"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"G(z) =" in latex
            assert r"\delta(\omega - \frac{1}{2} J)" in latex
        finally:
            nb.stop_engine()

    def test_greens_function_command_multimode(self):
        nb.start_engine(title="S16", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルク模型 3サイト鎖で S^x_2 のグリーン関数を求めて")
            assert r["status"] == "completed"
            assert r"\cfrac" in eng.expressions[r["expr_id"]].latex
        finally:
            nb.stop_engine()

    def test_terminator_command(self):
        nb.start_engine(title="S17", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "平方根終端の連続スペクトル密度を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spectral_terminator"
            latex = eng.expressions[r["expr_id"]].latex
            assert r"\sqrt{z^2 - 4 b_\infty^2}" in latex and r"S(\omega)" in latex
        finally:
            nb.stop_engine()


class TestComplexityLimit:
    """J-10 (scenario-narrative-notes review, item 14, 2026-07-08): a lattice larger
    than the 2-3 site systems this module is validated against (e.g. a 4-site chain's
    spectral function) makes moments/continued_fraction's term count blow up
    unboundedly — confirmed via direct profiling: 1 -> 4 -> 27 -> 164 -> 967 terms in
    4 commutator steps, ~20s and still climbing, heading toward a multi-minute,
    multi-GB hang with no result (the module has no reduction rule for two+ operator
    factors landing on the same site, e.g. repeated commutators on a chain
    reintroducing a factor on a site an earlier factor already occupies; this
    representation is deliberately left un-simplified elsewhere too — see
    test_double_commutator_command's (S^z_2)^2 — so "duplicate site" alone cannot be
    treated as an error).  _collect now raises ComplexityLimitExceeded once a term
    list passes a safety threshold, turning the silent hang into an immediate,
    catchable failure — NONE of the validated 2-3 site scenarios (term counts in the
    tens) come close to it.

    **A-5d, 2026-08-06: the reduction rule the docstring above says is missing now
    exists** (`spin_algebra._reduce_same_site_half`,
    `tests/test_spin_half_reduction.py`), so the 4-site chain that named this class
    completes in ~0.1s.  Two paths, two rules, because they assume different things:

    * **moments / continued_fraction** reduce ALWAYS — their inner product is the
      spin-½ trace by construction, so there is no other spin to preserve.
    * **commutator / nested_commutator** (what gets DISPLAYED) reduce only where the
      researcher declared spin-½ — the general-spin `(S^z_2)^2` in
      `test_double_commutator_command` is a deliberate result, not an unsimplified one.

    The limit stays, and stays tested: the wall moved from 4 sites to 8, it was not
    abolished.
    """
    def test_four_site_chain_spectral_function_now_completes(self):
        """A-5d (2026-08-06): this is the case the whole class was written for, and
        it no longer fails — it completes in ~0.1s.

        The reduction is unconditional HERE (not gated on a declaration) because the
        recursion method's inner product IS the spin-½ trace: `inner` → `expectation`,
        whose own docstring says spin-½, and `expectation_general` refuses a symbolic
        S outright.  Asking the researcher to declare ½ on this path would be asking
        them to declare what the calculation already assumes.
        """
        import time
        nb.start_engine(title="S18", mode="standard", language="en",
                        domains=["statistical_mechanics"])
        try:
            eng = nb.get_engine()
            t0 = time.time()
            r = _run(eng, "Find the spectral function of S^x_2 for the "
                          "Heisenberg model on a 4-site chain")
            elapsed = time.time() - t0
            assert r["status"] == "completed", r.get("error")
            assert elapsed < 30, f"took {elapsed}s"
        finally:
            nb.stop_engine()

    def test_the_limit_still_bounds_runtime_on_a_larger_chain(self):
        """The guard itself must stay working — the reduction moved the wall from 4
        sites to 8, it did not remove the need for a bound.  Measured: 4 sites 0.11s,
        6 sites 4.9s, 8 sites hits the limit in ~1s (a FAST failure, not a hang)."""
        import time
        import notebook.runtime.spin_algebra as spa
        h = spa.parse_spin_expression(" + ".join(
            f"S^{a}_{i} S^{a}_{i+1}" for i in range(1, 8) for a in "xyz"))
        t0 = time.time()
        with pytest.raises(spa.ComplexityLimitExceeded):
            spa.continued_fraction(h, spa.spin_operator("x", 1), 8)
        elapsed = time.time() - t0
        assert elapsed < 60, f"took {elapsed}s — safety limit did not bound runtime"

    def test_validated_two_and_three_site_scenarios_unaffected(self):
        """Regression guard: the safety limit must never fire for any of the
        documented, validated commands (would silently break a working scenario)."""
        nb.start_engine(title="S19", mode="standard", language="en",
                        domains=["statistical_mechanics"])
        try:
            eng = nb.get_engine()
            for cmd in (
                "Find the commutator of S^x_1 for the Ising model with 2 sites",
                "Find the double commutator of S^x_1 for the Ising model with 2 sites",
                "Find the moments up to 2nd order of S^x_1 for the Ising model with 2 sites",
                "Find the continued fraction of S^x_1 for the Ising model with 2 sites",
                "Find the spectral function of S^x_1 for the Ising model with 2 sites",
                "Find the spectral function of S^x_2 for the Heisenberg model on a 3-site chain",
            ):
                r = _run(eng, cmd)
                assert r["status"] == "completed", (cmd, r)
        finally:
            nb.stop_engine()

    def test_rotating_frame_general_workflow(self):
        # The rotating-frame derivation as a sequence of GENERAL spin operations:
        # register a spin H, ADD the frame term (H_rot = E1 + ω S^z_1), commutator EOM.
        nb.start_engine(title="S18", mode="standard", language="ja",
                        domains=["magnetic_resonance"])
        try:
            eng = nb.get_engine()
            _run(eng, r"H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1")     # E1
            r2 = _run(eng, r"H_{rot} = E1 + \omega S^z_1")              # E2 (add frame term)
            assert r2["operation"] == "spin"
            assert eng.expressions[r2["expr_id"]].latex == \
                r"H_{rot} = \left( \omega - \omega_0 \right) S^{z}_{1} - \omega_1 S^{x}_{1}"
            r4 = _run(eng, r"E2 と S^z_1 の交換子を求めて")              # precession EOM
            assert eng.expressions[r4["expr_id"]].latex == \
                r"[H_{rot}, S^z_1] = i \omega_1 S^{y}_{1}"
        finally:
            nb.stop_engine()
