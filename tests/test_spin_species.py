"""F-13: multi-species spin — a different spin magnitude per lattice site.

The spin algebra could always WRITE an electron-plus-nucleus system (different
sites commute, so ``A I·S = A Σ_a S^a_1 S^a_2`` and its commutators were fine);
what it could not do was EVALUATE one, because the infinite-temperature trace
used a single ``x = S(S+1)`` for every site.  These tests pin the per-site
generalization ``x_i = S_i(S_i+1)``.

Verification leans on two independent checks rather than on stored numbers:
setting every site to the same spin must reproduce the uniform-spin polynomial
evaluated at that x, and setting every site to ½ must reproduce
``spin_algebra.expectation()`` — a separate implementation that works in the
Pauli algebra instead of expanding in S^±.
"""
import asyncio
from fractions import Fraction
from unittest.mock import MagicMock

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime import spin_algebra as spa
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

F = Fraction

# I·S for an electron (site 1) and a nucleus (site 2) — the hyperfine coupling.
DOT_12 = "S^x_1 S^x_2 + S^y_1 S^y_2 + S^z_1 S^z_2"


def _dot():
    return spa.parse_spin_expression(DOT_12)


def _square(expr):
    return spa.multiply(expr, expr)


# ---------------------------------------------------------------------------
# The trace itself
# ---------------------------------------------------------------------------

class TestPerSiteTrace:
    def test_uniform_behaviour_is_untouched(self):
        # No spins argument → the pre-F-13 single-x polynomial, exactly.
        h = spa.build_hamiltonian([(1, 2), (2, 3)], "heisenberg")
        assert spa.expectation_general(_square(h)).to_latex() == \
            r"\frac{2}{3} J^{2} x^{2}"

    def test_first_moment_of_a_coupling_vanishes(self):
        # Every factor is traceless, so ⟨I·S⟩ = 0 whatever the spins are.
        for spins in (None, {1: None, 2: None}, {1: "1/2", 2: "1"}):
            assert spa.expectation_general(_dot(), spins).is_zero()

    def test_second_moment_separates_the_two_species(self):
        # ⟨(I·S)²⟩ = x₁x₂/3 — the whole point: one x per site, not x².
        got = spa.expectation_general(_square(_dot()), {1: None, 2: None})
        assert got.to_latex() == r"\frac{1}{3} x1 x2"

    @pytest.mark.parametrize("s1,s2,expected", [
        ("1/2", "1", F(1, 2)),        # electron + spin-1 nucleus
        ("1/2", "1/2", F(3, 16)),     # two spin-½
        ("1/2", "3/2", F(15, 16)),
        ("1", "1", F(4, 3)),
    ])
    def test_second_moment_at_definite_spins(self, s1, s2, expected):
        got = spa.expectation_general(_square(_dot()), {1: s1, 2: s2})
        assert got == spa.Coeff({(): (expected, F(0))})

    def test_one_site_definite_one_symbolic(self):
        # x₁ · I(I+1)/3 with I = 1 → 2x₁/3
        got = spa.expectation_general(_square(_dot()), {1: None, 2: "1"})
        assert got.to_latex() == r"\frac{2}{3} x1"

    @pytest.mark.parametrize("spin", ["1/2", "1", "3/2", "2"])
    def test_equal_spins_reproduce_the_uniform_polynomial(self, spin):
        h = spa.build_hamiltonian([(1, 2), (2, 3)], "heisenberg")
        uniform = spa.expectation_general(_square(h))
        x = F(spin) * (F(spin) + 1)
        per_site = spa.expectation_general(_square(h), {i: spin for i in (1, 2, 3)})
        assert per_site == spa.subst_coeff(uniform, "x", x)

    def test_all_spin_half_matches_the_independent_pauli_implementation(self):
        h = spa.build_hamiltonian([(1, 2), (2, 3)], "heisenberg")
        for expr in (_square(_dot()), _square(h), spa.commutator(h, _dot())):
            sites = {site for _, ops in expr for _, site in ops}
            assert spa.expectation_general(expr, {s: "1/2" for s in sites}) == \
                spa.expectation(expr)

    def test_unlisted_sites_stay_symbolic_and_separate(self):
        # Once the caller says the sites differ, no site falls back to shared x.
        got = spa.expectation_general(_square(_dot()), {1: "1/2"})
        assert got.to_latex() == r"\frac{1}{4} x2"

    @pytest.mark.parametrize("bad", ["1/3", "-1/2", "0.7", "abc"])
    def test_a_spin_that_is_not_a_half_integer_is_rejected(self, bad):
        with pytest.raises(ValueError):
            spa.expectation_general(_square(_dot()), {1: bad, 2: "1"})

    def test_site_variable_carries_no_subscript(self):
        # Regression guard: "x_1" is lowered to the indexed form x(1) on the way
        # to the CAS, and a later "substitute x_1 = 3/4" then matches nothing
        # while still reporting success.  The name must survive that trip.
        assert spa.site_spin_variable(1) == "x1"
        assert "_" not in spa.site_spin_variable(12)


class TestCoeffCanonical:
    @pytest.mark.parametrize("coeff,expected", [
        (spa.Coeff(), "0"),
        (spa.Coeff.int(1), "1"),
        (spa.Coeff({(): (F(-1, 2), F(0))}), "-1/2"),
        (spa.Coeff({(("J", 2), ("x1", 1)): (F(2, 3), F(0))}), "(2*J^2*x1)/3"),
        (spa.Coeff({(("J", 1),): (F(0), F(-3, 4))}), "-(3*%i*J)/4"),
    ])
    def test_canonical_rendering(self, coeff, expected):
        assert coeff.to_canonical() == expected

    def test_canonical_is_cas_parseable_and_substitutable(self):
        value = spa.expectation_general(_square(_dot()), {1: None, 2: None})
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            expr = e.register_expression(value.to_canonical())
            step, out = e.run_operation("substitute", {
                "expr": expr.display_id, "lhs": "x1", "rhs": "3/4"})
            assert step.status.value == "success"
            assert "x1" not in out          # the substitution actually happened


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _register_hyperfine(engine, label=r"H_{hf} = "):
    body = (r"A S^{x}_{1} S^{x}_{2} + A S^{y}_{1} S^{y}_{2} + A S^{z}_{1} S^{z}_{2}")
    return engine.register_dispersion(label + body, {"kind": "spin"})


class TestEngine:
    def test_declaration_is_recorded_and_readable(self):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            step = e.declare_site_spins({1: "1/2", 2: "1"})
            assert e.symbol_context.site_spins == {"1": "1/2", "2": "1"}
            assert "S_1 = 1/2" in step.visible_action
            assert e.site_spin_map() == {1: F(1, 2), 2: F(1)}

    def test_declaration_survives_save_and_load(self, tmp_path):
        path = tmp_path / "f13.json"
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            e.declare_site_spins({1: "1/2", 2: "1"})
            e.save(str(path))
        with NotebookEngine.load(str(path)) as loaded:
            assert loaded.symbol_context.site_spins == {"1": "1/2", "2": "1"}
            assert loaded.site_spin_map() == {1: F(1, 2), 2: F(1)}

    def test_a_session_saved_before_f13_loads_as_uniform_spin(self, tmp_path):
        path = tmp_path / "old.json"
        with NotebookEngine(title="old", mode=SessionMode.standard_symbolic) as e:
            e.save(str(path))
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("site_spins", None)                 # simulate the older file
        path.write_text(json.dumps(data), encoding="utf-8")
        with NotebookEngine.load(str(path)) as loaded:
            assert loaded.symbol_context.site_spins == {}
            assert loaded.site_spin_map() is None

    def test_expectation_uses_the_declared_spins(self):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            h = _register_hyperfine(e)
            e.declare_site_spins({1: "1/2", 2: "1"})
            assert e.spin_expectation(h.display_id, 1).canonical == "0"
            second = e.spin_expectation(h.display_id, 2)
            assert second.canonical == "A^2/2"
            assert second.latex.startswith(r"\langle {H_{hf}}^{2} \rangle")

    def test_without_a_declaration_the_uniform_result_is_kept(self):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            h = _register_hyperfine(e)
            assert e.spin_expectation(h.display_id, 2).canonical == "(A^2*x^2)/3"

    def test_bracket_falls_back_to_the_expression_id_when_unnamed(self):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            bare = e.register_dispersion(r"S^{z}_{1} S^{z}_{2}", {"kind": "spin"})
            e.declare_site_spins({1: "1/2", 2: "1"})
            got = e.spin_expectation(bare.display_id, 2)
            assert got.latex == rf"\langle {{{bare.display_id}}}^{{2}} \rangle = \frac{{1}}{{6}}"

    @pytest.mark.parametrize("spins", [{}, {1: "1/3"}, {1: "-1"}, {"a": "1/2"}])
    def test_bad_declarations_are_rejected(self, spins):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            with pytest.raises(ValueError):
                e.declare_site_spins(spins)

    def test_bad_moment_order_and_unknown_expression_are_rejected(self):
        with NotebookEngine(title="f13", mode=SessionMode.standard_symbolic) as e:
            h = _register_hyperfine(e)
            with pytest.raises(ValueError):
                e.spin_expectation(h.display_id, 0)
            with pytest.raises(ValueError):
                e.spin_expectation("E99", 1)


# ---------------------------------------------------------------------------
# Natural language — both languages
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
    @pytest.mark.parametrize("lang,text", [
        ("ja", "サイト1のスピンを1/2、サイト2のスピンを1とする"),
        ("en", "site 1 has spin 1/2 and site 2 has spin 1"),
        ("en", "spin 1/2 on site 1, spin 1 on site 2"),
    ])
    def test_declaration_in_both_languages(self, lang, text):
        nb.start_engine(title="f13", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r["status"] == "completed", r.get("error")
            assert r["operation"] == "declare_site_spins"
            assert nb.get_engine().symbol_context.site_spins == {"1": "1/2", "2": "1"}
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,declare,first,second", [
        ("ja", "サイト1のスピンを1/2、サイト2のスピンを1とする",
         "E1 の期待値を求めて", "E1 の2次モーメントを求めて"),
        ("en", "site 1 has spin 1/2 and site 2 has spin 1",
         "expectation value of E1", "compute the 2nd moment of E1"),
    ])
    def test_hyperfine_workflow(self, lang, declare, first, second):
        nb.start_engine(title="f13", mode="standard",
                        domains=["magnetic_resonance"], language=lang)
        try:
            engine = nb.get_engine()
            _register_hyperfine(engine)
            assert _run(engine, declare)["status"] == "completed"
            r1 = _run(engine, first)
            assert r1["operation"] == "spin_expectation"
            assert r1["registered_expressions"][-1]["canonical"] == "0"
            r2 = _run(engine, second)
            assert r2["operation"] == "spin_expectation"
            assert r2["registered_expressions"][-1]["canonical"] == "A^2/2"
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        ("ja", "ハイゼンベルク模型の3サイト鎖のエネルギーモーメントを求めて"),
        ("en", "compute the energy moments of the Heisenberg model on a 3-site chain"),
    ])
    def test_lattice_energy_moment_is_not_stolen(self, lang, text):
        # "モーメント"/"moment" also belongs to the F-1 lattice path, which
        # carries a model name and no E-id.
        nb.start_engine(title="f13", mode="standard", language=lang)
        try:
            r = _run(nb.get_engine(), text)
            assert r.get("operation") != "spin_expectation"
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を展開して"),
        ("en", "expand E1"),
        ("ja", "サイト1とサイト2の交換子を求めて"),
        ("en", "commutator of site 1 and site 2"),
    ])
    def test_unrelated_instructions_fall_through(self, lang, text):
        from ui import interpret_spin_species as iss
        for handler in (iss._try_declare_site_spins, iss._try_spin_expectation):
            assert asyncio.run(handler(None, None, lang, text, [])) is None
