"""P-1: classical Poisson bracket {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i).

Canonical pairs are declared once ("p_r を r の共役運動量とする") and reused.
The operation is built from diff() + a sum in the Maxima backend (no special
non-commutative algebra needed), routed rule-based before the LLM.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.models.symbol_context import SymbolContext


class TestCanonicalPairModel:
    def test_add_and_replace(self):
        c = SymbolContext()
        c.add_canonical_pair("r", "p_r")
        c.add_canonical_pair("theta", "p_theta")
        assert c.canonical_pairs == [("r", "p_r"), ("theta", "p_theta")]
        # same coordinate replaces its momentum (idempotent on q)
        c.add_canonical_pair("r", "p_r2")
        assert c.canonical_pairs == [("theta", "p_theta"), ("r", "p_r2")]

    def test_json_roundtrip_keeps_tuples(self):
        c = SymbolContext()
        c.add_canonical_pair("r", "p_r")
        c2 = SymbolContext.model_validate_json(c.model_dump_json())
        assert c2.canonical_pairs == [("r", "p_r")]
        assert all(isinstance(p, tuple) for p in c2.canonical_pairs)


class TestPoissonBracketPipeline:
    def _run(self, eng, sup, text):
        return asyncio.run(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))

    def _session(self):
        nb.start_engine(title="pb", mode="standard", language="ja",
                        domains=["classical_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        return nb, eng, sup

    def test_declare_pair_momentum_form(self):
        nb_, eng, sup = self._session()
        try:
            r = self._run(eng, sup, "p_r を r の共役運動量とする")
            assert r["status"] == "completed"
            assert eng.symbol_context.canonical_pairs == [("r", "p_r")]
        finally:
            nb_.stop_engine()

    def test_declare_pair_canonical_form(self):
        nb_, eng, sup = self._session()
        try:
            r = self._run(eng, sup, "theta と p_theta を正準対とする")
            assert r["status"] == "completed"
            assert eng.symbol_context.canonical_pairs == [("theta", "p_theta")]
        finally:
            nb_.stop_engine()

    def test_fundamental_brackets(self):
        # {q, p} = 1, {q, q'} = 0 for distinct pairs.
        nb_, eng, sup = self._session()
        try:
            self._run(eng, sup, "p_r を r の共役運動量とする")
            self._run(eng, sup, "p_theta を theta の共役運動量とする")
            assert self._run(eng, sup, "r と p_r のポアソン括弧を求めて")["canonical"] == "1"
            assert self._run(eng, sup, "r と p_theta のポアソン括弧を求めて")["canonical"] == "0"
        finally:
            nb_.stop_engine()

    def test_no_pairs_declared_is_helpful_error(self):
        nb_, eng, sup = self._session()
        try:
            r = self._run(eng, sup, "r と p_r のポアソン括弧を求めて")
            assert r["status"] == "failed"
            assert "正準座標対" in r["error"]
        finally:
            nb_.stop_engine()

    # pb_pair_momentum/pb_pair_canonical/pb_bracket externalized to
    # i18n/nl_patterns.json (所見G/(b) migration); no English regression test
    # existed for any of the three before.
    def _session_en(self):
        nb.start_engine(title="pb_en", mode="standard", language="en",
                        domains=["classical_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        return nb, eng, sup

    def test_declare_pair_momentum_form_en(self):
        nb_, eng, sup = self._session_en()
        try:
            r = self._run(eng, sup, "define p_r as the canonical momentum of r")
            assert r["status"] == "completed"
            assert eng.symbol_context.canonical_pairs == [("r", "p_r")]
        finally:
            nb_.stop_engine()

    def test_declare_pair_canonical_form_en(self):
        nb_, eng, sup = self._session_en()
        try:
            r = self._run(eng, sup, "define theta and p_theta as a canonical pair")
            assert r["status"] == "completed"
            assert eng.symbol_context.canonical_pairs == [("theta", "p_theta")]
        finally:
            nb_.stop_engine()

    def test_fundamental_brackets_en(self):
        nb_, eng, sup = self._session_en()
        try:
            self._run(eng, sup, "define p_r as the canonical momentum of r")
            self._run(eng, sup, "define p_theta as the canonical momentum of theta")
            assert self._run(eng, sup, "the Poisson bracket of r and p_r")["canonical"] == "1"
            assert self._run(eng, sup, "the Poisson bracket of r and p_theta")["canonical"] == "0"
        finally:
            nb_.stop_engine()

    def test_central_force_conservation(self):
        # {H, L_z} = 0 for the central-force Hamiltonian (angular momentum conserved).
        nb_, eng, sup = self._session()
        try:
            self._run(eng, sup, "p_r を r の共役運動量とする")
            self._run(eng, sup, "p_theta を theta の共役運動量とする")
            self._run(eng, sup, "H = p_r^2/(2*m) + p_theta^2/(2*m*r^2) + U(r)")
            self._run(eng, sup, "L_z = p_theta")
            r = self._run(eng, sup, "H と L_z のポアソン括弧を求めて")
            assert r["status"] == "completed"
            assert r["canonical"] == "0"
        finally:
            nb_.stop_engine()

    def test_hamilton_eom_via_bracket(self):
        # {H, r} = -∂H/∂p_r = -p_r/m  (a Hamilton equation of motion).
        nb_, eng, sup = self._session()
        try:
            self._run(eng, sup, "p_r を r の共役運動量とする")
            self._run(eng, sup, "H = p_r^2/(2*m) + U(r)")
            r = self._run(eng, sup, "H と r のポアソン括弧を求めて")
            assert r["status"] == "completed"
            assert r["canonical"] == "-(p_r/m)"
        finally:
            nb_.stop_engine()

    def test_bracket_notation_resolves_to_poisson_bracket_on_maxima(self):
        """J-11 (scenario-narrative-notes review, item 10b, 2026-07-08): [X, Y]
        bracket notation previously only resolved to the quantum `commutator`
        (julia_full-only); on a maxima session it fell through to the
        pure-formula fallback and was registered as a literal Maxima list
        `[X, Y]` instead of computing the Poisson bracket. Root cause:
        `operands_provided()` (which tells the pure-formula fallback to defer
        to the rule-based operator parser) only recognized bracket syntax in
        JuliaBackendHandler; moved to the shared BackendHandler base class
        since the check itself is backend-agnostic."""
        nb_, eng, sup = self._session()
        try:
            self._run(eng, sup, "p_r を r の共役運動量とする")
            self._run(eng, sup, "p_theta を theta の共役運動量とする")
            r = self._run(eng, sup, "[r, p_r]")
            assert r["status"] == "completed", r
            assert r["operation"] == "poisson_bracket"
            assert r["canonical"] == "1"
            r2 = self._run(eng, sup, "[r, p_theta]")
            assert r2["canonical"] == "0"
        finally:
            nb_.stop_engine()

    def test_matrix_literal_not_misrouted_to_poisson_bracket(self):
        """Regression guard for the fix above: a bare Maxima matrix literal
        ("[2, 1; 1, 2]", semicolon-separated rows) syntactically matches the
        same [X, Y] shape as a two-operand bracket (";" falls inside the
        bracket's unrestricted second capture group) and was briefly
        misrouted into poisson_bracket (failing with __no_canonical_pairs__)
        while developing the fix above — caught by
        test_matrix_literal.py::test_bare_matrix_eigenvalues and
        test_lotka_volterra_scenario.py::test_fixed_points_and_neutral_center
        (which registers a Jacobian as `[0, -b*c/d; a*d/b, 0]`)."""
        nb_, eng, sup = self._session()
        try:
            r = self._run(eng, sup, "[2, 1; 1, 2]")
            assert r["status"] == "registered", r
            assert eng.expressions["E1"].canonical == "matrix([2, 1], [1, 2])"
        finally:
            nb_.stop_engine()

    def test_declaration_visible_in_tex_export(self):
        """J-11 (item 10a): a canonical-pair declaration previously produced
        zero visible output in the exported TeX narrative (the confirmation
        message was returned in the API response but never recorded as a
        timeline step) — now recorded via record_quantity_update, mirroring
        declare_time_dependent's pattern, so it shows up between the input
        echo and the next step."""
        import asyncio as _asyncio
        nb_, eng, sup = self._session()
        try:
            self._run(eng, sup, "p_r を r の共役運動量とする")
            res = _asyncio.get_event_loop().run_until_complete(nb_.export_latex())
            import pathlib
            tex = pathlib.Path(res["path"]).read_text(encoding="utf-8")
            assert "正準座標対" in tex and "登録しました" in tex
        finally:
            nb_.stop_engine()
