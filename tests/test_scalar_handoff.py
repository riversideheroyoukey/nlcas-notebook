r"""SCALAR-HANDOFF: a finished scalar expression crossing to another backend.

Operator algebra is QuantumAlgebra's job; `(k, iω_n)`-space scalar algebra is
Maxima's.  A condensed-matter calculation that has finished with the operators
therefore has to cross a session boundary — and 2026-07-31 that crossing was
measured by hand end to end (docs/34 §6): the S-c Bogoliubov result was carried
into a Maxima session and reached the BCS closed form
Δ = ω_D/sinh(1/N₀V).  These tests pin the mechanised half.

Two things decide whether a handoff is honest, and both are tested here:

* **What may cross.**  Only operator-free expressions.  The target CAS has no
  operator algebra, so an operator that slipped through would arrive as an
  ordinary symbol and evaluate to something plausible and wrong — the failure
  shape this project keeps hunting.
* **What travels with it.**  The declarations INCLUDING their signs.  Measured
  during the same handoff (SIGN-DIALOG-ASSUME): with the type but not the sign,
  Maxima stops mid-integral to ask "Is omega_D positive, negative or zero?",
  which times out and takes the session with it.  A handoff that forgets the
  sign hands over an expression the target cannot evaluate.
"""
import pytest

import ui.notebook_bridge as nb
from notebook.models.quantity import Quantity
from notebook.runtime.i18n import t
from notebook.runtime.scalar_handoff import (
    HandoffRefused, collect_declarations, operator_symbols, seed_operations,
)


class TestWhatMayCross:
    def _engine(self):
        nb.start_engine(title="handoff_src", mode="standard", language="ja",
                        domains=[])
        return nb.get_engine()

    def test_operator_free_expression_crosses(self):
        eng = self._engine()
        try:
            eng.symbol_context.register("Delta", Quantity(
                name="Delta", kind="scalar", domain="real", sign="positive"))
            e = eng.register_expression("Delta^2 + xi^2")
            h = eng.fork_scalar_handoff(e.display_id)
            assert h.source_expression_ids == [e.display_id]
            assert h.target_backend == "maxima"
        finally:
            nb.stop_engine()

    def test_operator_bearing_expression_is_refused(self):
        eng = self._engine()
        try:
            eng.symbol_context.register("c", Quantity(name="c", kind="operator"))
            e = eng.register_expression("c * Delta")
            with pytest.raises(HandoffRefused) as ei:
                eng.fork_scalar_handoff(e.display_id)
            assert ei.value.i18n_key == "handoff.operators_present"
            assert "c" in ei.value.fmt_kwargs["names"]
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_refusal_says_what_to_do(self, lang):
        msg = t("handoff.operators_present", lang, names="c", expr_id="E1")
        assert msg != "handoff.operators_present"
        assert "c" in msg and "E1" in msg
        assert "{names}" not in msg

    def test_refusal_leaves_no_step(self):
        """A refused handoff is not a thing that happened."""
        eng = self._engine()
        try:
            eng.symbol_context.register("c", Quantity(name="c", kind="operator"))
            e = eng.register_expression("c * Delta")
            before = len(eng.steps)
            with pytest.raises(HandoffRefused):
                eng.fork_scalar_handoff(e.display_id)
            assert len(eng.steps) == before
        finally:
            nb.stop_engine()

    def test_crossing_is_recorded_as_a_step(self):
        """Which session a result came from is part of what it means."""
        from notebook.models.step import StepKind
        eng = self._engine()
        try:
            e = eng.register_expression("a + b")
            before = len(eng.steps)
            h = eng.fork_scalar_handoff(e.display_id)
            assert len(eng.steps) == before + 1
            step = list(eng.steps.values())[-1]
            assert step.kind == StepKind.scalar_handoff
            assert step.input_refs == [e.display_id]
            assert step.operation_args["handoff_id"] == h.handoff_id
        finally:
            nb.stop_engine()


class TestWhatTravelsWithIt:
    def test_sign_and_domain_are_carried(self):
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        ctx.register("omega_D", Quantity(name="omega_D", kind="scalar",
                                         domain="real", sign="positive"))
        decls, undeclared = collect_declarations({"omega_D"}, ctx)
        assert len(decls) == 1
        assert (decls[0].domain, decls[0].sign) == ("real", "positive")
        assert undeclared == []

    def test_free_variables_are_reported_not_refused(self):
        """`x`, `t`, `k` are ordinary — both CASes handle them natively."""
        from notebook.models.symbol_context import SymbolContext
        decls, undeclared = collect_declarations({"x", "k"}, SymbolContext())
        assert decls == []
        assert undeclared == ["k", "x"]

    def test_seed_operations_forward_the_sign(self):
        """The sign must reach `assume()`, via the wiring SIGN-DIALOG-ASSUME
        repaired — reused, not restated."""
        from notebook.models.handoff import HandoffDeclaration, ScalarHandoff
        h = ScalarHandoff(handoff_id="H1", declarations=[
            HandoffDeclaration(name="omega_D", domain="real", sign="positive"),
            HandoffDeclaration(name="xi", domain="real"),
        ])
        ops = seed_operations(h)
        assert ops[0] == ("define_scalar_parameter",
                          {"name": "omega_D", "domain": "real",
                           "sign": "positive"})
        assert ops[1] == ("define_scalar_parameter",
                          {"name": "xi", "domain": "real"})

    def test_operator_symbols_sees_statistics_too(self):
        """A particle-name declaration carries statistics rather than
        kind="operator"; both mean "not a c-number"."""
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        ctx.register("c", quantity=Quantity(name="c", kind="anonymous"),
                     statistics_kind="fermi")
        assert operator_symbols({"c"}, ctx) == ["c"]


class TestTheMeasuredPathStillWorks:
    r"""End to end on the case that motivated the feature: the seed operations
    must leave the target session able to evaluate the S-c integral — the exact
    step that timed out when the sign was lost."""

    def test_seeded_session_can_evaluate_the_gap_integral(self):
        from notebook.models.handoff import HandoffDeclaration, ScalarHandoff
        h = ScalarHandoff(handoff_id="H1", declarations=[
            HandoffDeclaration(name="Delta", domain="real", sign="positive"),
            HandoffDeclaration(name="omega_D", domain="real", sign="positive"),
            HandoffDeclaration(name="xi", domain="real"),
        ])
        nb.start_engine(title="handoff_dst", mode="standard", language="ja",
                        domains=[])
        eng = nb.get_engine()
        try:
            for op, args in seed_operations(h):
                eng.run_operation(op, args)
            step, val = eng.run_operation("integrate_definite", {
                "expr": "1/sqrt(xi^2 + Delta^2)", "var": "xi",
                "lower": "0", "upper": "omega_D"})
            assert val.replace(" ", "") == "asinh(omega_D/Delta)"
        finally:
            nb.stop_engine()


class TestTheTargetSessionIsReal:
    r"""`perform_handoff` must produce a session that actually works — not a
    file describing one.  Every declaration is executed in a fresh engine of
    the target backend, so the CAS state matches what the timeline claims."""

    def _handoff(self):
        from notebook.models.handoff import HandoffDeclaration, ScalarHandoff
        return ScalarHandoff(
            handoff_id="H1",
            source_expression_ids=["E18"],
            source_title="S-c",
            source_backend="julia_full",
            source_branch_id="main",
            export_latex=r"1/\sqrt{\xi^2 + \Delta^2}",
            declarations=[
                HandoffDeclaration(name="Delta", domain="real", sign="positive"),
                HandoffDeclaration(name="omega_D", domain="real", sign="positive"),
                HandoffDeclaration(name="xi", domain="real"),
            ])

    def test_target_file_loads_and_carries_the_expression(self, tmp_path):
        import json
        from notebook.runtime.scalar_handoff import perform_handoff
        path = perform_handoff(self._handoff(), tmp_path)
        data = json.loads(open(path).read())
        exprs = list(data["expressions"].values())
        assert any("sqrt" in (e["canonical"] or "") for e in exprs), exprs
        assert data["session"]["primary_backend"] == "maxima"

    def test_provenance_is_written_into_the_target_timeline(self, tmp_path):
        import json
        from notebook.runtime.scalar_handoff import perform_handoff
        path = perform_handoff(self._handoff(), tmp_path)
        data = json.loads(open(path).read())
        notes = [s for s in data["steps"].values()
                 if s.get("kind") == "manual_note"]
        assert notes, "no provenance note"
        blob = " ".join(str(n) for n in notes)
        assert "H1" in blob and "S-c" in blob and "julia_full" in blob
        assert "E18" in blob

    def test_the_carried_expression_is_computable_there(self, tmp_path):
        """The point of the whole feature: the target session can continue the
        calculation — including the integral whose sign question killed the
        session when the declarations were incomplete.

        Loaded the way the app loads (`NotebookEngine.load` is a CLASSMETHOD
        returning a NEW engine, and the CAS state comes back by replaying the
        branch), so this exercises the real path rather than a hand-built one.
        """
        from notebook.engine import NotebookEngine
        from notebook.runtime.scalar_handoff import perform_handoff
        path = perform_handoff(self._handoff(), tmp_path)
        with NotebookEngine.load(path) as eng:
            eng.replay_current_branch()
            _step, val = eng.run_operation("integrate_definite", {
                "expr": "1/sqrt(xi^2 + Delta^2)", "var": "xi",
                "lower": "0", "upper": "omega_D"})
            assert val.replace(" ", "") == "asinh(omega_D/Delta)"


class TestTheWorkingSetTravels:
    r"""Declarations carried are the source session's c-numbers, not only the
    symbols inside the expression that crosses.

    Measured 2026-08-01 through the NL route: the S-c continuation integrates
    the carried integrand up to `omega_D`, a bound declared in the source that
    appears nowhere in the expression.  Carrying only the expression's symbols
    left it undeclared in the target, where an undeclared subscript reads as an
    INDEX — the answer came back `asinh(omega(D)/Delta)` instead of
    `asinh(omega_D/Delta)`, with no error anywhere.

    A declaration is not a value, so carrying the working set costs nothing;
    making the researcher re-declare by hand reintroduces the friction this
    feature exists to remove.  Operators are still excluded — they mean nothing
    in a scalar CAS.
    """

    def _engine(self):
        nb.start_engine(title="carry", mode="standard", language="ja", domains=[])
        return nb.get_engine()

    def test_declared_symbol_outside_the_expression_is_carried(self):
        eng = self._engine()
        try:
            eng.symbol_context.register("Delta", Quantity(
                name="Delta", kind="scalar", domain="real", sign="positive"))
            eng.symbol_context.register("omega_D", Quantity(
                name="omega_D", kind="scalar", domain="real", sign="positive"))
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            h = eng.fork_scalar_handoff(e.display_id)
            carried = {d.name for d in h.declarations}
            assert {"Delta", "omega_D"} <= carried
            assert next(d for d in h.declarations
                        if d.name == "omega_D").sign == "positive"
        finally:
            nb.stop_engine()

    def test_operator_declarations_are_not_carried(self):
        eng = self._engine()
        try:
            eng.symbol_context.register("c", Quantity(name="c", kind="operator"))
            eng.symbol_context.register("Delta", Quantity(
                name="Delta", kind="scalar", domain="real"))
            e = eng.register_expression("Delta^2")
            h = eng.fork_scalar_handoff(e.display_id)
            carried = {d.name for d in h.declarations}
            assert "Delta" in carried
            assert "c" not in carried
        finally:
            nb.stop_engine()

    def test_undeclared_still_reports_only_the_expression_s_own(self):
        """`undeclared` answers "what in THIS expression has no declaration",
        so it must not grow with the carried working set."""
        eng = self._engine()
        try:
            eng.symbol_context.register("Delta", Quantity(
                name="Delta", kind="scalar", domain="real"))
            e = eng.register_expression("Delta + zeta")
            h = eng.fork_scalar_handoff(e.display_id)
            assert h.undeclared == ["zeta"]
        finally:
            nb.stop_engine()


class TestTheCrossingIsVerified:
    r"""The check docs/34 §5 item 4 asked for, and the limit it must admit.

    A handoff crosses backends, so `ratsimp(before - after) = 0` is
    unavailable — two processes, two notations.  What can be compared is the
    VALUE at a point, computed twice from the two canonicals through the
    numeric layer's whitelist-bound lowering.

    This checks the CROSSING (did the export→re-read round trip stay the same
    expression), which is the failure class this project keeps finding:
    docs/32's `\frac{1}{\sqrt{x}}` → the PRODUCT `frac1*sqrt(x)`, and slice 2's
    uncarried declaration turning `omega_D` into the indexed `omega(D)`.  It
    does NOT compare the two CASes against each other.

    When the expression uses something outside the numeric whitelist the check
    cannot run — and then it must say so.  An unverified crossing reported as a
    verified one is exactly what this feature must not become.
    """
    from notebook.runtime.scalar_handoff import verify_crossing as _v

    def test_same_expression_agrees(self):
        from notebook.runtime.scalar_handoff import verify_crossing
        r = verify_crossing("1/sqrt(xi^2+Delta^2)", "(1)/(sqrt(xi^2 + Delta^2))")
        assert r["status"] == "agreed"

    def test_the_docs32_corruption_is_caught(self):
        """`\\frac` failing to convert leaves a PRODUCT with an extra symbol."""
        from notebook.runtime.scalar_handoff import verify_crossing
        r = verify_crossing("1/sqrt(xi^2+Delta^2)", "frac1*sqrt(xi^2+Delta^2)")
        assert r["status"] == "differed"
        assert r["reason"] == "free_symbols"

    def test_a_different_value_is_caught(self):
        from notebook.runtime.scalar_handoff import verify_crossing
        r = verify_crossing("xi^2", "xi^3")
        assert r["status"] == "differed"
        assert r["reason"] == "value"

    def test_unverifiable_says_so(self):
        """`asinh` is outside the numeric whitelist — report it, do not guess."""
        from notebook.runtime.scalar_handoff import verify_crossing
        r = verify_crossing("asinh(x)", "asinh(x)")
        assert r["status"] == "not_checkable"
        assert r.get("reason")

    def test_sample_points_respect_the_declared_sign(self):
        """A positive quantity must not be probed at a negative value: the
        expression may only be real there (`sqrt`, `log`)."""
        from notebook.models.handoff import HandoffDeclaration
        from notebook.runtime.scalar_handoff import sample_bindings
        decls = [HandoffDeclaration(name="omega_D", sign="positive"),
                 HandoffDeclaration(name="q", sign="negative")]
        for i in range(3):
            b = sample_bindings({"omega_D", "q", "x"}, decls, point=i)
            assert b["omega_D"] > 0
            assert b["q"] < 0
            assert b["x"] != 0          # never probe at a singularity

    def test_the_verdict_is_recorded_in_the_target_timeline(self, tmp_path):
        from notebook.models.handoff import HandoffDeclaration, ScalarHandoff
        from notebook.runtime.scalar_handoff import perform_handoff
        import json
        h = ScalarHandoff(
            handoff_id="H1", source_canonical="1/sqrt(xi^2 + Delta^2)",
            export_latex=r"1/\sqrt{\xi^2 + \Delta^2}",
            declarations=[HandoffDeclaration(name="Delta", domain="real",
                                             sign="positive")])
        path = perform_handoff(h, tmp_path)
        assert h.verification.get("status") == "agreed", h.verification
        data = json.loads(open(path).read())
        blob = " ".join(str(s) for s in data["steps"].values())
        assert "handoff verification" in blob and "agreed" in blob


class TestALabelledResultCrosses:
    r"""The S-c re-run (2026-08-03) found the feature turning away the one
    expression it exists to carry.

    A vacuum expectation value arrives wearing a LABEL saying what it is of:

        c(-k,d) c(k,u) = \frac{u v}{(u^2+v^2)^2}

    The operator check read the whole line, found `c`, and refused — but an
    expectation value is a c-number BY DEFINITION; the operators are in the
    label, not the value.  So the check now runs on the value, and the label,
    which a scalar CAS has no meaning for, is dropped from what is registered
    and kept in the provenance note instead.

    This is not a loosening: an operator in the VALUE is still refused, which
    the first test here pins alongside the second.
    """

    def _engine(self):
        nb.start_engine(title="labelled", mode="standard", language="ja",
                        domains=[])
        eng = nb.get_engine()
        eng.symbol_context.register("c", Quantity(name="c", kind="operator"))
        eng.symbol_context.register("u", Quantity(name="u", kind="scalar"))
        eng.symbol_context.register("v", Quantity(name="v", kind="scalar"))
        return eng

    def test_operator_label_over_a_scalar_value_crosses(self):
        eng = self._engine()
        try:
            e = eng.register_expression("(u*v)/((u^2+v^2)^2)")
            # the shape a vacuum_expectation result has: label = value
            e.latex = r"c(-k,d) c(k,u) = \frac{u v}{\left( u^{2} + v^{2} \right)^{2}}"
            # a Julia RESULT names nothing here: the handle is all that is left,
            # so the latex is what the handoff has to read (as measured)
            e.canonical = "_nb_E16"
            e.input_canonical = ""
            h = eng.build_scalar_handoff(e.display_id)
            # the value crosses, the label does not
            assert "c(" not in h.export_latex, h.export_latex
            assert "u v" in h.export_latex or "u*v" in h.export_latex
            # …and the label is not lost: it says what the value is OF
            assert "c(-k,d)" in h.source_label_latex
        finally:
            nb.stop_engine()

    def test_an_operator_in_the_value_is_still_refused(self):
        eng = self._engine()
        try:
            e = eng.register_expression("u")
            e.latex = r"X = u \cdot c(1)"
            e.canonical = "_nb_E9"
            e.input_canonical = ""
            with pytest.raises(HandoffRefused) as ei:
                eng.build_scalar_handoff(e.display_id)
            assert ei.value.i18n_key == "handoff.operators_present"
        finally:
            nb.stop_engine()

    def test_an_h1_definition_keeps_its_name(self):
        """`H = J x y` is the other kind of label — a symbol NAME, worth
        carrying.  Only an OPERATOR label is dropped."""
        eng = self._engine()
        try:
            e = eng.register_expression("J*x*y")
            e.latex = "H = J x y"
            h = eng.build_scalar_handoff(e.display_id)
            assert h.export_latex.startswith("H ="), h.export_latex
            assert h.source_label_latex == ""
        finally:
            nb.stop_engine()

    def test_equals_inside_braces_is_not_a_separator(self):
        from notebook.runtime.scalar_handoff import split_export_form
        assert split_export_form(r"\sum_{k=1}^{N} a_k") == ("", r"\sum_{k=1}^{N} a_k")
        assert split_export_form("a = b") == ("a", "b")
        assert split_export_form("x <= y") == ("", "x <= y")

    def test_an_opaque_source_canonical_is_not_reported_as_a_mismatch(self, tmp_path):
        r"""A Julia result's canonical is the handle `_nb_Ek`.  Comparing a
        handle with a value made the crossing check answer **differed** — a
        false alarm saying a correct result is wrong, which is worse than
        admitting it cannot check.  The source's own reading of the exported
        LaTeX is what the check compares against."""
        from notebook.models.handoff import ScalarHandoff
        from notebook.runtime.scalar_handoff import perform_handoff
        h = ScalarHandoff(handoff_id="H1", source_canonical="_nb_E16",
                          export_latex=r"\frac{u v}{\left( u^{2} + v^{2} \right)^{2}}")
        perform_handoff(h, tmp_path)
        assert h.verification.get("status") != "differed", h.verification

        h2 = ScalarHandoff(handoff_id="H2", source_canonical="_nb_E16",
                           source_value_canonical="(u*v)/((u^2+v^2)^2)",
                           export_latex=r"\frac{u v}{\left( u^{2} + v^{2} \right)^{2}}")
        perform_handoff(h2, tmp_path)
        assert h2.verification.get("status") == "agreed", h2.verification
