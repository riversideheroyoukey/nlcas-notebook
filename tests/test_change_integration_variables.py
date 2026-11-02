"""Tests for change_integration_variables (F-5): multivariable change of
variables in an integral with the Jacobian determinant auto-inserted.

The operation produces the new integrand  f(transform) * det(J)  which the user
then integrates over the new variables.  Maxima-only.
"""
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.operation_validator import OperationValidator


def _handler() -> MaximaBackendHandler:
    return MaximaBackendHandler(None)


def _julia() -> JuliaBackendHandler:
    return JuliaBackendHandler(None)


class TestBuildCommand:
    def test_polar(self):
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x^2+y^2", "old_vars": "x, y", "new_vars": "r, theta",
             "transform": "r*cos(theta), r*sin(theta)"},
            {"x^2+y^2": "x^2+y^2"},
        )
        assert r.ok
        assert "subst([x = r * cos(theta), y = r * sin(theta)]" in r.command
        assert "determinant(jacobian([r * cos(theta), r * sin(theta)], [r, theta]))" in r.command

    def test_spherical_volume_element(self):
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "1", "old_vars": "x, y, z", "new_vars": "r, theta, phi",
             "transform": "r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)"},
            {"1": "1"},
        )
        assert r.ok
        assert "jacobian(" in r.command
        assert "[r, theta, phi]" in r.command

    def test_mismatched_lengths_rejected(self):
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x", "old_vars": "x, y", "new_vars": "u",
             "transform": "u"},  # 2 old vars but 1 transform
            {"x": "x"},
        )
        assert not r.ok

    def test_top_level_comma_split_ignores_function_args(self):
        # a transform containing a function with comma args must not be split
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x", "old_vars": "x", "new_vars": "u, v",
             "transform": "atan2(v, u)"},
            {"x": "x"},
        )
        assert r.ok
        assert "atan2(v, u)" in r.command

    def test_full_equation_transform_accepted(self):
        """D-6 (open_items.md): a plan step may supply `transform` as full
        equations ("x = r*cos(theta)") rather than bare RHS expressions — this
        must not crash (the dispatch-ordering bug) and must produce the SAME
        RHS-only substitution as the bare-RHS form (the _to_maxima Equation-
        stripping fix), not the literal equation text."""
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x^2+y^2", "old_vars": "x, y", "new_vars": "r, theta",
             "transform": "x = r*cos(theta), y = r*sin(theta)"},
            {"x^2+y^2": "x^2+y^2"},
        )
        assert r.ok, r.error
        assert "subst([x = r * cos(theta), y = r * sin(theta)]" in r.command
        assert "determinant(jacobian([r * cos(theta), r * sin(theta)], [r, theta]))" in r.command

    def test_bracket_wrapped_var_lists_accepted(self):
        """D-6: old_vars/new_vars given as a single bracketed list ("[x, y]",
        the exact form the real Adviser LLM proposed in S-aa) must split into
        two variables, not be treated as one unsplit blob."""
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x^2+y^2", "old_vars": "[x, y]", "new_vars": "[r, theta]",
             "transform": "r*cos(theta), r*sin(theta)"},
            {"x^2+y^2": "x^2+y^2"},
        )
        assert r.ok, r.error
        assert "subst([x = r * cos(theta), y = r * sin(theta)]" in r.command

    def test_paren_wrapped_var_lists_accepted(self):
        """D-6 re-verification (2026-07-17, live S-aa re-run): a FRESH real
        Adviser call proposed old_vars/new_vars as "(x, y)" (parens, not
        brackets) — _strip_outer_wrap must handle all three bracket kinds
        uniformly, not just "[...]"."""
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x^2+y^2", "old_vars": "(x, y)", "new_vars": "(r, theta)",
             "transform": "r*cos(theta), r*sin(theta)"},
            {"x^2+y^2": "x^2+y^2"},
        )
        assert r.ok, r.error
        assert "subst([x = r * cos(theta), y = r * sin(theta)]" in r.command

    def test_two_separately_bracketed_items_not_merged(self):
        """_strip_outer_wrap must only strip a bracket pair that wraps the
        WHOLE string — "[x],[y]" is two already-split items, not one list."""
        r = _handler().build_command(
            "change_integration_variables",
            {"expr": "x", "old_vars": "[x],[y]", "new_vars": "u, v",
             "transform": "u, v"},
            {"x": "x"},
        )
        assert r.ok, r.error
        assert "subst([[x] = u, [y] = v]" in r.command

    def test_julia_polar(self):
        r = _julia().build_command(
            "change_integration_variables",
            {"expr": "x^2+y^2", "old_vars": "x, y", "new_vars": "r, theta",
             "transform": "r*cos(theta), r*sin(theta)"},
            {"x^2+y^2": "x^2+y^2", "__input__x^2+y^2": "x^2+y^2"},
        )
        assert r.ok
        assert "substitute(" in r.command
        assert "x => r * cos(theta)" in r.command
        assert "det(Symbolics.jacobian([r * cos(theta), r * sin(theta)], [r, theta]))" in r.command


class TestValidation:
    def test_required_fields(self):
        v = OperationValidator()
        ok = v.validate("change_integration_variables",
                        {"expr": "x^2+y^2", "old_vars": "x, y",
                         "new_vars": "r, theta", "transform": "r*cos(theta), r*sin(theta)"},
                        "maxima", set())
        assert ok.valid, ok.error

    def test_missing_transform_rejected(self):
        v = OperationValidator()
        bad = v.validate("change_integration_variables",
                         {"expr": "x", "old_vars": "x", "new_vars": "u"},
                         "maxima", set())
        assert not bad.valid
        assert "transform" in bad.missing_fields

    def test_both_backends_supported(self):
        v = OperationValidator()
        for backend in ("maxima", "julia_full"):
            r = v.validate("change_integration_variables",
                           {"expr": "x^2+y^2", "old_vars": "x, y", "new_vars": "r, theta",
                            "transform": "r*cos(theta), r*sin(theta)"},
                           backend, set())
            assert r.valid, f"{backend}: {r.error}"


class TestEngineIntegration:
    """End-to-end through a live Maxima session."""

    def test_polar_jacobian_result(self):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="jacobian test") as e:
            ex = e.register_expression("x^2+y^2")
            step, _ = e.run_operation(
                "change_integration_variables",
                {"expr": ex.display_id, "old_vars": "x, y",
                 "new_vars": "r, theta", "transform": "r*cos(theta), r*sin(theta)"},
            )
            out = e.expressions[step.output_refs[0]]
            # ∫(x²+y²) dx dy → r·(r²) = r³ after simplification; the result must
            # at least contain the r-dependence from the Jacobian.
            canon = out.canonical.replace(" ", "")
            assert "r" in canon
            # not still in terms of x/y
            assert "x" not in canon and "y" not in canon


class TestPlanStepPathRegression:
    """D-6 (open_items.md): reproduces the exact args the real Adviser LLM
    proposed in the S-aa evaluation (docs/evaluations.md) — bracketed var
    lists and a full-equation transform — end-to-end through
    engine.run_operation, the same call path a plan step uses (as opposed to
    the S-o NL interceptor, which pre-normalizes these args itself before
    ever reaching the engine).  Before the fix this raised UnknownNotationError
    ("chained relation") at the generic per-arg parser in build_command."""

    def test_gaussian_integral_polar_transform_from_llm_plan(self):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="d6 s-aa repro") as e:
            # Register through the LaTeX door, as every product path does
            # (ui/interpret_pipeline converts before calling this).  Handing
            # raw LaTeX in as the CANONICAL was test scaffolding, and it hid
            # two things at once: the command sent to Maxima was unparseable,
            # and the adapter reported that as success (2026-08-20).
            _tex = r"\int_{-\infty}^{\infty} e^{-x^2}\,dx"
            ex = e.register_expression(e._handler.latex_to_canonical(_tex),
                                       latex=_tex)
            step, _ = e.run_operation("change_integration_variables", {
                "expr": ex.display_id,
                "old_vars": "[x, y]",
                "new_vars": "[r, theta]",
                "transform": "x = r*cos(theta), y = r*sin(theta)",
            })
            assert step.status.value == "success"
            # The RHS-only substitution actually sent to Maxima is what matters
            # here — the "x = r*cos(theta), ..." transform must have been split
            # correctly and stripped to its RHS, not the (opaque, by-design)
            # canonical of an integral-form object's result.
            cmd = step.rendered_command.replace(" ", "")
            assert "subst([x=r*cos(theta),y=r*sin(theta)]" in cmd
            assert "jacobian([r*cos(theta),r*sin(theta)],[r,theta])" in cmd


class TestScenarioSO:
    """S-o end-to-end: change_integration_variables → simplify yields the
    textbook volume elements.  The operation returns the RAW Jacobian product;
    a simplify step is required.

    Guards docs/scenario_coordinate_jacobian.md AND the keyword unification
    (SIMP-UNIFY): the user keyword `簡約`/simplify (and `三角簡約`/trig_simplify)
    must apply sin²+cos²=1 on BOTH backends — no CAS-specific keyword.
      - Maxima: simplify = trigsimp(ratsimp(...)) → polar r³, spherical r²sin(theta)
      - Julia:  simplify (Symbolics) → polar r³ clean; spherical only partly
                reduced (sin²φ+cos²φ left — a CAS capability gap, not a keyword one).
    """

    def _change(self, e, expr, old, new, transform):
        ex = e.register_expression(expr)
        step, _ = e.run_operation("change_integration_variables", {
            "expr": ex.display_id, "old_vars": old,
            "new_vars": new, "transform": transform})
        return step.output_refs[0]

    def test_maxima_simplify_keyword_applies_trig(self):
        """Maxima `簡約`(simplify) now applies sin²+cos²=1 (was ratsimp-only)."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="s_o maxima") as e:
            cid = self._change(e, "x^2+y^2", "x, y", "r, theta",
                               "r*cos(theta), r*sin(theta)")
            _, polar = e.run_operation("simplify", {"expr": cid})
            assert polar.replace(" ", "") == "r^3"

            cid2 = self._change(
                e, "1", "x, y, z", "r, theta, phi",
                "r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)")
            _, sph = e.run_operation("simplify", {"expr": cid2})
            assert sph.replace(" ", "") == "r^2*sin(theta)"

    def test_julia_keywords_unified(self):
        """On Julia, both `簡約`(simplify) and `三角簡約`(trig_simplify) work and
        give the same clean r³ for the 2-D polar case (keyword parity)."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="s_o julia", primary_backend="julia_full") as e:
            cid = self._change(e, "x^2+y^2", "x, y", "r, theta",
                               "r*cos(theta), r*sin(theta)")
            _, via_simplify = e.run_operation("simplify", {"expr": cid})
            assert via_simplify.replace(" ", "") == "r^3"
            # trig_simplify is no longer Maxima-only — it must run on Julia too.
            cid2 = self._change(e, "x^2+y^2", "x, y", "r, theta",
                                "r*cos(theta), r*sin(theta)")
            _, via_trig = e.run_operation("trig_simplify", {"expr": cid2})
            assert via_trig.replace(" ", "") == "r^3"
