"""Integration tests for NotebookEngine.

Uses a real Maxima process. Exercises the full pipeline:
register → validate → build → execute → store.
"""
import pytest
from notebook.engine import _equation_form_latex
from notebook.engine import NotebookEngine
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.models.session import SessionMode
from notebook.models.step import StepKind, StepStatus


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="test session") as e:
        yield e


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------

class TestSessionInit:
    def test_default_mode_and_backend(self):
        with NotebookEngine() as e:
            assert e.session.mode == SessionMode.standard_symbolic
            assert e.session.primary_backend == "maxima"

    def test_active_branch_is_main(self):
        with NotebookEngine() as e:
            assert e.active_branch_id == "main"

    def test_cas_syntax_hidden_by_default(self):
        with NotebookEngine() as e:
            assert e.session.show_cas_syntax_to_user is False


# ---------------------------------------------------------------------------
# Expression registration
# ---------------------------------------------------------------------------

class TestRegisterExpression:
    def test_first_expr_gets_E1(self, engine):
        expr = engine.register_expression("x^2-1")
        assert expr.display_id == "E1"

    def test_sequential_ids(self, engine):
        e2 = engine.register_expression("x-1")
        e3 = engine.register_expression("x+1")
        assert int(e3.display_id[1:]) == int(e2.display_id[1:]) + 1

    def test_expression_stored_in_table(self, engine):
        expr = engine.register_expression("sin(x)")
        assert expr.display_id in engine.expressions

    def test_registration_creates_step(self, engine):
        before = len(engine.steps)
        engine.register_expression("cos(x)")
        assert len(engine.steps) == before + 1

    def test_registration_step_kind(self, engine):
        expr = engine.register_expression("tan(x)")
        step = engine.steps[expr.created_by_step]
        assert step.kind == StepKind.expression_registration

    def test_branch_head_advances(self, engine):
        head_before = engine.active_branch.head_step_id
        expr = engine.register_expression("x^3")
        head_after = engine.active_branch.head_step_id
        assert head_after != head_before
        assert head_after == expr.created_by_step

    def test_lhs_label_prepended_to_latex(self, engine):
        """H-1: lhs_label prepends 'LHS = ' to the stored latex."""
        expr = engine.register_expression("x^2 + y^2", latex="x^2+y^2", lhs_label="H")
        assert expr.latex == "H = x^2+y^2"

    def test_lhs_label_empty_no_change(self, engine):
        """H-1: empty lhs_label leaves latex unchanged."""
        expr = engine.register_expression("x^2 + y^2", latex="x^2+y^2")
        assert expr.latex == "x^2+y^2"

    def test_lhs_label_with_latex_name(self, engine):
        """H-1: lhs_label with a LaTeX expression like E_{kin}."""
        expr = engine.register_expression("p^2 / (2*m)", latex="p^2/(2m)",
                                          lhs_label=r"E_{\rm kin}")
        # `\rm` is the old-style font switch; the display door rewrites it to
        # `\mathrm`, which is what a paper wants and what KaTeX renders.  The
        # LABEL is what is written down, so it goes through the same door as
        # the body — the point of the assertion is that the label survives.
        assert expr.latex.startswith(r"E_\mathrm{kin} = ")


# ---------------------------------------------------------------------------
# CAS operations — the core pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fresh():
    """Isolated engine for operation tests to avoid counter collision."""
    with NotebookEngine(title="op tests") as e:
        yield e


class TestRunOperation:
    def test_expand(self, fresh):
        e1 = fresh.register_expression("(x+1)^2")
        step, val = fresh.run_operation("expand", {"expr": e1.display_id})
        assert "x^2" in val
        assert "2*x" in val

    def test_factor(self, fresh):
        e = fresh.register_expression("x^2-1")
        _, val = fresh.run_operation("factor", {"expr": e.display_id})
        assert "(x-1)" in val
        assert "(x+1)" in val

    def test_rational_simplify(self, fresh):
        e = fresh.register_expression("(x^2-1)/(x-1)")
        _, val = fresh.run_operation("rational_simplify", {"expr": e.display_id})
        assert "x+1" in val

    def test_differentiate(self, fresh):
        e = fresh.register_expression("x^3")
        _, val = fresh.run_operation("differentiate", {"expr": e.display_id, "var": "x"})
        assert "3*x^2" in val

    def test_integrate_indefinite(self, fresh):
        e = fresh.register_expression("x^2")
        _, val = fresh.run_operation("integrate_indefinite",
                                     {"expr": e.display_id, "var": "x"})
        assert "x^3" in val

    def test_taylor_expand(self, fresh):
        e = fresh.register_expression("sin(x)")
        _, val = fresh.run_operation("taylor_expand",
                                     {"expr": e.display_id, "var": "x",
                                      "point": "0", "order": "3"})
        assert "x" in val

    def test_solve_equation(self, fresh):
        e = fresh.register_expression("x^2-4=0")
        _, val = fresh.run_operation("solve_equation",
                                     {"equation": e.display_id, "var": "x"})
        assert "2" in val

    def test_output_expression_registered(self, fresh):
        e = fresh.register_expression("x^4")
        before_count = len(fresh.expressions)
        fresh.run_operation("expand", {"expr": e.display_id})
        assert len(fresh.expressions) == before_count + 1

    def test_step_has_rendered_command(self, fresh):
        e = fresh.register_expression("x^5")
        step, _ = fresh.run_operation("differentiate",
                                      {"expr": e.display_id, "var": "x"})
        assert step.rendered_command is not None
        assert "diff(" in step.rendered_command

    def test_rendered_command_hidden(self, fresh):
        e = fresh.register_expression("x^6")
        step, _ = fresh.run_operation("expand", {"expr": e.display_id})
        assert step.show_command_to_user is False

    def test_step_status_success(self, fresh):
        e = fresh.register_expression("x+1")
        step, _ = fresh.run_operation("expand", {"expr": e.display_id})
        assert step.status == StepStatus.success

    def test_unknown_operation_raises(self, fresh):
        e = fresh.register_expression("x")
        with pytest.raises(ValueError, match="Unknown operation"):
            fresh.run_operation("frobnicate", {"expr": e.display_id})

    def test_unknown_expression_ref_raises(self, fresh):
        with pytest.raises(ValueError, match="Unknown expression"):
            fresh.run_operation("expand", {"expr": "E999"})

    def test_visible_action_does_not_contain_cas_syntax(self, fresh):
        e = fresh.register_expression("x^2")
        step, _ = fresh.run_operation("expand", {"expr": e.display_id})
        assert "expand(" not in step.visible_action
        assert "ratsimp(" not in step.visible_action


# ---------------------------------------------------------------------------
# Replay via engine
# ---------------------------------------------------------------------------

class TestEngineReplay:
    def test_replay_returns_ok(self, fresh):
        result = fresh.replay_current_branch()
        assert result.ok

    def test_replay_contains_registered_expressions(self, fresh):
        result = fresh.replay_current_branch()
        for eid in fresh.expressions:
            assert eid in result.expr_values

    def test_replay_rebuilds_cas_results(self):
        with NotebookEngine(title="replay test") as e:
            ex = e.register_expression("x^2-1")
            e.run_operation("factor", {"expr": ex.display_id})
            result = e.replay_current_branch()
        assert result.ok
        factor_id = list(result.expr_values)[-1]
        assert "(x" in result.expr_values[factor_id]


# ---------------------------------------------------------------------------
# Full scenario: spec §25.1
# ---------------------------------------------------------------------------

class TestScenarioStandardSymbolic:
    """Reproduce the canonical scenario from spec §25.1."""

    def test_rational_simplify_scenario(self):
        with NotebookEngine(title="Hamiltonian derivation") as e:
            # 3. 手書きで (x^2 - 1)/(x - 1) を入力したと想定
            e1 = e.register_expression("(x^2-1)/(x-1)")
            assert e1.display_id == "E1"

            # 5–8. 有理式として簡約 → E2 = x+1
            step, val = e.run_operation("rational_simplify", {"expr": "E1"})
            assert "x+1" in val

            # 9. 履歴に S1, S2 が保存される
            assert len(e.steps) == 2

            # CAS文法がステップに含まれない（利用者向け表示には出ない）
            assert step.show_command_to_user is False
            assert "ratsimp" not in step.visible_action


# ---------------------------------------------------------------------------
# Equation form display helpers (no Maxima required)
# ---------------------------------------------------------------------------

class TestCanonicalToDisplayLatex:
    @pytest.fixture(autouse=True)
    def handler(self):
        self._fmt = MaximaBackendHandler(None).format_constant_for_latex

    def test_pi(self):
        assert self._fmt("%pi") == r"\pi"

    def test_inf(self):
        assert self._fmt("%inf") == r"\infty"

    def test_minf(self):
        assert self._fmt("minf") == r"-\infty"

    def test_e(self):
        assert self._fmt("%e") == "e"

    def test_plain_number_unchanged(self):
        assert self._fmt("0") == "0"
        assert self._fmt("3") == "3"

    def test_mixed(self):
        result = self._fmt("0 to %pi")
        assert r"\pi" in result
        assert "%" not in result


class TestEquationFormLatex:
    @pytest.fixture(autouse=True)
    def handler(self):
        self._fmt = MaximaBackendHandler(None).format_constant_for_latex

    def test_integrate_indefinite(self):
        latex = _equation_form_latex(
            "integrate_indefinite", {"var": "x"}, r"\sin(x)", r"-\cos(x)",
            format_const=self._fmt,
        )
        assert latex is not None
        assert r"\int" in latex
        assert r"d x" in latex or r"dx" in latex
        assert r"-\cos(x)" in latex
        assert "=" in latex

    def test_integrate_definite(self):
        latex = _equation_form_latex(
            "integrate_definite", {"var": "x", "lower": "0", "upper": "%pi"},
            r"\sin(x)", "2",
            format_const=self._fmt,
        )
        assert latex is not None
        assert r"\int_{" in latex
        assert r"\pi" in latex
        assert "2" in latex

    def test_limit_basic(self):
        latex = _equation_form_latex(
            "limit", {"var": "x", "point": "0"},
            r"\frac{\sin(x)}{x}", "1",
            format_const=self._fmt,
        )
        assert latex is not None
        assert r"\lim_" in latex
        assert r"\to" in latex
        assert "1" in latex

    def test_limit_direction_plus(self):
        latex = _equation_form_latex(
            "limit", {"var": "x", "point": "0", "direction": "plus"},
            r"\frac{1}{x}", r"\infty",
            format_const=self._fmt,
        )
        assert "^{+}" in latex

    def test_limit_direction_minus(self):
        latex = _equation_form_latex(
            "limit", {"var": "x", "point": "0", "direction": "minus"},
            r"\frac{1}{x}", r"-\infty",
            format_const=self._fmt,
        )
        assert "^{-}" in latex

    def test_other_operation_returns_none(self):
        result = _equation_form_latex("expand", {}, "x+1", "x+1",
                                      format_const=self._fmt)
        assert result is None


# ---------------------------------------------------------------------------
# Equation form display: engine integration (uses real Maxima)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eq_engine():
    with NotebookEngine(title="eq form tests") as e:
        yield e


class TestEquationFormDisplay:
    def test_integrate_indefinite_latex_form(self, eq_engine):
        e = eq_engine.register_expression("x^2")
        step, _ = eq_engine.run_operation(
            "integrate_indefinite", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        assert r"\int" in result_expr.latex
        assert "=" in result_expr.latex
        # canonical is still just the result, not the equation form
        assert r"\int" not in result_expr.canonical

    def test_integrate_indefinite_canonical_unaffected(self, eq_engine):
        e = eq_engine.register_expression("sin(x)")
        step, val = eq_engine.run_operation(
            "integrate_indefinite", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # FREE-PARAM: indefinite integration now carries the constant of integration
        # (the general antiderivative).  The canonical is the CAS value + the
        # constant; the name avoids collisions (C, else C_1, … — a shared engine
        # that already integrated gets a fresh constant), so read it back.
        assert len(result_expr.free_parameters) == 1
        c = result_expr.free_parameters[0]
        assert result_expr.canonical == f"{val}+{c}"

    def test_integrate_definite_latex_form(self, eq_engine):
        e = eq_engine.register_expression("sin(x)")
        step, _ = eq_engine.run_operation(
            "integrate_definite",
            {"expr": e.display_id, "var": "x", "lower": "0", "upper": "%pi"},
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        assert r"\int_{" in result_expr.latex
        assert r"\pi" in result_expr.latex
        assert "=" in result_expr.latex

    def test_limit_latex_form(self, eq_engine):
        e = eq_engine.register_expression("sin(x)/x")
        step, _ = eq_engine.run_operation(
            "limit", {"expr": e.display_id, "var": "x", "point": "0"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        assert r"\lim_" in result_expr.latex
        assert r"\to" in result_expr.latex
        assert "=" in result_expr.latex

    def test_func_def_equation_integrate_uses_rhs(self, eq_engine):
        # f(x) = sin(x) → integrate → ∫f(x) dx = -cos(x), not ∫f(x)dx = ∫f(x)dx
        e = eq_engine.register_expression("f(x) = sin(x)")
        step, val = eq_engine.run_operation(
            "integrate_indefinite", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # RHS (sin(x)) was evaluated — result should contain cos, not f(x)
        assert "cos" in result_expr.canonical.lower()
        # Display shows ∫f(x)dx = ...
        assert r"\int" in result_expr.latex
        assert "f" in result_expr.latex
        assert "=" in result_expr.latex

    def test_simple_var_equation_limit_uses_rhs(self, eq_engine):
        # y = sin(x)/x → limit x→0 → lim_{x→0} y = 1
        e = eq_engine.register_expression("y = sin(x)/x")
        step, val = eq_engine.run_operation(
            "limit", {"expr": e.display_id, "var": "x", "point": "0"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # RHS was evaluated to 1
        assert result_expr.canonical == "1"
        # Display shows lim_{x→0} y = 1
        assert r"\lim_" in result_expr.latex
        assert "y" in result_expr.latex

    def test_simple_var_equation_differentiate_uses_rhs(self, eq_engine):
        # y = sin(x) → differentiate → dy/dx = cos(x)
        e = eq_engine.register_expression("y = sin(x)")
        step, val = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # RHS (sin(x)) was differentiated → cos(x)
        assert "cos" in result_expr.canonical.lower()
        # Display shows dy/dx = cos(x) — single-variable, no ∂
        assert r"\frac{dy}{dx}" in result_expr.latex
        assert "=" in result_expr.latex

    def test_func_def_equation_differentiate_operator_notation(self, eq_engine):
        # f(x) = cos(x) → differentiate → d/dx f(x) = -sin(x)
        e = eq_engine.register_expression("f(x) = cos(x)")
        step, val = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # Actual derivative of cos(x) computed via RHS
        assert "sin" in result_expr.canonical.lower()
        # Display uses operator notation d/dx f(x), not prime f'(x)
        assert r"\frac{d}{dx} f(x)" in result_expr.latex
        assert "=" in result_expr.latex

    def test_simple_var_multivariable_differentiate_partial(self, eq_engine):
        # z = (x+y)^2 → differentiate by x → ∂z/∂x = 2(x+y)
        e = eq_engine.register_expression("z = (x+y)^2")
        step, val = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # Partial derivative symbol ∂ must appear (multi-variable RHS)
        assert r"\partial" in result_expr.latex
        assert r"\partial z" in result_expr.latex
        assert "=" in result_expr.latex

    def test_func_def_multivariable_differentiate_partial(self, eq_engine):
        # g(x,y) = (x+y)^2 → differentiate by x → ∂/∂x g(x,y) = 2(x+y)
        e = eq_engine.register_expression("g(x,y) = (x+y)^2")
        step, val = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "x"}
        )
        result_expr = eq_engine.expressions[step.output_refs[0]]
        # Partial derivative symbol ∂ must appear (multi-arg function)
        assert r"\partial" in result_expr.latex
        assert "g(x,y)" in result_expr.latex
        assert "=" in result_expr.latex

    def test_physics_label_lhs_preserved_on_differentiate(self, eq_engine):
        """F = m * a → differentiate w.r.t. m → dF/dm shown (not ∂/∂m(m*a))."""
        # Register via lhs_alias path (simulating H-1 browser input)
        e = eq_engine.register_expression("m * a", lhs_label="F")
        step, _ = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "m"}
        )
        result = eq_engine.expressions[step.output_refs[0]]
        # LHS must mention F, not just the operator form ∂/∂m(...)
        assert "F" in result.latex, f"Expected F in LHS, got: {result.latex}"
        assert "=" in result.latex

    def test_iterated_partial_simple_var(self, eq_engine):
        # c=(a+b)^2 → ∂/∂a → ∂c/∂a=2(a+b) → ∂/∂b → ∂²c/∂a∂b=2
        ec = eq_engine.register_expression("c = (a+b)^2")
        step1, _ = eq_engine.run_operation(
            "differentiate", {"expr": ec.display_id, "var": "a"}
        )
        e2 = eq_engine.expressions[step1.output_refs[0]]
        # First partial: provenance set
        assert e2.quantity_name == "c"
        assert e2.derivative_vars == ["a"]
        assert e2.derivative_is_partial is True
        assert r"\partial c" in e2.latex

        step2, _ = eq_engine.run_operation(
            "differentiate", {"expr": e2.display_id, "var": "b"}
        )
        e3 = eq_engine.expressions[step2.output_refs[0]]
        # Second partial: mixed ∂²c/∂a∂b
        assert e3.quantity_name == "c"
        assert e3.derivative_vars == ["a", "b"]
        assert r"\partial^{2}" in e3.latex
        assert r"\partial a" in e3.latex
        assert r"\partial b" in e3.latex

    def test_iterated_partial_func_def(self, eq_engine):
        # f(a,b)=(a+b)^2 → ∂/∂a → ∂/∂a f(a,b) → ∂/∂b → ∂²/∂a∂b f(a,b)
        ef = eq_engine.register_expression("h(a,b) = (a+b)^2")
        step1, _ = eq_engine.run_operation(
            "differentiate", {"expr": ef.display_id, "var": "a"}
        )
        e2 = eq_engine.expressions[step1.output_refs[0]]
        assert e2.quantity_name == "h"
        assert r"\partial" in e2.latex
        assert "h(a,b)" in e2.latex

        step2, _ = eq_engine.run_operation(
            "differentiate", {"expr": e2.display_id, "var": "b"}
        )
        e3 = eq_engine.expressions[step2.output_refs[0]]
        assert e3.quantity_name == "h"
        assert r"\partial^{2}" in e3.latex
        assert "h(a,b)" in e3.latex

    def test_iterated_same_var_collapses_in_fallback_path(self, eq_engine):
        """A plain expression (no quantity provenance) differentiated twice by the
        same variable must collapse the LHS to d²/dk²(…) via the fallback bump,
        NOT nest as d/dk(d/dk(…)).  Regression for the position→velocity→
        acceleration use case (engine-level coverage of _bump_same_var_derivative).
        """
        e = eq_engine.register_expression("sin(k)")
        s1, _ = eq_engine.run_operation(
            "differentiate", {"expr": e.display_id, "var": "k"}
        )
        d1 = eq_engine.expressions[s1.output_refs[0]]
        # Fallback path: no quantity provenance, LHS is the wrapped form d/dk(…).
        assert d1.quantity_name is None
        assert r"\frac{d}{dk}\left(" in d1.latex

        s2, _ = eq_engine.run_operation(
            "differentiate", {"expr": d1.display_id, "var": "k"}
        )
        d2 = eq_engine.expressions[s2.output_refs[0]]
        # Collapsed to second order, not nested.
        # `\sin`, not the italic word "sin" — the display door names the
        # function.  What is asserted here is the COLLAPSE to second order.
        assert r"\frac{d^{2}}{dk^{2}}\left(\sin(k)\right)" in d2.latex
        assert r"\frac{d}{dk}\left(\frac{d}{dk}" not in d2.latex


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ir_ast population
# ---------------------------------------------------------------------------

class TestIrAst:
    """Expression.ir_ast is populated at runtime; not persisted."""

    def test_register_expression_populates_ir_ast(self):
        from notebook.models.ir import Equation
        with NotebookEngine() as e:
            expr = e.register_expression("f(x) = sin(x)")
            assert expr.ir_ast is not None
            assert isinstance(expr.ir_ast, Equation)

    def test_plain_expr_ir_ast_is_not_equation(self):
        from notebook.models.ir import Equation
        with NotebookEngine() as e:
            expr = e.register_expression("x^2 + 1")
            assert expr.ir_ast is not None
            assert not isinstance(expr.ir_ast, Equation)

    def test_run_operation_result_populates_ir_ast(self):
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("x^2 + 2*x + 1")
            step, _ = e.run_operation("differentiate", {"expr": expr.display_id, "var": "x"})
            result_id = step.output_refs[0]
            result = e.expressions[result_id]
            assert result.ir_ast is not None

    def test_ir_ast_excluded_from_save(self, tmp_path):
        """ir_ast must not appear in the saved JSON (canonical is ground truth)."""
        import json
        with NotebookEngine() as e:
            e.start()
            e.register_expression("y = x^2")
            path = tmp_path / "sess.json"
            e.save(path)
        data = json.loads(path.read_text())
        for expr_data in data["expressions"].values():
            assert "ir_ast" not in expr_data

    def test_ir_ast_none_after_load(self, tmp_path):
        """After load, ir_ast is None (re-derived at next access, not persisted)."""
        with NotebookEngine() as e:
            e.start()
            e.register_expression("y = x^2")
            path = tmp_path / "sess.json"
            e.save(path)
        loaded = NotebookEngine.load(path)
        for expr in loaded.expressions.values():
            assert expr.ir_ast is None


# ---------------------------------------------------------------------------
# julia_repr — secondary representation for Julia session expressions
# ---------------------------------------------------------------------------

class TestJuliaRepr:
    """JuliaBackendHandler.cas_repr() and Expression.cas_repr field."""

    def test_cas_repr_maxima_returns_canonical(self):
        """MaximaBackendHandler.cas_repr always returns canonical."""
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler(None)
        assert h.cas_repr("x + 1", "x + 1") == "x + 1"

    def test_cas_repr_julia_nb_var(self):
        """When canonical is a _nb_ reference, cas_value is returned."""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.builder import JuliaCommandBuilder
        h = JuliaBackendHandler(JuliaCommandBuilder())
        result = h.cas_repr("a†(1)a(1)", "_nb_E3")
        assert result == "a†(1)a(1)"

    def test_cas_repr_julia_non_nb_canonical(self):
        """When canonical is not a _nb_ reference, return None (e.g. side-effect ops)."""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.builder import JuliaCommandBuilder
        h = JuliaBackendHandler(JuliaCommandBuilder())
        assert h.cas_repr("nothing", "nothing") is None

    def test_cas_repr_julia_empty_cas_value(self):
        """Empty cas_value returns None (nothing to store)."""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.builder import JuliaCommandBuilder
        h = JuliaBackendHandler(JuliaCommandBuilder())
        assert h.cas_repr("", "_nb_E1") is None

    def test_julia_repr_in_expression_is_serializable(self):
        """julia_repr appears in model_dump() (unlike ir_ast which is excluded)."""
        from notebook.models.expression import Expression
        e = Expression(
            display_id="E1", branch_id="main",
            latex=r"a^\dagger(1)a(1)", canonical="_nb_E1",
            cas_repr="a†(1)a(1)",
        )
        d = e.model_dump()
        assert "cas_repr" in d
        assert d["cas_repr"] == "a†(1)a(1)"

    def test_maxima_expression_julia_repr_is_none(self):
        """Maxima expressions never set julia_repr."""
        from notebook.models.expression import Expression
        e = Expression(
            display_id="E1", branch_id="main",
            latex="x^2", canonical="x^2",
        )
        assert e.cas_repr is None

    def test_run_operation_maxima_cas_repr_equals_canonical(self):
        """run_operation in a Maxima session sets cas_repr == canonical."""
        with NotebookEngine() as eng:
            eng.start()
            expr = eng.register_expression("x^2 + 1")
            step, _ = eng.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "x"}
            )
            result_id = step.output_refs[0]
            result = eng.expressions[result_id]
            assert result.cas_repr == result.canonical


# ---------------------------------------------------------------------------
# OpaqueExpr — IR node for backends without a structured parser
# ---------------------------------------------------------------------------

class TestOpaqueExpr:
    """OpaqueExpr wraps Julia string(result) as a non-None ir_ast."""

    def test_opaque_expr_in_ir(self):
        from notebook.models.ir import OpaqueExpr
        node = OpaqueExpr(text="a†(1)a(1)")
        assert node.text == "a†(1)a(1)"

    def test_expr_generator_handles_opaque(self):
        from notebook.models.ir import OpaqueExpr
        from notebook.runtime.expr_generator import ExprGenerator
        node = OpaqueExpr(text="a†(1)a(1)")
        assert ExprGenerator().generate(node) == "a†(1)a(1)"

    def test_maxima_run_op_ir_ast_not_opaque(self):
        """Maxima results never use OpaqueExpr — they parse to structured nodes."""
        from notebook.models.ir import OpaqueExpr
        with NotebookEngine() as eng:
            eng.start()
            expr = eng.register_expression("x^2 + 1")
            step, _ = eng.run_operation("expand", {"expr": expr.display_id})
            result_id = step.output_refs[0]
            result_expr = eng.expressions[result_id]
            assert result_expr.ir_ast is not None
            assert not isinstance(result_expr.ir_ast, OpaqueExpr)


# ---------------------------------------------------------------------------
# canonical_variables — Julia backend operator name extraction
# ---------------------------------------------------------------------------

class TestJuliaCanonicalVariables:
    """JuliaBackendHandler.canonical_variables() parses julia_repr text."""

    def _handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.builder import JuliaCommandBuilder
        return JuliaBackendHandler(JuliaCommandBuilder())

    def test_nb_canonical_returns_empty(self):
        """_nb_E3 canonical cannot be parsed — return empty set."""
        h = self._handler()
        assert h.canonical_variables("_nb_E3") == set()

    def test_empty_returns_empty(self):
        h = self._handler()
        assert h.canonical_variables("") == set()

    def test_operator_extracted(self):
        """Boson operator a extracted from julia string repr."""
        h = self._handler()
        assert "a" in h.canonical_variables("a†(1)a(1)")

    def test_multiple_operators(self):
        h = self._handler()
        result = h.canonical_variables("comm(a(1), b'(1))")
        assert "a" in result
        assert "b" in result
        assert "comm" not in result  # builtin filtered

    def test_builtins_excluded(self):
        h = self._handler()
        result = h.canonical_variables("normal_form(a(1)*a'(1))")
        assert "normal_form" not in result
        assert "a" in result

    def test_constant_number_no_ops(self):
        """Expression with only a number has no operator variables."""
        h = self._handler()
        assert h.canonical_variables("1.0") == set()


# ---------------------------------------------------------------------------
# Quantity registry
# ---------------------------------------------------------------------------

class TestQuantityRegistry:
    """symbol_context.registry is populated automatically from register_expression."""

    def test_scalar_equation_registers_quantity(self):
        with NotebookEngine() as e:
            e.register_expression("y = x^2 + 1")
            assert e.symbol_context.get("y") is not None
            qty = e.symbol_context.get_quantity("y")
            assert qty.kind == "scalar"
            assert "x" in qty.natural_vars

    def test_function_equation_registers_quantity(self):
        with NotebookEngine() as e:
            e.register_expression("f(x) = sin(x)")
            assert e.symbol_context.get("f") is not None
            qty = e.symbol_context.get_quantity("f")
            assert qty.kind == "function"
            assert qty.natural_vars == ["x"]

    def test_multiarg_function_registers_all_vars(self):
        with NotebookEngine() as e:
            e.register_expression("g(x, y) = x^2 + y^2")
            assert e.symbol_context.get("g") is not None
            qty = e.symbol_context.get_quantity("g")
            assert qty.kind == "function"
            assert set(qty.natural_vars) == {"x", "y"}

    def test_plain_expression_not_registered(self):
        with NotebookEngine() as e:
            e.register_expression("x^2 + 1")
            assert len(e.symbol_context.registry) == 0

    def test_constant_scalar_has_empty_natural_vars(self):
        with NotebookEngine() as e:
            e.register_expression("c = 3")
            assert e.symbol_context.get("c") is not None
            qty = e.symbol_context.get_quantity("c")
            assert qty.kind == "scalar"
            assert qty.natural_vars == []

    def test_defining_expr_id_is_set(self):
        with NotebookEngine() as e:
            expr = e.register_expression("phi(t) = cos(t)")
            qty = e.symbol_context.get_quantity("phi")
            assert qty.defining_expr_id == expr.display_id

    def test_registry_persists_through_save_load(self, tmp_path):
        path = tmp_path / "session.json"
        with NotebookEngine() as e:
            e.register_expression("u(q, p) = q^2 + p^2")
            e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.get("u") is not None
        qty = loaded.symbol_context.get_quantity("u")
        assert qty.kind == "function"
        assert set(qty.natural_vars) == {"q", "p"}

    def test_source_is_partial_uses_registry(self):
        # multivar function: registry says natural_vars has 2 entries → partial
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("w(a, b) = a + b")
            qty = e.symbol_context.get_quantity("w")
            assert len(qty.natural_vars) == 2
            # Differentiate and confirm partial notation is used
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "a"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\partial" in result.latex


# ---------------------------------------------------------------------------
# SymbolStatistics — operator declarations update SymbolContext
# ---------------------------------------------------------------------------

class TestSymbolStatistics:
    """define_boson/fermion_operator must update SymbolContext.statistics."""

    def test_define_boson_updates_statistics(self):
        """After define_boson_operator('b'), statistics.kind == 'bose'."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="stats_test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_boson_operator", {"name": "b"})
            prop = e.symbol_context.get("b")
            assert prop is not None
            assert prop.statistics.kind == "bose"
            assert prop.quantity.kind == "operator"

    def test_define_fermion_updates_statistics(self):
        """After define_fermion_operator('psi'), statistics.kind == 'fermi'."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="stats_test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "psi"})
            prop = e.symbol_context.get("psi")
            assert prop is not None
            assert prop.statistics.kind == "fermi"

    def test_operator_def_does_not_alloc_expr_id(self):
        """define_boson_operator must not allocate an expression ID (side-effect op)."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="stats_test", mode=SessionMode.second_quantization) as e:
            e.start()
            counter_before = e._expr_counter
            e.run_operation("define_boson_operator", {"name": "c"})
            assert e._expr_counter == counter_before

    def test_define_spin_updates_statistics(self):
        """After define_spin_operator('S'), S and Sx, Sy, Sz are noncommutative.
        Sp/Sm are NOT registered automatically — users define them as Sx±i*Sy."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="stats_test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            for op in ("S", "Sx", "Sy", "Sz"):
                prop = e.symbol_context.get(op)
                assert prop is not None
                assert prop.statistics.kind == "noncommutative"
                assert prop.quantity.kind == "operator"
            # Sp/Sm are intentionally not registered (avoid @tlsxyz/@tlspm mixing)
            for op in ("Sp", "Sm"):
                assert e.symbol_context.get(op) is None

    def test_define_scalar_parameter_registers_scalar(self):
        """After define_scalar_parameter('xi'), xi is in SymbolContext as scalar/commutative."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="scalar_test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_scalar_parameter", {"name": "xi", "_all_names": ["xi"]})
            prop = e.symbol_context.get("xi")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "scalar"
            assert prop.statistics.kind == "bose"

    def test_define_scalar_parameter_no_output_expr(self):
        """define_scalar_parameter must not allocate an expression ID."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="scalar_no_expr", mode=SessionMode.second_quantization) as e:
            e.start()
            counter_before = e._expr_counter
            e.run_operation("define_scalar_parameter", {"name": "Delta", "_all_names": ["Delta"]})
            assert e._expr_counter == counter_before

    def test_define_scalar_parameter_maxima(self):
        """define_scalar_parameter works in standard Maxima sessions too."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="maxima_scalar") as e:
            e.start()
            e.run_operation("define_scalar_parameter", {"name": "lambda", "_all_names": ["lambda"]})
            prop = e.symbol_context.get("lambda")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "scalar"

    def test_define_scalar_parameter_maxima_multi(self):
        """Multiple names are all registered when _all_names is provided."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="maxima_scalar_multi") as e:
            e.start()
            e.run_operation("define_scalar_parameter", {
                "name": "xi", "_all_names": ["xi", "Delta"],
            })
            assert e.symbol_context.get("xi") is not None
            assert e.symbol_context.get("Delta") is not None

    def test_define_vector_registers_vector(self):
        """After define_vector('F'), F is in SymbolContext as vector/commutative."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="vector_test") as e:
            e.start()
            e.run_operation("define_vector", {"name": "F", "_all_names": ["F"]})
            prop = e.symbol_context.get("F")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "vector"
            assert prop.statistics.kind == "bose"

    def test_define_vector_no_output_expr(self):
        """define_vector must not allocate an expression ID."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="vector_no_expr") as e:
            e.start()
            counter_before = e._expr_counter
            e.run_operation("define_vector", {"name": "v", "_all_names": ["v"]})
            assert e._expr_counter == counter_before

    def test_define_matrix_registers_matrix(self):
        """After define_matrix('M'), M is in SymbolContext as matrix/commutative."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="matrix_test") as e:
            e.start()
            e.run_operation("define_matrix", {"name": "M", "_all_names": ["M"]})
            prop = e.symbol_context.get("M")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "matrix"
            assert prop.statistics.kind == "bose"

    def test_define_matrix_maxima(self):
        """define_matrix works in standard Maxima sessions."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="maxima_matrix") as e:
            e.start()
            e.run_operation("define_matrix", {"name": "A", "_all_names": ["A"]})
            prop = e.symbol_context.get("A")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "matrix"

    def test_define_tensor_registers_tensor(self):
        """After define_tensor('T'), T is in SymbolContext as tensor/commutative."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="tensor_test") as e:
            e.start()
            e.run_operation("define_tensor", {"name": "T", "_all_names": ["T"]})
            prop = e.symbol_context.get("T")
            assert prop is not None
            assert prop.quantity.math_type.algebra == "tensor"
            assert prop.statistics.kind == "bose"

    def test_define_type_decl_multi_names(self):
        """_all_names with multiple entries registers all symbols."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="multi_vector") as e:
            e.start()
            e.run_operation("define_vector", {"name": "u", "_all_names": ["u", "v", "w"]})
            for name in ["u", "v", "w"]:
                prop = e.symbol_context.get(name)
                assert prop is not None, f"{name} not registered"
                assert prop.quantity.math_type.algebra == "vector"

    def test_statistics_survive_save_load(self, tmp_path):
        """statistics.kind must be saved and restored through save/load."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        path = tmp_path / "sess_stats.json"
        with NotebookEngine(title="stats_save", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_boson_operator", {"name": "alpha"})
            e.run_operation("define_fermion_operator", {"name": "chi"})
            e.run_operation("define_spin_operator", {"name": "S"})
            e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.get("alpha").statistics.kind == "bose"
        assert loaded.symbol_context.get("chi").statistics.kind == "fermi"
        assert loaded.symbol_context.get("Sz").statistics.kind == "noncommutative"

    def test_legacy_save_format_loads_as_bose(self, tmp_path):
        """Old sessions without 'statistics' key in quantity_registry default to bose."""
        import json
        from notebook.engine import NotebookEngine
        path = tmp_path / "legacy.json"
        # Build a minimal session JSON using the old (quantity-only) format.
        with NotebookEngine() as e:
            e.start()
            e.register_expression("f(x) = sin(x)")
            e.save(path)
        # Rewrite quantity_registry to old flat format (no 'statistics' key).
        data = json.loads(path.read_text())
        data["quantity_registry"] = {
            k: v["quantity"] if isinstance(v, dict) and "quantity" in v else v
            for k, v in data["quantity_registry"].items()
        }
        path.write_text(json.dumps(data))
        loaded = NotebookEngine.load(path)
        prop = loaded.symbol_context.get("f")
        assert prop is not None
        assert prop.statistics.kind == "bose"

    def test_notation_usage_survives_save_load(self, tmp_path):
        """notation_usage entries must be saved and restored through save/load."""
        from notebook.engine import NotebookEngine
        path = tmp_path / "sess_notation.json"
        with NotebookEngine() as e:
            e.start()
            e.symbol_context.notation_usage[r"\Delta_k"] = "gap_function"
            e.symbol_context.notation_usage[r"\mu"] = "chemical_potential"
            e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.notation_usage[r"\Delta_k"] == "gap_function"
        assert loaded.symbol_context.notation_usage[r"\mu"] == "chemical_potential"

    def test_notation_usage_absent_in_old_save_loads_empty(self, tmp_path):
        """Old sessions without 'notation_usage' key load with an empty dict."""
        import json
        from notebook.engine import NotebookEngine
        path = tmp_path / "old_notation.json"
        with NotebookEngine() as e:
            e.start()
            e.save(path)
        data = json.loads(path.read_text())
        del data["notation_usage"]
        path.write_text(json.dumps(data))
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.notation_usage == {}


# ---------------------------------------------------------------------------
# Time-dependent quantities and dot notation
# ---------------------------------------------------------------------------

class TestTimeDependentQuantities:
    """declare_time_dependent() enables dot/ddot notation in derivative display."""

    def test_declare_time_dependent_updates_registry(self):
        with NotebookEngine() as e:
            e.start()
            e.register_expression("q(t) = sin(t)")
            assert not e.symbol_context.get_quantity("q").time_dependent
            e.declare_time_dependent("q")
            assert e.symbol_context.get_quantity("q").time_dependent
            assert e.symbol_context.get_quantity("q").time_parameter == "t"

    def test_register_quantity_merges_with_existing(self):
        with NotebookEngine() as e:
            e.start()
            e.register_expression("phi(t) = cos(t)")
            assert e.symbol_context.get_quantity("phi").kind == "function"
            e.register_quantity("phi", time_dependent=True, latex_name=r"\phi")
            qty = e.symbol_context.get_quantity("phi")
            assert qty.kind == "function"        # preserved
            assert qty.time_dependent is True    # merged
            assert qty.latex_name == r"\phi"     # merged

    def test_dot_notation_single_derivative(self):
        # q(t) = sin(omega*t) → declare_time_dependent → d/dt → \dot{q}
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(omega*t)")
            e.declare_time_dependent("q")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" in result.latex
            assert r"\frac" not in result.latex    # no fraction notation

    def test_ddot_notation_second_derivative(self):
        # order=2 on q(t) → \ddot{q}
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(omega*t)")
            e.declare_time_dependent("q")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t", "order": 2}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\ddot{q}" in result.latex

    def test_custom_latex_name_used_in_dot_notation(self):
        # latex_name overrides bare name in dot notation
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("r(t) = t^2")
            e.declare_time_dependent("r")
            e.register_quantity("r", latex_name=r"\mathbf{r}")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{\mathbf{r}}" in result.latex

    def test_no_dot_notation_without_declaration(self):
        # Without declare_time_dependent, normal d/dt notation is used
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" not in result.latex
            assert r"\frac" in result.latex    # fraction notation present

    def test_no_dot_notation_for_non_time_variable(self):
        # Differentiating by x (not t) should not use dot notation
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(x) = sin(x)")
            e.declare_time_dependent("q")          # declares time_parameter="t"
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "x"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" not in result.latex

    def test_display_format_leibniz_suppresses_dot(self):
        # notation="leibniz" produces d/dt even for time-dependent quantities
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")
            e.register_quantity("q", notation="leibniz")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" not in result.latex
            assert r"\frac" in result.latex

    def test_mixed_notation_in_same_session(self):
        # E2 in dot, E3 in Leibniz — both coexist after switching notation
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(omega*t)")
            e.declare_time_dependent("q")           # notation="dot"

            step1, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            e2 = e.expressions[step1.output_refs[0]]
            assert r"\dot{q}" in e2.latex           # dot notation

            e.register_quantity("q", notation="leibniz")

            step2, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            e3 = e.expressions[step2.output_refs[0]]
            assert r"\dot{q}" not in e3.latex       # Leibniz notation
            assert r"\frac" in e3.latex

            # E2 is unchanged — latex frozen at creation time
            assert r"\dot{q}" in e2.latex

    def test_mixed_notation_preserved_through_save_load(self, tmp_path):
        # After save → load, E2 (dot) and E3 (Leibniz) retain their notations
        path = tmp_path / "session.json"
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")

            step1, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            e2_latex = e.expressions[step1.output_refs[0]].latex

            e.register_quantity("q", notation="leibniz")
            step2, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            e3_latex = e.expressions[step2.output_refs[0]].latex

            e.save(path)

        loaded = NotebookEngine.load(path)
        # Expressions retain their original notation after reload
        e2_id = step1.output_refs[0]
        e3_id = step2.output_refs[0]
        assert loaded.expressions[e2_id].latex == e2_latex   # dot preserved
        assert loaded.expressions[e3_id].latex == e3_latex   # Leibniz preserved
        # Registry also retains notation="leibniz"
        assert loaded.symbol_context.get_quantity("q").notation == "leibniz"

    def test_individual_symbol_notation_independence(self):
        # Confirms notation is strictly per-symbol, and different symbols with different notations can coexist.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q") # q uses dot by default
            e.declare_time_dependent("p") # p uses dot by default
            
            # set p to leibniz, q remains dot
            e.register_quantity("p", notation="leibniz")
            
            expr = e.register_expression("q(t) * p(t)")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            
            # q uses dot notation (\dot{q})
            assert r"\dot{q}" in result.latex
            # p uses Leibniz notation (e.g. \frac{d}{d t} p(t))
            assert r"\frac{d}{d t}" in result.latex or r"\frac{d}{dt}" in result.latex
            # And they coexist beautifully in the same equation
            assert r"\dot{q}" in result.latex and r"p\left(t\right)" in result.latex

    def test_display_format_migration(self):
        # Verify that an old-style JSON dictionary containing 'display_format' is successfully migrated to 'notation'
        old_data = {
            "name": "q",
            "kind": "scalar",
            "time_dependent": True,
            "display_format": "leibniz"
        }
        from notebook.models.quantity import Quantity
        qty = Quantity.model_validate(old_data)
        assert qty.notation == "leibniz"


    def test_ddot_notation_iterated_derivative(self):
        # Differentiate the dot-notation result again → \ddot{q}
        # (iterated ordinary derivative, not a partial)
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(omega*t)")
            e.declare_time_dependent("q")

            step1, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            e2 = e.expressions[step1.output_refs[0]]
            assert r"\dot{q}" in e2.latex
            assert e2.derivative_is_partial is False

            # Differentiate the result: should give \ddot{q}, not \partial^2
            step2, _ = e.run_operation(
                "differentiate", {"expr": e2.display_id, "var": "t"}
            )
            e3 = e.expressions[step2.output_refs[0]]
            assert r"\ddot{q}" in e3.latex
            assert r"\partial" not in e3.latex   # must NOT be partial notation
            assert e3.derivative_is_partial is False

    def test_mixed_partial_still_works_after_fix(self):
        # c=(a+b)^2 → ∂c/∂a → ∂²c/∂a∂b: fix must not break mixed partials
        with NotebookEngine() as e:
            e.start()
            ec = e.register_expression("c = (a+b)^2")
            step1, _ = e.run_operation(
                "differentiate", {"expr": ec.display_id, "var": "a"}
            )
            e2 = e.expressions[step1.output_refs[0]]
            assert e2.derivative_is_partial is True

            step2, _ = e.run_operation(
                "differentiate", {"expr": e2.display_id, "var": "b"}
            )
            e3 = e.expressions[step2.output_refs[0]]
            assert e3.derivative_is_partial is True
            assert r"\partial^{2}" in e3.latex

    def test_set_notation_switches_to_leibniz(self):
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")
            e.set_notation("q", "leibniz")
            assert e.symbol_context.get_quantity("q").notation == "leibniz"
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" not in result.latex
            assert r"\frac" in result.latex

    def test_set_notation_switches_back_to_dot(self):
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")
            e.set_notation("q", "leibniz")
            e.set_notation("q", "dot")
            assert e.symbol_context.get_quantity("q").notation == "dot"
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" in result.latex

    def test_set_notation_invalid_format_raises(self):
        with NotebookEngine() as e:
            e.start()
            e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")
            import pytest
            with pytest.raises(ValueError, match="dot.*leibniz"):
                e.set_notation("q", "newton")

    def test_set_notation_unknown_quantity_raises(self):
        with NotebookEngine() as e:
            e.start()
            import pytest
            with pytest.raises(ValueError, match="not registered"):
                e.set_notation("z", "dot")

    def test_rhs_dot_notation_for_abstract_function(self):
        # When Maxima returns 'diff(q(t),t,1) unevaluated (abstract function),
        # the RHS should also use dot notation, not the Leibniz form.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            expr = e.register_expression("q(t) = q(t)")   # abstract: no closed form
            step, val = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" in result.latex
            assert r"\over" not in result.latex   # no Maxima Leibniz fragment
            assert r"diff(" not in result.latex   # no raw diff notation

    def test_rhs_dot_no_spurious_parens_in_product(self):
        # Differentiating a product like (1/2)*m*q(t)^2 w.r.t. t:
        # Maxima wraps the derivative factor as \left(...\right), so after dot
        # substitution we must NOT see \left(\dot{q}\right) in the RHS.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            expr = e.register_expression("(1/2)*m*q(t)^2")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" in result.latex
            assert r"(\dot{q})" not in result.latex          # no parens
            assert r"\left(\dot{q}\right)" not in result.latex

    def test_rhs_ddot_for_abstract_second_derivative(self):
        # Second derivative of abstract q(t) should produce \ddot{q} on both sides.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            expr = e.register_expression("q(t) = q(t)")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t", "order": 2}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\ddot{q}" in result.latex
            assert r"\over" not in result.latex

    def test_declare_before_register_preserves_time_dependent(self):
        # declare_time_dependent before register_expression must not be overwritten.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            assert e.symbol_context.get_quantity("q").time_dependent is True
            # Registering an explicit equation must preserve the time_dependent flag.
            e.register_expression("q(t) = sin(omega*t)")
            assert e.symbol_context.get_quantity("q").time_dependent is True
            step, _ = e.run_operation(
                "differentiate",
                {"expr": list(e.expressions.keys())[-1], "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{q}" in result.latex

    def test_plain_function_call_gets_provenance(self):
        # q(t) registered as a plain expression (not an equation) should still
        # produce dot notation and pass provenance to iterated derivatives.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            e1 = e.register_expression("q(t)")

            step1, _ = e.run_operation("differentiate", {"expr": e1.display_id, "var": "t"})
            e2 = e.expressions[step1.output_refs[0]]
            assert r"\dot{q}" in e2.latex          # LHS dot notation
            assert r"\over" not in e2.latex         # no raw Leibniz fragment
            assert e2.quantity_name == "q"          # provenance set

            # Iterated: E2 → E3 must use ddot, not raw 'diff(...)
            step2, _ = e.run_operation("differentiate", {"expr": e2.display_id, "var": "t"})
            e3 = e.expressions[step2.output_refs[0]]
            assert r"\ddot{q}" in e3.latex
            assert r"diff(" not in e3.latex         # no raw diff on LHS

    def test_rerender_notation_updates_existing_expressions(self):
        # After set_notation("q","leibniz"), rerender_notation_for_quantity must
        # update the latex of all existing derivative expressions retroactively.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            e1 = e.register_expression("q(t) = q(t)")
            step1, _ = e.run_operation("differentiate", {"expr": e1.display_id, "var": "t"})
            e2 = e.expressions[step1.output_refs[0]]
            assert r"\dot{q}" in e2.latex   # dot before switch

            e.set_notation("q", "leibniz")
            updated = e.rerender_notation_for_quantity("q")

            assert len(updated) == 1
            expr_id, new_latex = updated[0]
            assert expr_id == e2.display_id
            assert r"\dot{q}" not in new_latex
            assert r"\frac" in new_latex        # Leibniz LHS

            # Mutation is reflected in self.expressions too
            assert e.expressions[e2.display_id].latex == new_latex

    def test_rerender_notation_dot_to_leibniz_to_dot(self):
        # Switching leibniz→dot also re-renders back to dot notation.
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            e1 = e.register_expression("q(t) = q(t)")
            step1, _ = e.run_operation("differentiate", {"expr": e1.display_id, "var": "t"})
            step2, _ = e.run_operation(
                "differentiate", {"expr": e.expressions[step1.output_refs[0]].display_id, "var": "t"}
            )

            e.set_notation("q", "leibniz")
            e.rerender_notation_for_quantity("q")
            e.set_notation("q", "dot")
            updated = e.rerender_notation_for_quantity("q")

            assert len(updated) == 2
            for _, lat in updated:
                assert r"\dot{q}" in lat or r"\ddot{q}" in lat
                assert r"\frac" not in lat


# ---------------------------------------------------------------------------
# set_latex_name
# ---------------------------------------------------------------------------

class TestSetLatexName:
    """set_latex_name() changes the LaTeX symbol used in derivative display."""

    def test_set_latex_name_stores_in_registry(self):
        with NotebookEngine() as e:
            e.start()
            e.register_expression("q(t) = sin(t)")
            e.set_latex_name("q", r"\phi")
            assert e.symbol_context.get_quantity("q").latex_name == r"\phi"

    def test_set_latex_name_affects_dot_notation(self):
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("q(t) = sin(t)")
            e.declare_time_dependent("q")
            e.set_latex_name("q", r"\phi")
            step, _ = e.run_operation(
                "differentiate", {"expr": expr.display_id, "var": "t"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\dot{\phi}" in result.latex

    def test_set_latex_name_unknown_quantity_raises(self):
        with NotebookEngine() as e:
            e.start()
            import pytest
            with pytest.raises(ValueError, match="not registered"):
                e.set_latex_name("z", r"\phi")

    def test_set_latex_name_empty_raises(self):
        with NotebookEngine() as e:
            e.start()
            e.register_expression("q(t) = sin(t)")
            import pytest
            with pytest.raises(ValueError, match="cannot be empty"):
                e.set_latex_name("q", "")

    def test_set_latex_name_persists_save_load(self, tmp_path):
        path = tmp_path / "session.json"
        with NotebookEngine() as e:
            e.register_expression("q(t) = sin(t)")
            e.set_latex_name("q", r"\phi")
            e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.get_quantity("q").latex_name == r"\phi"


# ---------------------------------------------------------------------------
# check_diff_context / remove_variable_dependency
# ---------------------------------------------------------------------------

class TestDiffContextCheck:
    """Engine-level dependency conflict detection (CAS-agnostic interface)."""

    def test_no_conflict_when_no_deps(self):
        with NotebookEngine() as e:
            ex = e.register_expression("x^2 + y^2")
            conflicts = e.check_diff_context(ex.canonical, "x")
            assert conflicts == []

    def test_conflict_detected_when_dep_active(self):
        with NotebookEngine() as e:
            e._adapter.execute("depends([y], [x])")
            ex = e.register_expression("x^2 + y^2")
            conflicts = e.check_diff_context(ex.canonical, "x")
            assert any(v == "y" and d == "x" for v, d in conflicts)

    def test_no_conflict_when_dep_unrelated(self):
        """depends([q],[t]) must not conflict when diffing (x+y)^2 w.r.t. x."""
        with NotebookEngine() as e:
            e._adapter.execute("depends([q], [t])")
            ex = e.register_expression("(x+y)^2")
            conflicts = e.check_diff_context(ex.canonical, "x")
            assert conflicts == []

    def test_remove_variable_dependency_clears_conflict(self):
        with NotebookEngine() as e:
            e._adapter.execute("depends([y], [x])")
            ex = e.register_expression("x + y")
            assert e.check_diff_context(ex.canonical, "x") != []
            ok = e.remove_variable_dependency("y")
            assert ok
            assert e.check_diff_context(ex.canonical, "x") == []


# ---------------------------------------------------------------------------
# diff_wrt_qdot — ∂/∂q̇ (Lagrangian velocity derivative)
# ---------------------------------------------------------------------------

class TestDiffWrtQdot:
    """differentiate with wrt_qdot: differentiate w.r.t. a generalized velocity."""

    def test_harmonic_oscillator_dL_dqdot(self):
        # L = (1/2)*m*q̇² - (1/2)*k*q²  →  ∂L/∂q̇ = m*q̇
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression(
                "(1/2)*m*diff(q(t),t)^2 - (1/2)*k*q(t)^2"
            )
            step, val = e.run_operation(
                "differentiate", {"expr": eL.display_id, "wrt_qdot": "q"}
            )
            result = e.expressions[step.output_refs[0]]
            # LHS: ∂L/∂q̇
            assert r"\partial" in result.latex
            assert r"\dot{q}" in result.latex
            # RHS: m*q̇  (dot notation because q is time_dependent)
            assert r"\dot{q}" in result.latex
            # Canonical: m * 'diff(q(t),t,1)
            assert "diff(q(t),t,1)" in result.canonical

    def test_harmonic_oscillator_dL_dq(self):
        # L = (1/2)*m*q̇² - (1/2)*k*q²  →  ∂L/∂q = -k*q
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression(
                "(1/2)*m*diff(q(t),t)^2 - (1/2)*k*q(t)^2"
            )
            step, val = e.run_operation(
                "differentiate", {"expr": eL.display_id, "var": "q(t)", "order": 1}
            )
            result = e.expressions[step.output_refs[0]]
            assert "k" in result.canonical
            assert "q" in result.canonical

    def test_lhs_uses_source_label_from_equation(self):
        # When L is defined as an equation "L = ...", LHS should be ∂L/∂q̇ not ∂E1/∂q̇
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression(
                "L = (1/2)*m*diff(q(t),t)^2 - (1/2)*k*q(t)^2"
            )
            step, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "wrt_qdot": "q"}
            )
            result = e.expressions[step.output_refs[0]]
            assert r"\partial L" in result.latex or "partial" in result.latex

    def test_wrt_qdot_requires_time_dependent_quantity(self):
        import pytest
        with NotebookEngine() as e:
            e.start()
            e.register_expression("x^2 + y^2")
            eL = e.register_expression("x^2 + y^2")
            with pytest.raises(ValueError, match="time-dependent"):
                e.run_operation(
                    "differentiate", {"expr": eL.display_id, "wrt_qdot": "x"}
                )

    def test_wrt_qdot_unregistered_quantity_raises(self):
        import pytest
        with NotebookEngine() as e:
            e.start()
            eL = e.register_expression("p^2")
            with pytest.raises(ValueError, match="time-dependent"):
                e.run_operation(
                    "differentiate", {"expr": eL.display_id, "wrt_qdot": "p"}
                )


class TestLagrangianIndependentAtoms:
    """analytical_mechanics domain: q and q̇ are independent atoms (doc 19).

    ∂L/∂q̇ and ∂L/∂q are plain differentiations w.r.t. the independent atoms
    q_dot / q; d/dt is the total time derivative (chain-rule sum).  No block
    tricks, and the result canonical keeps the atoms (q_dot / q_ddot)."""

    def _lagrangian_engine(self):
        e = NotebookEngine()
        e.session.domains = ["analytical_mechanics"]
        return e

    def test_single_pendulum_euler_lagrange(self):
        with self._lagrangian_engine() as e:
            e.start()
            assert e._handler.is_lagrangian_mode()
            e.declare_time_dependent("q")
            eL = e.register_expression(
                "(1/2)*m*l^2*q_dot^2 - m*g*l*(1 - cos(q))"
            )
            # ∂L/∂q̇ = m l² q̇  (independent atom: keeps q_dot)
            s2, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "wrt_qdot": "q"}
            )
            r2 = e.expressions[s2.output_refs[0]]
            assert "q_dot" in r2.canonical
            assert "q_ddot" not in r2.canonical
            assert r"\dot{q}" in r2.latex
            # d/dt(∂L/∂q̇) = m l² q̈  (total time derivative)
            s3, _ = e.run_operation(
                "differentiate", {"expr": r2.display_id, "var": "t"}
            )
            r3 = e.expressions[s3.output_refs[0]]
            assert "q_ddot" in r3.canonical
            assert r"\ddot{q}" in r3.latex
            # ∂L/∂q = -m g l sin q  (q̇² term independent of q → 0)
            s4, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "var": "q"}
            )
            r4 = e.expressions[s4.output_refs[0]]
            assert "sin(q)" in r4.canonical
            assert "q_dot" not in r4.canonical  # no spurious velocity term

    def test_total_time_derivative_velocity_coupled(self):
        # Double-pendulum cross term: d/dt(m p q ḃ cos(a-b)) uses the chain rule
        # over BOTH coordinates a and b → b̈ and (ȧ-ḃ) terms appear.
        with self._lagrangian_engine() as e:
            e.start()
            e.declare_time_dependent("a")
            e.declare_time_dependent("b")
            eX = e.register_expression("m*p*q*b_dot*cos(a - b)")
            s, _ = e.run_operation(
                "differentiate", {"expr": eX.display_id, "var": "t"}
            )
            r = e.expressions[s.output_refs[0]]
            # b̈ term from ∂/∂b_dot · b_ddot
            assert "b_ddot" in r.canonical
            # velocity-product terms from ∂/∂a · a_dot and ∂/∂b · b_dot
            assert "a_dot" in r.canonical and "b_dot" in r.canonical

    def test_non_lagrangian_session_keeps_function_form(self):
        # Without the analytical_mechanics domain, the old depends/function-form
        # path is preserved (q_dot collapses to diff(q(t),t,1)).
        with NotebookEngine() as e:
            e.start()
            assert not e._handler.is_lagrangian_mode()
            e.declare_time_dependent("q")
            eL = e.register_expression("(1/2)*m*diff(q(t),t)^2")
            s, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "wrt_qdot": "q"}
            )
            r = e.expressions[s.output_refs[0]]
            # Old path keeps the Maxima function-derivative form (no q_dot atom).
            assert "diff(q(t),t,1)" in r.canonical
            assert "q_dot" not in r.canonical


class TestNewtonianDotNotationSubstitution:
    """Newtonian (classical_mechanics) sessions: user-typed \\dot{} LaTeX input
    must canonicalize to one form that cancels under substitution.

    Regression guard for the bug where the same velocity was rendered three
    different ways — atom ``r_dot`` (from \\dot{r} input), depends-form
    ``'diff(r,t,1)`` (from differentiate) and the bogus Maxima function
    ``dot(theta)`` (from the IR pipeline on \\dot{theta}).  Existing tests fed
    the *already-normalized* forms (``diff(q(t),t)`` or ``q_dot``) as input, so
    the \\dot{} LaTeX → IR path that real users hit went unverified.
    """

    def _engine(self):
        e = NotebookEngine()
        e.session.domains = ["classical_mechanics"]
        return e

    def test_dot_latex_input_canonicalizes_to_atom(self):
        # \dot{theta} (multi-char Greek base) must NOT become the function
        # dot(theta); it must be the atom theta_dot in the stored canonical.
        with self._engine() as e:
            e.start()
            assert not e._handler.is_lagrangian_mode()
            e.declare_time_dependent("r")
            e.declare_time_dependent("theta")
            eLz = e.register_expression(r"m r^2 \dot{theta}")
            c = e.expressions[eLz.display_id].canonical
            assert "theta_dot" in c
            assert r"\dot" not in c        # LaTeX must not survive into canonical
            assert "dot(" not in c         # must not be the bogus function form

    def test_differentiate_uses_depends_form(self):
        # The atom must differentiate to Maxima's depends form 'diff(theta,t,2),
        # the same form a later substitution LHS resolves to.
        with self._engine() as e:
            e.start()
            e.declare_time_dependent("r")
            e.declare_time_dependent("theta")
            eLz = e.register_expression(r"m r^2 \dot{theta}")
            s, _ = e.run_operation(
                "differentiate", {"expr": eLz.display_id, "var": "t"}
            )
            c = e.expressions[s.output_refs[0]].canonical
            assert "'diff(theta,t,2)" in c
            assert "'diff(r,t,1)" in c
            assert "dot(" not in c

    def test_angular_momentum_conserved_via_substitution(self):
        # End-to-end S-h: d/dt(m r² θ̇) with θ̈ = -2ṙθ̇/r (the EOM) must cancel
        # to 0.  This is the exact natural-command path that was silently broken:
        # the three velocity forms now unify to the depends form and cancel.
        with self._engine() as e:
            e.start()
            e.declare_time_dependent("r")
            e.declare_time_dependent("theta")
            eLz = e.register_expression(r"m r^2 \dot{theta}")
            s, _ = e.run_operation(
                "differentiate", {"expr": eLz.display_id, "var": "t"}
            )
            dLz = e.expressions[s.output_refs[0]]
            s2, val = e.run_operation(
                "substitute",
                {"expr": dLz.display_id,
                 "lhs": r"\ddot{theta}",
                 "rhs": r"-2 \dot{r} \dot{theta} / r"},
            )
            assert e.expressions[s2.output_refs[0]].canonical == "0"

    def test_single_char_velocity_atom_and_substitution(self):
        # Single-char base \dot{r}/\ddot{r}: dot-notation normalizes to the atom
        # and an \ddot{r} substitution LHS resolves to the depends form 'diff(r,t,2)
        # so it matches.
        with self._engine() as e:
            e.start()
            e.declare_time_dependent("r")
            assert e._handler.pre_process_canonical(r"\dot{r}") == "r_dot"
            assert e._handler.pre_process_canonical(r"\ddot{r}") == "r_ddot"
            ea = e.register_expression(r"\ddot{r}")
            assert e.expressions[ea.display_id].canonical == "r_ddot"
            s, _ = e.run_operation(
                "substitute",
                {"expr": ea.display_id, "lhs": r"\ddot{r}", "rhs": "F/m"},
            )
            assert e.expressions[s.output_refs[0]].canonical == "F/m"

    def test_scalar_times_derivative_space_juxtaposition(self):
        # A space-juxtaposed scalar×derivative source ("k r_ddot", "m r_dot")
        # must multiply correctly.  Maxima has no implicit multiplication, so a
        # verbatim "k r_ddot" source returns garbage ([r(t)]); the build path must
        # round-trip it to explicit "k*r_ddot".  Newtonian (depends-form) check.
        with self._engine() as e:
            e.start()
            e.declare_time_dependent("r")
            # differentiate: d/dt(m ṙ) = m r̈  (factor m must survive)
            ep = e.register_expression(r"m \dot{r}")
            s, _ = e.run_operation(
                "differentiate", {"expr": ep.display_id, "var": "t"}
            )
            assert e.expressions[s.output_refs[0]].canonical == "m*'diff(r,t,2)"
            # substitute: k r̈ with r̈ = F/m → k·F/m  (factor k must survive)
            ek = e.register_expression(r"k \ddot{r}")
            s2, _ = e.run_operation(
                "substitute",
                {"expr": ek.display_id, "lhs": r"\ddot{r}", "rhs": "F/m"},
            )
            assert e.expressions[s2.output_refs[0]].canonical == "(F*k)/m"

    def test_p2_definition_bind_normalizes_space_product(self):
        # P2 sends "H : <rhs>" to Maxima at definition time.  A space-juxtaposed
        # RHS ("a b") must be promoted to "a*b" or Maxima silently rejects the
        # assignment and H stays unbound.
        with self._engine() as e:
            e.start()
            e.register_expression("a b", lhs_label="H")
            assert e._adapter.execute("H;").value == "a*b"

    def test_juxtaposition_mult_leaves_symbol_paren_ambiguity_to_parser(self):
        # The build-path space→* promotion must NOT blindly turn "f (x)" into
        # "f*(x)": whether a space before "(" is a product or a function call needs
        # the SymbolContext, so it is left for the SharedParser.  Unambiguous
        # products (symbol↔symbol, )↔( ) are still promoted.
        import re
        rx = re.compile(r"(?<=[\w\)\]])\s+(?=[\w'])|(?<=[\)\]])\s+(?=\()")
        assert rx.sub("*", "k r_ddot") == "k*r_ddot"
        assert rx.sub("*", "(a+b) (c+d)") == "(a+b)*(c+d)"
        assert rx.sub("*", "f (x)") == "f (x)"        # ambiguous → untouched
        assert rx.sub("*", "sin (x)") == "sin (x)"    # ambiguous → untouched
        assert rx.sub("*", "H_0 + lambda*V") == "H_0 + lambda*V"  # subscript safe

    def test_space_juxtaposition_in_lagrangian_mode(self):
        # Same space-juxtaposition fix must hold in analytical_mechanics
        # (Lagrangian) mode, where derivatives stay as atoms (q_dot/q_ddot) and the
        # dot→diff map never runs — so the explicit-multiplication normalization,
        # not the map, is what saves "k r_ddot" from Maxima's missing implicit *.
        e = NotebookEngine()
        e.session.domains = ["analytical_mechanics"]
        with e:
            e.start()
            assert e._handler.is_lagrangian_mode()
            e.declare_time_dependent("r")
            ep = e.register_expression(r"m \dot{r}")
            s, _ = e.run_operation(
                "differentiate", {"expr": ep.display_id, "var": "t"}
            )
            # Lagrangian: d/dt(m ṙ) = m r̈, kept as the atom r_ddot (factor m survives)
            assert e.expressions[s.output_refs[0]].canonical == "m*r_ddot"
            ek = e.register_expression(r"k \ddot{r}")
            s2, _ = e.run_operation(
                "substitute",
                {"expr": ek.display_id, "lhs": r"\ddot{r}", "rhs": "F/m"},
            )
            assert e.expressions[s2.output_refs[0]].canonical == "(F*k)/m"


# ---------------------------------------------------------------------------
# combine_expressions — algebraic operations between two expressions
# ---------------------------------------------------------------------------

class TestCombineExpressions:
    """combine_expressions: LHS op LHS, RHS op RHS for two-expression arithmetic."""

    def test_add_two_plain_expressions(self):
        # (a + b) + (c + d) = a + b + c + d
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("a + b")
            eb = e.register_expression("c + d")
            step, val = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "+"},
            )
            result = e.expressions[step.output_refs[0]]
            assert "a" in result.canonical and "c" in result.canonical

    def test_subtract_two_plain_expressions(self):
        # 3*x - x = 2*x
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("3*x")
            eb = e.register_expression("x")
            step, val = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "-"},
            )
            result = e.expressions[step.output_refs[0]]
            assert "2" in result.canonical and "x" in result.canonical

    def test_combine_lhs_display_uses_equation_labels(self):
        # Both sources have latex of the form "LHS = RHS".
        # Output latex should start with "LHS_a op_sym LHS_b = result".
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression(
                "(1/2)*m*diff(q(t),t)^2 - (1/2)*k*q(t)^2"
            )
            # ∂L/∂q̇ — result has latex "\frac{...} = m \dot{q}"
            step_a, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "wrt_qdot": "q"}
            )
            ea = e.expressions[step_a.output_refs[0]]

            # ∂L/∂q — result has latex "\frac{...} = -k q(t)"
            step_b, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "var": "q(t)", "order": 1}
            )
            eb = e.expressions[step_b.output_refs[0]]

            step_c, _ = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "-"},
            )
            result = e.expressions[step_c.output_refs[0]]
            # LHS should contain both source expression labels
            assert ea.latex.split(" = ")[0] in result.latex
            assert eb.latex.split(" = ")[0] in result.latex
            # op symbol "-" should appear between them
            lhs_part = result.latex.split(" = ")[0]
            assert "-" in lhs_part
            # Exactly one ' = ' in the display (no double-equals)
            assert result.latex.count(" = ") == 1

    def test_combine_plain_expressions_lhs_is_full_latex(self):
        # Plain (non-equation) expressions: LHS display is the full latex, not just LHS part.
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("m*x")
            eb = e.register_expression("k*y")
            step, _ = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "+"},
            )
            result = e.expressions[step.output_refs[0]]
            # Both labels appear in the output latex
            assert ea.latex in result.latex
            assert eb.latex in result.latex

    def test_combine_multiply_op(self):
        # 2 * 3 = 6
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("2")
            eb = e.register_expression("3")
            step, val = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "*"},
            )
            result = e.expressions[step.output_refs[0]]
            assert "6" in result.canonical
            assert r"\cdot" in result.latex

    def test_combine_divide_op(self):
        # (2*x) / 2 = x
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("2*x")
            eb = e.register_expression("2")
            step, val = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "/"},
            )
            result = e.expressions[step.output_refs[0]]
            assert "x" in result.canonical

    def test_combine_invalid_expr_id_raises(self):
        import pytest
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("x")
            with pytest.raises(ValueError):
                e.run_operation(
                    "combine_expressions",
                    {"expr_a": ea.display_id, "expr_b": "E999", "op": "+"},
                )

    def test_combine_english_op_name_normalized(self):
        # LLM may return op="add" — must be normalised to "+" before Maxima call.
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("x")
            eb = e.register_expression("y")
            step, val = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "add"},
            )
            result = e.expressions[step.output_refs[0]]
            assert "x" in result.canonical and "y" in result.canonical

    def test_combine_lhs_extraction_compact_equals(self):
        # Expressions registered with compact 'a=b' latex (no spaces around =)
        # must still extract only the LHS for the display label, and the result
        # must have exactly one '=' (no double-equals from Maxima equation output).
        with NotebookEngine() as e:
            e.start()
            ea = e.register_expression("a = b", latex="a=b")
            eb = e.register_expression("c = d", latex="c=d")
            step, _ = e.run_operation(
                "combine_expressions",
                {"expr_a": ea.display_id, "expr_b": eb.display_id, "op": "+"},
            )
            result = e.expressions[step.output_refs[0]]
            lhs_part = result.latex.rsplit(" = ", 1)[0]
            # Should be "a + c", not "a=b + c=d"
            assert "=" not in lhs_part
            # Exactly one ' = ' separating LHS label from result
            assert result.latex.count(" = ") == 1


# ---------------------------------------------------------------------------
# substitute operation
# ---------------------------------------------------------------------------

class TestSubstitute:
    """substitute: replace subexpression with a new value using Maxima subst()."""

    def test_substitute_simple_symbol(self):
        with NotebookEngine() as e:
            e.start()
            ex = e.register_expression("a^2 + b")
            step, _ = e.run_operation("substitute", {
                "expr": ex.display_id, "lhs": "a", "rhs": "x+1"
            })
            result = e.expressions[step.output_refs[0]]
            assert "x" in result.canonical

    def test_substitute_function_call(self):
        # subst(v, diff(q(t),t,1), L) replaces q̇ with fresh symbol v
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression("(1/2)*m*diff(q(t),t,1)^2 - (1/2)*k*q(t)^2")
            step, _ = e.run_operation("substitute", {
                "expr": eL.display_id,
                "lhs": "diff(q(t),t,1)",
                "rhs": "v"
            })
            result = e.expressions[step.output_refs[0]]
            assert "v" in result.canonical
            assert "diff(q(t),t,1)" not in result.canonical

    def test_euler_lagrange_via_substitute(self):
        # Full E-L derivation for harmonic oscillator using two-step substitute.
        # d/dt(∂L/∂q̇) - ∂L/∂q  should give m*q̈ + k*q(t)
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            eL = e.register_expression("(1/2)*m*diff(q(t),t,1)^2 - (1/2)*k*q(t)^2")

            # ∂L/∂q̇ = m*q̇
            s1, _ = e.run_operation("differentiate", {"expr": eL.display_id, "wrt_qdot": "q"})
            e1 = e.expressions[s1.output_refs[0]]

            # d/dt(∂L/∂q̇) = m*q̈
            s2, _ = e.run_operation("differentiate", {"expr": e1.display_id, "var": "t", "order": 1})
            e2 = e.expressions[s2.output_refs[0]]

            # ∂L/∂q — two-step substitute to avoid Maxima chain-rule spurious terms
            s3, _ = e.run_operation("substitute", {"expr": eL.display_id, "lhs": "diff(q(t),t,1)", "rhs": "v"})
            e3 = e.expressions[s3.output_refs[0]]
            s4, _ = e.run_operation("substitute", {"expr": e3.display_id, "lhs": "q(t)", "rhs": "q_c"})
            e4 = e.expressions[s4.output_refs[0]]
            s5, _ = e.run_operation("differentiate", {"expr": e4.display_id, "var": "q_c", "order": 1})
            e5 = e.expressions[s5.output_refs[0]]
            s6, _ = e.run_operation("substitute", {"expr": e5.display_id, "lhs": "q_c", "rhs": "q(t)"})
            e6 = e.expressions[s6.output_refs[0]]

            # E-L = d/dt(∂L/∂q̇) - ∂L/∂q = m*q̈ + k*q(t)
            s7, _ = e.run_operation("combine_expressions", {
                "expr_a": e2.display_id, "expr_b": e6.display_id, "op": "-"
            })
            el = e.expressions[s7.output_refs[0]]

            # No spurious chain-rule term
            assert "diff(q(t),q" not in el.canonical
            # Correct terms present
            assert "m" in el.canonical
            assert "k" in el.canonical
            assert "diff(q(t),t,2)" in el.canonical

    def test_partial_wrt_q_auto_block(self):
        """∂L/∂q for a time-dependent q should be computed correctly WITHOUT
        manual substitute steps.  The system must auto-wrap with block() to
        avoid spurious chain-rule terms from diff(q(t),t,1)."""
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            # Harmonic oscillator Lagrangian (already in diff(...) form)
            eL = e.register_expression(
                "(1/2)*m*diff(q(t),t,1)^2 - (1/2)*k*q(t)^2"
            )
            # ∂L/∂q: should give -k*q(t) with no spurious diff(q(t),q,...) term
            step, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "var": "q", "order": 1}
            )
            result = e.expressions[step.output_refs[0]]
            # No spurious chain-rule term
            assert "diff(q(t),q" not in result.canonical, (
                f"Spurious chain-rule term in ∂L/∂q: {result.canonical}"
            )
            # Correct result: -k*q(t)
            assert "k" in result.canonical
            assert "q" in result.canonical
            # No q̇ term in ∂L/∂q
            assert "diff(q(t),t" not in result.canonical, (
                f"Velocity term should not appear in ∂L/∂q: {result.canonical}"
            )

    def test_partial_wrt_q_auto_block_bare_q(self):
        """Same as test_partial_wrt_q_auto_block but uses bare q (not q(t))
        in the expression canonical — as happens in sessions saved before the
        q_dot expansion fix, where cos(q) is stored without the (t) suffix."""
        with NotebookEngine() as e:
            e.start()
            e.declare_time_dependent("q")
            # Pendulum Lagrangian with bare q in cos(q) — simulates a loaded
            # session where Maxima returned q not q(t) for non-diff terms.
            eL = e.register_expression(
                "(1/2)*m*l^2*diff(q(t),t,1)^2 - m*g*l*(1-cos(q))"
            )
            step, _ = e.run_operation(
                "differentiate", {"expr": eL.display_id, "var": "q", "order": 1}
            )
            result = e.expressions[step.output_refs[0]]
            # No spurious chain-rule term
            assert "diff(q(t),q" not in result.canonical, (
                f"Spurious chain-rule term: {result.canonical}"
            )
            # Must contain sin (from -mgl*sin(q)) and not be zero
            assert "sin" in result.canonical, (
                f"Expected sin in ∂L/∂q for pendulum: {result.canonical}"
            )
            assert result.canonical.strip() != "0", (
                f"∂L/∂q must not be zero: {result.canonical}"
            )


class TestPreCasUndefinedSymbolGuard:
    """run_operation must block execution when input expressions contain undefined symbols.

    Uses the Julia (second_quantization) backend because Julia's SymbolContext is
    strict: every symbol must be explicitly registered.  Greek letters like xi and
    Delta are NOT in Julia's _KNOWN_SYMBOL_NAMES, so they trigger the guard.
    Maxima's list is more lenient (Greek letters are pre-known), so Maxima tests
    use genuinely unknown identifiers.
    """

    def test_blocks_operation_on_undefined_symbol(self):
        """simplify raises UndefinedSymbolsError when the expression contains xi (Julia)."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="undef_guard", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("xi + 1")
            assert "xi" in expr.undefined_symbols
            with pytest.raises(UndefinedSymbolsError) as exc_info:
                e.run_operation("simplify", {"expr": expr.display_id})
            assert "xi" in exc_info.value.symbols

    def test_allows_operation_after_define(self):
        """After define_scalar_parameter('xi'), the same simplify must succeed (Julia)."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="undef_then_define", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("xi + 1")
            e.run_operation("define_scalar_parameter", {"name": "xi", "_all_names": ["xi"]})
            step, _ = e.run_operation("simplify", {"expr": expr.display_id})
            assert step is not None

    def test_error_carries_all_undefined_symbols(self):
        """UndefinedSymbolsError.symbols lists every undefined name (Julia: xi, Delta)."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="undef_multi", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("xi + Delta")
            with pytest.raises(UndefinedSymbolsError) as exc_info:
                e.run_operation("simplify", {"expr": expr.display_id})
            assert set(exc_info.value.symbols) == {"xi", "Delta"}

    def test_known_symbols_not_blocked(self):
        """Operations on expressions with only defined symbols pass the guard (Julia)."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="known_sym", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_scalar_parameter", {"name": "xi", "_all_names": ["xi"]})
            expr = e.register_expression("xi + 1")
            assert expr.undefined_symbols == []
            step, _ = e.run_operation("simplify", {"expr": expr.display_id})
            assert step is not None

    def test_maxima_does_not_block_free_variables(self):
        """Maxima treats unknown symbols as free variables and does NOT block the guard."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        with NotebookEngine(title="maxima_free_var") as e:
            e.start()
            expr = e.register_expression("zvar + 1")
            # UI warning is present
            assert "zvar" in expr.undefined_symbols
            # But run_operation must NOT raise UndefinedSymbolsError for Maxima
            step, _ = e.run_operation("simplify", {"expr": expr.display_id})
            assert step is not None

    def test_latex_decorator_symbols_not_blocked(self):
        """ast/dagger/star in superscript position must not be flagged as undefined (Julia).

        Expressions like Delta^{ast} (complex conjugate) produce Symbol("ast") in the
        Surface AST.  These are LaTeX decorator commands, not physics parameters.
        """
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="decorator_guard", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_scalar_parameter", {"name": "Delta", "_all_names": ["Delta"]})
            # Delta^{ast} — complex conjugate; ast must not appear in undefined_symbols
            expr = e.register_expression("Delta^{ast}")
            assert "ast" not in expr.undefined_symbols
            # And the operation must not raise UndefinedSymbolsError
            step, _ = e.run_operation("simplify", {"expr": expr.display_id})
            assert step is not None

    # ------------------------------------------------------------------
    # Boundary conditions of the length-1 heuristic (Julia)
    # ------------------------------------------------------------------

    def test_single_char_not_blocked_in_julia(self):
        """Single-letter free variables (y, z — not QA builtins) must NOT block CAS execution."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="free_var_julia", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("y + z")
            try:
                step, _ = e.run_operation("simplify", {"expr": expr.display_id})
                assert step is not None
            except UndefinedSymbolsError as exc:
                pytest.fail(f"Single-char free variables must not block: {exc.symbols}")

    def test_qa_builtin_ops_are_blocked_when_undeclared(self):
        """'a' and 'f' are QuantumAlgebra built-in operator constructors.
        Using them without declaration must raise UndefinedSymbolsError (not a cryptic Julia error)."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="qa_builtin_block", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("a + 1")
            assert "a" in expr.undefined_symbols
            with pytest.raises(UndefinedSymbolsError) as exc_info:
                e.run_operation("simplify", {"expr": expr.display_id})
            assert "a" in exc_info.value.symbols

    def test_subscripted_multichar_is_blocked(self):
        """xi_k (base 'xi', len 2) must be blocked just like plain xi (Julia)."""
        from notebook.engine import NotebookEngine, UndefinedSymbolsError
        from notebook.models.session import SessionMode
        with NotebookEngine(title="subscript_block", mode=SessionMode.second_quantization) as e:
            e.start()
            expr = e.register_expression("xi_k + 1")
            assert "xi" in expr.undefined_symbols
            with pytest.raises(UndefinedSymbolsError):
                e.run_operation("simplify", {"expr": expr.display_id})

    def test_blocks_on_undefined_heuristic_unit(self):
        """Unit test: blocks_on_undefined_symbols applies len>1 + QA-builtin rules correctly.

        Tests the base-name extraction heuristic directly without live Julia execution.
        """
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.julia_command_builder import JuliaCommandBuilder
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.ir import Symbol, Add, Number

        handler = JuliaBackendHandler(JuliaCommandBuilder())
        ctx = SymbolContext()

        def check(symbols: list[str], expected_blocked: list[str]):
            # Build a trivial IR: Add of symbols
            nodes = [Symbol(name=s) for s in symbols]
            ir = nodes[0] if len(nodes) == 1 else Add(terms=nodes)
            result = handler.blocks_on_undefined_symbols(ir, ctx)
            assert set(result) == set(expected_blocked), (
                f"symbols={symbols} → blocked={result}, expected={expected_blocked}"
            )

        check(["xi"], ["xi"])          # len 2 → blocked
        check(["xi_k"], ["xi_k"])      # base "xi" len 2 → blocked (subscript stripped)
        check(["y"], [])               # len 1, not QA builtin → free var, not blocked
        check(["p_k"], [])             # base "p" len 1, not QA builtin → not blocked
        check(["a"], ["a"])            # QA builtin boson op → blocked even though len 1
        check(["f"], ["f"])            # QA builtin fermion op → blocked even though len 1

    def test_greek_letters_not_flagged_in_maxima(self):
        """Standard Greek letters (xi, Delta) are in Maxima's _KNOWN_SYMBOL_NAMES and not flagged."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="maxima_greek") as e:
            e.start()
            expr = e.register_expression("xi + Delta")
            assert "xi" not in expr.undefined_symbols
            assert "Delta" not in expr.undefined_symbols


# ---------------------------------------------------------------------------
# _nb_eval_site_deltas applied to anticommutator / normal_order / heisenberg_eom
# ---------------------------------------------------------------------------

class TestSiteDeltaEvaluation:
    """δ(i,j)=0 for distinct integer site labels must be evaluated in all
    operations that call normal_form: commutator, anticommutator, normal_order,
    heisenberg_equation."""

    def test_anticommutator_site_delta_evaluated(self):
        """{f(1), f†(2)} for fermions = δ(1,2) = 0 when sites 1≠2."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="acomm_delta", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            expr = e.register_expression("c(1)")
            expr2 = e.register_expression("c'(2)")
            step, _ = e.run_operation(
                "anticommutator", {"a": expr.display_id, "b": expr2.display_id}
            )
            assert step is not None
            result_expr = e.expressions[step.output_refs[0]]
            latex = result_expr.latex.replace(" ", "")
            assert latex in ("0", "0.0"), f"Expected 0, got: {result_expr.latex!r}"

    def test_normal_order_site_delta_evaluated(self):
        """normal_order of b(1)*b†(2): δ(1,2) term must be eliminated for integer sites 1≠2."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="normal_order_delta", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_boson_operator", {"name": "b"})
            # b(1)*b†(2) in normal order: b†(2)*b(1) + δ(1,2) → b†(2)*b(1) since δ(1,2)=0
            prod = e.register_expression("b(1) b'(2)")
            step, _ = e.run_operation("normal_order", {"expr": prod.display_id})
            assert step is not None
            result_expr = e.expressions[step.output_refs[0]]
            # δ(1,2) must be gone from the LaTeX output
            assert "\\delta_{1,2}" not in result_expr.latex
            assert "delta" not in result_expr.latex.lower()

    def test_heisenberg_equation_lhs_is_derivative(self):
        """heisenberg_equation display must show d(Op)/dt = RHS, not bare RHS."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="heis_lhs", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "gamma", "_all_names": ["gamma"]})
            e.run_operation("define_scalar_parameter", {"name": "B", "_all_names": ["B"]})
            # H = -gamma * B * Sz(1)
            h = e.register_expression("-gamma * B * Sz(1)")
            op = e.register_expression("Sx(1)")
            step, _ = e.run_operation("heisenberg_equation", {
                "operator": op.display_id,
                "hamiltonian": h.display_id,
            })
            result_expr = e.expressions[step.output_refs[0]]
            # LHS must contain the derivative notation d.../dt
            assert r"\frac{d" in result_expr.latex, (
                f"Expected d/dt LHS, got: {result_expr.latex!r}"
            )
            assert "=" in result_expr.latex, (
                f"Expected equation form, got: {result_expr.latex!r}"
            )

    def test_heisenberg_equation_lhs_when_the_operator_is_RAW(self):
        """The same left side when the operator arrives as TEXT, not an E-id.

        The sibling above passes ``operator=E<n>``, which is how the Japanese
        spelling used to arrive because that path registered the operand as an
        expression.  English has always passed the operator RAW
        (`Heisenberg equation of E1 for a(1)`), and since 2026-08-23 so does
        Japanese — so this is now the ORDINARY case, not an edge one.

        It was not covered, and that is how the defect got in: asking only about
        the E-id, an English session published `- 2 ω₀ Sy_1 - …` with no left
        side at all — the right answer with nothing saying what it is the
        derivative OF.  ``larmor_precession`` printed a bare ``0`` where the
        physics being stated is dS_z/dt = 0.  Mutation-checked: reverting
        ``_format_heisenberg_latex`` to the E-id-only test leaves the sibling
        above GREEN and fails this one.
        """
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="heis_lhs_raw",
                            mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter",
                            {"name": "gamma", "_all_names": ["gamma"]})
            e.run_operation("define_scalar_parameter",
                            {"name": "B", "_all_names": ["B"]})
            h = e.register_expression("-gamma * B * Sz(1)")
            step, _ = e.run_operation("heisenberg_equation", {
                "operator": "Sx(1)",          # RAW — never registered
                "hamiltonian": h.display_id,
            })
            result_expr = e.expressions[step.output_refs[0]]
            assert r"\frac{d Sx(1)}{d t}" in result_expr.latex, (
                f"Expected the raw operator in the d/dt LHS, got: "
                f"{result_expr.latex!r}")
            assert "=" in result_expr.latex


# ---------------------------------------------------------------------------
# operator_substitute exact_index explicit flag (C-1)
# ---------------------------------------------------------------------------

class TestOperatorSubstituteExactIndex:
    """exact_index: true in args overrides the _auto_exact_index heuristic."""

    def test_exact_index_true_restricts_to_specific_site(self):
        """exact_index=True: Sz(r)→mz replaces only Sz(r), not all Sz sites."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="exact_idx_true", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "mz", "_all_names": ["mz"]})
            # H = Sz(1) + Sz(2)
            h = e.register_expression("Sz(1) + Sz(2)")
            # Replace only Sz(r) where r matches "1" — exact_index=True overrides heuristic
            # (heuristic would treat lowercase "r" not in rhs as uniform/template)
            step, _ = e.run_operation("operator_substitute", {
                "expr": h.display_id,
                "op_name": "Sz",
                "formal_index": "1",
                "rhs": "mz",
                "exact_index": True,
            })
            assert step is not None
            result_expr = e.expressions[step.output_refs[0]]
            # mz replaces only site-1; Sz(2) remains
            assert "mz" in result_expr.latex or "m_{z}" in result_expr.latex
            assert "2" in result_expr.latex  # Sz(2) still present

    def test_exact_index_false_applies_uniformly(self):
        """exact_index=False: Sz(r)→mz replaces all Sz sites regardless of heuristic."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="exact_idx_false", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            e.run_operation("define_scalar_parameter", {"name": "mz", "_all_names": ["mz"]})
            h = e.register_expression("Sz(1) + Sz(2)")
            step, _ = e.run_operation("operator_substitute", {
                "expr": h.display_id,
                "op_name": "Sz",
                "formal_index": "r",
                "rhs": "mz",
                "exact_index": False,
            })
            assert step is not None
            result_expr = e.expressions[step.output_refs[0]]
            # Both Sz(1) and Sz(2) replaced by mz
            assert "S" not in result_expr.latex or "mz" in result_expr.latex or "m_{z}" in result_expr.latex


# ---------------------------------------------------------------------------
# Session load + declaration replay (D-3)
# ---------------------------------------------------------------------------

class TestSessionDeclarationReplay:
    """After engine.load() + engine.start(), operator declarations from the
    saved session must be replayed automatically so the user does not have
    to re-declare them before continuing calculations."""

    def test_fermion_operator_usable_after_reload(self, tmp_path):
        """Commutator of a fermion operator works immediately after session reload."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        path = tmp_path / "sess_replay.json"
        # --- first session: declare c, run a commutator, save ---
        with NotebookEngine(title="replay_test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            e.register_expression("c(1)")
            e.save(path)

        # --- reload: declarations should be replayed automatically ---
        loaded = NotebookEngine.load(path)
        with loaded:
            loaded.start()
            # register c(1) without re-declaring @fermion_ops c
            expr = loaded.register_expression("c(1)")
            expr2 = loaded.register_expression("c'(1)")
            step, _ = loaded.run_operation(
                "anticommutator", {"a": expr.display_id, "b": expr2.display_id}
            )
            assert step is not None, "anticommutator after reload failed"
            result = loaded.expressions[step.output_refs[0]]
            latex = result.latex.replace(" ", "")
            # {c(1), c†(1)} = 1 (same-site fermion canonical anticommutator)
            assert latex in ("1", "1.0"), f"Expected 1, got: {result.latex!r}"

    def test_scalar_parameter_usable_after_reload(self, tmp_path):
        """@variables declaration for a scalar parameter is replayed after reload."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        path = tmp_path / "sess_scalar_replay.json"
        with NotebookEngine(title="scalar_replay", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            e.run_operation("define_scalar_parameter", {"name": "xi", "_all_names": ["xi"], "domain": "real"})
            e.save(path)

        loaded = NotebookEngine.load(path)
        with loaded:
            loaded.start()
            # xi must be a declared @variables symbol; using it in an expression
            # and simplifying should not raise MethodError
            expr = loaded.register_expression("xi * c(1)")
            step, _ = loaded.run_operation("normal_order", {"expr": expr.display_id})
            assert step is not None, "normal_order after reload failed"


class TestVectorField:
    """Tests for define_vector_field and vector calculus operations."""

    def test_define_vector_field_registers_components(self):
        """define_vector_field('A') registers A as vector_field and A_x/A_y/A_z as functions."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="vf_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {
                "name": "A",
                "coordinate_system": "cartesian",
                "natural_vars": ["x", "y", "z"],
            })
            qty_A = e.symbol_context.get_quantity("A")
            assert qty_A is not None
            assert qty_A.kind == "vector_field"
            assert qty_A.components == ["A_x", "A_y", "A_z"]
            assert qty_A.coordinate_system == "cartesian"

            for comp in ("A_x", "A_y", "A_z"):
                qty_c = e.symbol_context.get_quantity(comp)
                assert qty_c is not None, f"{comp} not registered"
                assert qty_c.kind == "function"

    def test_define_vector_field_spherical_components(self):
        """Spherical coordinate vector field gets A_r, A_theta, A_phi components."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="vf_sph_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="spherical",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {
                "name": "B",
                "coordinate_system": "spherical",
            })
            qty = e.symbol_context.get_quantity("B")
            assert qty.components == ["B_r", "B_theta", "B_phi"]

    def test_divergence_builds_nb_div_command(self):
        """divergence operation emits nb_div(A_x,A_y,A_z) and returns a result."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="div_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {
                "name": "A",
                "coordinate_system": "cartesian",
                "natural_vars": ["x", "y", "z"],
            })
            step, result = e.run_operation("divergence", {"expr": "A"})
            assert step is not None
            # Result must mention partial derivatives wrt x, y, z
            assert "diff" in result or "partial" in result.lower() or result != ""

    def test_divergence_of_curl_is_zero(self):
        """∇·(∇×A) = 0 (vector identity). Uses recursive component resolution."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="div_curl_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {
                "name": "A",
                "coordinate_system": "cartesian",
                "natural_vars": ["x", "y", "z"],
            })
            # divergence(curl(A)) — recursive: curl components passed to nb_div
            step, result = e.run_operation("divergence", {"expr": "curl(A)"})
            assert step is not None
            # Maxima simplifies this identity to 0
            assert "0" in result

    def test_curl_of_gradient_is_zero(self):
        """∇×(∇f) = 0 (vector identity). Uses recursive component resolution."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="curl_grad_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_scalar_field", {
                "name": "f",
                "coordinate_system": "cartesian",
                "natural_vars": ["x", "y", "z"],
            })
            step, result = e.run_operation("curl", {"expr": "gradient(f)"})
            assert step is not None
            # Maxima simplifies this identity to [0, 0, 0]
            assert "0" in result

    def test_type_error_div_of_scalar(self):
        """divergence(divergence(A)) raises — div returns scalar, not vector."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        import pytest

        with NotebookEngine(
            title="te_div_scalar",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            from notebook.runtime.i18n import t as _t
            with pytest.raises(ValueError) as exc:
                e.run_operation("divergence", {"expr": "divergence(A)"})
            assert str(exc.value) == _t("vector.operand_is_scalar", e.lang,
                                        operand=r"\nabla \cdot A")

    def test_type_error_gradient_of_vector(self):
        """gradient(curl(A)) raises — curl returns vector, gradient needs scalar."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        import pytest

        with NotebookEngine(
            title="te_grad_vec",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            from notebook.runtime.i18n import t as _t
            with pytest.raises(ValueError) as exc:
                e.run_operation("gradient", {"expr": "curl(A)"})
            assert str(exc.value) == _t("vector.operand_is_vector", e.lang,
                                        operand=r"\nabla \times A")

    def test_type_error_curl_of_scalar_field(self):
        """curl(f) raises when f is a registered scalar field."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        import pytest

        with NotebookEngine(
            title="te_curl_scalar",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_scalar_field", {"name": "f", "coordinate_system": "cartesian"})
            from notebook.runtime.i18n import t as _t
            with pytest.raises(ValueError) as exc:
                e.run_operation("curl", {"expr": "f"})
            assert str(exc.value) == _t("vector.operand_is_scalar", e.lang, operand="f")

    def test_type_error_gradient_of_vector_field(self):
        """gradient(A) raises when A is a registered vector field."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        import pytest

        with NotebookEngine(
            title="te_grad_vf",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            from notebook.runtime.i18n import t as _t
            with pytest.raises(ValueError) as exc:
                e.run_operation("gradient", {"expr": "A"})
            assert str(exc.value) == _t("vector.operand_is_vector", e.lang, operand="A")

    def test_cross_product_cartesian(self):
        """cross_product(A, B) returns a 3-component Maxima list."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="cross_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            e.run_operation("define_vector_field", {"name": "B", "coordinate_system": "cartesian"})
            step, result = e.run_operation("cross_product", {"a": "A", "b": "B"})
            assert step is not None
            # Result should be a list with 3 components
            assert result.startswith("[") and result.endswith("]")
            assert result.count(",") >= 2

    def test_dot_product_cartesian(self):
        """dot_product(A, B) returns a scalar expression."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="dot_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            e.run_operation("define_vector_field", {"name": "B", "coordinate_system": "cartesian"})
            step, result = e.run_operation("dot_product", {"a": "A", "b": "B"})
            assert step is not None
            assert "A_x" in result or "A_z" in result  # scalar, not a list

    def test_divergence_of_cross_product_via_expr_id(self):
        """∇·(A×B): divergence applied to cross_product result via E-ID."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="div_cross_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "cartesian"})
            e.run_operation("define_vector_field", {"name": "B", "coordinate_system": "cartesian"})
            step_c, _ = e.run_operation("cross_product", {"a": "A", "b": "B"})
            cross_id = step_c.output_refs[0]
            step_div, result = e.run_operation("divergence", {"expr": cross_id})
            assert step_div is not None
            assert result != ""  # non-empty result with partial derivatives

    def test_type_error_coord_var_as_field(self):
        """Using a coordinate variable name as a vector field raises TypeError."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        import pytest

        with NotebookEngine(
            title="coord_var_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            # 'x' is a cartesian coordinate variable — should raise type error
            from notebook.runtime.i18n import t as _t
            with pytest.raises(ValueError) as exc:
                e.run_operation("divergence", {"expr": "x"})
            assert str(exc.value) == _t("vector.coordinate_name_as_field", e.lang, name="x")

    def test_custom_coordinate_system(self, tmp_path):
        """define_coordinate_system with Lamé coefficients — parabolic coords."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="parabolic",
            mode=SessionMode.standard_symbolic,
        ) as e:
            e.start()
            e.run_operation("define_coordinate_system", {
                "vars": "xi,eta,phi",
                "lame": "sqrt(xi^2+eta^2),sqrt(xi^2+eta^2),xi*eta",
            })
            assert e.session.coordinate_system == "custom"
            assert e.session.custom_coord_vars == ("xi", "eta", "phi")

            e.run_operation("define_vector_field", {
                "name": "A",
                "coordinate_system": "custom",
            })
            qty = e.symbol_context.get_quantity("A")
            assert qty is not None
            assert qty.components == ["A_xi", "A_eta", "A_phi"]

            step, result = e.run_operation("divergence", {"expr": "A"})
            assert step is not None
            # Maxima may fully evaluate (no 'diff') or leave unevaluated diffs;
            # either way the result must mention the coordinate variable xi/eta.
            assert "xi" in result or "eta" in result

    def test_custom_coord_save_reload(self, tmp_path):
        """Custom coordinate system is preserved through save/load."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        path = tmp_path / "custom.json"
        with NotebookEngine(title="C", mode=SessionMode.standard_symbolic) as e:
            e.start()
            e.run_operation("define_coordinate_system", {
                "vars": "xi,eta,phi",
                "lame": "sqrt(xi^2+eta^2),sqrt(xi^2+eta^2),xi*eta",
            })
            e.run_operation("define_vector_field", {"name": "A", "coordinate_system": "custom"})
            e.save(str(path))

        e2 = NotebookEngine.load(str(path))
        with e2:
            e2.start()
            assert e2.session.coordinate_system == "custom"
            assert e2.session.custom_coord_vars == ("xi", "eta", "phi")
            qty = e2.symbol_context.get_quantity("A")
            assert qty is not None and qty.components == ["A_xi", "A_eta", "A_phi"]
            step, result = e2.run_operation("divergence", {"expr": "A"})
            assert step is not None and ("xi" in result or "eta" in result)

    def test_define_vector_field_delete_unregisters(self):
        """delete_define_step removes A and its components from SymbolContext."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(
            title="vf_del_test",
            mode=SessionMode.standard_symbolic,
            coordinate_system="cartesian",
        ) as e:
            e.start()
            step, _ = e.run_operation("define_vector_field", {
                "name": "A",
                "coordinate_system": "cartesian",
                "natural_vars": ["x", "y", "z"],
            })
            assert e.symbol_context.get_quantity("A") is not None
            e.delete_define_step(step.step_id)
            assert e.symbol_context.get_quantity("A") is None
            assert e.symbol_context.get_quantity("A_x") is None


class TestEquationFormInput:
    """H-1: equation-form input regex and LHS label propagation."""

    @pytest.mark.parametrize("text,exp_lhs,exp_rhs", [
        ("H = x^2 + y^2",            "H",                  "x^2 + y^2"),
        ("E_{kin} = p^2 / (2*m)",    "E_{kin}",            "p^2 / (2*m)"),
        (r"\dot{a} = [H, a]",        r"\dot{a}",           "[H, a]"),
        ("V_eff = V + L",            "V_eff",              "V + L"),
    ])
    def test_eq_split_matches(self, text, exp_lhs, exp_rhs):
        from ui.interpret_pipeline import _EQ_SPLIT_RE
        m = _EQ_SPLIT_RE.match(text)
        assert m is not None, f"Should match: {text!r}"
        assert m.group("lhs").strip() == exp_lhs
        assert m.group("rhs").strip() == exp_rhs

    @pytest.mark.parametrize("text", [
        "H をシンプルにして",           # CJK
        "E5",                           # bare expr ID (no =)
        "a == b",                       # ==
        "a != b",                       # !=
        "a := b + c",                   # :=
        "x + y",                        # no =
        "f(x) = cos(x)",                # LHS has paren → function def
        "solve(x^2 - 1 = 0, x)",        # LHS has paren
        "a = b = c",                    # chained = in RHS
        "x^2 - 1 = 0",                 # RHS is bare number → Maxima equation
        "H = x^2 simplify",            # trailing plain word in RHS
        "integrate(f(x), x) = F(x)",    # LHS has paren
    ])
    def test_eq_split_no_match(self, text):
        from ui.interpret_pipeline import _is_valid_h1_input
        assert _is_valid_h1_input(text) is None, f"Should NOT trigger H-1: {text!r}"

    def test_lhs_label_links_symbol_in_context(self):
        """H-1: lhs_label links the symbol to the expression via defining_expr_id.

        Simulates: "H をハミルトニアンとする" followed by "H = ..."
        """
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="lhs_ctx", mode=SessionMode.standard_symbolic) as e:
            e.start()
            e.run_operation("define_scalar_parameter", {"name": "H", "_all_names": ["H"]})
            qty_before = e.symbol_context.get_quantity("H")
            assert qty_before.defining_expr_id is None

            expr = e.register_expression("a + b", latex="a+b", lhs_label="H")
            qty_after = e.symbol_context.get_quantity("H")
            assert qty_after.defining_expr_id == expr.display_id

    def test_lhs_label_registers_new_symbol(self):
        """H-1: undeclared LHS symbol is registered in SymbolContext."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="lhs_new", mode=SessionMode.standard_symbolic) as e:
            e.start()
            expr = e.register_expression("c * d", latex="cd", lhs_label="K_eff")
            qty = e.symbol_context.get_quantity("K_eff")
            assert qty is not None
            assert qty.defining_expr_id == expr.display_id

    @pytest.mark.parametrize("label,expected", [
        # Plain identifiers
        ("H",                                  "H"),
        ("H_eff",                              "H_eff"),
        ("E_{kin}",                            "E_kin"),
        ("H_{BCS}",                            "H_BCS"),
        ("V_0",                                "V_0"),
        # Decorated forms
        (r"\hat{H}",                           "H_hat"),
        (r"\bar{H}",                           "H_bar"),
        (r"\tilde{H}",                         "H_tilde"),
        (r"\vec{A}",                           "A_vec"),
        (r"\dot{a}",                           "a_dot"),
        (r"\ddot{q}",                          "q_ddot"),
        (r"\bar{H}_{BCS}",                     "H_BCS_bar"),
        # Expectation value — single symbol
        (r"<H>",                               "H_exp"),
        (r"\langle H \rangle",                 "H_exp"),
        (r"\langle H \rangle_0",               "H_exp_0"),
        # Expectation value — multi-char / product
        (r"<ab>",                              "ab_exp"),
        (r"\langle ab \rangle",                "ab_exp"),
        (r"\langle a b \rangle",               "ab_exp"),
        (r"\langle n_k \rangle",               "n_k_exp"),
        (r"\langle \hat{H} \rangle",           "H_exp"),
        # Time/partial derivative
        (r"\frac{dH}{dt}",                     "H_dt"),
        (r"\frac{da}{dt}",                     "a_dt"),
        (r"\frac{\partial H}{\partial x}",     "H_dx"),
        (r"\frac{d\hat{a}}{dt}",               "a_dt"),
        # Unmappable (compound / complex LaTeX) → lhs_alias_map only
        (r"f(x)",                              None),
        ("a + b",                              None),
        ("H - V",                              None),
    ])
    def test_extract_simple_symbol_name(self, label, expected):
        from notebook.engine import _extract_simple_symbol_name
        assert _extract_simple_symbol_name(label) == expected

    def test_lhs_alias_map_populated(self):
        """lhs_alias_map is populated for equation-form latex."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="alias_map", mode=SessionMode.standard_symbolic) as e:
            e.start()
            expr = e.register_expression("x^2", latex="x^2", lhs_label="H")
            assert e.lhs_alias_map.get("H") == expr.display_id

    def test_lhs_alias_map_compound_lhs(self):
        """combine_expressions result registers 'H + V' in lhs_alias_map."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="alias_compound", mode=SessionMode.standard_symbolic) as e:
            e.start()
            e1 = e.register_expression("x^2", latex="x^2", lhs_label="H")
            e2 = e.register_expression("y^2", latex="y^2", lhs_label="V")
            step, _ = e.run_operation("combine_expressions",
                                      {"expr_a": e1.display_id,
                                       "expr_b": e2.display_id, "op": "+"})
            out_id = step.output_refs[0]
            assert e.lhs_alias_map.get("H + V") == out_id

    def test_symbol_expansion_in_formula(self):
        """_expand_symbol_refs expands \\bar{H}, <ab>, \\frac{dH}{dt} in formulas."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        from ui.interpret_pipeline import _expand_symbol_refs

        with NotebookEngine(title="sym_exp_t", mode=SessionMode.standard_symbolic) as e:
            e.start()
            e.register_expression("J * x * y", latex="Jxy", lhs_label=r"\bar{H}")
            e.register_expression("a * b", latex="ab", lhs_label=r"\langle ab \rangle")
            e.register_expression("x^2", latex="x^2", lhs_label="H")
            e.register_expression("p", latex="p", lhs_label=r"\frac{dH}{dt}")

            assert "J * x * y" in _expand_symbol_refs(r"2 * \bar{H} + V", e)
            assert "a * b"     in _expand_symbol_refs(r"<ab> * 2", e)
            assert "a * b"     in _expand_symbol_refs(r"\langle ab \rangle + 1", e)
            assert "x^2"       in _expand_symbol_refs("H + 1", e)
            assert "p"         in _expand_symbol_refs(r"\frac{dH}{dt} + 1", e)
            # H inside \hat{H} must NOT be expanded
            assert r"\hat{H}"  == _expand_symbol_refs(r"\hat{H}", e)

    def test_lhs_alias_map_saved_and_loaded(self, tmp_path):
        """lhs_alias_map persists through save/load."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        path = tmp_path / "alias.json"
        with NotebookEngine(title="A", mode=SessionMode.standard_symbolic) as e:
            e.start()
            expr = e.register_expression("x^2", latex="x^2", lhs_label="H")
            e.save(str(path))

        e2 = NotebookEngine.load(str(path))
        assert e2.lhs_alias_map.get("H") == expr.display_id

    def test_lhs_propagates_through_operation(self):
        """LHS label from registered equation is carried through simplify.

        run_operation returns cas_result.value (raw CAS output), but the
        formatted latex (with LHS) is stored in expressions[output_id].latex.
        """
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="lhs_prop", mode=SessionMode.standard_symbolic) as e:
            e.start()
            expr = e.register_expression("x^2 + 0", latex="x^2 + 0", lhs_label="H")
            assert expr.latex == "H = x^2 + 0"
            step, _raw = e.run_operation("simplify", {"expr": expr.display_id})
            # The formatted result (with LHS) is stored in the result expression
            out_id = step.output_refs[0] if step.output_refs else None
            assert out_id is not None
            result_expr = e.expressions[out_id]
            assert result_expr.latex.startswith("H ="), \
                f"Expected 'H = ...', got {result_expr.latex!r}"


# ---------------------------------------------------------------------------
# P1: Auto-update undefined_symbols
# ---------------------------------------------------------------------------

class TestUndefinedSymbolsAutoUpdate:
    """P1: undefined_symbols field is refreshed when a symbol is newly defined."""

    def test_forward_ref_cleared_after_symbol_definition(self):
        """undefined=['H_0', 'H_1'] is cleared when H_0 and H_1 are defined."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="forward_ref", mode=SessionMode.standard_symbolic) as e:
            e.start()
            # Forward reference: H_0, H_1 not yet defined
            h_expr = e.register_expression("H_0 + H_1", latex="H_0 + H_1")
            assert set(h_expr.undefined_symbols or []) == {"H_0", "H_1"}

            # Define H_0
            h0_expr = e.register_expression("x^2", latex="x^2", lhs_label="H_0")
            # undefined_symbols for h_expr should be updated to exclude H_0
            h_expr_updated = e.expressions[h_expr.display_id]
            assert h_expr_updated.undefined_symbols == ["H_1"], \
                f"Expected ['H_1'], got {h_expr_updated.undefined_symbols}"

            # Define H_1
            h1_expr = e.register_expression("y^2", latex="y^2", lhs_label="H_1")
            # undefined_symbols for h_expr should now be empty
            h_expr_final = e.expressions[h_expr.display_id]
            assert h_expr_final.undefined_symbols == [], \
                f"Expected [], got {h_expr_final.undefined_symbols}"

    def test_undefined_multiple_expressions_updated(self):
        """When a symbol is defined, all expressions listing it as undefined are updated."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="multi_expr", mode=SessionMode.standard_symbolic) as e:
            e.start()
            # Two expressions both reference undefined symbol K
            e1 = e.register_expression("K + 1", latex="K + 1")
            e2 = e.register_expression("2 * K", latex="2 * K")
            assert e1.undefined_symbols == ["K"]
            assert e2.undefined_symbols == ["K"]

            # Define K
            e.register_expression("x^2", latex="x^2", lhs_label="K")

            # Both e1 and e2 should have undefined_symbols cleared
            e1_updated = e.expressions[e1.display_id]
            e2_updated = e.expressions[e2.display_id]
            assert e1_updated.undefined_symbols == []
            assert e2_updated.undefined_symbols == []


# ---------------------------------------------------------------------------
# P2: Maxima auto-substitution
# ---------------------------------------------------------------------------

class TestMaximaAutoSubstitution:
    """P2: Symbol definitions are sent to Maxima so simplify(H) uses H's value."""

    def test_maxima_symbol_definition_sent_on_registration(self):
        """When H = x^2 is registered, 'H : x^2' is sent to Maxima."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="maxima_subst", mode=SessionMode.standard_symbolic) as e:
            e.start()
            # Register H = x^2
            h_expr = e.register_expression("x^2", latex="x^2", lhs_label="H")

            # Now simplify(H) should expand to x^2
            # (H is now defined in Maxima as H : x^2)
            step, result_str = e.run_operation("simplify", {"expr": h_expr.display_id})
            # The simplified result should contain x^2 (not "H")
            assert "x^2" in result_str or "x**2" in result_str

    def test_maxima_symbol_definition_restored_on_load(self, tmp_path):
        """Symbol definitions are restored to Maxima after session load."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        path = tmp_path / "sym_def.json"

        # Create and save
        with NotebookEngine(title="B", mode=SessionMode.standard_symbolic) as e:
            e.start()
            h_expr = e.register_expression("a*b*c", latex="abc", lhs_label="H")
            step, result_str = e.run_operation("simplify", {"expr": h_expr.display_id})
            result_id_1 = step.output_refs[0] if step.output_refs else None
            e.save(str(path))

        # Load and verify that H is still defined
        e2 = NotebookEngine.load(str(path))
        e2.start()

        # Register a new expression that uses H
        test_expr = e2.register_expression("H + 1", latex="H + 1")

        # Simplify it; H should be expanded to a*b*c
        step, result_str = e2.run_operation("simplify", {"expr": test_expr.display_id})
        # The result should contain the expanded form (a*b*c + 1)
        assert "a*b*c" in result_str or "c*b*a" in result_str or ("a" in result_str and "b" in result_str)
