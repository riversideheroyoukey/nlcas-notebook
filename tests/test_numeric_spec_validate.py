"""Gatekeeper (validate_numeric_spec) + intent-kind registry (docs/24 §7, §3.1).
Pure — no engine / bridge / conversation."""
import pytest

from notebook.models.numeric import (
    InputVar, IntentSpec, MethodSpec, NumericSpec, OutputSpec,
)
from notebook.numeric.spec_projection import flat_to_spec
from notebook.numeric.intent_registry import (
    codegen_route, default_method_for, get_intent_kind, is_known_kind, known_kinds,
)
from notebook.numeric.spec_validate import validate_numeric_spec
from notebook.runtime.i18n import t


def _codes(issues) -> set[str]:
    return {i["code"] for i in issues}


def _sweep_spec(**over):
    base = dict(
        intent=IntentSpec(kind="parameter_sweep"),
        inputs=[InputVar(name="x", role="sweep",
                         spec={"min": 0, "max": 10, "points": 100}),
                InputVar(name="k", role="fixed", spec={"value": "2"})],
        outputs=[OutputSpec(name="y", shape="series", over=["x"])],
    )
    base.update(over)
    return NumericSpec(**base)


class TestIntentRegistry:
    def test_known_and_unknown_kinds(self):
        assert is_known_kind("parameter_sweep")
        assert not is_known_kind("monte_carlo_integration")
        assert "eigenvalue_problem" in known_kinds()

    def test_codegen_route(self):
        # only the whitelist-covered kinds are deterministic
        assert codegen_route("parameter_sweep") == "deterministic"
        assert codegen_route("root_finding") == "llm"
        # unknown/novel kind routes to the LLM path
        assert codegen_route("monte_carlo_integration") == "llm"

    def test_default_method(self):
        assert default_method_for("eigenvalue_problem") == "numpy_eig"
        assert get_intent_kind("root_finding").default_method == "fixed_point"
        assert default_method_for("nope") == ""


class TestValidateHappyPath:
    def test_clean_sweep_spec_has_no_issues(self):
        assert validate_numeric_spec(_sweep_spec()) == []

    def test_projected_slice1_spec_is_valid(self):
        from notebook.models.numeric import NumericTask
        task = NumericTask(task_id="N1", target_quantity="y",
                           fixed_values={"k": "2", "a": "0.3"},
                           sweep_parameters={"x": {"min": 0, "max": 5, "points": 20}})
        spec = flat_to_spec(task)
        assert validate_numeric_spec(spec, free_symbols={"x", "k", "a"}) == []

    def test_convergence_trace_over_iteration_is_allowed(self):
        spec = NumericSpec(
            inputs=[InputVar(name="x", role="free", spec={"initial": 0.5})],
            outputs=[OutputSpec(name="x_star", shape="scalar"),
                     OutputSpec(name="trace", shape="trace", over=["iteration"])],
            method=MethodSpec(name="fixed_point",
                              params={"tol": 1e-8, "max_iter": 1000}))
        assert validate_numeric_spec(spec) == []


class TestValidateCatchesProblems:
    def test_symbol_membership_when_free_symbols_given(self):
        issues = validate_numeric_spec(_sweep_spec(), free_symbols={"x"})  # k missing
        assert "numeric.spec.unknown_symbol" in _codes(issues)

    def test_symbol_check_skipped_without_free_symbols(self):
        assert "numeric.spec.unknown_symbol" not in _codes(
            validate_numeric_spec(_sweep_spec()))

    def test_bad_sweep_range(self):
        spec = _sweep_spec(inputs=[
            InputVar(name="x", role="sweep", spec={"min": 5, "max": 1, "points": 10})],
            outputs=[OutputSpec(name="y", over=["x"])])
        assert "numeric.spec.bad_sweep_range" in _codes(validate_numeric_spec(spec))

    def test_too_few_points(self):
        spec = _sweep_spec(inputs=[
            InputVar(name="x", role="sweep", spec={"min": 0, "max": 1, "points": 1})],
            outputs=[OutputSpec(name="y", over=["x"])])
        assert "numeric.spec.too_few_points" in _codes(validate_numeric_spec(spec))

    def test_bad_fixed_value(self):
        spec = _sweep_spec(inputs=[
            InputVar(name="x", role="sweep", spec={"min": 0, "max": 1, "points": 5}),
            InputVar(name="k", role="fixed", spec={"value": "two"})],
            outputs=[OutputSpec(name="y", over=["x"])])
        assert "numeric.spec.bad_fixed_value" in _codes(validate_numeric_spec(spec))

    def test_dangling_over(self):
        spec = _sweep_spec(outputs=[OutputSpec(name="y", over=["q"])])  # q not an input
        assert "numeric.spec.dangling_over" in _codes(validate_numeric_spec(spec))

    def test_no_outputs(self):
        spec = _sweep_spec(outputs=[])
        assert "numeric.spec.no_outputs" in _codes(validate_numeric_spec(spec))

    def test_bad_method_param(self):
        spec = _sweep_spec(method=MethodSpec(name="fixed_point",
                                             params={"tol": -1, "max_iter": 0}))
        codes = _codes(validate_numeric_spec(spec))
        assert "numeric.spec.bad_method_param" in codes


class TestValidationMessagesAreBilingual:
    def test_every_issue_code_has_ja_and_en(self):
        codes = ["numeric.spec.unknown_symbol", "numeric.spec.bad_sweep_range",
                 "numeric.spec.too_few_points", "numeric.spec.bad_fixed_value",
                 "numeric.spec.bad_free_initial", "numeric.spec.no_outputs",
                 "numeric.spec.dangling_over", "numeric.spec.bad_method_param"]
        for code in codes:
            for lang in ("ja", "en"):
                msg = t(code, lang, name="x", param="tol", output="y", ref="q")
                assert msg and msg != code   # resolved, not the bare key
