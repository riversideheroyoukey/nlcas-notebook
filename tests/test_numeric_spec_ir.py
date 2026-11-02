"""Faceted numeric-spec IR (docs/24): model, flat<->faceted projections,
serialization/backward-compat, and the four worked examples expressible with
ZERO schema change.  Model-level only — no engine / bridge / conversation.
"""
import pytest

from notebook.models.numeric import (
    IntentSpec, InputVar, MethodSpec, NumericSpec, NumericTask,
    OutputSpec, PresentationSpec,
)
from notebook.numeric.spec_projection import (
    flat_to_spec, is_flat_representable, spec_to_flat,
)


def _sweep_task(**over) -> NumericTask:
    base = dict(
        task_id="N1",
        problem_type="parameter_sweep",
        target_quantity="y",
        fixed_values={"k": "2", "a": "0.3"},
        sweep_parameters={"x": {"min": 0, "max": 10, "points": 100}},
    )
    base.update(over)
    return NumericTask(**base)


class TestModelDefaults:
    def test_spec_absent_by_default(self):
        # additive: a task without a faceted spec is the legacy shape
        assert _sweep_task().spec is None

    def test_facets_have_sensible_defaults(self):
        s = NumericSpec()
        assert s.intent.kind == "" and s.method.name == "direct_eval"
        assert s.inputs == [] and s.outputs == []
        assert s.presentation.kind == "line"

    def test_input_role_is_closed(self):
        # closed set (docs/24 §8-2): an unknown role is rejected
        with pytest.raises(Exception):
            InputVar(name="x", role="wiggle")


class TestFlatToSpec:
    def test_parameter_sweep_projects_to_facets(self):
        spec = flat_to_spec(_sweep_task())
        assert spec.intent.kind == "parameter_sweep"
        # swept var first, then fixed sorted by name
        assert [(v.name, v.role) for v in spec.inputs] == [
            ("x", "sweep"), ("a", "fixed"), ("k", "fixed")]
        assert spec.inputs[0].spec == {"min": 0, "max": 10, "points": 100}
        assert spec.inputs[1].spec == {"value": "0.3"}
        # single sweep -> series over [x], line plot
        assert spec.outputs == [
            OutputSpec(name="y", shape="series", over=["x"], role="primary")]
        assert spec.presentation.kind == "line"

    def test_two_sweeps_give_a_field(self):
        task = _sweep_task(
            sweep_parameters={"x": {"min": 0, "max": 1, "points": 10},
                              "y": {"min": 0, "max": 1, "points": 10}},
            target_quantity="z")
        spec = flat_to_spec(task)
        assert spec.outputs[0].shape == "field"
        assert spec.outputs[0].over == ["x", "y"]
        assert spec.presentation.kind == "heatmap"


class TestRoundTrip:
    def test_flat_to_spec_to_flat_is_identity_for_sweep(self):
        task = _sweep_task()
        flat = spec_to_flat(flat_to_spec(task))
        assert flat["problem_type"] == "parameter_sweep"
        assert flat["target_quantity"] == "y"
        assert flat["fixed_values"] == {"k": "2", "a": "0.3"}
        assert flat["sweep_parameters"] == {"x": {"min": 0, "max": 10, "points": 100}}


class TestFlatRepresentable:
    def test_single_sweep_direct_eval_is_flat(self):
        assert is_flat_representable(flat_to_spec(_sweep_task())) is True

    def test_two_sweeps_not_flat(self):
        spec = NumericSpec(inputs=[
            InputVar(name="x", role="sweep", spec={"min": 0, "max": 1, "points": 5}),
            InputVar(name="y", role="sweep", spec={"min": 0, "max": 1, "points": 5})])
        assert is_flat_representable(spec) is False

    def test_solver_method_not_flat(self):
        spec = NumericSpec(
            inputs=[InputVar(name="x", role="sweep",
                             spec={"min": 0, "max": 1, "points": 5})],
            method=MethodSpec(name="fixed_point", params={"tol": 1e-8}))
        assert is_flat_representable(spec) is False

    def test_free_input_not_flat(self):
        spec = NumericSpec(inputs=[
            InputVar(name="x", role="free", spec={"initial": 0.5})])
        assert is_flat_representable(spec) is False


class TestSerializationBackCompat:
    def test_spec_round_trips_through_json(self):
        task = _sweep_task(spec=flat_to_spec(_sweep_task()))
        restored = NumericTask.model_validate(task.model_dump(mode="json"))
        assert restored.spec is not None
        assert restored.spec.inputs[0].name == "x"
        assert restored.spec.outputs[0].shape == "series"

    def test_legacy_task_without_spec_key_loads(self):
        # a save file predating the faceted IR has no "spec" key
        raw = _sweep_task().model_dump(mode="json")
        raw.pop("spec", None)
        restored = NumericTask.model_validate(raw)
        assert restored.spec is None


class TestWorkedExamples:
    """docs/24 §4: the intents that broke the flat model are all expressible
    with the SAME schema — only the data differs."""

    def test_convergence_spec(self):
        spec = NumericSpec(
            intent=IntentSpec(kind="root_finding", goal="iterate until |dx|<tol"),
            inputs=[InputVar(name="x", role="free", spec={"initial": 0.5})],
            outputs=[OutputSpec(name="x_star", shape="scalar"),
                     OutputSpec(name="trace", shape="trace", over=["iteration"],
                                role="diagnostic")],
            method=MethodSpec(name="fixed_point",
                              params={"tol": 1e-8, "max_iter": 1000}),
            presentation=PresentationSpec(kind="convergence"))
        assert spec.method.params["tol"] == 1e-8
        assert [o.shape for o in spec.outputs] == ["scalar", "trace"]
        assert is_flat_representable(spec) is False

    def test_diagonalization_spec_has_multiple_outputs(self):
        spec = NumericSpec(
            intent=IntentSpec(kind="eigenvalue_problem"),
            outputs=[OutputSpec(name="eigenvalues", shape="vector"),
                     OutputSpec(name="eigenvectors", shape="matrix")],
            method=MethodSpec(name="numpy_eig"),
            presentation=PresentationSpec(kind="table"))
        assert len(spec.outputs) == 2
        assert {o.shape for o in spec.outputs} == {"vector", "matrix"}

    def test_unknown_kind_is_allowed(self):
        # open kind (docs/24 §8-1): a novel intent is not a validation error
        spec = NumericSpec(intent=IntentSpec(kind="monte_carlo_integration"))
        assert spec.intent.kind == "monte_carlo_integration"
