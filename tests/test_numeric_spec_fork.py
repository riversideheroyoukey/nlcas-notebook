"""Engine hook: fork_numeric_task(spec=…) — build a NumericTask from a faceted
spec (docs/24), validating it and projecting it onto the flat fields the
deterministic path / assumption guard consume.  No sandbox / bridge / UI."""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.numeric import (
    InputVar, IntentSpec, MethodSpec, NumericSpec, OutputSpec,
)


def _sweep_spec():
    return NumericSpec(
        intent=IntentSpec(kind="parameter_sweep", goal="sweep x"),
        inputs=[InputVar(name="x", role="sweep",
                         spec={"min": 0, "max": 6.28, "points": 50}),
                InputVar(name="k", role="fixed", spec={"value": "2"})],
        outputs=[OutputSpec(name="y", shape="series", over=["x"])])


class TestForkFromSpec:
    def test_flat_representable_spec_derives_flat_fields(self):
        e = NotebookEngine(title="spec fork")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(ex.display_id, spec=_sweep_spec())
        assert task.spec is not None and task.spec.intent.kind == "parameter_sweep"
        # flat fields derived from the spec, so the deterministic path works
        assert task.problem_type == "parameter_sweep"
        assert task.target_quantity == "y"
        assert task.fixed_values == {"k": "2"}
        assert task.sweep_parameters == {"x": {"min": 0, "max": 6.28, "points": 50}}
        art = e.generate_numeric_code(task.task_id)
        assert "np.linspace" in art.content

    def test_non_flat_spec_is_carried_with_empty_sweep(self):
        e = NotebookEngine(title="conv fork")
        ex = e.register_expression("sin(k*x)")
        spec = NumericSpec(
            intent=IntentSpec(kind="root_finding"),
            inputs=[InputVar(name="x", role="free", spec={"initial": 0.5}),
                    InputVar(name="k", role="fixed", spec={"value": "2"})],
            outputs=[OutputSpec(name="x_star", shape="scalar")],
            method=MethodSpec(name="fixed_point",
                              params={"tol": 1e-8, "max_iter": 100}))
        task = e.fork_numeric_task(ex.display_id, spec=spec)
        assert task.spec.intent.kind == "root_finding"
        assert task.sweep_parameters == {}    # not flat-representable
        assert task.fixed_values == {"k": "2"}

    def test_invalid_spec_is_rejected_before_task_created(self):
        e = NotebookEngine(title="bad fork")
        ex = e.register_expression("sin(k*x)")
        bad = NumericSpec(
            inputs=[InputVar(name="x", role="sweep",
                             spec={"min": 5, "max": 1, "points": 10})],  # min>max
            outputs=[OutputSpec(name="y", over=["x"])])
        n_before = len(e.numeric_tasks)
        with pytest.raises(ValueError):
            e.fork_numeric_task(ex.display_id, spec=bad)
        assert len(e.numeric_tasks) == n_before   # nothing created

    def test_spec_symbol_membership_is_checked_against_expression(self):
        # a spec naming an input the expression doesn't have is rejected
        e = NotebookEngine(title="sym fork")
        ex = e.register_expression("sin(k*x)")
        spec = NumericSpec(
            inputs=[InputVar(name="x", role="sweep",
                             spec={"min": 0, "max": 1, "points": 10}),
                    InputVar(name="w", role="fixed", spec={"value": "1"})],  # w absent
            outputs=[OutputSpec(name="y", over=["x"])])
        with pytest.raises(ValueError):
            e.fork_numeric_task(ex.display_id, spec=spec)

    def test_spec_survives_save_load(self, tmp_path):
        e = NotebookEngine(title="persist spec")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(ex.display_id, spec=_sweep_spec())
        p = tmp_path / "s.json"
        e.save(p)
        loaded = NotebookEngine.load(p)
        lt = loaded.numeric_tasks[task.task_id]
        assert lt.spec is not None
        assert [(v.name, v.role) for v in lt.spec.inputs] == [("x", "sweep"), ("k", "fixed")]
        assert lt.spec.outputs[0].shape == "series"

    def test_legacy_flat_fork_still_works(self):
        # no spec -> unchanged slice-1 behaviour, task.spec is None
        e = NotebookEngine(title="legacy fork")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1, "points": 10}},
            fixed_values={"k": 2})
        assert task.spec is None
        assert task.sweep_parameters == {"x": {"min": 0, "max": 1, "points": 10}}
