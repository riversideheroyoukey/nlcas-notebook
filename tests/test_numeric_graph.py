"""G1 of the typed numeric computation graph (docs/26): evaluate template,
NB_MANIFEST contract, typed artifacts, value_from chaining, stale detection.

Sandbox-dependent tests are skipped when macOS sandbox-exec or .venv-numeric
is unavailable (same policy as test_numeric_layer.py).
"""
import json
import math

import pytest

from notebook.engine import NotebookEngine
from notebook.models.expression import Assumption
from notebook.models.numeric import (
    InputVar, IntentSpec, NumericArtifact, NumericSpec, NumericTask,
    OutputSpec,
)
from notebook.models.step import StepKind
from notebook.numeric.graph import can_bind, source_hash
from notebook.numeric.intent_registry import codegen_route
from notebook.numeric.sandbox import SandboxRunner
from notebook.numeric.spec_projection import is_flat_representable, spec_to_flat
from notebook.numeric.spec_validate import validate_numeric_spec
from notebook.numeric.templates import MANIFEST_MARKER, RESULT_MARKER, generate_code
from notebook.runtime.i18n import t
from notebook.runtime.shared_parser import UnknownNotationError

_runner = SandboxRunner()
needs_sandbox = pytest.mark.skipif(
    not _runner.available(),
    reason="macOS sandbox-exec or .venv-numeric python unavailable",
)


def _eval_task(**over) -> NumericTask:
    base = dict(
        task_id="N1",
        source_expression_ids=["E1"],
        source_canonical="a*b",
        problem_type="function_evaluation",
        target_quantity="P",
        fixed_values={"a": "3", "b": "4"},
    )
    base.update(over)
    return NumericTask(**base)


# ---------------------------------------------------------------------------
# evaluate template (docs/26 §10 — the chaining workhorse)
# ---------------------------------------------------------------------------

class TestEvaluateTemplate:
    def test_source_compiles_with_provenance_and_both_contract_lines(self):
        src, meta = generate_code(_eval_task())
        compile(src, "<t>", "exec")
        assert "task_id   : N1" in src
        assert RESULT_MARKER in src and MANIFEST_MARKER in src
        assert "matplotlib" not in src           # point evaluation: no figure
        assert meta["numpy_expr"] == "a * b"

    def test_sweep_parameter_rejected(self):
        task = _eval_task(sweep_parameters={"a": {"min": 0, "max": 1, "points": 5}})
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(task)
        assert ei.value.i18n_key == "numeric.evaluate_no_sweep"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)

    def test_unbound_symbol_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_eval_task(fixed_values={"a": "3"}))
        assert ei.value.i18n_key == "numeric.missing_values"

    def test_registry_routes_function_evaluation_deterministically(self):
        assert codegen_route("function_evaluation") == "deterministic"


# ---------------------------------------------------------------------------
# NB_MANIFEST emission by the sweep/plotting templates
# ---------------------------------------------------------------------------

class TestManifestEmission:
    def _sweep_task(self, problem_type="parameter_sweep") -> NumericTask:
        return NumericTask(
            task_id="N1", source_expression_ids=["E1"],
            source_canonical="sin(k*x)", problem_type=problem_type,
            target_quantity="y", fixed_values={"k": "2"},
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 50}},
        )

    def test_sweep_declares_series_backed_by_table(self):
        src, _ = generate_code(self._sweep_task())
        assert MANIFEST_MARKER in src
        assert '"value_kind": "series"' in src
        assert '"file": "table.csv"' in src

    def test_plotting_declares_no_bindable_artifacts(self):
        src, _ = generate_code(self._sweep_task("plotting"))
        assert MANIFEST_MARKER in src
        assert '"file": "table.csv"' not in src


# ---------------------------------------------------------------------------
# pure helpers: hash + binding table + projections + validator
# ---------------------------------------------------------------------------

class TestGraphHelpers:
    def test_hash_is_whitespace_and_order_insensitive(self):
        h1 = source_hash("a *  x\n+ b", ["x > 0", "a > 0"])
        h2 = source_hash("a * x + b", ["a > 0", "x > 0"])
        assert h1 == h2 and h1.startswith("sha256:")

    def test_hash_changes_when_content_changes(self):
        base = source_hash("a*x", ["x > 0"])
        assert source_hash("a*x + 1", ["x > 0"]) != base
        assert source_hash("a*x", ["x > 1"]) != base
        assert source_hash("a*x", []) != base

    def test_binding_table(self):
        assert can_bind("fixed", "scalar") and can_bind("free", "scalar")
        assert not can_bind("fixed", "series")
        assert not can_bind("sweep", "scalar")
        assert not can_bind("fixed", "")

    def test_zero_sweep_spec_is_flat_and_projects_to_evaluate(self):
        spec = NumericSpec(
            intent=IntentSpec(kind="function_evaluation"),
            inputs=[InputVar(name="a", role="fixed", spec={"value": 3})],
            outputs=[OutputSpec(name="P", shape="scalar")],
        )
        assert is_flat_representable(spec)
        flat = spec_to_flat(spec)
        assert flat["problem_type"] == "function_evaluation"
        assert flat["sweep_parameters"] == {}

    def test_validator_defers_value_from_to_the_engine(self):
        spec = NumericSpec(
            intent=IntentSpec(kind="function_evaluation"),
            inputs=[InputVar(name="c", role="fixed",
                             spec={"value_from": "na_x"})],
            outputs=[OutputSpec(name="Q", shape="scalar")],
        )
        codes = {i["code"] for i in validate_numeric_spec(spec)}
        assert "numeric.spec.bad_fixed_value" not in codes


# ---------------------------------------------------------------------------
# value_from binding: engine-side resolution + type checks (no sandbox)
# ---------------------------------------------------------------------------

def _scalar_artifact(engine, task_id="N000", value=12.0, value_kind="scalar",
                     value_type="real_float64"):
    art = NumericArtifact(task_id=task_id, artifact_type="result",
                          value_kind=value_kind, value_type=value_type,
                          value=value)
    engine.numeric_artifacts[art.artifact_id] = art
    return art


def _bind_spec(artifact_id, role="fixed", name="c"):
    return NumericSpec(
        intent=IntentSpec(kind="function_evaluation"),
        inputs=[InputVar(name=name, role=role,
                         spec={"value_from": artifact_id})],
        outputs=[OutputSpec(name="Q", shape="scalar")],
    )


class TestValueFromBinding:
    @pytest.fixture
    def engine(self):
        e = NotebookEngine(title="graph binding")
        e.register_expression("c^2")
        return e

    def _expr_id(self, engine):
        return next(iter(engine.expressions))

    def test_scalar_binds_and_records_edge(self, engine):
        art = _scalar_artifact(engine)
        task = engine.fork_numeric_task(self._expr_id(engine),
                                        spec=_bind_spec(art.artifact_id))
        assert task.input_artifact_refs == [art.artifact_id]
        assert float(task.fixed_values["c"]) == 12.0
        assert task.problem_type == "function_evaluation"
        # value_from stays in the resolved spec as provenance
        assert task.spec.inputs[0].spec["value_from"] == art.artifact_id

    def test_missing_artifact_fails_before_execution(self, engine):
        with pytest.raises(ValueError) as ei:
            engine.fork_numeric_task(self._expr_id(engine),
                                     spec=_bind_spec("na_nope"))
        assert str(ei.value) == t("numeric.spec.artifact_not_found",
                                  engine.lang, artifact_id="na_nope")

    def test_series_artifact_refused_for_fixed_input(self, engine):
        art = _scalar_artifact(engine, value_kind="series", value=None)
        with pytest.raises(ValueError) as ei:
            engine.fork_numeric_task(self._expr_id(engine),
                                     spec=_bind_spec(art.artifact_id))
        assert t("numeric.spec.artifact_not_bindable", engine.lang,
                 artifact_id=art.artifact_id, value_kind="series",
                 role="fixed") == str(ei.value)

    def test_value_from_on_sweep_role_refused(self, engine):
        art = _scalar_artifact(engine)
        with pytest.raises(ValueError) as ei:
            engine.fork_numeric_task(self._expr_id(engine),
                                     spec=_bind_spec(art.artifact_id,
                                                     role="sweep"))
        assert "value_from" in str(ei.value)

    def test_non_finite_scalar_refused(self, engine):
        art = _scalar_artifact(engine, value=float("nan"))
        with pytest.raises(ValueError):
            engine.fork_numeric_task(self._expr_id(engine),
                                     spec=_bind_spec(art.artifact_id))

    def test_complex_scalar_refused_for_real_slot(self, engine):
        art = _scalar_artifact(engine, value={"re": 1.0, "im": 2.0},
                               value_type="complex_float64")
        with pytest.raises(ValueError) as ei:
            engine.fork_numeric_task(self._expr_id(engine),
                                     spec=_bind_spec(art.artifact_id))
        assert "complex_float64" in str(ei.value)

    def test_new_messages_render_in_both_languages(self):
        for key, kw in (
            ("numeric.spec.artifact_not_found", {"artifact_id": "x"}),
            ("numeric.spec.artifact_not_bindable",
             {"artifact_id": "x", "value_kind": "series", "role": "fixed"}),
            ("numeric.spec.bad_value_from_role", {"name": "c", "role": "sweep"}),
            ("numeric.evaluate_no_sweep", {"count": 1}),
            ("numeric.executed_value", {"task_id": "N1", "target": "P",
                                        "value": "12"}),
        ):
            for lang in ("ja", "en"):
                assert t(key, lang, **kw)


# ---------------------------------------------------------------------------
# minimize_1d template (G2-2, docs/26 §10)
# ---------------------------------------------------------------------------

def _min_task(**over) -> NumericTask:
    base = dict(
        task_id="N1",
        source_expression_ids=["E1"],
        source_canonical="(x-2)^2 + c",
        problem_type="minimize_1d",
        target_quantity="y",
        fixed_values={"c": "1"},
        sweep_parameters={"x": {"min": 0, "max": 10}},
    )
    base.update(over)
    return NumericTask(**base)


class TestMinimizeTemplate:
    def test_source_compiles_with_scipy_and_typed_outputs(self):
        src, meta = generate_code(_min_task())
        compile(src, "<t>", "exec")
        assert "from scipy.optimize import minimize_scalar" in src
        assert "x_star" in src and "y_min" in src
        assert "boundary_solution" in src
        assert meta["bounds"] == {"min": "0", "max": "10"}

    def test_exactly_one_bounded_variable_required(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_min_task(sweep_parameters={}))
        assert ei.value.i18n_key == "numeric.need_one_bounded"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)

    def test_tol_comes_from_spec_method_params(self):
        from notebook.models.numeric import MethodSpec
        spec = NumericSpec(
            intent=IntentSpec(kind="minimize_1d"),
            inputs=[InputVar(name="x", role="sweep",
                             spec={"min": 0, "max": 10}),
                    InputVar(name="c", role="fixed", spec={"value": 1})],
            outputs=[OutputSpec(name="y", shape="scalar")],
            method=MethodSpec(name="brent_bounded", params={"tol": 1e-4}),
        )
        src, meta = generate_code(_min_task(spec=spec))
        assert meta["tol"] == 1e-4
        assert '"xatol": 0.0001' in src

    def test_registry_and_validator_accept_minimize_spec(self):
        assert codegen_route("minimize_1d") == "deterministic"
        spec = NumericSpec(
            intent=IntentSpec(kind="minimize_1d"),
            inputs=[InputVar(name="x", role="sweep",
                             spec={"min": 0, "max": 10})],   # bounds, no points
            outputs=[OutputSpec(name="y", shape="scalar")],
        )
        codes = {i["code"] for i in validate_numeric_spec(spec)}
        assert "numeric.spec.too_few_points" not in codes
        # …but a sweep kind still demands a grid
        spec2 = spec.model_copy(update={
            "intent": IntentSpec(kind="parameter_sweep")})
        codes2 = {i["code"] for i in validate_numeric_spec(spec2)}
        assert "numeric.spec.too_few_points" in codes2


@needs_sandbox
class TestMinimizeE2E:
    def test_interior_minimum_with_typed_artifacts(self, tmp_path):
        e = NotebookEngine(title="minimize e2e")
        e.save(tmp_path / "min.json")
        ex = e.register_expression("(x-2)^2 + c")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="minimize_1d", target_quantity="y",
            fixed_values={"c": 1}, sweep_parameters={"x": {"min": 0, "max": 10}})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr

        typed = {e.numeric_artifacts[a].metadata.get("name"):
                 e.numeric_artifacts[a]
                 for a in run.result_artifact_ids
                 if e.numeric_artifacts[a].value_kind == "scalar"}
        assert abs(typed["x_star"].value - 2.0) < 1e-5
        assert abs(typed["y_min"].value - 1.0) < 1e-9

        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.operation_args["diagnostics"]["boundary_solution"] is False
        assert t("numeric.boundary_note", e.lang) not in reflux.visible_action

        # the argmin chains into a follow-up evaluation (optimize → evaluate)
        e.register_expression("c^2")
        eid2 = [k for k in e.expressions if k != ex.display_id][0]
        task2 = e.fork_numeric_task(
            eid2, spec=_bind_spec(typed["x_star"].artifact_id))
        run2 = e.run_numeric_task(task2.task_id)
        assert run2.ok, run2.stderr
        vals = [e.numeric_artifacts[a].value for a in run2.result_artifact_ids
                if e.numeric_artifacts[a].value_kind == "scalar"]
        assert vals and abs(vals[0] - 4.0) < 1e-8

    def test_boundary_solution_is_flagged_loudly(self, tmp_path):
        e = NotebookEngine(title="boundary")
        e.save(tmp_path / "b.json")
        ex = e.register_expression("x")     # min of x on [1,5] sits at x=1
        task = e.fork_numeric_task(
            ex.display_id, problem_type="minimize_1d", target_quantity="y",
            sweep_parameters={"x": {"min": 1, "max": 5}})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.operation_args["diagnostics"]["boundary_solution"] is True
        assert t("numeric.boundary_note", e.lang) in reflux.visible_action


# ---------------------------------------------------------------------------
# matrix evaluate (G2-4, docs/26 §10): surface-level decomposition
# ---------------------------------------------------------------------------

class TestMatrixLowering:
    def test_cells_lowered_through_scalar_whitelist(self):
        from notebook.numeric.expr_to_numpy import canonical_matrix_to_numpy
        rows, free, shape = canonical_matrix_to_numpy(
            "matrix([eps, Delta], [Delta, -eps])")
        assert shape == (2, 2)
        assert rows[0][0] == "eps"
        assert rows[1][1] in ("-1 * eps", "-eps", "(-1) * eps", "-1*eps")
        assert free == {"eps", "Delta"}

    def test_two_element_row_is_not_a_commutator(self):
        """The surface-level split must protect [a, b] rows from the
        bracket→Commutator normalization (right for operator algebra,
        wrong inside matrix())."""
        from notebook.numeric.expr_to_numpy import canonical_matrix_to_numpy
        rows, _free, _shape = canonical_matrix_to_numpy(
            "matrix([a, b], [c, d])")
        assert rows == [["a", "b"], ["c", "d"]]

    def test_ragged_matrix_rejected(self):
        from notebook.numeric.expr_to_numpy import canonical_matrix_to_numpy
        with pytest.raises(UnknownNotationError) as ei:
            canonical_matrix_to_numpy("matrix([a, b], [c])")
        assert ei.value.i18n_key == "numeric.matrix_ragged"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang)

    def test_whitelist_still_guards_cells(self):
        from notebook.numeric.expr_to_numpy import canonical_matrix_to_numpy
        with pytest.raises(UnknownNotationError) as ei:
            canonical_matrix_to_numpy("matrix([zeta(x)], [x])")
        assert ei.value.i18n_key == "numeric.unsupported_function"


class TestMatrixEvaluateTemplate:
    def _task(self, canonical="matrix([eps, Delta], [Delta, -eps])", **over):
        base = dict(task_id="N1", source_expression_ids=["E1"],
                    source_canonical=canonical,
                    problem_type="function_evaluation", target_quantity="H",
                    fixed_values={"eps": "1.0", "Delta": "0.5"})
        base.update(over)
        return NumericTask(**base)

    def test_source_compiles_and_declares_matrix(self):
        src, meta = generate_code(self._task())
        compile(src, "<t>", "exec")
        assert '"value_kind": "matrix"' in src
        assert meta["matrix_shape"] == [2, 2]

    def test_unbound_symbol_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(self._task(fixed_values={"eps": "1.0"}))
        assert ei.value.i18n_key == "numeric.missing_values"


@needs_sandbox
class TestMatrixEvaluateE2E:
    def test_real_matrix_artifact_with_shape(self, tmp_path):
        e = NotebookEngine(title="matrix eval")
        e.save(tmp_path / "m.json")
        ex = e.register_expression("matrix([eps, Delta], [Delta, -eps])")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="function_evaluation",
            target_quantity="H", fixed_values={"eps": 1.0, "Delta": 0.5})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        mats = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                if e.numeric_artifacts[a].value_kind == "matrix"]
        assert mats, "expected a typed matrix artifact"
        m = mats[0]
        assert m.shape == [2, 2]
        assert m.value_type == "real_float64"
        assert m.value == [[1.0, 0.5], [0.5, -1.0]]
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.visible_action == t(
            "numeric.executed_matrix", e.lang, task_id=task.task_id,
            target="H", m=2, n=2)
        # …and a matrix refuses to bind into a scalar fixed slot
        e.register_expression("c + 1")
        eid2 = [k for k in e.expressions if k != ex.display_id][0]
        with pytest.raises(ValueError):
            e.fork_numeric_task(eid2, spec=_bind_spec(m.artifact_id))

    def test_complex_matrix_typed_honestly(self, tmp_path):
        e = NotebookEngine(title="pauli y")
        e.save(tmp_path / "py.json")
        ex = e.register_expression("matrix([0, -%i], [%i, 0])")
        task = e.fork_numeric_task(ex.display_id,
                                   problem_type="function_evaluation",
                                   target_quantity="sigma_y")
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        m = [e.numeric_artifacts[a] for a in run.result_artifact_ids
             if e.numeric_artifacts[a].value_kind == "matrix"][0]
        assert m.value_type == "complex_float64"
        assert m.value[0][1] == {"re": 0.0, "im": -1.0}
        assert m.value[1][0] == {"re": 0.0, "im": 1.0}


# ---------------------------------------------------------------------------
# eigenvalue_problem: fused evaluate+diagonalize (G3-1, docs/26 §10)
# ---------------------------------------------------------------------------

def _eigen_task(**over) -> NumericTask:
    base = dict(task_id="N1", source_expression_ids=["E1"],
                source_canonical="matrix([eps, Delta], [Delta, -eps])",
                problem_type="eigenvalue_problem", target_quantity="E",
                fixed_values={"eps": "1.0", "Delta": "0.5"})
    base.update(over)
    return NumericTask(**base)


class TestEigenTemplate:
    def test_point_mode_compiles_with_ordering(self):
        src, meta = generate_code(_eigen_task())
        compile(src, "<t>", "exec")
        assert "np.linalg.eigh" in src and "np.linalg.eig" in src
        assert "sorted_by_value" in src and "eigenvalue_order" in src
        assert meta["mode"] == "point"

    def test_sweep_mode_compiles_with_branches(self):
        task = _eigen_task(fixed_values={"Delta": "0.5"},
                           sweep_parameters={"eps": {"min": -2, "max": 2,
                                                     "points": 81}})
        src, meta = generate_code(task)
        compile(src, "<t>", "exec")
        assert "eigvalsh" in src and "branches.csv" in src
        assert meta["mode"] == "sweep"

    def test_non_matrix_source_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_eigen_task(source_canonical="a*x"))
        assert ei.value.i18n_key == "numeric.eigen_needs_matrix"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang)

    def test_non_square_matrix_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_eigen_task(
                source_canonical="matrix([a, b, c], [d, e, f])",
                fixed_values={k: "1" for k in "abcdef"}))
        assert ei.value.i18n_key == "numeric.eigen_needs_square"

    def test_registry_routes_deterministically(self):
        assert codegen_route("eigenvalue_problem") == "deterministic"


@needs_sandbox
class TestEigenE2E:
    def test_bdg_point_spectrum_with_typed_artifacts(self, tmp_path):
        """BdG-style 2x2: eigenvalues ±sqrt(eps²+Δ²), ascending, with an
        eigenvector matrix in matching column order."""
        e = NotebookEngine(title="eigen point")
        e.save(tmp_path / "e.json")
        ex = e.register_expression("matrix([eps, Delta], [Delta, -eps])")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="eigenvalue_problem",
            target_quantity="E", fixed_values={"eps": 1.0, "Delta": 0.5})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr

        arts = {e.numeric_artifacts[a].metadata.get("name"):
                e.numeric_artifacts[a]
                for a in run.result_artifact_ids
                if e.numeric_artifacts[a].value_kind}
        lam = math.sqrt(1.0**2 + 0.5**2)
        vals = arts["eigenvalues"]
        assert vals.value_kind == "vector" and vals.shape == [2]
        assert abs(vals.value[0] + lam) < 1e-12    # ascending: -λ first
        assert abs(vals.value[1] - lam) < 1e-12
        assert vals.ordering == {"kind": "sorted_by_value", "ascending": True}
        vecs = arts["eigenvectors"]
        assert vecs.value_kind == "matrix" and vecs.shape == [2, 2]
        assert vecs.ordering == {"kind": "eigenvalue_order"}

        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.operation_args["diagnostics"]["hermitian"] is True
        assert "固有値" in reflux.visible_action or \
               "eigenvalues" in reflux.visible_action

    def test_avoided_crossing_sweep(self, tmp_path):
        """The S-y use case: eigenvalue branches of ((eps,Δ),(Δ,-eps)) over
        eps show an avoided crossing with gap 2Δ at eps=0."""
        e = NotebookEngine(title="avoided crossing")
        e.save(tmp_path / "ac.json")
        ex = e.register_expression("matrix([eps, Delta], [Delta, -eps])")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="eigenvalue_problem",
            target_quantity="E", fixed_values={"Delta": 0.5},
            sweep_parameters={"eps": {"min": -2, "max": 2, "points": 81}})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr

        tables = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                  if e.numeric_artifacts[a].artifact_type == "table"]
        assert tables and tables[0].value_kind == "series"
        lines = open(tables[0].path).read().strip().splitlines()
        rows = [[float(x) for x in ln.split(",")] for ln in lines[1:]]
        # gap at eps=0 (grid midpoint, 81 points → index 40) equals 2Δ = 1
        mid = rows[40]
        assert abs(mid[0]) < 1e-9
        assert abs((mid[2] - mid[1]) - 1.0) < 1e-9
        figs = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                if e.numeric_artifacts[a].artifact_type == "figure"]
        assert figs

    def test_non_hermitian_flagged_loudly(self, tmp_path):
        e = NotebookEngine(title="non hermitian")
        e.save(tmp_path / "nh.json")
        ex = e.register_expression("matrix([1, 2], [0, 3])")
        task = e.fork_numeric_task(ex.display_id,
                                   problem_type="eigenvalue_problem")
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.operation_args["diagnostics"]["non_hermitian"] is True
        assert t("numeric.non_hermitian_note", e.lang) in reflux.visible_action

    def test_complex_hermitian_pauli_y(self, tmp_path):
        """σ_y is complex Hermitian: eigh path, real spectrum ±1."""
        e = NotebookEngine(title="pauli y eigen")
        e.save(tmp_path / "pye.json")
        ex = e.register_expression("matrix([0, -%i], [%i, 0])")
        task = e.fork_numeric_task(ex.display_id,
                                   problem_type="eigenvalue_problem")
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        vals = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                if e.numeric_artifacts[a].metadata.get("name") == "eigenvalues"]
        assert vals and vals[0].value == [-1.0, 1.0]


# ---------------------------------------------------------------------------
# self_consistent_loop: structured loop, region-fused (G2-3, docs/26 §8)
# ---------------------------------------------------------------------------

def _loop_spec(body, update, initials, fixed=(), tol=1e-10, max_iter=200):
    from notebook.models.numeric import MethodSpec
    inputs = [InputVar(name=n, role="free", spec={"initial": v})
              for n, v in initials.items()]
    inputs += [InputVar(name=n, role="fixed", spec={"value": v})
               for n, v in dict(fixed).items()]
    return NumericSpec(
        intent=IntentSpec(kind="self_consistent_loop"),
        inputs=inputs,
        outputs=[OutputSpec(name=n, shape="scalar") for n in initials]
                + [OutputSpec(name="trace", shape="trace",
                              over=["iteration"], role="diagnostic")],
        method=MethodSpec(name="fixed_point",
                          params={"body": list(body), "update": dict(update),
                                  "tol": tol, "max_iter": max_iter}),
    )


class TestLoopSpecValidation:
    def test_valid_loop_spec_passes(self):
        spec = _loop_spec(body=[{"expr_id": "E1", "name": "y"}],
                          update={"x": "y"}, initials={"x": 0.5})
        assert validate_numeric_spec(spec) == []

    def test_missing_body_and_update_reported(self):
        spec = _loop_spec(body=[], update={}, initials={"x": 0.5})
        codes = {i["code"] for i in validate_numeric_spec(spec)}
        assert {"numeric.spec.loop_no_body",
                "numeric.spec.loop_no_update"} <= codes

    def test_update_edge_must_connect_free_input_to_body_output(self):
        spec = _loop_spec(body=[{"expr_id": "E1", "name": "y"}],
                          update={"nope": "y"}, initials={"x": 0.5})
        codes = {i["code"] for i in validate_numeric_spec(spec)}
        assert "numeric.spec.loop_bad_update" in codes

    def test_loop_messages_render_in_both_languages(self):
        for key, kw in (
            ("numeric.spec.loop_no_body", {}),
            ("numeric.spec.loop_no_update", {}),
            ("numeric.spec.loop_bad_update", {"var": "x", "source": "y"}),
            ("numeric.executed_loop", {"task_id": "N1", "state": "x=0.739",
                                       "iterations": 30, "residual": "1e-10"}),
            ("numeric.loop_not_converged", {"task_id": "N1", "state": "x=512",
                                            "iterations": 10,
                                            "residual": "256"}),
        ):
            for lang in ("ja", "en"):
                assert t(key, lang, **kw)


class TestLoopTemplate:
    def _task(self, spec, body_canonicals):
        return NumericTask(
            task_id="N1", source_expression_ids=["E1"],
            source_canonical=body_canonicals[0][1],
            problem_type="self_consistent_loop",
            fixed_values={n: str(v)
                          for v in ()},   # filled below when needed
            spec=spec,
            source_body=[{"expr_id": f"E{i+1}", "name": n, "canonical": c}
                         for i, (n, c) in enumerate(body_canonicals)],
        )

    def test_fused_source_compiles_with_loop_and_trace(self):
        spec = _loop_spec(body=[{"expr_id": "E1", "name": "y"}],
                          update={"x": "y"}, initials={"x": 0.5})
        task = self._task(spec, [("y", "cos(x)")])
        src, meta = generate_code(task)
        compile(src, "<t>", "exec")
        assert "for _it in range(1," in src
        assert "y = float(np.cos(x))" in src
        assert "trace.csv" in src and MANIFEST_MARKER in src
        assert meta["update"] == {"x": "y"}

    def test_missing_initial_rejected(self):
        spec = _loop_spec(body=[{"expr_id": "E1", "name": "y"}],
                          update={"x": "y"}, initials={})
        spec = spec.model_copy(update={"inputs": []})
        task = self._task(spec, [("y", "cos(x)")])
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(task)
        assert ei.value.i18n_key == "numeric.missing_values"

    def test_body_symbol_bound_by_earlier_body_output(self):
        # z uses y (assigned one line earlier) — must lower cleanly
        spec = _loop_spec(body=[{"expr_id": "E1", "name": "y"},
                                {"expr_id": "E2", "name": "z"}],
                          update={"x": "z"}, initials={"x": 0.5})
        task = self._task(spec, [("y", "cos(x)"), ("z", "(x + y)/2")])
        src, _ = generate_code(task)
        compile(src, "<t>", "exec")
        assert src.index("y = float") < src.index("z = float")


@needs_sandbox
class TestLoopE2E:
    def test_single_expression_fixed_point_converges(self, tmp_path):
        """x ← cos(x): the Dottie number, the smallest self-consistent loop."""
        e = NotebookEngine(title="dottie")
        e.save(tmp_path / "d.json")
        ex = e.register_expression("cos(x)")
        spec = _loop_spec(body=[{"expr_id": ex.display_id, "name": "y"}],
                          update={"x": "y"}, initials={"x": 0.5})
        task = e.fork_numeric_task(ex.display_id, spec=spec)
        # fork snapshotted the body canonical
        assert task.source_body == [{"expr_id": ex.display_id, "name": "y",
                                     "canonical": "cos(x)"}]
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr

        typed = {e.numeric_artifacts[a].metadata.get("name"):
                 e.numeric_artifacts[a]
                 for a in run.result_artifact_ids
                 if e.numeric_artifacts[a].value_kind}
        assert abs(typed["x"].value - 0.7390851332151607) < 1e-8
        assert typed["trace"].value_kind == "trace"
        assert typed["trace"].path.endswith("trace.csv")

        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        diag = reflux.operation_args["diagnostics"]
        assert diag["converged"] is True and diag["iterations"] > 1
        assert "x=0.739" in reflux.visible_action

    def test_two_expression_body_fuses_into_one_run(self, tmp_path):
        """y = cos(x); z = (x+y)/2; x ← z — a multi-expression fused region
        (damped iteration, same fixed point), ONE sandbox execution."""
        e = NotebookEngine(title="fused")
        e.save(tmp_path / "f.json")
        e1 = e.register_expression("cos(x)")
        e2 = e.register_expression("(x + y)/2")
        spec = _loop_spec(body=[{"expr_id": e1.display_id, "name": "y"},
                                {"expr_id": e2.display_id, "name": "z"}],
                          update={"x": "z"}, initials={"x": 0.5})
        task = e.fork_numeric_task(e1.display_id, spec=spec)
        assert set(task.source_expression_ids) == {e1.display_id, e2.display_id}
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        assert len(e.numeric_executions) == 1        # ONE fused run
        typed = {e.numeric_artifacts[a].metadata.get("name"):
                 e.numeric_artifacts[a]
                 for a in run.result_artifact_ids
                 if e.numeric_artifacts[a].value_kind == "scalar"}
        assert abs(typed["x"].value - 0.7390851332151607) < 1e-8

    def test_divergent_loop_is_an_honest_failed_step(self, tmp_path):
        """x ← 2x diverges: exit code 0, but diagnostics.success=False must
        yield a FAILED step (docs/26 §8 — never a clean success wrapping a
        garbage number)."""
        e = NotebookEngine(title="diverge")
        e.save(tmp_path / "dv.json")
        ex = e.register_expression("2*x")
        spec = _loop_spec(body=[{"expr_id": ex.display_id, "name": "y"}],
                          update={"x": "y"}, initials={"x": 0.5},
                          max_iter=10)
        task = e.fork_numeric_task(ex.display_id, spec=spec)
        run = e.run_numeric_task(task.task_id)
        assert run.ok                                 # the PROCESS succeeded
        assert e.numeric_tasks[task.task_id].status == "failed"
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result][-1]
        assert reflux.status.value == "failed"
        assert reflux.visible_action == t(
            "numeric.loop_not_converged", e.lang, task_id=task.task_id,
            state="x=512", iterations=10, residual="256")

    def test_edited_body_expression_marks_loop_task_stale(self, tmp_path):
        e = NotebookEngine(title="loop stale")
        e.save(tmp_path / "ls.json")
        ex = e.register_expression("cos(x)")
        spec = _loop_spec(body=[{"expr_id": ex.display_id, "name": "y"}],
                          update={"x": "y"}, initials={"x": 0.5})
        task = e.fork_numeric_task(ex.display_id, spec=spec)
        assert not e.numeric_task_stale_info(task.task_id)["stale"]
        e.expressions[ex.display_id] = ex.model_copy(
            update={"canonical": "sin(x)"})
        info = e.numeric_task_stale_info(task.task_id)
        assert info["stale"]
        assert any(r["kind"] == "source_changed" for r in info["reasons"])


# ---------------------------------------------------------------------------
# reflux as expression (G2-1, docs/26 §5): artifact → theory layer
# ---------------------------------------------------------------------------

class TestRefluxAsExpression:
    @pytest.fixture
    def engine(self):
        return NotebookEngine(title="reflux")

    def test_scalar_becomes_h1_equation_with_provenance(self, engine):
        art = _scalar_artifact(engine, task_id="N001", value=2.274)
        expr = engine.reflux_numeric_value(art.artifact_id, "a_star")
        assert expr.canonical == "2.274"          # H-1: canonical is RHS only
        assert expr.latex == "a_star = 2.274"
        assert engine.lhs_alias_map.get("a_star") == expr.display_id
        step = engine.steps[expr.created_by_step]
        assert step.operation == "reflux_numeric_value"
        assert step.operation_args["artifact_id"] == art.artifact_id
        assert step.operation_args["task_id"] == "N001"
        assert step.visible_action == t("numeric.refluxed", engine.lang,
                                        name="a_star", value="2.274",
                                        task_id="N001")

    def test_integral_value_formats_without_trailing_zero(self, engine):
        art = _scalar_artifact(engine, value=12.0)
        expr = engine.reflux_numeric_value(art.artifact_id, "P")
        assert expr.canonical == "12"

    def test_full_precision_survives(self, engine):
        art = _scalar_artifact(engine, value=math.pi)
        expr = engine.reflux_numeric_value(art.artifact_id, "pival")
        assert float(expr.canonical) == math.pi   # repr() precision, not %.4g

    def test_series_artifact_refused(self, engine):
        art = _scalar_artifact(engine, value_kind="series", value=None)
        with pytest.raises(ValueError) as ei:
            engine.reflux_numeric_value(art.artifact_id, "s")
        assert str(ei.value) == t("numeric.reflux_not_scalar", engine.lang,
                                  artifact_id=art.artifact_id,
                                  value_kind="series")

    def test_complex_and_nan_refused(self, engine):
        cx = _scalar_artifact(engine, value={"re": 1.0, "im": 2.0},
                              value_type="complex_float64")
        with pytest.raises(ValueError):
            engine.reflux_numeric_value(cx.artifact_id, "z")
        nan = _scalar_artifact(engine, value=float("nan"))
        with pytest.raises(ValueError):
            engine.reflux_numeric_value(nan.artifact_id, "n")

    def test_invalid_symbol_name_refused(self, engine):
        art = _scalar_artifact(engine)
        for bad in ("2bad", "a-b", "", "a b"):
            with pytest.raises(ValueError) as ei:
                engine.reflux_numeric_value(art.artifact_id, bad)
            assert str(ei.value) == t("numeric.reflux_bad_name",
                                      engine.lang, name=bad)

    def test_missing_artifact_refused(self, engine):
        with pytest.raises(ValueError):
            engine.reflux_numeric_value("na_nope", "x")

    def test_reflux_messages_render_in_both_languages(self):
        for key, kw in (
            ("numeric.refluxed", {"name": "a", "value": "1", "task_id": "N1"}),
            ("numeric.reflux_not_scalar", {"artifact_id": "x",
                                           "value_kind": "series"}),
            ("numeric.reflux_bad_name", {"name": "2bad"}),
        ):
            for lang in ("ja", "en"):
                assert t(key, lang, **kw)


# ---------------------------------------------------------------------------
# Julia-session applicability (docs/26 §10 補注): the numeric runtime is
# backend-independent; the deterministic templates are Maxima-flavored.
# ---------------------------------------------------------------------------

class TestJuliaSessionNumeric:
    def test_reflux_works_in_a_julia_session(self):
        """G2-1 reflux rides H-1, whose P2 semantics differ per backend
        (docs/03) — a numeric equation must still register cleanly in a
        second-quantization (julia_full) session and leave the symbol
        usable without an undefined-symbol warning."""
        from notebook.models.session import SessionMode
        e = NotebookEngine(title="julia reflux",
                           mode=SessionMode.second_quantization)
        assert e.session.primary_backend == "julia_full"
        art = _scalar_artifact(e, task_id="N001", value=2.274)
        expr = e.reflux_numeric_value(art.artifact_id, "a_star")
        assert expr.canonical == "2.274"
        assert expr.latex == "a_star = 2.274"
        assert e.lhs_alias_map.get("a_star") == expr.display_id
        follow = e.register_expression("a_star * 2")
        assert follow.undefined_symbols == []

    def test_opaque_canonical_fails_template_loudly(self):
        """A Julia opaque canonical (_nb_E*) must never be silently
        mistranslated by the deterministic template — it raises, and the
        LLM path (source_latex) is the designed route (docs/22 §14,
        verified live in S-ac stage ③)."""
        task = _eval_task(source_canonical="_nb_E2")
        with pytest.raises(UnknownNotationError):
            generate_code(task)


# ---------------------------------------------------------------------------
# stale detection (docs/26 §7) — report-only
# ---------------------------------------------------------------------------

class TestStaleDetection:
    @pytest.fixture
    def engine(self):
        return NotebookEngine(title="stale")

    def _fork(self, engine, canonical="a*x"):
        ex = engine.register_expression(canonical)
        return ex, engine.fork_numeric_task(
            ex.display_id, problem_type="function_evaluation",
            fixed_values={"a": 1, "x": 2})

    def test_fresh_task_is_not_stale(self, engine):
        _ex, task = self._fork(engine)
        assert task.source_hash.startswith("sha256:")
        assert engine.numeric_task_stale_info(task.task_id) == {
            "stale": False, "reasons": []}

    def test_source_edit_marks_stale(self, engine):
        ex, task = self._fork(engine)
        engine.expressions[ex.display_id] = ex.model_copy(
            update={"canonical": "a*x + 1"})
        info = engine.numeric_task_stale_info(task.task_id)
        assert info["stale"]
        assert info["reasons"][0]["kind"] == "source_changed"

    def test_assumption_change_marks_stale(self, engine):
        _ex, task = self._fork(engine)
        engine.assumptions["as1"] = Assumption(
            assumption_id="as1", display="x>0", canonical="x > 0",
            scope="session")
        info = engine.numeric_task_stale_info(task.task_id)
        assert info["stale"]

    def test_deleted_source_marks_stale(self, engine):
        ex, task = self._fork(engine)
        del engine.expressions[ex.display_id]
        info = engine.numeric_task_stale_info(task.task_id)
        assert info["reasons"][0]["kind"] == "source_missing"

    def test_upstream_staleness_propagates(self, engine):
        ex1, task1 = self._fork(engine)
        art = _scalar_artifact(engine, task_id=task1.task_id)
        engine.register_expression("c^2")
        eid2 = [k for k in engine.expressions if k != ex1.display_id][0]
        task2 = engine.fork_numeric_task(eid2,
                                         spec=_bind_spec(art.artifact_id))
        assert not engine.numeric_task_stale_info(task2.task_id)["stale"]
        engine.expressions[ex1.display_id] = ex1.model_copy(
            update={"canonical": "a*x + 1"})
        info = engine.numeric_task_stale_info(task2.task_id)
        assert info["stale"]
        assert any(r["kind"] == "upstream_stale" for r in info["reasons"])

    def test_legacy_task_without_hash_never_stale(self, engine):
        ex, task = self._fork(engine)
        engine.numeric_tasks[task.task_id] = task.model_copy(
            update={"source_hash": ""})
        engine.expressions[ex.display_id] = ex.model_copy(
            update={"canonical": "changed"})
        assert not engine.numeric_task_stale_info(task.task_id)["stale"]


# ---------------------------------------------------------------------------
# graph view (G3-2, docs/26 §6): the DAG derived from refs
# ---------------------------------------------------------------------------

class TestGraphView:
    def test_empty_session_yields_empty_graph(self):
        e = NotebookEngine(title="empty graph")
        g = e.numeric_graph_view()
        assert g == {"expressions": [], "tasks": [], "edges": []}

    def test_chain_and_reflux_edges_derived(self):
        e = NotebookEngine(title="graph view")
        ex1 = e.register_expression("a*b")
        task1 = e.fork_numeric_task(ex1.display_id,
                                    problem_type="function_evaluation",
                                    fixed_values={"a": 3, "b": 4})
        art = _scalar_artifact(e, task_id=task1.task_id, value=12.0)
        ex2 = e.register_expression("c^2")
        task2 = e.fork_numeric_task(ex2.display_id,
                                    spec=_bind_spec(art.artifact_id))
        out = e.reflux_numeric_value(art.artifact_id, "P_val")

        g = e.numeric_graph_view()
        expr_ids = {x["expr_id"] for x in g["expressions"]}
        # numeric-touching expressions only: sources + the refluxed one
        assert {ex1.display_id, ex2.display_id, out.display_id} <= expr_ids
        kinds = {(ed["from"], ed["to"], ed["kind"]) for ed in g["edges"]}
        assert (ex1.display_id, task1.task_id, "source") in kinds
        assert (ex2.display_id, task2.task_id, "source") in kinds
        assert (task1.task_id, task2.task_id, "value") in kinds
        assert (task1.task_id, out.display_id, "reflux") in kinds
        value_edge = next(ed for ed in g["edges"] if ed["kind"] == "value")
        assert value_edge["value_kind"] == "scalar"
        reflux_edge = next(ed for ed in g["edges"] if ed["kind"] == "reflux")
        assert reflux_edge["name"] == "P_val"
        # stale report rides on each task node
        t1 = next(tk for tk in g["tasks"] if tk["task_id"] == task1.task_id)
        assert t1["stale"] == {"stale": False, "reasons": []}

    def test_unrelated_expressions_stay_out(self):
        e = NotebookEngine(title="graph scope")
        e.register_expression("x + 1")            # never forked
        ex = e.register_expression("a*b")
        e.fork_numeric_task(ex.display_id,
                            problem_type="function_evaluation",
                            fixed_values={"a": 1, "b": 2})
        g = e.numeric_graph_view()
        assert [x["expr_id"] for x in g["expressions"]] == [ex.display_id]


# ---------------------------------------------------------------------------
# branch comparison (G3-3, docs/26 §10-c): read-only, loose grouping
# ---------------------------------------------------------------------------

class TestBranchComparison:
    def _fork(self, e, canonical="a*b", target="P", branch=None):
        ex = e.register_expression(canonical)
        task = e.fork_numeric_task(ex.display_id,
                                   problem_type="function_evaluation",
                                   target_quantity=target,
                                   fixed_values={"a": 1, "b": 2})
        if branch is not None:
            e.numeric_tasks[task.task_id] = task.model_copy(
                update={"source_branch_id": branch})
        return e.numeric_tasks[task.task_id]

    def test_single_branch_yields_no_comparable_group(self):
        e = NotebookEngine(title="one branch")
        self._fork(e)
        cmp = e.numeric_branch_comparison()
        assert cmp["branch_ids"] == ["main"]
        assert all(not g["comparable"] for g in cmp["groups"])

    def test_same_kind_and_target_across_branches_is_comparable(self):
        e = NotebookEngine(title="two branches")
        t1 = self._fork(e)                          # main
        t2 = self._fork(e, branch="B2")
        self._fork(e, target="OTHER", branch="B2")  # different target
        cmp = e.numeric_branch_comparison()
        assert cmp["branch_ids"] == ["B2", "main"]
        comparable = [g for g in cmp["groups"] if g["comparable"]]
        assert len(comparable) == 1
        g = comparable[0]
        assert g["kind"] == "function_evaluation" and g["target"] == "P"
        assert {e_["task_id"] for e_ in g["branches"]["main"]} == {t1.task_id}
        assert {e_["task_id"] for e_ in g["branches"]["B2"]} == {t2.task_id}
        other = next(gr for gr in cmp["groups"] if gr["target"] == "OTHER")
        assert not other["comparable"]

    @needs_sandbox
    def test_summary_and_figure_ride_on_entries(self, tmp_path):
        e = NotebookEngine(title="cmp summary")
        e.save(tmp_path / "c.json")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="parameter_sweep",
            target_quantity="y", fixed_values={"k": 2.0},
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 50}})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        cmp = e.numeric_branch_comparison()
        entry = cmp["groups"][0]["branches"]["main"][0]
        assert entry["task_id"] == task.task_id
        assert entry["status"] == "executed"
        assert entry["summary"]                       # the i18n'd reflux line
        assert entry["figure_artifact_id"]


# ---------------------------------------------------------------------------
# persistence of the new additive fields
# ---------------------------------------------------------------------------

class TestTypedPersistence:
    def test_typed_fields_roundtrip(self, tmp_path):
        e = NotebookEngine(title="typed persist")
        ex = e.register_expression("a*b")
        task = e.fork_numeric_task(ex.display_id,
                                   problem_type="function_evaluation",
                                   fixed_values={"a": 1, "b": 2})
        art = _scalar_artifact(e, task_id=task.task_id, value=2.0)
        path = tmp_path / "typed.json"
        e.save(path)
        loaded = NotebookEngine.load(path)
        lt = loaded.numeric_tasks[task.task_id]
        assert lt.source_hash == task.source_hash
        assert lt.input_artifact_refs == []
        la = loaded.numeric_artifacts[art.artifact_id]
        assert la.value_kind == "scalar" and la.value == 2.0
        assert la.value_type == "real_float64"

    def test_pre_g1_artifact_json_loads_with_defaults(self):
        # A slice-1 artifact dict (no typed keys) must validate untouched.
        old = {"artifact_id": "na_old", "task_id": "N1",
               "artifact_type": "figure", "path": "/tmp/x.png"}
        art = NumericArtifact.model_validate(old)
        assert art.value_kind == "" and art.value is None
        assert art.shape == [] and art.ordering == {}


# ---------------------------------------------------------------------------
# E2E: evaluate → typed artifact → bind into next task → evaluate (sandbox)
# ---------------------------------------------------------------------------

@needs_sandbox
class TestChainE2E:
    def test_two_node_chain(self, tmp_path):
        e = NotebookEngine(title="chain e2e")
        e.save(tmp_path / "chain.json")

        ex1 = e.register_expression("a*b")
        task1 = e.fork_numeric_task(
            ex1.display_id, problem_type="function_evaluation",
            target_quantity="P", fixed_values={"a": 3, "b": 4})
        run1 = e.run_numeric_task(task1.task_id)
        assert run1.ok, run1.stderr

        typed = [e.numeric_artifacts[a] for a in run1.result_artifact_ids
                 if e.numeric_artifacts[a].value_kind == "scalar"]
        assert typed, "evaluate must produce a typed scalar artifact"
        art = typed[0]
        assert art.value == 12.0
        assert art.value_type == "real_float64"
        assert art.metadata.get("name") == "P"

        # reflux message is the single-value form
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result]
        assert reflux[0].visible_action == t(
            "numeric.executed_value", e.lang, task_id=task1.task_id,
            target="P", value="12")
        assert reflux[0].operation_args["diagnostics"].get("success") is True

        ex2 = e.register_expression("c^2")
        task2 = e.fork_numeric_task(ex2.display_id,
                                    spec=_bind_spec(art.artifact_id))
        assert task2.input_artifact_refs == [art.artifact_id]
        run2 = e.run_numeric_task(task2.task_id)
        assert run2.ok, run2.stderr
        typed2 = [e.numeric_artifacts[a] for a in run2.result_artifact_ids
                  if e.numeric_artifacts[a].value_kind == "scalar"]
        assert typed2 and typed2[0].value == 144.0

        # persistence of the whole chain
        e.save(tmp_path / "chain.json")
        loaded = NotebookEngine.load(tmp_path / "chain.json")
        lt2 = loaded.numeric_tasks[task2.task_id]
        assert lt2.input_artifact_refs == [art.artifact_id]

    def test_complex_result_is_typed_complex(self, tmp_path):
        e = NotebookEngine(title="complex eval")
        e.save(tmp_path / "cx.json")
        ex = e.register_expression("exp(%i*x)")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="function_evaluation",
            target_quantity="z", fixed_values={"x": 1.0})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        typed = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                 if e.numeric_artifacts[a].value_kind == "scalar"]
        assert typed and typed[0].value_type == "complex_float64"
        v = typed[0].value
        assert abs(v["re"] - math.cos(1.0)) < 1e-9
        assert abs(v["im"] - math.sin(1.0)) < 1e-9
        # ...and a complex scalar refuses to bind into a real-only slot
        e.register_expression("c + 1")
        eid2 = [k for k in e.expressions if k != ex.display_id][0]
        with pytest.raises(ValueError):
            e.fork_numeric_task(eid2, spec=_bind_spec(typed[0].artifact_id))

    def test_sweep_series_artifact_gets_typed_from_manifest(self, tmp_path):
        e = NotebookEngine(title="sweep typed")
        e.save(tmp_path / "sw.json")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, problem_type="parameter_sweep",
            target_quantity="y", fixed_values={"k": 2.0},
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 50}})
        run = e.run_numeric_task(task.task_id)
        assert run.ok, run.stderr
        tables = [e.numeric_artifacts[a] for a in run.result_artifact_ids
                  if e.numeric_artifacts[a].artifact_type == "table"]
        assert tables and tables[0].value_kind == "series"
        assert tables[0].metadata.get("name") == "y"
