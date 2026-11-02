"""N-1 numeric layer: lowering, templates, sandbox, engine E2E, persistence.

Sandbox-dependent tests are skipped when macOS sandbox-exec or the dedicated
numeric interpreter (.venv-numeric) is unavailable, so the suite stays green
on machines without the numeric environment.
"""
import json

import pytest

from notebook.engine import NotebookEngine
from notebook.models.numeric import NumericTask
from notebook.models.step import StepKind
from notebook.numeric.expr_to_numpy import canonical_to_numpy
from notebook.numeric.sandbox import SandboxRunner
from notebook.numeric.templates import generate_code, RESULT_MARKER
from notebook.runtime.i18n import t
from notebook.runtime.shared_parser import UnknownNotationError

_runner = SandboxRunner()
needs_sandbox = pytest.mark.skipif(
    not _runner.available(),
    reason="macOS sandbox-exec or .venv-numeric python unavailable",
)


# ---------------------------------------------------------------------------
# canonical → numpy lowering
# ---------------------------------------------------------------------------

class TestNumpyLowering:
    def test_basic_functions_and_power(self):
        expr, free = canonical_to_numpy("sin(k*x)*exp(-a*x) + x^2")
        assert "np.sin(k * x)" in expr
        assert "np.exp" in expr
        assert "(x)**(2)" in expr
        assert free == {"k", "x", "a"}

    def test_pi_constant(self):
        expr, free = canonical_to_numpy("sin(2*%pi*x)")
        assert "np.pi" in expr
        assert free == {"x"}

    def test_unknown_function_fails_loudly(self):
        with pytest.raises(UnknownNotationError) as ei:
            canonical_to_numpy("zeta(x) + bessel(0, x)")
        assert ei.value.i18n_key == "numeric.unsupported_function"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)

    def test_indexed_construct_fails_loudly(self):
        with pytest.raises(UnknownNotationError) as ei:
            canonical_to_numpy("x_i * delta_ij")
        assert ei.value.i18n_key == "numeric.unsupported_construct"


# ---------------------------------------------------------------------------
# deterministic templates
# ---------------------------------------------------------------------------

def _task(**over) -> NumericTask:
    base = dict(
        task_id="N1",
        source_expression_ids=["E1"],
        source_canonical="sin(k*x)",
        problem_type="parameter_sweep",
        sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 100}},
        fixed_values={"k": "2.0"},
    )
    base.update(over)
    return NumericTask(**base)


class TestTemplates:
    def test_generates_reviewable_source_with_provenance(self):
        src, meta = generate_code(_task())
        assert "task_id   : N1" in src        # provenance header
        assert "E1" in src
        assert "np.linspace(0, 6.28, 100)" in src
        assert RESULT_MARKER in src
        assert "table.csv" in src             # parameter_sweep → CSV
        assert meta["numpy_expr"] == "np.sin(k * x)"
        compile(src, "<generated>", "exec")   # syntactically valid Python

    def test_plotting_type_skips_table(self):
        src, _ = generate_code(_task(problem_type="plotting"))
        assert "table.csv" not in src
        assert "figure.png" in src

    def test_unbound_symbol_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_task(fixed_values={}))
        assert ei.value.i18n_key == "numeric.missing_values"
        assert "k" in ei.value.fmt_kwargs["symbols"]

    def test_non_numeric_fixed_value_rejected(self):
        """fixed_values are injected into generated code — they must be plain
        numbers, never expressions (adversarial-input rule)."""
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_task(fixed_values={"k": "__import__('os')"}))
        assert ei.value.i18n_key == "numeric.bad_value"

    def test_fullwidth_digits_normalized_not_injected_verbatim(self):
        """Full-width digits (Japanese IME) reach the template unnormalized via
        the numeric modal's direct /numeric/fork POST.  float('０.３') parses,
        so they pass the numeric check — but '０.３' is a SyntaxError as Python
        source.  They must be normalized to half-width (project accepts
        full-width, CLAUDE.md H-5z), and the generated code must COMPILE."""
        src, _ = generate_code(_task(
            fixed_values={"k": "０.３"},
            sweep_parameters={"x": {"min": "０", "max": "１０", "points": 50}}))
        assert "k = 0.3" in src
        assert "np.linspace(0, 10, 50)" in src
        compile(src, "<generated>", "exec")   # would SyntaxError if injected verbatim

    def test_ascii_integer_values_kept_verbatim(self):
        # normalization must NOT churn plain ASCII values (0 stays 0, not 0.0)
        src, _ = generate_code(_task(
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 100}}))
        assert "np.linspace(0, 6.28, 100)" in src

    def test_non_finite_fixed_value_rejected(self):
        # inf/nan parse via float() but aren't literals -> reject, don't inject
        for bad in ("inf", "nan"):
            with pytest.raises(UnknownNotationError) as ei:
                generate_code(_task(fixed_values={"k": bad}))
            assert ei.value.i18n_key == "numeric.bad_value"

    def test_exactly_one_sweep_required(self):
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_task(sweep_parameters={}))
        assert ei.value.i18n_key == "numeric.need_one_sweep"

    def test_deterministic_type_gate(self):
        # root_finding has no deterministic template — it is the LLM path's
        # territory (eigenvalue_problem graduated to a template in G3).
        with pytest.raises(UnknownNotationError) as ei:
            generate_code(_task(problem_type="root_finding"))
        assert ei.value.i18n_key == "numeric.unsupported_problem_type"

    def test_figure_caption_names_target_and_fixed_values(self):
        """The figure caption (ax.set_title) must name the target(sweep_var)
        and every fixed symbol=value, so a plot pulled out of context (or
        several runs stacked in the timeline with different parameters)
        stays self-describing."""
        src, _ = generate_code(_task(target_quantity="y",
                                     fixed_values={"k": "2.0"}))
        assert "ax.set_title('y(x)   |   k=2.0')" in src

    def test_figure_caption_and_axis_labels_render_greek_symbols(self):
        """Greek-named parameters (gamma, omega_0 — the damped-oscillator
        symbols) must render via mathtext ($\\gamma$, $\\omega_{0}$), not as
        the bare canonical identifier, in both the caption and axis labels.
        omega_0 must be REGISTERED in the SymbolContext for this — an
        undeclared '_digit' subscript resolves as an IndexedSymbol, not a
        plain name (CLAUDE.md: undeclared subscript in an operation → index)."""
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        ctx.register_quantity("omega_0", kind="scalar")
        ctx.register_quantity("gamma", kind="scalar")
        src, _ = generate_code(_task(
            source_canonical="gamma*omega_0*exp(-t)",
            sweep_parameters={"t": {"min": 0, "max": 5, "points": 20}},
            fixed_values={"gamma": "0.3", "omega_0": "2.0"},
            target_quantity="x"), context=ctx)
        assert r"ax.set_title('x(t)   |   $\\gamma$=0.3, $\\omega_{0}$=2.0')" in src
        assert "ax.set_xlabel('t')" in src           # t has no Greek form
        assert "ax.set_ylabel('x')" in src            # x has no Greek form

    def test_figure_caption_omits_separator_when_no_fixed_values(self):
        # x is both the sweep var and the only free symbol, so nothing is
        # unbound even with fixed_values={}.
        src, _ = generate_code(_task(source_canonical="sin(x)",
                                     fixed_values={}, target_quantity="y"))
        assert "ax.set_title('y(x)')" in src

    def test_multiline_canonical_stays_valid_python(self):
        """Maxima pretty-prints long expressions across multiple lines
        (continuation-indented).  Embedded verbatim into the single-line
        '# --- theory expression: ... ---' comment, the continuation lines
        lose their leading '#' and become a Python IndentationError — found
        while building the S-x damped-oscillator scenario (fork of a
        solve_ode_general/specialize_by_conditions RHS, which Maxima wraps)."""
        wrapped = "sin(k*x)\n  + cos(k*x)"
        src, _ = generate_code(_task(source_canonical=wrapped))
        compile(src, "<generated>", "exec")   # would raise IndentationError pre-fix


# ---------------------------------------------------------------------------
# sandbox (empirically verified profile: net denied, writes confined)
# ---------------------------------------------------------------------------

@needs_sandbox
class TestSandbox:
    def test_numpy_runs_and_writes_inside_workdir(self, tmp_path):
        code = tmp_path / "ok.py"
        code.write_text(
            "import os, numpy as np\n"
            "open(os.path.join(os.environ['NB_NUMERIC_OUT'], 'out.txt'), 'w')"
            ".write(str(float(np.sum(np.arange(10)))))\n"
            "print('done')\n")
        r = _runner.run_file(code, tmp_path / "work")
        assert r.ok, r.stderr
        assert (tmp_path / "work" / "out.txt").read_text() == "45.0"
        assert "out.txt" in r.created_files

    def test_network_denied(self, tmp_path):
        code = tmp_path / "net.py"
        code.write_text(
            "import urllib.request\n"
            "urllib.request.urlopen('http://example.com', timeout=5)\n")
        r = _runner.run_file(code, tmp_path / "work")
        assert not r.ok
        assert "URLError" in r.stderr or "urlopen error" in r.stderr

    def test_write_outside_workdir_denied(self, tmp_path):
        target = tmp_path / "outside.txt"   # OUTSIDE the workdir subpath
        code = tmp_path / "w.py"
        code.write_text(f"open({str(target)!r}, 'w').write('x')\n")
        r = _runner.run_file(code, tmp_path / "work")
        assert not r.ok
        assert "PermissionError" in r.stderr
        assert not target.exists()

    def test_timeout_kills_runaway_code(self, tmp_path):
        code = tmp_path / "loop.py"
        code.write_text("while True:\n    pass\n")
        runner = SandboxRunner(timeout_s=3)
        r = runner.run_file(code, tmp_path / "work")
        assert r.timed_out and not r.ok
        assert r.duration_s < 20


# ---------------------------------------------------------------------------
# engine E2E: fork → generate → run → reflux → save/load
# ---------------------------------------------------------------------------

@needs_sandbox
class TestEngineVerticalSlice:
    def test_full_cycle(self, tmp_path):
        e = NotebookEngine(title="numeric e2e")
        ex = e.register_expression("sin(k*x)*exp(-a*x)")
        # numeric outputs are anchored to the saved session's folder/stem
        path = tmp_path / "numeric.json"
        e.save(path)
        task = e.fork_numeric_task(
            ex.display_id, problem_type="parameter_sweep",
            target_quantity="y",
            sweep_parameters={"x": {"min": 0, "max": 10, "points": 200}},
            fixed_values={"k": 2.0, "a": 0.3},
        )
        # provenance snapshot
        assert task.source_canonical == ex.canonical
        assert task.source_branch_id == "main"
        fork_steps = [s for s in e.steps.values()
                      if s.kind == StepKind.numeric_fork]
        assert fork_steps and fork_steps[0].input_refs == [ex.display_id]

        execution = e.run_numeric_task(task.task_id)
        assert execution.ok, execution.stderr
        assert "numpy" in execution.package_info

        kinds = {e.numeric_artifacts[a].artifact_type
                 for a in execution.result_artifact_ids}
        assert {"figure", "table", "result"} <= kinds

        result_art = next(a for a in execution.result_artifact_ids
                          if e.numeric_artifacts[a].artifact_type == "result")
        summary = json.loads(e.numeric_artifacts[result_art].content)
        # physics check: first max of sin(kx)e^{-ax} at x = atan(k/a)/k
        import math
        expected = math.atan(2.0 / 0.3) / 2.0
        assert abs(summary["max"]["x"] - expected) < 0.05

        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result]
        assert len(reflux) == 1
        assert reflux[0].operation_args["source_expression_ids"] == [ex.display_id]
        assert reflux[0].operation_args["summary"] == summary

        # persistence roundtrip
        e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.numeric_tasks[task.task_id].status == "executed"
        assert len(loaded.numeric_executions) == 1
        assert loaded._numeric_counter == 1

    def test_failed_run_is_recorded_not_raised(self, tmp_path):
        """A crashing execution must produce a failed status + log artifact,
        not an exception — the researcher reviews the log in the timeline."""
        e = NotebookEngine(title="numeric fail")
        e.save(tmp_path / "fail.json")
        ex = e.register_expression("log(x)")
        task = e.fork_numeric_task(
            ex.display_id,
            sweep_parameters={"x": {"min": 0, "max": 1, "points": 10}},
        )
        art = e.generate_numeric_code(task.task_id)
        # sabotage the code to guarantee a crash
        broken = art.content + "\nraise RuntimeError('boom')\n"
        e.numeric_artifacts[art.artifact_id] = art.model_copy(
            update={"content": broken})
        execution = e.run_numeric_task(task.task_id)
        assert not execution.ok
        assert e.numeric_tasks[task.task_id].status == "failed"
        reflux = [s for s in e.steps.values()
                  if s.kind == StepKind.numeric_result]
        assert reflux and reflux[0].status.value == "failed"


# ---------------------------------------------------------------------------
# persistence without sandbox (runs everywhere)
# ---------------------------------------------------------------------------

class TestNumericPersistence:
    def test_old_session_without_numeric_keys_loads(self, tmp_path):
        e = NotebookEngine(title="no numeric")
        e.register_expression("x+1")
        path = tmp_path / "plain.json"
        e.save(path)
        data = json.loads(path.read_text())
        for key in ("numeric_tasks", "numeric_artifacts",
                    "numeric_executions", "numeric_counter"):
            data.pop(key, None)
        path.write_text(json.dumps(data, ensure_ascii=False))
        loaded = NotebookEngine.load(path)
        assert loaded.numeric_tasks == {}
        assert loaded._numeric_counter == 0

    def test_task_roundtrip_without_execution(self, tmp_path):
        e = NotebookEngine(title="numeric rt")
        ex = e.register_expression("k*x^2")
        e.fork_numeric_task(ex.display_id,
                            sweep_parameters={"x": {"min": 0, "max": 1}},
                            fixed_values={"k": 1.5})
        path = tmp_path / "rt.json"
        e.save(path)
        loaded = NotebookEngine.load(path)
        task = loaded.numeric_tasks["N001"]
        assert task.source_canonical == "k*x^2"
        assert task.fixed_values == {"k": "1.5"}
        assert task.status == "draft"


class TestListResultGuard:
    """Found while verifying scenario S-y: forking the eigenvalues LIST
    output must give a clear guidance error, not a confusing
    'Commutator unsupported' from the lowering."""

    def test_list_canonical_fork_gives_guidance(self):
        with NotebookEngine(title="numeric list guard") as e:
            m = e.register_expression("[eps, Delta; Delta, -eps]")
            step, _ = e.run_operation("eigenvalues", {"expr": m.display_id})
            ev_id = step.output_refs[0]
            with pytest.raises(ValueError) as ei:
                e.fork_numeric_task(
                    ev_id, sweep_parameters={"eps": {"min": -1, "max": 1}})
            assert "solve" in str(ei.value)      # points at the working route
            for lang in ("ja", "en"):
                assert t("numeric.list_not_supported", lang, expr_id="E2")

    def test_solve_route_still_works(self, tmp_path):
        """The verified S-y route: charpoly -> solve -> fork a solution."""
        with NotebookEngine(title="numeric solve route") as e:
            e.save(tmp_path / "solve_route.json")
            m = e.register_expression("[eps, Delta; Delta, -eps]")
            s1, _ = e.run_operation("characteristic_polynomial",
                                    {"expr": m.display_id, "var": "x"})
            s2, _ = e.run_operation("solve_equation",
                                    {"equation": s1.output_refs[0], "var": "x"})
            sol = [oid for oid in s2.output_refs
                   if not e.expressions[oid].hidden][0]
            task = e.fork_numeric_task(
                sol, sweep_parameters={"eps": {"min": -3, "max": 3, "points": 100}},
                fixed_values={"Delta": 1.0})
            if SandboxRunner().available():
                execution = e.run_numeric_task(task.task_id)
                assert execution.ok, execution.stderr


class TestNumericOutputAnchor:
    """Numeric outputs are anchored to the saved session's folder/stem
    (user decision 2026-07-05: reject a separate sessions/<name>_files/
    bundle — reuse the JSON's own location instead)."""

    def test_unsaved_session_blocks_run(self):
        e = NotebookEngine(title="numeric unsaved")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
            fixed_values={"k": 1.0})
        with pytest.raises(ValueError) as ei:
            e.run_numeric_task(task.task_id)
        for lang in ("ja", "en"):
            assert t("numeric.save_session_first", lang)
        assert "save" in str(ei.value).lower() or "保存" in str(ei.value)

    @needs_sandbox
    def test_output_lands_under_stem_numeric_beside_json(self, tmp_path):
        e = NotebookEngine(title="numeric anchor")
        path = tmp_path / "sub" / "myexp.json"
        path.parent.mkdir()
        e.save(path)
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
            fixed_values={"k": 1.0})
        execution = e.run_numeric_task(task.task_id)
        assert execution.ok, execution.stderr
        expected_root = tmp_path / "sub" / "myexp_numeric"
        workdir = expected_root / task.task_id / execution.execution_id
        assert (workdir / "code.py").exists()

    @needs_sandbox
    def test_autosave_does_not_move_anchor(self, tmp_path):
        e = NotebookEngine(title="numeric autosave")
        anchor_path = tmp_path / "anchored.json"
        e.save(anchor_path)
        # simulate autosave rotation elsewhere — must not shift the anchor
        autosave_dir = tmp_path / "autosave"
        autosave_dir.mkdir()
        e.save(autosave_dir / "slot_1.json", is_autosave=True)
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
            fixed_values={"k": 1.0})
        execution = e.run_numeric_task(task.task_id)
        assert execution.ok, execution.stderr
        workdir = (tmp_path / "anchored_numeric" / task.task_id
                   / execution.execution_id)
        assert (workdir / "code.py").exists()

    def test_save_as_relocates_numeric_outputs(self, tmp_path):
        """Save As after a run (which auto-saved under a throwaway stem) must
        MOVE <old>_numeric/ → <new>_numeric/ and re-point artifact paths, so
        the outputs stay beside the session and the figure route (which anchors
        to the current stem) still finds them.  Regression for the 2026-07-09
        UI-E2E finding: without this, Save As orphaned the code/figure/table and
        404'd the figure.  Pure-filesystem test (no sandbox): simulate a run's
        output dir, then Save As."""
        from notebook.models.numeric import NumericArtifact
        e = NotebookEngine(title="relocate")
        e.save(tmp_path / "auto.json")                 # run-time anchor
        old_fig = tmp_path / "auto_numeric" / "N001" / "run1" / "figure.png"
        old_fig.parent.mkdir(parents=True)
        old_fig.write_bytes(b"\x89PNG\r\n")
        art = NumericArtifact(task_id="N001", artifact_type="figure",
                              path=str(old_fig))
        e.numeric_artifacts[art.artifact_id] = art

        e.save(tmp_path / "chosen.json")               # Save As
        new_fig = tmp_path / "chosen_numeric" / "N001" / "run1" / "figure.png"
        assert new_fig.is_file()
        assert not old_fig.exists()
        assert not (tmp_path / "auto_numeric").exists()
        assert e.numeric_artifacts[art.artifact_id].path == str(new_fig)

    def test_autosave_does_not_relocate_numeric_outputs(self, tmp_path):
        """An autosave (rotation) must NOT move the numeric anchor even when
        outputs already exist (the anchor belongs to the explicit save)."""
        from notebook.models.numeric import NumericArtifact
        e = NotebookEngine(title="reloc autosave")
        e.save(tmp_path / "anchor.json")
        fig = tmp_path / "anchor_numeric" / "N001" / "run1" / "figure.png"
        fig.parent.mkdir(parents=True)
        fig.write_bytes(b"\x89PNG")
        art = NumericArtifact(task_id="N001", artifact_type="figure",
                              path=str(fig))
        e.numeric_artifacts[art.artifact_id] = art
        (tmp_path / "autosave").mkdir()
        e.save(tmp_path / "autosave" / "slot_1.json", is_autosave=True)
        # outputs stay put; artifact path unchanged
        assert fig.is_file()
        assert e.numeric_artifacts[art.artifact_id].path == str(fig)
        assert not (tmp_path / "autosave" / "slot_1_numeric").exists()


class TestIdFormats:
    """task_id/execution_id readability fixes (user request 2026-07-05):
    zero-padded task_id sorts correctly past N9→N10, and a timestamp-based
    execution_id makes newest-run visible at a glance instead of a random
    UUID fragment."""

    def test_task_id_zero_padded_and_sorts_correctly(self):
        e = NotebookEngine(title="numeric id format")
        ex = e.register_expression("k*x")
        ids = [e.fork_numeric_task(
                   ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
                   fixed_values={"k": 1.0}).task_id
               for _ in range(11)]
        assert ids[0] == "N001"
        assert ids[9] == "N010"
        assert ids[10] == "N011"
        assert sorted(ids) == ids   # lexicographic order == allocation order

    def test_execution_id_is_timestamp_based_and_sortable(self):
        import re
        e = NotebookEngine(title="numeric exec id")
        ex = e.register_expression("k*x")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
            fixed_values={"k": 1.0})
        from notebook.models.numeric import NumericExecution
        execs = [NumericExecution(task_id=task.task_id,
                                  code_artifact_id="na_dummy")
                 for _ in range(3)]
        for ex_ in execs:
            assert re.fullmatch(r"nx_\d{8}T\d{12}", ex_.execution_id)
        assert sorted(x.execution_id for x in execs) == \
            [x.execution_id for x in execs]


class TestSandboxProfileReadsOwnWorkdir:
    """Regression (2026-07-09, found via the numeric-UI E2E): the sandbox
    profile deny-reads PROJECT_ROOT/sessions to protect other sessions'
    research data, but session-anchored numeric output puts the task's own
    exec dir UNDER sessions/ (<stem>_numeric/<task>/<exec>/).  SBPL is
    last-match-wins, so without an explicit allow-read of the work dir AFTER
    that deny, the sandbox cannot even read its OWN code.py → every numeric
    run from a normally-saved session (in sessions/) failed with
    'Operation not permitted'.  The existing sandbox tests all used tmp_path
    (outside sessions/), so this was never caught."""

    def test_workdir_under_sessions_is_readable(self):
        from notebook.numeric.sandbox import build_profile, _PROJECT_ROOT
        workdir = str(_PROJECT_ROOT / "sessions" / "demo_numeric" / "N001" / "nx_x")
        prof = build_profile(workdir)
        deny_idx = prof.index(str(_PROJECT_ROOT / "sessions") + '"')
        allow_read_idx = prof.index(f'(allow file-read*  (subpath "{workdir}"))')
        # the workdir allow-read must come AFTER the sessions deny (last wins)
        assert allow_read_idx > deny_idx
        # and the broad sessions deny is still present (other sessions protected)
        assert '(deny file-read* (subpath "' in prof

    def test_secret_dirs_still_denied(self):
        from notebook.numeric.sandbox import build_profile
        prof = build_profile("/tmp/whatever_numeric/N001/nx")
        for secret in (".ssh", ".aws", ".gnupg"):
            assert secret in prof
