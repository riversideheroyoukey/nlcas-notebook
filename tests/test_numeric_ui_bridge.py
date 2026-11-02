"""Numeric-layer UI bridge endpoints (N-1 slice 1, 2026-07-09).

The deterministic fork→generate→run→reflux loop is now driveable from the
notebook UI.  These tests cover the three bridge additions the UI relies on:

  * GET  /numeric/params/{expr_id}   — auto-enumerate the free symbols the
      task-editor modal must ask a value for (reuses the exact template
      lowering, so the modal and codegen never disagree).
  * GET  /timeline                   — now emits numeric_fork / numeric_result
      entries (with the figure artifact id) so a reloaded session shows them.
  * GET  /numeric/figure/{id}        — serves a figure PNG with a path-
      traversal guard (must be a figure artifact under the session's own
      numeric output root).

Full-loop tests that actually execute code are skipped when the macOS
sandbox / .venv-numeric is unavailable (same policy as test_numeric_layer).
"""
import asyncio
import json

import pytest

import ui.notebook_bridge as nb
import ui.session_io as session_io
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from ui.notebook_bridge import NumericForkRequest, NumericRunRequest
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.numeric.sandbox import SandboxRunner

needs_sandbox = pytest.mark.skipif(
    not SandboxRunner().available(),
    reason="macOS sandbox-exec or .venv-numeric python unavailable",
)


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]),
            eng, sup))


def _await(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestNumericParamsEndpoint:
    def test_scalar_expression_lists_free_symbols(self):
        nb.start_engine(title="np1", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            r = _await(nb.numeric_params("E1"))
            assert r["ok"] is True
            # exactly the symbols the template will demand (one becomes the
            # sweep var, the rest need fixed values)
            assert r["parameters"] == ["a", "k", "x"]
        finally:
            nb.stop_engine()

    def test_matrix_canonical_is_rejected_with_guidance(self):
        nb.start_engine(title="np2", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "M = [1, 2; 3, 4]")
            r = _await(nb.numeric_params("E1"))
            assert r["ok"] is False
            assert r["detail"]
        finally:
            nb.stop_engine()

    def test_display_map_renders_greek_and_subscripts(self):
        """The modal renders each fixed-value label via this map instead of
        the bare canonical identifier — gamma/omega_0 must come back as
        LaTeX (\\gamma, \\omega_{0}), not literal text."""
        nb.start_engine(title="np_disp", mode="standard", language="ja", domains=[])
        try:
            eng = nb.get_engine()
            s = _sup()
            _run(eng, s, "\\gamma を正の実数とする")
            _run(eng, s, "\\omega_0 を正の実数とする")
            _run(eng, s, "f = gamma * omega_0 * exp(-t)")
            eid = [k for k, e in eng.expressions.items()
                   if "exp" in (e.canonical or "")][-1]
            r = _await(nb.numeric_params(eid))
            assert r["ok"] is True
            assert r["display"]["gamma"] == r"\gamma"
            assert r["display"]["omega_0"] == r"\omega_{0}"
            assert r["display"]["t"] == "t"          # no Greek form: unchanged
        finally:
            nb.stop_engine()

    def test_unknown_expression_is_rejected(self):
        nb.start_engine(title="np3", mode="standard", language="en", domains=[])
        try:
            r = _await(nb.numeric_params("E99"))
            assert r["ok"] is False
        finally:
            nb.stop_engine()

    def test_unlowerable_expression_reports_reason(self):
        nb.start_engine(title="np4", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            # gamma() is not in the numpy whitelist → template can't lower it
            _run(eng, _sup(), "z = gamma(x)")
            r = _await(nb.numeric_params("E1"))
            assert r["ok"] is False
            assert r["detail"]
        finally:
            nb.stop_engine()


class TestNumericForkFromSpec:
    """POST /numeric/fork with a `spec` body (docs/24/25): the faceted spec
    takes precedence over the flat fields, and the response's codegen_route
    tells the caller which generate endpoint to call next without
    re-deriving that dispatch client-side."""

    def test_flat_representable_spec_routes_deterministic(self):
        nb.start_engine(title="forkspec1", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            spec = {
                "intent": {"kind": "parameter_sweep", "goal": "sweep x"},
                "inputs": [
                    {"name": "x", "role": "sweep", "spec": {"min": 0, "max": 10, "points": 50}},
                    {"name": "k", "role": "fixed", "spec": {"value": 2}},
                    {"name": "a", "role": "fixed", "spec": {"value": 0.3}},
                ],
                "outputs": [{"name": "y", "shape": "series", "over": ["x"]}],
            }
            r = _await(nb.numeric_fork(NumericForkRequest(expr_id="E1", spec=spec)))
            assert r["ok"] is True
            assert r["codegen_route"] == "deterministic"
            assert r["task"]["spec"]["intent"]["kind"] == "parameter_sweep"
            # flat fields are DERIVED from the spec (so numeric_generate works)
            assert r["task"]["sweep_parameters"] == {"x": {"min": 0, "max": 10, "points": 50}}
        finally:
            nb.stop_engine()

    def test_non_flat_spec_routes_llm(self):
        nb.start_engine(title="forkspec2", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            spec = {
                "intent": {"kind": "root_finding"},
                "inputs": [{"name": "x", "role": "free", "spec": {"initial": 0.5}},
                          {"name": "k", "role": "fixed", "spec": {"value": 2}}],
                "outputs": [{"name": "x_star", "shape": "scalar"}],
                "method": {"name": "fixed_point", "params": {"tol": 1e-8, "max_iter": 100}},
            }
            r = _await(nb.numeric_fork(NumericForkRequest(expr_id="E1", spec=spec)))
            assert r["ok"] is True
            assert r["codegen_route"] == "llm"
        finally:
            nb.stop_engine()

    def test_invalid_spec_rejected(self):
        nb.start_engine(title="forkspec3", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            spec = {"inputs": [{"name": "x", "role": "sweep",
                               "spec": {"min": 5, "max": 1, "points": 10}}],
                   "outputs": [{"name": "y", "over": ["x"]}]}
            r = _await(nb.numeric_fork(NumericForkRequest(expr_id="E1", spec=spec)))
            assert r["ok"] is False
        finally:
            nb.stop_engine()

    def test_no_spec_defaults_to_deterministic_route(self):
        # legacy flat fork (no spec) still reports a route, for a uniform
        # client-side branch regardless of which entry path was used
        nb.start_engine(title="forkspec4", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            r = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 1, "points": 10}},
                fixed_values={"k": "2"})))
            assert r["ok"] is True
            assert r["codegen_route"] == "deterministic"
        finally:
            nb.stop_engine()


class TestNumericAssumptionGuard:
    """(a) A fork whose fixed values violate a snapshotted session assumption
    (e.g. the underdamped condition D>0) is refused with a localized message,
    so the researcher re-enters instead of getting a NaN plot."""

    def _setup(self, sup):
        # Declare the parameters, define D = 4ω₀²-(2γ)², assume D>0, and make a
        # fork target that carries both γ and ω₀ (so both are asked for).
        nb.start_engine(title="asmguard", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        _run(eng, sup, "Define gamma as a real number")
        _run(eng, sup, "Define omega_0 as a real number")
        _run(eng, sup, "D = 4*omega_0^2 - (2*gamma)^2")
        _run(eng, sup, "assume D > 0")
        _run(eng, sup, "f = gamma * omega_0 * exp(-t)")
        # find the fork target (the f expression)
        eid = [k for k, e in eng.expressions.items()
               if "exp" in (e.canonical or "")][-1]
        return eng, eid

    def test_fork_blocked_when_values_violate_assumption(self):
        s = _sup()
        eng, eid = self._setup(s)
        try:
            r = _await(nb.numeric_fork(NumericForkRequest(
                expr_id=eid, target_quantity="f",
                sweep_parameters={"t": {"min": 0, "max": 5, "points": 20}},
                fixed_values={"gamma": "3", "omega_0": "2"})))   # D = -20 < 0
            assert r["ok"] is False
            assert "D > 0" in r["detail"]
        finally:
            nb.stop_engine()

    def test_fork_allowed_when_assumption_satisfied(self):
        s = _sup()
        eng, eid = self._setup(s)
        try:
            r = _await(nb.numeric_fork(NumericForkRequest(
                expr_id=eid, target_quantity="f",
                sweep_parameters={"t": {"min": 0, "max": 5, "points": 20}},
                fixed_values={"gamma": "0.3", "omega_0": "2"})))  # D = 15.64 > 0
            assert r["ok"] is True
        finally:
            nb.stop_engine()


@needs_sandbox
class TestNumericFullLoopBridge:
    """fork → generate → run through the bridge, then verify the timeline
    serializes the numeric steps and the figure route serves the plot."""

    def test_loop_refluxes_figure_and_timeline_entry(self, tmp_path):
        nb.start_engine(title="loop", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            # anchor numeric output to a real saved path
            path = tmp_path / "loop.json"
            eng.save(path)
            session_io._current_filename = str(path)

            fork = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", problem_type="parameter_sweep",
                target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 10, "points": 50}},
                fixed_values={"k": "2", "a": "0.3"})))
            assert fork["ok"]
            tid = fork["task"]["task_id"]

            gen = _await(nb.numeric_generate(tid))
            assert gen["ok"] and "np.linspace" in gen["code"]

            run = _await(nb.numeric_run(tid))
            assert run["ok"], run
            assert run["figure_artifact_id"]
            assert run["result_text"]

            # timeline serializes numeric_fork + numeric_result
            tl = _await(nb.get_timeline())
            types = [e["type"] for e in tl["entries"]]
            assert "numeric_fork" in types
            nres = [e for e in tl["entries"] if e["type"] == "numeric_result"]
            assert nres and nres[0]["ok"] is True
            assert nres[0]["figure_artifact_id"] == run["figure_artifact_id"]

            # figure route serves the PNG
            resp = _await(nb.numeric_figure(run["figure_artifact_id"]))
            from fastapi.responses import FileResponse
            assert isinstance(resp, FileResponse)
        finally:
            nb.stop_engine()

    def test_figure_route_rejects_unknown_id(self):
        from fastapi import HTTPException
        nb.start_engine(title="figguard", mode="standard", language="en", domains=[])
        try:
            with pytest.raises(HTTPException) as ei:
                _await(nb.numeric_figure("na_doesnotexist"))
            assert ei.value.status_code == 404
        finally:
            nb.stop_engine()

    def test_nan_result_is_flagged_not_clean_success(self, tmp_path):
        """(b) NaN net: a swept expression that leaves its valid regime
        (sqrt of a negative) exits 0 but yields NaN — the run must be flagged
        (ok=False, figure suppressed) and name the parameter range, not report
        a clean success with a blank plot."""
        nb.start_engine(title="nan", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sqrt(1 - x)")
            eng.save(tmp_path / "nan.json")
            session_io._current_filename = str(tmp_path / "nan.json")
            fork = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 5, "points": 30}},
                fixed_values={})))
            tid = fork["task"]["task_id"]
            _await(nb.numeric_generate(tid))
            run = _await(nb.numeric_run(tid))
            assert run["ok"] is False            # flagged, not clean success
            assert run["figure_artifact_id"] is None   # blank plot suppressed
            assert "NaN" in run["result_text"] or "nan" in run["result_text"].lower()
        finally:
            nb.stop_engine()

    def test_figure_still_serves_after_save_as(self, tmp_path):
        """End-to-end: run (auto-saves under a throwaway stem) → Save As →
        the figure route must still serve (outputs relocated to the new stem).
        This is the browser save-experiment the relocation fix targets."""
        nb.start_engine(title="saveas_e2e", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            eng.save(tmp_path / "auto.json")
            session_io._current_filename = str(tmp_path / "auto.json")
            fork = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 10, "points": 40}},
                fixed_values={"k": "2", "a": "0.3"})))
            tid = fork["task"]["task_id"]
            _await(nb.numeric_generate(tid))
            run = _await(nb.numeric_run(tid))
            figid = run["figure_artifact_id"]
            # Save As a chosen name
            _await(session_io.do_save(eng, filename=str(tmp_path / "chosen.json")))
            # figure still serves, and its file now lives under chosen_numeric/
            from fastapi.responses import FileResponse
            resp = _await(nb.numeric_figure(figid))
            assert isinstance(resp, FileResponse)
            assert "chosen_numeric" in eng.numeric_artifacts[figid].path
        finally:
            nb.stop_engine()


class TestNumericRunTimeoutOverride:
    """POST /numeric/run/{task_id} threads an optional timeout_s override
    through to SandboxRunner, falling back to the configured default
    (config/default_settings.json numeric_timeout_s) when omitted."""

    @needs_sandbox
    def test_default_timeout_used_when_omitted(self, tmp_path, monkeypatch):
        from notebook.numeric import sandbox as sandbox_mod
        import ui.routes.numeric as numeric_routes

        monkeypatch.setattr(numeric_routes, "get_default_numeric_timeout_s", lambda: 42.0)
        seen = {}
        real_init = sandbox_mod.SandboxRunner.__init__

        def _spy_init(self, python_path=None, timeout_s=60.0):
            seen["timeout_s"] = timeout_s
            real_init(self, python_path=python_path, timeout_s=timeout_s)

        monkeypatch.setattr(sandbox_mod.SandboxRunner, "__init__", _spy_init)

        nb.start_engine(title="to1", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            eng.save(tmp_path / "to1.json")
            session_io._current_filename = str(tmp_path / "to1.json")
            fork = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 5, "points": 10}},
                fixed_values={"k": "2"})))
            tid = fork["task"]["task_id"]
            _await(nb.numeric_generate(tid))
            _await(nb.numeric_run(tid))
            assert seen["timeout_s"] == 42.0
        finally:
            nb.stop_engine()

    @needs_sandbox
    def test_explicit_timeout_overrides_default(self, tmp_path, monkeypatch):
        from notebook.numeric import sandbox as sandbox_mod

        seen = {}
        real_init = sandbox_mod.SandboxRunner.__init__

        def _spy_init(self, python_path=None, timeout_s=60.0):
            seen["timeout_s"] = timeout_s
            real_init(self, python_path=python_path, timeout_s=timeout_s)

        monkeypatch.setattr(sandbox_mod.SandboxRunner, "__init__", _spy_init)

        nb.start_engine(title="to2", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            eng.save(tmp_path / "to2.json")
            session_io._current_filename = str(tmp_path / "to2.json")
            fork = _await(nb.numeric_fork(NumericForkRequest(
                expr_id="E1", target_quantity="y",
                sweep_parameters={"x": {"min": 0, "max": 5, "points": 10}},
                fixed_values={"k": "2"})))
            tid = fork["task"]["task_id"]
            _await(nb.numeric_generate(tid))
            _await(nb.numeric_run(tid, NumericRunRequest(timeout_s=5.0)))
            assert seen["timeout_s"] == 5.0
        finally:
            nb.stop_engine()


_GOOD_SPEC_REPLY = json.dumps({
    "intent": {"kind": "parameter_sweep", "goal": "sweep x"},
    "inputs": [
        {"name": "x", "role": "sweep", "spec": {"min": 0, "max": 10, "points": 100}},
        {"name": "k", "role": "fixed", "spec": {"value": 2}},
        {"name": "a", "role": "fixed", "spec": {"value": 0.3}},
    ],
    "outputs": [{"name": "y", "shape": "series", "over": ["x"], "role": "primary"}],
    "method": {"name": "direct_eval", "params": {}},
})


class TestNumericProposeSpec:
    """POST /numeric/propose_spec (docs/25 §3, B-γ turn) — the Adviser LLM is
    resolved internally via ui.interpret_plan._resolve_adviser_chat, so tests
    monkeypatch that resolver rather than injecting a chat callable directly
    (matching how the endpoint actually calls it in production)."""

    def _patch_adviser(self, monkeypatch, reply):
        import ui.interpret_plan as interpret_plan
        monkeypatch.setattr(interpret_plan, "_resolve_adviser_chat",
                            lambda: (lambda system, user: reply))

    def test_first_turn_returns_spec_and_issues(self, monkeypatch):
        self._patch_adviser(monkeypatch, _GOOD_SPEC_REPLY)
        nb.start_engine(title="propose1", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            r = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E1", goal="sweep x")))
            assert r["ok"] is True
            assert r["spec"]["intent"]["kind"] == "parameter_sweep"
            assert r["issues"] == []
        finally:
            nb.stop_engine()

    def test_refine_turn_threads_prior_spec_and_correction(self, monkeypatch):
        seen = {}

        def spy_chat(system, user):
            seen["user"] = user
            return _GOOD_SPEC_REPLY

        import ui.interpret_plan as interpret_plan
        monkeypatch.setattr(interpret_plan, "_resolve_adviser_chat", lambda: spy_chat)
        nb.start_engine(title="propose2", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            prior = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E1", goal="sweep x")))
            r = _await(nb.numeric_propose_spec(nb.NumericProposeSpecRequest(
                expr_id="E1", goal="sweep x", prior_spec=prior["spec"],
                refinement="sweep k instead")))
            assert r["ok"] is True
            assert "sweep k instead" in seen["user"]
            assert "Previous proposal" in seen["user"]
        finally:
            nb.stop_engine()

    def test_unknown_expr_id_rejected(self, monkeypatch):
        self._patch_adviser(monkeypatch, _GOOD_SPEC_REPLY)
        nb.start_engine(title="propose3", mode="standard", language="en", domains=[])
        try:
            r = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E99", goal="x")))
            assert r["ok"] is False
        finally:
            nb.stop_engine()

    def test_adviser_unavailable_reported(self, monkeypatch):
        import ui.interpret_plan as interpret_plan
        monkeypatch.setattr(interpret_plan, "_resolve_adviser_chat", lambda: None)
        nb.start_engine(title="propose4", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            r = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E1", goal="x")))
            assert r["ok"] is False
            assert r["detail"]
        finally:
            nb.stop_engine()

    def test_unparseable_reply_reported_not_500(self, monkeypatch):
        self._patch_adviser(monkeypatch, "not json at all")
        nb.start_engine(title="propose5", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x) * exp(-a*x)")
            r = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E1", goal="x")))
            assert r["ok"] is False
            assert r["detail"]
        finally:
            nb.stop_engine()

    def test_spec_with_gatekeeper_issues_still_returned(self, monkeypatch):
        # "k"/"a" here are not free symbols of a bare "sin(x)" expression
        self._patch_adviser(monkeypatch, _GOOD_SPEC_REPLY)
        nb.start_engine(title="propose6", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(x)")
            r = _await(nb.numeric_propose_spec(
                nb.NumericProposeSpecRequest(expr_id="E1", goal="x")))
            assert r["ok"] is True
            assert r["spec"] is not None
            # issues are translated server-side (same policy as every other
            # endpoint) — the response carries display strings, not raw codes
            assert any("k" in msg for msg in r["issues"])
        finally:
            nb.stop_engine()


class TestNumericApproveEdit:
    """POST /numeric/approve/{artifact_id} with an edited_source body
    (docs/25 §5 decision 3): the edit must become a NEW approved artifact
    that run_numeric_task actually picks up and executes."""

    def _fork_and_generate_llm(self, eng, chat):
        # Goes through the engine directly (not the bridge's numeric_generate_llm,
        # which resolves a REAL coder endpoint) — these tests are about
        # /numeric/approve's edit-handling, not about coder resolution.
        fork = _await(nb.numeric_fork(NumericForkRequest(
            expr_id="E1", target_quantity="y",
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 50}},
            fixed_values={"k": "2"})))
        tid = fork["task"]["task_id"]
        art = eng.generate_numeric_code_llm(tid, chat=chat)
        return tid, art.artifact_id

    def test_approve_without_edit_keeps_original_content(self):
        code = ('import json, os\n'
                'OUT = os.environ.get("NB_NUMERIC_OUT", ".")\n'
                'print("NB_RESULT", json.dumps({"target":"y","sweep_var":"x",'
                '"n_points":1,"min":{"x":0,"y":0},"max":{"x":0,"y":0}}))\n')
        nb.start_engine(title="approve1", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            tid, aid = self._fork_and_generate_llm(
                eng, lambda s, u: f"```python\n{code}\n```")
            r = _await(nb.numeric_approve(aid))
            assert r["ok"] is True
            assert r["edited"] is False
            approved = eng.numeric_artifacts[r["artifact_id"]]
            assert approved.content == eng.numeric_artifacts[aid].content
            assert approved.artifact_id != aid   # a NEW artifact, per docs/25 §5
        finally:
            nb.stop_engine()

    def test_approve_with_edit_creates_new_artifact_with_edited_content(self):
        original = ('import json\n'
                    'print("NB_RESULT", json.dumps({"target":"y","sweep_var":"x",'
                    '"n_points":1,"min":{"x":0,"y":0},"max":{"x":0,"y":0}}))\n')
        edited = original.replace("y", "y_edited")
        nb.start_engine(title="approve2", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            tid, aid = self._fork_and_generate_llm(
                eng, lambda s, u: f"```python\n{original}\n```")
            r = _await(nb.numeric_approve(
                aid, nb.NumericApproveRequest(edited_source=edited)))
            assert r["ok"] is True
            assert r["edited"] is True
            approved = eng.numeric_artifacts[r["artifact_id"]]
            assert approved.content == edited
            assert approved.metadata.get("approved_from") == aid
        finally:
            nb.stop_engine()

    @needs_sandbox
    def test_edited_code_actually_runs(self, tmp_path):
        original = 'raise SystemExit("should never run")\n'
        edited = ('import json, os\n'
                  'OUT = os.environ.get("NB_NUMERIC_OUT", ".")\n'
                  'print("NB_RESULT", json.dumps({"target":"y","sweep_var":"x",'
                  '"n_points":1,"min":{"x":0.0,"y":0.0},"max":{"x":0.0,"y":0.0}}))\n')
        nb.start_engine(title="approve3", mode="standard", language="en", domains=[])
        try:
            eng = nb.get_engine()
            _run(eng, _sup(), "y = sin(k*x)")
            eng.save(tmp_path / "approve3.json")
            session_io._current_filename = str(tmp_path / "approve3.json")
            tid, aid = self._fork_and_generate_llm(
                eng, lambda s, u: f"```python\n{original}\n```")
            _await(nb.numeric_approve(aid, nb.NumericApproveRequest(edited_source=edited)))
            run = _await(nb.numeric_run(tid))
            assert run["ok"], run   # the EDITED code ran, not the original
        finally:
            nb.stop_engine()
