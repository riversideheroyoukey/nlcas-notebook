"""NotebookEngine mixin: the numeric layer (N-1 phase-1 vertical slice).

theory expression → fork_numeric_task (provenance snapshot)
                 → generate_numeric_code (deterministic template)
                 → run_numeric_task (sandbox-exec, net-denied, scratch-only)
                 → numeric_result step refluxed into the timeline.

Follows the EngineHistoryMixin pattern: state via self, NotebookEngine only
in annotations, no import from engine.py.  The LLM is deliberately absent
from this slice — spec elicitation and result interpretation are later
phases; code generation itself stays deterministic (core principle #3).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from notebook.models.numeric import (
    NumericArtifact, NumericExecution, NumericSpec, NumericTask,
)
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.i18n import t
from notebook.runtime.shared_parser import UnknownNotationError, strip_lhs

_RESULT_LINE_RE = re.compile(r"^NB_RESULT (\{.*\})\s*$", re.M)
# Typed manifest line (docs/26 §4) — read IN ADDITION to NB_RESULT, so old
# code artifacts (which only print NB_RESULT) keep working unchanged.
_MANIFEST_LINE_RE = re.compile(r"^NB_MANIFEST (\{.*\})\s*$", re.M)
# Comparison operators in an assumption canonical (">=" before ">" so the
# two-char forms win).  Equalities are not enforced (not an inequality).
_ASM_OP_RE = re.compile(r"(>=|<=|>|<)")


def _fmt_num(x: float) -> str:
    """Compact numeric label for user messages (trim trailing zeros)."""
    import math
    if not math.isfinite(x):
        return str(x)          # inf/nan: a diverged loop's residual
    if x == int(x):
        return str(int(x))
    return f"{x:.4g}"


# Scalar stand-in for `np` when evaluating an assumption inequality in-process.
# The numeric interpreter (.venv-numeric) has numpy, but THIS app process
# (.venv) does not — so the assumption check maps the whitelist-lowered
# `np.<fn>` calls to `math` equivalents.  A function outside this shim (or a
# domain error like sqrt of a negative) raises → the check treats the
# assumption as un-evaluable and skips it (the run-time NaN net still covers
# that parameter point).
def _make_scalar_np():
    import math
    from types import SimpleNamespace
    return SimpleNamespace(
        pi=math.pi, e=math.e,
        sin=math.sin, cos=math.cos, tan=math.tan,
        arcsin=math.asin, arccos=math.acos, arctan=math.atan, arctan2=math.atan2,
        sinh=math.sinh, cosh=math.cosh, tanh=math.tanh,
        exp=math.exp, log=math.log, sqrt=math.sqrt,
        abs=abs, floor=math.floor, round=round,
        sign=lambda x: (x > 0) - (x < 0),
    )


_SCALAR_NP = _make_scalar_np()


def _summary_has_nan(summary: dict) -> bool:
    """True if the NB_RESULT min/max — or an evaluate template's single
    "value" — carry a NaN (the expression left its valid regime, e.g. sqrt
    of a negative discriminant)."""
    import math

    def _bad(v) -> bool:
        return isinstance(v, float) and math.isnan(v)

    if summary.get("has_nan") is True:     # matrix evaluate declares it itself
        return True
    for key in ("min", "max"):
        pt = summary.get(key) or {}
        if any(_bad(pt.get(coord)) for coord in ("x", "y")):
            return True
    if _bad(summary.get("argmin")) or _bad(summary.get("minimum")):
        return True
    val = summary.get("value")
    if _bad(val):
        return True
    if isinstance(val, dict) and (_bad(val.get("re")) or _bad(val.get("im"))):
        return True
    return False


class EngineNumericMixin:
    """Numeric-layer operations for NotebookEngine."""

    # ------------------------------------------------------------------
    # Fork: theory expression → NumericTask
    # ------------------------------------------------------------------

    def _alloc_numeric_id(self) -> str:
        # Zero-padded so lexicographic and numeric sort agree past N9 → N10
        # (directory listings, UI dropdowns, log greps all sort as strings).
        self._numeric_counter += 1
        return f"N{self._numeric_counter:03d}"

    def fork_numeric_task(self, expr_id: str,
                          problem_type: str = "parameter_sweep",
                          target_quantity: str = "",
                          sweep_parameters: "dict | None" = None,
                          fixed_values: "dict | None" = None,
                          approximations: "list[str] | None" = None,
                          created_by: str = "user",
                          spec: "NumericSpec | None" = None) -> NumericTask:
        """Create a NumericTask from a theory expression, snapshotting its
        canonical/latex and the session's assumptions (provenance survives
        later edits to the theory layer).

        When a faceted ``spec`` (docs/24) is given it is validated and attached,
        and the flat fields (problem_type/target_quantity/fixed_values/
        sweep_parameters) are DERIVED from it via ``spec_to_flat`` so the
        existing deterministic template and the assumption guard keep working
        unchanged.  For a non-flat spec (convergence/eigen) the derived flat
        fields are best-effort/empty — the task still carries the full spec for
        the LLM codegen path.
        """
        expr = self.expressions.get(expr_id)
        if expr is None:
            raise ValueError(t("numeric.task_source_missing", self.lang,
                               expr_id=expr_id))
        # A list/matrix canonical ([[λ1,λ2],[…]] from eigenvalues, matrix
        # literals, solve raw results) is not a scalar expression.  Without
        # this early check the lowering rejects it too, but with a confusing
        # "Commutator unsupported" message (a 2-element bracket resolves to a
        # commutator) — found while verifying scenario S-y.
        if (expr.canonical or "").lstrip().startswith("["):
            raise ValueError(t("numeric.list_not_supported", self.lang,
                               expr_id=expr_id))
        # Faceted-spec fork: resolve artifact bindings (docs/26 §5) FIRST so
        # the resolved numeric values pass the literal checks, then validate,
        # then project onto the flat fields the assumption guard and
        # deterministic path consume.
        input_artifact_refs: list[str] = []
        source_body: list[dict] = []
        plot_sources: list[dict] = []
        if spec is not None:
            from notebook.numeric.spec_projection import spec_to_flat
            from notebook.numeric.spec_validate import validate_numeric_spec
            spec, input_artifact_refs = self._resolve_value_from_bindings(spec)
            # Loop body (docs/26 §8): snapshot each body expression's
            # canonical NOW, like source_canonical — provenance survives
            # later edits.  Also widen the free-symbol universe to the whole
            # body: a loop's fixed/free inputs may only appear in a body
            # expression, not in the primary one.
            free_syms = self._numeric_free_symbols(expr.canonical)
            if spec.intent.kind == "self_consistent_loop":
                body_entries = (spec.method.params or {}).get("body") or []
                for entry in body_entries:
                    if not isinstance(entry, dict):
                        continue
                    beid = str(entry.get("expr_id", ""))
                    bexpr = self.expressions.get(beid)
                    if bexpr is None:
                        raise ValueError(t("numeric.task_source_missing",
                                           self.lang, expr_id=beid))
                    source_body.append({"expr_id": beid,
                                        "name": str(entry.get("name", "")),
                                        "canonical": bexpr.canonical})
                if free_syms is not None:
                    for b in source_body:
                        s = self._numeric_free_symbols(b["canonical"])
                        if s is None:
                            free_syms = None
                            break
                        free_syms = free_syms | s | {b["name"]}
            from notebook.numeric.plot_spec import is_cartesian, snapshot_plot_sources
            if is_cartesian(spec):
                try:
                    plot_sources, plot_free = snapshot_plot_sources(self, spec)
                except UnknownNotationError as exc:
                    raise self._translate_unknown_notation(exc) from exc
                free_syms = plot_free | (free_syms or set())
            issues = validate_numeric_spec(spec, free_symbols=free_syms)
            if issues:
                first = issues[0]
                raise ValueError(t(first["code"], self.lang,
                                   **{k: v for k, v in first.items() if k != "code"}))
            flat = spec_to_flat(spec)
            problem_type = flat["problem_type"]
            target_quantity = flat["target_quantity"]
            fixed_values = flat["fixed_values"]
            sweep_parameters = flat["sweep_parameters"]
        # (a) Assumption guard: the expression may have been DERIVED under a
        # session assumption (e.g. the damped-oscillator general solution is
        # only valid for D=4ω₀²-(2γ)²>0, underdamped).  If the chosen fixed
        # values definitively violate such an assumption, refuse the fork and
        # ask the researcher to re-enter — don't silently produce a formula
        # evaluated outside its regime (which would go NaN, see the run-time
        # NaN net).  Assumptions that involve the swept variable, or symbols
        # with no value yet, can't be checked pointwise here and are skipped.
        _viol = self._numeric_assumption_violations(
            {k: str(v) for k, v in (fixed_values or {}).items()},
            list((sweep_parameters or {}).keys()))
        if _viol:
            disp, vals, computed = _viol[0]
            raise ValueError(t("numeric.assumption_violated", self.lang,
                               values=vals, assumption=disp,
                               computed=_fmt_num(computed)))
        task = NumericTask(
            task_id=self._alloc_numeric_id(),
            source_expression_ids=[expr_id] + [
                b["expr_id"] for b in source_body + plot_sources if b["expr_id"] != expr_id],
            source_canonical=expr.canonical,
            source_latex=expr.latex,
            source_assumption_ids=sorted(self.assumptions.keys()),
            source_branch_id=self.active_branch_id,
            problem_type=problem_type,
            target_quantity=target_quantity,
            fixed_values={k: str(v) for k, v in (fixed_values or {}).items()},
            sweep_parameters=dict(sweep_parameters or {}),
            approximations=list(approximations or []),
            created_by=created_by,
            spec=spec,
            input_artifact_refs=input_artifact_refs,
            source_hash=self._numeric_source_hash(expr.canonical),
            source_body=source_body,
            plot_sources=plot_sources,
        )
        self.numeric_tasks[task.task_id] = task

        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.numeric_fork,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("numeric.forked", self.lang,
                             task_id=task.task_id, expr_id=expr_id),
            operation="fork_numeric_task",
            operation_args={"task_id": task.task_id,
                            "problem_type": problem_type},
            input_refs=task.source_expression_ids,
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return task

    # ------------------------------------------------------------------
    # Deterministic code generation
    # ------------------------------------------------------------------

    def generate_numeric_code(self, task_id: str) -> NumericArtifact:
        task = self._numeric_task(task_id)
        from notebook.numeric.templates import generate_code
        try:
            source, meta = generate_code(task, self.symbol_context, lang=self.lang)
        except UnknownNotationError as exc:
            raise self._translate_unknown_notation(exc) from exc
        artifact = NumericArtifact(
            task_id=task.task_id, artifact_type="code",
            content=source, metadata=meta,
        )
        self.numeric_artifacts[artifact.artifact_id] = artifact
        self.numeric_tasks[task.task_id] = task.model_copy(
            update={"status": "code_generated"})
        return artifact

    def generate_numeric_code_llm(self, task_id: str, chat=None,
                                  ) -> NumericArtifact:
        """Vibe-coding path: the coding LLM writes the script from the
        TeX/canonical form.  This is the GENERAL path — it works where the
        deterministic template cannot (whitelist-external functions, Julia
        opaque canonicals via the latex snapshot).

        The artifact is stored UNREVIEWED: run_numeric_task refuses to
        execute it until approve_numeric_code() — researcher supervision is
        enforced engine-side, not left to UI convention."""
        from notebook.numeric.llm_codegen import generate_llm_code

        task = self._numeric_task(task_id)
        if chat is None:
            raise ValueError(t("numeric.coder_unavailable", self.lang))

        # Offer the deterministic template as a scaffold when it exists —
        # and remember it for the differential check.
        scaffold = None
        try:
            from notebook.numeric.templates import generate_code
            scaffold, _ = generate_code(task, self.symbol_context, lang=self.lang)
        except UnknownNotationError:
            scaffold = None    # whitelist-external: the vibe path's home turf

        try:
            source, meta = generate_llm_code(chat, task,
                                             latex=task.source_latex,
                                             scaffold=scaffold)
        except UnknownNotationError as exc:
            raise self._translate_unknown_notation(exc) from exc
        if scaffold is not None:
            meta["template_scaffold"] = scaffold
        artifact = NumericArtifact(
            task_id=task.task_id, artifact_type="code",
            content=source, metadata=meta,
        )
        self.numeric_artifacts[artifact.artifact_id] = artifact
        self.numeric_tasks[task.task_id] = task.model_copy(
            update={"status": "code_generated"})
        return artifact

    def approve_numeric_code(self, artifact_id: str,
                             edited_source: "str | None" = None) -> NumericArtifact:
        """The researcher's explicit review approval of LLM-generated code.

        docs/25 §5 decision 3: the code panel is editable, and an edit must
        actually run — so approval creates a NEW artifact carrying the
        (possibly edited) source, rather than mutating the reviewed artifact
        in place.  The new artifact has a fresher ``created_at``, so
        ``_latest_code_artifact``/``run_numeric_task`` pick it up automatically.
        ``approved_from`` keeps the edit's provenance; ``edited`` records
        whether the source actually changed.
        """
        art = self.numeric_artifacts.get(artifact_id)
        if art is None or art.artifact_type != "code":
            raise ValueError(t("numeric.no_code", self.lang,
                               artifact_id=artifact_id))
        content = art.content if edited_source is None else edited_source
        approved = NumericArtifact(
            task_id=art.task_id, artifact_type="code", content=content,
            metadata={**art.metadata, "reviewed": True,
                     "edited": content != art.content,
                     "approved_from": art.artifact_id},
        )
        self.numeric_artifacts[approved.artifact_id] = approved
        return approved

    def differential_check_llm_code(self, task_id: str,
                                    timeout_s: float = 60.0) -> dict:
        """Run the latest LLM code AND the template side by side, compare
        their NB_RESULT summaries.  Only possible for whitelist-covered
        tasks — exactly where a deterministic ground truth exists."""
        from notebook.numeric.llm_codegen import differential_check
        from notebook.numeric.sandbox import SandboxRunner

        task = self._numeric_task(task_id)
        code_art = self._latest_code_artifact(task_id)
        if code_art is None or code_art.metadata.get("source") != "llm":
            raise ValueError(t("numeric.no_code", self.lang,
                               artifact_id=task_id))
        template = code_art.metadata.get("template_scaffold")
        if not template:
            from notebook.numeric.templates import generate_code
            template, _ = generate_code(task, self.symbol_context, lang=self.lang)
        runner = SandboxRunner(timeout_s=timeout_s)
        if not runner.available():
            raise ValueError(t("numeric.sandbox_unavailable", self.lang,
                               python=runner.python_path))
        workdir = self._numeric_output_root() / task.task_id / "diffcheck"
        verdict = differential_check(runner, code_art.content, template,
                                     workdir)
        self.numeric_artifacts[code_art.artifact_id] = code_art.model_copy(
            update={"metadata": {**code_art.metadata,
                                 "differential": {"match": verdict["match"]}}})
        return verdict

    # ------------------------------------------------------------------
    # Sandbox execution + reflux
    # ------------------------------------------------------------------

    def run_numeric_task(self, task_id: str,
                         timeout_s: float = 60.0) -> NumericExecution:
        """Run the task's latest code artifact in the sandbox, record the
        execution + result artifacts, and reflux a numeric_result step into
        the timeline (with full provenance in operation_args)."""
        from notebook.numeric.sandbox import SandboxRunner

        task = self._numeric_task(task_id)
        code_art = self._latest_code_artifact(task_id)
        if code_art is None:
            code_art = self.generate_numeric_code(task_id)
            task = self._numeric_task(task_id)
        # Review gate (vibe-coding path): LLM-generated code cannot run until
        # the researcher approved it.  Template code carries no such flag —
        # it is deterministic and needs no review.
        if (code_art.metadata.get("source") == "llm"
                and not code_art.metadata.get("reviewed")):
            raise ValueError(t("numeric.code_not_reviewed", self.lang,
                               artifact_id=code_art.artifact_id))

        runner = SandboxRunner(timeout_s=timeout_s)
        if not runner.available():
            raise ValueError(t("numeric.sandbox_unavailable", self.lang,
                               python=runner.python_path))

        execution = NumericExecution(
            task_id=task.task_id,
            code_artifact_id=code_art.artifact_id,
            runtime=runner.python_path,
            package_info=runner.package_info(),
            parameter_snapshot={"fixed_values": task.fixed_values,
                                "sweep_parameters": task.sweep_parameters},
        )
        workdir = self._numeric_output_root() / task.task_id / execution.execution_id
        workdir.mkdir(parents=True, exist_ok=True)
        code_path = workdir / "code.py"
        code_path.write_text(code_art.content)

        sb = runner.run_file(code_path, workdir)
        execution.stdout = sb.stdout
        execution.stderr = sb.stderr
        execution.return_status = sb.returncode
        execution.timed_out = sb.timed_out
        execution.duration_s = round(sb.duration_s, 3)

        # File artifacts (figures / tables) created inside the work dir
        result_ids: list[str] = []
        for name in sb.created_files:
            if name == "code.py":
                continue
            kind = ("figure" if name.endswith(".png")
                    else "table" if name.endswith(".csv") else "log")
            art = NumericArtifact(
                task_id=task.task_id, artifact_type=kind,
                path=str(workdir / name),
                metadata={"execution_id": execution.execution_id},
            )
            self.numeric_artifacts[art.artifact_id] = art
            result_ids.append(art.artifact_id)

        # Machine-readable summary printed by the template
        summary: dict = {}
        m = _RESULT_LINE_RE.search(sb.stdout or "")
        if m:
            try:
                summary = json.loads(m.group(1))
            except ValueError:
                summary = {}

        # Typed manifest (docs/26 §4): declared value artifacts + diagnostics.
        # Value entries become typed result artifacts (bindable downstream);
        # file entries attach type info to the file artifact created above.
        diagnostics: dict = {}
        mf = _MANIFEST_LINE_RE.search(sb.stdout or "")
        if mf:
            try:
                manifest = json.loads(mf.group(1))
            except ValueError:
                manifest = {}
            diagnostics = manifest.get("diagnostics") or {}
            for entry in manifest.get("artifacts") or []:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                typed = {
                    "value_kind": str(entry.get("value_kind", "")),
                    "value_type": str(entry.get("value_type", "")),
                    "shape": list(entry.get("shape") or []),
                    "ordering": dict(entry.get("ordering") or {}),
                }
                fname = entry.get("file")
                if fname:
                    for aid in result_ids:
                        a = self.numeric_artifacts[aid]
                        if a.path and Path(a.path).name == fname:
                            self.numeric_artifacts[aid] = a.model_copy(
                                update={**typed,
                                        "metadata": {**a.metadata,
                                                     "name": entry["name"]}})
                            break
                elif "value" in entry:
                    art = NumericArtifact(
                        task_id=task.task_id, artifact_type="result",
                        content=json.dumps(entry, ensure_ascii=False),
                        value=entry["value"], **typed,
                        metadata={"execution_id": execution.execution_id,
                                  "name": entry["name"]},
                    )
                    self.numeric_artifacts[art.artifact_id] = art
                    result_ids.append(art.artifact_id)

        if summary or not sb.ok:
            art = NumericArtifact(
                task_id=task.task_id,
                artifact_type="result" if sb.ok else "log",
                content=json.dumps(summary, ensure_ascii=False)
                        if summary else (sb.stderr or sb.stdout)[-2000:],
                metadata={"execution_id": execution.execution_id},
            )
            self.numeric_artifacts[art.artifact_id] = art
            result_ids.append(art.artifact_id)

        execution.result_artifact_ids = result_ids
        self.numeric_executions[execution.execution_id] = execution

        # (b) NaN net: the sandbox exited 0, but if the swept expression left
        # its valid regime (e.g. sqrt of a negative discriminant when D<0), the
        # values are NaN and the plot is empty.  Surface the offending parameter
        # combination and flag the step as failed rather than reporting a clean
        # success with a blank figure.
        _nan = bool(sb.ok and summary and _summary_has_nan(summary))
        # Honest-diagnostics gate (docs/26 §8): a run that exits 0 but
        # DECLARES failure (non-converged loop, failed optimizer) is a failed
        # step, not a clean success wrapping a garbage number.
        _diag_fail = diagnostics.get("success") is False
        _result_ok = sb.ok and not _nan and not _diag_fail
        self.numeric_tasks[task.task_id] = task.model_copy(
            update={"status": "executed" if _result_ok else "failed"})

        # Reflux: the timeline entry that carries the result AND its provenance
        if _nan:
            _params = ", ".join(f"{k}={v}" for k, v in sorted(task.fixed_values.items()))
            if task.sweep_parameters:
                _sv, _sp = next(iter(task.sweep_parameters.items()))
                _seg = f"{_sv}∈[{_sp.get('min')},{_sp.get('max')}]"
                _params = f"{_params}; {_seg}" if _params else _seg
            action = t("numeric.result_has_nan", self.lang,
                       task_id=task.task_id, params=_params)
        elif sb.ok and summary and "plot_layers" in summary:
            from notebook.numeric.plot_spec import plot_result_text
            action = plot_result_text(task_id, summary, diagnostics, self.lang)
        elif sb.ok and summary and "state" in summary:
            # fused loop: converged state or an honest non-convergence
            _st = ", ".join(f"{k}={_fmt_num(float(v))}"
                            for k, v in (summary.get("state") or {}).items())
            _key = ("numeric.executed_loop" if summary.get("converged")
                    else "numeric.loop_not_converged")
            action = t(_key, self.lang, task_id=task.task_id, state=_st,
                       iterations=summary.get("iterations", "?"),
                       residual=_fmt_num(float(summary.get("residual", 0.0)))
                       if summary.get("residual") is not None else "?")
        elif sb.ok and summary and "eigenvalues" in summary:
            # eigen point mode: list the (small) spectrum in the timeline
            def _fmt_ev(v):
                if isinstance(v, dict):
                    sgn = "+" if v.get("im", 0) >= 0 else "-"
                    return (f"{_fmt_num(v.get('re', 0))}{sgn}"
                            f"{_fmt_num(abs(v.get('im', 0)))}i")
                return _fmt_num(float(v))
            _vals = ", ".join(_fmt_ev(v) for v in summary["eigenvalues"])
            _note = ("" if summary.get("hermitian")
                     else t("numeric.non_hermitian_note", self.lang))
            action = t("numeric.executed_eigen", self.lang,
                       task_id=task.task_id, values=_vals, note=_note)
        elif sb.ok and summary and "n_branches" in summary:
            _note = ("" if summary.get("hermitian")
                     else t("numeric.non_hermitian_note", self.lang))
            action = t("numeric.executed_eigen_sweep", self.lang,
                       task_id=task.task_id,
                       n=summary["n_branches"],
                       sweep_var=summary.get("sweep_var", "?"),
                       lo=_fmt_num(float(summary.get("branch_min", 0))),
                       hi=_fmt_num(float(summary.get("branch_max", 0))),
                       note=_note)
        elif sb.ok and summary and "matrix_shape" in summary:
            _m, _n = (list(summary["matrix_shape"]) + ["?", "?"])[:2]
            action = t("numeric.executed_matrix", self.lang,
                       task_id=task.task_id,
                       target=summary.get("target", "M"), m=_m, n=_n)
        elif sb.ok and summary and "argmin" in summary:
            # minimize template: argmin + minimum, with a loud boundary note
            _bnote = (t("numeric.boundary_note", self.lang)
                      if diagnostics.get("boundary_solution") else "")
            action = t("numeric.executed_minimize", self.lang,
                       task_id=task.task_id,
                       target=summary.get("target", "y"),
                       variable=summary.get("variable", "?"),
                       argmin=_fmt_num(float(summary["argmin"])),
                       minimum=_fmt_num(float(summary["minimum"])),
                       boundary=_bnote)
        elif sb.ok and summary and "value" in summary:
            # evaluate template: a single (possibly complex) value
            _v = summary["value"]
            if isinstance(_v, dict):
                _sign = "+" if _v.get("im", 0) >= 0 else "-"
                _vs = (f"{_fmt_num(_v.get('re', 0))}{_sign}"
                       f"{_fmt_num(abs(_v.get('im', 0)))}i")
            elif isinstance(_v, (int, float)):
                _vs = _fmt_num(_v)
            else:
                _vs = str(_v)
            action = t("numeric.executed_value", self.lang,
                       task_id=task.task_id,
                       target=summary.get("target", "y"), value=_vs)
        elif sb.ok and summary:
            mn, mx = summary.get("min", {}), summary.get("max", {})
            action = t("numeric.executed_summary", self.lang,
                       task_id=task.task_id,
                       target=summary.get("target", "y"),
                       sweep_var=summary.get("sweep_var", "?"),
                       min_y=mn.get("y"), min_x=mn.get("x"),
                       max_y=mx.get("y"), max_x=mx.get("x"))
        elif sb.ok:
            action = t("numeric.executed", self.lang, task_id=task.task_id)
        else:
            action = t("numeric.exec_failed", self.lang,
                       task_id=task.task_id,
                       reason=("timeout" if sb.timed_out
                               else (sb.stderr or "").strip().splitlines()[-1]
                               if (sb.stderr or "").strip() else f"rc={sb.returncode}"))
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.numeric_result,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=action,
            operation="run_numeric_task",
            operation_args={
                "task_id": task.task_id,
                "execution_id": execution.execution_id,
                "source_expression_ids": task.source_expression_ids,
                "source_assumption_ids": task.source_assumption_ids,
                "source_branch_id": task.source_branch_id,
                "approximations": task.approximations,
                "parameter_snapshot": execution.parameter_snapshot,
                "summary": summary,
                "diagnostics": diagnostics,
                "artifact_ids": result_ids,
            },
            input_refs=list(task.source_expression_ids),
            state_affecting=False,
            status=StepStatus.success if _result_ok else StepStatus.failed,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return execution

    # ------------------------------------------------------------------
    # Output location: derived from the saved session JSON
    # ------------------------------------------------------------------
    # Numeric artifacts (executed code copy, figures, tables) live NEXT TO
    # the session file that owns them: sessions/<dir>/<stem>_numeric/<task>/
    # <exec>/.  A session must therefore be SAVED before numeric execution —
    # the JSON's folder/filename anchor the outputs, so save/load keeps the
    # session and its numeric products together.  Autosave never moves this
    # anchor (slot rotation must not scatter outputs).

    def _numeric_output_root(self) -> "Path":
        if getattr(self, "_last_save_path", None) is None:
            raise ValueError(t("numeric.save_session_first", self.lang))
        p = self._last_save_path
        return p.parent / f"{p.stem}_numeric"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _numeric_task(self, task_id: str) -> NumericTask:
        task = self.numeric_tasks.get(task_id)
        if task is None:
            raise ValueError(t("numeric.task_not_found", self.lang,
                               task_id=task_id))
        return task

    # ------------------------------------------------------------------
    # Typed-graph chaining (docs/26 §5): value_from bindings
    # ------------------------------------------------------------------

    def _resolve_value_from_bindings(self, spec) -> "tuple[object, list[str]]":
        """Resolve every ``{"value_from": artifact_id}`` in *spec*'s inputs to
        the referenced artifact's inline value, type-checking the binding
        against ``BINDABLE_VALUE_KINDS`` (docs/26 §5) — an incompatible kind
        (ordered_list into a scalar slot, complex into a real-only fixed
        value) fails HERE, before any code exists.

        Returns ``(resolved_spec, input_artifact_refs)``.  The original spec
        is not mutated; ``value_from`` stays in the resolved input's spec dict
        as provenance next to the resolved ``value``/``initial``.
        """
        import math
        from notebook.numeric.graph import can_bind

        refs: list[str] = []
        new_inputs = []
        for v in spec.inputs:
            sp = dict(v.spec or {})
            ref = sp.get("value_from")
            if not ref:
                new_inputs.append(v)
                continue
            art = self.numeric_artifacts.get(ref)
            if art is None:
                raise ValueError(t("numeric.spec.artifact_not_found",
                                   self.lang, artifact_id=ref))
            if v.role not in ("fixed", "free"):
                raise ValueError(t("numeric.spec.bad_value_from_role",
                                   self.lang, name=v.name, role=v.role))
            if not can_bind(v.role, art.value_kind):
                raise ValueError(t("numeric.spec.artifact_not_bindable",
                                   self.lang, artifact_id=ref,
                                   value_kind=art.value_kind or "?",
                                   role=v.role))
            val = art.value
            # can_bind guarantees value_kind == "scalar", but the VALUE must
            # also be a plain finite real number (a complex scalar carries a
            # {re, im} dict) — the adversarial-input rule extends to values
            # that arrive via the graph, not just via the UI.
            if (isinstance(val, bool) or not isinstance(val, (int, float))
                    or not math.isfinite(float(val))):
                raise ValueError(t("numeric.spec.artifact_not_bindable",
                                   self.lang, artifact_id=ref,
                                   value_kind=art.value_type or art.value_kind
                                   or "?", role=v.role))
            sp["value" if v.role == "fixed" else "initial"] = val
            new_inputs.append(v.model_copy(update={"spec": sp}))
            refs.append(ref)
        if not refs:
            return spec, []
        return spec.model_copy(update={"inputs": new_inputs}), refs

    # ------------------------------------------------------------------
    # Graph view (docs/26 §6): the DAG derived from refs — never stored.
    # ------------------------------------------------------------------

    def numeric_graph_view(self) -> dict:
        """Derive the numeric-computation graph from the reference fields:
        expression → task (source_expression_ids), task → task (an
        input_artifact_refs entry, compressed into one edge labeled by the
        carrying artifact), task → expression (reflux steps).  Only theory
        expressions that actually touch the numeric layer appear."""
        expr_ids: set = set()
        tasks: list[dict] = []
        edges: list[dict] = []

        for tid, task in self.numeric_tasks.items():
            kind = (task.spec.intent.kind
                    if task.spec and task.spec.intent.kind
                    else task.problem_type)
            tasks.append({
                "task_id": tid,
                "kind": kind,
                "status": task.status,
                "target": task.target_quantity,
                "branch_id": task.source_branch_id,
                "stale": self.numeric_task_stale_info(tid),
            })
            for eid in task.source_expression_ids:
                expr_ids.add(eid)
                edges.append({"from": eid, "to": tid, "kind": "source"})
            for ref in task.input_artifact_refs:
                art = self.numeric_artifacts.get(ref)
                if art is None or art.task_id not in self.numeric_tasks:
                    continue
                edges.append({
                    "from": art.task_id, "to": tid, "kind": "value",
                    "artifact_id": ref,
                    "name": art.metadata.get("name", ""),
                    "value_kind": art.value_kind,
                })

        # Reflux edges: task → theory expression (registration steps stamped
        # by reflux_numeric_value carry the producing task in operation_args).
        for step in self.steps.values():
            if step.operation != "reflux_numeric_value":
                continue
            tid = (step.operation_args or {}).get("task_id", "")
            for out in step.output_refs or []:
                if tid in self.numeric_tasks and out in self.expressions:
                    expr_ids.add(out)
                    edges.append({"from": tid, "to": out, "kind": "reflux",
                                  "name": self._reflux_symbol_name(out)})

        expressions = []
        for eid in sorted(expr_ids):
            expr = self.expressions.get(eid)
            expressions.append({
                "expr_id": eid,
                "latex": (expr.latex if expr else ""),
                "missing": expr is None,
            })
        return {"expressions": expressions, "tasks": tasks, "edges": edges}

    def _reflux_symbol_name(self, expr_id: str) -> str:
        for name, eid in self.lhs_alias_map.items():
            if eid == expr_id:
                return name
        return ""

    # ------------------------------------------------------------------
    # Branch comparison (docs/26 §10-c, G3-3): read-only, artifacts and
    # snapshots only — never touches CAS session state (the B-4 boundary).
    # ------------------------------------------------------------------

    def numeric_branch_comparison(self) -> dict:
        """Group numeric tasks by (kind, target_quantity); a group is
        "comparable" when its tasks span 2+ theory branches.  Deliberately
        LOOSE matching (docs/26 §10-c): the view bundles candidates and the
        researcher judges — automatic lineage matching that silently equates
        different computations would be the worst failure mode here."""
        latest_result: dict = {}
        for step in self.steps.values():
            if step.kind != StepKind.numeric_result:
                continue
            tid = (step.operation_args or {}).get("task_id")
            if tid and (tid not in latest_result
                        or step.timestamp > latest_result[tid].timestamp):
                latest_result[tid] = step

        latest_figure: dict = {}
        for art in self.numeric_artifacts.values():
            if art.artifact_type != "figure":
                continue
            cur = latest_figure.get(art.task_id)
            if cur is None or art.created_at > cur.created_at:
                latest_figure[art.task_id] = art

        groups: dict = {}
        branch_ids: set = set()
        for tid, task in sorted(self.numeric_tasks.items()):
            kind = (task.spec.intent.kind
                    if task.spec and task.spec.intent.kind
                    else task.problem_type)
            key = (kind, task.target_quantity)
            g = groups.setdefault(key, {"kind": kind,
                                        "target": task.target_quantity,
                                        "branches": {}})
            step = latest_result.get(tid)
            fig = latest_figure.get(tid)
            g["branches"].setdefault(task.source_branch_id, []).append({
                "task_id": tid,
                "status": task.status,
                "summary": step.visible_action if step else "",
                "figure_artifact_id": fig.artifact_id if fig else None,
            })
            branch_ids.add(task.source_branch_id)

        out = [{**g, "comparable": len(g["branches"]) >= 2}
               for g in groups.values()]
        return {"groups": out, "branch_ids": sorted(branch_ids)}

    # ------------------------------------------------------------------
    # Reflux as expression (docs/26 §5 / docs/22 §13 phase-2 leftover):
    # a typed scalar artifact becomes a theory-layer expression.
    # ------------------------------------------------------------------

    def reflux_numeric_value(self, artifact_id: str, name: str):
        """Register a typed scalar artifact's value as a theory expression
        ``name = value``, riding the H-1 equation machinery — so the name
        becomes a substitutable symbol (lhs_alias_map + SymbolContext +
        Maxima binding) exactly as if the researcher had typed it, and the
        numeric provenance is recorded on the registration step.

        Same eligibility rule as value_from binding: only a finite REAL
        scalar crosses the layer boundary (a complex value or a series has
        no equation form the theory layer can consume this way)."""
        import math

        art = self.numeric_artifacts.get(artifact_id)
        if art is None:
            raise ValueError(t("numeric.spec.artifact_not_found", self.lang,
                               artifact_id=artifact_id))
        val = art.value
        if (art.value_kind != "scalar" or isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))):
            raise ValueError(t("numeric.reflux_not_scalar", self.lang,
                               artifact_id=artifact_id,
                               value_kind=art.value_kind or "?"))
        if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
            raise ValueError(t("numeric.reflux_bad_name", self.lang,
                               name=str(name)))

        value_str = (str(int(val)) if float(val) == int(val)
                     else repr(float(val)))
        expr = self.register_expression(value_str, lhs_label=name)

        # Stamp the numeric provenance onto the registration step: which
        # artifact/task/execution produced this number.  The step itself was
        # just created by register_expression, so enriching it here keeps
        # "one timeline entry = full provenance" (same principle as the
        # numeric_result step).
        step = self.steps.get(expr.created_by_step)
        if step is not None:
            task = self.numeric_tasks.get(art.task_id)
            step.operation = "reflux_numeric_value"
            step.operation_args = {
                "artifact_id": artifact_id,
                "task_id": art.task_id,
                "execution_id": art.metadata.get("execution_id", ""),
                "source_expression_ids":
                    list(task.source_expression_ids) if task else [],
                "value": val,
            }
            step.visible_action = t("numeric.refluxed", self.lang,
                                    name=name, value=_fmt_num(float(val)),
                                    task_id=art.task_id)
        return expr

    # ------------------------------------------------------------------
    # Stale detection (docs/26 §7): snapshot hash vs current theory layer
    # ------------------------------------------------------------------

    def _numeric_source_hash(self, canonical: str) -> str:
        from notebook.numeric.graph import source_hash
        asm = [getattr(a, "canonical", "") or ""
               for _k, a in sorted(self.assumptions.items())]
        return source_hash(canonical, asm)

    def numeric_task_stale_info(self, task_id: str) -> dict:
        """Report whether *task_id* is stale: the source expression changed
        (or vanished), the assumption set changed, or — transitively — an
        upstream task whose artifact this task consumed is itself stale.
        Legacy tasks (``source_hash == ""``) predate hashing and are never
        reported stale (unknown ≠ stale).  Report-only: nothing is blocked
        (docs/26 §7 — the researcher decides whether to re-run)."""
        task = self._numeric_task(task_id)
        reasons: list[dict] = []
        if task.source_hash:
            # Assumption drift, isolated: re-hash the SNAPSHOT canonical with
            # the CURRENT assumption set — differs iff assumptions changed.
            if self._numeric_source_hash(task.source_canonical) != task.source_hash:
                reasons.append({"kind": "assumptions_changed"})
            # Expression drift: exact canonical comparison against the
            # snapshot, per expression (primary + loop body).
            body_ids = {b.get("expr_id") for b in task.source_body + task.plot_sources}
            for b in task.source_body + task.plot_sources:
                bexpr = self.expressions.get(b.get("expr_id", ""))
                if bexpr is None:
                    reasons.append({"kind": "source_missing",
                                    "expr_id": b.get("expr_id", "")})
                elif bexpr.canonical != b.get("canonical", ""):
                    reasons.append({"kind": "source_changed",
                                    "expr_id": b["expr_id"]})
            for eid in task.source_expression_ids:
                if eid in body_ids:
                    continue
                expr = self.expressions.get(eid)
                if expr is None:
                    reasons.append({"kind": "source_missing", "expr_id": eid})
                elif expr.canonical != task.source_canonical:
                    reasons.append({"kind": "source_changed", "expr_id": eid})
        for ref in task.input_artifact_refs:
            art = self.numeric_artifacts.get(ref)
            if art is None:
                reasons.append({"kind": "input_artifact_missing",
                                "artifact_id": ref})
                continue
            if art.task_id in self.numeric_tasks and \
                    self.numeric_task_stale_info(art.task_id)["stale"]:
                reasons.append({"kind": "upstream_stale",
                                "task_id": art.task_id})
        return {"stale": bool(reasons), "reasons": reasons}

    def _latest_code_artifact(self, task_id: str) -> "NumericArtifact | None":
        arts = [a for a in self.numeric_artifacts.values()
                if a.task_id == task_id and a.artifact_type == "code"]
        return max(arts, key=lambda a: a.created_at) if arts else None

    def _numeric_free_symbols(self, canonical: str) -> "set[str] | None":
        """Free symbols of a canonical via the numpy whitelist lowering, for
        the spec gatekeeper's symbol-membership check.  Returns None when the
        expression is not whitelist-lowerable (a convergence/eigen source may
        not be) — the caller then skips that check rather than blocking."""
        try:
            from notebook.numeric.expr_to_numpy import (
                canonical_matrix_to_numpy, canonical_to_numpy,
                is_matrix_canonical,
            )
            if is_matrix_canonical(canonical):
                _rows, free, _shape = canonical_matrix_to_numpy(
                    canonical, self.symbol_context)
                return free
            _expr, free = canonical_to_numpy(canonical, self.symbol_context)
            return free
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Assumption guard (a): reject fixed values that violate a snapshotted
    # session assumption BEFORE the run.
    # ------------------------------------------------------------------

    def _expand_h1_defs(self, text: str) -> str:
        """Replace each H-1-defined name (lhs_alias_map, e.g. ``D``) with its
        defining RHS, so an assumption like ``D > 0`` can be evaluated from the
        underlying free symbols (``4*omega_0^2-(2*gamma)^2``)."""
        for name, eid in self.lhs_alias_map.items():
            expr = self.expressions.get(eid)
            if expr is None:
                continue
            # The spaced `" = "` also missed a CAS-produced equation, which is
            # written without spaces.  This text feeds the numeric ASSUMPTION
            # guard (`D > 0`), so a wrong body is a guard evaluating the wrong
            # thing before a sandboxed run.
            rhs = strip_lhs(expr.canonical or "")
            text = re.sub(
                r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
                f"({rhs})", text)
        return text

    def _numeric_assumption_violations(
            self, fixed_values: dict, sweep_var_names: list) -> list:
        """Return ``[(assumption_display, values_str, lhs_value), …]`` for each
        snapshotted inequality assumption the *fixed_values* definitively break.

        Reuses the numeric layer's own lowering (``canonical_to_numpy``) so the
        check agrees with how the code will be generated — no live CAS needed.
        Skips assumptions that (i) aren't inequalities, (ii) depend on the swept
        variable (can't be checked at a single point — the NaN net covers that),
        or (iii) still reference a symbol with no value.
        """
        from notebook.numeric.expr_to_numpy import canonical_to_numpy
        try:
            point = {k: float(v) for k, v in fixed_values.items()}
        except (TypeError, ValueError):
            return []                            # non-numeric → template rejects later
        sweep = set(sweep_var_names)
        out: list = []
        for asm in self.assumptions.values():
            cond = (getattr(asm, "canonical", "") or "").strip()
            m = _ASM_OP_RE.search(cond)
            if not m:
                continue
            op = m.group(1)
            lhs = self._expand_h1_defs(cond[:m.start()].strip())
            rhs = self._expand_h1_defs(cond[m.end():].strip())
            try:
                lhs_np, lf = canonical_to_numpy(lhs, self.symbol_context)
                rhs_np, rf = canonical_to_numpy(rhs, self.symbol_context)
            except UnknownNotationError:
                continue
            free = lf | rf
            if free & sweep or not free <= set(point):
                continue                         # pointwise-uncheckable → skip
            try:
                g = {"__builtins__": {}}
                ns = {"np": _SCALAR_NP, **point}
                lv = float(eval(lhs_np, g, ns))  # whitelist-lowered arithmetic only
                rv = float(eval(rhs_np, g, ns))
            except Exception:
                continue
            ok = {">": lv > rv, "<": lv < rv, ">=": lv >= rv, "<=": lv <= rv}[op]
            if not ok:
                vals = ", ".join(f"{k}={v}" for k, v in sorted(fixed_values.items()))
                out.append((getattr(asm, "display", None) or cond, vals, lv))
        return out
