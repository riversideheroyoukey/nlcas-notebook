"""NotebookEngine mixin: session persistence (save / load / CAS snapshots).

Extracted from engine.py (AUD-4 stage 3).  Serializes session state to JSON and
reconstructs a NotebookEngine from it (load is a classmethod using cls), plus
CAS state snapshots.  Mixed into NotebookEngine; state is reached via self / cls.
NotebookEngine appears only in annotations (PEP 563), so there is no import from
engine.py and no circular import.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from notebook.runtime.shared_parser import split_relation
from notebook.models.branch import Branch
from notebook.models.expression import Assumption, Expression
from notebook.models.quantity import Quantity
from notebook.models.session import Session
from notebook.models.step import Step
from notebook.runtime.backend_factory import make_backend
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.placeholder_sum_bounds import BoundLedger

_logger = logging.getLogger(__name__)

# Session-file schema version written by save().
# History:
#   "1" (implicit for files without the key): sessions saved before 2026-07-03.
#       Loading runs the legacy repair migrations (wrong-func-call rewrite,
#       raw-solve-result hiding/promotion, broken-LaTeX rebuild) below.
#   "2": sessions saved from 2026-07-03 on.  No legacy migrations are applied —
#       the bugs those migrations repair cannot occur in files written at v2.
#       Adds canonical_pairs / spin_display_styles / einstein_summation keys.
_SAVE_VERSION = "2"


class EnginePersistenceMixin:
    """Session persistence (save/load + CAS snapshots) for NotebookEngine."""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _relocate_numeric_outputs(self, new_path: "Path") -> None:
        """When an explicit save changes the numeric anchor, bring the existing
        numeric outputs along so they stay beside the session file.

        Numeric run-time output lives under ``<stem>_numeric/`` derived from the
        save path in effect AT RUN TIME (often a throwaway auto-save name, since
        run auto-saves an unsaved session).  A later "Save As <name>" would
        otherwise leave the code/figure/table orphaned under the old stem AND
        break figure serving (the figure route anchors to the CURRENT stem, so
        the old paths fail its containment check → 404, even after reload).

        Move ``<old_stem>_numeric/`` → ``<new_stem>_numeric/`` and rewrite every
        numeric artifact ``path`` that pointed under the old root, so the JSON
        written next records the new locations.  Only for explicit saves — an
        autosave must never move the anchor.
        """
        old_path = getattr(self, "_last_save_path", None)
        if old_path is None or not self.numeric_artifacts:
            return
        old_root = old_path.parent / f"{old_path.stem}_numeric"
        new_root = new_path.parent / f"{new_path.stem}_numeric"
        if old_root == new_root or not old_root.exists():
            return
        new_root.mkdir(parents=True, exist_ok=True)
        for child in old_root.iterdir():
            target = new_root / child.name
            if target.exists():
                # A same-named task dir at the destination is replaced — the
                # artifacts we're re-pointing are THIS session's, and task ids
                # are per-session, so our copy is authoritative.
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        try:
            old_root.rmdir()          # now-empty; leave it if not (best effort)
        except OSError:
            pass
        old_s, new_s = str(old_root), str(new_root)
        for aid, art in list(self.numeric_artifacts.items()):
            if art.path and (art.path == old_s or art.path.startswith(old_s + "/")):
                self.numeric_artifacts[aid] = art.model_copy(
                    update={"path": new_s + art.path[len(old_s):]})

    def save(self, path: str | Path, is_autosave: bool = False) -> None:
        """Serialize session state to a JSON file.

        Explicit saves anchor the numeric-output location (the JSON's
        folder/stem names sessions/<stem>_numeric/).  Autosave rotation must
        NOT move that anchor, hence the flag."""
        if not is_autosave:
            # Relocate numeric outputs to follow the new anchor BEFORE the JSON
            # is built, so the serialized artifact paths record the new location.
            self._relocate_numeric_outputs(Path(path).resolve())
            self._last_save_path = Path(path).resolve()
        data = {
            "version": _SAVE_VERSION,
            "session": self.session.model_dump(mode="json"),
            "expressions": {k: v.model_dump(mode="json") for k, v in self.expressions.items()},
            "assumptions": {k: v.model_dump(mode="json") for k, v in self.assumptions.items()},
            "steps": {k: v.model_dump(mode="json") for k, v in self.steps.items()},
            "branches": {k: v.model_dump(mode="json") for k, v in self.branches.items()},
            "quantity_registry": {
                k: {
                    "quantity": prop.quantity.model_dump(mode="json"),
                    "statistics": prop.statistics.model_dump(mode="json"),
                }
                for k, prop in self.symbol_context.registry.items()
            },
            "notation_usage": dict(self.symbol_context.notation_usage),
            "canonical_pairs": [list(p) for p in self.symbol_context.canonical_pairs],
            "spin_display_styles": dict(self.symbol_context.spin_display_styles),
            # F-13 — additive key; absent in older files (uniform spin).
            "site_spins": dict(self.symbol_context.site_spins),
            # スピン射影の値集合（分子）。追加キーで、古いファイルは空で読める
            # ——空は「まだ言われていない」であって「射影が無い」ではないので、
            # 読み直したセッションは `\sum_\sigma` を形式和のまま扱う（誤らない）。
            "spin_projections": {k: list(v) for k, v
                                 in self.symbol_context.spin_projections.items()},
            "indexed_bases": dict(self.symbol_context.indexed_bases),   # docs/32 Phase 4
            "called_bases": list(self.symbol_context.called_bases),   # docs/32 Phase 4
            "einstein_summation": self.symbol_context.einstein_summation,
            "lhs_alias_map": dict(self.lhs_alias_map),
            "expr_counter": self._expr_counter,
            # Scaffolding key counter (INTERNAL-INTERMEDIATE-EID-GAP) — additive
            # key; an older file has none and loads as 0, which is correct: its
            # intermediates were keyed from the E space and stay that way.
            "internal_counter": self._internal_counter,
            # Placeholder bounds for range-less sums (B-10 ②) — additive keys.
            # The counter must not restart: a reloaded session would hand a NEW
            # sum the same pair an existing one already carries in its canonical.
            # The MAP is what lets an existing sum get its own pair back — the
            # IR that carried it is `exclude=True` and does not survive the save,
            # so without this a reloaded sum went to the CAS rangeless and Maxima
            # refused it (B-10 ②-c, measured 2026-08-08).
            **self._sum_bounds.as_saved(),
            "step_counter": self._step_counter,
            # Live branch (fork-in-session) — additive key; absent in older files.
            "branch_counter": self._branch_counter,
            # Numeric layer (N-1) — additive keys; absent in older files.
            "numeric_tasks": {k: v.model_dump(mode="json")
                              for k, v in self.numeric_tasks.items()},
            "numeric_artifacts": {k: v.model_dump(mode="json")
                                  for k, v in self.numeric_artifacts.items()},
            "numeric_executions": {k: v.model_dump(mode="json")
                                   for k, v in self.numeric_executions.items()},
            "numeric_counter": self._numeric_counter,
            # Plan layer (C-Ext-5) — additive keys; absent in older files.
            "plans": {k: v.model_dump(mode="json")
                      for k, v in self.plans.items()},
            "plan_counter": self._plan_counter,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> NotebookEngine:
        """Restore a session from a JSON file.

        Returns an engine whose Maxima adapter has NOT been started yet.
        Use as a context manager or call start() explicitly before running operations.
        """
        data = json.loads(Path(path).read_text())
        engine: NotebookEngine = cls.__new__(cls)
        engine._last_save_path = Path(path).resolve()
        engine.session = Session.model_validate(data["session"])
        engine.expressions = {
            k: Expression.model_validate(v) for k, v in data["expressions"].items()
        }
        engine.assumptions = {
            k: Assumption.model_validate(v)
            for k, v in data.get("assumptions", {}).items()
        }
        engine.steps = {
            k: Step.model_validate(v) for k, v in data["steps"].items()
        }
        engine.branches = {
            k: Branch.model_validate(v) for k, v in data["branches"].items()
        }
        from notebook.models.symbol_context import SymbolContext, SymbolStatistics
        engine.symbol_context = SymbolContext()
        engine.symbol_context.domains = engine.session.domains
        engine.symbol_context.session_mode = engine.session.mode.value
        for k, raw in data.get("quantity_registry", {}).items():
            if isinstance(raw, dict) and "quantity" in raw:
                # new format: {quantity: {...}, statistics: {...}}
                qty = Quantity.model_validate(raw["quantity"])
                stats_kind = raw.get("statistics", {}).get("kind", "bose")
            else:
                # legacy format: quantity fields directly (no statistics saved)
                qty = Quantity.model_validate(raw)
                stats_kind = "bose"
            engine.symbol_context.register(k, quantity=qty, statistics_kind=stats_kind)
        engine.symbol_context.notation_usage = data.get("notation_usage", {})
        engine.symbol_context.canonical_pairs = [
            tuple(p) for p in data.get("canonical_pairs", []) if len(p) == 2
        ]
        engine.symbol_context.spin_display_styles = dict(
            data.get("spin_display_styles", {})
        )
        engine.symbol_context.indexed_bases = {
            str(k): int(v) for k, v in (data.get("indexed_bases") or {}).items()
        }   # docs/32 Phase 4 — additive key; an older file loads as empty
        engine.symbol_context.called_bases = [
            str(v) for v in (data.get("called_bases") or [])
        ]   # docs/32 Phase 4 — additive key
        engine.symbol_context.site_spins = {
            str(k): str(v) for k, v in (data.get("site_spins") or {}).items()
        }
        engine.symbol_context.spin_projections = {
            str(k): [int(n) for n in v]
            for k, v in (data.get("spin_projections") or {}).items()
        }   # 追加キー — 古いファイルは空で読める
        engine.symbol_context.einstein_summation = bool(
            data.get("einstein_summation", False)
        )
        engine.lhs_alias_map = dict(data.get("lhs_alias_map", {}))
        from notebook.models.numeric import (
            NumericArtifact, NumericExecution, NumericTask,
        )
        engine.numeric_tasks = {
            k: NumericTask.model_validate(v)
            for k, v in data.get("numeric_tasks", {}).items()
        }
        engine.numeric_artifacts = {
            k: NumericArtifact.model_validate(v)
            for k, v in data.get("numeric_artifacts", {}).items()
        }
        engine.numeric_executions = {
            k: NumericExecution.model_validate(v)
            for k, v in data.get("numeric_executions", {}).items()
        }
        engine._numeric_counter = int(data.get("numeric_counter", 0))
        from notebook.models.plan import SupervisorPlan
        engine.plans = {
            k: SupervisorPlan.model_validate(v)
            for k, v in data.get("plans", {}).items()
        }
        engine._plan_counter = int(data.get("plan_counter", 0))
        engine._branch_counter = int(data.get("branch_counter", 0))
        engine._expr_counter = data["expr_counter"]
        engine._internal_counter = int(data.get("internal_counter", 0))
        engine._sum_bounds = BoundLedger.from_saved(data)
        engine._step_counter = data["step_counter"]
        engine._adapter, engine._builder, engine._handler = make_backend(engine.session.primary_backend)
        engine._handler.set_symbol_context(engine.symbol_context)
        engine._handler.bind_language_source(lambda: engine.lang)   # I18N-1
        engine._validator = OperationValidator()
        engine.pending_op = None

        # docs/32 §2.7: a free parameter is a NAME the CAS holds (k_1, k_2, C).
        # Re-declare them from what the file itself records, so a reloaded
        # session resolves k_1 the same way the live one did — undeclared it
        # lowers to k(1) and every substitution onto it silently misses.
        # Version-independent: this restores state from the file's own data,
        # it is not one of the pre-v2 damage repairs gated below.
        for _e in engine.expressions.values():
            engine._declare_free_parameters(getattr(_e, "free_parameters", None))

        # Schema-version gate: the migrations below repair damage produced by
        # bugs that existed before v2 files were introduced (P2 substitution
        # artifacts, raw solve results, \\, comma-split LaTeX).  Files written
        # at v2 or later cannot contain that damage, so loading them must not
        # rewrite anything (a v2 file round-trips byte-identically).
        try:
            _file_version = int(str(data.get("version", "1")))
        except (TypeError, ValueError):
            _file_version = 1
        if _file_version >= 2:
            return engine

        # Migration: incorrect function-call canonicals (e.g. l*m(l*omega^2))
        # This arises when P2 sent symbol definitions to Maxima which substituted
        # lhs_alias symbols (like g = l*omega^2) producing m(l*omega^2) notation.
        # Fix: replace single-letter-function-calls with multiplication in both
        # canonical and latex fields.
        # Fix canonicals where a single-letter symbol was wrongly stored as a function
        # call with an arithmetic argument, e.g. m(l*omega^2) → m*(l*omega^2).
        # This happens when P2 substitution caused g=l*omega^2 to be inserted and
        # Maxima kept the grouped form as a function-call-like canonical.
        # Guard: only fix single-letter "functions" whose argument contains * or ^
        # (arithmetic expression), NOT things like q(t) (t is a single plain variable).
        import re as _re_fc
        _WRONG_FUNC_RE = _re_fc.compile(
            r'\b([a-zA-Z])\(([^()]*(?:[*^][^()]*)+)\)'
        )

        # A call whose callee the user explicitly declared callable is a REAL
        # function call, not P2-substitution damage — never rewrite those.
        # Without this guard a legitimate f(x^2) becomes f*(x^2) on every load.
        _CALLABLE_KINDS_FC = {"function", "operator", "vector_field", "scalar_field"}
        _callable_names_fc = {
            name for name, prop in engine.symbol_context.registry.items()
            if prop.quantity.kind in _CALLABLE_KINDS_FC
        }

        def _fix_wrong_func_calls(s: str) -> str:
            return _WRONG_FUNC_RE.sub(
                lambda m2: m2.group(0) if m2.group(1) in _callable_names_fc
                else f"{m2.group(1)}*({m2.group(2)})", s)

        _GREEK_MAP_FC = {
            "omega": "\\omega", "alpha": "\\alpha", "beta": "\\beta",
            "gamma": "\\gamma", "delta": "\\delta", "theta": "\\theta",
            "lambda": "\\lambda", "mu": "\\mu", "nu": "\\nu",
            "xi": "\\xi", "pi": "\\pi", "phi": "\\phi",
            "psi": "\\psi", "sigma": "\\sigma", "tau": "\\tau",
            "Omega": "\\Omega", "Delta": "\\Delta", "Gamma": "\\Gamma",
            "Phi": "\\Phi", "Psi": "\\Psi", "Sigma": "\\Sigma",
        }

        for _eid, _ex in list(engine.expressions.items()):
            _can = _ex.canonical or ""
            if not _WRONG_FUNC_RE.search(_can):
                continue
            _fixed_can = _fix_wrong_func_calls(_can)
            if _fixed_can == _can:
                continue  # all matches were declared callables — nothing to repair
            _rough = _fixed_can
            for _gk, _gl in _GREEK_MAP_FC.items():
                _rough = _re_fc.sub(rf'\b{_gk}\b', lambda _, r=_gl: r, _rough)
            _rough = _rough.replace("*", " ")
            _rough = _re_fc.sub(r'\^(\d+)', lambda m2: "^{" + m2.group(1) + "}", _rough)
            _new_lat = (
                engine._handler.postprocess_result_latex(
                    _fixed_can, _rough, engine.symbol_context
                ) or _rough
            )
            _stored_lat = _ex.latex or ""
            # `" = "` was the SPACED spelling of "is this an equation", and this
            # runs on LOAD over stored latex — where Maxima's own `tex1` writes
            # `x=e^{-t\gamma}…` with no spaces (present in three saved sessions).
            # No input was found that changes the OUTCOME here (the repair is
            # gated on `_WRONG_FUNC_RE` as well), so this is the one definition,
            # not a measured fix.
            _rel_lat = split_relation(_stored_lat)
            if _rel_lat is not None and _rel_lat[1] == "=":
                _new_lat = f"{_rel_lat[0]} = {_new_lat}"
            engine.expressions[_eid] = _ex.model_copy(
                update={"canonical": _fixed_can, "latex": _new_lat}
            )

        # Migration: raw solve results (expressions whose canonical starts with "["
        # and whose step is a solve_equation/solve_system operation) should be hidden.
        # This fixes sessions saved before the hidden-raw-result change.
        # Hides WITHOUT reclaiming the E number, deliberately — a saved session's
        # steps and user_text already name the numbers above it (docs/02 §20b).
        from notebook.models.step import StepKind as _SK
        _solve_raw_ids: set[str] = set()
        for _s in engine.steps.values():
            if _s.operation not in ("solve_equation", "solve_system", "build_inverse_transform"):
                continue
            # The raw result is the expression whose canonical has outer brackets
            for _oid in (_s.output_refs or []):
                _ex = engine.expressions.get(_oid)
                if _ex and (_ex.canonical or "").strip().startswith("["):
                    _solve_raw_ids.add(_oid)
        for _rid in _solve_raw_ids:
            if not engine.expressions[_rid].hidden:
                engine.expressions[_rid] = engine.expressions[_rid].model_copy(
                    update={"hidden": True}
                )

        # Migration: promote clean solve results to the raw predecessor's ID slot.
        # Old sessions have E9=raw(hidden) and E10=clean; users typed "E9" to refer to
        # the result, but E9 is hidden.  Swap so clean result uses the lower ID.
        import re as _re_mig
        def _pred_expr_id(eid: str) -> str | None:
            _m = _re_mig.match(r'^([A-Za-z]+)(\d+)$', eid)
            if _m:
                _n = int(_m.group(2))
                return f"{_m.group(1)}{_n - 1}" if _n > 1 else None
            return None

        for _s in engine.steps.values():
            if _s.operation not in ("solve_equation", "solve_system", "build_inverse_transform"):
                continue
            _new_refs = list(_s.output_refs or [])
            _changed = False
            for _ci, _clean_id in enumerate(list(_new_refs)):
                _clean_ex = engine.expressions.get(_clean_id)
                if not _clean_ex or _clean_ex.hidden:
                    continue
                _raw_id = _pred_expr_id(_clean_id)
                if not _raw_id:
                    continue
                _raw_ex = engine.expressions.get(_raw_id)
                if not (_raw_ex and _raw_ex.hidden
                        and (_raw_ex.canonical or "").strip().startswith("[")):
                    continue
                # Promote: copy clean into raw slot, remove higher-numbered entry.
                engine.expressions[_raw_id] = _clean_ex.model_copy(
                    update={"display_id": _raw_id}
                )
                engine.expressions.pop(_clean_id, None)
                for _ak, _av in list(engine.lhs_alias_map.items()):
                    if _av == _clean_id:
                        engine.lhs_alias_map[_ak] = _raw_id
                _new_refs[_ci] = _raw_id
                _changed = True
            if _changed:
                try:
                    engine.steps[_s.step_id] = _s.model_copy(
                        update={"output_refs": _new_refs}
                    )
                except Exception:
                    _logger.warning(
                        "failed to update step %s output_refs during load cleanup; "
                        "history may be incomplete", _s.step_id, exc_info=True)

        # Migration: clean solve result expressions saved with broken LaTeX
        # (e.g. "g = l\" due to \, comma-split bug) — rebuild from lhs_alias_map + canonical.
        _inv_alias: dict[str, str] = {
            eid: lhs for lhs, eid in engine.lhs_alias_map.items()
            if "[" not in lhs
        }
        for _s in engine.steps.values():
            if _s.operation not in ("solve_equation", "solve_system", "build_inverse_transform"):
                continue
            for _oid in (_s.output_refs or []):
                _ex = engine.expressions.get(_oid)
                if not _ex or _ex.hidden:
                    continue
                _lhs = _inv_alias.get(_oid, "")
                if not _lhs:
                    continue
                # Detect broken latex:
                #  (a) ends with backslash — \, comma-split bug left truncated string
                #  (b) RHS contains "*" — Maxima canonical leaked through without LaTeX render
                _lat = _ex.latex or ""
                # Feeds the "is this latex broken" decision below.
                _rel_b = split_relation(_lat)
                _rhs_part = (_rel_b[2] if (_rel_b is not None and "=" in _rel_b[1])
                             else _lat)
                _is_broken = (
                    _lat.endswith("\\") or _lat.endswith("\\\\")
                    or ("*" in _rhs_part and "\\" not in _rhs_part)
                )
                if _is_broken:
                    _canonical = _ex.canonical or ""
                    # Convert Maxima canonical notation to rough LaTeX:
                    # omega → \omega, alpha → \alpha, etc.; a*b → a b; a^n → a^{n}
                    _GREEK = {
                        "omega": r"\omega", "alpha": r"\alpha", "beta": r"\beta",
                        "gamma": r"\gamma", "delta": r"\delta", "epsilon": r"\epsilon",
                        "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta",
                        "lambda": r"\lambda", "mu": r"\mu", "nu": r"\nu",
                        "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
                        "sigma": r"\sigma", "tau": r"\tau", "phi": r"\phi",
                        "chi": r"\chi", "psi": r"\psi", "Omega": r"\Omega",
                        "Delta": r"\Delta", "Gamma": r"\Gamma", "Lambda": r"\Lambda",
                        "Phi": r"\Phi", "Pi": r"\Pi", "Psi": r"\Psi", "Sigma": r"\Sigma",
                        "Theta": r"\Theta", "Xi": r"\Xi",
                    }
                    import re as _re_mig2
                    _rough = _canonical
                    for _gk, _gl in _GREEK.items():
                        _rough = _re_mig2.sub(rf'\b{_gk}\b', _gl, _rough)
                    _rough = _rough.replace("*", " ")
                    _rough = _re_mig2.sub(r'\^(\d+)', r'^{\1}', _rough)
                    _new_lat = (
                        engine._handler.postprocess_result_latex(
                            _canonical, _rough, engine.symbol_context
                        ) or _rough
                    )
                    engine.expressions[_oid] = _ex.model_copy(
                        update={"latex": f"{_lhs} = {_new_lat}"}
                    )

        return engine

    # ------------------------------------------------------------------
    # CAS snapshot
    # ------------------------------------------------------------------

    def save_cas_snapshot(self, path: str | Path) -> bool:
        """Save CAS state snapshot. Returns False if not running."""
        return self._adapter.save_snapshot(path, rendered_cmds=self._collect_rendered_commands())

    def load_cas_snapshot(self, path: str | Path) -> bool:
        """Load CAS state snapshot. Returns False if not running or file missing."""
        return self._adapter.load_snapshot(path)
