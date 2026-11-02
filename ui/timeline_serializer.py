"""Timeline serialization (R2, docs/27): the /timeline endpoint's entry
builder, extracted from notebook_bridge.

``build_timeline_entries(engine)`` is a pure read of engine state: it walks the
active branch's lineage and turns each step into the JSON entry shape the
front-end renders.  Kept separate from the HTTP layer so the bridge endpoint is
a one-line delegation and this logic can be unit-tested against an engine
directly.  Read-only — it never mutates the engine.
"""
from __future__ import annotations

from notebook.models.step import StepKind as SK
from ui.numeric_products import numeric_downloads
from notebook.runtime.replay import compute_lineage

# **YOU カードは研究者が打った文をそのまま出す。** ここには長く、10操作を
# 手で並べた表と「先頭の式参照を削る」整形（`expr_ref_prefix`）があった。
# 削る対象は `ui/interpret_pipeline` が補う式番号——**内部の足場**——だと
# 書かれていたが、実測 2026-09-10、足場が乗っていたのは `llm_supervision`
# ステップのほうで、ここが読む `user_input` ステップには**1件も無かった**
# （公開実例の6件はすべて前者）。つまりこの整形は、
#
#   * 直すべき場所ではないところに当てられており（発火しない）、
#   * 発火したときは**研究者自身の言葉を削る**——`E5 の x をスカラーとする`
#     の `E5 の` は研究者が打った文の一部で、実際に保存済みセッションに在る。
#
# 足場が記録に入らないよう `interpret_pipeline` 側を直したので、整形は消した。


def build_timeline_entries(engine) -> list[dict]:
    """Timeline entries for the active branch, in chronological order."""
    lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)
    lineage_ids = {s.step_id for s in lineage}

    # user_input steps do not advance branch head, so they are absent from the
    # parent-chain lineage.  Collect them as "companions" keyed by the parent
    # they share with the immediately following state-affecting step.
    companions: dict = {}
    for s in engine.steps.values():
        if s.step_id in lineage_ids:
            continue
        if s.branch_id != engine.active_branch_id:
            continue
        if s.kind != SK.user_input or not s.user_text:
            continue
        parent_id = s.parent_step_ids[0] if s.parent_step_ids else None
        companions.setdefault(parent_id, []).append(s)
    _snum = lambda s: int(s.step_id[1:]) if s.step_id[1:].isdigit() else 0
    for key in companions:
        companions[key].sort(key=_snum)

    referenced_expr_ids: set[str] = set()
    for s in engine.steps.values():
        referenced_expr_ids.update(s.input_refs)

    # Build a lookup: parent_step_ids key → llm_supervision step args.
    # Used to recover _raw_names / _op_expr_display that are stored on the
    # supervision step (not on the cas_operation step in the lineage).
    _supervision_args: dict[str, dict] = {}
    for s in engine.steps.values():
        if s.kind == SK.llm_supervision and s.operation_args:
            key = tuple(sorted(s.parent_step_ids))
            _supervision_args[key] = s.operation_args

    entries = []
    for step in lineage:
        # Insert companion user_input entries that preceded this step
        parent_id = step.parent_step_ids[0] if step.parent_step_ids else None
        for comp in companions.get(parent_id, []):
            entries.append({
                "type":    "user_input",
                "step_id": comp.step_id,
                "text":    comp.user_text,
            })

        if step.kind == SK.user_input and step.user_text:
            entries.append({
                "type":    "user_input",
                "step_id": step.step_id,
                "text":    step.user_text,
            })
        elif step.kind in (SK.expression_registration, SK.cas_operation):
            for ref in step.output_refs:
                expr = engine.expressions.get(ref)
                # `internal` as well as `hidden`: a scaffolding expression is
                # reachable (hidden=False) for as long as the operation that
                # needs it runs, so `hidden` alone would show it to anyone who
                # rendered the timeline during that window
                # (INTERNAL-INTERMEDIATE-EID-GAP).
                if expr and not (expr.hidden or expr.internal):
                    entries.append({
                        "type":            "expression",
                        "step_id":         step.step_id,
                        "expr_id":         expr.display_id,
                        "latex":           expr.latex,
                        "canonical":       expr.canonical,
                        "cas_repr":        expr.cas_repr,
                        "deletable":       expr.display_id not in referenced_expr_ids,
                        "operation_label": step.visible_action if step.kind == SK.cas_operation else None,
                    })
            if not step.output_refs and step.kind == SK.cas_operation and step.visible_action:
                # Side-effect operation (e.g. define_fermion_operator): no output expression
                _args = step.operation_args or {}
                # Merge in supervision step args to recover _raw_names / _op_expr_display.
                _sup_key = tuple(sorted(step.parent_step_ids))
                _sup_args = _supervision_args.get(_sup_key, {})
                _args = {**_sup_args, **_args}  # cas_operation fields take precedence
                _reg_names = _args.get("registered_names") or []
                _raw_names = _args.get("_raw_names") or []
                _single_name = (_raw_names[0] if _raw_names else None) or _args.get("name", "")
                _all_names = _reg_names if _reg_names else (_raw_names if _raw_names else ([_single_name] if _single_name else []))
                _display_latex = ", ".join(
                    engine._handler._canonical_name_to_latex(n) for n in _all_names
                )
                entries.append({
                    "type":          "side_effect",
                    "step_id":       step.step_id,
                    "text":          step.visible_action,
                    "name":          _single_name,
                    "display_latex": _display_latex,
                })
        elif step.kind == SK.assumption_add:
            assumption_id = step.operation_args.get("assumption_id", "")
            canonical     = step.operation_args.get("canonical", "")
            display_str   = step.operation_args.get("display", "") or canonical
            if assumption_id in engine.assumptions:
                entries.append({
                    "type":          "assumption",
                    "step_id":       step.step_id,
                    "assumption_id": assumption_id,
                    "canonical":     canonical,
                    "display":       display_str,
                })
        elif step.kind == SK.manual_note:
            entries.append({
                "type":      "note",
                "step_id":   step.step_id,
                "text":      step.user_text or "",
                "timestamp": step.timestamp.strftime("%Y-%m-%d %H:%M") if step.timestamp else "",
            })
        elif step.kind == SK.numeric_fork:
            entries.append({
                "type":    "numeric_fork",
                "step_id": step.step_id,
                "text":    step.visible_action,
                "task_id": (step.operation_args or {}).get("task_id", ""),
            })
        elif step.kind == SK.numeric_result:
            _nargs = step.operation_args or {}
            _art_ids = _nargs.get("artifact_ids") or []
            # Pick the figure artifact (if any) so the entry can render the
            # plot; its file is served by GET /numeric/figure/{artifact_id}.
            _fig_id = next(
                (a for a in _art_ids
                 if (art := engine.numeric_artifacts.get(a)) is not None
                 and art.artifact_type == "figure"),
                None)
            entries.append({
                "type":               "numeric_result",
                "step_id":            step.step_id,
                "text":               step.visible_action,
                "task_id":            _nargs.get("task_id", ""),
                "summary":            _nargs.get("summary") or {},
                "figure_artifact_id": _fig_id,
                "downloads": numeric_downloads(engine, _art_ids),
                "ok":                 getattr(step.status, "value", "") != "failed",
            })
        elif step.kind == SK.plan_execution:
            # D-5 (open_items.md): the live confirm-click render bypasses this
            # serializer entirely (it renders the raw /interpret or /operation
            # response), so without this branch a plan's step-by-step summary
            # — success or stopped alike — silently vanished on reload even
            # though engine.plans/engine.steps still had the data intact.
            _plargs = step.operation_args or {}
            entries.append({
                "type":               "plan_execution",
                "step_id":            step.step_id,
                "message":            step.visible_action,
                "plan_id":            _plargs.get("plan_id", ""),
                "status":             "completed" if getattr(step.status, "value", "") == "success" else "failed",
                "plan_step_results":  _plargs.get("step_results") or [],
            })

    return entries
