"""NotebookEngine mixin: step/history bookkeeping, recording and deletion.

Extracted from engine.py (AUD-4 stage 3): ID allocation, head/branch advance,
user-input / supervision / note / quantity-update / ambiguity recording, step
and expression deletion with parent-chain rewiring, plus active-expr and
visible-action helpers.  Mixed into NotebookEngine; all state is reached via
self / MRO.  No dependency on engine.py module-level constants.
"""
from __future__ import annotations

import re

from notebook.runtime.shared_parser import split_relation

from datetime import datetime, timezone

from notebook.models.branch import Branch
from notebook.models.expression import Expression
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.branch_manager import BranchManager
from notebook.runtime.i18n import t
from notebook.runtime.shared_parser import find_top_level_relation


#: The prefix `_alloc_internal_expr_id` below hands out.  One place makes these
#: keys and, until 2026-08-27, no place recognised them — the docstring there
#: says "the display layers skip it explicitly", which is true of the EXPRESSION
#: (no card is drawn for `_T1`) and false of its NAME: 40 published labels read
#: `simplify [_T1]` / `_T1 を登録する`.  A predicate so the question has one
#: answer.
_INTERNAL_EXPR_PREFIX = "_T"


def is_internal_expr_id(name: str) -> bool:
    """Is *name* a scaffolding key the researcher has never seen?"""
    return bool(name) and str(name).startswith(_INTERNAL_EXPR_PREFIX) \
        and str(name)[len(_INTERNAL_EXPR_PREFIX):].isdigit()


def label_names_internal_scaffold(label: str) -> bool:
    """Does *label* name a scaffolding key?  Then it is not fit to show."""
    import re as _re
    return bool(label) and bool(
        _re.search(r"(?<![A-Za-z0-9_])" + _INTERNAL_EXPR_PREFIX + r"\d+\b", label))


class EngineHistoryMixin:
    """Step/history bookkeeping, recording and deletion for NotebookEngine."""

    def _alloc_expr_id(self) -> str:
        self._expr_counter += 1
        return f"E{self._expr_counter}"

    def _alloc_internal_expr_id(self) -> str:
        """A key for scaffolding the researcher never sees.

        INTERNAL-INTERMEDIATE-EID-GAP: several operations build a temporary
        expression, run a CAS operation on it, then hide it.  While they took
        their key from the E counter, every such temporary punched a HOLE in the
        researcher's numbering — `E1` followed by `E3` — and a hole reads as
        "there is an E2 somewhere".  Measured: three misreadings in a few days,
        one of which wrote a substitute command against the empty number.

        A separate namespace is the fix rather than renumbering afterwards: an
        E number, once shown, is what the researcher writes down and refers to,
        so the numbers that exist must never move.  Nothing here parses the
        digits out of a key (the four sites that do all guard with isdigit()),
        so a non-E key is inert everywhere except the display layers, which skip
        it explicitly.
        """
        self._internal_counter += 1
        return f"_T{self._internal_counter}"

    def _alloc_step_id(self) -> str:
        self._step_counter += 1
        return f"S{self._step_counter}"

    def _alloc_branch_id(self) -> str:
        self._branch_counter += 1
        return f"branch_{self._branch_counter}"

    def _current_parent_ids(self) -> list[str]:
        head = self.active_branch.head_step_id
        return [head] if head else []

    def _advance_head(self, step: Step) -> None:
        updated = BranchManager.advance_head(self.active_branch, step)
        self.branches[updated.branch_id] = updated

    def _restore_stored_expr_var(self, expr_id: str) -> bool:
        """Re-execute CAS commands to restore a missing CAS-side expression variable.

        Called when the handler recognises the backend's "no such variable"
        error (``BackendHandler.missing_variable_expr_id``) — today only Julia
        stores results this way, as ``_nb_E{n}``.  The restoration itself is
        generic replay machinery, which is why it lives here and not in the
        handler.
        Two cases:
          - CAS-derived expression: replay rendered commands up to the creator step.
          - Registered expression: replay preceding CAS commands (operator/param
            definitions) then materialize _nb_En via build_expr_restore_command.
        Returns True if the target variable was successfully restored.
        """
        from notebook.runtime.replay import compute_lineage

        creator_step_id = next(
            (sid for sid, s in self.steps.items() if expr_id in (s.output_refs or [])),
            None,
        )
        if creator_step_id is None:
            return False

        creator = self.steps.get(creator_step_id)
        lineage = compute_lineage(self.active_branch_id, self.steps, self.branches)

        if creator and creator.kind == StepKind.expression_registration:
            # Expression was registered (no Julia command was ever executed for it).
            # First restore the Julia environment (operator/param definitions),
            # then materialize _nb_En from the expression's input_canonical.
            expr = self.expressions.get(expr_id)
            if expr is None:
                return False
            restore_cmd = self._handler.build_expr_restore_command(expr_id, expr)
            if restore_cmd is None:
                return False
            # Replay all preceding state-affecting CAS commands.
            for step in lineage:
                if step.step_id == creator_step_id:
                    break
                if not step.state_affecting or step.kind != StepKind.cas_operation:
                    continue
                if step.status != StepStatus.success or not step.rendered_command:
                    continue
                result = self._adapter.execute(step.rendered_command)
                if not result.ok:
                    return False
            # Materialize _nb_En.
            result = self._adapter.execute(restore_cmd)
            return result.ok

        # CAS-derived expression: replay rendered commands up to creator step.
        for step in lineage:
            if not step.state_affecting:
                continue
            if step.kind != StepKind.cas_operation:
                continue
            if step.status != StepStatus.success:
                continue
            if not step.rendered_command:
                continue

            result = self._adapter.execute(step.rendered_command)
            if not result.ok:
                return False

            if step.step_id == creator_step_id:
                return True

        return False

    # ------------------------------------------------------------------
    # Meta-step recording (non-CAS events for the decision trail)
    # ------------------------------------------------------------------

    def record_user_input(self, text: str) -> Step:
        """Record raw user input as a non-state-affecting step."""
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.user_input,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.user_input", self.lang),
            user_text=text,
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        # user_input does not advance the branch head — the next real step will
        return step

    def record_llm_supervision(
        self,
        user_text: str,
        operation: str | None,
        args: dict | None,
        candidates: list[dict] | None = None,
        confirmed: bool = True,
    ) -> Step:
        """Record an LLM supervisor interpretation as a non-state-affecting step.

        **``user_text`` is what the RESEARCHER TYPED — never the text handed to
        the interpreter.**  Those are two different things and this field is the
        first: everything downstream (the timeline YOU card, the TeX export,
        the advisory feed, the saved session, the published examples) reads it
        as the researcher's own writing.

        The text handed to the interpreter has been rewritten twice by the time
        it reaches here — 2a-sym replaces the researcher's symbol names with
        expression ids, and step 2d PREPENDS the most recent expression when the
        instruction names none.  Both are scaffolding.  Measured 2026-09-10
        (open_items DECL-BARE-NOUN-EN（残）②), passing that text in put **34
        strings a researcher never wrote** into the committed published
        examples::

            "user_text": "E19 に対して G を行列とする"   ← typed: G を行列とする
            "user_text": "E1 と Sx_1 の交換子を計算して"  ← typed: H と Sx_1 …
            "user_text": "Expand E4"                     ← typed: Expand F

        One field was carrying two meanings, so both rewrites leaked together.
        """
        step_id = self._alloc_step_id()
        extra: dict = {}
        if candidates:
            extra["candidates"] = candidates
        step = Step(
            step_id=step_id,
            kind=StepKind.llm_supervision,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.llm_interpreted", self.lang,
                             operation=operation or t("engine.llm_unknown", self.lang)),
            user_text=user_text,
            operation=operation,
            operation_args={**(args or {}), **extra},
            state_affecting=False,
            status=StepStatus.success if confirmed else StepStatus.failed,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        return step

    def delete_expression(self, expr_id: str) -> None:
        """Delete an expression that has not been used as input to any operation.

        Finds the expression_registration step that created it, re-wires the
        parent chain around that step (same pattern as delete_note_step), and
        removes both the step and the expression entry.
        """
        if expr_id not in self.expressions:
            raise ValueError(f"Expression '{expr_id}' not found")

        # Refuse if any step references this expression as an input
        for step in self.steps.values():
            if expr_id in step.input_refs:
                raise ValueError(
                    f"Expression '{expr_id}' is referenced by step '{step.step_id}' "
                    "and cannot be deleted"
                )

        # Find the step that created this expression
        creator_step_id = next(
            (sid for sid, s in self.steps.items() if expr_id in s.output_refs),
            None,
        )
        if creator_step_id is None:
            # Expression exists but has no creation step — just remove it
            del self.expressions[expr_id]
            return

        creator = self.steps[creator_step_id]
        predecessor_ids = creator.parent_step_ids

        # Re-wire: replace every reference to creator_step_id with its predecessor
        for sid, s in list(self.steps.items()):
            if creator_step_id in s.parent_step_ids:
                new_parents = [
                    (predecessor_ids[0] if predecessor_ids else "")
                    if p == creator_step_id else p
                    for p in s.parent_step_ids
                ]
                new_parents = [p for p in new_parents if p]
                self.steps[sid] = s.model_copy(update={"parent_step_ids": new_parents})

        # Update branch head if it pointed to the creator step
        for bid, branch in list(self.branches.items()):
            if branch.head_step_id == creator_step_id:
                new_head = predecessor_ids[0] if predecessor_ids else None
                self.branches[bid] = branch.model_copy(update={"head_step_id": new_head})

        del self.steps[creator_step_id]
        del self.expressions[expr_id]

    def delete_user_input_step(self, step_id: str) -> None:
        """Delete a user_input step and re-wire the parent chain around it.

        user_input steps are non-state-affecting (they carry only the raw text
        the user typed) so removal never breaks CAS state.
        """
        step = self.steps.get(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found")
        if step.kind != StepKind.user_input:
            raise ValueError(f"Only user_input steps can be deleted via this method (got {step.kind})")
        self._rewire_parent_chain(step_id, step.parent_step_ids)
        del self.steps[step_id]

    def _rewire_parent_chain(self, step_id: str, predecessor_ids: list) -> None:
        """Replace every reference to step_id in parent chains with its predecessors."""
        for sid, s in list(self.steps.items()):
            if step_id in s.parent_step_ids:
                new_parents = [
                    (predecessor_ids[0] if predecessor_ids else "")
                    if p == step_id else p
                    for p in s.parent_step_ids
                ]
                new_parents = [p for p in new_parents if p]
                self.steps[sid] = s.model_copy(update={"parent_step_ids": new_parents})
        for bid, branch in list(self.branches.items()):
            if branch.head_step_id == step_id:
                new_head = predecessor_ids[0] if predecessor_ids else None
                self.branches[bid] = branch.model_copy(update={"head_step_id": new_head})

    def delete_note_step(self, step_id: str) -> None:
        """Delete a manual_note step and re-wire the parent chain around it.

        Only manual_note steps may be deleted (they are state_affecting=False,
        so removal never breaks CAS state).  Any step that references the
        deleted step as a parent is updated to point to the deleted step's own
        parent, keeping the lineage intact.
        """
        step = self.steps.get(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found")
        if step.kind != StepKind.manual_note:
            raise ValueError(f"Only manual_note steps can be deleted (got {step.kind})")

        self._rewire_parent_chain(step_id, step.parent_step_ids)
        del self.steps[step_id]

    _CANCELLABLE_DEFINE_OPS: frozenset = frozenset({
        "define_scalar_parameter", "define_vector", "define_matrix", "define_tensor",
        "define_boson_operator", "define_fermion_operator", "define_spin_operator",
        "define_function", "define_vector_field", "define_scalar_field",
    })

    def delete_define_step(self, step_id: str) -> None:
        """Delete a define operation step, unregister its symbols, and rewire the parent chain.

        Removes the symbol(s) from SymbolContext so the UI no longer treats them
        as defined.  The CAS-side declaration persists in the running process but
        is harmless — it is simply a phantom parameter that will never appear in
        a well-formed expression going forward.
        """
        step = self.steps.get(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found")
        if step.operation not in self._CANCELLABLE_DEFINE_OPS:
            raise ValueError(
                f"Step operation '{step.operation}' is not a cancellable define operation"
            )

        op   = step.operation
        args = step.operation_args

        # Determine all symbols registered by this step.
        if "registered_names" in args:
            names_to_unregister: list[str] = list(args["registered_names"])
        elif op in ("define_scalar_parameter", "define_vector", "define_matrix", "define_tensor",
                    "define_boson_operator", "define_fermion_operator", "define_function"):
            names_to_unregister = [args.get("name", "")]
        elif op == "define_spin_operator":
            base = args.get("name", "")
            names_to_unregister = [base] + [f"{base}{s}" for s in ("x", "y", "z", "p", "m")]
        elif op in ("define_vector_field", "define_scalar_field"):
            names_to_unregister = list(args.get("registered_names", [args.get("name", "")]))
        else:
            names_to_unregister = []

        for name in names_to_unregister:
            if name:
                self.symbol_context.registry.pop(name, None)

        predecessor_ids = step.parent_step_ids

        for sid, s in list(self.steps.items()):
            if step_id in s.parent_step_ids:
                new_parents = [
                    (predecessor_ids[0] if predecessor_ids else "")
                    if p == step_id else p
                    for p in s.parent_step_ids
                ]
                new_parents = [p for p in new_parents if p]
                self.steps[sid] = s.model_copy(update={"parent_step_ids": new_parents})

        for bid, branch in list(self.branches.items()):
            if branch.head_step_id == step_id:
                new_head = predecessor_ids[0] if predecessor_ids else None
                self.branches[bid] = branch.model_copy(update={"head_step_id": new_head})

        del self.steps[step_id]

    def record_quantity_update(self, label: str, text: str) -> Step:
        """Record a quantity-registry change as a named timeline entry.

        Appears in the session timeline so that operations like set_latex_name,
        set_notation, and declare_time_dependent are visible after reloading.
        """
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.manual_note,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=label,
            user_text=text,
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return step

    def record_manual_note(self, text: str) -> Step:
        """Record a user-written research note as a first-class timeline step.

        Unlike user_input/llm_supervision, manual notes advance the branch head
        so that they appear in the correct chronological position in the lineage.
        """
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.manual_note,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.memo", self.lang),
            user_text=text,
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return step

    def fork_branch(self, step_id: str, label: str = "") -> Branch:
        """Live in-session fork (docs/developer_manuals/02_history_spec.md §11):
        create a new Branch headed at step_id and switch to it. The source
        branch's own head_step_id is never touched, so it is left unchanged.

        Unlike /session/fork (ui/routes/session.py), which saves a trimmed
        session as a NEW FILE, this creates a second branch that coexists with
        the others in the SAME loaded engine — required for
        numeric_branch_comparison() to ever have two branches to compare.
        """
        if step_id not in self.steps:
            raise ValueError(f"Step '{step_id}' not found")
        source_branch_id = self.steps[step_id].branch_id
        new_id = self._alloc_branch_id()
        new_branch = Branch(
            branch_id=new_id, label=label,
            head_step_id=step_id, forked_from_step_id=step_id,
        )
        self.branches[new_id] = new_branch
        self.session.active_branch_id = new_id
        fork_step_id = self._alloc_step_id()
        fork_step = Step(
            step_id=fork_step_id,
            kind=StepKind.branch_fork,
            branch_id=new_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.branch_forked", self.lang, branch_id=new_id),
            operation_args={
                "forked_from_step_id": step_id,
                "source_branch_id": source_branch_id,
                "new_branch_id": new_id,
            },
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[fork_step_id] = fork_step
        self._advance_head(fork_step)
        return new_branch

    def switch_branch(self, branch_id: str) -> None:
        """Change which branch subsequent operations are recorded against.

        Pure session-cursor state (like session_io._current_filename) — no
        history step is recorded, since switching is not notebook content.
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch '{branch_id}' not found")
        self.session.active_branch_id = branch_id

    def record_ambiguity_resolution(
        self,
        user_text: str,
        chosen_operation: str,
        chosen_args: dict,
        candidates: list[dict],
    ) -> Step:
        """Record that the user chose one candidate from ambiguous interpretations."""
        step_id = self._alloc_step_id()
        args_summary = ", ".join(f"{k}={v}" for k, v in chosen_args.items())
        step = Step(
            step_id=step_id,
            kind=StepKind.ambiguity_resolution,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.disambiguate", self.lang, chosen_operation=chosen_operation),
            user_text=user_text,
            operation=chosen_operation,
            operation_args={"chosen": chosen_args, "candidates": candidates},
            user_confirmation=f"{chosen_operation}（{args_summary}）",
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        return step

    def _active_expr_ids(self) -> set[str]:
        """Return expression IDs accessible from the active branch's lineage.

        Expressions created on other branches after the fork point are excluded,
        so operations on a forked branch cannot accidentally reference main-branch
        expressions that did not exist at fork time.
        """
        from notebook.runtime.replay import compute_lineage
        lineage = compute_lineage(self.active_branch_id, self.steps, self.branches)
        ids = {ref for s in lineage for ref in (s.output_refs or [])}
        visible = {eid for eid, e in self.expressions.items() if not e.hidden}
        available = ids & visible
        # Fall back to all non-hidden expressions only for an empty session
        return available if available else visible

    #: 式番号の綴り。`most_recent_expr_id` だけが読む。
    _EXPR_NUM_RE = re.compile(r"E(\d+)$")

    def most_recent_expr_id(self) -> "str | None":
        """この枝から見た**直前の式**。名指しの無い指示が落ちる先。

        **この問いに、この系は3通りで答えていた**（実測 2026-09-11、
        open_items OPERAND-BOUNDARY 残②）——登録順の最後（行列単項・q̇ 規則）、
        E 番号の最大（`interpret_plan._find_target`・候補畳み込み）、そして
        足場（step 2d）は自前の系譜 walk を持っていた。落ちてよいかは既に1つに
        畳んである（`operand_boundary.may_fall_back_to_the_recent_expression`）
        ので、**どれに落ちるか**もここで1つにする。

        実測した限り**到達できる状態では3つとも一致する**——コーパス1102行、
        保存済みセッション18本、分岐（fork → 枝で登録 → main に戻る）で差分0。
        にもかかわらず畳むのは、**割れる状態が2つ具体的に在る**からである:

        - **隠れた式が最大番号を持つとき。** 足場と候補畳み込みだけが `hidden`
          を除いておらず、研究者から**意図的に隠した式**を答えに選ぶ
          （実測: 他が E2、この2箇所が E3）。いま到達しないのは、E 番号を
          消費する唯一の隠し場所（`engine_plan_mixin`）が `PlanStep.present`
          が False のときだけ働き、それを False にする側がリポジトリに
          1箇所も無いからにすぎない。
        - **可視の内部式（`_Tn`）が登録順の最後に来たとき。** 登録順で答える
          2箇所だけが `_T1` を返す——研究者が一度も見たことのない鍵である
          （open_items SCAFFOLD-LABEL が同じ漏れを別の口で記録している）。
          いま到達しないのは、8つある内部登録がすべて直後に E 番号の結果を
          作るか隠されるかしているからで、9つ目が同じ形とは限らない。

        採るのは**いちばん保守的な答え**——非 hidden かつ系譜内（`_active_expr_ids`）
        の中で E 番号が最大のもの。隠した式も、研究者が見ていない鍵も返さない。
        """
        active = self._active_expr_ids()
        return max((eid for eid in active if eid in self.expressions),
                   key=lambda e: (int(m.group(1)) if (m := self._EXPR_NUM_RE.match(e))
                                  else -1),
                   default=None)

    def _visible_action(self, operation: str, args: dict,
                        is_total_derivative: bool = False,
                        display_hints: dict | None = None) -> str:
        lang = self.lang
        hints = display_hints or {}
        # A handler that ran the operation on a SCAFFOLD says here what the
        # researcher would call it; the wording (particles, both-sides short
        # forms) still happens below, once (open_items SCAFFOLD-LABEL).
        operation = hints.get("label_operation") or operation
        spec = self._validator.get_spec(operation)
        display = (spec["display"].get(lang) or spec["display"].get("en") or operation) if spec else operation

        expr_ref = (hints.get("label_expr_ref") or hints.get("label_rhs_of")
                    or args.get("expr") or args.get("equation")
                    or args.get("a") or args.get("operator"))
        if hints.get("label_rhs_of"):
            # "E1 の右辺を簡約する" — the operation ran on the right-hand side,
            # which is a thing the researcher wrote and `E1` alone does not say.
            return t("engine.rhs_operation", lang,
                     eid=hints["label_rhs_of"], display=display)

        # "両辺" operations on a non-equation expression: drop the "both sides" prefix
        _BOTH_SIDES_OPS = {
            "divide_both_sides":        (t("engine.divide_both_sides", lang),
                                         t("engine.divide_both_sides_short", lang)),
            "multiply_both_sides":      (t("engine.multiply_both_sides", lang),
                                         t("engine.multiply_both_sides_short", lang)),
            "add_to_both_sides":        (t("engine.add_to_both_sides", lang),
                                         t("engine.add_to_both_sides_short", lang)),
            "subtract_from_both_sides": (t("engine.subtract_from_both_sides", lang),
                                         t("engine.subtract_from_both_sides_short", lang)),
        }
        if operation in _BOTH_SIDES_OPS and expr_ref:
            src_expr = self.expressions.get(expr_ref, None)
            # "Is this an equation?" asked of BOTH forms, through the parser's
            # own bracket-aware scan.  Asking `"=" in canonical` was measured
            # wrong on 2026-08-27: an H-1 equation keeps `=` in its LATEX only
            # (CLAUDE.md — the canonical of `f = …` stores the RHS alone), so
            # every `E1 の両辺を 2 で割る` on a NAMED equation took the
            # non-equation short form and read `E1 を割る`.  It went unseen
            # because the label itself never reached a card until today.
            if src_expr is not None and not any(
                    find_top_level_relation(t or "")
                    for t in (src_expr.canonical, src_expr.latex)):
                display = _BOTH_SIDES_OPS[operation][1]

        # For commutator/anticommutator: show bracket notation with both operands.
        if operation in ("commutator", "anticommutator"):
            a_disp = hints.get("bracket_a") or args.get("a", "")
            b_disp = hints.get("bracket_b") or args.get("b", "")
            open_b, close_b = ("[", "]") if operation == "commutator" else ("{", "}")
            return t("engine.action_on_expr", lang,
                     expr_ref=f"{open_b}{a_disp}, {b_disp}{close_b}",
                     display=display)

        # For substitute: show "lhs → rhs in ExprID".
        # Through the display door, the same one `_format_substitute_latex`
        # uses — this label is shown to the RESEARCHER, so it must carry their
        # spelling, not the canonical.  Two places answered "how do I show a
        # substitution's two sides?" and only one of them asked (measured
        # 2026-08-23: the timeline read `rho → lambda/mu` where the researcher
        # wrote `\rho` and `\lambda/\mu`).
        # Substitution has its own sentence, and RETURNS — like commutator above.
        # It used to build `[{expr_ref}] lhs → rhs` and then fall through to
        # `engine.action_on_expr`, which attaches the reference AGAIN: the
        # timeline read `E1 を[E1] x → …` in Japanese (the template's own
        # `{expr_ref} を` prefix leaving the bracket mid-sentence) and
        # `[E1] q → p [E1]` in English — measured 2026-08-27, the E-id printed
        # twice in both.  Its own key rather than the generic wrapper because
        # that wrapper's Japanese is `{expr_ref} を{display}`, written for a
        # display that BEGINS with a Japanese verb; a substitution's begins with
        # the researcher's own symbol, so `E1 をx → …` runs the two together.
        if operation == "substitute":
            lhs_s = self._display_arg(str(args.get("lhs", "")))
            rhs_s = self._display_arg(str(args.get("rhs", "")))
            if expr_ref and lhs_s and rhs_s:
                return t("engine.substitute_action", lang,
                         expr_ref=expr_ref, lhs=lhs_s, rhs=rhs_s)

        # For differentiate: label as total/partial based on context.
        if operation == "differentiate" and expr_ref:
            # **同じ扉**——このラベルは研究者に見せるものなので、正準形ではなく
            # 研究者の綴りを運ぶ（すぐ上の substitute の註と同じ理由）。変数を
            # `latex_to_canonical` に通すようにした日（2026-09-19、
            # DECORATED-NAME-AS-DIFF-VAR）に、ここだけが取り残されてカードが
            # `E2 をlambda で偏微分する` / `E66 をy_bar で偏微分する` と**内部名**を
            # 印字するようになった。コーパス差分が捕まえた——ただし**欄の名前を
            # 取り違えた比較器では 0 と出ていた**（`args` を `operation_args` と
            # 書いていた）。
            var = self._display_arg(str(args.get('var', '')))
            if is_total_derivative:
                display = (t("engine.total_diff", lang, var=var) if var
                           else t("engine.total_diff_novar", lang))
            else:
                src = self.expressions.get(expr_ref)
                if src:
                    if src.quantity_name or src.derivative_origin:
                        if src.derivative_is_partial:
                            display = (t("engine.partial_diff", lang, var=var) if var
                                       else t("engine.partial_diff_novar", lang))
                    else:
                        is_partial = False
                        _rel = split_relation(src.canonical)
                        if _rel is not None and _rel[1] == "=":
                            lhs = _rel[0]
                            qty = self.symbol_context.get_quantity(self._handler.qty_name_from_origin(lhs))
                            if qty is not None:
                                is_partial = len(qty.natural_vars) > 1
                            else:
                                rhs = _rel[2]
                                is_partial = len(self._handler.canonical_variables(rhs)) > 1
                        else:
                            is_partial = len(self._handler.canonical_variables(
                                src.display_repr
                            )) > 1
                        if is_partial:
                            display = (t("engine.partial_diff", lang, var=var) if var
                                       else t("engine.partial_diff_novar", lang))
                        elif var:
                            display = t("engine.diff", lang, var=var)

        if not expr_ref:
            return display
        # **A display that already takes its own object is a possessor, not an
        # object.**  The Japanese wrapper is `{expr_ref} を{display}`, written for
        # a display like `展開する`; 70 of the 103 operations' Japanese displays
        # already carry a `を` of their own (`方程式を解く`, `固有値を求める`,
        # `実部を取り出す`), and those came out as `E5 を方程式を解く` — two
        # objects, one verb.  Measured 2026-08-27: **32 such labels were already
        # published** in the worked examples, and every one of the 70 reads
        # correctly with `の` (`E5 の方程式を解く`).  English needs no such
        # distinction, and its two templates are identical.
        # …and a display whose own object is 両辺 needs it too, even though the
        # particle marking that object is に or から rather than を.  Measured
        # 2026-09-05 over all 103 displays: eight carry a case particle without
        # a を, and only the three 両辺 ones want の — 両辺 is something the
        # EXPRESSION HAS, while `極形式に変換する` / `数値的に解く` mark a goal
        # and a manner, and those read correctly as `E1 を…`.  The three are
        # already enumerated just above, so this asks that table rather than
        # spelling a Japanese word here (and only for the LONG form: the short
        # one is chosen exactly when the expression is not an equation, and
        # `E1 を乗じる` is right).
        _long_both_sides = (operation in _BOTH_SIDES_OPS
                            and display == _BOTH_SIDES_OPS[operation][0])
        _key = ("engine.action_on_expr_of" if ("を" in display or _long_both_sides)
                else "engine.action_on_expr")
        return t(_key, lang, expr_ref=expr_ref, display=display)

