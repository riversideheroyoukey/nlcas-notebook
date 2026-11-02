"""SCALAR-HANDOFF: hand a finished scalar expression to another backend's session.

The source-session half.  Validates that the expression may cross, snapshots
its provenance, and records the crossing as a TIMELINE STEP — which session a
result came from is part of what the result means, the same reason F-13 made
the spin declaration a step rather than a hidden setting.

Design and the manual measurement this mechanises:
docs/developer_manuals/34 §5-6.
"""
from __future__ import annotations

from datetime import datetime, timezone

from notebook.models.handoff import ScalarHandoff
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.i18n import t
from notebook.runtime.scalar_handoff import (
    HandoffRefused, collect_declarations, operator_symbols, split_export_form,
)
from notebook.runtime.shared_parser import UnknownNotationError


def _export_latex(expr, drop_label: bool) -> str:
    """The LaTeX that travels: the whole equation, or its value half."""
    text = expr.latex or expr.canonical or ""
    return split_export_form(text)[1] if drop_label else text


def _label_latex(expr) -> str:
    return split_export_form(expr.latex or expr.canonical or "")[0]


class EngineHandoffMixin:
    """Mixed into NotebookEngine."""

    def _handoff_source_text(self, expr) -> str:
        """The expression as the SOURCE session can still read it.

        Prefer the user's own writing: a Julia RESULT's canonical is an opaque
        `_nb_Ek` handle that names nothing, while `input_canonical` / the LaTeX
        keep what the researcher wrote (docs/32).
        """
        text = (getattr(expr, "input_canonical", "") or "").strip()
        if not text or text.startswith("_nb_"):
            text = (expr.canonical or "").strip()
        if not text or text.startswith("_nb_"):
            text = expr.latex or ""
            text = self._handler.latex_to_canonical(text) if text else ""
        return text

    def _handoff_symbol_names(self, text: str) -> "set[str]":
        """Every symbol name appearing in *text*, as the source session reads it."""
        from notebook.runtime.maxima_semantic_resolver import _collect_symbols
        names: set[str] = set()
        if not text:
            return names
        try:
            from notebook.runtime.shared_parser import SharedParser
            ir = self._handler.parse_ir(text) if hasattr(self._handler, "parse_ir") else None
            if ir is None:
                from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
                ir = MaximaSemanticResolver().resolve(
                    SharedParser().parse(text), self.symbol_context)
            _collect_symbols(ir, names)
        except (UnknownNotationError, ValueError) as _exc:
            # An expression we cannot parse cannot be checked for operators, so
            # the caller must not be told it is clean — surface it as a refusal
            # rather than silently handing over an unchecked expression.
            # Narrow on purpose: these two are what a parse/resolve failure
            # raises here (pydantic's ValidationError is a ValueError).  Any
            # OTHER exception is a bug in this code path and must not be
            # dressed up as "the researcher wrote something unparsable".
            raise HandoffRefused("handoff.unparsable", expr=text[:80]) from _exc
        return names

    def build_scalar_handoff(self, expr_id: str,
                             target_backend: str = "maxima") -> ScalarHandoff:
        """Everything ``fork_scalar_handoff`` does EXCEPT recording it.

        Split so the UI can show what a handoff WOULD carry before the
        researcher commits to it, without the preview leaving a step in the
        timeline.  One body, two callers — a second implementation of "what
        crosses and what travels with it" is exactly how the two would drift.
        """
        expr = self.expressions.get(expr_id)
        if expr is None:
            raise ValueError(f"Expression '{expr_id}' not found")

        # What crosses is the VALUE; a result may wear a LABEL saying what it is
        # of (`⟨c c⟩ = uv/…`).  Check the value — an expectation value is a
        # c-number however operator-laden its label (see split_export_form).
        label_text, value_text = split_export_form(self._handoff_source_text(expr))
        names = self._handoff_symbol_names(value_text)
        ops = operator_symbols(names, self.symbol_context)
        if ops:
            raise HandoffRefused("handoff.operators_present",
                                 names=", ".join(ops), expr_id=expr_id)
        # An operator label cannot travel — there is nothing in a scalar CAS for
        # it to mean.  Drop it from what is registered, but keep it in the
        # provenance: which quantity this is the value of is part of what the
        # value means, and the target session's note is where that has to survive.
        label_ops = (operator_symbols(self._handoff_symbol_names(label_text),
                                      self.symbol_context)
                     if label_text else [])

        # Declarations to carry: every c-number the SOURCE SESSION knows, not
        # only the ones appearing in this expression.  Measured 2026-07-31: the
        # S-c continuation integrates the carried integrand up to `omega_D`, a
        # bound that is declared in the source but appears nowhere in the
        # expression that crosses — carrying only the expression's symbols left
        # it undeclared in the target, where it read as the INDEXED `omega(D)`
        # and the answer came back `asinh(omega(D)/Delta)`.  A declaration is
        # not a value, so carrying the working set costs nothing; making the
        # researcher re-declare by hand reintroduces exactly the friction this
        # feature removes.  Operator declarations are excluded — they have no
        # meaning in a scalar CAS.
        _carry = {n for n in self.symbol_context.registry
                  if not operator_symbols({n}, self.symbol_context)}
        decls, _ = collect_declarations(_carry, self.symbol_context)
        _, undeclared = collect_declarations(names, self.symbol_context)
        handoff = ScalarHandoff(
            handoff_id=f"H{len(getattr(self, 'scalar_handoffs', {})) + 1}",
            source_expression_ids=[expr_id],
            source_latex=expr.latex or "",
            source_canonical=expr.canonical or "",
            source_assumption_ids=sorted(self.assumptions.keys()),
            source_branch_id=self.active_branch_id,
            source_backend=self.session.primary_backend,
            source_title=self.session.title or "",
            target_backend=target_backend,   # type: ignore[arg-type]
            declarations=decls,
            undeclared=undeclared,
            export_latex=_export_latex(expr, drop_label=bool(label_ops)),
            source_label_latex=(_label_latex(expr) if label_ops else ""),
            source_value_canonical=value_text,
        )
        return handoff

    def fork_scalar_handoff(self, expr_id: str,
                            target_backend: str = "maxima") -> ScalarHandoff:
        """Prepare *expr_id* to continue in a *target_backend* session, and
        RECORD the crossing — which session a result came from is part of what
        it means (the reason F-13 made its declaration a step, not a setting)."""
        handoff = self.build_scalar_handoff(expr_id, target_backend)
        if not hasattr(self, "scalar_handoffs"):
            self.scalar_handoffs = {}
        self.scalar_handoffs[handoff.handoff_id] = handoff

        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.scalar_handoff,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("handoff.forked", self.lang,
                             expr_id=expr_id, backend=target_backend),
            operation="fork_scalar_handoff",
            operation_args={"handoff_id": handoff.handoff_id,
                            "target_backend": target_backend},
            input_refs=[expr_id],
            state_affecting=False,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return handoff
