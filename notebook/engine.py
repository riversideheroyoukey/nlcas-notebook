from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger(__name__)

from notebook.models.branch import Branch
from notebook.models.expression import Assumption, Expression
from notebook.models.quantity import MathType, Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.models.session import Session, SessionMode
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.expr_ref_fields import is_expr_ref
from notebook.runtime.backend_factory import default_backend_for_mode, make_backend
from notebook.runtime.indexed_display import (
    record_greek_glyphs, record_symbol_usage,
)
from notebook.runtime.backend_handler import (
    BackendHandler, MATH_FUNC_NAMES, beautify_bare_canonical_latex,
    canonical_name_to_latex)
from notebook.runtime.branch_manager import BranchManager
from notebook.runtime.operand_guards import (
    reject_conflicting_redeclaration, reject_declared_object_without_value,
    reject_structure_only_operation, reject_unusable_operands,
    translate_indexed_variable_failure,
    translate_value_bearing_variable_failure, word_operand_warnings)
from notebook.runtime.placeholder_sum_bounds import BoundLedger
from notebook.runtime.i18n import DEFAULT_LANG, t
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.replay import ReplayResult, replay
from notebook.runtime.shared_parser import (
    UnknownNotationError, find_top_level_relation, split_relation, strip_lhs)
from notebook.engine_history_mixin import EngineHistoryMixin
from notebook.engine_pair_rep_mixin import EnginePairRepMixin
from notebook.engine_deferred_forms_mixin import EngineDeferredFormsMixin
from notebook.engine_angular_momentum_mixin import EngineAngularMomentumMixin
from notebook.engine_spin_species_mixin import EngineSpinSpeciesMixin
from notebook.engine_fourier_mixin import EngineFourierMixin
from notebook.engine_free_params_mixin import EngineFreeParamsMixin
from notebook.engine_laplace_mixin import EngineLaplaceMixin
from notebook.engine_handoff_mixin import EngineHandoffMixin
from notebook.engine_numeric_mixin import EngineNumericMixin
from notebook.engine_persistence_mixin import EnginePersistenceMixin
from notebook.engine_plan_mixin import EnginePlanMixin
from notebook.engine_wick_mixin import EngineWickMixin


# Operations whose rendered_command must be replayed into a fresh CAS session
# after engine.load() / engine.start() so the CAS has the needed declarations.
_DECLARATION_OPS: frozenset[str] = frozenset({
    "define_fermion_operator",
    "define_boson_operator",
    "define_spin_operator",
    "define_scalar_parameter",
    "define_vector_field",
    "define_scalar_field",
    "define_coordinate_system",
})


class UndefinedSymbolsError(ValueError):
    """Raised when a CAS operation references expressions with undefined symbols."""
    def __init__(self, message: str, symbols: list[str]) -> None:
        super().__init__(message)
        self.symbols = symbols

# Operations that compute a quantity FROM the expression value.
# When the source expression is a function-definition equation (f(x) = rhs),
# only the LHS is passed so the operation acts on the function itself,
# not on the equation object (which would transform both sides independently).
_ANALYTICAL_OPS = frozenset({
    "differentiate",
    "integrate_indefinite", "integrate_definite",
    "limit",
    "taylor_expand", "series_expand",
    "laplace_transform", "inverse_laplace",
    "summation", "product_expr",
    "numerical_eval", "numerical_eval_hp",
})

#: Operations whose result is a DIFFERENT quantity from the source, and which
#: therefore show the operator wrapped around the source's name:
#: ``\int_{0}^{1} Y \, dx = …``, ``\sum_{i=1}^{3} Y = 3Y``.  Re-attaching a
#: bare ``Y =`` would assert something false.
#:
#: One definition because it was two — ``_INTEGRAL_LIMIT_OPS`` (which decides
#: whether to strip the LHS) and ``_EQ_FORM_OPS`` (which decides how to write
#: the result) were the same frozenset spelled twice, so Σ and Π had to be
#: added to both or neither.  They were in neither, and sent the equation WHOLE
#: (open_items EQ-SOURCE-BARE-LHS).
_EQ_OPERATOR_FORM_OPS = frozenset({
    "integrate_indefinite", "integrate_definite", "limit",
    "summation", "product_expr",
})

#: Operations that preserve the VALUE, so the source's name still names the
#: result: ``Y = 5.0``.  ``numerical_eval`` is here rather than above because
#: evaluating a definition numerically does not change what it is.
_EQ_VALUE_PRESERVING_OPS = frozenset({
    "simplify", "expand", "factor", "ratsimp",
    "normal_order", "vacuum_expectation",
    "numerical_eval", "numerical_eval_hp",
})

# Complex-analysis value ops (realpart/imagpart/rectform/conjugate).  On an
# equation Z = expr these must act on the RHS value, not on the equation object:
# realpart(Z = expr) mangles it (imagpart happened to look right only because
# imagpart(Z)=0).  Unlike simplify/factor they do NOT preserve equality, so the
# result is shown plainly (R, ωL−1/(ωC)) without re-attaching "Z =".
_COMPLEX_VALUE_OPS = frozenset({
    "real_part", "imaginary_part", "rectform", "complex_conjugate",
})

# Standard math function names excluded from variable counting.
# Alias for the shared frozenset exported from backend_handler.
_MATH_FUNC_NAMES_ENGINE: frozenset[str] = MATH_FUNC_NAMES



from notebook.runtime.time_order import find_bare_ordering
from notebook.runtime.bound_name_collision import (
    find_collisions as find_bound_name_collisions,
)
from notebook.runtime.lhs_formatting import (
    _equation_form_latex,
    _format_provenance_lhs,
    _format_derivative_lhs,
    _DERIV_BUMP_RE,
    _bump_same_var_derivative,
    _format_wrt_qdot_lhs,
    _SIMPLE_SYM_RE,
    _DECORATED_SYM_RE,
    _EXPECT_RE,
    _DERIV_RE,
    _DECO_SUFFIX,
    _extract_simple_symbol_name,
    _normalize_lhs,
    _lhs_latex_from_expr,
    _rhs_of_latex,
    _COMBINE_OP_SYMBOLS,
    _has_toplevel_additive,
)


class OperationRefused(ValueError):
    """A backend builder refused to build the command, with a reason.

    A ``ValueError`` still, so every existing caller keeps working; what it adds
    is ``reason_code`` — a stable identifier for callers that must tell one
    refusal from another.  Before this, the only thing distinguishing them was
    the researcher-facing message, which is translated, so a caller keying on it
    would be matching Japanese prose (TRANSFORM-SUBST-CHAIN, 2026-08-15).

    Set deliberately and rarely.  A refusal with ``reason_code is None`` is one
    nobody has needed to distinguish, which is the honest default — inventing a
    code per refusal would produce a vocabulary that no caller reads and that
    drifts from the messages.
    """

    def __init__(self, message: str, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _args_with_derivative_kind(args: dict, is_total: bool,
                               is_partial: bool) -> dict:
    r"""研究者が言った微分の種別を、ビルダにも渡す。

    **∂ と d は違う量なので、種別はラベルだけでなくコマンドを決める。**
    `_preprocess_args` が `_is_*` を剥がすのは**検証器**に見せないためであって、
    ビルダに隠すためではない——隠れていた結果、偏微分を頼まれたバックエンドが
    変数だけを見て全微分を組み、**ラベルだけが研究者の要求どおり**だった
    （実測 2026-09-06: 駆動振動子 `L = q̇²/2 − q²/2 + F q cos(ωt)` の `∂L/∂t` が
    `−F q ω sin(ωt) + F q̇ cos(ωt) + q̇q̈ − q q̇`、正しくは第1項だけ。
    `dH/dt = −∂L/∂t` はまさにこの量についての主張である）。

    何を組むかはバックエンドの知識なので、ここは**言われたことを渡すだけ**で、
    どう組むかは決めない（原則#10）。
    """
    if not (is_total or is_partial):
        return args
    return {**args, "_derivative_kind": "total" if is_total else "partial"}


class NotebookEngine(EngineHistoryMixin, EnginePersistenceMixin,
                     EngineLaplaceMixin, EngineFreeParamsMixin,
                     EngineDeferredFormsMixin, EngineWickMixin,
                     EngineNumericMixin, EngineHandoffMixin, EnginePlanMixin,
                                          EngineFourierMixin,
                     EngineAngularMomentumMixin, EngineSpinSpeciesMixin, EnginePairRepMixin):
    """In-memory session state and high-level notebook operations.

    Wires together adapter → validator → builder → branch_manager → replay.
    The CLI and tests call this class; none of them touch CAS syntax directly.
    """

    def __init__(
        self,
        title: str = "Untitled",
        mode: SessionMode = SessionMode.standard_symbolic,
        primary_backend: str | None = None,
        coordinate_system: str | None = None,
    ) -> None:
        # Explicit primary_backend takes priority; otherwise the mode's default,
        # which backend_factory owns along with the backend-name registry — the
        # engine must not hold a second table of CAS names.
        if primary_backend is None:
            primary_backend = default_backend_for_mode(mode)
        self.session = Session(
            title=title, mode=mode, primary_backend=primary_backend,
            coordinate_system=coordinate_system,
        )
        self.expressions: dict[str, Expression] = {}
        self.assumptions: dict[str, Assumption] = {}
        self.steps: dict[str, Step] = {}
        self.branches: dict[str, Branch] = {
            "main": Branch(branch_id="main", label="main")
        }
        self.session.active_branch_id = "main"
        self._branch_counter = 0
        self.symbol_context = SymbolContext()
        self.symbol_context.domains = self.session.domains
        self.symbol_context.session_mode = self.session.mode.value
        # LHS alias table: maps normalized LHS display string → expr_id.
        # Populated automatically whenever an expression with an equation-form
        # latex ("LHS = RHS") is registered or produced by an operation.
        # Enables "H + V を展開して", "\bar{H} を計算して" etc. to resolve
        # directly to the expression without knowing its E-ID.
        self.lhs_alias_map: dict[str, str] = {}

        # Numeric layer (N-1): tasks / artifacts / executions, keyed by id.
        from notebook.models.numeric import (
            NumericArtifact, NumericExecution, NumericTask,
        )
        self.numeric_tasks: dict[str, NumericTask] = {}
        self.numeric_artifacts: dict[str, NumericArtifact] = {}
        self.numeric_executions: dict[str, NumericExecution] = {}
        self._numeric_counter = 0

        # Plan layer (C-Ext-5): multi-step plans, keyed by plan_id.
        from notebook.models.plan import SupervisorPlan
        self.plans: dict[str, SupervisorPlan] = {}
        self._plan_counter = 0

        # Set by explicit save()/load(); anchors sessions/<stem>_numeric/.
        self._last_save_path = None

        self._expr_counter = 0
        self._internal_counter = 0
        # Placeholder bounds for sums written WITHOUT a range (open_items B-10
        # ②).  Session-wide so two such sums never share a pair — sharing one
        # makes Maxima identify their ranges, which for ranges we were never
        # told is a claim we invented.
        self._sum_bounds = BoundLedger()
        self._step_counter = 0
        self._adapter, self._builder, self._handler = make_backend(primary_backend)
        self._handler.set_symbol_context(self.symbol_context)
        # I18N-1: let the handler reach the session language for its user-facing
        # messages.  A live accessor, not a snapshot — start_engine assigns
        # session.language only AFTER this constructor returns.
        self._handler.bind_language_source(lambda: self.lang)
        self._validator = OperationValidator()
        # Transient state for the step-by-step transfer-function derivation (S-e):
        # the source Laplace equation and the detected output/input symbols.
        self._tf_source_eid: str | None = None
        self._tf_output: str = "q"
        self._tf_input: str = "v"
        # Transient: holds a partial operation waiting for one missing argument.
        # Cleared after the follow-up is processed or when any unrelated input arrives.
        self.pending_op: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Re-sync domains: callers often set session.domains AFTER construction
        # (debug_repl, the example generator, session load), so the __init__ copy
        # would otherwise be stale and domain-gated behaviour (e.g. Lagrangian mode)
        # would not fire.
        self.symbol_context.domains = list(self.session.domains)
        self._adapter.start()
        self._init_coordinate_system()
        self._replay_declarations()
        self._replay_symbol_definitions()  # P2: Restore symbol definitions to Maxima

    def _init_coordinate_system(self) -> None:
        """Send coordinate-system setup commands to the CAS at engine start.

        Delegates to the backend handler: Maxima emits ``nb_*`` function
        definitions from ``session.coordinate_system``; Julia is a no-op.
        """
        self._handler.init_coordinate_system(self._adapter, self.session)

    def _replay_declarations(self) -> None:
        """Re-execute declaration commands into the CAS after session load.

        Collects rendered_command strings for every successful declaration step
        in the active branch lineage and passes them to the backend handler.
        For a brand-new session this is a no-op (no steps exist yet).
        """
        from notebook.runtime.replay import compute_lineage
        from notebook.models.step import StepKind, StepStatus
        lineage = compute_lineage(self.active_branch_id, self.steps, self.branches)
        decl_commands = [
            step.rendered_command
            for step in lineage
            if (
                step.kind == StepKind.cas_operation
                and step.status == StepStatus.success
                and step.operation in _DECLARATION_OPS
                and step.rendered_command
            )
        ]
        if decl_commands:
            self._handler.replay_session_declarations(self._adapter, decl_commands)

    def _replay_symbol_definitions(self) -> None:
        """P2: Re-send symbol definitions to Maxima after session start.

        When a session is loaded, symbol definitions (H = x^2) are stored in
        expressions but not sent back to Maxima unless we replay them.
        This method iterates lhs_alias_map and re-sends each symbol definition
        so that simplify(H) etc. will have access to the definitions.

        Only applies to backends that bind symbol definitions into CAS state
        (Maxima's evaluate-on-assignment); a no-op for structure-preserving
        backends (Julia).
        """
        if not self._handler.binds_symbol_definitions:
            return
        for lhs_key, expr_id in self.lhs_alias_map.items():
            if expr_id not in self.expressions:
                continue
            expr = self.expressions[expr_id]
            # Is the stored latex an equation at all?  `find(" = ")` was the
            # SPACED spelling of that question and answered it twice wrongly:
            # Maxima's `tex1` writes an equation UNSPACED (`x=e^{-t\gamma}…`,
            # present in three saved sessions), so a CAS-produced definition was
            # skipped here in silence; and `\sum_{i = 1}^{3}` — a space a
            # researcher may write — matched INSIDE the subscript and threw the
            # head away.  `split_relation` is the one definition, bracket-aware
            # and blind to spacing.
            _rel_lat = split_relation(expr.latex)
            if _rel_lat is None or _rel_lat[1] != "=" or not _rel_lat[2]:
                continue
            # The RHS the CAS is given comes from the CANONICAL, never the latex.
            # (The old `rhs_canonical = … if expr.canonical else rhs_latex`
            # fallback was dead — the next statement overwrote it before use —
            # and reviving it would send LaTeX to Maxima, so an expression with
            # no canonical is skipped instead.  It used to send `sym : $`, which
            # failed and was swallowed by the except below.)
            if not expr.canonical:
                continue
            rhs_canonical = strip_lhs(expr.canonical)

            # Extract symbol name from lhs_key
            sym = _extract_simple_symbol_name(lhs_key)
            if sym:
                # Skip "solve result" style definitions where the LHS symbol
                # appears on the RHS (e.g. q = -'diff(q,t,2)/omega^2 from
                # solve).  These are conditional equalities, not global
                # symbol bindings — replaying them would corrupt Maxima's
                # state and cause erroneous substitutions everywhere.
                import re as _re_rpl
                _lhs_in_rhs = bool(_re_rpl.search(
                    r'(?<![A-Za-z0-9_])' + _re_rpl.escape(sym) + r'(?![A-Za-z0-9_])',
                    rhs_canonical,
                ))
                if _lhs_in_rhs:
                    continue
                maxima_cmd = f"{sym} : {rhs_canonical}$"
                try:
                    self._adapter.execute(maxima_cmd)
                except Exception:
                    # Silently ignore failures (same as _send_symbol_definition_to_maxima)
                    pass

    def close(self) -> None:
        self._adapter.close()

    def __enter__(self) -> NotebookEngine:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_branch(self) -> Branch:
        return self.branches[self.session.active_branch_id]

    @property
    def active_branch_id(self) -> str:
        return self.session.active_branch_id

    @property
    def lang(self) -> str:
        # No recorded language = nobody to identify → i18n.DEFAULT_LANG
        # (docs/developer_manuals/15_i18n_design.md "Default language").
        return self.session.language[0] if self.session.language else DEFAULT_LANG

    def _translate_unknown_notation(self, exc: "Exception") -> ValueError:
        """Turn an UnknownNotationError (i18n key + kwargs) into the
        user-readable ValueError that run_operation / register_expression
        promise to raise on failure."""
        return ValueError(t(exc.i18n_key, self.lang, **exc.fmt_kwargs))

    # ------------------------------------------------------------------
    # Variable dependency declaration
    # ------------------------------------------------------------------

    def declare_depends(self, var: str, depends_on: list[str]) -> None:
        """Register a CAS-side variable dependency."""
        self._handler.declare_depends(self._adapter, var, depends_on)

    def _send_symbol_definition_to_maxima(self, symbol_name: str, expr_id: str) -> None:
        """P2: Send symbol definition to Maxima for auto-substitution in operations.

        When a new symbol is defined (H = x^2), extract the RHS canonical form
        and send `H : x^2` to Maxima so that later operations like simplify(H)
        automatically expand to the definition.

        No-op if:
        - Backend is not Maxima
        - expr_id does not exist
        - The expression is not an equation (no RHS to extract)
        """
        if expr_id not in self.expressions:
            return
        expr = self.expressions[expr_id]

        # Extract RHS from canonical
        # Two cases:
        # 1. canonical contains " = " (equation form): H = x^2
        # 2. canonical is RHS only (lhs_label used): x^2
        #    In this case, canonical itself is the RHS to substitute

        # Case 1 (equation form) and case 2 (canonical is the RHS already) are
        # the two answers `strip_lhs` gives, and it asks the bracket-aware
        # question: `\sum_{i = 1}^{3} i^2 = S` split at the subscript's `=` used
        # to bind the symbol to `1}^3*i^2 = S` (measured).  What goes to Maxima
        # below is this string, so a mis-split here is a wrong DEFINITION, not a
        # wrong display.
        rhs_canonical = strip_lhs(expr.canonical).strip()

        if not rhs_canonical:
            return

        # Detect user-defined symbols in the RHS that are already assigned in Maxima.
        # If any exist, temporarily kill them before the assignment so that Maxima
        # stores V symbolically (e.g. V : m*g*l*(1-cos(q)) keeps g unevaluated even
        # when g : l*omega^2 is already in Maxima).  After the assignment, restore
        # each killed symbol.
        #
        # We only kill symbols whose names appear as whole-word tokens in rhs_canonical
        # and that have a known expression in the engine (so we can restore them).
        import re as _re
        _to_restore: list[tuple[str, str]] = []  # (name, rhs_to_restore)
        for _sym, _eid in self.lhs_alias_map.items():
            if _sym == symbol_name:
                continue
            if not _re.search(rf'(?<![A-Za-z0-9_]){_re.escape(_sym)}(?![A-Za-z0-9_])', rhs_canonical):
                continue
            _dep_expr = self.expressions.get(_eid)
            if _dep_expr is None:
                continue
            _dep_rhs = strip_lhs(_dep_expr.canonical).strip()
            if _dep_rhs:
                _to_restore.append((_sym, _dep_rhs))

        # Apply q_dot / q_ddot → diff(q(t),t,N) so that Maxima can differentiate
        # time-dependent expressions stored via P2.  Without this, T stored as
        # (1/2)*m*l^2*q_dot^2 has q_dot as a plain symbol and diff(T,t) → 0.
        # The RHS is the researcher's writing, not CAS code, so it goes through
        # the lowering door.  Measured: `V = \int_0^{\infty} v e^{-s t} dt` bound
        # `V` to `%e^-(s*t)*dt*int_0^inf*v` — the `%int_…` family, an integral
        # never formed, carried silently into every later use.  A refusal SKIPS
        # the bind: an undefined symbol makes the next operation say so, a symbol
        # bound to garbage says nothing.  (This makes the bind SUCCEED where it
        # used to fail; `_shadow_own_definition` in the Maxima handler is what
        # keeps "solve this definition for its variable" from becoming vacuous.)
        # A refusal from the CODE GENERATOR (`\int d^3x …`) arrives as an
        # UnknownNotationError, not as status="refused": `to_backend_code`
        # re-raises that class so an OPERATION can translate it.  There is no
        # operation to fail here — this is the researcher writing something
        # down — so letting it out killed the whole registration and answered
        # `H = \int d^3x \psi^* T \psi` with a raw i18n key (measured
        # 2026-08-20).  Both shapes mean the same thing on this path: no bind,
        # and the equation is still written down.
        try:
            _lowered = self._handler.to_backend_code(rhs_canonical)
        except UnknownNotationError:
            return
        if _lowered.status == "refused":
            return
        rhs_for_maxima = self._handler.ensure_explicit_mult(
            self._handler.apply_time_dep_map(_lowered.code)
        )

        try:
            if _to_restore:
                # Kill conflicting assignments so RHS stays symbolic
                _kill_names = ", ".join(n for n, _ in _to_restore)
                self._adapter.execute(f"kill({_kill_names})$")
            self._adapter.execute(f"{symbol_name} : {rhs_for_maxima}$")
            # Restore previously killed symbols (also apply time-dep map)
            for _sym, _dep_rhs in _to_restore:
                _dep_rhs_maxima = self._handler.ensure_explicit_mult(
                    self._handler.apply_time_dep_map(_dep_rhs)
                )
                self._adapter.execute(f"{_sym} : {_dep_rhs_maxima}$")
        except Exception:
            # Log but don't raise — symbol definition in notebook is valid even if
            # CAS assignment fails (e.g., syntax error in RHS)
            pass

    def _refresh_undefined_symbols(self, symbol_name: str) -> None:
        """P1: Auto-update undefined_symbols when a symbol is newly defined.

        When symbol_name is registered, re-scan all expressions that listed it
        as undefined and update their undefined_symbols field.  This clears
        false positives and ensures forward references are tracked correctly.
        """
        for expr_id, expr in self.expressions.items():
            # Skip if symbol was not listed as undefined in this expression
            if symbol_name not in (expr.undefined_symbols or []):
                continue

            # Re-compute undefined symbols for this expression
            ir_ast_val = expr.ir_ast
            if ir_ast_val is None and expr.canonical and not expr.canonical.startswith("_nb_"):
                from notebook.runtime.shared_parser import SharedParser
                ir_ast_val = SharedParser().parse(expr.canonical)
            new_undefined = self._handler.find_undefined_symbols(ir_ast_val, self.symbol_context)

            # Update the expression if the list changed
            if new_undefined != expr.undefined_symbols:
                self.expressions[expr_id] = expr.model_copy(
                    update={"undefined_symbols": new_undefined}
                )

    def update_lhs_alias(self, expr_id: str, label: str) -> None:
        """Reassign a named label to point to a different expression (B-alias update).

        This implements the "blackboard overwrite" pattern: when the user applies
        an approximation or substitution to H and wants the symbol H to refer to
        the updated expression, call this method to redirect the alias.

        The original expression is preserved in self.expressions (history intact).
        Only the lhs_alias_map entry and SymbolContext.defining_expr_id are updated.

        Example::

            # H (E1) → operator_substitute → H_MF (E2)
            step, _ = engine.run_operation("operator_substitute", {...})
            new_id = step.output_refs[0]   # E2
            engine.update_lhs_alias(new_id, "H")
            # Now "H" in lhs_alias_map → E2; E1 still accessible by ID

        Args:
            expr_id: The new expression ID that the label should point to.
            label:   The label to reassign (e.g. "H", "H_BCS").
        """
        if expr_id not in self.expressions:
            raise KeyError(f"update_lhs_alias: expression {expr_id!r} not found")

        key = _normalize_lhs(label)
        if key:
            self.lhs_alias_map[key] = expr_id

        # Update SymbolContext.defining_expr_id for simple identifiers.
        sym = _extract_simple_symbol_name(label)
        if sym:
            existing = self.symbol_context.get(sym)
            if existing is not None:
                new_qty = existing.quantity.model_copy(
                    update={"defining_expr_id": expr_id}
                )
                self.symbol_context.registry[sym] = existing.model_copy(
                    update={"quantity": new_qty}
                )

    def _alias_keys(self, lhs: str) -> "list[str]":
        r"""The spellings an LHS may be looked up under — as written, and
        canonical.

        The map has consumers on BOTH sides of the LaTeX/canonical line and
        one key cannot serve them (all measured 2026-08-09):

        * `_auto_detect_substitutions` scans an expression's CANONICAL, and
          the P2 bind sends the name to the CAS — keyed on the display,
          `alpha_k = …` registered as `\alpha_{k}`, nothing matched it, and an
          inverse transform returned the degenerate `c_k = 0` as a success.
        * `_expand_symbol_refs` expands a formula written in LATEX — keyed
          only canonically, `2 * \bar{H} + V` stopped finding `H_bar`.
        * the same equation typed as plain text and picked from the math chip
          arrives spelled two ways (`sigma` / `\sigma`) and must land on one
          entry.

        So both are registered.  They point at the same expression, and every
        consumer already reduces a key to a symbol name before using it.
        """
        keys = [lhs]
        try:
            canonical = _normalize_lhs(self._handler.latex_to_canonical(lhs))
        except (UnknownNotationError, ValueError, TypeError):
            # Unreadable as LaTeX — the text as written is still a usable key,
            # and refusing to register one would lose the definition entirely.
            return keys
        if canonical and canonical not in keys:
            keys.append(canonical)
        return keys

    def _update_lhs_alias(self, expr_id: str, latex: str,
                          skip_cas_bind: bool = False) -> None:
        """Extract the LHS from *latex* and register it in lhs_alias_map.

        Called after any expression registration or operation result storage.
        No-op when *latex* contains no '=' (plain expression, not an equation).

        Also updates SymbolContext.defining_expr_id for simple identifiers and
        known decorated forms (\\bar{H} → H_bar, \\langle H \\rangle → H_exp, etc.).

        skip_cas_bind: if True, skip P2 (do not send the definition to the CAS).
        Used for solve results, which should remain explicit substitutions rather
        than becoming permanent session-wide variable bindings.
        """
        lhs = _lhs_latex_from_expr(latex)
        if not lhs or lhs == latex:
            # No '=' in latex → not an equation form
            return
        key = _normalize_lhs(lhs)
        if key:
            for _k in self._alias_keys(key):
                self.lhs_alias_map[_k] = expr_id

        # Additionally link simple / decorated symbol names in SymbolContext
        sym = _extract_simple_symbol_name(lhs)
        if sym:
            existing_prop = self.symbol_context.get(sym)
            is_new_symbol = existing_prop is None

            if existing_prop is not None:
                new_qty = existing_prop.quantity.model_copy(
                    update={"defining_expr_id": expr_id}
                )
                self.symbol_context.registry[sym] = existing_prop.model_copy(
                    update={"quantity": new_qty}
                )
            else:
                self.symbol_context.register(
                    sym,
                    quantity=Quantity(
                        name=sym,
                        kind="anonymous",
                        defining_expr_id=expr_id,
                    ),
                )

            # P1: Auto-update undefined_symbols for all expressions that
            # listed sym as undefined (only if newly defined)
            if is_new_symbol:
                self._refresh_undefined_symbols(sym)

            # P2: CAS auto-substitution — bind the symbol definition into CAS
            # state so that simplify(H) automatically uses H's value.  Only for
            # backends with evaluate-on-assignment semantics (Maxima); Julia
            # preserves structure and does not bind.
            # Send for both new symbols AND redefinitions (updated definitions).
            # skip_cas_bind=True suppresses this for solve results (which are
            # meant to be used via explicit substitution, not session-wide binds)
            if self._handler.binds_symbol_definitions and not skip_cas_bind:
                self._send_symbol_definition_to_maxima(sym, expr_id)

    def register_quantity(self, name: str, **kwargs) -> Quantity:
        """Add or update a Quantity in the registry by merging kwargs."""
        return self.symbol_context.register_quantity(name, **kwargs)

    def set_notation(self, name: str, fmt: str) -> Quantity:
        """Switch the display notation for a time-dependent quantity.

        fmt must be 'dot' or 'leibniz'.
        Raises ValueError when the quantity is unknown or fmt is invalid.
        Call rerender_notation_for_quantity(name) afterwards to update existing
        expression tiles.
        """
        if fmt not in ("dot", "leibniz"):
            raise ValueError(f"Unknown notation format: {fmt!r}. Use 'dot' or 'leibniz'.")
        if not self.symbol_context.get(name):
            raise ValueError(f"Quantity '{name}' is not registered. Declare it with declare_time_dependent first.")
        return self.register_quantity(name, notation=fmt)

    def rerender_notation_for_quantity(self, name: str) -> list[tuple[str, str]]:
        """Re-render latex for all derivative expressions of quantity `name`.

        Must be called AFTER set_notation() has updated notation in the
        registry.  Mutates self.expressions in place.

        Returns list of (expr_id, new_latex) pairs for every updated expression.
        """
        if not self._handler.supports_equation_source:
            return []  # Julia canonicals are variable names; no re-rendering needed.
        qty = self.symbol_context.get_quantity(name)
        if qty is None:
            return []
        updated: list[tuple[str, str]] = []
        for expr_id, expr in self.expressions.items():
            if expr.quantity_name != name or not expr.derivative_vars:
                continue
            # Re-compute LHS from stored metadata + current registry settings
            lhs = _format_provenance_lhs(
                name, expr.derivative_vars, expr.derivative_is_partial,
                qty=qty, parse_ir=self._handler.parse_ir,
                context=self.symbol_context,
            )
            # Re-compute RHS: rebuild Maxima Leibniz raw form from canonical, then
            # re-apply dot notation (which skips conversion when
            # notation="leibniz", leaving the Leibniz form intact).
            m = self._handler.uneval_diff_match(expr.canonical.strip())
            if m and m.group(1) == name:
                # Simple pure-derivative canonical: lhs alone is the canonical label.
                new_latex = lhs
            else:
                # Complex canonical (e.g. m*'diff(q(t),t,2)+k*q(t)): keep equation form.
                # `strip_lhs` returns the text unchanged when it is not an
                # equation, which is exactly the old `else expr.latex` branch —
                # only now a CAS-produced (unspaced) equation is recognised
                # instead of being re-emitted whole into `{lhs} = {rhs}`.
                new_latex = f"{lhs} = {strip_lhs(expr.latex)}"
            self.expressions[expr_id] = expr.model_copy(update={"latex": new_latex})
            updated.append((expr_id, new_latex))
        return updated

    def set_latex_name(self, name: str, latex: str) -> Quantity:
        """Set the LaTeX display symbol for a registered quantity.

        Affects future differentiations only; existing expressions retain their
        frozen latex.  Raises ValueError when the quantity is unknown or latex
        is empty.
        """
        if not self.symbol_context.get(name):
            raise ValueError(f"Quantity '{name}' is not registered.")
        latex = latex.strip()
        if not latex:
            raise ValueError("latex_name cannot be empty.")
        return self.register_quantity(name, latex_name=latex)

    def declare_time_dependent(self, name: str, time_parameter: str = "t") -> Quantity:
        """Mark a quantity as time-dependent so differentiation uses dot notation.

        Also runs depends([name],[...]) in Maxima so that diff(name, time_parameter)
        is treated symbolically (not as zero).  Maxima's depends() REPLACES a
        symbol's dependency list on each call rather than merging it — so if the
        symbol already has spatial dependencies (e.g. via define_scalar_field's
        depends(name,[x,y,z])), redeclaring with only [time_parameter] would
        silently drop them.  Re-send the union of prior natural_vars and
        time_parameter in a single call to keep all dependencies.
        """
        existing_vars = list(self.symbol_context.get_quantity(name).natural_vars) \
            if self.symbol_context.get_quantity(name) else []
        merged_vars = existing_vars + [v for v in [time_parameter] if v not in existing_vars]
        qty = self.register_quantity(
            name, time_dependent=True, time_parameter=time_parameter,
            natural_vars=merged_vars,
        )
        self.declare_depends(name, merged_vars)
        return qty

    # ------------------------------------------------------------------
    # Expression registration
    # ------------------------------------------------------------------

    def _fill_placeholder_sum_bounds(self, ir, expr_id: str) -> None:
        """Give every range-less sum in *ir* its placeholder range (B-10 ②).

        The rule — one pair per index NAME per expression, never shared between
        expressions, and the SAME pair when asked again — and the measurements
        behind it live in notebook/runtime/placeholder_sum_bounds.py.
        """
        from notebook.runtime.placeholder_sum_bounds import fill
        fill(ir, lambda name: self._sum_bounds.alloc(expr_id, name))

    def _sum_bounds_ir(self, expr_id: str, canonical: str):
        """Rebuild an expression's IR with the SAME placeholder ranges (B-10 ②-c).

        Cached back onto the expression: from here on it behaves like the live
        session it came from, and no later operation pays for the parse again.
        """
        ir = self._handler.parse_ir(
            canonical,
            before_alphatize=lambda node: self._fill_placeholder_sum_bounds(node, expr_id))
        if ir is not None and expr_id in self.expressions:
            self.expressions[expr_id].ir_ast = ir
        return ir

    def register_expression(self, canonical: str, latex: str = "",
                            lhs_label: str = "",
                            skip_cas_bind: bool = False,
                            _state_affecting: bool = True,
                            _advance_head: bool = True,
                            internal: bool = False,
                            reuse_display_id: str = "") -> Expression:
        """Register a new expression and return it with an assigned display_id.

        *internal* marks scaffolding the researcher never sees (INTERNAL-
        INTERMEDIATE-EID-GAP): it is keyed ``_T1``, ``_T2`` … instead of
        consuming an E number, so the visible numbering has no holes.  See
        :meth:`_alloc_internal_expr_id`.

        *reuse_display_id* registers into an ALREADY-ALLOCATED slot instead of
        taking a new number — used by the solve family, whose output id is
        allocated before the CAS runs (the command wrapping needs it) and whose
        first solution row belongs in exactly that slot.
        """
        canonical = self._handler.pre_process_canonical(canonical)
        display_id = (reuse_display_id or
                      (self._alloc_internal_expr_id() if internal
                       else self._alloc_expr_id()))
        step_id = self._alloc_step_id()

        from notebook.runtime.shared_parser import SharedParser, UnknownNotationError
        try:
            # B-10 ②: a sum written without a range gets its own placeholder
            # range here, once, so it keeps it for the life of the expression.
            # It runs INSIDE the parse, before the dummy indices are renamed:
            # the allocation is keyed on the name the researcher wrote, and
            # `\sum_i x_i + \sum_i y_i` is one binding written twice.
            ir_ast_val = self._handler.parse_ir(
                canonical,
                before_alphatize=lambda node: self._fill_placeholder_sum_bounds(
                    node, display_id))
        except UnknownNotationError as exc:
            raise self._translate_unknown_notation(exc) from exc
        log = getattr(ir_ast_val, "simplification_log", []) if ir_ast_val else []
        # For backends where parse_ir returns None (e.g. Julia), fall back to
        # SharedParser so undefined symbol detection still works on user input.
        _undef_ast = ir_ast_val
        if _undef_ast is None and canonical and not canonical.startswith("_nb_"):
            try:
                _undef_ast = SharedParser().parse(canonical)
            except UnknownNotationError as exc:
                raise self._translate_unknown_notation(exc) from exc
        undefined = self._handler.find_undefined_symbols(_undef_ast, self.symbol_context)
        # A bound index that reuses a name the expression uses elsewhere is
        # legitimate writing, so it is reported and nothing is refused — the
        # researcher is the only one who knows whether it was meant.  Computed
        # from the RESOLVED IR, where a Σ's dummies are actually bound; the
        # surface text cannot tell a binding from an ordinary occurrence.
        _warnings = [
            t(reason, self.lang, name=name)
            for name, reason in (
                find_bound_name_collisions(ir_ast_val, self.symbol_context)
                + find_bare_ordering(ir_ast_val))
        ]
        # docs/32 Phase 4: remember which bases the researcher wrote with a
        # subscript, so the display bridge can turn the CAS's `V(i)` back into
        # `V_i`.  It must be learned HERE, from the source expression's IR — a
        # result canonical `a*V(i)+b` parses as an ordinary function call and no
        # longer says that `V` was ever subscripted.
        record_symbol_usage(ir_ast_val, self.symbol_context)
        # Which SPELLING of φ/ε/θ the researcher used, for the same reason and
        # from the same place: the CAS answers in its own (`tex1` gives
        # `\varphi` for `phi`), so without this the paper carries one quantity
        # in two glyphs on adjacent lines (open_items PHI-GLYPH-SPLIT).  Read
        # from the SOURCE latex — a result's spelling is the backend's.
        record_greek_glyphs(latex, self.symbol_context)
        # If an LHS label is provided, prepend it: "LHS = RHS"
        # ONE door for "what does a registered expression look like".  Every
        # caller used to answer it for itself and they disagreed: the same
        # session published `\xi_{k}` for a RESULT and a bare `xi_k` for the
        # input beside it, and `\frac{\sqrt{…}}{2}` beside `sqrt(4*V^2*…)`
        # (measured 2026-08-09 over the whole example corpus).  The CANONICAL is
        # untouched — that is the researcher's own writing and what the CAS is
        # sent; only what is WRITTEN DOWN is normalised.
        _display_latex = self._handler.normalize_display_latex(latex or canonical)
        if lhs_label:
            # The label goes through the same door as the body — it is written
            # down too.  Left raw, `phi0 = \phi` published its left side as the
            # italic word "phi0" beside a right side that read φ
            # (measured 2026-08-09).
            _display_latex = (f"{self._handler.normalize_display_latex(lhs_label)}"
                              f" = {_display_latex}")
        # Render a matrix literal "[a, b; c, d]" as a pmatrix in the DISPLAY latex
        # only (canonical stays the CAS-ready matrix(...) form).  Idempotent.
        _display_latex = self._handler.matrix_literal_to_pmatrix(_display_latex)
        expr = Expression(
            display_id=display_id,
            branch_id=self.active_branch_id,
            latex=_display_latex,
            canonical=canonical,
            created_by_step=step_id,
            ir_ast=ir_ast_val,
            simplification_log=log,
            undefined_symbols=undefined,
            warnings=_warnings,
            # Preserve the user-facing canonical so JuliaCodeGenerator can
            # reconstruct the IR for commutator distribution later.
            input_canonical=canonical,
            internal=internal,
        )
        step = Step(
            step_id=step_id,
            kind=StepKind.expression_registration,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.register", self.lang, display_id=display_id),
            output_refs=[display_id],
            state_affecting=_state_affecting,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )

        self.expressions[display_id] = expr
        self.steps[step_id] = step
        if _advance_head:
            self._advance_head(step)

        qty = self._handler.parse_quantity(canonical, display_id)
        if qty is not None:
            # Use register_quantity (merge) so that previously declared
            # time_dependent / notation / latex_name flags are preserved.
            self.register_quantity(
                qty.name,
                kind=qty.kind,
                natural_vars=qty.natural_vars,
                defining_expr_id=qty.defining_expr_id,
            )

        # H-1: register LHS label in lhs_alias_map and (for simple identifiers)
        # in SymbolContext.  _update_lhs_alias handles both paths.
        #
        # The alias is keyed on what the RESEARCHER WROTE, never on the
        # normalised display.  The two are different questions and the display
        # door above answers only the second: `_auto_detect_substitutions`
        # scans an expression's CANONICAL for alias keys, and the P2 bind sends
        # the key to the CAS.  Keyed on the display, `alpha_k = u c_k + v c_mk`
        # registered as `\alpha_{k}`, nothing in a canonical ever matched it,
        # and an inverse transform came back `c_k = 0` — a degenerate solution
        # reported as success (measured 2026-08-09, tests/test_debug_repl.py).
        _alias_latex = latex or canonical
        if lhs_label:
            _alias_latex = f"{lhs_label} = {_alias_latex}"
        self._update_lhs_alias(display_id, _alias_latex, skip_cas_bind=skip_cas_bind)

        return expr


    def canonical_display_latex(self, canonical: str) -> str:
        """Render a canonical string to display LaTeX *without* evaluating it:
        ``tex1`` then the backend display postprocess (e.g. 'diff(q(t),t,2) → q̈).
        Used to show the explicit content of an E-ID-referenced formula at input
        time (display→evaluate, like the integral form)."""
        try:
            r = self._adapter.execute(f"tex1({canonical})")
            raw = r.value.strip() if (r.ok and r.value) else canonical
            return self._handler.postprocess_result_latex(
                canonical, raw, self.symbol_context) or raw
        except Exception:
            return canonical






















    # ------------------------------------------------------------------
    # CAS operation — helpers
    # ------------------------------------------------------------------

    def _guard_type_declaration(self, operation: str, args: dict,
                                all_names: "list[str] | None",
                                redeclare: bool) -> dict:
        """型宣言の前提を確かめ、いま宣言されている代数を返す。

        **宣言は既に書いた式の読み方を決めている**ので、型を変える宣言は既定で
        断る——判断そのものは `operand_guards.reject_conflicting_redeclaration`
        にあり、ここはその呼び出しと、取り消しに要る「前は何だったか」を集める
        ところ。`run_operation` から切り出してあるのは、ハイジーンの ratchet が
        「伸ばすな、切り出せ」と言うためである。
        """
        prior: dict[str, str] = {}
        for name in (all_names or [args.get("name", "")]):
            prop = self.symbol_context.get(name) if name else None
            math_type = getattr(getattr(prop, "quantity", None), "math_type", None)
            algebra = getattr(math_type, "algebra", None) if math_type else None
            if algebra:
                prior[name] = algebra
        reject_conflicting_redeclaration(
            operation, {**args, "_all_names": all_names, "_redeclare": redeclare},
            symbol_context=self.symbol_context, lang=self.lang)
        return prior

    def _withdraw_prior_declarations(self, prior_algebras: dict) -> None:
        """前の型宣言を CAS から取り消す（「宣言し直して」のときだけ）。

        取り消さずに宣言を撃つと CAS が断り、返るのは内部語だけになる
        （実測 2026-08-22: `declare(x, scalar)` のあとの `declare(x, nonscalar)` は
        `Maxima evaluation error`、`remove(x, scalar)` を挟むと通る）。
        何を撃つかはバックエンドの知識（`undeclare_command`）。
        """
        for name, algebra in prior_algebras.items():
            command = self._handler.undeclare_command(name, algebra)
            if command:
                self._adapter.execute(command)

    def _operation_label(self, operation: str) -> str:
        """操作を**研究者の言語で**呼ぶ（内部名を断り文に出さないため）。"""
        spec = self._validator.get_spec(operation) or {}
        display = spec.get("display") or {}
        return display.get(self.lang) or display.get("en") or operation

    def _reject_display_only_operands(self, operation: str, args: dict) -> None:
        """断りの判断は notebook/runtime/operand_guards.py（材料が増える場所なので）。"""
        reject_structure_only_operation(operation, lang=self.lang)
        reject_unusable_operands(operation, args, expressions=self.expressions,
                                 symbol_context=self.symbol_context, lang=self.lang,
                                 operation_label=self._operation_label(operation))
        # 宣言されただけの対象は値を持たない。**どの操作が宣言から組み立てるか**は
        # バックエンドの知識なので、そちらに訊く。案内する操作名も、研究者の言語で。
        _sep = "、" if self.lang == "ja" else ", "
        reject_declared_object_without_value(
            operation, args, symbol_context=self.symbol_context,
            reads_declaration=self._handler.reads_declared_object,
            operation_label=self._operation_label(operation),
            reading_ops_label=lambda kind, algebra: _sep.join(
                self._operation_label(op)
                for op in self._handler.declaration_computing_ops(kind, algebra)),
            lang=self.lang)

    def _raise_translated_cas_failure(self, operation: str, args: dict,
                                      error: str) -> None:
        """CAS の失敗を**研究者の言葉**にして送出する——ここが唯一の出口。

        読み替えは2段ある。**エラー本文から分かるもの**はバックエンドが名指し
        （`classify_cas_error`）、**本文には無いもの**は帳簿と CAS に訊く
        （値を持つ名前が変数の欄に立っていたのか）。Maxima のこの失敗の本文は
        文字どおり `Maxima evaluation error` で手掛かりが無いので、後者が要る。

        **断ったあとにだけ訊く**——境界は欄ではなく CAS にあり、同じ形で正しく
        答える操作がある（`solve` は「その名前が指すもの」について解く）。
        """
        translate_value_bearing_variable_failure(
            operation, args, symbol_context=self.symbol_context,
            cas_holds_value=lambda name: self._handler.cas_holds_value(
                self._adapter, name),
            value_of_name=self._defining_value_latex,
            operation_label=self._operation_label(operation), lang=self.lang)
        translate_indexed_variable_failure(
            operation, args, lowered_of=self._lowered_name,
            operation_label=self._operation_label(operation), lang=self.lang)
        _err_key = self._handler.classify_cas_error(error)
        if _err_key:
            raise ValueError(t(_err_key, self.lang,
                               **self._handler.cas_error_args(error)))
        raise ValueError(error)

    def _lowered_name(self, name: str) -> str:
        """その名前が CAS へどう届くか（`x_1` → `x(1)`）。読めなければ空文字。"""
        from notebook.runtime.shared_parser import UnknownNotationError
        try:
            return (self._handler.to_backend_code(name).code or "").strip()
        except (UnknownNotationError, ValueError, AttributeError, TypeError):
            # 読めない名前は「添字付き」とは言えない——`vector_expression_code`
            # と同じ並びで、断りの根拠にできないものは黙って降りる。
            return ""

    def _defining_value_latex(self, eid: str) -> str:
        """その式が**研究者の綴りで**与えている値（右辺）。

        断り文に出すのは内部の正準形ではなく研究者が書いたものである
        （Core Principle #1）——`y = x^3` なら `x^3`。
        """
        expr = self.expressions.get(eid)
        if expr is None:
            return ""
        return strip_lhs(expr.latex or expr.canonical or "").strip()

    def _preprocess_args(self, operation: str, args: dict) -> tuple[dict, str | None, bool, bool, dict]:
        """Normalize args before validation.

        Returns (normalized_args, wrt_qdot_qty_name, is_total_derivative,
        is_partial_derivative, display_hints).  Internal pipeline fields (prefixed
        with ``_``) and display-only hints (``bracket_a``, ``bracket_b``) are
        extracted and stripped here so OperationValidator never sees them.
        """
        # Extract and strip internal fields before validation.  is_total/is_partial
        # are the explicit derivative-kind flags the NL layer sets after resolving
        # the ∂-vs-d choice (asking the user when it is ambiguous).
        is_total = bool(args.get("_is_total_derivative", False))
        is_partial = bool(args.get("_is_partial_derivative", False))
        # Display-only hints: the handler says WHAT it did, `_visible_action`
        # says how to word it.  `label_expr_ref` / `label_operation` /
        # `label_rhs_of` exist because a few interceptors operate on a SCAFFOLD
        # (`_T1`) rather than on the researcher's own expression, so the label
        # built from `args` names a key nobody has seen (open_items
        # SCAFFOLD-LABEL, measured in 40 published labels).  Only the handler
        # knows what the scaffold stands for; only this layer knows the wording.
        display_hints = {
            k: args[k] for k in ("bracket_a", "bracket_b", "label_expr_ref",
                                 "label_operation", "label_rhs_of")
            if k in args
        }
        # define_scalar_field/define_vector_field combined with "時間依存の…場と
        # して定義して": fold the time parameter into natural_vars while the
        # "_also_time_dependent" flag is still present (it is stripped below along
        # with all other "_"-prefixed internal fields).  Must happen before the
        # strip so build_command's depends() call covers space AND time in one
        # declaration — see declare_time_dependent's docstring for why a later
        # separate call would silently drop the spatial vars.
        if (operation in ("define_scalar_field", "define_vector_field")
                and args.get("_also_time_dependent") and not args.get("natural_vars")):
            _coord_sys = str(
                args.get("coordinate_system") or self.session.coordinate_system or "cartesian"
            )
            args = {**args, "natural_vars":
                    list(self._handler.reserved_coord_vars(_coord_sys)) + ["t"]}
        # The particle the researcher NAMED decides the spin projection set, and
        # `_`-prefixed args are display hints that the next line drops — so it is
        # read here, the same place `_also_time_dependent` is read above.
        if operation in ("define_boson_operator", "define_fermion_operator"):
            self._record_spin_projections(str(args.get("name", "")).strip(),
                                          str(args.get("_particle_key", "")))
        if operation == "define_spin_operator" and args.get("_spin_desc"):
            self._record_nuclide_projections(str(args.get("name", "")).strip(),
                                             str(args.get("_spin_desc", "")))
        args = {k: v for k, v in args.items()
                if not k.startswith("_") and k not in display_hints}
        if operation in ("define_boson_operator", "define_fermion_operator", "define_spin_operator") and "name" in args:
            # Strip site index: "a(1)" → "a", "b(k)" → "b"
            _raw_name = str(args["name"]).strip()
            _bare = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", _raw_name)
            if _bare:
                args = {**args, "name": _bare.group(1)}

        if operation == "combine_expressions" and "op" in args:
            _raw_op = str(args["op"]).strip().lower()
            _OP_NORM: dict[str, str] = {
                "add": "+", "plus": "+", "+": "+",
                "subtract": "-", "minus": "-", "-": "-",
                "multiply": "*", "times": "*", "×": "*", "*": "*",
                "divide": "/", "÷": "/", "/": "/",
            }
            _normed = _OP_NORM.get(_raw_op, _raw_op)
            if _normed not in ("+", "-", "*", "/"):
                raise ValueError(
                    f"combine_expressions: unknown operator '{args['op']}'. "
                    "Use +, -, *, / (or add, subtract, multiply, divide)."
                )
            args = {**args, "op": _normed}

        if operation == "differentiate" and args.get("wrt_qdot"):
            _wrt_name = str(args["wrt_qdot"])
            _qty = self.symbol_context.get_quantity(_wrt_name)
            if _qty is None or not _qty.time_dependent:
                raise ValueError(
                    f"'{_wrt_name}' is not a registered time-dependent quantity. "
                    "Declare it with declare_time_dependent first."
                )
            _t = _qty.time_parameter or "t"
            _src = self.expressions.get(str(args.get("expr") or ""))
            _qdot = self._handler.wrt_qdot_canonical(
                _wrt_name, _t, (_src.canonical if _src else "") or "")
            args = {k: v for k, v in args.items() if k != "wrt_qdot"}
            args = {**args, "var": _qdot, "order": "1"}
            return args, _wrt_name, is_total, is_partial, display_hints

        return args, None, is_total, is_partial, display_hints

    def _resolve_equation_source(
        self, operation: str, args: dict, expr_lookup: dict
    ) -> tuple[dict, str, bool, bool]:
        """Detect function-definition equation sources for analytical ops.

        When the source expression is f(x) = rhs (or y = rhs for diff/integrate/
        limit), substitutes the RHS into expr_lookup so the operation acts on the
        function body rather than the equation object.

        Returns (expr_lookup, source_lhs, source_was_equation, source_is_partial).
        """
        if not self._handler.supports_equation_source:
            return expr_lookup, "", False, False
        # _SINGLE_EXPR_OPS (simplify/expand/factor/…) must also strip the LHS
        # from expr_lookup so Maxima receives only the RHS canonical.  Without
        # this, "factor(EOM = rhs)" causes Maxima to evaluate the EOM variable
        # (from H-1 definition) and produce a doubled "lhs = rhs = rhs" result.
        # source_was_equation is returned as True here only to strip expr_lookup;
        # _format_result_latex re-attaches the LHS via the _SINGLE_EXPR_OPS branch.
        _SINGLE_EXPR_OPS_NAMES = _EQ_VALUE_PRESERVING_OPS
        _EQ_AWARE_OPS = _ANALYTICAL_OPS | {"substitute"} | _SINGLE_EXPR_OPS_NAMES | _COMPLEX_VALUE_OPS
        if operation not in _EQ_AWARE_OPS:
            return expr_lookup, "", False, False

        expr_arg = args.get("expr")
        if not expr_arg or expr_arg not in expr_lookup:
            return expr_lookup, "", False, False

        cf = expr_lookup[expr_arg]
        # An `=` inside brackets is not an equation.  `"=" in cf` read
        # `\sum_{i=1}^{3} i^2` as the equation `\sum_{i` = `1}^{3} i^2` and
        # replaced the expression with that bogus right-hand side, so a bounded
        # sum reached the CAS as `trigsimp(ratsimp(1}^{3} i^2))` — measured
        # 2026-08-08, and present long before the sum work (open_items B-10).
        # `find_top_level_relation` is the brace/paren-aware scan `parse()`
        # already uses for exactly this question; there is no second answer.
        from notebook.runtime.shared_parser import find_top_level_relation
        _rel = find_top_level_relation(cf)
        if _rel is None or _rel[1] != "=" or ":=" in cf:
            return expr_lookup, "", False, False

        _rel_pos = _rel[0]
        lhs = cf[:_rel_pos].strip()
        rhs = cf[_rel_pos + 1:].strip()
        _lhs_qty = self.symbol_context.get_quantity(self._handler.qty_name_from_origin(lhs))

        if _lhs_qty is not None and _lhs_qty.kind == "function":
            return (
                {**expr_lookup, expr_arg: rhs},
                lhs, True, len(_lhs_qty.natural_vars) > 1,
            )

        from notebook.models.ir import FunctionCall, Symbol
        lhs_ir = self._handler.parse_ir(lhs)

        if lhs_ir is not None and isinstance(lhs_ir, FunctionCall) and isinstance(lhs_ir.func, Symbol):
            func_args = [a.name for a in lhs_ir.args if isinstance(a, Symbol)]
            return (
                {**expr_lookup, expr_arg: rhs},
                lhs, True, len(func_args) > 1,
            )

        if (lhs_ir is not None and isinstance(lhs_ir, Symbol)
                and operation in (_EQ_OPERATOR_FORM_OPS | {"differentiate"})):
            # Use natural_vars if they are explicitly set; otherwise fall back to
            # counting variables in the RHS canonical (handles L = f(q, q̇, t)
            # without explicit natural_vars declaration).
            is_partial = (
                len(_lhs_qty.natural_vars) > 1
                if (_lhs_qty is not None and _lhs_qty.natural_vars)
                else len(self._handler.canonical_variables(rhs)) > 1
            )
            return ({**expr_lookup, expr_arg: rhs}, lhs, True, is_partial)

        # For simplify/expand/factor/… on a named equation (e.g. EOM = expr),
        # strip the LHS so Maxima receives only the RHS.  Without this, Maxima
        # evaluates the LHS symbol (from H-1 definition) and produces a doubled
        # "lhs_value = rhs_value" result.  _format_result_latex re-attaches the
        # LHS label via the _SINGLE_EXPR_OPS branch.
        if lhs_ir is not None and isinstance(lhs_ir, Symbol) and operation in _SINGLE_EXPR_OPS_NAMES:
            return ({**expr_lookup, expr_arg: rhs}, lhs, True, False)

        # Complex-value ops on a named equation (Z = expr): strip to the RHS so
        # realpart/imagpart act on the value, not the equation.  source_lhs="" /
        # source_was_equation=False so the result is shown plainly (Re Z = R,
        # Im Z = ωL−1/(ωC)) without the (incorrect) "Z =" re-attachment, since
        # these ops do not preserve equality.
        # An INTEGRAL TRANSFORM of a named equation is the same case: the
        # researcher means the transform of the VALUE.  Measured: `inverse_laplace`
        # on the s-domain solution `X = …` returned `X*delta(t) = …` as a SUCCESS
        # (Maxima read the transform symbol X as an ordinary constant).  Five more
        # _ANALYTICAL_OPS are in none of the three tests above — open_items
        # EQ-SOURCE-BARE-LHS; scenario_forced_oscillator_laplace_jp.md ③.
        # A TRUNCATED SERIES is the third kind: it is neither the same quantity
        # (so no operator form) nor equal to it (so no `Y =`).  `taylor_expand`
        # of `Y = x^2+1` to order 2 is an approximation of Y, and writing
        # `Y = …` would state an equality the researcher did not ask for.  Until
        # 2026-08-16 it sent the equation whole and registered `x^2+1 = x^2+1`.
        _TRANSFORM_VALUE_OPS = frozenset({"laplace_transform", "inverse_laplace",
                                          "taylor_expand", "series_expand"})
        if (lhs_ir is not None and isinstance(lhs_ir, Symbol)
                and operation in (_COMPLEX_VALUE_OPS | _TRANSFORM_VALUE_OPS)):
            # Unlike Re/Im, an inverse transform HAS a name for its result: the
            # inverse of X(s) is x(t), and the engine holds that pairing already
            # (define_laplace_transform's table).  Leaving it off published the
            # time-domain solution as a bare expression with nothing saying what
            # it is — the researcher reads `x(t) = …`.
            _tl = self._laplace_time_lhs(lhs, args) if operation == "inverse_laplace" else ""
            if _tl:
                return ({**expr_lookup, expr_arg: rhs}, _tl, True, False)
            return ({**expr_lookup, expr_arg: rhs}, "", False, False)

        return expr_lookup, "", False, False

    def differentiate_is_ambiguous(self, expr_id: str, var: str) -> bool:
        """True when ∂/∂var and d/dvar may differ, so the system must ask, not guess.

        The expression contains — besides the differentiation variable itself — at
        least one symbol that is NOT a declared constant: either a free symbol
        (whose dependence on *var* is unknown) or one declared to depend on *var*.
        Any such symbol can make the total derivative (chain rule) differ from the
        partial, so the user must choose.  Symbols declared as scalar constants are
        excluded — they do not depend on *var* — so declaring constants is how the
        user removes the ambiguity.  This is NOT physics- or time-specific: it
        holds for any variable.  (The time-dependent declaration is just the
        CAS-level way to state a dependence; it gets no special treatment here.)"""
        expr = self.expressions.get(expr_id)
        if expr is None:
            return False
        canon = expr.canonical
        # `"=" in canon` also fired on `<=` / `>=`, and an inequality DOES reach
        # here (measured: `y \le x^2 + a` differentiated by x), so the operator
        # test is kept as written — only the SCAN becomes bracket-aware, which is
        # what `\sum_{i=1}^{N} a_i x` needed: split naively, the variables of
        # `1}^N*a_i*x` decided partial-vs-total (measured).
        # `:=` needs no guard of its own; find_top_level_relation excludes it.
        _rel = split_relation(canon)
        rhs = _rel[2] if (_rel is not None and "=" in _rel[1]) else canon
        for _name in self._handler.canonical_variables(rhs):
            # r_dot / theta_ddot → r / theta, and y' → y.  The PRIME is the same
            # convention as the dot in this physics (q and q̇ are independent
            # atoms, CLAUDE.md), so it gets the same treatment — and doing so
            # KEEPS today's answer: before `canonical_variables` learned to see
            # the prime, `y'` was reported as `y` and already compared equal to a
            # differentiation variable `y` (measured 2026-09-24).
            _base = re.sub(r"(?:_d+ot|'+)$", "", _name)
            if _name == var or _base == var:
                continue                            # the differentiation variable
            _q = (self.symbol_context.get_quantity(_name)
                  or self.symbol_context.get_quantity(_base))
            if _q is not None and _q.kind == "scalar":
                continue                            # declared constant → no ambiguity
            return True                             # free / dependent symbol remains
        return False

    def _resolve_derivative_provenance(
        self,
        operation: str,
        args: dict,
        source_was_equation: bool,
        source_lhs: str,
        source_is_partial: bool,
        wrt_qdot_name: str | None,
        force_total: bool = False,
        force_partial: bool = False,
    ) -> tuple[str | None, list[str], bool]:
        """Track derivative origin across iterated/mixed partials.

        Returns (d_qty_name, d_vars, d_is_partial).
        All three are empty/False for non-differentiate operations or wrt_qdot.
        """
        if operation != "differentiate" or wrt_qdot_name is not None:
            return None, [], False

        _dvar = str(args.get("var", "x"))
        _order = int(args.get("order", 1))
        # Explicit derivative-kind flags resolved upstream (NL layer asks the user
        # when the choice is ambiguous): "全微分して" forces d/dt (total) notation,
        # "偏微分して" forces ∂/∂var (partial) — both override the generic
        # >1-variable heuristic that otherwise guesses partial.  Passed in (the
        # underscore-prefixed args were already stripped before this call).
        _force_total = force_total
        _force_partial = force_partial

        if source_was_equation:
            d_qty_name = self._handler.qty_name_from_origin(source_lhs)
            d_vars = [_dvar] * _order
            _early_qty = self.symbol_context.get_quantity(d_qty_name)
            d_is_partial = (
                (
                    len(_early_qty.natural_vars) > 1
                    if (_early_qty and _early_qty.natural_vars)
                    else source_is_partial
                )
                or len(set(d_vars)) > 1
            )
            if _force_total:
                d_is_partial = False
            elif _force_partial:
                d_is_partial = True
            return d_qty_name, d_vars, d_is_partial

        _ea = args.get("expr")
        if not _ea or _ea not in self.expressions:
            return None, [], False

        _se = self.expressions[_ea]
        _se_qty_name = _se.quantity_name or (
            self._handler.qty_name_from_origin(_se.derivative_origin)
            if _se.derivative_origin else None
        )

        if self._handler.supports_equation_source and not _se_qty_name:
            from notebook.models.ir import FunctionCall, Symbol
            ir = self._handler.parse_ir(_se.canonical.strip())
            if ir is not None and isinstance(ir, FunctionCall) and isinstance(ir.func, Symbol):
                if ir.func.name not in MATH_FUNC_NAMES:
                    _fn_candidate = ir.func.name
                    if self.symbol_context.get(_fn_candidate):
                        _se_qty_name = _fn_candidate

        if not _se_qty_name:
            return None, [], False

        d_qty_name = _se_qty_name
        d_vars = _se.derivative_vars + [_dvar] * _order
        _iter_qty = self.symbol_context.get_quantity(d_qty_name)
        d_is_partial = (
            (len(_iter_qty.natural_vars) > 1 if (_iter_qty and _iter_qty.natural_vars) else False)
            or len(set(d_vars)) > 1
        )
        if _force_total:
            d_is_partial = False
        elif _force_partial:
            d_is_partial = True
        return d_qty_name, d_vars, d_is_partial

    def _complete_operator_sites(self, args: dict, expr_lookup: dict) -> dict:
        """A-11 case B: if the session/operands use exactly one site, complete any
        site-unspecified spin operator in a literal operand to that site.  With 0 or
        ≥2 sites it leaves args unchanged (the case-A ambiguity error then fires)."""
        h = self._handler
        if not getattr(h, "completes_operator_sites", False):
            return args
        sites: "set[str]" = set()
        for ex in self.expressions.values():
            sites |= h.spin_sites_in(ex.canonical)
        for v in args.values():
            if isinstance(v, str):
                sites |= h.spin_sites_in(expr_lookup.get(v, v))
        if len(sites) != 1:
            return args
        site = next(iter(sites))
        new_args, changed = dict(args), False
        for k, v in args.items():
            if isinstance(v, str) and v not in self.expressions:
                completed = h.complete_bare_spin_sites(v, site)
                if completed != v:
                    new_args[k], changed = completed, True
        return new_args if changed else args

    def _build_cas_command(
        self, operation: str, args: dict, expr_lookup: dict, backend: str
    ) -> "tuple[str, str, str | None]":
        """Compile operation + args to a CAS command string.

        Returns (command, sum_index_hint, input_canonical_override) where
        sum_index_hint is a comma-separated list of dummy index names and
        input_canonical_override (if not None) should be stored as the new
        expression's input_canonical instead of the default propagation.
        """
        from notebook.runtime.shared_parser import UnknownNotationError
        try:
            build = self._handler.build_command(operation, args, expr_lookup)
        except UnknownNotationError as exc:
            raise self._translate_unknown_notation(exc) from exc
        if not build.ok:
            raise OperationRefused(build.error or "",
                                   reason_code=getattr(build, "reason_code", None))
        return (build.command,
                getattr(build, "sum_index_hint", ""),
                getattr(build, "input_canonical", None))

    def _format_result_latex(
        self,
        operation: str,
        args: dict,
        out_latex: str,
        source_was_equation: bool,
        source_lhs: str,
        wrt_qdot_name: str | None,
        d_qty_name: str | None,
        d_vars: list[str],
        d_is_partial: bool,
        is_total: bool,
        is_partial_flag: bool = False,
    ) -> str:
        """Apply operation-specific LaTeX equation formatting to `out_latex`.

        R3 (docs/27): each per-operation formatter below keeps its own guard and
        returns *out_latex* unchanged for other operations, so this dispatcher is
        a simple chain — exactly one formatter transforms the string."""
        out_latex = self._format_differentiate_latex(
            operation, args, out_latex, source_was_equation, source_lhs,
            wrt_qdot_name, d_qty_name, d_vars, d_is_partial, is_total, is_partial_flag)
        out_latex = self._format_substitute_latex(
            operation, args, out_latex, source_was_equation, source_lhs)
        out_latex = self._format_operator_substitute_latex(operation, args, out_latex)
        out_latex = self._format_relabel_index_latex(operation, args, out_latex)
        out_latex = self._format_integral_latex(
            operation, args, out_latex, source_was_equation, source_lhs)
        out_latex = self._format_eigenvalues_latex(operation, args, out_latex)
        out_latex = self._format_pairs_latex(operation, args, out_latex)
        out_latex = self._format_heisenberg_latex(operation, args, out_latex)
        out_latex = self._format_inverse_transform_latex(
            operation, args, out_latex, source_was_equation, source_lhs)
        out_latex = self._format_single_expr_latex(operation, args, out_latex)
        out_latex = self._format_combine_latex(operation, args, out_latex)
        return out_latex

    # ------------------------------------------------------------------
    # _format_result_latex phases (R3, docs/27): per-operation LaTeX formatters
    # extracted verbatim.  Each keeps its guard and passes *out_latex* through
    # untouched for other operations, so the dispatcher above just chains them.
    # ------------------------------------------------------------------
    def _format_differentiate_latex(self, operation, args, out_latex,
                                    source_was_equation, source_lhs, wrt_qdot_name,
                                    d_qty_name, d_vars, d_is_partial, is_total,
                                    is_partial_flag) -> str:
        if operation == "differentiate":
            if wrt_qdot_name is not None:
                _wqty = self.symbol_context.get_quantity(wrt_qdot_name)
                _src_lbl = source_lhs if source_was_equation else args.get("expr", "expr")
                return f"{_format_wrt_qdot_lhs(_src_lbl, _wqty)} = {out_latex}"
            if d_qty_name:
                _d_qty = self.symbol_context.get_quantity(d_qty_name)
                deriv_lhs = _format_provenance_lhs(d_qty_name, d_vars, d_is_partial,
                                                   qty=_d_qty, parse_ir=self._handler.parse_ir,
                                                   context=self.symbol_context)
                # A result that is ALREADY a statement gets no second left-hand
                # side.  `_format_single_expr_latex` learned this in 2026-08-13
                # (`x + x = 2 x=y`) and asks `find_top_level_relation`; this
                # sibling did not, so differentiating an equation published
                # `\frac{\partial}{\partial t}(G) = \frac{d}{dt}G = -(…)` — the
                # same statement twice, with two left-hand sides
                # (measured 2026-08-21, the Fourier step of S-ak).
                if (deriv_lhs and deriv_lhs.strip() != out_latex.strip()
                        and find_top_level_relation(out_latex) is None):
                    return f"{deriv_lhs} = {out_latex}"
                return out_latex
            expr_arg = args.get("expr")
            if expr_arg and expr_arg in self.expressions:
                src_expr = self.expressions[expr_arg]
                src_vars = self._handler.canonical_variables(
                    src_expr.display_repr
                )
                # Explicit kind wins; otherwise fall back to the >1-variable guess.
                if is_total:
                    is_partial = False
                elif is_partial_flag:
                    is_partial = True
                else:
                    is_partial = len(src_vars) > 1
                # Prefer the display LaTeX of the source expression (already
                # post-processed: \dot{q}, \ddot{q} etc.) over display_repr
                # (raw Maxima canonical like l^2*m*'diff(q(t),t,1)).
                # If the stored latex is an equation "LHS = RHS", use the LHS
                # (mathematical identity, e.g. ∂L/∂q̇) rather than the RHS
                # (computed value, e.g. ml²q̇) — this gives the conventional
                # d/dt(∂L/∂q̇) notation rather than d/dt(ml²q̇).
                # When using LaTeX as the origin string, skip parse_ir
                # (which expects Maxima canonical, not LaTeX).
                # `_lhs_latex_from_expr` returns the whole string when there is
                # no top-level `=`, which is the old "leave it alone" branch —
                # and it is the same scan the rest of this file now uses, so a
                # bounded sum's own `=` can no longer be read as the equation's.
                _latex_val = _lhs_latex_from_expr(src_expr.latex or "")
                if _latex_val:
                    _src_display = _latex_val
                    _parse_ir_fn = None  # LaTeX origin: skip IR parsing
                else:
                    _src_display = src_expr.display_repr
                    _parse_ir_fn = self._handler.parse_ir
                _var = str(args.get("var", "x"))
                _ord = int(args.get("order", 1))
                # Collapse repeated same-variable differentiation into a
                # higher-order derivative (d/dk(d/dk(X)) → d²/dk²(X)) rather than
                # nesting.  Only applies when the source LHS is already a
                # derivative w.r.t. the same variable.
                _bumped = _bump_same_var_derivative(
                    _src_display, _var, _ord, parse_ir=_parse_ir_fn,
                    context=self.symbol_context
                )
                if _bumped is not None:
                    deriv_lhs = _bumped
                else:
                    deriv_lhs = _format_derivative_lhs(
                        _src_display, _var, _ord,
                        is_partial=is_partial, parse_ir=_parse_ir_fn,
                        context=self.symbol_context,
                    )
                # Same rule as the branch above and as
                # `_format_single_expr_latex`: a result that is already a
                # statement gets no second left-hand side.
                if find_top_level_relation(out_latex) is not None:
                    return out_latex
                return f"{deriv_lhs} = {out_latex}"
        return out_latex

    def _display_arg(self, text: str) -> str:
        r"""An operation argument as the researcher should SEE it.

        Only worth a CAS round trip when the text is not already plain: a NAME
        or a number needs no `tex1`, which would cost a call per label.

        But a name is not always its own display — `rho` is written `\rho` and
        `theta_ddot` is written `\ddot{\theta}`.  Returning it raw published the
        internal spelling: the substitution label read `rho → lambda/mu` and
        `theta_ddot → …` where the researcher had written `\rho` and
        `\ddot{\theta}` (measured 2026-08-23; Core Principle #1, and the same
        shape as the `\[ H_hat_0 \]` defect a researcher reported the day
        before).  `canonical_name_to_latex` is the door that already answers
        this — it leaves `q`, `F`, `T` alone and knows the decorations — so this
        asks it rather than deciding again."""
        if not text:
            return text
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
            return canonical_name_to_latex(text, self.symbol_context)
        if re.fullmatch(r"\d+", text):
            return text
        return self.canonical_display_latex(text) or text

    def _format_substitute_latex(self, operation, args, out_latex,
                                 source_was_equation, source_lhs) -> str:
        if operation == "substitute":
            expr_arg = args.get("expr")
            # The label is shown to the RESEARCHER, so the two sides go through
            # the display door rather than being pasted in as CAS text.  Measured
            # 2026-08-20: substituting a defined expectation value labelled the
            # step `[E2]_{expectation(H) \to E_0}` — the system's own functional
            # name, an English word the researcher never wrote, in a formula
            # where they had written `\langle H \rangle` (Core Principle #1).
            lhs_arg = self._display_arg(str(args.get("lhs", "")))
            rhs_arg = self._display_arg(str(args.get("rhs", "")))
            if expr_arg and expr_arg in self.expressions:
                src_lbl = self.expressions[expr_arg].display_id
                # Determine the effective source LHS: prefer source_was_equation
                # (set when source is a function-definition equation), then fall
                # back to checking the source expression's latex directly.
                # This handles H-1 named equations (EOM = …, L = …) and H-8
                # both-sides results whose latex has "LHS = RHS" form but whose
                # canonical is RHS-only.
                _eff_source_lhs = source_lhs if source_was_equation else ""
                if not _eff_source_lhs:
                    _src_latex = self.expressions[expr_arg].latex
                    if "=" in _src_latex and "==" not in _src_latex:
                        _eff_source_lhs = _lhs_latex_from_expr(_src_latex)
                # `source_lhs` は expr_lookup 由来＝**正準形**である。ラベルは
                # 研究者が見るものなので、右辺と同じく表示の扉を通す——通って
                # いなかったため、公開 TeX に `G_{\sigma}(vec(r)t,vec(r)'t')` が
                # 出ていた（利用者報告、2026-08-22、E18）。LaTeX 側の枝を通しても
                # 無害（既にコマンドになっている名前は二重変換されない）。
                if _eff_source_lhs:
                    _eff_source_lhs = beautify_bare_canonical_latex(_eff_source_lhs)
                if _eff_source_lhs:
                    # A result that is ALREADY a statement gets no second left
                    # side — the same guard, and the same bracket-aware
                    # spacing-blind test, as _format_single_expr_latex above.
                    # It became reachable here on 2026-08-14, when the substitute
                    # builder stopped cutting the source equation down to its
                    # right-hand side: the CAS now answers with the whole
                    # equation, and prepending produced `H = H=2 J y`.
                    if find_top_level_relation(out_latex) is not None:
                        return out_latex
                    import re as _re
                    _repl_val = rhs_arg
                    subst_lhs = _re.sub(
                        r'(?<![A-Za-z0-9_])' + _re.escape(lhs_arg) + r'(?![A-Za-z0-9_}])',
                        lambda _, r=_repl_val: r, _eff_source_lhs
                    )
                    return f"{subst_lhs} = {out_latex}"
                return f"[{src_lbl}]_{{{lhs_arg} \\to {rhs_arg}}} = {out_latex}"
        return out_latex

    def _format_operator_substitute_latex(self, operation, args, out_latex) -> str:
        if operation in ("operator_substitute", "operator_substitute_pair", "collect_by_operators"):
            expr_arg = args.get("expr")
            if expr_arg and expr_arg in self.expressions:
                src_lhs = _lhs_latex_from_expr(self.expressions[expr_arg].latex)
                if src_lhs:
                    return f"{src_lhs} = {out_latex}"
        return out_latex

    def _format_relabel_index_latex(self, operation, args, out_latex) -> str:
        if operation == "relabel_index":
            expr_arg = args.get("expr")
            old_idx  = str(args.get("old_index", ""))
            new_idx  = str(args.get("new_index", ""))
            if expr_arg and expr_arg in self.expressions:
                src_lhs = _lhs_latex_from_expr(self.expressions[expr_arg].latex)
                if src_lhs and old_idx and new_idx:
                    return f"{src_lhs}\\big|_{{{old_idx}\\to {new_idx}}} = {out_latex}"
        return out_latex

    def _format_integral_latex(self, operation, args, out_latex,
                               source_was_equation, source_lhs) -> str:
        if operation in _EQ_OPERATOR_FORM_OPS:
            expr_arg = args.get("expr")
            if expr_arg and expr_arg in self.expressions:
                src_canonical = self.expressions[expr_arg].canonical
                src_is_equation = "=" in src_canonical and ":=" not in src_canonical
                if source_was_equation:
                    eq_form = _equation_form_latex(
                        operation, args, source_lhs, out_latex,
                        self._handler.format_constant_for_latex,
                    )
                    return eq_form or out_latex
                if not src_is_equation:
                    # Use the source's *value* latex as the integrand: when the
                    # source is itself an integral/limit result (latex carries an
                    # equation form like "\int…=1/4" while its canonical is the bare
                    # value), nesting its full latex would duplicate the equation.
                    integrand_latex = _rhs_of_latex(self.expressions[expr_arg].latex)
                    eq_form = _equation_form_latex(
                        operation, args, integrand_latex, out_latex,
                        self._handler.format_constant_for_latex,
                    )
                    return eq_form or out_latex
        return out_latex

    @staticmethod
    def _split_top_level(text: str, opener: str = "{[(", closer: str = "})]") -> "list[str]":
        """Split at commas that are not inside any bracket."""
        parts, depth, start = [], 0, 0
        for k, ch in enumerate(text):
            if ch in opener:
                depth += 1
            elif ch in closer:
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(text[start:k]); start = k + 1
        parts.append(text[start:])
        return [p.strip() for p in parts]

    def _format_pairs_latex(self, operation, args, out_latex) -> str:
        r"""``[[1,1],[2,2]]`` from `pure_nash_equilibria` → ``\{(1,1), (2,2)\}``.

        The CAS returns the equilibria as a list of index PAIRS — which row and
        which column — and printed literally that is a bracket of brackets a
        reader will take for a payoff matrix (raised 2026-08-09 on the
        battle-of-the-sexes scenario).  It is not one: `\begin{pmatrix}1&1\\2&2
        \end{pmatrix}` would state a 2×2 array of numbers where the answer is
        "two equilibria, at (1,1) and (2,2)".  A set of pairs says exactly that
        and needs no CAS convention to read.
        """
        if operation != "pure_nash_equilibria":
            return out_latex
        rows = self._unwrap_bracket_lists(out_latex)
        if rows is None:
            return out_latex
        return (r"\left\{ "
                + " , ".join(r"\left( " + " , ".join(r) + r" \right)" for r in rows)
                + r" \right\}")

    def _unwrap_bracket_lists(self, latex: str) -> "list[list[str]] | None":
        """``\\left[ \\left[a,b\\right] , \\left[c,d\\right] \\right]`` → [[a,b],[c,d]].

        ``None`` when the text is not exactly that shape — a display is never
        invented for something this did not understand.
        """
        def _strip(text: str) -> "str | None":
            text = text.strip()
            for open_tok, close_tok in ((r"\left[", r"\right]"), ("[", "]")):
                if text.startswith(open_tok) and text.endswith(close_tok):
                    return text[len(open_tok):-len(close_tok)].strip()
            return None

        body = _strip(latex)
        if body is None:
            return None
        rows = []
        for part in self._split_top_level(body):
            inner = _strip(part)
            if inner is None:
                return None
            rows.append(self._split_top_level(inner))
        return rows or None

    def _format_eigenvalues_latex(self, operation, args, out_latex) -> str:
        r"""Maxima answers `eigenvalues` with ``[[values],[multiplicities]]``.

        That pairing is the CAS's RETURN CONVENTION, not notation: printed
        literally it is a bracket of brackets, and a researcher reading
        `\left[\left[λ_1, λ_2\right], \left[1, 1\right]\right]` has to know
        Maxima to know which half is which (raised on the two-level
        perturbation scenario, 2026-08-09).

        Written as the SPECTRUM instead — each eigenvalue repeated as often as
        its multiplicity says — it needs no convention: `\{2, 2\}` says
        "2 twice" to anyone.  Anything that does not parse as that exact shape
        is returned untouched; a display is never invented.
        """
        if operation != "eigenvalues":
            return out_latex
        body = out_latex.strip()
        for open_tok, close_tok in ((r"\left[", r"\right]"), ("[", "]")):
            if body.startswith(open_tok) and body.endswith(close_tok):
                body = body[len(open_tok):-len(close_tok)].strip()
                break
        else:
            return out_latex
        halves = self._split_top_level(body)
        if len(halves) != 2:
            return out_latex
        lists = []
        for half in halves:
            half = half.strip()
            for open_tok, close_tok in ((r"\left[", r"\right]"), ("[", "]")):
                if half.startswith(open_tok) and half.endswith(close_tok):
                    lists.append(self._split_top_level(
                        half[len(open_tok):-len(close_tok)].strip()))
                    break
            else:
                return out_latex
        values, mults = lists
        if len(values) != len(mults) or not all(m.isdigit() for m in mults):
            return out_latex
        spectrum = [v for v, m in zip(values, mults) for _ in range(int(m))]
        return r"\left\{ " + " , ".join(spectrum) + r" \right\}"

    def _format_heisenberg_latex(self, operation, args, out_latex) -> str:
        # heisenberg_equation: display as d(Op)/dt = RHS
        if operation == "heisenberg_equation":
            op_ref = args.get("operator")
            # The operator reaches here EITHER as an E-id or as the researcher's
            # own text — `Heisenberg equation of E1 for a(1)` has always passed
            # it raw, and since 2026-08-23 so does the Japanese spelling (the
            # operand of a non-`expr` operation is no longer registered as an
            # expression, so the two languages number their expressions alike).
            # Asking only about the E-id meant the English session NEVER got the
            # `d(Op)/dt =` left side — the same result, published without saying
            # what it is the derivative OF.  Measured 2026-08-23 from the
            # regenerated examples, which is also the only thing that caught it:
            # the instruction corpus compared operation and operands, and a
            # DISPLAY divergence is invisible to both.
            if op_ref:
                op_latex = _lhs_latex_from_expr(
                    self.expressions[op_ref].latex if op_ref in self.expressions
                    else op_ref)
                # The space after `d` is load-bearing, not cosmetic: without it
                # this system's own display of a Heisenberg equation does not
                # read back.  `\frac{dc(k,u)}{dt}` (emitted before 2026-08-11,
                # and sitting in sessions/S-c_redrive_20260731.json) lowers to
                # the PRODUCT `dc(k, u) * dt^-1`, because an unseparated `d` may
                # only be split off a single-letter base — the no-auto-split rule
                # that keeps `\frac{delta}{dx}` from becoming a derivative.
                # Spelled `\frac{d c(k,u)}{d t}` it lowers to
                # `'diff(c(k, u),t,1)`.  Typographically it is also the more
                # correct setting: the differential d is an operator, not part
                # of the operator's name.
                lhs = rf"\frac{{d {op_latex}}}{{d t}}"
                return f"{lhs} = {out_latex}"
        return out_latex

    def _format_single_expr_latex(self, operation, args, out_latex) -> str:
        # Generic: single-expr operations (simplify, expand, normal_order, etc.)
        # If the source expression had a LHS label, carry it forward to the result.
        # The THIRD copy of this list, found while adding `numerical_eval` to
        # it (2026-08-16): the strip decision and the re-attachment were spelled
        # separately, so an operation could be told to drop the label and never
        # told to put it back — which is exactly what `numerical_eval` did.
        if operation in _EQ_VALUE_PRESERVING_OPS:
            expr_arg = args.get("expr")
            if expr_arg and expr_arg in self.expressions:
                _src_latex = self.expressions[expr_arg].latex
                # Only carry forward the LHS label when the source expression
                # was in equation form "LHS = RHS".  When there is no "=" the
                # full string would be returned by _lhs_latex_from_expr, which
                # would prepend the entire source canonical as a false LHS.
                if "=" in _src_latex and "==" not in _src_latex:
                    src_lhs = _lhs_latex_from_expr(_src_latex)
                    if src_lhs:
                        # Guard: a result that is ALREADY a statement gets no
                        # second left-hand side.  This used to ask whether the
                        # result STARTS WITH the source's LHS text, which is a
                        # different question and answers wrongly the moment the
                        # CAS simplifies that side: `x + x = y` expanded gives
                        # Maxima's `2 x=y`, the prefix test fails, and the
                        # published latex was `x + x = 2 x=y` — a double
                        # equation, with the canonical `2*x = y` beside it
                        # perfectly correct (measured 2026-08-13).  Nothing to
                        # do with sums; any compound left side reaches it.
                        # `find_top_level_relation` is the bracket-aware scan
                        # this file already uses 500 lines above for the mirror
                        # question ("is the SOURCE an equation") — and it is
                        # spacing-blind, which the old test was not: Maxima's
                        # `tex1` writes `2 x=y` while our own display writes
                        # ` = `, so a `" = "` test cannot see a CAS equation.
                        if find_top_level_relation(out_latex) is not None:
                            return out_latex
                        return f"{src_lhs} = {out_latex}"
        return out_latex

    def _format_combine_latex(self, operation, args, out_latex) -> str:
        if operation == "combine_expressions":
            _cea = args.get("expr_a")
            _ceb = args.get("expr_b")
            _op = str(args.get("op", "+"))
            _cop_sym = _COMBINE_OP_SYMBOLS.get(_op, _op)
            if _cea in self.expressions and _ceb in self.expressions:
                _lhs_a = _lhs_latex_from_expr(self.expressions[_cea].latex)
                _lhs_b = _lhs_latex_from_expr(self.expressions[_ceb].latex)
                if _op in ("*", "/") and _has_toplevel_additive(_lhs_a):
                    _lhs_a = f"({_lhs_a})"
                if _op in ("*", "/", "-") and _has_toplevel_additive(_lhs_b):
                    _lhs_b = f"({_lhs_b})"
                return rf"{_lhs_a} {_cop_sym} {_lhs_b} = {_rhs_of_latex(out_latex)}"
        return out_latex

    # ------------------------------------------------------------------
    # Re-evaluation (C-Ext Phase 1-A)
    # ------------------------------------------------------------------

    def _auto_detect_substitutions(self, expr: "Expression") -> "dict[str, str]":
        """Detect substitutions automatically from lhs_alias_map.

        Scans expr.canonical for symbol names that appear in lhs_alias_map,
        resolves their current canonical values, and returns the mapping.

        Uses word-boundary matching to avoid false positives (e.g. "H" matching
        inside "H_0").  Falls back to substring check if regex unavailable.
        """
        import re as _re
        result: dict[str, str] = {}
        canonical = expr.canonical or ""
        # Sort by length descending so longer names match before shorter prefixes
        for lhs_key, ref_id in sorted(
            self.lhs_alias_map.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            # Use word-boundary match: lhs_key surrounded by non-word chars or string edges
            pattern = r"(?<![A-Za-z0-9_])" + _re.escape(lhs_key) + r"(?![A-Za-z0-9_])"
            if not _re.search(pattern, canonical):
                continue
            ref_expr = self.expressions.get(ref_id)
            if ref_expr and ref_expr.canonical:
                result[lhs_key] = ref_expr.canonical
        return result

    def re_evaluate_expr(
        self,
        expr_id: str,
        substitutions: "dict[str, str] | None" = None,
        latex: str = "",
        lhs_label: str = "",
    ) -> "Expression | None":
        """Re-evaluate an expression with updated symbol values.

        This is the primary API for self-consistent iterations in mean-field
        theory and parametric sweeps.  The original expression is not modified;
        a *new* expression is registered and returned.

        If *substitutions* is None or empty, symbols are auto-detected from
        ``lhs_alias_map``: every symbol in the canonical that has a registered
        definition is substituted with its current value.  This makes the
        operation backend-agnostic — the user does not need to know whether
        they are in a Maxima or Julia session.

        Example (Hartree-Fock mean-field update)::

            # Initial mean-field result registered as E5
            # After one iteration: n_k ← 0.42
            new_expr = engine.re_evaluate_expr(
                "E5",
                substitutions={"n_k": "0.42"},
            )

            # Auto-detect from lhs_alias_map:
            new_expr = engine.re_evaluate_expr("E5")

        Args:
            expr_id:       Display-ID of the expression to re-evaluate (e.g. "E5").
            substitutions: {symbol_name: new_canonical_value} pairs, or None to
                           auto-detect from lhs_alias_map.
                           Values may be numeric literals, canonical expressions,
                           or E-refs (e.g. "E3").
            latex:         Optional LaTeX string for the new expression.
            lhs_label:     Optional LHS label (e.g. "H_MF'").

        Returns:
            New Expression, or None if the backend does not support re-evaluation.

        Raises:
            KeyError: if *expr_id* is not found in this session.
        """
        if expr_id not in self.expressions:
            raise KeyError(f"re_evaluate_expr: expression {expr_id!r} not found")

        src_expr = self.expressions[expr_id]
        canonical = src_expr.canonical or ""

        # Auto-detect substitutions when not provided
        if not substitutions:
            substitutions = self._auto_detect_substitutions(src_expr)

        # Resolve any E-refs in substitution values.
        expr_lookup: dict[str, str] = {
            eid: (expr.canonical or "") for eid, expr in self.expressions.items()
        }
        resolved_subs: dict[str, str] = {}
        for sym, val in substitutions.items():
            if val in expr_lookup:
                resolved_subs[sym] = expr_lookup[val]
            else:
                resolved_subs[sym] = val

        new_canonical = self._handler.re_evaluate_expr(
            canonical, resolved_subs, expr_lookup, adapter=self._adapter,
            sympy_canonical=src_expr.sympy_canonical,
        )
        if new_canonical is None:
            return None

        new_expr = self.register_expression(
            new_canonical,
            latex=latex,
            lhs_label=lhs_label,
        )
        # Propagate sympy_canonical: if the source was a SymPy expression and the
        # result is still symbolic (not a bare number), mark the new expression too.
        if new_expr is not None and src_expr.sympy_canonical and new_canonical:
            # new_canonical is the value_str from _nb_sympy_substitute; use it as
            # sympy_canonical so further chaining (collect_by_operators etc.) works.
            new_expr = new_expr.model_copy(update={"sympy_canonical": new_canonical})
            self.expressions[new_expr.display_id] = new_expr
        return new_expr



    # ------------------------------------------------------------------
    # CAS operation — public entry point
    # ------------------------------------------------------------------

    def evaluate_formula(self, canonical: str) -> str:
        """CAS-evaluate a raw canonical expression without creating history.

        Returns the simplified canonical, or the original *canonical* on failure.
        """
        simplified, _ = self.evaluate_formula_with_latex(canonical)
        return simplified

    def evaluate_formula_with_latex(self, canonical: str) -> tuple[str, str]:
        """CAS-evaluate *canonical* and return (simplified_canonical, display_latex).

        Runs the expression through the backend's simplify command, then
        renders it to LaTeX via postprocess_result_latex so that dot notation
        (``\\ddot{q}``), q(t)→q stripping, etc. are all applied.
        Returns (canonical, "") on failure so the caller can fall back gracefully.
        Intended for the H-1 input pipeline (``EOM = E3 - E4`` style).
        """
        try:
            # Prefer simplifying the raw canonical directly: the expanded RHS is
            # already a valid CAS string, and re-parsing it via build_command can
            # drop parens around a subtracted multi-term group (A−(−b−c)→A−b−c).
            _raw_cmd = self._handler.simplify_raw_command(canonical)
            if _raw_cmd is not None:
                result = self._adapter.execute(_raw_cmd)
            else:
                _tmp_id = "E0"
                build = self._handler.build_command(
                    "simplify", {"expr": _tmp_id}, {_tmp_id: canonical}
                )
                if not build.ok:
                    return canonical, ""
                result = self._adapter.execute(build.command)
            if not (result.ok and result.value and not result.value.startswith("ERROR")):
                return canonical, ""
            simplified = result.value.strip()
            # execute() already renders LaTeX as a side effect of the same
            # command (Maxima: tex1() wrapper; Julia: driver-side tex output)
            # — reuse it instead of re-issuing a Maxima-only tex1() call,
            # which errors under the Julia backend.
            raw_latex = result.latex.strip() if result.latex else ""
            if not raw_latex:
                try:
                    tex_result = self._adapter.execute(f"tex1({simplified})")
                    if tex_result.ok and tex_result.value:
                        raw_latex = tex_result.value.strip()
                except Exception:
                    pass
            if not raw_latex:
                raw_latex = simplified
            display_latex = self._handler.postprocess_result_latex(
                simplified, raw_latex, self.symbol_context
            )
            return simplified, display_latex
        except Exception:
            # Best-effort simplification: fall back to the original canonical.
            # Logged so a genuine simplification failure leaves a trace.
            _logger.debug("simplified-form computation failed; returning original",
                          exc_info=True)
        return canonical, ""

    # ------------------------------------------------------------------

    def run_operation(self, operation: str, args: dict,
                      internal: bool = False) -> tuple[Step, str]:
        """Validate, build, and execute a common operation.

        Returns (step, result_value) on success.
        Raises ValueError with a user-readable message on failure.

        *internal* marks the RESULT as scaffolding: it is keyed ``_T1``, ``_T2``
        … instead of consuming an E number (INTERNAL-INTERMEDIATE-EID-GAP).
        Callers that build a temporary, run an operation on it and then hide it
        must say so HERE rather than hiding afterwards, because hiding after the
        fact cannot give the number back — the next expression has already taken
        the one after it.  The result stays reachable (``hidden`` is not set), so
        a following step can still use it as an operand.

        Special key in *args*:
            ``_update_alias``: if present and non-empty, the result expression's
            lhs_alias_map entry is updated to point to the new expression.
            This implements the "blackboard overwrite" pattern — the named symbol
            (e.g. "H") now refers to the post-operation result while the original
            expression is preserved in history.

            Example::

                engine.run_operation("operator_substitute", {
                    "expr": "E1",
                    "op_name": "Sz", "rhs": "mz",
                    "formal_index": "2", "exact_index": True,
                    "_update_alias": "H",   # ← H now points to the result
                })
        """
        # Extract and strip _update_alias before validation.
        _update_alias: str = str(args.pop("_update_alias", "") or "")

        # Save underscore-internal fields that the handler needs AFTER arg stripping.
        _TYPE_DECL_OPS = frozenset({
            "define_scalar_parameter", "define_vector", "define_matrix", "define_tensor",
        })
        _saved_all_names: list[str] | None = (
            args.get("_all_names") if operation in _TYPE_DECL_OPS else None
        )
        # 「宣言し直して」と明示されたか（`_preprocess_args` が `_` 付きを剥がす）。
        _saved_redeclare: bool = bool(args.get("_redeclare"))

        # 1. Normalize args; strip internal fields; detect wrt_qdot shorthand
        args, wrt_qdot_name, is_total, is_partial_flag, display_hints = self._preprocess_args(operation, args)

        # 2. Validate — restrict to expressions reachable in the active branch
        known_ids = self._active_expr_ids()
        val = self._validator.validate(operation, args, self.session.primary_backend, known_ids)
        if not val.valid:
            raise ValueError(val.error)

        # 2b. Refuse display-only operands BEFORE the parser sees them
        self._reject_display_only_operands(operation, args)

        # 2c. A type declaration must not silently REPLACE a different one.
        _prior_algebras = self._guard_type_declaration(
            operation, args, _saved_all_names, _saved_redeclare)

        # 3. Resolve equation source (f(x)=rhs → operate on rhs, display as f(x))
        expr_lookup, source_lhs, source_was_equation, source_is_partial = (
            self._build_expr_lookup(operation, args, known_ids)
        )

        # 4. Ensure explicit order for differentiate before provenance tracking
        if operation == "differentiate":
            args = {**args, "order": str(int(args.get("order", 1)))}

        # 5. Track derivative provenance for iterated/mixed partials
        d_qty_name, d_vars, d_is_partial = self._resolve_derivative_provenance(
            operation, args, source_was_equation, source_lhs, source_is_partial, wrt_qdot_name,
            force_total=is_total, force_partial=is_partial_flag,
        )

        # 6. Pre-CAS undefined symbol guard
        self._pre_cas_undefined_guard(args, known_ids)

        # Operation-specific argument preparation (vector-field natural_vars,
        # site completion, Laplace time-function form).
        args, expr_lookup = self._prepare_op_specific_args(operation, args, expr_lookup)

        # 7. Build CAS command via IR compiler
        command, _sum_hint, _input_canon_override = self._build_cas_command(
            operation, _args_with_derivative_kind(args, is_total, is_partial_flag),
            expr_lookup, val.backend)

        # Propagate sum_index_hint from source expressions when the operation itself
        # does not produce one (combine_expressions, substitute, operator_substitute,
        # relabel_index all act on summed Hamiltonians but may not set the hint).
        if not _sum_hint:
            _src_eids = [v for v in args.values()
                         if isinstance(v, str) and v in self.expressions]
            for _src_eid in _src_eids:
                _expr = self.expressions[_src_eid]
                _inherited = _expr.sum_index_hint
                if not _inherited:
                    # Backfill for sessions saved before sum_index_hint was stored:
                    # detect \sum_{k} at the start of the expression's LaTeX.
                    _m = re.search(r'\\sum_\{?([A-Za-z,]+)\}?', _expr.latex)
                    if _m:
                        _inherited = _m.group(1).strip("{}")
                if _inherited:
                    _sum_hint = _inherited
                    break

        # 7. Allocate IDs and execute
        _SIDE_EFFECT_OPS = frozenset({
            "define_function", "to_latex", "to_plain_text",
            "define_boson_operator", "define_fermion_operator",
            "define_spin_operator",
            "define_scalar_parameter", "define_vector",
            "define_matrix", "define_tensor",
            "define_vector_field", "define_scalar_field",
            "define_coordinate_system",
            "declare_symmetry",
            # Display-only operations: return a formatted string but do NOT
            # register a new expression (no E-number assigned).  The result
            # appears in the timeline as a display annotation.
            "collect_spin_pm",
        })
        is_side_effect = operation in _SIDE_EFFECT_OPS
        output_id = None if is_side_effect else (
            self._alloc_internal_expr_id() if internal else self._alloc_expr_id())
        step_id = self._alloc_step_id()

        # Wrap the command for result storage (e.g. Julia assigns _nb_Ek = (cmd)).
        command = self._handler.wrap_command_for_storage(command, output_id, operation)

        if _saved_redeclare:
            self._withdraw_prior_declarations(_prior_algebras)

        cas_result = self._execute_cas_command(operation, command)
        status = StepStatus.success if cas_result.ok else StepStatus.failed
        output_refs = [output_id] if (output_id and cas_result.ok) else []
        # Something the researcher should see that is not a failure — asked of
        # the backend, because what counts as worth saying is its knowledge.
        _result_warns = (self._handler.result_warnings(operation, self._adapter)
                         if cas_result.ok else [])
        # ②の共通の仕組み: オペランドは、セッションが知っているものである。
        # 判断は notebook/runtime/operand_guards.py（材料が増える場所なので）。
        _word_operand_warns = word_operand_warnings(
            operation, args, validator=self._validator, lang=self.lang)
        input_refs = [v for v in args.values() if isinstance(v, str) and v in self.expressions]

        step = Step(
            step_id=step_id,
            kind=StepKind.cas_operation,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=self._visible_action(operation, args, is_total_derivative=is_total,
                                                display_hints=display_hints),
            operation=operation,
            operation_args=args,
            backend=self.session.primary_backend,
            rendered_command=command,
            show_command_to_user=False,
            input_refs=input_refs,
            output_refs=output_refs,
            state_affecting=True,
            status=status,
            timestamp=datetime.now(timezone.utc),
        )
        self.steps[step_id] = step

        if not cas_result.ok:
            if not is_side_effect:
                self._expr_counter -= 1
            self._raise_translated_cas_failure(operation, args, cas_result.error or "")

        self._advance_head(step)

        step = self._update_symbol_context_after_exec(
            operation, args, step, step_id, _saved_all_names, _TYPE_DECL_OPS
        )

        # Display the imaginary unit with the domain's symbol (EE: %i → j), done at
        # the VALUE level so a subscript/superscript/exponent index i is untouched.
        cas_result = self._handler.localize_imaginary_unit(
            cas_result, self._adapter, self.symbol_context)

        if is_side_effect:
            # Apply the same LaTeX postprocessing used for regular results.
            # Skip postprocessing when the value is a backend void sentinel
            # ("nothing" for Julia, "true"/"done" for Maxima) — those are
            # implementation details and must not be passed through latex pipelines.
            _side_raw = cas_result.latex or cas_result.value
            _is_void = not self._handler.clean_void_result(_side_raw)
            _side_val = self._handler.postprocess_result_latex(
                cas_result.value, _side_raw, self.symbol_context
            ) if _side_raw and not _is_void else cas_result.value
            return step, _side_val

        # 8. Format output LaTeX
        if operation in ("to_latex", "to_plain_text"):
            out_latex = cas_result.value
        else:
            raw_latex = cas_result.latex or cas_result.value
            out_latex = self._handler.postprocess_result_latex(
                cas_result.value, raw_latex, self.symbol_context, _sum_hint
            )
            # Wrap with Σ notation before equation formatting so that
            # _format_result_latex can produce "H = Σ_k[...]" correctly.
            #
            # Only when the backend did NOT write one itself.  A backend whose
            # sum is a real object renders its own Σ, and pasting a second one
            # on top gave `\sum_{k}[\sum_{j} …]` (measured 2026-08-07).  The
            # test is on the RESULT, not on the backend's name, so this stays
            # inside the encapsulation rule.
            #
            # The names are the backend's NEUTRAL dummy names, not the ones the
            # researcher wrote: the body carries those same neutral names, and a
            # Σ that disagrees with its own body is worse than one that does not
            # use the researcher's letters (open_items SUM-DUMMY-NAMES).
            _hint_names = [x.strip() for x in _sum_hint.split(",") if x.strip()]
            # …and only when the RESULT still carries one of those indices.  A
            # sum the CAS contracted away has no index left, and wrapping it
            # produced a Σ over an index the summand does not contain
            # (measured 2026-08-22 — see `result_carries_index`).
            if (_hint_names and self._handler.needs_sum_prefix
                    and not self._handler.result_latex_has_sum(out_latex)
                    and self._handler.result_carries_index(out_latex, _hint_names)):
                indices = ",".join(f"_{{{x}}}" for x in _hint_names)
                out_latex = f"\\sum{indices}\\left[{out_latex}\\right]"
            out_latex = self._format_result_latex(
                operation, args, out_latex,
                source_was_equation, source_lhs,
                wrt_qdot_name, d_qty_name, d_vars, d_is_partial, is_total,
                is_partial_flag,
            )

        # 9. Store result expression
        canonical_val = self._handler.canonical_for_result(cas_result.value, output_id, operation)
        # FREE-PARAM: tag the degrees of freedom a general solution carries, and show
        # them.  Indefinite integration carries the constant of integration C; the
        # result is the GENERAL antiderivative F(x)+C (math workflow — a later
        # specialization step fixes C from a condition).  ODE / Laplace cases plug
        # in here as they are implemented.
        _free_params: list[str] = []
        if operation == "integrate_indefinite" and cas_result.ok:
            _C = self._fresh_free_param("C", avoid_in=canonical_val)
            canonical_val = f"{canonical_val}+{_C}"
            out_latex = f"{out_latex} + {_C}"
            _free_params = [_C]
        elif operation == "solve_ode_general" and cas_result.ok:
            # The general ODE solution carries one (1st order) or two (2nd order)
            # arbitrary constants; the names were chosen collision-free by the
            # caller (solve_ode_general) and renamed in-CAS.  Tag whichever appear.
            _cn = args.get("const_names") or {"k1": "k_1", "k2": "k_2", "c": "C"}
            _free_params = [nm for nm in (_cn.get("k1"), _cn.get("k2"), _cn.get("c"))
                            if nm and re.search(rf"(?<!\w){re.escape(nm)}(?!\w)",
                                                canonical_val)]
        cas_repr_val = self._handler.cas_repr(cas_result.value, canonical_val)
        # parse_ir may raise on raw CAS list output (e.g. a solve / inverse-transform
        # result like "[[c_k = …, c_mk = …]]"): that raw expression is hidden and the
        # meaningful per-solution registration happens in _register_solve_results, so a
        # parse failure here must degrade to OpaqueExpr rather than abort the operation.
        try:
            ir_ast_val = self._handler.parse_ir(canonical_val)
        except Exception:
            _logger.debug("parse_ir failed for %r; falling back to OpaqueExpr",
                          canonical_val, exc_info=True)
            ir_ast_val = None
        if ir_ast_val is None and cas_repr_val:
            from notebook.models.ir import OpaqueExpr
            ir_ast_val = OpaqueExpr(text=cas_repr_val)
            
        log = getattr(ir_ast_val, "simplification_log", []) if ir_ast_val else []
        # input_canonical: preserve the user-facing canonical BEFORE Julia renames it
        # to "_nb_Ek".  Priority: build_command override (e.g. relabel_index stores
        # the relabeled string) > args["expr"] for register ops > source expr propagation.
        if _input_canon_override is not None:
            _input_canon: str | None = _input_canon_override
        else:
            _expr_val = args.get("expr")
            if isinstance(_expr_val, str) and not is_expr_ref(_expr_val):
                # args["expr"] is an actual canonical string (register/compute op)
                _input_canon = _expr_val
            else:
                # For transform ops, propagate source expression's input_canonical.
                _src_eid = next(
                    (v for v in args.values()
                     if isinstance(v, str) and v in self.expressions),
                    None,
                )
                _input_canon = self.expressions[_src_eid].input_canonical if _src_eid else None
        # Mark SymPy-derived results so downstream operations can chain via SymPy.
        # _nb_sympy_* Julia functions return (value_str, latex_str); the value_str
        # is a SymPy-evaluatable string stored in cas_result.value.
        # We store it in sympy_canonical so future operations can re-feed it to SymPy.
        _sympy_canonical: str | None = None
        if cas_result.ok and command and "_nb_sympy_" in command:
            _sympy_canonical = cas_result.value or None

        # Through the SAME door a registered expression's display goes through
        # (`register_expression`).  A result is written down too, and it was
        # written down by a different rule: a canonical `phi_bar` — H-1's
        # spelling of `\bar{\phi}` — came back as φ subscripted with the WORD
        # "bar" (measured 2026-08-09).
        out_latex = self._handler.normalize_display_latex(out_latex)
        self.expressions[output_id] = Expression(
            display_id=output_id,
            branch_id=self.active_branch_id,
            latex=out_latex,
            canonical=canonical_val,
            created_by_step=step_id,
            quantity_name=d_qty_name,
            derivative_vars=d_vars,
            derivative_is_partial=d_is_partial,
            ir_ast=ir_ast_val,
            cas_repr=cas_repr_val,
            simplification_log=log,
            input_canonical=_input_canon,
            sum_index_hint=_sum_hint,
            sympy_canonical=_sympy_canonical,
            free_parameters=_free_params,
            internal=internal,
            warnings=([t(_k, self.lang) for _k in _result_warns]
                      + _word_operand_warns),
        )
        # docs/32 §2.7: a free parameter is a NAME the CAS already holds, so it
        # must be declared here or it lowers as an indexed symbol (k_1 → k(1))
        # and every later substitution on it silently misses.
        self._declare_free_parameters(_free_params)
        # Register LHS alias for any equation-form result (e.g. "H + V = ...").
        # Skip solve operations — _register_solve_results registers proper canonical
        # keys from each var=val pair; registering the raw Maxima LaTeX here would
        # store a LaTeX-form key (e.g. \ddot{q}) that causes wrong substitutions.
        #
        # skip_cas_bind=True (BP-7): this call registers an OPERATION's computed
        # RESULT, not user-typed H-1 input (user input goes through
        # register_expression, which calls _update_lhs_alias separately without
        # this flag). An operation's result can coincidentally simplify to a bare
        # "symbol = expr" form (e.g. small-integer-payoff mixed-strategy
        # indifference equations) that is indistinguishable from H-1 syntax; P2
        # CAS-binding it would permanently rebind that symbol to a
        # self-referential value and silently corrupt every later solve() call
        # for it. lhs_alias_map / SymbolContext linking still happens (needed for
        # formula-ref lookups on the result) — only the CAS-side "sym: rhs" send
        # is suppressed.
        if operation not in ("solve_equation", "solve_system", "build_inverse_transform",
                             "solve_ode_general"):
            self._update_lhs_alias(output_id, out_latex, skip_cas_bind=True)

        # B-alias update: redirect the named symbol to the new expression.
        if _update_alias and output_id:
            self.update_lhs_alias(output_id, _update_alias)

        # H-9: for solve operations, parse the result and register each
        # solution (var = val) as a separate named expression.
        if operation in ("solve_equation", "solve_system", "build_inverse_transform") and cas_result.ok:
            self._finalize_solve_results(cas_result, step_id, output_id)

        # Return the freshest step object: solve / inverse-transform replace
        # self.steps[step_id] with a model_copy carrying the full output_refs
        # (all solution rows), so the stale local `step` would otherwise hide every
        # solution past the first from callers that read step.output_refs.
        return self.steps.get(step_id, step), cas_result.value

    # ------------------------------------------------------------------
    # run_operation phases (R3, docs/27): extracted from the 432-line body so
    # each phase reads as one unit.  Pure moves — same locals threaded in/out.
    # ------------------------------------------------------------------
    def _with_placeholder_sum_bounds(self, expr_lookup: dict, known_ids) -> dict:
        """CAS view for expressions whose IR carries a placeholder sum range."""
        from notebook.runtime.placeholder_sum_bounds import cas_view
        return cas_view(expr_lookup, self.expressions,
                        self._handler.generate_backend_code,
                        rebuild=self._sum_bounds_ir)

    def _with_vector_cas_view(self, expr_lookup: dict, known_ids) -> dict:
        """外積を含む式の CAS 表示へ差し替える（研究者の綴りは `__user__` に残す）。"""
        from notebook.runtime.vector_op_types import value_lookup
        _vo = value_lookup(self.symbol_context, expr_lookup.get)
        out = None
        for eid in list(expr_lookup):
            text = expr_lookup.get(eid)
            if not isinstance(text, str) or eid.startswith(("__input__", "__sympy__", "__user__")):
                continue
            view = self._handler.vector_cas_view(text, _vo)
            if view and view != text:
                out = dict(expr_lookup) if out is None else out
                out[f"__user__{eid}"] = text
                out[eid] = view
        return out if out is not None else expr_lookup

    def placeholder_cas_view(self, expr_id: str) -> "str | None":
        """One expression's CAS view — see `cas_view_of` (EXPR-REF-PROVENANCE)."""
        from notebook.runtime.placeholder_sum_bounds import cas_view_of
        return cas_view_of(expr_id, self.expressions.get(expr_id),
                           self._handler.generate_backend_code,
                           rebuild=self._sum_bounds_ir)

    def _build_expr_lookup(self, operation: str, args: dict,
                           known_ids) -> "tuple[dict, str | None, bool, bool]":
        """Phase 3: build the canonical lookup (+ __input__/__sympy__ variants and
        H-1 equation restoration), then resolve the equation source.  Returns
        (expr_lookup, source_lhs, source_was_equation, source_is_partial)."""
        expr_lookup = {k: v.canonical for k, v in self.expressions.items() if k in known_ids}
        # B-10 ②: a sum written WITHOUT a range has no CAS form in its canonical
        # (that is the researcher's own writing).  Its placeholder range was
        # allocated once, at registration, and lives in the stored IR, so the
        # CAS view of such an expression is generated FROM THAT IR.  Deriving it
        # here — per call, never persisted — is the CAS-code rule, and it is
        # what keeps ONE expression's range stable while two expressions that
        # merely both lack a range keep different ones.
        expr_lookup = self._with_placeholder_sum_bounds(expr_lookup, known_ids)
        # 外積は研究者の綴り（`~`）のままでは CAS が評価できず、一般の操作に入ると
        # **生の CAS エラー**が返っていた（Core Principle #1）。綴り直せるかは
        # バックエンドの知識なので訊く——`__user__<eid>` に研究者の綴りを残すのは、
        # 成分の読みが**型を利用者書式から**読むからである（`~` は CrossProduct
        # 節点、落とした後は未知の関数呼び出し）。約束は `lower_expr_lookup` と同じ。
        expr_lookup = self._with_vector_cas_view(expr_lookup, known_ids)
        # Parallel lookup: input_canonical (user-facing formula before Julia opaque renaming).
        # Used by JuliaCodeGenerator to reconstruct IR for commutator distribution.
        # Keys are prefixed with "__input__" to coexist in the same dict.
        for _eid, _expr in self.expressions.items():
            if _eid in known_ids and _expr.input_canonical:
                expr_lookup[f"__input__{_eid}"] = _expr.input_canonical
            if _eid in known_ids and getattr(_expr, "sympy_canonical", None):
                expr_lookup[f"__sympy__{_eid}"] = _expr.sympy_canonical
        # H-1 equation restoration: expressions registered via H-1 store only the
        # RHS in canonical, but _resolve_equation_source needs the full "LHS = RHS"
        # form to recognise them as equations.  Reconstruct using lhs_alias_map.
        # Guard: only simple identifier / function-call LHS keys (e.g. "F", "h(a,b)").
        # Keys like "\frac{\partial c}{\partial a}" (derivative results) must be
        # excluded — parse_ir cannot handle them as equation LHS.
        import re as _re_h1
        _SIMPLE_LHS_RE = _re_h1.compile(
            r'^(?:'
            r'[A-Za-z_Α-ω][A-Za-z0-9_Α-ω]*(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?'  # plain id
            r'|\\[A-Za-z]+(?:\^[0-9]+)?(?:_[A-Za-z0-9]+)?'  # \cmd, \cmd^n, \cmd_n
            r')$'
        )
        _lhs_by_eid: dict[str, str] = {}
        for _lhs_key, _lhs_eid in self.lhs_alias_map.items():
            if (_lhs_eid in known_ids
                    and _lhs_eid not in _lhs_by_eid
                    and _SIMPLE_LHS_RE.match(_lhs_key)):
                _lhs_by_eid[_lhs_eid] = _lhs_key
        for _eid, _lhs_key in _lhs_by_eid.items():
            _cur = expr_lookup.get(_eid, "")
            if _cur and "=" not in _cur:
                # Convert LaTeX LHS to backend canonical so that solve() and
                # subst() receive valid CAS syntax (e.g. \omega^2 → omega^2).
                _maxima_lhs = self._handler.latex_to_canonical(_lhs_key) or _lhs_key
                expr_lookup[_eid] = f"{_maxima_lhs} = {_cur}"

        # D-7 (open_items.md): a compound LHS containing literal-division
        # derivative notation (e.g. "dy/dx+y") is not a single simple atom,
        # so the loop above skips it — restore_derivative_lhs (BackendHandler,
        # Maxima-only today) handles this narrower case separately, aborting
        # safely (None) rather than guessing at anything it doesn't recognise.
        for _lhs_key, _lhs_eid in self.lhs_alias_map.items():
            if _lhs_eid not in known_ids or _lhs_eid in _lhs_by_eid:
                continue
            _cur = expr_lookup.get(_lhs_eid, "")
            if not _cur or "=" in _cur:
                continue
            _restored = self._handler.restore_derivative_lhs(_lhs_key)
            if _restored is not None:
                expr_lookup[_lhs_eid] = f"{_restored} = {_cur}"
        return self._resolve_equation_source(operation, args, expr_lookup)

    def _pre_cas_undefined_guard(self, args: dict, known_ids) -> None:
        """Phase 5: raise UndefinedSymbolsError if any operand expression carries a
        symbol the backend requires to be declared before CAS execution."""
        _input_eids = [v for v in args.values()
                       if isinstance(v, str) and v in self.expressions and v in known_ids]
        _pre_undef: list[str] = []
        _seen_pre: set[str] = set()
        for _eid in _input_eids:
            _expr_obj = self.expressions[_eid]
            _ast = _expr_obj.ir_ast
            # Backends where parse_ir returns None (e.g. Julia) cannot store an ir_ast
            # for user-input expressions.  **Which AST this scan should read is the
            # backend's knowledge**, so ask the handler rather than parsing here:
            # unresolved, `\lim_{x,to,0} (…)` reads as the symbols `lim`, `to`.
            if _ast is None and _expr_obj.canonical and not _expr_obj.canonical.startswith("_nb_"):
                _ast = self._handler.parse_for_undefined_scan(_expr_obj.canonical)
            for _sym in self._handler.blocks_on_undefined_symbols(_ast, self.symbol_context):
                if _sym not in _seen_pre:
                    _seen_pre.add(_sym)
                    _pre_undef.append(_sym)
        if _pre_undef:
            raise UndefinedSymbolsError(
                t("engine.undefined_symbols", self.lang, symbols=", ".join(_pre_undef)),
                symbols=_pre_undef,
            )

    def _prepare_op_specific_args(self, operation: str, args: dict,
                                  expr_lookup: dict) -> "tuple[dict, dict]":
        """Phase 6: op-specific argument prep (vector-field natural_vars, spin-site
        completion, Laplace time-function form).  Returns (args, expr_lookup)."""
        # For define_vector_field with custom coordinate system, inject variable names
        # into args so build_command can use the correct component suffixes.
        if (operation == "define_vector_field"
                and str(args.get("coordinate_system") or self.session.coordinate_system or "") == "custom"
                and self.session.custom_coord_vars
                and not args.get("natural_vars")):
            args = dict(args)
            args["natural_vars"] = ",".join(self.session.custom_coord_vars)

        # A-11 case B: complete site-unspecified spin operators (Sx → Sx_1) when
        # the session/operands reference a single unambiguous site.
        args = self._complete_operator_sites(args, expr_lookup)

        # Laplace transform: rewrite time-dependent symbols to explicit function
        # form (q → q(t)).  Maxima's laplace ignores depends() on a bare atom and
        # would mistreat q as a constant (q/s); it needs q(t) to give s²Q(s)−…
        if operation == "laplace_transform":
            _t = str(args.get("t", "t"))
            _td_names = [n for n, p in self.symbol_context.registry.items()
                         if getattr(p.quantity, "time_dependent", False)
                         and str(getattr(p.quantity, "time_parameter", "t")) == _t]
            _expr_arg = args.get("expr")
            if _td_names and _expr_arg in expr_lookup:
                expr_lookup = {**expr_lookup,
                               _expr_arg: self._handler.to_time_function_form(
                                   expr_lookup[_expr_arg], _td_names, _t)}
        return args, expr_lookup

    def _execute_cas_command(self, operation: str, command: str):
        """Phase 8: run the command on the CAS adapter.  A backend whose command
        may span several statements executes them itself (see
        ``BackendHandler.execute_multi_statement``); a Julia UndefVarError for a
        cleared _nb_Ek triggers one restore-and-retry."""
        # Whether a command can be multi-statement, and how to combine the
        # per-statement results, is the backend's business — the engine must not
        # know the operation name or build a CAS-specific result type
        # (open_items AUDIT-SHARED-CAS-LEAK).  None = not applicable.
        cas_result = self._handler.execute_multi_statement(
            self._adapter, operation, command)
        if cas_result is None:
            cas_result = self._adapter.execute(command)
        # A backend that stores results in CAS-side variables can lose them (the
        # Python process restarted while the in-memory expressions survived).
        # Recognising THAT error is the backend's job — the wording and the
        # variable naming are its own (open_items AUDIT-SHARED-CAS-LEAK); the
        # replay-based restoration below is generic session machinery.
        if not cas_result.ok and cas_result.error:
            _eid = self._handler.missing_variable_expr_id(cas_result.error)
            if _eid and self._restore_stored_expr_var(_eid):
                cas_result = self._adapter.execute(command)
        return cas_result

    def _finalize_solve_results(self, cas_result, step_id: str, output_id: str) -> None:
        """Phase 14: parse a solve / inverse-transform result and register each
        (var = val) solution row, promoting the first into the pre-allocated
        output_id slot so the result keeps its expected E-number."""
        # FREE-PARAM instance 4: an under-determined solve returns free variables
        # as Maxima %rN — rename them to named free parameters (the original free
        # variable where possible) and tag the solution rows with them.
        _clean_val, _free_names = self._rename_free_solve_vars(cas_result.value or "")
        # An empty solution list is NOT a result.  Maxima answers `[]` when the
        # system has no solution (or it cannot find one), and with nothing to
        # register the pre-allocated slot kept the raw `[]` — the timeline
        # showed `E3 = \left[ \right]`, a dead end that consumed an expression
        # number and could be referenced by later steps as if it were a value
        # (measured 2026-08-01 on the S-c Bogoliubov condition × normalisation,
        # and on a plainly inconsistent `x+y=1, x+y=2`).  Fail with a reason
        # instead, the same shape as any other CAS failure.
        if _clean_val.replace(" ", "") in ("[]", "[[]]"):
            self.expressions.pop(output_id, None)
            self._expr_counter -= 1
            raise ValueError(t("error.solve_no_solution", self.lang))
        # The first row registers straight INTO output_id, which is what
        # removed the promote-and-delete that used to live here (and the
        # lhs_alias_map redirect it needed).  Why, measured, is in
        # _register_solve_results' docstring.
        _solve_exprs = self._register_solve_results(
            _clean_val, step_id,
            cas_latex="" if _free_names else (cas_result.latex or ""),
            output_id=output_id,
        )
        for _se in _solve_exprs:
            _se.free_parameters = list(_free_names)
        self._declare_free_parameters(_free_names)   # docs/32 §2.7
        if _solve_exprs:
            _extra_ids = [e.display_id for e in _solve_exprs]
            _cur_step = self.steps[step_id]
            try:
                self.steps[step_id] = _cur_step.model_copy(
                    update={"output_refs": _extra_ids}
                )
            except Exception:
                _logger.warning(
                    "failed to update step %s output_refs after solve; "
                    "history may be incomplete", step_id, exc_info=True)

    def _update_symbol_context_after_exec(
        self,
        operation: str,
        args: dict,
        step: "Step",
        step_id: str,
        saved_all_names: "list[str] | None",
        type_decl_ops: "frozenset[str]",
    ) -> "Step":
        """Update SymbolContext and Step metadata after successful CAS execution.

        Handles operator declarations, type declarations, symmetry declarations,
        coordinate system registration, and vector/scalar field registration.
        Returns the (possibly updated) *step* object.
        """
        self._register_operator_declaration(operation, args)
        self._register_type_declaration(operation, args, saved_all_names)
        self._register_symmetry_declaration(operation, args)
        step = self._register_coordinate_system(operation, args, step, step_id)
        step = self._register_field_declaration(operation, args, step, step_id)
        step = self._record_registered_names(
            operation, args, step, step_id, saved_all_names, type_decl_ops)
        return step

    # ------------------------------------------------------------------
    # _update_symbol_context_after_exec phases (R3, docs/27): per-declaration-
    # family helpers extracted verbatim so the dispatcher above reads as a list.
    # The first three mutate only SymbolContext (no step); the last three thread
    # the Step (returning it possibly updated).
    # ------------------------------------------------------------------
    def _register_operator_declaration(self, operation: str, args: dict) -> None:
        """Operator declarations (boson/fermion/spin) → SymbolContext statistics."""
        from notebook.models.symbol_context import SymbolStatistics
        if operation in ("define_boson_operator", "define_fermion_operator", "define_spin_operator"):
            _op_name = args.get("name", "")
            if _op_name:
                if operation == "define_spin_operator":
                    # Register the base operator and Cartesian projections: Sx, Sy, Sz.
                    # Sp/Sm (@tlspm_ops) are intentionally NOT registered — define S+/S-
                    # manually as Sx+i*Sy / Sx-i*Sy to avoid normal_form mixing issues.
                    targets = [_op_name] + [f"{_op_name}{s}" for s in ("x", "y", "z")]
                    for _tgt in targets:
                        existing = self.symbol_context.get(_tgt)
                        if existing is not None:
                            self.symbol_context.registry[_tgt] = existing.model_copy(
                                update={"statistics": SymbolStatistics(kind="noncommutative")}
                            )
                        else:
                            self.symbol_context.register(
                                _tgt,
                                quantity=Quantity(name=_tgt, kind="operator"),
                                statistics_kind="noncommutative",
                            )
                else:
                    _stats = "bose" if operation == "define_boson_operator" else "fermi"
                    existing = self.symbol_context.get(_op_name)
                    if existing is not None:
                        self.symbol_context.registry[_op_name] = existing.model_copy(
                            update={"statistics": SymbolStatistics(kind=_stats)}
                        )
                    else:
                        self.symbol_context.register(
                            _op_name,
                            quantity=Quantity(name=_op_name, kind="operator"),
                            statistics_kind=_stats,
                        )

    def _record_spin_projections(self, op_name: str, particle_key: str) -> None:
        r"""宣言に書かれた粒子から、スピン射影の値集合（分子）を記録する。

        統計は分子の**偶奇**を決めるが**大きさ**は決めない（スピン 3/2 の
        フェルミオンは実在する）ので、$2S$ は粒子ごとのデータ
        （`models/particles.json`）から来る。決まらない粒子では何も記録しない
        ——空は「まだ言われていない」であって「射影が無い」ではない。
        """
        if not op_name or not particle_key:
            return
        from notebook.runtime.spin_value_set import projections_for_particle
        nums = projections_for_particle(particle_key)
        if nums:
            self.symbol_context.spin_projections[op_name] = nums

    def _record_nuclide_projections(self, op_name: str, description: str) -> None:
        r"""宣言文に書かれた**核種**から、スピン射影の値集合（分子）を記録する。

        $^{13}$C のスピンは「炭素」からは決まらない（$^{12}$C は 0、$^{14}$C も 0）
        ので、同一性は核種 (Z, A) である。書かれていなければ何も記録しない
        ——空は「まだ言われていない」。
        """
        if not op_name or not description:
            return
        from notebook.runtime.spin_value_set import projections_for_nuclide
        nums = projections_for_nuclide(description)
        if nums:
            self.symbol_context.spin_projections[op_name] = nums

    def _register_type_declaration(self, operation: str, args: dict,
                                   saved_all_names: "list[str] | None") -> None:
        """Algebraic type declarations (scalar_parameter/vector/matrix/tensor);
        a sign qualifier on a scalar parameter is also assume()'d in the CAS."""
        from notebook.models.symbol_context import SymbolStatistics
        _TYPE_DECL_ALGEBRA: dict[str, str] = {
            "define_scalar_parameter": "scalar",
            "define_vector":           "vector",
            "define_matrix":           "matrix",
            "define_tensor":           "tensor",
        }
        if operation in _TYPE_DECL_ALGEBRA:
            from notebook.models.quantity import MathType
            _algebra = _TYPE_DECL_ALGEBRA[operation]
            _domain = str(args.get("domain", "") or "").lower().strip() or None
            _sign = str(args.get("sign", "") or "").lower().strip() or None
            for _sym_name in (saved_all_names or [args.get("name", "")]):
                if not _sym_name:
                    continue
                existing = self.symbol_context.get(_sym_name)
                if existing is not None:
                    _qty_update: dict = {"math_type": MathType(algebra=_algebra)}
                    if _domain is not None:
                        _qty_update["domain"] = _domain
                    if _sign is not None:
                        _qty_update["sign"] = _sign
                    self.symbol_context.registry[_sym_name] = existing.model_copy(
                        update={
                            "statistics": SymbolStatistics(kind="bose"),
                            "quantity": existing.quantity.model_copy(update=_qty_update),
                        }
                    )
                else:
                    self.symbol_context.register(
                        _sym_name,
                        quantity=Quantity(
                            name=_sym_name, kind="scalar",
                            math_type=MathType(algebra=_algebra),
                            domain=_domain,
                            sign=_sign,
                        ),
                        statistics_kind="bose",
                    )
                # A sign qualifier ("positive"/"negative"/"nonneg"/"nonpos") is not just
                # metadata on a scalar parameter — it is a real constraint the CAS should
                # know about (e.g. so subsequent integrals/limits don't hit an unresolved
                # sign branch).  Delegated to the backend handler (Backend Encapsulation
                # Rule): Maxima sends an actual assume(); backends with no assume()
                # mechanism (Julia/Symbolics has no sign predicate) no-op by default.
                if operation == "define_scalar_parameter" and _sign is not None:
                    self._handler.assume_sign(self, _sym_name, _sign, _domain)
                # The number DOMAIN is the same kind of thing as the sign and was
                # the half that never left the notebook: recorded on the Quantity,
                # shown as accepted, never sent (open_items COMPLEX-DECL-NOT-SENT).
                if operation == "define_scalar_parameter" and _domain is not None:
                    self._handler.declare_domain(self, _sym_name, _domain)

    def _register_symmetry_declaration(self, operation: str, args: dict) -> None:
        """declare_symmetry: store the index symmetry on the quantity.

        `対称行列` のように**型も言っている**ラベルは `algebra` を伴う。それも
        記録しないと、対称性を取った代わりに「行列であること」が落ちる
        ——研究者はどちらも書いている（実測 2026-09-05）。
        """
        if operation == "declare_symmetry":
            _sym_name = str(args.get("name", ""))
            _symmetry = str(args.get("symmetry", ""))
            _algebra = str(args.get("algebra", "") or "")
            if _sym_name and _symmetry:
                existing = self.symbol_context.get(_sym_name)
                if existing is not None:
                    self.symbol_context.registry[_sym_name] = existing.model_copy(
                        update={
                            "quantity": existing.quantity.model_copy(
                                update={"symmetry": _symmetry}
                            )
                        }
                    )
                else:
                    self.symbol_context.register(
                        _sym_name,
                        quantity=Quantity(name=_sym_name, kind="anonymous", symmetry=_symmetry),
                    )
            # **代数型は対称性の後に書く。** 量がまだ無いときは上の `else` が
            # 作るので、先に書くと「まだ無い」ため取りこぼす（実測 2026-09-05:
            # `A を対称行列とする` で algebra が args に在るのに記録されなかった）。
            if _sym_name and _algebra:
                from notebook.models.quantity import MathType
                _ex = self.symbol_context.get(_sym_name)
                if _ex is not None:
                    self.symbol_context.registry[_sym_name] = _ex.model_copy(
                        update={"quantity": _ex.quantity.model_copy(
                            update={"math_type": MathType(algebra=_algebra)})})

    def _register_coordinate_system(self, operation: str, args: dict,
                                    step: "Step", step_id: str) -> "Step":
        """define_coordinate_system: persist the custom coord system to the session
        and normalize vars/lame into the step args."""
        if operation == "define_coordinate_system":
            _raw_vars = args.get("vars", "")
            _raw_lame = args.get("lame", "")
            _cv = (
                tuple(v.strip() for v in _raw_vars.split(",") if v.strip())
                if isinstance(_raw_vars, str)
                else tuple(str(v).strip() for v in _raw_vars)
            )
            _cl = (
                tuple(h.strip() for h in _raw_lame.split(",") if h.strip())
                if isinstance(_raw_lame, str)
                else tuple(str(h).strip() for h in _raw_lame)
            )
            if len(_cv) == 3 and len(_cl) == 3:
                self.session = self.session.model_copy(update={
                    "coordinate_system": "custom",
                    "custom_coord_vars": _cv,
                    "custom_coord_lame": _cl,
                })
            step = step.model_copy(update={"operation_args": {
                **step.operation_args,
                "vars": ",".join(_cv) if _cv else "",
                "lame": ",".join(_cl) if _cl else "",
            }})
            self.steps[step_id] = step
        return step

    def _register_field_declaration(self, operation: str, args: dict,
                                    step: "Step", step_id: str) -> "Step":
        """define_vector_field / define_scalar_field: register the field (and, for
        a vector field, its components) and record the names on the step."""
        if operation in ("define_vector_field", "define_scalar_field"):
            from notebook.models.quantity import MathType
            _field_name = args.get("name", "")
            _coord_sys  = str(
                args.get("coordinate_system")
                or getattr(self.session, "coordinate_system", None)
                or "cartesian"
            )
            _raw_vars_f = args.get("natural_vars") or args.get("vars") or []
            if isinstance(_raw_vars_f, str):
                _raw_vars_f = [v.strip() for v in _raw_vars_f.split(",") if v.strip()]
            if not _raw_vars_f:
                if _coord_sys == "custom" and self.session.custom_coord_vars:
                    _raw_vars_f = list(self.session.custom_coord_vars)
                else:
                    _raw_vars_f = list(self._handler.reserved_coord_vars(_coord_sys))

            # "時間依存のスカラー場/ベクトル場として定義して" (see backend_handler.py
            # _label_wants_time_dependent, folded into natural_vars by
            # _preprocess_args before "_"-prefixed args were stripped): mark the
            # registered Quantity itself time-dependent so differentiate's
            # partial/total ambiguity check and dot notation work, matching
            # declare_time_dependent's semantics.
            _field_qty_kwargs = (
                {"time_dependent": True, "time_parameter": "t"}
                if "t" in _raw_vars_f else {}
            )

            if operation == "define_vector_field":
                _suffixes = (
                    tuple(self.session.custom_coord_vars)
                    if _coord_sys == "custom" and self.session.custom_coord_vars
                    else self._handler.coord_component_suffixes(_coord_sys)
                )
                _components = [f"{_field_name}_{s}" for s in _suffixes]
                self.symbol_context.register(
                    _field_name,
                    quantity=Quantity(
                        name=_field_name, kind="vector_field",
                        components=_components,
                        coordinate_system=_coord_sys,
                        natural_vars=list(_raw_vars_f),
                        **_field_qty_kwargs,
                    ),
                )
                for _comp in _components:
                    self.symbol_context.register(
                        _comp,
                        quantity=Quantity(
                            name=_comp, kind="function",
                            math_type=MathType(algebra="scalar"),
                            natural_vars=list(_raw_vars_f),
                            **_field_qty_kwargs,
                        ),
                    )
                step = step.model_copy(update={"operation_args": {
                    **step.operation_args,
                    "registered_names": [_field_name] + _components,
                    "coordinate_system": _coord_sys,
                    "natural_vars": list(_raw_vars_f),
                }})
                self.steps[step_id] = step
            else:  # define_scalar_field
                self.symbol_context.register(
                    _field_name,
                    quantity=Quantity(
                        name=_field_name, kind="scalar_field",
                        coordinate_system=_coord_sys,
                        natural_vars=list(_raw_vars_f),
                        **_field_qty_kwargs,
                    ),
                )
                step = step.model_copy(update={"operation_args": {
                    **step.operation_args,
                    "registered_names": [_field_name],
                    "coordinate_system": _coord_sys,
                    "natural_vars": list(_raw_vars_f),
                }})
                self.steps[step_id] = step
        return step

    def _record_registered_names(self, operation: str, args: dict, step: "Step",
                                 step_id: str, saved_all_names: "list[str] | None",
                                 type_decl_ops: "frozenset[str]") -> "Step":
        """Store registered_names on the step so delete_define_step can undo a
        multi-name type declaration or a spin-operator declaration."""
        if operation in type_decl_ops and saved_all_names and len(saved_all_names) > 1:
            step = step.model_copy(update={"operation_args": {
                **step.operation_args, "registered_names": saved_all_names
            }})
            self.steps[step_id] = step
        elif operation == "define_spin_operator" and (base_op := args.get("name")):
            spin_names = [base_op] + [f"{base_op}{s}" for s in ("x", "y", "z")]
            step = step.model_copy(update={"operation_args": {
                **step.operation_args, "registered_names": spin_names
            }})
            self.steps[step_id] = step
        return step

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Assumption management
    # ------------------------------------------------------------------

    def _alloc_assumption_id(self) -> str:
        n = len(self.assumptions) + 1
        return f"ASM{n}"

    def add_assumption(self, canonical: str, display: str = "") -> tuple[Step, Assumption]:
        """Add a CAS assumption (e.g. assume(x > 0)) and record the step.

        The stored ``canonical`` stays the researcher's own spelling; only the
        RENDERED command is put into the backend's dialect, via
        ``assumption_canonical``.  Maxima writes a not-equal assumption as
        ``notequal(a, b)`` even though the expression language spells it ``a # b``.
        """
        # The COMMAND is the backend's to spell, not this layer's.  Built here,
        # `assume(...)` reached a Julia session verbatim and the researcher's
        # "X > 0 と仮定する" came back as `Sym assume is not callable` — raw CAS
        # text naming a function nobody wrote (open_items BP-5).
        from notebook.runtime.backend_handler import UnsupportedAssumption
        try:
            rendered = self._handler.assumption_command(canonical)
        except UnsupportedAssumption as exc:
            raise ValueError(t(exc.i18n_key, self.lang,
                               condition=display or canonical, **exc.fmt)) from None
        result = self._adapter.execute(rendered)
        if not result.ok:
            raise ValueError(result.error)

        assumption_id = self._alloc_assumption_id()
        step_id = self._alloc_step_id()
        assumption = Assumption(
            assumption_id=assumption_id,
            display=display or canonical,
            canonical=canonical,
            scope="branch",
            branch_id=self.active_branch_id,
        )
        step = Step(
            step_id=step_id,
            kind=StepKind.assumption_add,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.assume", self.lang, condition=display or canonical),
            operation_args={"assumption_id": assumption_id, "canonical": canonical,
                            "display": display or canonical},
            rendered_command=rendered,
            state_affecting=True,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.assumptions[assumption_id] = assumption
        self.steps[step_id] = step
        self._advance_head(step)
        return step, assumption

    def remove_assumption_by_canonical(self, canonical: str) -> Step:
        """Remove an assumption by its canonical condition string."""
        for aid, asm in self.assumptions.items():
            if asm.canonical == canonical:
                return self.remove_assumption(aid)
        raise ValueError(f"No active assumption with condition '{canonical}'")

    def remove_assumption(self, assumption_id: str) -> Step:
        """Remove a previously added assumption and record the step."""
        if assumption_id not in self.assumptions:
            raise ValueError(f"Assumption '{assumption_id}' not found")
        assumption = self.assumptions[assumption_id]
        # Same door as add_assumption.  `None` means this backend never sent the
        # assumption anywhere, so there is nothing to take back — the notebook
        # still drops it, which is the half that was always ours.
        rendered = self._handler.forget_command(assumption.canonical)
        if rendered:
            self._adapter.execute(rendered)  # best-effort; Maxima may warn if not found

        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.assumption_remove,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.forget", self.lang, condition=assumption.display),
            rendered_command=rendered,
            state_affecting=True,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        del self.assumptions[assumption_id]
        self.steps[step_id] = step
        self._advance_head(step)
        return step

    # ------------------------------------------------------------------
    # Dependency context checks (CAS-agnostic interface)
    # ------------------------------------------------------------------

    def check_diff_context(self, expr_canonical: str, diff_var: str) -> list[tuple[str, str]]:
        """Return (dependent_var, independent_var) pairs that would affect diff.

        Queries the active CAS state and compares against variables present in
        expr_canonical.  Returns [] when no conflicts exist or adapter is not running.
        This method is CAS-agnostic: callers need not know Maxima syntax.
        """
        if not self._adapter or not self._adapter.is_running:
            return []
        all_deps = self._adapter.get_active_dependencies()
        if not all_deps:
            return []
        expr_vars = set(re.findall(r'\b([A-Za-z_]\w*)\b', expr_canonical)) - _MATH_FUNC_NAMES_ENGINE
        return [(v, d) for v, d in all_deps if v in expr_vars and d == diff_var]

    def remove_variable_dependency(self, var: str) -> bool:
        """Remove all CAS dependency declarations for var.

        CAS-agnostic: delegates to the adapter without exposing backend syntax.
        """
        if not self._adapter or not self._adapter.is_running:
            return False
        return self._adapter.remove_dependency(var)


    def _collect_rendered_commands(self) -> list[str]:
        """Return rendered CAS commands for the active branch in chronological order.

        Only includes successful, state-affecting cas_operation steps.
        Used by save_maxima_snapshot() to build a self-contained replay script.
        """
        from notebook.runtime.replay import compute_lineage
        from notebook.models.step import StepKind, StepStatus
        lineage = compute_lineage(self.active_branch_id, self.steps, self.branches)
        return [
            step.rendered_command
            for step in lineage
            if step.state_affecting
            and step.kind == StepKind.cas_operation
            and step.status == StepStatus.success
            and step.rendered_command
        ]

    # Replay
    # ------------------------------------------------------------------

    def replay_current_branch(self) -> ReplayResult:
        """Replay the active branch and return the rebuilt expression table."""
        return replay(
            self.active_branch_id,
            self.steps,
            self.branches,
            self.expressions,
            self._adapter,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


