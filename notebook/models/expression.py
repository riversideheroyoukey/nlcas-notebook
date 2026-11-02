from __future__ import annotations
from typing import Literal, Optional, Any
import uuid
from pydantic import BaseModel, Field, model_validator

from notebook.models.ir import Expr

class Expression(BaseModel):
    ir_ast: Optional[Expr] = Field(default=None, exclude=True)
    expr_uid: str = Field(default_factory=lambda: f"expr_{uuid.uuid4().hex[:6]}")
    display_id: str
    branch_id: str
    latex: str
    canonical: str
    created_by_step: Optional[str] = None
    simplification_log: list[str] = Field(default_factory=list)
    # Derivative provenance — set when this expression is the result of differentiate.
    # Enables iterated/mixed partial display: ∂²c/∂a∂b, ∂²f(a,b)/∂a∂b, …
    quantity_name: Optional[str] = None           # bare quantity name, e.g. "c" or "f"
    derivative_vars: list[str] = Field(default_factory=list)  # accumulated vars ["a","b"]
    derivative_is_partial: bool = False
    # deprecated: superseded by quantity_name; kept for loading pre-migration sessions
    derivative_origin: Optional[str] = None
    # CAS result representation: human-readable string returned by the CAS.
    # Always set for CAS results; None for directly-registered expressions.
    # Use display_repr (property) instead of cas_repr directly.
    cas_repr: Optional[str] = None

    # Symbols that appear in this expression but are not registered in SymbolContext.
    # Set by engine.register_expression for Julia sessions; empty for Maxima.
    # Format: bare symbol name, e.g. ["xi", "Delta"] (without indices).
    undefined_symbols: list[str] = Field(default_factory=list)
    #: Non-blocking notices about this expression, already translated.  A
    #: WARNING is for something that is legitimate writing but might not be
    #: what was meant — a bound index reusing a name the expression uses
    #: elsewhere.  Neither reading can be ruled out from the text, so nothing
    #: is refused; the only unacceptable outcome is silence.  Additive: an
    #: older saved expression loads with none.
    warnings: list[str] = Field(default_factory=list)

    # The user-facing canonical expression string BEFORE the CAS assigned a variable
    # name (e.g. before Julia's canonical_for_result transforms it to "_nb_E1").
    # Used by JuliaCodeGenerator to reconstruct the IR of stored expressions so that
    # commutator linearity can be distributed over algebraic index terms.
    input_canonical: str | None = None

    # Comma-separated dummy index names for sum display (e.g. "k" → \sum_{k}[...]).
    # Propagated from build_command.sum_index_hint and carried forward through
    # combine_expressions, substitute, relabel_index, and operator_substitute.
    sum_index_hint: str = ""

    # SymPy 計算結果の場合にセット（canonical は表示用のまま変更しない）
    # SymPy チェーン継続に使う。None なら SymPy 結果でない。
    sympy_canonical: str | None = None

    # If True, this expression is an internal intermediate (e.g. H-8 temp RHS)
    # and should not appear in the timeline or be reachable via _active_expr_ids.
    hidden: bool = False

    # True when the expression was created as SCAFFOLDING and never had a
    # researcher-facing number: its display_id is `_T1`, `_T2`, … rather than
    # `E7` (INTERNAL-INTERMEDIATE-EID-GAP).  `hidden` says "stop offering this
    # as an operand"; `internal` says "this never was one of the researcher's
    # numbered expressions", and the two are set at different times — an
    # internal intermediate stays reachable (hidden=False) while the operation
    # that needs it runs, and is hidden afterwards.  Display layers must skip
    # either.
    internal: bool = False

    # B (symbolic lattice sums): when set, this expression is an UNEXPANDED
    # bond sum Σ_C body(i,j) kept symbolically. Holds the descriptor needed to
    # concretize it later ({body, index1, index2, pair_constraint, sum_label,
    # lattice_shape?}). The latex shows the \sum_{…} form; the canonical is not
    # a backend-computable string until concretized via expand_neighbor_sum.
    lattice_descriptor: Optional[dict] = None

    # Unevaluated definite integral (S-o change-of-variables workflow): when set,
    # this expression is a display-only multiple definite integral that is NOT sent
    # to the CAS until explicitly evaluated.  Holds {integrand, bounds:[[var,lo,hi],
    # …], region?}.  The latex shows the ∫…∫ form; change-of-variables rewrites the
    # descriptor (new integrand·|J| and new bounds), and an evaluate step assembles
    # the iterated CAS integrate.  Mirrors lattice_descriptor's deferred-eval design.
    integral_descriptor: Optional[dict] = None
    # PAIR-REP: a Heisenberg/interaction-picture operator, held as the SUM OF
    # (scalar coefficient, QA operator) PAIRS it actually is.  QuantumAlgebra
    # cannot hold the coefficient (`e^{-i\varepsilon t}` is decomposed into
    # cos/sin with an `exp(0)` remnant — docs/33 §6.1), so the coefficient lives
    # here and the operator stays a QA object.  Same deferred-evaluation shape
    # as lattice_/integral_descriptor: the display is the researcher's notation
    # `\hat c_n(t)`, the substance is the pair.
    pair_representation: Optional[dict] = None

    # Extremum binder (EXTREMUM-BINDER): `\max_i x_i`, `\sup_{x \in X} f(x)`.
    # Display-only, and for a different reason than the two above — it is not
    # that evaluation is DEFERRED but that the backend has no form for it at
    # all (Maxima's `lmax` needs a concrete list, so a symbolic range has no
    # command).  Holds {op, index_latex, body_latex}.  Written down and shown
    # exactly as typed; a generic operation on it is refused at the point of
    # USE (operand_guards) rather than at registration — measured 2026-08-16,
    # refusing at registration stopped the researcher writing an ordinary
    # objective function in their own notebook.
    extremum_descriptor: Optional[dict] = None

    # Angular-momentum coupling provenance (F-9): the quantum numbers this
    # object was computed from, e.g. {"kind": "clebsch_gordan", "j1": "1/2",
    # "m1": "1/2", …} or {"kind": "coupling", "j1": "3/2", "j2": "1/2",
    # "allowed": ["1", "2"]}.  The VALUE is exact and already in `canonical`
    # (F-9 never calls the CAS), so unlike lattice_/integral_descriptor this
    # carries no deferred computation — it records what the numbers MEAN, so a
    # follow-up step (the j1·j2 eigenvalue of a coupling, a Landé g factor)
    # can be taken from a registered expression instead of re-typed.
    angular_descriptor: Optional[dict] = None

    # Free parameters / arbitrary constants this result carries (FREE-PARAM): the
    # degrees of freedom of a general solution — the integration constant of an
    # indefinite integral (C), the arbitrary constants of an ODE general solution
    # (k_1, k_2), the initial-value placeholders of a Laplace-solved ODE (q_0, …).
    # Kept symbolic by default (math workflow); a specialization operation
    # determines them by value/condition (physics workflow).  Names are the
    # display/canonical symbols (e.g. "C").
    free_parameters: list[str] = Field(default_factory=list)

    @property
    def display_repr(self) -> str:
        """Human-readable representation for display; falls back to canonical."""
        return self.cas_repr if self.cas_repr is not None else self.canonical


class Assumption(BaseModel):
    assumption_id: str
    display: str
    canonical: str
    scope: Literal["branch", "session"]
    branch_id: Optional[str] = None

    @model_validator(mode="after")
    def branch_id_required_for_branch_scope(self) -> Assumption:
        if self.scope == "branch" and self.branch_id is None:
            raise ValueError("branch_id is required when scope is 'branch'")
        return self
