"""A scalar expression handed from one backend's session to another's.

SCALAR-HANDOFF (docs/developer_manuals/34 §5): operator algebra is
QuantumAlgebra's job and `(k, iω_n)`-space scalar algebra is Maxima's, so a
condensed-matter calculation that has finished with the operators has to cross
a session boundary to continue.  The crossing is a FORK, not a switch — Core
Principle #6 (the backend is fixed at session creation) stands, because nothing
is carried over as live state: what travels is the researcher's own writing plus
the declarations needed to re-establish it, and the target session executes
every step for real.

The provenance fields mirror ``NumericTask``'s deliberately: both are "this
expression left the theory timeline and continued elsewhere", and in both cases
the snapshot must survive later edits to the source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HandoffDeclaration(BaseModel):
    """One symbol the target session must be told about before the expression.

    Carries the SIGN as well as the type.  Measured 2026-07-31 (open_items
    SIGN-DIALOG-ASSUME): with the type alone, Maxima stops mid-integral to ask
    "Is omega_D positive, negative or zero?" — so a handoff that forgets the
    sign hands over an expression the target session cannot actually evaluate.
    """
    name: str
    kind: str = "scalar"                     # Quantity.kind
    domain: Optional[str] = None             # "real" / "integer" / …
    sign: Optional[str] = None               # "positive" / "negative" / …


class ScalarHandoff(BaseModel):
    """The descriptor produced in the SOURCE session."""
    handoff_id: str
    # --- provenance (snapshot; must survive later edits to the source) ---
    source_expression_ids: list[str] = Field(default_factory=list)
    source_latex: str = ""
    source_canonical: str = ""
    source_assumption_ids: list[str] = Field(default_factory=list)
    source_branch_id: str = "main"
    source_backend: str = ""
    source_title: str = ""
    source_file: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # --- what the target session needs ---
    target_backend: Literal["maxima", "julia_full"] = "maxima"
    declarations: list[HandoffDeclaration] = Field(default_factory=list)
    #: Symbols with no declaration in the source session.  Not an error — free
    #: variables (x, t, k) are ordinary — but the caller may want to ask.
    undeclared: list[str] = Field(default_factory=list)
    #: The expression as the researcher wrote it.  A Julia RESULT's canonical is
    #: an opaque `_nb_Ek` handle, so the LaTeX display form is what travels
    #: (docs/32 made that form trustworthy; docs/34 §5 measured it working).
    export_latex: str = ""
    #: The label the value wore in the source, when that label could NOT travel
    #: — an operator product like ``⟨c_{-k↓}c_{k↑}⟩``, which a scalar CAS has no
    #: meaning for.  The value crosses without it, so this is where "what the
    #: number is the value OF" survives; the target session's note quotes it.
    source_label_latex: str = ""
    #: The source's OWN reading of the exported LaTeX, produced by the source
    #: backend's ``latex_to_canonical``.  This is what the crossing check
    #: compares against, not ``source_canonical``: a Julia result's canonical is
    #: the opaque handle ``_nb_Ek``, and comparing a handle with a value made the
    #: check report **differed** — a false alarm that tells the researcher a
    #: correct result is wrong, which is worse than admitting it cannot check.
    #: Two independent readings of the same LaTeX (source handler vs target
    #: handler) is exactly the round-trip the check exists to test.
    source_value_canonical: str = ""
    #: Outcome of the value spot-check on the crossing (see
    #: ``scalar_handoff.verify_crossing``).  ``not_checkable`` is a real
    #: outcome and is never reported as agreement.
    verification: dict = Field(default_factory=dict)
