"""NotebookEngine mixin: what the session knows about a Laplace transform PAIR.

Extracted verbatim from engine.py (refactor Stage 3, 2026-07-04),
following the EngineHistoryMixin / EnginePersistenceMixin pattern:
state is reached via self; NotebookEngine appears only in annotations
(PEP 563), so there is no import from engine.py and no cycle.

**`laplace_by_definition` was deleted on 2026-08-15** (SCENARIO-MACRO case 7)
together with its by-parts, linearity and IC-display helpers — 272 lines.  It
was a macro: measured, both of its scenarios reproduce their published results
using general operations only, because the two things it did are separately
available.

* an EXPLICIT function's transform is the definite integral itself
  (`\\int_0^\\infty \\sin(\\omega t)e^{-st}dt` through `integrate_definite`), and
* a DERIVATIVE's transform is a THEOREM, not a computation — the researcher
  writes `s^2 X - s x_0 - v_0` and the system composes it in.  There is no
  choice to make there (`u = e^{-st}` is fixed by the transform), which is
  exactly why it can be written down rather than derived.

Its convergence guard is not missed either: without an assumption the general
path stops with the CAS's own question, names the symbol, and tells the
researcher to assume its sign — an explicit, visible step instead of an
assume/forget the macro performed silently (measured; the session survives).

`define_laplace_transform` STAYS, but not for the reason first written here.

It was claimed that the transform PAIR cannot be expressed with general
operations.  **Measured, that is false** — and the measurement is worth keeping,
because it is what the alternative design rests on:

    laplace(x(t),t,s)             → 'laplace(x(t),t,s)   (Maxima's own noun)
    ilt(laplace(x(t),t,s),s,t)    → x(t)                 ← the CAS knows the pair
    ilt(X,s,t)                    → X*delta(t)           ← it does not know `X`

So the pair IS knowledge the CAS holds, keyed on its noun form; what this table
holds is only the researcher's NAMING of that object (capital = image, the
engineering convention).  Naming is what H-1 already does for every other
quantity, so anchoring `X` to the noun form would make the table redundant and
let the label come from the CAS itself — at the cost of anchoring an s-domain
symbol to the builtin's noun, which is the line these scenarios deliberately do
not cross.  Kept as a DECLARATION for now, with that route recorded as
open_items LAPLACE-PAIR-ANCHOR and taken only if this one gives trouble.
"""
from __future__ import annotations

from notebook.models.expression import (  # noqa: F401
    Expression,
)
from notebook.runtime.i18n import (  # noqa: F401
    t,
)


class EngineLaplaceMixin:
    """The Laplace transform PAIR: its declaration, and the labels it enables."""

    def _lap_cap(self, base: str) -> str:
        """The transform symbol for a time-domain base (q→Q), honouring an explicit
        ``define_laplace_transform`` declaration, else the uppercased base."""
        return getattr(self, "_laplace_symbols", {}).get(base, base.upper())

    def _laplace_time_lhs(self, lhs: str, args: dict) -> str:
        """``x(t)`` for an equation whose left side is the transform symbol ``X``.

        The label an inverse transform's result should carry.  Only a base this
        session actually DECLARED a transform for qualifies — uppercasing back
        would invent a name (`ABC` → `abc`) for any capitalised symbol that never
        was a transform, and a wrong label is worse than none.
        """
        lhs = (lhs or "").strip()
        t = str(args.get("t") or "t")
        for base in getattr(self, "_laplace_symbols", {}) or {}:
            if self._lap_cap(base) == lhs:
                return f"{base}({t})"
        return ""

    def _format_inverse_transform_latex(self, operation, args, out_latex,
                                        source_was_equation, source_lhs) -> str:
        """One link of `_format_result_latex`'s chain (R3, docs/27): label an
        inverse transform's result with its TIME-domain name.

        `_format_single_expr_latex` cannot serve here — it carries the source's
        LHS forward verbatim, and the source's LHS is `X`, the s-domain symbol.
        The label is `x(t)`, which only the transform table knows; that is why
        this link lives beside it rather than in engine.py.
        """
        if operation != "inverse_laplace" or not (source_was_equation and source_lhs):
            return out_latex
        if "=" in out_latex:          # already a statement — no second left side
            return out_latex
        return f"{source_lhs} = {out_latex}"
    def define_laplace_transform(self, base: str, cap: "str | None" = None,
                                 t: str = "t", s: str = "s") -> Expression:
        """Lay out the Laplace-transform DEFINITION as a card:
        ``Cap(s) = ∫₀^∞ base(t) e^{-st} dt`` — so the s-domain symbol Cap(s) used by
        the transform is anchored to its definition, not a magic symbol."""
        cap = cap or base.upper()
        syms = getattr(self, "_laplace_symbols", None)
        if syms is None:
            syms = self._laplace_symbols = {}
        syms[base] = cap
        latex = (rf"{cap}({s}) = \int_0^{{\infty}} {base}({t})\,"
                 rf" e^{{-{s} {t}}}\, d{t}")
        canonical = f"'integrate({base}*exp(-({s})*({t})),{t},0,inf)"
        return self._register_plain(latex, canonical)
    def _raw_tex(self, expr_str: str) -> str:
        """tex1 of a Maxima expression string, without display postprocessing.

        Its only caller is `substitute_free_parameters`.  The display bridge this
        used to name was deleted with `laplace_by_definition` (2026-08-15) —
        nothing produced the `x_ic{n}` tokens it rewrote any more.
        """
        try:
            r = self._adapter.execute(f"tex1({expr_str})")
            return r.value.strip() if (r.ok and r.value) else expr_str
        except Exception:
            return expr_str
