"""NotebookEngine mixin: F-13 multi-species spin — per-site spin magnitudes.

The spin algebra (``notebook/runtime/spin_algebra.py``) represents operators as
"component × integer site" and already treats different sites as commuting, so
an electron at site 1 and a nucleus at site 2 could always be WRITTEN
(``A I·S = A Σ_a S^a_1 S^a_2``) and commuted.  What could not be done was
EVALUATED: the infinite-temperature trace assumed one spin magnitude for every
site (a single ``x = S(S+1)``).

This mixin adds the missing half: declare each site's spin, then take
expectation values with ``x_i = S_i(S_i+1)`` per site.  Both operations are
CAS-bypass — the trace is exact rational arithmetic in Python — but the RESULT
is registered with a CAS-ready canonical (``(x1*x2)/3``, ``1/2``) so it composes
with ordinary operations like any other value.
"""
from __future__ import annotations

from fractions import Fraction

from notebook.models.expression import Expression  # noqa: F401
from notebook.models.step import Step, StepKind, StepStatus  # noqa: F401
from notebook.runtime import spin_algebra as spa
from notebook.runtime.i18n import t


class EngineSpinSpeciesMixin:
    """F-13: per-site spin magnitudes (see module docstring)."""

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    def declare_site_spins(self, spins: "dict") -> Step:
        """Record the spin magnitude of one or more sites.

        ``{1: "1/2", 2: "1"}`` — an electron and a spin-1 nucleus.  A site
        mapped to ``None``/``""`` keeps its spin symbolic.

        The declaration is a step in the notebook, not a hidden setting: which
        spins a trace was taken with is part of the result's meaning, and the
        session is a research log.  It is stored in ``SymbolContext`` and
        survives save/load.
        """
        if not spins:
            raise ValueError(t("spin_species.no_sites", self.lang))
        normalized: "dict[str, str]" = {}
        for site, value in spins.items():
            site_key = str(site).strip()
            if not site_key.isdigit():
                raise ValueError(t("spin_species.bad_site", self.lang, site=site_key))
            if value in (None, ""):
                normalized[site_key] = ""
                continue
            spin = Fraction(str(value).strip())
            if spin < 0 or (2 * spin).denominator != 1:
                raise ValueError(t("spin_species.bad_spin", self.lang,
                                   site=site_key, spin=str(value)))
            normalized[site_key] = str(spin)
        self.symbol_context.site_spins.update(normalized)

        shown = ", ".join(
            f"S_{k} = {v}" if v else f"S_{k} = ?"
            for k, v in sorted(normalized.items(), key=lambda kv: int(kv[0])))
        step_id = self._alloc_step_id()
        step = Step(
            step_id=step_id,
            kind=StepKind.assumption_add,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("spin_species.declared", self.lang, spins=shown),
            operation_args={"site_spins": dict(normalized)},
            state_affecting=True,
            status=StepStatus.success,
            timestamp=_now(),
        )
        self.steps[step_id] = step
        self._advance_head(step)
        return step

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def spin_expectation(self, expr_id: str, power: int = 1) -> Expression:
        """``⟨A^n⟩`` at infinite temperature for a registered spin expression.

        Uses the declared per-site spins; with none declared it falls back to
        the uniform-spin polynomial in ``x = S(S+1)`` that predates F-13, so
        existing sessions are unaffected.

        ``power`` is what makes this useful: the FIRST moment of a coupling
        term vanishes (``⟨I·S⟩ = 0``, every factor traceless), and the physics
        — hyperfine second moments, the Van Vleck NMR line width — lives in
        ``⟨A²⟩``.
        """
        if power < 1:
            raise ValueError(t("spin_species.bad_power", self.lang, power=power))
        source = self.expressions.get(expr_id)
        if source is None:
            raise ValueError(t("spin_species.unknown_expr", self.lang, expr_id=expr_id))
        terms = spa.parse_spin_expression(source.latex or source.canonical)
        product = terms
        for _ in range(power - 1):
            product = spa.multiply(product, terms)
        value = spa.expectation_general(product, self.site_spin_map())

        return self.register_expression(
            value.to_canonical(),
            latex=f"{_bracket_latex(source, expr_id, power)} = {value.to_latex()}",
            skip_cas_bind=True,
        )

    def site_spin_map(self) -> "dict | None":
        """The declared site→spin map in ``spin_algebra`` form, or None.

        None means "nothing declared" — the uniform-spin path.  An entry with a
        blank value means "this site is a distinct species whose magnitude is
        still symbolic", which is NOT the same thing.
        """
        declared = self.symbol_context.site_spins
        if not declared:
            return None
        return {int(site): (Fraction(value) if value else None)
                for site, value in declared.items()}


def _bracket_latex(source, expr_id: str, power: int) -> str:
    """``⟨H_hf²⟩`` — by NAME when the source was registered as ``H = …``.

    The session is a research log; ``⟨E1²⟩`` makes the reader chase an id back
    up the timeline to learn what was averaged.
    """
    from notebook.runtime.lhs_formatting import _lhs_latex_from_expr
    latex = source.latex or ""
    # _lhs_latex_from_expr returns the WHOLE string when there is no "=", which
    # would put the entire Hamiltonian inside the bracket — so require the "=".
    name = (_lhs_latex_from_expr(latex).strip() if "=" in latex else "") or expr_id
    inner = name if power == 1 else f"{{{name}}}^{{{power}}}"
    return rf"\langle {inner} \rangle"


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
