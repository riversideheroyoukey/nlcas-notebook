"""NotebookEngine mixin: F-8 definition-based transforms (signal_processing /
control_theory domains): Fourier transform, inverse Fourier transform, and
Z-transform.

Mirrors EngineLaplaceMixin.laplace_by_definition's EXPLICIT-FUNCTION branch
only (evaluate the definition integral/sum directly) — deliberately narrower
than the full Laplace mechanism, which also handles derivatives of an
unknown time-dependent function (by-parts, IC free parameters) and whole
equations (by linearity). Signal_processing's own domain hint states its
primary use case as an explicit signal's transform
(``X(omega) = integral x(t) e^{-i omega t} dt``); extend to the
derivative/linearity sub-cases only if a future scenario needs them (same
"add only the missing primitive" discipline as F-14/F-16/F-17).

Extracted as its own mixin (not appended to engine_laplace_mixin.py or
engine.py) following the EngineLaplaceMixin/EngineGameMixin pattern: state is
reached via self; NotebookEngine appears only in annotations (PEP 563), so
there is no import from engine.py and no cycle.
"""
from __future__ import annotations

from notebook.runtime.shared_parser import strip_lhs
from notebook.models.expression import (  # noqa: F401
    Expression,
)


class EngineFourierMixin:
    """F-8: definition-based transforms (see module docstring)."""

    def _transform_source_body(self, target: str) -> str:
        """Resolve *target* (an E-id or a raw expression string) to its bare
        body.  UNLIKE EngineLaplaceMixin.laplace_by_definition's target
        resolution, this does NOT fall back to reconstructing an equation
        from the expression's LATEX when its canonical lacks "=" — that
        fallback exists in the Laplace mixin specifically for H-1-registered
        ODEs (whose canonical stores only the RHS, with the real "LHS = RHS"
        living in latex). A transform's own OWN prior result (e.g. feeding a
        fourier_transform_by_definition output into
        inverse_fourier_by_definition) is a "definition card" whose LATEX is
        a multi-part DISPLAY string (containing "=" signs for presentation:
        "F{f}(omega) = integral... = value") — reconstructing an equation
        from THAT latex misparses it into nonsense (confirmed live: it
        produced a garbled `%int_...=...` string that then hung Maxima's
        persistent adapter on the resulting malformed integrate() call). For
        an explicit-function transform target, the canonical is always
        already correct as-is."""
        if target in self.expressions:
            cano = self.expressions[target].canonical or ""
        else:
            cano = self._handler.latex_to_canonical(target) or target
        return strip_lhs(cano).strip()

    def z_transform_by_definition(self, target: str, n: str = "n",
                                  z: str = "z") -> Expression:
        """Z-transform of an EXPLICIT discrete signal straight from the
        definition sum ``X(z) = sum_{n=0}^{inf} x[n] z^{-n}``, via Maxima's
        nusum. If nusum cannot find a closed form the result is registered
        AS THE UNEVALUATED SUM, not silently mistranslated (matches the
        numeric layer's whitelist philosophy: an unsupported case is a
        visible non-result, never a guess)."""
        x = self._transform_source_body(target)
        x_disp = self.canonical_display_latex(x)
        cmd = self._handler.z_transform_definition_command(x, n, z)
        val, val_latex = self._cas_value_and_latex(cmd)
        defn = (rf"\mathcal{{Z}}\{{{x_disp}\}}({z}) = "
                rf"\sum_{{{n}=0}}^{{\infty}} {x_disp}\, {z}^{{-{n}}} = {val_latex}")
        return self._register_plain(defn, val)
