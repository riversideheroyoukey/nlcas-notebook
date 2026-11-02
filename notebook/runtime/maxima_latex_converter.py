"""Maxima-specific LaTeX → canonical converter."""
from __future__ import annotations

import re

from notebook.runtime.latex_typography import upright_operator_spellings
from notebook.runtime.latex_converter import LatexConverter, TEX_GREEK_TO_NAME


class MaximaLatexConverter(LatexConverter):
    """Converts LaTeX to Maxima canonical form.

    Maxima-specific constants: ``%pi``, ``%e``, ``%i``, ``inf``.
    Maxima-specific functions: ``determinant``, ``mattrace``.
    """

    _TRIG_MAP: dict[str, str] = {
        r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
        r"\cot": "cot", r"\sec": "sec", r"\csc": "csc",
        # Maxima spells these `asin`/`acos`/`atan`; it does NOT know `arcsin`.
        # Measured 2026-08-18, it treated `arcsin` as an UNDEFINED function —
        # `arcsin(1)` stayed `arcsin(1)` instead of %pi/2, `diff` returned a noun
        # form, `integrate` did nothing, `sin(arcsin(x))` would not simplify to
        # x, and its `tex1` was `{\it arcsin}(x)`: italic prose in a published
        # paper.  Every inverse-trig calculation quietly did nothing.  `asin`
        # fixes all of it and DISPLAYS as `\arcsin x` — the researcher's own
        # notation — so the round trip closes.
        r"\arcsin": "asin", r"\arccos": "acos", r"\arctan": "atan",
        r"\sinh": "sinh", r"\cosh": "cosh", r"\tanh": "tanh",
        r"\log": "log", r"\ln": "log",
        r"\exp": "exp",
        r"\det": "determinant",
        # 立体で組まれた作用素名。**綴りは生成する**（`\text{tr}` /
        # `\mathrm{tr}` / `\operatorname{tr}` …）——ラッパの違いは組版の
        # 好みでしかない。`Tr` は物理の標準的な綴り（$\mathrm{Tr}\,\rho$）。
        **upright_operator_spellings({"tr": "mattrace", "Tr": "mattrace",
                                      "det": "determinant"}),
    }

    # Letters come from the CAS-neutral table; only Maxima's CONSTANT
    # spellings are added here (see latex_converter.TEX_GREEK_TO_NAME).
    _GREEK_MAP: dict[str, str] = {
        **TEX_GREEK_TO_NAME,
        r"\pi": "%pi", r"\infty": "inf",
        r"\i": "%i",
        r"\mathrm{i}": "%i",
        r"\text{i}": "%i",
    }

    _KNOWN_FUNCTIONS: frozenset[str] = frozenset(set(_TRIG_MAP.values()) | {"sqrt"})
    _FN_ALT: str = "|".join(sorted(_KNOWN_FUNCTIONS, key=len, reverse=True))

    def _greek_and_constants(self, s: str) -> str:
        s = super()._greek_and_constants(s)
        # Standalone 'e' before '^' → Maxima's %e (Euler's number)
        s = re.sub(r"(?<![A-Za-z%])e(?=\^)", "%e", s)
        return s


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def latex_to_canonical(latex: str, max_passes: int = 6) -> str:
    """Convert a LaTeX expression to Maxima-compatible canonical form.

    Kept for backward compatibility, and kept HERE rather than in the shared
    ``latex_converter`` base: it hardcodes one backend, so the shared module
    would have had to import this one (open_items AUDIT-SHARED-CAS-LEAK).
    New code should call ``backend_handler.latex_to_canonical(latex)``.
    """
    return MaximaLatexConverter().to_canonical(latex, max_passes)
