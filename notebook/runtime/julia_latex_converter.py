"""Julia-specific LaTeX → canonical converter."""
from __future__ import annotations

from notebook.runtime.latex_converter import LatexConverter, TEX_GREEK_TO_NAME


class JuliaLatexConverter(LatexConverter):
    """Converts LaTeX to Julia-compatible canonical form.

    Julia-specific constants: ``pi``, ``Inf``, ``im``.
    Julia-specific functions: ``det``, ``tr`` (from LinearAlgebra).
    Trig inverse functions use Julia names: ``asin``, ``acos``, ``atan``.
    """

    _TRIG_MAP: dict[str, str] = {
        r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
        r"\cot": "cot", r"\sec": "sec", r"\csc": "csc",
        r"\arcsin": "asin", r"\arccos": "acos", r"\arctan": "atan",
        r"\sinh": "sinh", r"\cosh": "cosh", r"\tanh": "tanh",
        r"\log": "log", r"\ln": "log",
        r"\exp": "exp",
        r"\det": "det", r"\text{tr}": "tr",
    }

    # Letters come from the CAS-neutral table; only Julia's CONSTANT
    # spellings are added here (see latex_converter.TEX_GREEK_TO_NAME).
    _GREEK_MAP: dict[str, str] = {
        **TEX_GREEK_TO_NAME,
        r"\pi": "pi", r"\infty": "Inf",
        r"\i": "im",
        r"\mathrm{i}": "im",
        r"\text{i}": "im",
    }

    _KNOWN_FUNCTIONS: frozenset[str] = frozenset(set(_TRIG_MAP.values()) | {"sqrt"})
    _FN_ALT: str = "|".join(sorted(_KNOWN_FUNCTIONS, key=len, reverse=True))

    def _implicit_multiplication(self, s: str) -> str:
        import re
        # Subscripted identifiers (e.g. xi_k) before ( are multiplication, not function calls.
        # This must run before the base rules so that xi_k*(... is stable.
        s = re.sub(r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9-]+)+)\s+\(", r"\1*(", s)
        # Any identifier not in _KNOWN_FUNCTIONS followed by space+( is multiplication.
        # Covers physics parameters: Gamma (Sx+Sy) → Gamma*(Sx+Sy).
        _known = self._KNOWN_FUNCTIONS

        def _maybe_mul(m: "re.Match[str]") -> str:
            word = m.group(1)
            if word in _known:
                return m.group(0)   # sin(x), cos(x) etc. → keep as function call
            return word + "*("

        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\s+\(", _maybe_mul, s)
        return super()._implicit_multiplication(s)
