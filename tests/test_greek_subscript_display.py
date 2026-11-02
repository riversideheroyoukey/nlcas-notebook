r"""GREEK-SUB-ON-GREEK-BASE (2026-09-11) — a Greek subscript is a name too.

`x_\sigma` displayed as `x_{\sigma}` and `\psi_\sigma` as `\psi_{sigma}` — the
italic word.  The answer depended on the BASE, because the same question ("is
this subscript a Greek name?") was answered in two places and the one that fires
first said no: `_BARE_GREEK_RE` reads "a Greek base with a glued subscript"
itself and used to emit the subscript RAW, so `_GREEK_SUBSCRIPT_RE` — which does
convert — never saw it.

This is the `\dot{theta}` shape (EXAMPLES-DRIFT): it typesets, as prose.
`\psi_\sigma` is the ordinary field-operator spelling.

No language axis (the display bridge has no session language); the real axis is
the BACKEND, so both are measured.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.backend_handler import beautify_bare_canonical_latex
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler

_BACKENDS = [MaximaBackendHandler, JuliaBackendHandler]


def _shown(cls, latex: str) -> str:
    h = cls(None)
    h.set_symbol_context(SymbolContext())
    return h.normalize_display_latex(h.latex_to_canonical(latex))


class TestAGreekSubscriptIsRestoredWhateverTheBase:
    @pytest.mark.parametrize("cls", _BACKENDS)
    @pytest.mark.parametrize("latex,expected", [
        (r"\psi_\sigma",       r"\psi_{\sigma}"),
        (r"\Sigma_\sigma",     r"\Sigma_{\sigma}"),
        (r"\phi_\mu",          r"\phi_{\mu}"),
        (r"\psi_\sigma(x)",    r"\psi_{\sigma}(x)"),
        # the base that always worked — the pass that handles it converts
        (r"x_\sigma",          r"x_{\sigma}"),
        (r"p_\theta",          r"p_{\theta}"),
        (r"\rho_{\alpha\beta}", r"\rho_{\alpha,\beta}"),
    ])
    def test_display(self, cls, latex, expected):
        assert _shown(cls, latex) == expected

    @pytest.mark.parametrize("cls", _BACKENDS)
    @pytest.mark.parametrize("latex,expected", [
        (r"\psi_x",   r"\psi_{x}"),     # a Latin subscript is not a Greek name
        (r"\phi_2",   r"\phi_{2}"),     # nor is a digit
    ])
    def test_a_subscript_that_is_not_greek_is_left_alone(self, cls, latex, expected):
        assert _shown(cls, latex) == expected

    @pytest.mark.parametrize("cls", _BACKENDS)
    def test_only_the_bare_name_converts(self, cls):
        """`mu2` is not the letter — the same carve-out `_GREEK_SUBSCRIPT_RE` has."""
        assert _shown(cls, r"\phi_mu2") == r"\phi_{mu2}"


class TestThePrimeStaysInsideTheIndex:
    r"""A prime marks a DISTINCT VARIABLE and is part of the index name."""

    @pytest.mark.parametrize("cls", _BACKENDS)
    def test_greek_base(self, cls):
        assert _shown(cls, r"\psi_{\sigma'}") == r"\psi_{\sigma'}"

    @pytest.mark.parametrize("cls", _BACKENDS)
    def test_latin_base(self, cls):
        assert _shown(cls, r"x_{\sigma'}") == r"x_{\sigma'}"


class TestBothDisplayEntrancesAgree:
    """`beautify_bare_canonical_latex` is the other entrance, and it reads the
    same two rules — so it must give the same answer."""

    @pytest.mark.parametrize("canonical,expected", [
        ("psi_sigma",  r"\psi_{\sigma}"),
        ("x_sigma",    r"x_{\sigma}"),
        ("psi_sigma'", r"\psi_{\sigma'}"),
    ])
    def test_same_answer(self, canonical, expected):
        assert beautify_bare_canonical_latex(canonical) == expected
