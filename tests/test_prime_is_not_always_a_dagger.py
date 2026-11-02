r"""PRIME-DISPLAY-DAGGER (2026-09-10) — a prime is an adjoint only on an OPERATOR.

In a Julia session every prime became a dagger on the way to the screen
(`_PRIME_ADJOINT_RE` was ``(\w)'``, substituted unconditionally), so the most
ordinary spellings in the Green's-function family were displayed as something
the researcher never wrote::

    G(t, t')                     →  G(t, t^{\dagger})
    f(x, x')                     →  f(x, x^{\dagger})
    \sigma'                      →  \sigma^{\dagger}
    \psi^\dagger_{\sigma'}(x')   →  \psi^{\dagger}_{\sigma^{\dagger}}(x^{\dagger})

**The declaration is the discriminator**, and the published Julia sessions say
so: every primed operator base there (`c`, `g`, `a`, `b`) carries
``kind == "operator"``, and `sigma` — whose prime marks a DISTINCT VARIABLE — is
declared nowhere.

There is no language axis here: the display bridge has no session language (the
i18n rule's own reason for `bind_language_source`).  The axis that IS real is the
BACKEND, so both are measured — Maxima must be untouched, because there a prime
is never QuantumAlgebra's adjoint.
"""
import pytest

from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.julia_handler_common import quantum_operator_names


def _handler(cls, *operators: str):
    h = cls(None)
    ctx = SymbolContext()
    for name in operators:
        ctx.register(name, quantity=Quantity(name=name, kind="operator"))
    h.set_symbol_context(ctx)
    return h


def _shown(cls, latex: str, *operators: str) -> str:
    """What the researcher sees after typing *latex* — through both doors."""
    h = _handler(cls, *operators)
    return h.normalize_display_latex(h.latex_to_canonical(latex))


class TestAPrimeOnADeclaredOperatorIsTheAdjoint:
    @pytest.mark.parametrize("latex,operator,expected", [
        (r"a'(k)",                  "a",       r"a^{\dagger}(k)"),
        (r"a^\dagger(k)",           "a",       r"a^{\dagger}(k)"),
        (r"c^\dagger_k",            "c",       r"c^{\dagger}_k"),
        # A DECORATED operator — the ordinary field operator.  Before this fix
        # the adjoint was decided AFTER `psi_hat'` had become `\hat{\psi}'`,
        # where the prime no longer follows an identifier, so it stayed a prime.
        (r"\hat\psi^\dagger(x)",    "psi_hat", r"\hat{\psi}^{\dagger}(x)"),
    ])
    def test_julia_shows_a_dagger(self, latex, operator, expected):
        assert _shown(JuliaBackendHandler, latex, operator) == expected

    def test_the_adjoint_survives_a_product(self):
        out = _shown(JuliaBackendHandler, "a'(k) a(k)", "a")
        assert r"a^{\dagger}(k)" in out and "a(k)" in out


class TestAPrimeOnAnythingElseIsTheResearchersOwnMark:
    @pytest.mark.parametrize("latex,expected", [
        (r"G(t, t')",                    r"G(t, t')"),
        (r"f(x, x')",                    r"f(x, x')"),
        (r"v(\mathbf r - \mathbf r')",   r"v(\mathbf{r} - \mathbf{r}')"),
        (r"\sigma'",                     r"\sigma'"),
        (r"\delta_{\sigma\sigma'}",      r"\delta_{\sigma,\sigma'}"),
    ])
    def test_julia_keeps_the_prime(self, latex, expected):
        assert _shown(JuliaBackendHandler, latex) == expected

    def test_an_undeclared_name_keeps_its_prime(self):
        """Nothing is declared, so nothing is an adjoint — the safe direction."""
        assert _shown(JuliaBackendHandler, "a'(k)") == r"a'(k)"

    @pytest.mark.parametrize("latex,expected", [
        (r"G(t, t')",                    r"G(t, t')"),
        (r"v(\mathbf r - \mathbf r')",   r"v(\mathbf{r} - \mathbf{r}')"),
        (r"\sigma'",                     r"\sigma'"),
    ])
    def test_maxima_is_untouched(self, latex, expected):
        assert _shown(MaximaBackendHandler, latex) == expected


class TestThePrimeBelongsToTheIndexItSitsOn:
    r"""`_{\sigma}'` is the prime detached from the index it names."""

    @pytest.mark.parametrize("cls,expected", [
        (JuliaBackendHandler,  r"\psi^{\dagger}_{\sigma'}(x')"),
        (MaximaBackendHandler, r"\psi^\dagger_{\sigma'}(x')"),
    ])
    def test_the_field_operator_line(self, cls, expected):
        ops = ("psi",) if cls is JuliaBackendHandler else ()
        assert _shown(cls, r"\psi^\dagger_{\sigma'}(x')", *ops) == expected

    def test_the_decorated_field_operator_line(self):
        assert _shown(JuliaBackendHandler,
                      r"\hat\psi^\dagger_{\sigma'}(\mathbf r')", "psi_hat") == (
            r"\hat{\psi}^{\dagger}_{\sigma'}(\mathbf{r}')")


class TestVecIsNeverShown:
    r"""`vec(...)` is this system's wrapper, not the researcher's notation.

    `beautify_bare_canonical_latex` restored it and called itself "the shared
    entrance"; `normalize_display_latex` — the entrance `register_expression`
    and the operation route both use — did not ask (Core Principle #1).
    """

    @pytest.mark.parametrize("cls", [JuliaBackendHandler, MaximaBackendHandler])
    def test_a_bare_vector(self, cls):
        assert _shown(cls, r"\mathbf k") == r"\mathbf{k}"


class TestTheOperatorNamesAreAskedInOnePlace:
    def test_a_declaration_answers(self):
        ctx = SymbolContext()
        ctx.register("c", quantity=Quantity(name="c", kind="operator"))
        ctx.register("xi", quantity=Quantity(name="xi", kind="scalar"))
        names = quantum_operator_names(ctx)
        assert {"c", "c'"} <= names
        assert "xi" not in names and "xi'" not in names

    def test_no_context_names_nothing(self):
        assert quantum_operator_names(None) == set()

    def test_an_ir_adds_an_undeclared_primed_base(self):
        """A Bogoliubov `beta'(-k)` introduced by a substitution is an operator
        even though the session never declared it."""
        from notebook.models.ir import FunctionCall, Symbol
        ir = FunctionCall(func=Symbol(name="beta'"), args=[Symbol(name="k")])
        names = quantum_operator_names(SymbolContext(), ir)
        assert {"beta", "beta'"} <= names
