"""R-6 (display bridge): input → display LaTeX → re-input must preserve meaning.

The display bridge shows users natural notation (x_i, δ_ij, a†(k), \\hat{H})
for backend canonicals (x_i, delta_ij, a'(k), H_hat).  If a displayed form
does not convert BACK to the same canonical, copying a shown result into the
input box silently changes its meaning.  These tests pin the round trip for
the representative notation set from the audit (docs/archive/audit/, §C table).
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.backend_factory import make_backend
from notebook.runtime.latex_converter import restore_decorations


# ---------------------------------------------------------------------------
# Maxima: canonical → display LaTeX → canonical
# ---------------------------------------------------------------------------

class TestMaximaDisplayRoundtrip:

    @pytest.fixture(scope="class")
    def engine(self):
        with NotebookEngine(title="rt display") as e:
            yield e

    @pytest.mark.parametrize("canonical", [
        "x_i",                # indexed symbol
        "A_ij",               # two-index symbol
        "delta_ij*f_j",       # Kronecker delta in context
        "omega_0^2*q",        # greek + numeric subscript + power
        "a*x+b",              # plain scalars
        "sin(x)*cos(y)",      # function calls
    ])
    def test_canonical_display_canonical(self, engine, canonical):
        display = engine.canonical_display_latex(canonical)
        back = engine._handler.latex_to_canonical(display)
        assert back == canonical

    def test_decoration_roundtrip(self, engine):
        """\\hat{H} → H_hat → \\hat{H} → H_hat (restore_decorations path)."""
        c1 = engine._handler.latex_to_canonical(r"\hat{H}")
        assert c1 == "H_hat"
        display = restore_decorations(c1)
        assert display == r"\hat{H}"
        c2 = engine._handler.latex_to_canonical(display)
        assert c2 == c1


# ---------------------------------------------------------------------------
# Julia: creation-operator notation (both accepted input styles)
# ---------------------------------------------------------------------------

class TestJuliaDisplayRoundtrip:

    @pytest.fixture(scope="class")
    def handler(self):
        _, _, h = make_backend("julia_full")
        # `a` is DECLARED, because a prime is an adjoint only on an operator
        # this session declared one (PRIME-DISPLAY-DAGGER, 2026-09-10).  The
        # empty context this used to carry was convenience, not fidelity —
        # measured over the published Julia sessions, every primed operator
        # base there (`c`, `g`, `a`, `b`) carries `kind == "operator"`, and
        # QuantumAlgebra's own `a`/`f` must be declared before algebraic use.
        ctx = SymbolContext()
        ctx.register("a", quantity=Quantity(name="a", kind="operator"))
        h.set_symbol_context(ctx)
        return h

    def test_dagger_and_prime_normalize_to_same_canonical(self, handler):
        """a^\\dagger(k) and a'(k) are both accepted and mean the same."""
        assert handler.latex_to_canonical(r"a^\dagger(k)") == "a'(k)"
        assert handler.latex_to_canonical(r"a'(k)") == "a'(k)"

    def test_creation_operator_roundtrip(self, handler):
        """canonical a'(k) displays as a^{\\dagger}(k) and converts back."""
        c1 = handler.latex_to_canonical(r"a'(k)")
        display = handler.normalize_display_latex(r"a'(k)")
        assert r"\dagger" in display
        c2 = handler.latex_to_canonical(display)
        assert c2 == c1

    def test_bilinear_roundtrip(self, handler):
        """a†_k a_k (subscript style) survives display → re-input."""
        c1 = handler.latex_to_canonical(r"a^\dagger_k a_k")
        display = handler.normalize_display_latex(r"a^\dagger_k a_k")
        c2 = handler.latex_to_canonical(display)
        assert c2 == c1
