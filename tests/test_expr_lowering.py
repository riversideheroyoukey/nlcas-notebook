"""User form → backend code: the single derivation point (docs/32 Phase 0).

The bug this boundary exists to prevent (open_items AUDIT-SUBST-INDEXED) is a
SILENT one: an expression body reached Maxima as the raw user string while the
operation's argument was lowered through the IR, so ``subst(V(i)=5, a*V_i+b)``
matched nothing and returned success.  These tests pin the properties that make
that impossible: one derivation point, idempotent, context-dependent (so the
result must never be cached), and never a silent pass-through on failure.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.quantity import Quantity
from notebook.runtime.expr_lowering import Lowered, to_backend_code
from notebook.runtime.shared_parser import UnknownNotationError


@pytest.fixture(scope="module")
def handler():
    with NotebookEngine(title="lowering") as e:
        yield e._handler


class TestMaximaLowering:

    @pytest.mark.parametrize("text,expected", [
        ("a*V_i + b", "a * V(i) + b"),   # the AUDIT-SUBST-INDEXED case
        ("x_1", "x(1)"),
        ("A_{i,j}", "A(i, j)"),   # indices need a separator (AUDIT-SUBSCRIPT-SPLIT)
    ])
    def test_indexed_symbols_are_lowered(self, handler, text, expected):
        r = handler.to_backend_code(text)
        assert r.status == "lowered"
        assert r.code == expected

    @pytest.mark.parametrize("text", [
        "'diff(q(t),t,2)",              # derivative noun form
        "'diff(q,t,2)",                 # depends form
        "q(t)",
        "block([q], ratsimp(q))",
        "abs(x)",
    ])
    def test_cas_own_forms_survive_verbatim(self, handler, text):
        """A CAS result may be re-lowered on its way back into a command; the
        forms Maxima itself emits must come through untouched."""
        assert handler.to_backend_code(text).code == text

    @pytest.mark.parametrize("text", [
        "a*V_i + b", "x_1", "'diff(q(t),t,2)", "sin(theta)^2+cos(theta)^2",
        "matrix([a,b],[c,d])", "%e^(-a*t)", "H = J*x*y", "integrate(f(x),x,0,1)",
    ])
    def test_idempotent(self, handler, text):
        """R1.  Lowering an already-lowered string must be a no-op, otherwise
        the same expression would drift each time it passes through a command."""
        once = handler.to_backend_code(text).code
        assert handler.to_backend_code(once).code == once

    def test_empty_is_passthrough(self, handler):
        assert handler.to_backend_code("").status == "passthrough"
        assert handler.to_backend_code("   ").status == "passthrough"


class TestContextDependence:
    """R0.  The correct lowering is a function of what is declared RIGHT NOW,
    so the result must be re-derived per call and never stored on Expression.
    """

    def test_declaration_changes_the_lowering(self):
        with NotebookEngine(title="ctx") as e:
            h = e._handler
            assert h.to_backend_code("k_B*T").code == "k(B) * T"
            e.symbol_context.register("k_B", Quantity(name="k_B", kind="scalar"))
            assert h.to_backend_code("k_B*T").code == "k_B * T"

    def test_partial_declaration_is_per_symbol(self):
        with NotebookEngine(title="ctx2") as e:
            h = e._handler
            e.symbol_context.register("H_0", Quantity(name="H_0", kind="scalar"))
            # H_0 is a name; H_1 is still an undeclared indexed symbol.
            assert h.to_backend_code("H_0 + H_1").code == "H_0 + H(1)"


class _StubHandler:
    """Minimal stand-in exercising the contract of the two hooks."""

    def __init__(self, *, opaque=False, raises=None, ir="IR", code="CODE"):
        self._opaque, self._raises, self._ir, self._code = opaque, raises, ir, code

    def is_opaque_handle(self, text):
        return self._opaque

    def parse_ir(self, text):
        if self._raises is not None:
            raise self._raises
        return self._ir

    def generate_backend_code(self, ir):
        return self._code


class TestContract:

    def test_opaque_handle_is_never_touched(self):
        """A backend handle (Julia's ``_nb_E3``) names a value living in the CAS
        session; parsing it as maths would destroy the reference."""
        r = to_backend_code(_StubHandler(opaque=True), "_nb_E3")
        assert r == Lowered("_nb_E3", "passthrough", "backend handle")

    def test_unparseable_is_passthrough(self):
        r = to_backend_code(_StubHandler(ir=None), "whatever")
        assert r.status == "passthrough"
        assert r.code == "whatever"

    def test_backend_without_generator_is_passthrough(self):
        r = to_backend_code(_StubHandler(code=None), "whatever")
        assert r.status == "passthrough"

    def test_failure_is_refused_not_silent(self):
        """R4.  A parse/resolve failure must NOT hand the raw string to the CAS —
        that is exactly how a wrong answer gets returned as a success."""
        r = to_backend_code(_StubHandler(raises=RuntimeError("boom")), "x")
        assert r.status == "refused"
        assert not r.ok
        assert "RuntimeError" in r.reason and "boom" in r.reason

    def test_unknown_notation_still_propagates(self):
        """The engine boundary owns the translation of this one (it knows the
        session language), so it must not be swallowed into a reason string."""
        exc = UnknownNotationError("parser.unsupported_notation.chained_relation")
        with pytest.raises(UnknownNotationError):
            to_backend_code(_StubHandler(raises=exc), "a = b = c")


class TestJuliaHandle:

    def test_nb_reference_is_opaque(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        assert h.is_opaque_handle("_nb_E3")
        assert h.is_opaque_handle("  _nb_E12  ")
        assert not h.is_opaque_handle("a*x+b")

    def test_maxima_canonicals_are_not_opaque(self, handler):
        assert not handler.is_opaque_handle("a*V_i + b")
