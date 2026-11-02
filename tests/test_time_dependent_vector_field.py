r"""VEC-APPLIED (2026-09-11) — `\vec A(t)` was the vector TIMES the time.

`_vector` wrapped the NAME and left the argument list standing beside it, so the
whole of electromagnetism was unwritable and answered as a success::

    \vec{A}(t)                             →  vec(A) * t
    \nabla \times \vec{A}(t)               →  curl(vec(A) * t)      ← B = ∇×A
    \frac{\partial \vec{A}(t)}{\partial t} →  'diff(vec(A)*t,t,1) = A  ← E = −∂A/∂t

**A vector that is APPLIED is still one quantity**, so the wrapper goes around
the application.  Only the GLUED form is an application — the parser's own rule
for `f(x)` vs `f (x)`, where a space means a product.

Two more of the same family were measured while fixing it:

* the components of a `vec(...)` EXPRESSION were built by appending `_x` to the
  whole wrapper (`vec(A)_x`), which reached the researcher as a raw
  `Maxima evaluation error`;
* `\nabla \times \vec{A}` — the spelling every textbook uses — did not reach the
  operation at all (the field pattern accepted a bare identifier only), while
  `\nabla \times A` did.

And the applied field is REFUSED in vector calculus rather than answered: read
as `A_x(t)`, the coordinate derivatives in ∇× and ∇· are zero (measured against
a live Maxima), and inventing the coordinate dependence the researcher did not
write is the guess this system declines.  The refusal names the road that works
— `\nabla \times A`, which the instruction corpus itself uses (8 rows).
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.llm_supervisor import _try_single_vec_op
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler

_BACKENDS = [MaximaBackendHandler, JuliaBackendHandler]


def _canonical(cls, latex: str) -> str:
    h = cls(None)
    h.set_symbol_context(SymbolContext())
    return h.latex_to_canonical(latex)


class TestAnAppliedVectorIsOneQuantity:
    @pytest.mark.parametrize("cls", _BACKENDS)
    @pytest.mark.parametrize("latex,canonical", [
        (r"\vec{A}(t)",              "vec(A(t))"),
        (r"\vec A(t)",               "vec(A(t))"),
        (r"\mathbf{A}(t)",           "vec(A(t))"),
        (r"\mathbf A(t)",            "vec(A(t))"),
        (r"\mathbf A(\mathbf r, t)", "vec(A(vec(r), t))"),
    ])
    def test_the_wrapper_goes_around_the_application(self, cls, latex, canonical):
        assert _canonical(cls, latex) == canonical

    @pytest.mark.parametrize("cls", _BACKENDS)
    @pytest.mark.parametrize("latex,canonical", [
        (r"\vec{A}",     "vec(A)"),      # nothing applied — unchanged
        (r"\vec A",      "vec(A)"),
        (r"\vec{A} (t)", "vec(A)*(t)"),  # a SPACE is a product (the `f (x)` rule)
    ])
    def test_the_forms_that_were_already_right(self, cls, latex, canonical):
        assert _canonical(cls, latex) == canonical

    @pytest.mark.parametrize("latex,canonical", [
        (r"\nabla \times \vec{A}(t)",             "nabla ~ vec(A(t))"),
        (r"\frac{\partial \vec{A}(t)}{\partial t}", "'diff(vec(A(t)),t,1)"),
        (r"\vec{A}(t) \cdot \vec{B}(t)",          "vec(A(t)) . vec(B(t))"),
    ])
    def test_the_electromagnetic_lines(self, latex, canonical):
        assert _canonical(MaximaBackendHandler, latex) == canonical


class TestTheVectorMarkIsDecorationOnTheName:
    r"""`\nabla \times \vec{A}` is the spelling every textbook uses, and it did
    not reach the operation while `\nabla \times A` did."""

    @pytest.mark.parametrize("text,operation,field", [
        (r"\nabla \times A",          "curl",       "A"),
        (r"\nabla \times \vec{A}",    "curl",       "A"),
        (r"\nabla \times \vec A",     "curl",       "A"),
        (r"\nabla \times \mathbf{A}", "curl",       "A"),
        (r"\nabla \cdot \vec{A}",     "divergence", "A"),
        (r"curl \vec{A}",             "curl",       "A"),
        (r"div \mathbf{B}",           "divergence", "B"),
        (r"\nabla \vec{f}",           "gradient",   "f"),
        (r"\nabla f",                 "gradient",   "f"),
    ])
    def test_the_field_is_read(self, text, operation, field):
        result = _try_single_vec_op(text)
        assert result is not None, text
        assert result.operation == operation
        assert result.args["expr"] == field


class TestAnAppliedFieldIsRefusedInVectorCalculus:
    r"""Read as `A_x(t)`, ∇× and ∇· are ZERO — measured against a live Maxima.
    The coordinate dependence the researcher did not write is not invented."""

    def test_the_components_of_a_wrapped_field_unwrap(self):
        h = MaximaBackendHandler(None)
        h.set_symbol_context(SymbolContext())
        assert h._build_vec_components("vec(A)", "cartesian", {}) == \
            ("A_x", "A_y", "A_z")

    def test_an_applied_field_is_refused_by_name(self):
        h = MaximaBackendHandler(None)
        h.set_symbol_context(SymbolContext())
        with pytest.raises(TypeError) as exc:
            h._build_vec_components("vec(A(t))", "cartesian", {})
        assert "A" in str(exc.value)

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_refusal_exists_in_both_languages(self, lang):
        import json
        msgs = json.load(open("i18n/messages.json", encoding="utf-8"))
        assert msgs["vector.applied_field_components"][lang]

    def test_the_refusal_names_a_road_that_works(self):
        r"""`\nabla \times A` is what the instruction corpus itself types."""
        import json
        msgs = json.load(open("i18n/messages.json", encoding="utf-8"))
        for lang in ("ja", "en"):
            assert r"\nabla \times {name}" in msgs["vector.applied_field_components"][lang]
        assert _try_single_vec_op(r"\nabla \times A").operation == "curl"
