r"""Opening a vector into components — the answer to `\int \frac{d^3k}{(2\pi)^3}`.

A Fourier transform is written with a volume measure and a dot product, and
neither can be handed to a one-variable CAS as it stands.  Before 2026-08-08 the
whole thing came back as a PRODUCT — `(d³·int·k·f(k))/(8π³)`, returned as a
success — and the first repair only stopped that, by refusing.  Refusing is not
an answer to notation this ordinary, so the dimension is opened explicitly
instead: `\int d^3k → \int dk_x dk_y dk_z`, the researcher's own suggestion.

**Opening the measure ALONE would have been the same bug wearing a new face.**
Measured: with the integrand untouched, Maxima reads `f(k)` as a constant in
`k_x` and returns `k_x k_y k_z · f(k)`.  So the measure, the dot product and the
function OF the vector are opened together, and a vector still standing
afterwards is refused by name.

Asserted on VALUES wherever a value exists — `∫d³k e^{-k·k} = π^{3/2}` is the
kind of check a shape assertion cannot fake.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.vector_components import (
    UnopenedVector, component_names, expand,
)
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
)


def _run_nl(engine, cmd):
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=[], primary_backend="maxima")
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[InputToken(type="text", value=cmd)]), engine, sup))


class _Session:
    """A Maxima session with `k` and `x` declared as vectors."""

    def setup_method(self):
        nb.start_engine(title="components", mode="standard",
                        language=getattr(self, "LANG", "ja"), domains=[])
        self.eng = nb.get_engine()
        self.eng.run_operation("define_vector", {"name": "k"})
        self.eng.run_operation("define_vector", {"name": "x"})

    def teardown_method(self):
        nb.stop_engine()

    def _register(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        return self.eng.register_expression(canonical, latex=latex)

    def _opened(self, latex):
        src = self._register(latex)
        expr, names = self.eng.expand_vector_components(src.display_id)
        return expr.canonical.replace(" ", ""), names

    def _evaluated(self, latex):
        """Open, THEN ask for the value — two requests, because they are two.

        The expansion HOLDS (`'integrate`): over an arbitrary basis the
        coefficient is a definition and there is nothing to carry out, and
        asking Maxima to try anyway does not merely fail to simplify — measured,
        it comes back `ok=False` with an empty answer.
        """
        src = self._register(latex)
        opened, names = self.eng.expand_vector_components(src.display_id)
        assert "'integrate(" in opened.canonical, opened.canonical
        value = self.eng.evaluate_held_integrals(opened.display_id)
        return value.canonical.replace(" ", ""), names


class TestTheValueIsRight(_Session):
    """Known integrals, computed the whole way through."""

    @pytest.mark.parametrize("text,value", [
        # ∫d³k e^{-k·k} = (√π)³
        (r"\int_{-\infty}^{\infty} d^3k e^{-(k \cdot k)}", "%pi^(3/2)"),
        # …divided by (2π)³
        (r"\int_{-\infty}^{\infty} \frac{d^3k}{(2\pi)^3} e^{-(k \cdot k)}",
         "1/(8*%pi^(3/2))"),
        # the DIMENSION is the one the researcher wrote, not the one the
        # coordinate system happens to have
        (r"\int_{-\infty}^{\infty} d^2k e^{-(k \cdot k)}", "%pi"),
        # …and with NO range written it is still all space: a volume measure is
        # not an antiderivative.  Read as one it gave
        # `π^{3/2} erf(k_x) erf(k_y) erf(k_z)/8` — the integration variables
        # still in the answer, close enough to skim past.
        (r"\int d^3k e^{-(k \cdot k)}", "%pi^(3/2)"),
    ])
    def test_it_computes(self, text, value):
        assert self._evaluated(text)[0] == value


class TestWhatGetsOpened(_Session):
    def test_the_measure_becomes_one_integral_per_component(self):
        canonical, names = self._opened(r"\int d^3k f(k)")
        assert names == ["k_x", "k_y", "k_z"]
        assert canonical.count("integrate(") == 3

    def test_a_function_of_the_vector_is_opened_too(self):
        r"""Without this Maxima reads `f(k)` as a constant in `k_x` and returns
        `k_x k_y k_z · f(k)` — the unopened bug, one layer down."""
        canonical, _ = self._opened(r"\int d^3k f(k)")
        assert "f(k_x,k_y,k_z)" in canonical, canonical

    def test_a_dot_product_becomes_a_sum_of_products(self):
        canonical, names = self._opened(r"\int d^3k e^{i (k \cdot x)}")
        assert set(names) == {"k_x", "k_y", "k_z", "x_x", "x_y", "x_z"}
        for pair in ("k_x*x_x", "k_y*x_y", "k_z*x_z"):
            assert pair in canonical, canonical

    def test_the_components_are_registered_as_scalars(self):
        """So a later substitution or derivative sees quantities, not unknowns."""
        self._opened(r"\int d^3k f(k)")
        for name in ("k_x", "k_y", "k_z"):
            quantity = self.eng.symbol_context.get_quantity(name)
            assert quantity is not None and quantity.math_type.algebra == "scalar"

    def test_a_component_can_then_be_an_INTEGRATION_VARIABLE(self):
        r"""The half `test_the_components_are_registered_as_scalars` does not
        reach, and the one where the two mechanisms could contradict each other.

        `\int d^3k` is refused (`multidim_measure`) and answered by opening the
        vector into components — that is this list's worked example.  But the
        components it produces are SUBSCRIPTED names (`k_x`; `k_1, k_2, …` once
        the dimension outruns the coordinate suffixes), and a subscripted name
        used as an integration variable is exactly what `integral.var_is_indexed`
        REFUSES (open_items INTEGRAL-VAR-INDEXED).  So the worked example's own
        output would be un-integrable — "what is displayed must be what can be
        computed with", broken across two mechanisms neither of which is wrong
        on its own.

        What reconciles them is that `expand_vector_components` REGISTERS the
        names: declared, `k_x` lowers to `k_x`, and the guard (which fires on a
        `(` in the lowered form) cannot trigger.  Measured 2026-08-17:
        after expanding, `k_x^2` integrated over `k_x` from 0 to 1 returns 1/3;
        in a fresh session the same request is refused, correctly, with the fix.

        That link is load-bearing and was untested — the registration test above
        names "a later substitution or derivative" as its consumer, not this one.
        """
        self._opened(r"\int d^3k f(k)")
        lowered = self.eng._handler.to_backend_code("k_x").code
        assert "(" not in lowered, (
            f"a component of the worked example lowers to {lowered!r}, which "
            "integral.var_is_indexed refuses — the expansion's own output "
            "would not be integrable")

    def test_the_guard_still_fires_for_a_name_nothing_has_declared(self):
        """The other direction: the reconciler is the DECLARATION, not a
        weakening of the guard.  A subscripted name with nothing behind it must
        still read as an index, or the refusal has no cases left.

        (`k_x` cannot be used here — this fixture declares `k` a VECTOR, which
        is already enough; see the three states in the test below.)
        """
        lowered = self.eng._handler.to_backend_code("w_x").code
        assert "(" in lowered, (
            "an undeclared subscripted name no longer reads as an index; "
            "integral.var_is_indexed can never fire")

    def test_the_guard_matches_what_maxima_actually_accepts(self):
        r"""There are THREE states, and the guard's test is the CAS's boundary.

        Measured 2026-08-17 against a live Maxima — the discriminator
        `"(" in lowered` is not a heuristic, it is exactly the line:

            nothing declared      k_x → k(x)   integrate(k(x)^2, k(x), 0, 1)  FAILS
            `k` declared a vector k_x → k[x]   integrate(k[x]^2, k[x], 0, 1)  = 1/3
            after the expansion   k_x → k_x    integrate(k_x^2, k_x, 0, 1)    = 1/3

        So the middle state — which the worked example passes through before it
        is expanded — is NOT a hole: Maxima takes a subscripted array reference
        as an integration variable, and only the function-call form is rejected.
        """
        assert "(" in self.eng._handler.to_backend_code("w_x").code      # none
        assert "(" not in self.eng._handler.to_backend_code("k_x").code  # vector
        self._opened(r"\int d^3k f(k)")
        assert self.eng._handler.to_backend_code("k_x").code == "k_x"    # opened

    def test_both_ways_out_named_by_the_refusal_actually_work(self):
        r"""`integral.var_is_indexed` names TWO declarations; both must work.

        Until 2026-08-17 it named only one — "declare the whole name" — and its
        stated reason was wrong: "the CAS cannot integrate an indexed quantity"
        is false, since Maxima takes `integrate(k[x]^2, k[x], 0, 1)` and answers
        1/3.  What it rejects is the FUNCTION-CALL form `k(x)`.

        The omission mattered for exactly the researcher this list's worked
        example produces: someone holding a vector `k` was told to declare `k_x`
        as an unrelated scalar, when declaring `k` a vector already suffices and
        keeps the components related.

        A refusal that names a fix nobody has measured is how these drift, so
        both are measured here.
        """
        assert "(" not in self.eng._handler.to_backend_code("k_x").code   # (1)
        self.eng.run_operation("define_scalar_parameter", {"name": "w_x"})
        assert "(" not in self.eng._handler.to_backend_code("w_x").code   # (2)

    def test_the_refusal_names_the_base_not_just_the_variable(self):
        from notebook.runtime.i18n import t
        for lang in ("ja", "en"):
            msg = t("integral.var_is_indexed", lang, var="k_x", base="k",
                    lowered="k(x)")
            assert msg != "integral.var_is_indexed"
            assert "k_x" in msg and "k(x)" in msg
            # the base must be offered as its own way out, not only the name
            assert msg.count("k") >= 3, msg

    def test_a_vector_left_standing_is_refused_by_name(self):
        r"""`\int d^3k k` has no componentwise reading, and letting it through
        would put an unopened vector under a component integral — silently."""
        src = self._register(r"\int d^3k k")
        with pytest.raises(UnopenedVector) as exc:
            self.eng.expand_vector_components(src.display_id)
        assert exc.value.name == "k"


class TestComponentNames:
    """Names follow the convention `define_vector_field` already uses."""

    def test_cartesian_suffixes(self):
        assert component_names("k", 3, ("x", "y", "z")) == ["k_x", "k_y", "k_z"]

    def test_fewer_than_the_coordinate_system_offers(self):
        assert component_names("k", 2, ("x", "y", "z")) == ["k_x", "k_y"]

    def test_more_than_it_offers_falls_back_to_numbers(self):
        """A dimension the researcher wrote is never silently reduced to the
        three a coordinate system happens to name."""
        assert component_names("k", 4, ("x", "y", "z")) == [
            "k_1", "k_2", "k_3", "k_4"]


class TestTheDimensionSurvivesTheWalks(_Session):
    def test_measure_dim_reaches_the_expander(self):
        r"""Every walker that rebuilt an `Integral` used to drop a field it did
        not know about, so `\int d^3k f(k)` arrived as the ONE-variable
        `integrate(f(k), k)` — silently wrong in a new way."""
        src = self._register(r"\int d^3k f(k)")
        assert src.ir_ast.measure_dim == 3


class TestTheInstruction(_Session):
    LANG = "ja"

    def _fourier(self):
        self._register(r"\int \frac{d^3k}{(2\pi)^3} e^{i (k \cdot x)} f(k)")

    def test_japanese(self):
        self._fourier()
        result = _run_nl(self.eng, "成分に展開して")
        assert result["status"] == "completed", result
        assert result["operation"] == "expand_components"
        assert "k_x" in result["message"]

    def test_an_explicit_expression_id(self):
        self._fourier()
        result = _run_nl(self.eng, "E1 を成分に展開して")
        assert result["status"] == "completed", result

    def test_it_reaches_for_the_expression_that_has_vectors(self):
        """Not simply the latest expression: an unrelated registration after the
        integral must not become the target."""
        self._fourier()
        self._register(r"y = t + 1")
        result = _run_nl(self.eng, "成分に展開して")
        assert result["status"] == "completed", result
        assert "k_x" in result["message"], result

    def test_a_component_is_not_mistaken_for_an_unopened_vector(self):
        r"""`x_1` is already a component of the declared vector `x`.  Reading its
        BASE as an unopened vector refused perfectly good componentwise
        notation."""
        # Convergent on purpose: with the all-space range filled in, a constant
        # integrand really IS divergent and Maxima says so — correctly.
        src = self._register(r"\int d^3k x_1 e^{-(k \cdot k)}")
        expr, names = self.eng.expand_vector_components(src.display_id)
        assert names == ["k_x", "k_y", "k_z"]
        assert "x" in expr.canonical


class TestTheInstructionInEnglish(_Session):
    """Bilingual by construction: the pattern is chosen by SESSION language, so
    an English instruction has to be sent to an English session."""

    LANG = "en"

    def test_english(self):
        self._register(r"\int \frac{d^3k}{(2\pi)^3} e^{i (k \cdot x)} f(k)")
        result = _run_nl(self.eng, "expand into components")
        assert result["status"] == "completed", result
        assert result["operation"] == "expand_components"
        assert "k_x" in result["message"], result


class TestTheMeasureIsItselfADeclaration:
    r"""`\int d^3x` says `x` is a three-dimensional vector.

    Requiring a separate `define_vector` first would be defensible; requiring it
    and NOT checking was not.  Measured 2026-08-08 with `x` undeclared:
    `\int d^3x f(x)` opened the measure, left `f(x)` alone, and returned
    ``f(x)·x_x·x_y·x_z`` — the exact silent wrong answer this module exists to
    prevent, reached by forgetting a declaration.
    """

    def setup_method(self):
        nb.start_engine(title="measure-declares", mode="standard",
                        language="ja", domains=[])
        self.eng = nb.get_engine()          # nothing declared on purpose

    def teardown_method(self):
        nb.stop_engine()

    def _register(self, latex):
        canonical = self.eng._handler.latex_to_canonical(latex) or latex
        return self.eng.register_expression(canonical, latex=latex)

    def test_nothing_is_declared_in_this_session(self):
        self._register(r"\int d^3x f(x)")
        assert self.eng.symbol_context.get_quantity("x") is None

    def test_the_integrand_is_opened_anyway(self):
        src = self._register(r"\int d^3x f(x)")
        expr, names = self.eng.expand_vector_components(src.display_id)
        assert names == ["x_x", "x_y", "x_z"]
        assert "f(x_x,x_y,x_z)" in expr.canonical.replace(" ", ""), expr.canonical

    def test_it_computes(self):
        src = self._register(r"\int_{-\infty}^{\infty} d^3x e^{-(x \cdot x)}")
        opened, _ = self.eng.expand_vector_components(src.display_id)
        value = self.eng.evaluate_held_integrals(opened.display_id)
        assert value.canonical.replace(" ", "") == "%pi^(3/2)"

    def test_a_dot_product_with_an_unknown_partner_is_refused(self):
        r"""`\int d^3k e^{i k\cdot x}` with `x` unknown: `k·x` cannot be opened,
        so `k` is still standing.  Refused out loud rather than multiplied."""
        src = self._register(r"\int d^3k e^{i (k \cdot x)}")
        with pytest.raises(UnopenedVector):
            self.eng.expand_vector_components(src.display_id)

    def test_an_unrelated_factor_is_left_where_it_is(self):
        """`g(x)` does not depend on k and is not touched."""
        src = self._register(r"\int d^3k f(k) g(x)")
        expr, _ = self.eng.expand_vector_components(src.display_id)
        assert "g(x)" in expr.canonical.replace(" ", ""), expr.canonical

    def test_the_session_declarations_are_not_edited(self):
        """The measure is honoured for the rewrite; it does not silently declare
        `x` a vector for every later expression in the session."""
        src = self._register(r"\int d^3x f(x)")
        self.eng.expand_vector_components(src.display_id)
        quantity = self.eng.symbol_context.get_quantity("x")
        assert quantity is None or quantity.math_type.algebra != "vector"


class TestADefinitionStaysADefinition(_Session):
    r"""The reason expansion must NOT evaluate (raised by the researcher).

    Introduce an arbitrary basis into a field-operator Hamiltonian and the
    expansion coefficients are all-space integrals BY DEFINITION —
    `h_{mn} = \int d^3x\, \phi_m^*(x)\,\hat h\,\phi_n(x)`.  The basis is
    arbitrary, so there is nothing to carry out: the integral is the definition,
    not a computation waiting to happen.

    Measured 2026-08-08, and this is what forced the design: asking Maxima to
    integrate such a thing over all space does not simply fail to simplify — it
    comes back `ok=False` with an EMPTY answer (a convergence question the
    persistent adapter cannot answer).  So an expansion that evaluated as a side
    effect would take the ordinary Hamiltonian workflow down.
    """

    def test_it_survives_the_expansion(self):
        src = self._register(r"\int d^3x conjugate(phi_m(x)) phi_n(x)")
        opened, _ = self.eng.expand_vector_components(src.display_id)
        assert "'integrate(" in opened.canonical, opened.canonical
        assert "phi_n(x_x,x_y,x_z)" in opened.canonical.replace(" ", "")

    def test_it_survives_being_asked_to_evaluate(self):
        """Asking is allowed, and the answer is "still this" — not a failure and
        not a wrong number."""
        src = self._register(r"\int d^3x conjugate(phi_m(x)) phi_n(x)")
        opened, _ = self.eng.expand_vector_components(src.display_id)
        value = self.eng.evaluate_held_integrals(opened.display_id)
        assert "'integrate(" in value.canonical, value.canonical

    def test_the_gaussian_next_to_it_still_evaluates(self):
        """Holding is not refusing: what CAN be carried out still is, when asked."""
        src = self._register(r"\int d^3x e^{-(x \cdot x)}")
        opened, _ = self.eng.expand_vector_components(src.display_id)
        value = self.eng.evaluate_held_integrals(opened.display_id)
        assert value.canonical.replace(" ", "") == "%pi^(3/2)"
