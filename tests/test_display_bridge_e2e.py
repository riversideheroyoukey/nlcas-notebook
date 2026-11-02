"""E2E tests for the Display Bridge: CAS output → display LaTeX (E-1).

The unit tests in test_latex_converter.py cover ``restore_decorations()`` in
isolation, and test_notation_bridge_safety.py covers kron_delta / matrix ops /
basic index lowering.  These tests exercise the *full* CAS-output → display
pipeline (``postprocess_result_latex`` / ``normalize_display_latex``) for the
remaining responsibilities: decoration restoration through the orchestrator,
multi-index lowering, the natural-vars / time-dependent guards, and the Julia
quantum-operator prime→dagger conversion.

Both backend handlers are constructed without a live CAS (``Handler(None)``);
``postprocess_result_latex`` is a pure transform over (canonical, raw_latex,
registry).
"""
from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import Quantity
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.julia_backend_handler import JuliaBackendHandler


def _maxima() -> MaximaBackendHandler:
    return MaximaBackendHandler(None)


def _julia() -> JuliaBackendHandler:
    return JuliaBackendHandler(None)


class TestDecorationRestoreE2E:
    """CAS output with canonical decoration suffixes → LaTeX decoration commands,
    through the full postprocess pipeline (not restore_decorations in isolation)."""

    def test_hat(self):
        ctx = SymbolContext()
        out = _maxima().postprocess_result_latex("H_hat", "H_hat", ctx)
        assert out == r"\hat{H}"

    def test_tilde_on_greek(self):
        ctx = SymbolContext()
        out = _maxima().postprocess_result_latex("omega_tilde", "omega_tilde", ctx)
        assert out == r"\tilde{\omega}"

    def test_decoration_in_sum_expression(self):
        ctx = SymbolContext()
        out = _maxima().postprocess_result_latex("H_hat + V_hat", "H_hat + V_hat", ctx)
        assert out == r"\hat{H} + \hat{V}"


class TestItalicIdentifierUnwrapE2E:
    """Maxima renders bare multi-char identifiers as {\\it name} (italic, escaped
    underscore) in solve / inverse-transform results — a backend artifact that must
    not reach the user.  Unwrap to a plain name with a braced subscript so it matches
    the other backends."""

    def test_bare_italic_with_subscript(self):
        ctx = SymbolContext()
        out = _maxima().postprocess_result_latex(
            "alpha_mk", r"{\it alpha\_mk}", ctx)
        assert r"\it" not in out and r"\_" not in out
        assert "alpha_{mk}" in out

    def test_italic_inside_fraction(self):
        ctx = SymbolContext()
        out = _maxima().postprocess_result_latex(
            "c_k", r"\frac{{\it alpha\_mk} v + {\it alpha\_k} u}{v^2+u^2}", ctx)
        assert r"\it" not in out
        assert "alpha_{mk}" in out and "alpha_{k}" in out


class TestIndexLoweringE2E:
    """Indexed-symbol function/array calls → subscript notation."""

    def test_scalar_single_index_maxima(self):
        ctx = SymbolContext()
        ctx.register_quantity("x", kind="scalar")
        out = _maxima().postprocess_result_latex("x(i)", r"x\left(i\right)", ctx)
        assert "x_{i}" in out

    def test_scalar_multi_index_maxima(self):
        ctx = SymbolContext()
        ctx.register_quantity("g", kind="scalar")
        out = _maxima().postprocess_result_latex("g(k,q)", r"g\left(k, q\right)", ctx)
        assert "g_{k,q}" in out

    def test_array_access_julia(self):
        ctx = SymbolContext()
        ctx.register_quantity("A", kind="scalar")
        out = _julia().postprocess_result_latex("A[i, j]", r"A\left[i, j\right]", ctx)
        assert "A_{i,j}" in out


class TestDisplayBridgeGuards:
    """The bridge must NOT lower a symbol's natural/time variables to subscripts."""

    def test_natural_var_not_lowered(self):
        ctx = SymbolContext()
        ctx.register("u", quantity=Quantity(name="u", kind="scalar", natural_vars=["x"]))
        # x is a natural variable of u → keep u(x), do NOT produce u_{x}
        out = _maxima().postprocess_result_latex("u(x)", r"u\left(x\right)", ctx)
        assert "u_{x}" not in out
        assert r"u\left(x\right)" in out

    def test_non_natural_index_is_lowered(self):
        ctx = SymbolContext()
        ctx.register("u", quantity=Quantity(name="u", kind="scalar", natural_vars=["x"]))
        # k is not a natural variable → lower to subscript
        out = _maxima().postprocess_result_latex("u(k)", r"u\left(k\right)", ctx)
        assert "u_{k}" in out

    def test_time_dependent_var_not_subscripted(self):
        ctx = SymbolContext()
        ctx.register("q", quantity=Quantity(
            name="q", kind="scalar", natural_vars=["t"], time_dependent=True))
        out = _maxima().postprocess_result_latex("q(t)", r"q\left(t\right)", ctx)
        # the time parameter must never become a subscript
        assert "q_{t}" not in out


class TestQuantumOperatorPrimeAdjoint:
    """Julia: creation-operator prime notation a'(k) → a^{\\dagger}(k) on display.

    The operator is DECLARED: a prime is an adjoint only on a name this session
    declared one, because `t'` / `\\sigma'` / `\\mathbf r'` carry the same mark
    for a DISTINCT VARIABLE (PRIME-DISPLAY-DAGGER, 2026-09-10,
    `tests/test_prime_is_not_always_a_dagger.py`).
    """

    @staticmethod
    def _with_a() -> JuliaBackendHandler:
        h = _julia()
        ctx = SymbolContext()
        ctx.register("a", quantity=Quantity(name="a", kind="operator"))
        h.set_symbol_context(ctx)
        return h

    def test_prime_to_dagger(self):
        out = self._with_a().normalize_display_latex("a'(k)")
        assert out == r"a^{\dagger}(k)"

    def test_prime_to_dagger_in_product(self):
        out = self._with_a().normalize_display_latex("a'(k)*a(k)")
        assert r"a^{\dagger}(k)" in out
        assert "a(k)" in out


class TestTheAdjointSurvivesTheDisplayBridge:
    r"""Found running S-l (squeezed states) 2026-08-04.

    Two defects in the same place, one shipped and one latent:

    * **Shipped.**  Julia's Latexify hands a bare operator name back WRAPPED —
      `b'(1)` comes back as ``\mathrm{b'}\left( 1 \right)``.  The bridge did not
      recognise that shape, so every solve / inverse-transform result showed the
      CAS's own prime spelling.  The committed S-c example has
      ``\mathrm{g'}\left( k, u \right)`` sitting beside a correctly rendered
      ``{c}_{k}^{\dagger}`` from QuantumAlgebra's latex — the same operator,
      three spellings in one timeline.

    * **Latent, and made reachable by fixing the first.**  The adjoint marks
      were counted from the REGISTERED NAME, and operators are registered under
      the bare base (`b`), so the count was always 0 and any matched form lost
      its dagger: ``b'(k) b(k)`` → ``b_{k} b_{k}``.  A number operator displayed
      as something else.  Measured unreachable at the time (a differential check
      over every generated Julia example found 0 collapsed displays) — but
      teaching the pattern the wrapped shape is exactly what would have made it
      live, so both are fixed together.
    """

    def _ctx(self):
        ctx = SymbolContext()
        ctx.session_mode = "second_quantization"
        ctx.register("b", Quantity(name="b", kind="operator", statistics="bose"))
        ctx.register("g", Quantity(name="g", kind="operator", statistics="fermi"))
        return ctx

    def test_bare_prime_keeps_its_dagger(self):
        h, ctx = _julia(), self._ctx()
        assert h._apply_display_bridge("b'(1)", ctx) == r"b^{\dagger}_{1}"
        assert h._apply_display_bridge(r"b^{\dagger}(1)", ctx) == r"b^{\dagger}_{1}"

    def test_the_number_operator_is_not_collapsed(self):
        """`b†b` and `bb` are different quantities; they must not display alike."""
        h, ctx = _julia(), self._ctx()
        out = h._apply_display_bridge("b'(k) b(k)", ctx)
        assert out == r"b^{\dagger}_{k} b_{k}", out
        assert out != h._apply_display_bridge("b(k) b(k)", ctx)

    def test_annihilation_is_untouched(self):
        h, ctx = _julia(), self._ctx()
        assert h._apply_display_bridge("b(1)", ctx) == "b_{1}"

    def test_the_latexify_wrapper_is_unwrapped(self):
        h, ctx = _julia(), self._ctx()
        assert h._apply_display_bridge(r"\mathrm{b'}\left( 1 \right)", ctx) \
            == r"b^{\dagger}_{1}"
        assert h._apply_display_bridge(r"\mathrm{b}\left( 1 \right)", ctx) \
            == "b_{1}"
        # the shape from the shipped S-c example, spaces and minus sign included
        assert h._apply_display_bridge(r"\mathrm{g'}\left(  - k, d \right)", ctx) \
            == r"g^{\dagger}_{-k,d}"

    def test_maxima_does_not_unwrap(self):
        r"""`\mathrm{}` wrapping is Latexify's habit, so only the Julia handler
        declares it (Backend Encapsulation Rule).  Maxima leaves it alone."""
        ctx = SymbolContext()
        ctx.register("b", Quantity(name="b", kind="scalar"))
        out = _maxima()._apply_display_bridge(r"\mathrm{b}\left( 1 \right)", ctx)
        assert out == r"\mathrm{b}\left( 1 \right)", out

    def test_already_subscripted_output_is_left_alone(self):
        """QuantumAlgebra's own latex is already in the researcher's notation."""
        h, ctx = _julia(), self._ctx()
        s = r"{b}_{1}^{\dagger}"
        assert h._apply_display_bridge(s, ctx) == s
