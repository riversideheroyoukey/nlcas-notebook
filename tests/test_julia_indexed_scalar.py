r"""A Julia session must not answer an indexed scalar with the CAS's own error.

docs/32 Phase 4 left "the Julia display leak" as a residual, on the reasoning
that `JuliaBackendHandler.parse_ir` returns None so `indexed_bases` stays empty
and the `V(i)` → `V_i` restore never fires.  Measured 2026-07-30, the leak is
NOT reachable that way: the expression never gets as far as being displayed.
`xi_k` lowers to `xi(k)`, and Julia answers

    Sym xi is not callable. Use @syms xi(var1, var2,...) to create it as a callable.

which the engine raises verbatim.  So the reachable defect is a hard failure
whose message is raw CAS syntax — Core Principle #1, in the one place the
researcher cannot avoid reading.  It hit the ordinary physics spellings: a
declared scalar (`xi_k`), a declared vector's component (`p_i`) and an
undeclared base (`V_i`).

Nothing in the Julia declaration path emits `@syms`, so the call form is never
valid; the generator refuses it instead, and says what to do.

DESIGN DECISION (user, 2026-07-30): refuse, do not auto-route.  Both ways of
making it "work" were measured and both cost more than they give —

* `Symbolics.@syms xi(..)` compiles, but the result displays as `xi(k)` (the
  very leak docs/32 predicted) and **cannot multiply a QuantumAlgebra
  operator**, so `xi_k c'_k c_k` — the reason the Julia backend exists — breaks;
* `param(:xi, "k")` composes with operators and displays as `xi_{k}`, but is a
  `QuExpr` rather than a Symbolics `Num` and QuantumAlgebra restricts index
  names to a single character (`-k` is rejected outright).

open_items JULIA-INDEXED-SCALAR carries the question of making it work.
"""
import pytest

from notebook.models.quantity import MathType, Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.i18n import t
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError

_KEY = "parser.unsupported_notation.julia_indexed_scalar"


def _generate(src: str, ctx: SymbolContext) -> str:
    ir = JuliaSemanticResolver().resolve(SharedParser().parse(src), ctx)
    return JuliaCodeGenerator(context=ctx).generate(ir)


def _ctx(**kinds: str) -> SymbolContext:
    """*kinds* maps a name to a Quantity.kind, or to "vector" — which is a
    ``scalar`` carrying ``MathType(algebra="vector")``, the shape
    ``define_vector`` actually registers."""
    ctx = SymbolContext()
    for name, kind in kinds.items():
        if kind == "vector":
            qty = Quantity(name=name, kind="scalar",
                           math_type=MathType(algebra="vector"))
        else:
            qty = Quantity(name=name, kind=kind)
        ctx.register(name, qty)
    return ctx


class TestTheRawCasErrorIsGone:
    r"""Each of these used to reach Julia and come back as `Sym … is not callable`.

    2026-08-02: the DECLARED cases (`xi_k`, `p_i`) no longer refuse — they lower
    to QuantumAlgebra `param`, which is what makes the ticket's headline
    expression `xi_k c'_k c_k` writable at last (see
    TestIndexedScalarNowWorks).  What stays refused is the case with no
    declaration to stand on, because routing that too turns a refusal into
    silent nonsense — measured on `\sum_k …`, where the structural `sum` itself
    became `param(:sum, k)`.
    """

    @pytest.mark.parametrize("src,ctx_kw", [
        ("V_i * 2",   {}),                    # undeclared base
        ("omega_0 + 1", {}),                  # digit subscript, undeclared
    ])
    def test_refused_before_the_cas(self, src, ctx_kw):
        with pytest.raises(UnknownNotationError) as exc:
            _generate(src, _ctx(**ctx_kw))
        assert exc.value.i18n_key == _KEY

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_message_is_translated_and_actionable(self, lang):
        msg = t(_KEY, lang, name="xi_k", base="xi", index="k")
        assert "unsupported_notation" not in msg          # key was resolved
        assert "{" not in msg                             # kwargs were filled
        assert "xi_k" in msg and "xi" in msg
        # it must not repeat the CAS's own vocabulary at the researcher
        assert "@syms" not in msg and "Sym " not in msg

    def test_both_languages_differ(self):
        assert t(_KEY, "ja") != t(_KEY, "en")


class TestWhatStillWorks:
    """The refusal is narrow: only the spelling that had no valid Julia form."""

    def test_an_operators_index_is_untouched(self):
        assert _generate("c_k", _ctx(c="operator")) == "c(_QA.QuIndex(\"k\"))"

    def test_a_bound_index_uses_the_param_form(self):
        r"""`\sum_k*xi_k` is the branch that always worked — QuantumAlgebra's
        indexed c-number.  It is what the refusal points the researcher at.

        (The explicit `*` is Julia's working spelling; the space-separated
        `\sum_k xi_k` is a separate historical gap, open_items
        AUDIT-JULIA-SUM-PREFIX.)"""
        code = _generate(r"\sum_k*xi_k", _ctx(xi="scalar"))
        assert "param(:xi, :k)" in code

    def test_the_scalar_still_commutes_with_operators_under_the_sum(self):
        """The form the refusal must not have broken: ξ_k c†_k c_k under Σ_k."""
        code = _generate(r"\sum_k*xi_k*c'_k*c_k",
                         _ctx(xi="scalar", c="operator"))
        assert "param(:xi, :k)" in code and "c'(" in code

    def test_the_full_name_declared_as_one_symbol_is_an_atom(self):
        """The route the message tells the researcher to take."""
        assert _generate("xi_k * 2", _ctx(xi_k="scalar")) == "xi_k * 2"

    def test_a_bare_scalar_is_untouched(self):
        assert _generate("xi * 2", _ctx(xi="scalar")) == "xi * 2"


class TestEndToEnd:
    """Through a real Julia session: the researcher sees the guidance, and the
    CAS is never handed the invalid text."""

    def test_a_declared_base_now_computes(self):
        """2026-08-02: this used to be the refusal case.  With `xi` declared,
        `xi_k` is a QuantumAlgebra c-number and the expression computes — and
        displays in the researcher's own notation."""
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="jl indexed scalar",
                            mode=SessionMode.second_quantization) as e:
            e.run_operation("define_scalar_parameter", {"name": "xi"})
            ex = e.register_expression("xi_k * 2")
            step, _ = e.run_operation("expand", {"expr": ex.display_id})
            assert step.status.value == "success"
            latex = e.expressions[step.output_refs[0]].latex
            assert "xi" in latex.replace("\\", "")
            assert "not callable" not in latex and "@syms" not in latex

    def test_an_undeclared_base_still_refuses_and_explains(self):
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="jl indexed scalar undecl",
                            mode=SessionMode.second_quantization) as e:
            ex = e.register_expression("V_i * 2")
            with pytest.raises(ValueError) as exc:
                e.run_operation("expand", {"expr": ex.display_id})
            msg = str(exc.value)
            assert "not callable" not in msg and "@syms" not in msg
            assert "V_i" in msg

    def test_the_declared_full_name_computes_and_displays(self):
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode

        with NotebookEngine(title="jl full name",
                            mode=SessionMode.second_quantization) as e:
            e.run_operation("define_scalar_parameter", {"name": "xi_k"})
            ex = e.register_expression("xi_k * 2")
            step, _ = e.run_operation("expand", {"expr": ex.display_id})
            assert step.status.value == "success"
            # the researcher's own notation, not the CAS's
            latex = e.expressions[step.output_refs[0]].latex
            assert r"\xi_{k}" in latex and "(" not in latex


class TestUndeclaredAppliedFormIsExplained:
    r"""An UNDECLARED name applied to arguments must not surface as raw CAS text.

    `a_i(k)`, `G_0(k)`, `Gzero(k)` and `g(x)` all come back from Julia as
    `Sym <name> is not callable. Use @syms …` (measured 2026-08-01).  `a_i(k)`
    was the one on the books (open_items JULIA-INDEXED-SCALAR, "同種の漏れ"),
    but measuring showed the ticket's premise was wrong twice over: the
    discriminator is NOT the underscore (`Gzero(k)` fails identically), and a
    Julia builtin-function list is not needed — against the running session
    only `time_ns` and `print_ir` carry the `name_x` shape at all.

    **This is translated after the fact, not refused before the CAS.**  A
    pre-CAS guard was implemented first and measured wrong: it had to allow-list
    every name the system itself emits, and two full-suite runs showed it still
    refused legitimate ones — `c(1)` (the second-quantization letter
    convention) and `adjoint(c(1))`, which is the GENERATOR'S OWN output
    re-entering as canonical text.  The defect is message quality, not a wrong
    answer, so a fix whose blast radius is every function call in a Julia
    session is the wrong trade.  Translating the failure catches every spelling,
    including the ones an allow-list would forget.
    """
    _KEY_CALL = "error.julia_undeclared_call"
    _KEY_INDEX = "error.julia_multichar_index"

    def _handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        return JuliaBackendHandler.__new__(JuliaBackendHandler)

    @pytest.mark.parametrize("name", ["a_i", "G_0", "Gzero", "g"])
    def test_not_callable_is_classified(self, name):
        h = self._handler()
        err = (f"Sym {name} is not callable. "
               f"Use @syms {name}(var1, var2,...) to create it as a callable.")
        assert h.classify_cas_error(err) == self._KEY_CALL
        assert h.cas_error_args(err) == {"name": name}

    def test_multichar_index_is_classified(self):
        h = self._handler()
        err = "ArgumentError: index names must be single character, got omega"
        assert h.classify_cas_error(err) == self._KEY_INDEX
        assert h.cas_error_args(err) == {"index": "omega"}

    def test_unrelated_errors_are_left_alone(self):
        h = self._handler()
        assert h.classify_cas_error("some other Julia error") is None
        assert h.cas_error_args("some other Julia error") == {}

    def test_existing_classification_still_wins(self):
        h = self._handler()
        err = "MethodError: no method matching *(::QuExpr, ::QuExprConstructor{X})"
        assert h.classify_cas_error(err) == "error.operator_missing_index"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_message_names_the_symbol_and_the_fix(self, lang):
        msg = t(self._KEY_CALL, lang, name="G_0")
        assert msg != self._KEY_CALL
        assert "G_0" in msg
        assert "{name}" not in msg

    def test_declaring_the_name_really_works(self):
        """The advice must be true: a declared name lowers to `param`."""
        assert _generate("G_0(k)", _ctx(G_0="scalar")) == "param(:G_0, :k)"

    def test_generation_is_not_pre_emptively_refused(self):
        """Generation stays permissive — that is the point of the redesign."""
        assert _generate("Gzero(k)", _ctx()) == "Gzero(k)"
        assert _generate("c(1)", _ctx()) == "c(1)"
        assert _generate("sin(x)", _ctx()) == "sin(x)"

    def test_multichar_index_is_still_refused_where_we_emit_it(self):
        """The one pre-CAS check that survived: we KNOW `param(:G_0, :k, :omega)`
        is about to hand `omega` to QA as an index, so it is refused there."""
        with pytest.raises(UnknownNotationError) as ei:
            _generate("G_0(k,omega)", _ctx(G_0="scalar"))
        assert ei.value.i18n_key == "parser.unsupported_notation.julia_multichar_index"


class TestIndexedScalarNowWorks:
    r"""The ticket's own headline case, finally computable.

    `xi_k c'_k c_k` is ordinary condensed-matter writing, and the ticket
    (JULIA-INDEXED-SCALAR) was open because it could not be written at all: an
    unbound indexed scalar was refused, and the only way through was to declare
    the whole `xi_k` as one name.

    Measured 2026-08-02, `param` carries the work — it multiplies QA operators,
    survives `normal_form`, adds/multiplies/powers with other params, takes
    scalar factors, and even multiplies a Symbolics variable (`t xi(k)`).  Its
    latex is the researcher's own subscript, so no display bridge is needed.
    The ticket's "QuExpr does not mix with Symbolics" was too broad: only
    DIFFERENTIATION fails, and that is now said in words.

    **One representation, not a mode split.**  Bound and unbound indices take
    the same road, so the dual representation AUDIT-SUBST-INDEXED was about
    never appears.
    """
    _KEY_REFUSE = "parser.unsupported_notation.julia_indexed_scalar"

    def test_declared_base_with_unbound_index_lowers_to_param(self):
        assert _generate("xi_k", _ctx(xi="scalar")) == 'param(:xi, _QA.QuIndex("k"))'

    def test_the_headline_expression_lowers(self):
        code = _generate("xi_k c'_k c_k", _ctx(xi="scalar", c="operator"))
        assert 'param(:xi, _QA.QuIndex("k"))' in code
        assert "c'(_QA.QuIndex(\"k\"))" in code

    def test_declared_vector_component_too(self):
        assert _generate("p_i", _ctx(p="vector")) == 'param(:p, _QA.QuIndex("i"))'

    def test_undeclared_base_is_still_refused(self):
        """The declaration is the warrant.  Routing undeclared bases as well
        turns a refusal into silent nonsense — measured: `\\sum_k xi_k c'_k c_k`
        resolves to a Mul in this backend, so the structural `sum` itself became
        `param(:sum, k)` and the whole expression a meaningless product."""
        with pytest.raises(UnknownNotationError) as ei:
            _generate("V_i", _ctx())
        assert ei.value.i18n_key == self._KEY_REFUSE

    def test_the_sum_token_is_not_turned_into_a_parameter(self):
        """The regression that measurement caught before it shipped.

        2026-08-07: this used to assert a REFUSAL, because `\\sum_k …` resolved
        to a Mul on this backend and the structural `sum` therefore arrived as
        an indexed scalar with an undeclared base — refused, which is what kept
        `param(:sum, k)` from shipping.  The Mul resolution was the actual
        defect (AUDIT-JULIA-SUM-PREFIX) and is fixed, so `sum` is no longer a
        symbol here at all: it is the sum.  The property this test exists for is
        unchanged, so it is asserted directly instead of through the refusal.
        """
        code = _generate(r"\sum_k xi_k c'_k c_k", _ctx(xi="scalar", c="operator"))
        assert "param(:sum" not in code           # never a parameter
        assert code.startswith("_QA.∑(:k,")       # …because it is a real sum
        assert "param(:xi, :k)" in code           # the coefficient still is one

    def test_multichar_index_still_refused(self):
        with pytest.raises(UnknownNotationError) as ei:
            _generate("xi_omega", _ctx(xi="scalar"))
        assert ei.value.i18n_key == \
            "parser.unsupported_notation.julia_multichar_index"

    def test_differentiation_limit_is_said_in_words(self):
        """The one thing param cannot do, translated rather than left as a
        Julia MethodError."""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        err = "MethodError: no method matching (::Differential)(::QuExpr)"
        assert h.classify_cas_error(err) == \
            "error.julia_quexpr_not_differentiable"

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_differentiation_message_points_somewhere(self, lang):
        msg = t("error.julia_quexpr_not_differentiable", lang)
        assert msg != "error.julia_quexpr_not_differentiable"
        assert "xi_k" in msg
        assert ("Maxima" in msg or "maxima" in msg)   # names the way out
