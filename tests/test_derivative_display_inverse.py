r"""BP-9: the derivative DISPLAY forms must survive latex_to_canonical.

Two producers emit LaTeX for an unevaluated derivative — Maxima's own `tex1`
(`\frac{d^2}{d t d z} A_{z}`) and `lhs_formatting._format_provenance_lhs` (the
H-1 LHS label, `\frac{\partial phi}{\partial x}` and friends).  Fed to the
generic `\frac{a}{b}` → `(a)/(b)` rule they became **silently wrong** Maxima:
`d` and `\partial` degrade to ordinary symbols, so `∂φ/∂x` cancels to `φ/x`
with no error and the wrong value.

`TestGeneratedFormsAreCovered` is the load-bearing one: it asks the real
generator for its output instead of hard-coding shapes, so a change to the
display format cannot quietly reopen the hole.
"""
import logging
from contextlib import contextmanager

import pytest

from notebook.engine import NotebookEngine
from notebook.models.quantity import Quantity
from notebook.runtime.i18n import t
from notebook.runtime.lhs_formatting import _format_provenance_lhs
from notebook.runtime.maxima_derivative_latex import (
    REASON_NO_SEPARATOR,
    REASON_ORDER,
    restore_derivative_display_latex,
)

import ui.notebook_bridge as nb


def _handler():
    return NotebookEngine()._handler


class TestMaximaTex1Forms:
    """What Maxima's own tex1 emits, after clean_latex."""

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{d z} \varphi",           "'diff(phi,z,1)"),
        (r"\frac{d^2}{d z^2} \varphi",       "'diff(phi,z,2)"),
        (r"\frac{d^2}{d t d z} A_{z}",       "'diff(A_z,t,1,z,1)"),
        (r"\frac{d^3}{d t d z^2} f",         "'diff(f,t,1,z,2)"),
        (r"\frac{d^3}{d x d y d z} f",       "'diff(f,x,1,y,1,z,1)"),
        (r"\frac{d^2}{d t d \theta} A_{z}",  "'diff(A_z,t,1,theta,1)"),
    ])
    def test_latex_to_canonical(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical


class TestProvenanceLhsForms:
    """What lhs_formatting._format_provenance_lhs writes as the H-1 LHS label."""

    @pytest.mark.parametrize("latex,canonical", [
        # operand carried in the numerator
        (r"\frac{\partial phi}{\partial x}",                "'diff(phi,x,1)"),
        (r"\frac{\partial^{2} phi}{\partial x^{2}}",        "'diff(phi,x,2)"),
        (r"\frac{\partial^{2} phi}{\partial z\partial t}",  "'diff(phi,z,1,t,1)"),
        (r"\frac{\partial^{3} phi}{\partial x\partial y\partial z}",
         "'diff(phi,x,1,y,1,z,1)"),
        (r"\frac{dy}{dx}",                                  "'diff(y,x,1)"),
        (r"\frac{d^{2}y}{dx^{2}}",                          "'diff(y,x,2)"),
        (r"\frac{\partial gauge}{\partial t}",              "'diff(gauge,t,1)"),
        # operator style, operand follows
        (r"\frac{\partial}{\partial x} f(x, t)",            "'diff(f(x, t),x,1)"),
        (r"\frac{\partial^{2}}{\partial x\partial t} f(x, t)",
         "'diff(f(x, t),x,1,t,1)"),
        (r"\frac{\partial}{\partial x}\left(a*b+c\right)",  "'diff(a*b+c,x,1)"),
        (r"\frac{d^{3}}{dx^{2}dy}\left(a*b+c\right)",       "'diff(a*b+c,x,2,y,1)"),
    ])
    def test_latex_to_canonical(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical


class TestGeneratedFormsAreCovered:
    """Round-trip against the REAL generator, so a display-format change that
    outruns the inverse fails here instead of silently mistranslating."""

    CASES = [
        ("phi",    ["x"],           True,  "'diff(phi,x,1)"),
        ("phi",    ["x", "x"],      True,  "'diff(phi,x,2)"),
        ("phi",    ["z", "t"],      True,  "'diff(phi,z,1,t,1)"),
        ("phi",    ["x", "y", "z"], True,  "'diff(phi,x,1,y,1,z,1)"),
        ("y",      ["x"],           False, "'diff(y,x,1)"),
        ("y",      ["x", "x"],      False, "'diff(y,x,2)"),
        ("f(x,t)", ["x"],           True,  "'diff(f(x, t),x,1)"),
        ("f(x,t)", ["x", "t"],      True,  "'diff(f(x, t),x,1,t,1)"),
        ("a*b+c",  ["x"],           True,  "'diff(a*b+c,x,1)"),
        ("a*b+c",  ["x", "x", "y"], False, "'diff(a*b+c,x,2,y,1)"),
    ]

    @pytest.mark.parametrize("origin,dvars,is_partial,canonical", CASES)
    def test_roundtrip(self, origin, dvars, is_partial, canonical):
        h = _handler()
        latex = _format_provenance_lhs(origin, dvars, is_partial,
                                       qty=None, parse_ir=h.parse_ir)
        assert h.latex_to_canonical(latex) == canonical, latex

    def test_declared_scalar_and_function_shapes(self):
        h = _handler()
        scalar = _format_provenance_lhs(
            "c", ["a", "b"], True,
            qty=Quantity(name="c", kind="scalar"), parse_ir=h.parse_ir)
        assert h.latex_to_canonical(scalar) == "'diff(c,a,1,b,1)"
        func = _format_provenance_lhs(
            "g", ["x", "y"], True,
            qty=Quantity(name="g", kind="function", natural_vars=["x", "y"]),
            parse_ir=h.parse_ir)
        assert h.latex_to_canonical(func) == "'diff(g(x,y),x,1,y,1)"


class TestOrdinaryFractionsUntouched:
    """Everything that is not unambiguously a derivative stays a division."""

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{a}{b}",       "((a)/(b))"),
        (r"\frac{x+1}{x-1}",   "((x+1)/(x-1))"),
        (r"\frac{1}{2}",       "((1)/(2))"),
        (r"\frac{d}{dt}",      "((d)/(dt))"),          # no operand at all
        (r"\frac{d^3}{d t d z} f", "((d^3)/(d*t*d*z))*f"),  # orders do not add up
    ])
    def test_not_a_derivative(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    def test_kinetic_energy_fraction(self):
        """The classic \\frac{1}{2}mv^2 must not be mistaken for anything."""
        assert _handler().latex_to_canonical(r"\frac{1}{2} m v^2") == "((1)/(2))*m*v^2"


class TestAmbiguityIsRefused:
    """`dxdy` has two readings (d·x d·y, and d·(xdy)); the numerator's total
    order picks one.  When nothing picks a unique reading, leave it alone."""

    def _restore(self, s, refusals=None):
        on_refused = (lambda frac, reason: refusals.append((frac, reason))
                      ) if refusals is not None else None
        return restore_derivative_display_latex(s, lambda x: x,
                                                on_refused=on_refused)

    def test_order_two_splits_into_two_factors(self):
        assert self._restore(r"\frac{d^{2}}{dxdy}\left(f\right)") == "'diff(f,x,1,y,1)"

    def test_unsatisfiable_order_is_left_verbatim(self):
        src = r"\frac{d^{5}}{dxdy}\left(f\right)"
        refusals = []
        assert self._restore(src, refusals) == src
        assert [r for _, r in refusals] == [REASON_ORDER]

    def test_multiletter_name_without_a_separator_is_refused(self):
        r"""`dxy` could be d·(xy) or d·x·y; without a separator we do not pick.
        Writing `d xy` (or using \partial) resolves it."""
        src = r"\frac{d}{dxy}\left(f\right)"
        refusals = []
        assert self._restore(src, refusals) == src
        assert [r for _, r in refusals] == [REASON_NO_SEPARATOR]
        # …and with the space it converts.
        assert self._restore(r"\frac{d}{d xy}\left(f\right)") == "'diff(f,xy,1)"


class TestOperatorAppliedToAnOperand:
    r"""`\frac{d}{dt} f(t)` — the operator form, with the operand following.

    The bracketed cases are the reason the operand is scanned with balanced
    delimiters instead of a regex: `\left( f(t) + g(t) \right)` contains
    ordinary parentheses of its own, and a non-counting match read `\left` as
    the operand and emitted `'diff(,t,1)` — an EMPTY operand, worse than
    refusing outright.
    """

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{dt} f(t)",                   "'diff(f(t),t,1)"),
        (r"\frac{d}{dt}f(t)",                    "'diff(f(t),t,1)"),   # no space
        (r"\frac{d}{d t} f(t)",                  "'diff(f(t),t,1)"),
        (r"\frac{d^{2}}{dt^{2}} f(t)",           "'diff(f(t),t,2)"),
        (r"\frac{d}{dt} x",                      "'diff(x,t,1)"),
        (r"\frac{\partial}{\partial t} f(x,t)",  "'diff(f(x,t),t,1)"),
    ])
    def test_plain_operand(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{dt} \left( a*b+c \right)",   "'diff(a*b+c,t,1)"),
        # operands carrying parentheses of their own
        (r"\frac{d}{dt} \left( f(t) + g(t) \right)", "'diff(f(t) + g(t),t,1)"),
        (r"\frac{d}{dt} ( f(t) + g(t) )",            "'diff(f(t) + g(t),t,1)"),
        (r"\frac{\partial^{2}}{\partial x\partial t} \left( f(x,t)*g(x) \right)",
         "'diff(f(x,t)*g(x),x,1,t,1)"),
    ])
    def test_bracketed_operand(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    def test_operator_with_no_operand_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _handler().latex_to_canonical(r"\frac{d}{dt}") == "((d)/(dt))"
        assert any("deriv" in r.getMessage() or "微分の表記" in r.getMessage()
                   for r in caplog.records)


class TestCalledOperandKeepsItsArguments:
    r"""AUDIT-DERIV-DISPLAY-RT: `\frac{d}{d x} f\left(x , y\right)`.

    Maxima's tex1 writes a call's arguments with `\left(`…`\right)`, and only
    the bare `(` form was read.  The arguments were left behind for the generic
    rule, which turned them into a PRODUCT — `'diff(f,x,1)*(x , y)`, a different
    expression, with no error anywhere.
    """

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{d x} f\left(x , y\right)",     "'diff(f(x , y),x,1)"),
        (r"\frac{d^2}{d x^2} f\left(x , y\right)", "'diff(f(x , y),x,2)"),
        (r"\frac{d^2}{d x d y} f\left(x , y\right)",
         "'diff(f(x , y),x,1,y,1)"),
        (r"\frac{\partial}{\partial t} \psi\left(x , t\right)",
         "'diff(psi(x , t),t,1)"),
        # a single argument, and a term following the derivative
        (r"\frac{d}{d t} f\left(t\right)",         "'diff(f(t),t,1)"),
        (r"\frac{d}{d x} f\left(x , y\right)+h\left(x\right)",
         "'diff(f(x , y),x,1)+h(x)"),
    ])
    def test_arguments_stay_with_the_operand(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    def test_unbalanced_arguments_are_refused_not_dropped(self):
        r"""Returning the bare name would leave `\left(x` behind to become a
        product — the failure this class is about.  Left verbatim instead."""
        src = r"\frac{d}{d x} f\left(x , y"
        assert "'diff(" not in _handler().latex_to_canonical(src)

    def test_space_before_the_bracket_is_still_ambiguous(self):
        r"""`f \left(x\right)` may be a product; deciding that needs the
        SymbolContext, so this module leaves the operand at the bare name
        (CLAUDE.md, Math Input Conventions)."""
        assert (_handler().latex_to_canonical(r"\frac{d}{d x} f \left(x\right)")
                == "'diff(f,x,1)*(x)")


class TestHighOrderDotFormsAreReadable:
    r"""AUDIT-DERIV-DISPLAY-RT: `\dddot{q}` and `{q}^{(4)}`.

    `_apply_dot_notation_to_latex` prints these for orders 3 and 4+, and neither
    could be read back: `\dddot` is not in the shared accent table so it decayed
    to the symbol `\dddotq`, and `{q}^{(4)}` parsed as the fourth POWER — a
    derivative displayed and re-read as something else entirely, silently.
    """

    @staticmethod
    @contextmanager
    def _session(time_var="t"):
        with NotebookEngine(title="dddot") as e:
            e.declare_time_dependent("q", time_var)
            yield e

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_generated_display_reads_back(self, order):
        """Drive the REAL generator: differentiate until the order is reached,
        then re-input exactly what the timeline showed."""
        with self._session() as e:
            cur = e.register_expression("q")
            for _ in range(order):
                step, _ = e.run_operation(
                    "differentiate", {"expr": cur.display_id, "var": "t"})
                cur = e.expressions[step.output_refs[0]]
            # the display is "<Leibniz LHS> = <dot RHS>"; both halves must read
            back = e._handler.latex_to_canonical(cur.latex)
            rhs = back.split("=")[-1].strip()
            assert "\\" not in back, back          # no LaTeX command survived
            if order <= 2:
                # documented atoms whose meaning is mode-dependent (docs/19)
                assert rhs == f"q_{'d' * order}ot"
            else:
                assert rhs == f"'diff(q,t,{order})"

    @pytest.mark.parametrize("latex,canonical", [
        (r"\dddot{q}",       "'diff(q,t,3)"),
        (r"\dddot{\varphi}", "'diff(phi,t,3)"),
        (r"a*\dddot{q} + b", "a*'diff(q,t,3) + b"),
    ])
    def test_dddot_needs_no_declaration(self, latex, canonical):
        r"""`\dddot` is an accent: it cannot mean anything but a derivative, so
        it is read for any base, with `t` when nothing is declared."""
        assert _handler().latex_to_canonical(latex) == canonical

    def test_dddot_uses_the_declared_time_parameter(self):
        with self._session("tau") as e:
            assert e._handler.latex_to_canonical(r"\dddot{q}") == "'diff(q,tau,3)"

    def test_order_superscript_needs_the_declaration(self):
        r"""`^{(n)}` is a superscript and collides with exponentiation, so it is
        read only for a quantity the generator would have printed it for.  An
        undeclared `x^{(4)}` is the fourth power, which is what it says."""
        assert _handler().latex_to_canonical(r"{x}^{(4)}") == "x^((4))"
        with self._session() as e:
            assert e._handler.latex_to_canonical(r"{q}^{(4)}") == "'diff(q,t,4)"
            assert e._handler.latex_to_canonical(r"q^{(7)}") == "'diff(q,t,7)"

    def test_ordinary_powers_are_untouched(self):
        h = _handler()
        for src in ("x^2", "x^{2}", "q^{(a+b)}", "(x+1)^{3}"):
            assert "diff" not in h.latex_to_canonical(src), src


class TestUnseparatedFalsePositivesRefused:
    r"""`\frac{delta}{dx}` must NOT become 'diff(elta,x,1).

    Reading a leading `d` as the differential operator is only safe when
    something separates it from the name — otherwise an ordinary fraction of two
    d-initial names is silently mangled, which is the very failure mode BP-9 is
    about.  Same rule as CLAUDE.md's no-auto-split convention for juxtaposition.
    """

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{delta}{dx}",    "((delta)/(dx))"),
        (r"\frac{data}{depth}",  "((data)/(depth))"),
        (r"\frac{dose}{day}",    "((dose)/(day))"),
        (r"\frac{drag}{dyne}",   "((drag)/(dyne))"),
    ])
    def test_stays_a_fraction(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{dy}{dx}",     "'diff(y,x,1)"),      # single letter
        (r"\frac{dV_x}{dt}",   "'diff(V_x,t,1)"),    # letter + subscript
        (r"\frac{dx_1}{dt}",   "'diff(x_1,t,1)"),    # letter + digit subscript
        (r"\frac{d\psi}{dx}",  "'diff(psi,x,1)"),    # LaTeX command separates
        (r"\frac{d^{2}y}{dx^{2}}", "'diff(y,x,2)"),  # the order separates
        (r"\frac{d T}{d t}",   "'diff(T,t,1)"),      # a space separates
    ])
    def test_separated_forms_still_convert(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical


class TestWarnsAndContinues:
    """A fraction that looks like a derivative but cannot be read leaves the
    expression alone (an ordinary division, as before) AND says so."""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("latex,reason", [
        (r"\frac{d^3}{d t d z} f", "deriv_reason_order"),
        (r"\frac{delta}{dx}",      "deriv_reason_no_separator"),
        (r"\frac{d}{dt}",          "deriv_reason_no_operand"),
    ])
    def test_warning_is_logged(self, caplog, lang, latex, reason):
        """Named reason, in the session's language (I18N-1)."""
        nb.start_engine(title="test_bp9_warn", mode="standard", language=lang)
        eng = nb.get_engine()
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                out = eng._handler.latex_to_canonical(latex)
            assert out and "'diff(" not in out          # processing continued
            text = " ".join(r.getMessage() for r in caplog.records)
            assert t("notation.deriv_unreadable", lang).split("{")[0] in text
            assert t(f"notation.{reason}", lang)[:12] in text
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("latex", [
        r"\frac{a}{b}", r"\frac{1}{2} m v^2", r"\frac{x+1}{x-1}",
        r"\frac{dy}{dx}", r"\frac{\partial phi}{\partial x}",
    ])
    def test_no_warning_for_plain_fractions_or_clean_derivatives(self, caplog, latex):
        """The warning must stay rare enough to mean something."""
        with caplog.at_level(logging.WARNING):
            _handler().latex_to_canonical(latex)
        assert not [r for r in caplog.records
                    if "deriv_unreadable" in r.getMessage()
                    or "微分の表記" in r.getMessage()
                    or "as a derivative" in r.getMessage()]


class TestSubstitutionUsesTheRestoredForm:
    """The BP-9 danger in situ: a partial-derivative display string reaching the
    CAS as a division silently cancels (∂φ/∂x → φ/x)."""

    def test_partial_display_substitutes_correctly(self):
        nb.start_engine(title="test_bp9", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        try:
            src = eng.register_expression("'diff(phi,x,1)+w")
            _, value = eng.run_operation("substitute", {
                "expr": src.display_id,
                "lhs":  r"\frac{\partial phi}{\partial x}",
                "rhs":  "0",
            })
            assert value.replace(" ", "") == "w"
        finally:
            nb.stop_engine()
