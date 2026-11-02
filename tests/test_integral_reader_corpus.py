r"""MEASURE-READ-TWICE: the two measure readers, on one corpus of spellings.

An integral written by a researcher is claimed by ONE of two routes, and which
one is decided before either has seen the mathematics:

* **①** ``ui.interpret_integral._try_integral_input`` reads the RAW TEXT with a
  regex (``integral_form.parse_latex_definite_integral``) and, when it can find
  explicit bounds and matching differentials, registers a display-only integral
  form.  The CAS is not called at input; it is called later, from the descriptor.
* **②** everything ① declines falls through to the general pipeline, where the
  measure is read from the IR (``sum_integral_prefix._read_measure``).

Both answer "which variable is this integrated over, between which limits, of
what" — so a spelling either route can claim must reach the CAS as the same
integral.  It did not, and the failures were silent (measured 2026-08-16,
through the interceptor):

    \int_0^{2\pi} r \, d\theta              → `integrate((r), theta, 0, 2\pi)`
                                              → `_res_`, the ADAPTER'S SCRATCH
                                                NAME, registered as a success
    \int_{-\infty}^{\infty} e^{-x^2} \, dx  → `sqrt(%pi)*erf(infty)` — the
                                              Gaussian integral, wrong, success
    \int_0^1 \sum_i f_i(x) \, dx            → `sum_i*'integrate(f_i(x),x,0,1)`,
                                              a bare symbol named `sum_i`

① is not a mistake — a range-bearing definite integral is a first-class
display-only object by design (S-o, docs/scenario_coordinate_jacobian) — and the
routes look at genuinely different things (surface text vs IR), so this file
does not assert that one of them goes away.  It asserts that **where they
overlap they agree**, and it asks the CAS rather than comparing strings: the two
routes legitimately spell the same integral differently (`integrate((x^2), …)`
vs `integrate(x^2, …)`), and two earlier silent-wrong bugs at this boundary
survived string round-trip checks (docs/32 §Phase 5).

One entry per MEANING, several spellings each — the notation-corpus rule.  A
family whose spellings disagree is the defect, whichever route is right.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.integral_form import (
    build_eval_command, parse_latex_definite_integral,
)
from notebook.runtime.maxima_adapter import MaximaAdapter
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler


@pytest.fixture(scope="module")
def handler():
    h = MaximaBackendHandler(None)
    h._context = SymbolContext()
    return h


@pytest.fixture(scope="module")
def cas():
    a = MaximaAdapter()
    a.start()
    yield a
    a.close()


def route1(h, latex: str) -> "str | None":
    """The CAS command route ① builds, or None when ① declines the spelling.

    Mirrors ``_try_integral_input`` + ``evaluate_integral_form``: bounds through
    ``normalize_bound``, integrand through ``to_backend_code``.  Kept in step
    with those by ``test_integral_workflow`` / ``test_change_vars_intercept``,
    which drive the real pipeline; this function exists so the corpus can be
    swept without a live engine per row.
    """
    desc = parse_latex_definite_integral(latex)
    if desc is None:
        return None
    bounds = [[v, h.normalize_bound(lo), h.normalize_bound(hi)]
              for v, lo, hi in desc["bounds"]]
    lowered = h.to_backend_code(h.latex_to_canonical(desc["integrand_latex"]))
    if not lowered.ok:
        return None
    return build_eval_command(lowered.code, bounds)


def route2(h, latex: str) -> "str | None":
    """The CAS command route ② builds, or None when it cannot read the text."""
    try:
        r = h.to_backend_code(h.latex_to_canonical(latex))
    except Exception:
        return None
    return r.code if r.ok else None


def _value(cas, command: str) -> "str | None":
    r = cas.execute(command)
    if not (r.ok and r.value and not r.value.startswith("ERROR")):
        return None
    return r.value.strip()


#: (family, [spellings]).  One family = one integral; every spelling in it is a
#: way a researcher (or this system's own display) writes that same integral.
CORPUS: list[tuple[str, list[str]]] = [
    ("power, unit interval", [
        r"\int_0^1 x^2 \, dx",
        r"\int_0^1 x^2 dx",
        r"\int_{0}^{1} x^2 \, dx",
        r"\int_0^1 x^2 \, \mathrm{d}x",
    ]),
    ("greek variable", [
        r"\int_0^{2\pi} r \, d\theta",
        r"\int_{0}^{2\pi} r \, d\theta",
    ]),
    ("pi bound", [
        r"\int_0^{\pi} \sin\theta \, d\theta",
        r"\int_{0}^{\pi} \sin\theta \, d\theta",
    ]),
    ("gaussian, infinite bounds", [
        r"\int_{-\infty}^{\infty} e^{-x^2} \, dx",
        r"\int_{-\infty}^{+\infty} e^{-x^2} \, dx",
    ]),
    ("polar jacobian (this system's own display)", [
        r"\int_0^{2\pi} \int_0^1 r \, dr \, d\theta",
        r"\int_{0}^{2\pi} \int_{0}^{1} r \, dr \, d\theta",
    ]),
    ("bracketed integrand", [
        r"\int_0^1 (x+1) \, dx",
        r"\int_0^1 \left(x+1\right) \, dx",
    ]),
    ("rational integrand", [
        r"\int_0^1 \frac{1}{1+x^2} \, dx",
    ]),
    ("applied function", [
        r"\int_0^1 f(t) \, dt",
    ]),
    ("symbolic bounds", [
        r"\int_{a}^{b} x^2 \, dx",
    ]),
    ("iterated, two variables", [
        r"\int_0^1 \int_0^2 x y \, dx \, dy",
    ]),
    # The two families below exist because the integrand needs LOWERING, not
    # just converting: without them a mutation that hands `latex_to_canonical`
    # output straight to the CAS passed every other row here (measured, and it
    # is the defect this file was written for — `\sum_i` reaching Maxima as a
    # bare symbol `sum_i`).  A corpus of spellings that were already going to
    # work cannot catch it.
    ("sum under the integral", [
        r"\int_0^1 \sum_{i=1}^{3} f_i(x) \, dx",
        r"\int_{0}^{1} \sum_{i=1}^{3} f_i(x) \, dx",
    ]),
    ("indexed integrand", [
        r"\int_0^1 V_i x \, dx",
        r"\int_{0}^{1} V_i x \, dx",
    ]),
    # `\iint` is an ABBREVIATION of `\int\int`, and only route ① knew it: the
    # unbounded form — which ① declines — reached the CAS as
    # `iint * x * y * dx * dy`, a product with an undefined symbol, ∫ gone.
    ("double integral, abbreviated command", [
        r"\iint x y \, dx \, dy",
        r"\int \int x y \, dx \, dy",
    ]),
    ("triple integral, abbreviated command", [
        r"\iiint x y z \, dx \, dy \, dz",
        r"\int \int \int x y z \, dx \, dy \, dz",
    ]),
]


class TestRoutesAgree:
    r"""Where both routes claim a spelling, the CAS must answer the same thing."""

    @pytest.mark.parametrize("family,spellings", CORPUS,
                             ids=[f for f, _ in CORPUS])
    def test_both_routes_give_the_same_value(self, handler, cas, family, spellings):
        for latex in spellings:
            c1, c2 = route1(handler, latex), route2(handler, latex)
            if c1 is None or c2 is None:
                continue                       # only one route claims it
            v1, v2 = _value(cas, c1), _value(cas, c2)
            assert v1 is not None, f"{latex}: route ① command failed: {c1}"
            assert v2 is not None, f"{latex}: route ② command failed: {c2}"
            diff = _value(cas, f"ratsimp(({v1}) - ({v2}))")
            assert diff == "0", (
                f"{family} / {latex}\n  ① {c1} = {v1}\n  ② {c2} = {v2}")

    @pytest.mark.parametrize("family,spellings", CORPUS,
                             ids=[f for f, _ in CORPUS])
    def test_every_spelling_of_one_meaning_agrees(self, handler, cas,
                                                  family, spellings):
        """Within a family the spellings differ only on the page."""
        values = []
        for latex in spellings:
            cmd = route1(handler, latex) or route2(handler, latex)
            assert cmd is not None, f"{family}: no route reads {latex}"
            v = _value(cas, cmd)
            assert v is not None, f"{family} / {latex}: CAS refused {cmd}"
            values.append((latex, v))
        first_latex, first = values[0]
        for latex, v in values[1:]:
            diff = _value(cas, f"ratsimp(({first}) - ({v}))")
            assert diff == "0", (
                f"{family}: {first_latex} = {first} but {latex} = {v}")


class TestNoRawLatexReachesTheCas:
    r"""A command carrying a backslash is LaTeX that was never converted.

    This is the shape all three measured defects had — `0, 2\pi` in the bounds,
    `\sum_i` in the integrand — and it is cheap enough to check on every row.
    """

    @pytest.mark.parametrize("latex",
                             [s for _f, sp in CORPUS for s in sp])
    def test_no_backslash_in_either_command(self, handler, latex):
        for name, cmd in (("①", route1(handler, latex)),
                          ("②", route2(handler, latex))):
            if cmd is not None:
                assert "\\" not in cmd, f"{name} sends raw LaTeX: {cmd}"


class TestClaimBoundary:
    r"""WHICH route claims a spelling — recorded, so a change to it is visible.

    Not a rule about what ought to claim what: it is the map the ticket asked
    for, and a diff here means the boundary moved.
    """

    @pytest.mark.parametrize("latex,claimed_by_one", [
        (r"\int_0^1 x^2 \, dx",                            True),
        (r"\int_0^{2\pi} r \, d\theta",                    True),
        (r"\int_0^1 \int_0^2 x y \, dx \, dy",             True),
        # 2026-08-16: these two MOVED (INTEGRAL-CLAIM-ASYMMETRY).  ①'s limits
        # were read by a regex that could not nest braces, so a bound as
        # ordinary as `\frac{\pi}{2}` or `\sqrt{2}` decided the OBJECT KIND —
        # claimed, it is an integral form that can be transformed and
        # evaluated; declined, an ordinary expression that cannot.  Reading the
        # limits with balanced braces is ①'s own repair, not a second copy of
        # ②'s IR walk (which would need the CAS to render `\frac{\pi}{2}`
        # back, and ① is defined by not calling the CAS at input).
        (r"\int_0^{\frac{\pi}{2}} \sin x \, dx",            True),
        (r"\int_0^{\sqrt{2}} x \, dx",                     True),
        (r"\int x^2 \, dx",                                False),  # no bounds
        # Still declined, and deliberately: these are about the MEASURE, not the
        # limits, and reading them needs ②'s IR walk.  Their answers are correct
        # (the corpus above checks that); what they lack is object status.
        (r"\int_0^1 dx \, x^2",                            False),  # leading measure
        (r"\int_{-\infty}^{\infty} \frac{d\omega}{2\pi} G(\omega)", False),
    ])
    def test_route1_claim(self, handler, latex, claimed_by_one):
        assert (route1(handler, latex) is not None) is claimed_by_one
