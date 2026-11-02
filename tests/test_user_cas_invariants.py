"""The user⇄CAS boundary invariants, pinned over a real corpus (docs/32 Phase 5).

R0–R7 in docs/developer_manuals/32 are what the repair is FOR; the phases that
established them are pinned by their own files.  This one closes the loop by
checking the properties that must hold for EVERY expression, against the saved
sessions in ``sessions/*.json`` — real research data, not hand-picked cases.

Two lessons from the earlier phases shape how these are written:

* **Ask the CAS, not the strings.**  AUDIT-POW-DENOM (``k/r^2`` → ``k*sqrt(r)``)
  and the ``^`` associativity bug both survived string round-trip checks and only
  showed up as a differential ``ratsimp(input - lowered) = 0``.  Equations and
  matrices subtract component-wise, so "zero" means every component is zero.

* **Assert the VALUE.**  The bug this whole effort repairs returned ``success``.
"""
import glob
import re

import pytest

from notebook.engine import NotebookEngine
from notebook.models.ir import IndexedSymbol, Symbol
from notebook.runtime.shared_parser import UnknownNotationError

_SESSION_FILES = sorted(glob.glob("sessions/*.json"))
#: ``0``, ``0 = 0``, ``[[0,0],[0,0]]``, ``matrix([0,0],[0,0])`` — a difference is
#: zero when every component of it is.
_ZERO_RE = re.compile(r"[\s0\[\],=;()]*|matrix\([\s0\[\],]*\)")


def _is_zero(cas_value: str) -> bool:
    return bool(_ZERO_RE.fullmatch((cas_value or "").strip()))


def _has_undeclared_index(ir, ctx) -> bool:
    """True when the IR contains an IndexedSymbol whose base is undeclared.

    Lowering such a symbol DELIBERATELY changes meaning (``V_i`` is not Maxima's
    ``V_i`` but ``V(i)``), so R2 does not apply to it — see the caveat on R2 in
    docs/32 §3."""
    seen: set[int] = set()
    found = False

    def walk(node) -> None:
        nonlocal found
        if node is None or found or id(node) in seen:
            return
        seen.add(id(node))
        if (isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol)
                and ctx.get(node.base.name) is None):
            found = True
            return
        for f in getattr(type(node), "model_fields", ()) or ():
            v = getattr(node, f, None)
            if isinstance(v, (list, tuple)):
                for item in v:
                    if hasattr(type(item), "model_fields"):
                        walk(item)
            elif hasattr(type(v), "model_fields"):
                walk(v)

    walk(ir)
    return found


@pytest.fixture(scope="module")
def corpus():
    """(session file, engine, expression canonical) for every saved expression.

    Engines are loaded through ``NotebookEngine.load`` on purpose: building a
    SymbolContext by hand drops the declarations and makes the numbers lie (that
    mistake cost an afternoon while measuring §2.6)."""
    engines, rows = [], []
    for path in _SESSION_FILES:
        try:
            e = NotebookEngine.load(path)
            e.start()
        except Exception:                      # a file we cannot load is not
            continue                           # this test's business
        engines.append(e)
        for expr in e.expressions.values():
            canonical = (expr.canonical or "").strip()
            if canonical:
                rows.append((path, e, canonical))
    yield rows
    for e in engines:
        e.close()


def test_the_corpus_is_not_empty(corpus):
    """Guard: a silently empty corpus would make every test below vacuous."""
    assert len(corpus) >= 40, f"only {len(corpus)} expressions found"


class TestR1Idempotence:
    """``lower(lower(x)) == lower(x)``.

    Not cosmetic: a non-idempotent boundary grows the expression every time it
    crosses.  Measured here — ``a - 4*g`` gained a ``1 *`` per pass because the
    generator's ``-1`` shortcut only fired at arity 2.
    """

    def test_every_corpus_expression_is_stable(self, corpus):
        drifting = []
        for path, e, canonical in corpus:
            try:
                once = e._handler.to_backend_code(canonical)
            except UnknownNotationError:
                continue
            if once.status != "lowered":
                continue
            twice = e._handler.to_backend_code(once.code)
            if twice.code != once.code:
                drifting.append(f"{path}: {canonical[:40]!r}\n"
                                f"    1st {once.code[:70]!r}\n"
                                f"    2nd {twice.code[:70]!r}")
        assert not drifting, "lowering is not idempotent for:\n" + "\n".join(drifting)

    @pytest.mark.parametrize("expr", [
        "a - b", "a - 4*g", "a - b*c", "a - (b+c)", "-2*x*y", "a - b - c",
        "-x", "k/r^2", "a*(-b)*c",
    ])
    def test_subtraction_does_not_accumulate_factors(self, expr):
        with NotebookEngine(title="r1") as e:
            first = e._handler.to_backend_code(expr).code
            for _ in range(3):
                assert e._handler.to_backend_code(first).code == first


class TestR2MeaningPreserved:
    """``ratsimp(raw - lower(raw)) == 0`` in a live CAS.

    Skipped for expressions containing an UNDECLARED indexed symbol: lowering
    those changes meaning on purpose (docs/32 §3, R2's caveat).
    """

    def test_every_applicable_corpus_expression(self, corpus):
        changed = []
        for path, e, canonical in corpus:
            if e.session.primary_backend != "maxima":
                continue
            try:
                lowered = e._handler.to_backend_code(canonical)
            except UnknownNotationError:
                continue
            if lowered.status != "lowered":
                continue
            if _has_undeclared_index(e._handler.parse_ir(canonical), e.symbol_context):
                continue
            diff = e._adapter.execute(f"ratsimp(({canonical}) - ({lowered.code}))")
            if diff.ok and not _is_zero(diff.value):
                changed.append(f"{path}: {canonical[:40]!r} → {lowered.code[:40]!r} "
                               f"(difference {diff.value[:40]!r})")
        assert not changed, "lowering changed the meaning of:\n" + "\n".join(changed)


class TestR4NoSilentPassThrough:
    """A lowering that cannot be done must fail the operation, never reach the
    CAS as raw text.  The corpus should need no escape hatch at all."""

    def test_nothing_in_the_corpus_is_refused(self, corpus):
        refused = []
        for path, e, canonical in corpus:
            try:
                r = e._handler.to_backend_code(canonical)
            except UnknownNotationError as exc:
                refused.append(f"{path}: {canonical[:40]!r} — {exc}")
                continue
            if r.status == "refused":
                refused.append(f"{path}: {canonical[:40]!r} — {r.reason[:50]}")
        assert not refused, (
            "expressions the boundary cannot lower (each is an operation that "
            "now fails for the researcher):\n" + "\n".join(refused))


class TestR3DisplayRoundTrip:
    """What is displayed must come back as the same expression.

    Curated rather than corpus-wide, because the known gaps below would otherwise
    hide a NEW regression behind a count.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        with NotebookEngine(title="r3") as e:
            for src in ("a*V_i + b*W_j", "omega_0*q", "A_{i,j}"):
                e.register_expression(src)
            yield e

    @pytest.mark.parametrize("canonical", [
        "a*x+b",
        "a - 4*g",
        "k/r^2",                  # AUDIT-POW-DENOM's shape
        "x^2/y^2",
        "1/2*x",
        "sin(x)*cos(y)",
        "sqrt(4*w^2-4*g^2)",
        "a*V(i)+b*W(j)",          # Phase 4: indexed display restored
        "omega(0)^2*q",
        "A(i,j)+b",
        "y = a*x+b",
        "%e^(-a*t)",
        "[[1,2],[3,4]]",
        # AUDIT-DERIV-DISPLAY-RT: a derivative whose operand carries arguments.
        # tex1 writes them as `f\left(x , y\right)` and only the bare `(` form
        # was read, so the arguments detached into a product.
        "'diff(f(x,y),x,1)",
        "'diff(f(x,y),x,1,y,1)",
        "'diff(g(x,y,z),y,1)+h(x)",
    ])
    def test_display_then_reinput(self, engine, canonical):
        h, ctx = engine._handler, engine.symbol_context
        raw = engine._adapter.execute(f"tex1({canonical})").value
        display = h.postprocess_result_latex(canonical, raw, ctx)
        back = h.to_backend_code(h.latex_to_canonical(display)).code
        diff = engine._adapter.execute(f"ratsimp(({canonical}) - ({back}))")
        assert diff.ok and _is_zero(diff.value), (
            f"{canonical!r} displays as {display!r}, which comes back as "
            f"{back!r} (difference {diff.value!r})")

    @pytest.mark.parametrize("canonical,why", [
        ("matrix([a,b],[c,d])",
         "tex1 emits a \\pmatrix macro wrapper that latex_to_canonical does not "
         "read back"),
        ("abs(x)",
         "displays as |x|, which is deliberately ambiguous on re-input "
         "(abs vs determinant) and raises for disambiguation — by design"),
    ])
    def test_known_gaps_are_still_gaps(self, engine, canonical, why):
        """Pinned so that fixing one is a visible, deliberate change — and so a
        NEW shape breaking cannot hide among them."""
        h, ctx = engine._handler, engine.symbol_context
        try:
            raw = engine._adapter.execute(f"tex1({canonical})").value
            display = h.postprocess_result_latex(canonical, raw, ctx)
            back = h.to_backend_code(h.latex_to_canonical(display)).code
            diff = engine._adapter.execute(f"ratsimp(({canonical}) - ({back}))")
            round_trips = diff.ok and _is_zero(diff.value)
        except UnknownNotationError:
            round_trips = False
        assert not round_trips, (
            f"{canonical!r} now round-trips — remove it from the known-gap list "
            f"(recorded reason: {why})")
