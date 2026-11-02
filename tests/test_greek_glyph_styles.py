r"""One quantity, one glyph: φ/ε/θ come back in the spelling they were written in.

PHI-GLYPH-SPLIT.  Three Greek letters have two LaTeX spellings for two
different glyphs, and the CAS answers in the one the researcher may not have
used — Maxima's ``tex1`` returns ``\varphi`` for ``phi``.  The converter maps
BOTH spellings to one canonical, so the canonical cannot say which was written,
and the published Coulomb-gauge example carried the same potential in two
glyphs on adjacent lines::

    \phi_{0} = \phi                                                ← written
    \frac{\partial \phi_{0}}{\partial x} = \frac{d}{d x} \varphi    ← returned

Not an error — an unevenness, which is why it survived: it is only visible in
the paper.

**The table is a measurement, not a guess.**  All 34 Greek names were put
through `tex1` (2026-08-16); exactly three come back in a variant, and the
ticket had only named one of them.  ε and θ split the same way and nobody had
looked.

The rule is `spin_display_styles`' — **first use is the declaration** — and it
restores what was WRITTEN rather than picking a favourite: a researcher who
writes ``\varphi`` keeps ``\varphi``, and a session that never writes one of
these letters records nothing at all.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.indexed_display import (
    GREEK_GLYPH_VARIANTS, record_greek_glyphs, restore_greek_glyphs,
)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestTheSpellingSurvivesTheRoundTrip:
    @pytest.mark.parametrize("written", [
        r"\phi + 1", r"\varphi + 1",
        r"\epsilon + 1", r"\varepsilon + 1",
        r"\theta + 1", r"\vartheta + 1",
    ], ids=["phi", "varphi", "epsilon", "varepsilon", "theta", "vartheta"])
    def test_the_result_uses_what_was_written(self, written):
        nb.start_engine(title="glyph", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup, written, "math")
            out = _run(eng, sup, f"{_eid(r)} を簡約して")
            assert out["status"] == "completed", out
            latex = eng.expressions[_eid(out)].latex
            command = written.split()[0]
            assert command in latex, f"{written} → {latex}"
        finally:
            nb.stop_engine()


class TestTheTableIsTheMeasurement:
    def test_exactly_the_three_letters_maxima_rewrites(self):
        """If Maxima ever rewrites a fourth, this fails and the table is wrong."""
        from notebook.runtime.latex_converter import TEX_GREEK_TO_NAME
        from notebook.runtime.maxima_adapter import MaximaAdapter
        cas = MaximaAdapter()
        cas.start()
        try:
            split = set()
            for name in sorted(set(TEX_GREEK_TO_NAME.values())):
                got = (cas.execute(f"tex1({name})").value or "").strip()
                if got not in ("\\" + name, "{\\it %s}" % name):
                    split.add(name)
        finally:
            cas.close()
        assert split == set(GREEK_GLYPH_VARIANTS), split


class TestItRecordsOnlyWhatWasWritten:
    def test_first_use_wins(self):
        ctx = SymbolContext()
        record_greek_glyphs(r"\varphi + 1", ctx)
        record_greek_glyphs(r"\phi + 2", ctx)          # later, and ignored
        assert ctx.greek_glyph_styles["phi"] == r"\varphi"

    def test_the_variant_is_not_read_as_the_plain_form(self):
        r"""The two commands do not overlap — the BACKSLASH separates them, so
        ``\varphi`` does not contain ``\phi``.

        Written first as "the longer spelling must be matched first", which a
        mutation then showed to be untrue: reversing the order changed nothing.
        The test is kept because the PROPERTY matters (a variant must never be
        recorded as the plain form); the claim about ordering was wrong and the
        comment in the source now says so."""
        ctx = SymbolContext()
        record_greek_glyphs(r"\varphi", ctx)
        assert ctx.greek_glyph_styles["phi"] == r"\varphi"

    def test_a_letter_never_written_is_never_touched(self):
        ctx = SymbolContext()
        record_greek_glyphs(r"\phi + x", ctx)
        assert "epsilon" not in ctx.greek_glyph_styles
        # …and the CAS spelling of the others stands unchanged
        out = restore_greek_glyphs(r"\varphi + \varepsilon + \vartheta", ctx)
        assert out == r"\phi + \varepsilon + \vartheta"

    def test_nothing_recorded_means_nothing_rewritten(self):
        ctx = SymbolContext()
        src = r"\varphi + \varepsilon + \vartheta"
        assert restore_greek_glyphs(src, ctx) == src

    @pytest.mark.parametrize("name", sorted(GREEK_GLYPH_VARIANTS))
    def test_both_directions_are_reachable(self, name):
        plain, variant = GREEK_GLYPH_VARIANTS[name]
        for written, other in ((plain, variant), (variant, plain)):
            ctx = SymbolContext()
            record_greek_glyphs(written, ctx)
            assert restore_greek_glyphs(other, ctx) == written

    def test_a_longer_command_starting_with_the_same_letters_is_left_alone(self):
        r"""``\thetahat`` is not ``\theta`` — the boundary matters."""
        ctx = SymbolContext()
        record_greek_glyphs(r"\theta", ctx)
        assert restore_greek_glyphs(r"\varthetaX", ctx) == r"\varthetaX"
