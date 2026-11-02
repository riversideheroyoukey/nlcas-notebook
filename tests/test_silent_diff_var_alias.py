"""SILENT-DIFF-VAR-ALIAS: differentiating a raw (non-E-id) formula that
reuses a symbol already solved for elsewhere in the session must not
silently substitute that symbol's solved VALUE into the formula before
differentiating.

Found via the lotka_volterra_stability practice-loop scenario's English
translation (2026-07-05): after `Solve E1 and E2 for x and y` registers
x=0 (among other roots) in lhs_alias_map, a later inline-formula
differentiate command containing "x" as a free variable of the formula
(not the E-id x was aliased to) got x replaced by its solved value 0
before differentiation ran — "differentiate a*x-b*x*y with respect to x"
silently returned 0 instead of the correct a-b*y, with no error.

Root cause: English input has no natural-language cue marking where an
inline formula ends and the instruction begins (unlike Japanese, where
_FORMULA_OP_RE splits on the ASCII/non-CJK script transition), so the raw
formula fell through to the generic pipeline's "known symbol -> E-id"
substitution pass (2a-sym in run_interpret_pipeline), which resolved the
formula's own free variable via lhs_alias_map. Fixed by adding a dedicated
English recognizer (_DIFF_INLINE_FORMULA_EN_RE in ui/interpret_pipeline.py)
that splits "differentiate/derivative of <formula> ... with respect to
<var>" using the explicit keyword boundary, registers the formula first
(bypassing symbol-alias substitution, mirroring the Japanese path), and
only then differentiates the resulting E-id.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, value):
    tok = InputToken(type="text", value=value)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestInlineFormulaDifferentiateAfterSolve:
    def _engine(self):
        nb.start_engine(title="silent_diff_var_alias", mode="standard",
                        language="en", domains=["mathematical_biology"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["mathematical_biology"],
                            primary_backend="maxima")
        return eng, sup

    def test_partial_derivative_wrt_a_solved_variable_is_not_corrupted(self):
        eng, sup = self._engine()
        try:
            _run(eng, sup, "a*x - b*x*y = 0")
            _run(eng, sup, "d*x*y - c*y = 0")
            _run(eng, sup, "Solve E1 and E2 for x and y")   # registers x=0, x=c/d, ...
            r = _run(eng, sup,
                     "Take the partial derivative of a*x - b*x*y with respect to x")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical == "a-b*y"
        finally:
            nb.stop_engine()

    def test_other_free_variable_in_the_same_formula_also_stays_symbolic(self):
        """y is also solved-for (y=a/b) but is not the diff variable here —
        it must stay free in the formula too, not get replaced by a/b."""
        eng, sup = self._engine()
        try:
            _run(eng, sup, "a*x - b*x*y = 0")
            _run(eng, sup, "d*x*y - c*y = 0")
            _run(eng, sup, "Solve E1 and E2 for x and y")
            r = _run(eng, sup,
                     "Take the partial derivative of a*x - b*x*y with respect to y")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical == "-(b*x)"
        finally:
            nb.stop_engine()

    def test_all_four_jacobian_entries_match_expected(self):
        """Reproduces the full S-q/lotka_volterra Jacobian derivation."""
        eng, sup = self._engine()
        try:
            _run(eng, sup, "a*x - b*x*y = 0")
            _run(eng, sup, "d*x*y - c*y = 0")
            _run(eng, sup, "Solve E1 and E2 for x and y")
            results = []
            for cmd in (
                "Take the partial derivative of a*x - b*x*y with respect to x",
                "Take the partial derivative of a*x - b*x*y with respect to y",
                "Take the partial derivative of d*x*y - c*y with respect to x",
                "Take the partial derivative of d*x*y - c*y with respect to y",
            ):
                r = _run(eng, sup, cmd)
                assert r["status"] == "completed", (cmd, r)
                results.append(eng.expressions[r["expr_id"]].canonical)
            assert results == ["a-b*y", "-(b*x)", "d*y", "d*x-c"]
        finally:
            nb.stop_engine()


class TestAliasSubstitutionIsNotRecorded:
    r"""H1-ALIAS-EID-LEAK: the same 2a-sym pass, seen from the other side.

    Rewriting a known symbol into its E-ID is for the SUPERVISOR's benefit —
    it stops the LLM re-resolving "H" to some derived expression.  But the
    pure-formula registration path a few blocks later registered the REWRITTEN
    text, so what got written down was an internal identifier the researcher
    never typed: after "G = a + b", typing "G^2 + 1" recorded `E1^2 + 1` in
    both the display and the canonical (measured 2026-08-01 — found in the
    S-c handoff measurement as `xi_k^2 + E17^2`, docs/34 §6.2).

    That breaks "canonical is the user's own writing" (CLAUDE.md) and shows a
    CAS-internal id in the timeline, the saved session and the TeX export.  It
    also makes the record depend on E1 staying put: edit or fork E1 and the
    stored expression quietly means something else.

    Only the RECORDING changed — the supervisor still receives the E-ID form
    (pinned below), and the value is unaffected because `G` resolves through
    the H-1 alias anyway (also pinned below).
    """

    def _engine(self, lang="ja"):
        nb.start_engine(title="alias_eid_leak", mode="standard", language=lang,
                        domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        return eng, sup

    def test_bare_expression_keeps_the_symbol_jp(self):
        eng, sup = self._engine("ja")
        try:
            _run(eng, sup, "G = a + b")
            _run(eng, sup, "G^2 + 1")
            e2 = eng.expressions["E2"]
            assert e2.canonical == "G^2 + 1"
            assert e2.latex == "G^2 + 1"
        finally:
            nb.stop_engine()

    def test_bare_expression_keeps_the_symbol_en(self):
        eng, sup = self._engine("en")
        try:
            _run(eng, sup, "G = a + b")
            _run(eng, sup, "G^2 + 1")
            assert eng.expressions["E2"].canonical == "G^2 + 1"
        finally:
            nb.stop_engine()

    def test_equation_whose_lhs_is_not_a_symbol(self):
        """`1 = …` cannot be an H-1 definition, so it falls through to the
        pure-formula path — the exact shape the S-c gap equation had."""
        eng, sup = self._engine("ja")
        try:
            _run(eng, sup, "G = a + b")
            _run(eng, sup, "1 = G^2 + 1")
            e2 = eng.expressions["E2"]
            assert "E1" not in e2.canonical
            assert e2.canonical == "1 = G^2 + 1"
        finally:
            nb.stop_engine()

    def test_the_value_still_resolves_through_the_alias(self):
        """Recording the user's spelling must not cost the resolution."""
        eng, sup = self._engine("ja")
        try:
            _run(eng, sup, "G = a + b")
            _run(eng, sup, "G^2 + 1")
            r = _run(eng, sup, "E2 を展開して")
            assert r["status"] == "completed", r
            got = eng.expressions[r["expr_id"]].canonical.replace(" ", "")
            assert got == "b^2+2*a*b+a^2+1", got
        finally:
            nb.stop_engine()

    def test_supervisor_still_sees_the_eid_form(self):
        """The rewrite itself must stay — it is why 2a-sym exists."""
        eng, sup = self._engine("ja")
        seen: list[str] = []
        _orig = sup.interpret

        def _spy(text, *a, **kw):
            seen.append(text)
            return _orig(text, *a, **kw)

        sup.interpret = _spy
        try:
            _run(eng, sup, "G = a + b")
            _run(eng, sup, "G を展開して")
            assert seen, "supervisor was never called"
            assert any("E1" in t for t in seen), seen
        finally:
            nb.stop_engine()
