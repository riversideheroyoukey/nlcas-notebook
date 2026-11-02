"""S-o end-to-end: display-only definite integral → change of variables → evaluate.

Exercises the full pipeline (no LLM): register an unevaluated ∫∫ object, rewrite it
under a coordinate change (integrand·J + new bounds), and evaluate to a number.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["electromagnetism"], primary_backend="maxima")


def _run(eng, sup, cmd):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=cmd)]), eng, sup))


class TestPolarDiskWorkflow:
    def test_disk_change_then_evaluate(self):
        nb.start_engine(title="s_o flow", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            # (1) display-only region integral over the unit disk
            r1 = _run(eng, sup, "x^2+y^2<=1 の領域で x^2+y^2 を定積分")
            assert r1["status"] == "registered", r1
            e1 = r1["expr_id"]
            assert eng.expressions[e1].integral_descriptor["region"]["kind"] == "disk"
            assert "\\iint" in eng.expressions[e1].latex
            # (2)+(3) change of variables → new bounds auto-derived (polar/disk)
            r2 = _run(eng, sup,
                      f"{e1} を x,y から r,theta へ "
                      "x=r*cos(theta), y=r*sin(theta) で変数変換して")
            assert r2["status"] == "registered", r2
            e2 = r2["expr_id"]
            desc = eng.expressions[e2].integral_descriptor
            assert desc["bounds"] == [["theta", "0", "2*%pi"], ["r", "0", "1"]]
            assert desc["integrand_canonical"].replace(" ", "") == "r^3"
            assert eng.expressions[e2].latex == \
                r"\int_{0}^{2\pi} \int_{0}^{1} r^3 \, dr \, d\theta"
            # (4) evaluate → π/2
            r3 = _run(eng, sup, "この定積分を評価して")
            assert r3["status"] == "completed", r3
            assert r3["canonical"].replace(" ", "") in ("%pi/2", "(%pi)/2")
            assert "= \\frac{\\pi}{2}" in eng.expressions[r3["expr_id"]].latex
        finally:
            nb.stop_engine()


class _StubLLM:
    """Minimal llm_client stub returning a fixed JSON bounds list."""
    def __init__(self, text):
        self._text = text
        self.calls = 0
    def chat(self, system, user, **kw):
        self.calls += 1
        from notebook.runtime.llm_client import ChatResponse
        return ChatResponse(ok=True, text=self._text)


class TestBoundResolutionOrder:
    def test_llm_fallback_when_no_table_or_user_bounds(self):
        """Non-standard transform, no user bounds → the LLM proposal is used."""
        nb.start_engine(title="s_o llm", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        sup._llm = _StubLLM('[["u","0","1"],["v","0","2"]]')
        try:
            r1 = _run(eng, sup, r"\int_{0}^{1} \int_{0}^{1} (x+y) \, dx \, dy")
            r2 = _run(eng, sup,
                      f"{r1['expr_id']} を x,y から u,v へ "
                      "x=2*u+v, y=u+3*v で変数変換して")
            assert r2["status"] == "registered", r2
            assert sup._llm.calls == 1
            assert eng.expressions[r2["expr_id"]].integral_descriptor["bounds"] == \
                [["u", "0", "1"], ["v", "0", "2"]]
        finally:
            nb.stop_engine()

    def test_user_bounds_override_even_with_llm(self):
        """Explicit user bounds win and the LLM is not consulted."""
        nb.start_engine(title="s_o ovr", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        sup._llm = _StubLLM('[["u","9","9"],["v","9","9"]]')
        try:
            r1 = _run(eng, sup, r"\int_{0}^{1} \int_{0}^{1} (x+y) \, dx \, dy")
            r2 = _run(eng, sup,
                      f"{r1['expr_id']} を x,y から u:0..1, v:0..1 へ "
                      "x=2*u+v, y=u+3*v で変数変換して")
            assert eng.expressions[r2["expr_id"]].integral_descriptor["bounds"] == \
                [["u", "0", "1"], ["v", "0", "1"]]
            assert sup._llm.calls == 0
        finally:
            nb.stop_engine()

    def test_table_beats_llm_for_standard_transform(self):
        """Standard polar/disk → auto table is used, LLM not consulted."""
        nb.start_engine(title="s_o tbl", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        sup._llm = _StubLLM('[["x","9","9"],["y","9","9"]]')
        try:
            r1 = _run(eng, sup, "x^2+y^2<=1 の領域で x^2+y^2 を定積分")
            r2 = _run(eng, sup,
                      f"{r1['expr_id']} を x,y から r,theta へ "
                      "x=r*cos(theta), y=r*sin(theta) で変数変換して")
            assert eng.expressions[r2["expr_id"]].integral_descriptor["bounds"] == \
                [["theta", "0", "2*%pi"], ["r", "0", "1"]]
            assert sup._llm.calls == 0
        finally:
            nb.stop_engine()


class TestLatexRectangleWorkflow:
    def test_latex_input_user_bounds_evaluate(self):
        nb.start_engine(title="s_o rect", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            # (1) LaTeX double integral over the unit square
            r1 = _run(eng, sup, r"\int_{0}^{1} \int_{0}^{1} (x^2+y^2) \, dx \, dy")
            assert r1["status"] == "registered", r1
            e1 = r1["expr_id"]
            assert eng.expressions[e1].integral_descriptor["bounds"] == \
                [["y", "0", "1"], ["x", "0", "1"]]
            # evaluate the original directly: ∫₀¹∫₀¹ (x²+y²) = 2/3
            r2 = _run(eng, sup, f"{e1} を評価して")
            assert r2["status"] == "completed", r2
            assert r2["canonical"].replace(" ", "") == "2/3"
        finally:
            nb.stop_engine()

    def test_general_jacobian_user_bounds(self):
        nb.start_engine(title="s_o gen", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r1 = _run(eng, sup, r"\int_{0}^{1} \int_{0}^{1} (x+y) \, dx \, dy")
            e1 = r1["expr_id"]
            # general linear transform with USER-supplied new bounds
            r2 = _run(eng, sup,
                      f"{e1} を x,y から u:0..1, v:0..1 へ "
                      "x=2*u+v, y=u+3*v で変数変換して")
            assert r2["status"] == "registered", r2
            desc = eng.expressions[r2["expr_id"]].integral_descriptor
            assert desc["bounds"] == [["u", "0", "1"], ["v", "0", "1"]]
            # integrand x+y → (3u+4v), × det J = 5 → 15u+20v
            assert desc["integrand_canonical"].replace(" ", "") == "20*v+15*u"
        finally:
            nb.stop_engine()


class TestDefiniteIntegralAcceptsTexSpelling:
    r"""The researcher writes back what the notebook DISPLAYED, which is TeX
    (S-C-MEASURE-RESIDUALS, measured during the S-c handoff, docs/34 §6.2).

    Two spellings used to fall out of the rule-based path:

    * the integration VARIABLE as ``\xi`` — the pattern's ``[A-Za-z_]\w*``
      could not match it, so the command missed the interceptor entirely and
      (with no LLM configured) failed outright;
    * the INTEGRAND in TeX — the pattern matched, but the text was passed on
      raw, and ``1/\sqrt{\xi^2+\Delta^2}`` came back as "IndexedSymbol index
      must resolve to a Symbol or IndexSymbol", an internal parser message for
      an expression whose plain-ascii spelling works fine.

    Same rule as AUDIT-DERIV-DISPLAY-RT: what is displayed must be re-enterable.
    """

    def _session(self, lang="ja"):
        nb.start_engine(title="tex_integral", mode="standard", language=lang,
                        domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        for _d in ("Delta を正の実数とする", "omega_D を正の実数とする",
                   "xi を実数とする"):
            r = _run(eng, sup, _d)
            if r["status"] == "needs_confirmation" and r.get("candidates"):
                _c = r["candidates"][-1]
                eng.run_operation(_c["operation"], _c["args"])
        return eng, sup

    def _result(self, eng, r):
        assert r["status"] == "completed", r
        return eng.expressions[r["expr_id"]].canonical.replace(" ", "")

    def test_plain_spelling_is_the_baseline(self):
        eng, sup = self._session()
        try:
            r = _run(eng, sup,
                     "1/sqrt(xi^2 + Delta^2) を xi で 0 から omega_D まで定積分して")
            assert self._result(eng, r) == "asinh(omega_D/Delta)"
        finally:
            nb.stop_engine()

    def test_tex_variable(self):
        eng, sup = self._session()
        try:
            r = _run(eng, sup,
                     r"1/sqrt(xi^2 + Delta^2) を \xi で 0 から omega_D まで定積分して")
            assert self._result(eng, r) == "asinh(omega_D/Delta)"
        finally:
            nb.stop_engine()

    def test_tex_integrand(self):
        eng, sup = self._session()
        try:
            r = _run(eng, sup,
                     r"1/\sqrt{\xi^2 + \Delta^2} を xi で 0 から omega_D まで定積分して")
            assert self._result(eng, r) == "asinh(omega_D/Delta)"
        finally:
            nb.stop_engine()

    def test_everything_in_tex(self):
        eng, sup = self._session()
        try:
            r = _run(eng, sup,
                     r"1/\sqrt{\xi^2 + \Delta^2} を \xi で 0 から \omega_D まで定積分して")
            assert self._result(eng, r) == "asinh(omega_D/Delta)"
        finally:
            nb.stop_engine()

    def test_tex_variable_en(self):
        nb.start_engine(title="tex_integral_en", mode="standard", language="en",
                        domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        try:
            r = _run(eng, sup,
                     r"integrate 1/\sqrt{x^2+1} with respect to \x from 0 to 1")
            # `\x` is not a TeX letter command; the point is the backslash no
            # longer breaks the match — the variable resolves to plain `x`.
            assert self._result(eng, r) == "asinh(1)"
        finally:
            nb.stop_engine()


class TestChangeOfVariablesAcceptsTexSpelling:
    r"""The same rule, one step earlier in the SAME workflow.

    This workflow's own display writes ``d\theta``, so writing the TeX form
    back is the natural thing to do — but the new-variable names were passed on
    raw, missed the standard-coordinate table (polar × disk), and the automatic
    bounds never fired: plain ``r,theta`` produced ∫₀^{2π}∫₀¹ r³, while
    ``r,\theta`` asked the researcher to supply bounds the table already knows
    (measured 2026-08-01).  Found by asking what the polar scenario's angular
    integration actually did — the fix to the definite-integral path alone
    would have left this half undone.
    """

    def _polar(self, new_vars: str, trans: str) -> str:
        """Run the S-o workflow with the given spelling; return the value."""
        nb.start_engine(title="cv_tex", mode="standard", language="ja",
                        domains=["electromagnetism"])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "x^2+y^2<=1 の領域で x^2+y^2 を定積分")
            r = _run(eng, sup, f"E1 を x,y から {new_vars} へ {trans} で変数変換して")
            assert r["status"] == "registered", r
            r2 = _run(eng, sup, "この定積分を評価して")
            assert r2["status"] == "completed", r2
            return eng.expressions[r2["expr_id"]].canonical.replace(" ", "")
        finally:
            nb.stop_engine()

    def test_plain_spelling_is_the_baseline(self):
        assert self._polar("r,theta",
                           "x=r*cos(theta), y=r*sin(theta)") == "%pi/2"

    def test_tex_new_variable_and_transform(self):
        assert self._polar(r"r,\theta",
                           r"x=r*cos(\theta), y=r*sin(\theta)") == "%pi/2"

    def test_mixed_spelling(self):
        """A TeX name declared and a plain name used in the transform is still
        one variable."""
        assert self._polar(r"r,\theta",
                           "x=r*cos(theta), y=r*sin(theta)") == "%pi/2"


class TestIntegrationVariableNotation:
    r"""What may be written as the integration variable (INTEGRAL-VAR-NOTATION).

    Measured 2026-08-02.  Two fixes, both about the researcher writing what the
    notebook showed them:

    * **Unicode Greek.**  CLAUDE.md promises `ω` is converted automatically, and
      `normalize_unicode_greek` does it — but it ran at the top of
      `LLMSupervisor.interpret()`, AFTER every rule-based interceptor.  So `θ`
      missed the definite-integral rule while `\theta` worked, and the failure
      message showed `\theta`: proof the conversion happened just too late.
      Moving it to run ONCE for the request (after the YOU card is recorded,
      before the interceptors) fixed every rule at the same time and added no
      spelling table — the point of the "one normalisation entry" note in the
      ticket.

    * **A subscripted, undeclared name.**  `x_1` reads as a component INDEX in
      an operation (CLAUDE.md), lowering to `x(1)`, and Maxima cannot integrate
      with respect to that.  It answered the bare "Maxima evaluation error",
      which says neither what is wrong nor that declaring the name fixes it.
      The refusal now says both — and the advice is TRUE, which the last test
      here checks rather than asserts.

    A third, 2026-08-06 (INTEGRAL-VAR-BRACED): the BRACED spellings `\xi_{k}` /
    `x_{1}` did not reach any of the above.  The variable capture was
    `\\?[A-Za-z_]\w*`, which stops at `{`, so the whole request fell through to
    the LLM and came back "Could not interpret" — while the identical unbraced
    `\xi_k` got the actionable refusal.  The ticket suspected the normalisation
    ran too late again, but measured, that is not it: normalising the WHOLE
    request would destroy `\int_{0}^{1} x^2 \, dx` (→ `\int_0^1*x^2*dx`), which
    is the input form `_try_integral_input` parses.  The capture is what was
    wrong, and it was wrong in three patterns at once — see
    tests/test_nl_symbol_fragment.py.
    """

    def _session(self):
        nb.start_engine(title="intvar", mode="standard", language="ja",
                        domains=[])
        return nb.get_engine(), _sup()

    @pytest.mark.parametrize("var", ["x", "theta", r"\theta", "θ", "ω"])
    def test_accepted_spellings_all_give_the_same_answer(self, var):
        eng, sup = self._session()
        try:
            r = _run(eng, sup, f"{var}^2 を {var} で 0 から 1 まで定積分して")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical.replace(" ", "") == "1/3"
        finally:
            nb.stop_engine()

    def test_unicode_greek_works_in_english_too(self):
        nb.start_engine(title="intvar_en", mode="standard", language="en",
                        domains=[])
        eng = nb.get_engine()
        try:
            r = _run(eng, _sup(), "integrate θ^2 with respect to θ from 0 to 1")
            assert r["status"] == "completed", r
            assert eng.expressions[r["expr_id"]].canonical.replace(" ", "") == "1/3"
        finally:
            nb.stop_engine()

    def test_the_you_card_keeps_what_was_typed(self):
        """Normalising for interpretation must not rewrite the researcher's own
        words in the log — the same rule as H1-ALIAS-EID-LEAK."""
        eng, sup = self._session()
        try:
            _run(eng, sup, "θ^2 を θ で 0 から 1 まで定積分して")
            typed = [s.user_text or "" for s in eng.steps.values()]
            assert any("θ" in t for t in typed), typed
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize(
        "var", ["x_1", r"\omega_D", "x_{1}", r"\xi_{k}"])
    def test_undeclared_subscripted_variable_is_refused_with_the_fix(self, var):
        """The BRACED spellings must land on this same refusal, not on the LLM's
        "Could not interpret" — which is what they did before INTEGRAL-VAR-BRACED."""
        eng, sup = self._session()
        try:
            r = _run(eng, sup, f"{var}^2 を {var} で 0 から 1 まで定積分して")
            assert r["status"] == "failed", r
            err = r["error"]
            assert "integral." not in err          # translated, not a raw key
            assert "(" in err                      # shows the indexed reading
            assert "宣言" in err                    # and says what to do
            assert "Could not interpret" not in err  # reached the rule, not the LLM
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("declared,written", [
        ("x_1", "x_1"),
        ("x_1", "x_{1}"),          # declared unbraced, written braced
        ("xi_k", r"\xi_{k}"),      # …and the TeX spelling of the same name
    ])
    def test_the_advice_is_true(self, declared, written):
        """Declaring the name really does make that very form work — including
        the braced spelling, which is the same name to the researcher."""
        eng, sup = self._session()
        try:
            r = _run(eng, sup, f"{declared} をスカラーパラメータとして定義して")
            if r["status"] == "needs_confirmation":
                chosen = r["candidates"][-1]        # 「名前の一部」
                eng.run_operation(chosen["operation"], chosen["args"])
            r2 = _run(eng, sup, f"{written}^2 を {written} で 0 から 1 まで定積分して")
            assert r2["status"] == "completed", r2
            assert eng.expressions[r2["expr_id"]].canonical.replace(" ", "") == "1/3"
        finally:
            nb.stop_engine()


class TestADisplayOnlyIntegralIsNotAPlainOperand:
    r"""Found by the practice loop, 2026-08-03 (S-c scenario work).

    A range-bearing definite integral is display-only BY DESIGN: the CAS is not
    invoked at input (INTEGRAL-COV), so its canonical is LaTeX, not a Maxima
    expression.  Handing that to a generic operation used to "succeed": Maxima
    was sent `changevar(\iint_{x^2+y^2 \le 1} …)` and answered with the literal
    `_res_` — the ADAPTER'S OWN scratch variable — which was then registered as
    an expression the researcher could go on to use.

    Two independent guards, because the shape matters more than this one route:
    the engine refuses the operand with a reason that names the right phrasing,
    and the adapter refuses to call its own scratch name an answer.
    """

    def _engine_with_region_integral(self):
        import ui.notebook_bridge as nb
        nb.start_engine(title="displayonly", mode="standard", language="ja",
                        domains=[])
        eng = nb.get_engine()
        e = eng.register_integral_form(
            r"\iint_{x^2+y^2 \le 1} (x^2 + y^2) \, dx \, dy",
            {"integrand_latex": "x^2 + y^2", "integrand_canonical": "x^2 + y^2",
             "bounds": [], "region": "x^2+y^2 <= 1"})
        return nb, eng, e

    def test_generic_operation_is_refused_with_the_fix_named(self):
        import pytest
        nb, eng, e = self._engine_with_region_integral()
        try:
            with pytest.raises(ValueError) as ei:
                eng.run_operation("change_variable", {
                    "expr": e.display_id, "old_var": "x",
                    "new_expr": "r*cos(theta)", "new_var": "r"})
            msg = str(ei.value)
            # the message must say what to do instead, not just "no"
            assert "変数変換" in msg and "評価" in msg
            assert e.display_id in msg
            # nothing was registered
            assert len(eng.expressions) == 1
        finally:
            nb.stop_engine()

    def test_the_english_message_is_english(self):
        import pytest
        import ui.notebook_bridge as nb
        nb.start_engine(title="displayonly_en", mode="standard", language="en",
                        domains=[])
        eng = nb.get_engine()
        try:
            e = eng.register_integral_form(
                r"\iint_{x^2+y^2 \le 1} (x^2 + y^2) \, dx \, dy",
                {"integrand_latex": "x^2 + y^2", "integrand_canonical": "x^2 + y^2",
                 "bounds": [], "region": "x^2+y^2 <= 1"})
            with pytest.raises(ValueError) as ei:
                eng.run_operation("change_variable", {
                    "expr": e.display_id, "old_var": "x",
                    "new_expr": "r*cos(theta)", "new_var": "r"})
            msg = str(ei.value)
            assert "display-only" in msg and "evaluate" in msg
            assert not any("぀" <= ch <= "ヿ" for ch in msg), msg
        finally:
            nb.stop_engine()

    def test_a_bare_scratch_name_is_ordinary_at_the_adapter(self):
        """Pins why the guard is at the ENGINE and not at the adapter — and what
        `_res_` at the adapter actually was.

        A first attempt put a guard in the adapter — "a result equal to our own
        scratch variable can never be an answer" — and it broke five passing
        tests.  The reason recorded then was that `_res_` is the ordinary
        outcome of any command that is not a single expression, `declare(Delta,
        scalar); true` among them.  **Measured 2026-08-20, that reason was
        wrong.**  `;` is not a separator inside parentheses, so the wrapped
        command is a SYNTAX error: Maxima's parser fails before `errcatch` ever
        runs, the declaration does NOT take effect (`featurep` says false), and
        the "success" came from the guard variable still holding the PREVIOUS
        command's value.  Clearing that flag in a statement of its own is what
        separates a real failure from a real result — without it every command
        this system could not lower correctly came back as a plausible answer
        (`simplify` on `x' + x` returned `_res_` with status success).

        Both halves are pinned below: a compound command fails AND changes
        nothing, an ordinary declaration succeeds and takes effect, and a
        `_res_` VALUE is still no business of the adapter's — the real defect,
        a value with no operand behind it reaching the timeline, is stopped
        upstream by `_reject_display_only_operands`.
        """
        from notebook.runtime.maxima_adapter import MaximaAdapter
        with MaximaAdapter() as ad:
            # `;` inside the parenthesised wrapper: a syntax error, and the
            # declaration really did not happen — so reporting failure is the
            # truthful answer, not a regression.
            assert not ad.execute("declare(zz1, scalar); true").ok
            assert ad.execute("featurep(zz1, scalar)").value.strip() == "false"
            # The shape the app actually sends still works, and takes effect.
            assert ad.execute("declare(zz2, scalar)").ok
            assert ad.execute("featurep(zz2, scalar)").value.strip() == "true"
            assert ad.execute("2+3").value.strip() == "5"


class TestEvaluateWithNoIntegralForm:
    r"""EVAL-INTEGRAL-EN-FALLTHROUGH, found by the practice loop 2026-08-03.

    `_try_evaluate_integral` used to DECLINE (return None) when the phrase
    matched but no integral form existed, and the request fell through to the
    alias scan.  There the bare noun `integral` is an alias of
    `integrate_indefinite`, whose positional field mapper took the word
    **`evaluate` as the integration variable**:

        evaluate this integral
        → \int x^2 + 1 \, devaluate = evaluate(x^2+1) + C     status=completed

    A sentence naming something that does not exist came back as a REGISTERED
    expression the researcher could go on to reference.

    Japanese escaped it for an accidental reason, not a good one: `積分を計算して`
    landed on `integrate_definite`, whose four required fields could not be
    filled, so it produced an EMPTY confirmation dialog.  Both are answered here
    now — which is why these tests are bilingual (the defect existed in exactly
    the language nobody wrote a test for).
    """

    def _session(self, lang):
        nb.start_engine(title="evalnoform", mode="standard", language=lang,
                        domains=[])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        _run(eng, sup, "x^2 + 1")
        return eng, sup

    def test_english_says_there_is_no_integral(self):
        eng, sup = self._session("en")
        try:
            for cmd in ("evaluate this integral", "evaluate the integral"):
                r = _run(eng, sup, cmd)
                assert r["status"] == "failed", r
                assert "no definite integral" in r["error"]
                # nothing bogus was registered
                assert len(eng.expressions) == 1, list(eng.expressions)
        finally:
            nb.stop_engine()

    def test_japanese_says_the_same(self):
        eng, sup = self._session("ja")
        try:
            for cmd in ("この定積分を評価して", "積分を計算して"):
                r = _run(eng, sup, cmd)
                assert r["status"] == "failed", r
                assert "評価できる定積分がありません" in r["error"]
                assert len(eng.expressions) == 1, list(eng.expressions)
        finally:
            nb.stop_engine()

    def test_a_named_expression_still_goes_to_the_supervisor(self):
        """`E5 を計算して` / `evaluate E5` is ordinary "compute this", not an
        integral-specific request — this handler must keep declining it so the
        supervisor gets its turn.  (Without an LLM configured that surfaces as
        the generic no-LLM failure, which is the point: it is NOT our message.)"""
        for lang, cmd in (("ja", "E1 を計算して"), ("en", "evaluate E1")):
            eng, sup = self._session(lang)
            try:
                r = _run(eng, sup, cmd)
                assert r["status"] == "failed", r
                assert "no definite integral" not in (r.get("error") or "")
                assert "評価できる定積分" not in (r.get("error") or "")
            finally:
                nb.stop_engine()

    def test_the_real_path_is_untouched(self):
        """The differential check: with an integral form present, both languages
        still evaluate it."""
        seq = {
            "ja": ["x^2 + y^2 <= 1 の領域で x^2 + y^2 を定積分して",
                   "E1 を x,y から r,θ へ x=r*cos(θ), y=r*sin(θ) で変数変換して",
                   "この定積分を評価して"],
            "en": ["definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1",
                   "change of variables in E1 from x,y to r,θ via x=r*cos(θ), y=r*sin(θ)",
                   "evaluate this integral"],
        }
        for lang, cmds in seq.items():
            nb.start_engine(title="evalreal", mode="standard", language=lang,
                            domains=[])
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[], primary_backend="maxima")
            try:
                for c in cmds:
                    r = _run(eng, sup, c)
                assert r["status"] == "completed", (lang, r)
                assert eng.expressions[r["expr_id"]].canonical.replace(" ", "") == "%pi/2"
            finally:
                nb.stop_engine()
