"""A driven ODE through the Laplace domain, using GENERAL operations only.

The forced-oscillator scenario (docs/scenarios/scenario_forced_oscillator_laplace_
{jp,en}.md) found four silent-wrongs, each in a column the RLC scenario never
enters.  Three were general defects and are pinned below; the fourth
(EN-ALIAS-MULTIWORD) is still open and is why the English inverse transform is
driven through `run_operation` rather than natural language.

**Rewritten 2026-08-15 (SCENARIO-MACRO case 7).**  `laplace_by_definition` was
deleted as a macro, so the route here is the one the scenario now takes:

* the drive's image is the DEFINITE INTEGRAL itself
  (`\\int_0^\\infty \\sin(\\omega t)e^{-st}dt` → `integrate_definite`), composed
  into the s-domain equation by its E-id (H-2);
* the derivative's image is a THEOREM the researcher writes
  (`s^2 X - s x_0 - v_0`), not something the system derives;
* the initial conditions are ordinary equations (`x_0 = 1`), not free parameters
  owned by a transform operation.

What that costs, and what it does not: defect 1 (a drive term passed through
untransformed) lived INSIDE the deleted linearity split and dies with it, so its
two tests are gone rather than moved — there is no longer a code path that can
pass a term through.  Defect 2's trap is the opposite case: Maxima splitting a
trailing digit run off an italicised name is a GENERAL display defect
(`k_B1`, `a_b2`), so its unit tests stay while the end-to-end form, which
depended on the macro's `x_ic1` parameters, is gone.

Bilingual per the project rule: the NL path is checked in both JA and EN.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler


def _sup(lang="ja"):
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["differential_equations"], primary_backend="maxima")


def _run(eng, sup, value, ttype="text", canonical=None):
    tok = (InputToken(type="math", value=value, latex=value, canonical=canonical or "")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


# The scenario's command list, in both languages.  Kept as data so the two
# language paths cannot drift into testing different things.
_COMMANDS = {
    "ja": {
        "scalar":   "F_0 をスカラーとする",
        "time_dep": "x を時間依存関数として定義する",
        "name_lap": "x のラプラス変換を X とおく",
        "assume":   "{sym} > 0 と仮定する",
        "defint":   r"{eid} を t で 0 から \infty まで定積分して",
        "solve":    "{eid} を X について解いて",
    },
    "en": {
        "scalar":   "Declare F_0 as a scalar",
        "time_dep": "Define x as a time-dependent function",
        "name_lap": "Let X be the Laplace transform of x",
        "assume":   "Assume {sym} > 0",
        "defint":   r"Integrate {eid} with respect to t from 0 to \infty",
        "solve":    "Solve {eid} for X",
    },
}


def _eid(result):
    """The expression id this step actually registered.

    NEVER write an E-number literal in a test.  A literal asserts the number the
    author ASSUMED, not what the run produced: when the step list shifts (a
    declaration registers no expression; a card is inserted), the reference still
    names an EXISTING expression, so the operation succeeds — on the wrong one —
    and the test reports a pass while proving nothing.  Measured twice while
    writing this file, and again when it was rewritten onto general operations.
    (A scenario doc is the opposite case: there the E-number is the researcher's
    own input and must stay literal.)
    """
    eid = result.get("expr_id")
    if not eid:
        regs = result.get("registered_expressions") or []
        eid = regs[-1]["expr_id"] if regs else None
    assert eid, f"step registered no expression: {result}"
    return eid


def _s_domain_session(lang):
    """Run the scenario up to the s-domain equation; return (engine, sup, result).

    General operations only — no operation here belongs to Laplace except the
    transform-pair DECLARATION, which is a naming (open_items LAPLACE-PAIR-ANCHOR).
    """
    c = _COMMANDS[lang]
    nb.start_engine(title=f"lapdrive_{lang}", mode="standard",
                    language=lang, domains=["differential_equations"])
    eng = nb.get_engine()
    sup = _sup(lang)
    _run(eng, sup, c["scalar"])
    _run(eng, sup, c["time_dep"])
    _run(eng, sup, c["name_lap"])
    # The convergence conditions, as visible steps.  Without them the CAS asks
    # ("Is s positive, negative or zero?") and the step fails with that reason —
    # measured; the session survives.  The deleted macro did an assume/forget
    # silently, which hid a modelling choice inside shared code.
    _run(eng, sup, c["assume"].format(sym="s"))
    _run(eng, sup, c["assume"].format(sym=r"\omega"))
    r_int = _run(eng, sup, r"\sin(\omega t) e^{-s t}", "math")
    r_img = _run(eng, sup, c["defint"].format(eid=_eid(r_int)))
    assert r_img["status"] == "completed", r_img
    r_eq = _run(eng, sup,
                rf"m (s^2 X - s x_0 - v_0) + k X = F_0 {_eid(r_img)}", "math")
    return eng, sup, r_eq


class TestTheDriveImageComesFromTheDefinition:
    """The drive's image is computed, not assumed — and it reaches the equation.

    This is what remains of defect 1 once the macro is gone: the question is no
    longer "did the linearity split pass a term through?" but "does the
    researcher's own definition integral compose into the s-domain equation?".
    """

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_definite_integral_is_the_transform(self, lang):
        try:
            c = _COMMANDS[lang]
            nb.start_engine(title=f"lapimg_{lang}", mode="standard",
                            language=lang, domains=["differential_equations"])
            eng = nb.get_engine()
            sup = _sup(lang)
            _run(eng, sup, c["assume"].format(sym="s"))
            _run(eng, sup, c["assume"].format(sym=r"\omega"))
            r_int = _run(eng, sup, r"\sin(\omega t) e^{-s t}", "math")
            r = _run(eng, sup, c["defint"].format(eid=_eid(r_int)))
            assert r["status"] == "completed", r
            assert r["canonical"].replace(" ", "") == "omega/(s^2+omega^2)", r
            # The researcher wrote `\infty`; it must come back as `\infty`, not
            # as Maxima's `inf` set in italics.
            assert r"\infty" in r["latex"], r["latex"]
            assert "{inf}" not in r["latex"], r["latex"]
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_no_t_domain_term_survives_into_the_s_domain_equation(self, lang):
        try:
            eng, _sup, r = _s_domain_session(lang)
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            # The E-id reference (H-2) put the computed image INTO the equation.
            assert "omega/(s^2+omega^2)" in canon, canon
            assert "sin(omega*t)" not in canon, canon
        finally:
            nb.stop_engine()


class TestTrailingDigitsInAnItalicisedName:
    """Defect 2's trap, kept at the level where it lives.

    Maxima's `tex1` splits a trailing digit run off an identifier it also
    italicises, so unwrapping the group alone produced `x_{ic}_{1}` — a double
    subscript, a hard LaTeX error that would stop the exported .tex from
    typesetting at all.  Nothing about Laplace: `k_B1` and `a_b2` do the same.
    The end-to-end form of this test is gone with the macro whose free
    parameters were the only names that reached it.
    """

    def test_underscore_name_with_trailing_digits(self):
        raw = r"{\it x\_ic}_{1}+{\it k\_B}_{2}"
        out = MaximaBackendHandler._unwrap_italic_identifiers(raw)
        assert out == "x_{ic1}+k_{B2}", out
        assert "}_{" not in out.replace("+", ""), out   # no double subscript

    def test_names_without_underscore_are_unchanged(self):
        # `foo1` has no second subscript to collide with — behaviour preserved.
        assert MaximaBackendHandler._unwrap_italic_identifiers(
            r"{\it foo}_{1}") == "foo_{1}"
        assert MaximaBackendHandler._unwrap_italic_identifiers(
            r"{\it alpha\_mk}") == "alpha_{mk}"


class TestInverseTransformOfNamedEquation:
    """Defect 3: `ilt` must act on the VALUE of `X = …`, not on the equation.

    The first test drives `run_operation` directly: the engine-side question
    ("does a bare-symbol LHS get stripped for this operation?") is language
    independent, and mixing the NL routing into it would make it a change
    detector for a different layer.

    The NL path is pinned SEPARATELY, in both languages, below.  It could not be
    until 2026-08-15: `Take the inverse Laplace transform of …` resolved to the
    FORWARD transform (open_items EN-ALIAS-MULTIWORD, now fixed), so an English
    assertion here would have pinned the defect instead of the behaviour.
    """

    def _solved(self, lang):
        eng, sup, r_eq = _s_domain_session(lang)
        _run(eng, sup, "$x_0 = 1$", "math")
        _run(eng, sup, "$v_0 = 0$", "math")
        rs = _run(eng, sup, _COMMANDS[lang]["solve"].format(eid=_eid(r_eq)))
        assert rs["status"] == "completed", rs
        for sym in ("m", "k"):
            _run(eng, sup, _COMMANDS[lang]["assume"].format(sym=sym))
        return eng, sup, rs

    def test_inverse_does_not_transform_the_lhs_symbol(self):
        try:
            eng, _sup, rs = self._solved("ja")
            step, _v = eng.run_operation(
                "inverse_laplace", {"expr": _eid(rs), "s": "s", "t": "t"})
            canon = eng.expressions[step.output_refs[0]].canonical
            canon = canon.replace(" ", "").replace("\n", "")
            # The measured silent-wrong: ilt(X = …) read X as a constant.
            assert "delta(t)" not in canon, canon
            assert "X*" not in canon and not canon.startswith("X"), canon
            # The physics: free term at sqrt(k/m) plus a forced term at omega.
            assert "cos((sqrt(k)*t)/sqrt(m))" in canon, canon
            assert "sin(omega*t)" in canon, canon
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_nl_path_reaches_the_inverse_transform(self, lang):
        r"""What the researcher actually types, in both languages.

        English went to the FORWARD transform until 2026-08-15, and registered
        the result — with the preposition `the` pulled in as a variable — as a
        SUCCESS.  Both halves are asserted: the operation, and that no English
        grammar word became part of the value.
        """
        invert = {"ja": "{eid} を s から t へ逆ラプラス変換して",
                  "en": "Take the inverse Laplace transform of {eid} from s to t"}[lang]
        try:
            eng, sup, rs = self._solved(lang)
            r = _run(eng, sup, invert.format(eid=_eid(rs)))
            assert r["status"] == "completed", r
            assert r["operation"] == "inverse_laplace", r["operation"]
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            for word in ("the", "*to", "(to)", "take", "from"):
                assert word not in canon, (word, canon)
            assert "cos((sqrt(k)*t)/sqrt(m))" in canon, canon
            assert eng.expressions[_eid(r)].latex.startswith("x(t) = ")
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_nonzero_initial_conditions_reach_the_solution(self, lang):
        """The column RLC does not enter: ICs solved at a NONZERO value.

        `x_0 = 1` must reach the s-domain solution — with `x_0 = 0` the free
        term disappears entirely, so its presence is what proves the condition
        was applied rather than dropped.
        """
        try:
            eng, _sup, rs = self._solved(lang)
            canon = eng.expressions[_eid(rs)].canonical.replace(" ", "")
            assert "x_0" not in canon and "v_0" not in canon, canon
            assert "m*s^3" in canon, canon          # x_0 = 1 → the m s^3 term
        finally:
            nb.stop_engine()


class TestResultLabels:
    """The forced-oscillator scenario's review findings (2026-08-13)."""

    def test_inverse_transform_result_is_labelled_x_of_t(self):
        """An inverse transform's result carries its TIME-domain name.

        `X = …` inverted is `x(t) = …`; published bare, the time-domain solution
        arrived with nothing saying what it was.  The pairing comes from
        define_laplace_transform's table, so a capitalised symbol that never was
        a transform gets NO label rather than an invented one.  That table is
        the researcher's NAMING, not mathematics the CAS lacks — see open_items
        LAPLACE-PAIR-ANCHOR for the measured alternative.
        """
        try:
            eng, _sup, rs = TestInverseTransformOfNamedEquation()._solved("ja")
            step, _v = eng.run_operation(
                "inverse_laplace", {"expr": _eid(rs), "s": "s", "t": "t"})
            latex = eng.expressions[step.output_refs[0]].latex
            assert latex.startswith("x(t) = "), latex
        finally:
            nb.stop_engine()

    def test_undeclared_capital_gets_no_invented_label(self):
        try:
            nb.start_engine(title="lapnolabel", mode="standard", language="ja",
                            domains=["differential_equations"])
            eng = nb.get_engine()
            assert eng._laplace_time_lhs("ABC", {"t": "t"}) == ""
        finally:
            nb.stop_engine()


class TestTexEscapesInequalities:
    """`<` / `>` are not LaTeX specials but are unprintable in OT1 text mode
    (the preamble omits fontenc on purpose): «D > 0» set as «D ¿ 0»."""

    def test_inequalities_go_to_math_mode(self):
        from ui.routes.export import _tex_escape
        assert _tex_escape("D > 0 と仮定する") == r"D $>$ 0 と仮定する"
        assert _tex_escape("a < b") == r"a $<$ b"
        # The two-character spellings must win: `<=` is ≤, not "< =".
        assert _tex_escape("x^2+y^2<=1") == r"x\^{}2+y\^{}2$\le$1"
        assert _tex_escape("n >= 3") == r"n $\ge$ 3"

    def test_a_replacement_is_never_re_escaped(self):
        """One pass, not a chain of replaces.  Several replacements contain
        characters that are themselves keys, so a sequential table escapes what
        an earlier row inserted: `x^2` came out `x\\^\\{\\}2`."""
        from ui.routes.export import _tex_escape
        assert _tex_escape("x^2") == r"x\^{}2"
        assert _tex_escape("a\\b") == r"a\textbackslash{}b"
        assert _tex_escape("$5 > $3") == r"\$5 $>$ \$3"
