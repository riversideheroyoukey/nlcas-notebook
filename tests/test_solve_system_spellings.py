r"""Solving a system: every spelling, both languages, and the same answer.

Filed as SOLVE-LIST-COMMA-EN — one defect, in English.  Measured 2026-08-16 it
was FOUR, and the ticket had only the first.  All four were reached by the
ticket's own example (`x+y+z=6`, `x-y=0`, `z=2`), because writing one equation
as `z = 2` is ordinary in a linear system and that is what triggers three of
them:

1. **EN comma list** — `solve E1, E2 and E3 for x, y, z`, the most natural
   English enumeration, fell through to a SINGLE-equation solve and returned a
   partial answer (`a = 4-b`) as a success.
2. **JA comma list** — `E1, E2 と E3 の連立…` matched only the TRAILING ids, so
   the first equation was silently dropped and a 2-equation system was solved.
3. **The solve-for VARIABLES were rewritten to E-IDs.**  2a-sym rewrites the
   researcher's symbols into expression ids so the supervisor can tell a name
   from a reference; the rule-based parser then read that same text for
   variable NAMES, and a variable that was also an equation's left side became
   its id — `solve([…], [a, b, E3])`.
4. **A P2-bound symbol could not be a solve target.**  H-1 binds `c: 2`
   CAS-side, so `[a, b, c]` evaluated to `[a, b, 2]` and Maxima refused the
   call.  `block([…])` unbinds them for one call — the mechanism
   `_build_inverse_transform` already used against the same binding.

**3 and 4 cancelled each other**: English replaced `c` with the unbound `E3`,
which dodged the P2 binding, so the case everyone believed correct returned a
plausible answer for the other two variables.  Fixing either alone makes the
failure visible; that is why they are pinned together here.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOMAIN = "mathematical_economics"

#: The ticket's own system, renamed off x/y/z so no variable collides with a
#: coordinate the domain may declare.  `c = 2` is the load-bearing part: it
#: makes `c` an H-1 symbol, which is what defects 3 and 4 need.
EQUATIONS = ["a + b + c = 6", "a - b = 0", "c = 2"]
ANSWER = {"a = 2", "b = 2", "c = 2"}

SPELLINGS = {
    "ja": [
        "E1 と E2 と E3 の連立方程式を a, b, c について解いて",
        "E1, E2 と E3 の連立方程式を a, b, c について解いて",
        "E1、E2、E3 の連立方程式を a, b, c について解いて",
        "E1, E2, E3 の連立方程式を a, b, c について解いて",
    ],
    "en": [
        "solve E1 and E2 and E3 for a, b, c",
        "solve E1, E2 and E3 for a, b, c",
        "solve E1, E2, E3 for a, b, c",
        "solve E1, E2, and E3 for a, b, c",
    ],
}


def _solve(lang, cmd):
    """(operation, status, {registered results}) for one fresh session."""
    nb.start_engine(title="solve", mode="standard", language=lang,
                    domains=[DOMAIN])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[DOMAIN], primary_backend="maxima")

        def run(text):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    eng, sup))

        for e in EQUATIONS:
            run(e)
        before = set(eng.expressions)
        r = run(cmd)
        new = {eng.expressions[e].latex.replace(" ", "")
               for e in eng.expressions if e not in before}
        return r.get("operation"), r.get("status"), new
    finally:
        nb.stop_engine()


class TestEverySpellingSolvesTheWholeSystem:
    @pytest.mark.parametrize("lang,cmd",
                             [(l, c) for l, cs in SPELLINGS.items() for c in cs],
                             ids=[f"{l}:{c[:34]}" for l, cs in SPELLINGS.items()
                                  for c in cs])
    def test_same_operation_and_same_answer(self, lang, cmd):
        op, status, results = _solve(lang, cmd)
        assert op == "solve_system", f"{cmd!r} ran {op!r}"
        assert status == "completed", f"{cmd!r} → {status}"
        assert results == {a.replace(" ", "") for a in ANSWER}, \
            f"{cmd!r} → {results}"


class TestTheFourDefectsIndividually:
    r"""Each named, so a regression says which one came back."""

    def test_the_comma_list_is_not_a_single_solve(self):
        """Defect 1: it answered `a = 4-b` — E1 alone, solved for `a`."""
        op, _st, results = _solve("en", "solve E1, E2 and E3 for a, b, c")
        assert op == "solve_system"
        assert "a=4-b" not in results

    def test_no_equation_is_dropped(self):
        """Defect 2: `E1, E2 と E3` matched only the trailing ids, so the first
        equation vanished and the 2-equation answer `b = a` came back."""
        _op, _st, results = _solve("ja", "E1, E2 と E3 の連立方程式を a, b, c について解いて")
        assert "b=a" not in results, results
        assert "a=2" in results

    def test_a_variable_that_is_also_a_symbol_stays_a_variable(self):
        """Defect 3: `c` is E3's left side, and the solve target became `E3`."""
        from ui.interpret_equation_ops import _parse_solve
        rewritten = "solve E1 and E2 and E3 for a, b, E3"       # what 2a-sym made
        written = "solve E1 and E2 and E3 for a, b, c"          # what was typed
        _op, args = _parse_solve(rewritten, "en", written)
        assert args["vars"] == "a, b, c"
        assert args["equations"] == "E1, E2, E3"

    def test_a_p2_bound_symbol_can_be_solved_for(self):
        """Defect 4: `c: 2` is bound CAS-side, so `[a, b, c]` became `[a, b, 2]`."""
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        build = MaximaBackendHandler(None).build_command(
            "solve_system",
            {"equations": "E1, E2", "vars": "a, b"},
            {"E1": "a + b = 3", "E2": "a - b = 1"})
        assert build.ok
        assert build.command.startswith("block([a, b],"), build.command


class TestWhatMustNotChange:
    def test_a_single_equation_still_solves(self):
        nb.start_engine(title="one", mode="standard", language="en",
                        domains=[DOMAIN])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[DOMAIN], primary_backend="maxima")

            def run(text):
                return asyncio.get_event_loop().run_until_complete(
                    run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        eng, sup))

            run("2*a + 4 = 10")
            r = run("solve E1 for a")
            assert r["status"] == "completed", r
            assert r["operation"] == "solve_equation"
        finally:
            nb.stop_engine()
