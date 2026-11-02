"""Constrained optimisation written with GENERAL OPERATIONS (SCENARIO-MACRO, case 3).

Replaces `tests/test_constrained_optimization.py`, deleted with the three macros
it tested — `form_lagrangian` ("objective − multiplier × constraint"),
`stationary_conditions` (`differentiate` once per variable) and
`implicit_derivative` (the template `-diff(F,p)/diff(F,x)`).  All three are
expressions the researcher writes; the Lagrangian's shape and the implicit
function theorem are economics, not system behaviour.

What is pinned here is the replacement route, in both languages, plus the two
things the rewrite MEASURED and that a reader would otherwise get wrong:

* a subscripted name must be DECLARED, or the first-order condition is silently
  wrong (the undeclared `x_1` resolves as the component index `x(1)`);
* the comparative-statics steps must come BEFORE the solve, because solving
  registers `x_1 = …` as an H-1 definition and P2 binds `x_1` CAS-side, which
  collapses `F = am − (a+b)p₁x₁` to 0.

The H-1 tautology trap the deleted file also recorded is NOT lost: it is pinned
independently in `tests/test_fourier_transform.py` and `tests/test_game_theory.py`
(measured before deleting).
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

DOMAIN = "mathematical_economics"


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[DOMAIN], primary_backend="maxima")


def _run(eng, sup, value, kind="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if kind == "math" else InputToken(type="text", value=value))
    r = asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))
    if r["status"] == "needs_confirmation" and r.get("candidates"):
        # The subscript dialog: the LAST candidate is 「名前の一部」 (keep the
        # full subscripted name), which is what a declaration means here.
        ch = r["candidates"][-1]
        eng.run_operation(ch["operation"], ch["args"])
        return {"status": "confirmed"}
    return r


def _eid(result):
    eid = result.get("expr_id")
    if not eid:
        regs = result.get("registered_expressions") or []
        eid = regs[-1]["expr_id"] if regs else None
    assert eid, f"step registered no expression: {result}"
    return eid


_CMD = {
    "ja": {"declare": "{n} をスカラーとする",
           "pdiff":   "{eid} を {v} で偏微分して",
           "set0":    "{eid} の右辺を 0 とする"},
    "en": {"declare": "declare {n} as a scalar",
           "pdiff":   "take the partial derivative of {eid} with respect to {v}",
           "set0":    "set the right-hand side of {eid} to 0"},
}


def _lagrangian_session(lang, declare=True):
    nb.start_engine(title=f"econ_{lang}_{declare}", mode="standard",
                    language=lang, domains=[DOMAIN])
    eng, sup = nb.get_engine(), _sup()
    c = _CMD[lang]
    if declare:
        for n in ("x_1", "x_2", "p_1", "p_2"):
            _run(eng, sup, c["declare"].format(n=n))
    _run(eng, sup, r"U = x_1^a x_2^b", "math")
    rl = _run(eng, sup, r"L = U - \lambda (p_1 x_1 + p_2 x_2 - m)", "math")
    return eng, sup, _eid(rl)


class TestFirstOrderConditions:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_declared_subscripts_give_the_right_condition(self, lang):
        try:
            eng, sup, lid = _lagrangian_session(lang)
            r = _run(eng, sup, _CMD[lang]["pdiff"].format(eid=lid, v="x_1"))
            assert r["status"] == "completed", r
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            # Factor ORDER moved on 2026-08-16, when `declare(x_1, scalar)`
            # started taking effect: the templates carried a `; true` suffix,
            # and inside the adapter's `errcatch(_res_:(…))` wrapper Maxima's
            # separator is `,` not `;`, so every declaration this app sent was
            # silently a no-op (`declare_symmetry` did nothing at all).  With
            # the declaration in force Maxima sorts the declared scalars last.
            # Same expression, different spelling — asserted as a CAS
            # difference so the ordering is not pinned a third time.
            assert canon == "x_1^(a-1)*x_2^b*a-p_1*lambda", canon
            diff = eng._adapter.execute(
                f"ratsimp(({canon}) - (a*x_1^(a-1)*x_2^b-p_1*lambda))")
            assert diff.value.strip() == "0", diff.value
        finally:
            nb.stop_engine()

    def test_undeclared_subscript_loses_the_utility_term(self):
        """Why steps 1–4 of the scenario exist.  Undeclared, `x_1` resolves as
        the component index `x(1)`, the U term contributes nothing, and the
        first-order condition comes back WRONG — as a success."""
        try:
            eng, sup, lid = _lagrangian_session("ja", declare=False)
            r = _run(eng, sup, _CMD["ja"]["pdiff"].format(eid=lid, v="x_1"))
            assert r["status"] == "completed", r
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            assert canon == "-(p(1)*lambda)", canon      # the measured wrong answer
            assert "x_1^(a-1)" not in canon
        finally:
            nb.stop_engine()


class TestComparativeStatics:

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_implicit_function_theorem_by_general_operations(self, lang):
        c = _CMD[lang]
        try:
            nb.start_engine(title=f"econ_ift_{lang}", mode="standard",
                            language=lang, domains=[DOMAIN])
            eng, sup = nb.get_engine(), _sup()
            for n in ("x_1", "p_1"):
                _run(eng, sup, c["declare"].format(n=n))
            rf = _run(eng, sup, r"F = a m - (a+b) p_1 x_1", "math")
            fm = _eid(_run(eng, sup, c["pdiff"].format(eid=_eid(rf), v="m")))
            fx = _eid(_run(eng, sup, c["pdiff"].format(eid=_eid(rf), v="x_1")))
            r = _run(eng, sup, rf"\frac{{dx_1}}{{dm}} = -\frac{{{fm}}}{{{fx}}}", "math")
            latex = eng.expressions[_eid(r)].latex.replace(" ", "")
            # Factor order moved with the declaration repair (see the note in
            # TestFirstOrderConditions); `p_1(b+a)` is if anything the more
            # natural reading of the two.
            assert r"\frac{a}{p_{1}\left(b+a\right)}" in latex, latex
        finally:
            nb.stop_engine()

    def test_solving_first_would_collapse_F(self):
        """Why the scenario orders the steps as it does: after the solve, P2
        binds `x_1` CAS-side, so `F = am − (a+b)p₁x₁` is identically 0 and its
        m-derivative is 0 — a plausible-looking wrong answer."""
        try:
            nb.start_engine(title="econ_order", mode="standard", language="ja",
                            domains=[DOMAIN])
            eng, sup = nb.get_engine(), _sup()
            for n in ("x_1", "p_1"):
                _run(eng, sup, _CMD["ja"]["declare"].format(n=n))
            _run(eng, sup, r"x_1 = \frac{a m}{(a+b) p_1}", "math")   # as solve would
            rf = _run(eng, sup, r"F = a m - (a+b) p_1 x_1", "math")
            r = _run(eng, sup, _CMD["ja"]["pdiff"].format(eid=_eid(rf), v="m"))
            assert r["status"] == "completed", r
            assert eng.expressions[_eid(r)].canonical.strip() == "0"
        finally:
            nb.stop_engine()


def test_the_three_macros_are_gone():
    from notebook.engine import NotebookEngine
    v = OperationValidator()
    assert "form_lagrangian" not in v.known_operations
    assert "implicit_derivative" not in v.known_operations
    assert not hasattr(NotebookEngine, "stationary_conditions")
