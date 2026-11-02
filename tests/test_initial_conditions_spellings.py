r"""Setting initial conditions: the spellings a researcher actually writes.

IC-NL-ORPHANED.  The bare form — ``q(0)=0, \dot q(0)=0 とする``, no target and
no verb — built the free-parameter names ``q_ic0`` / ``q_ic1``, and the only
thing that ever PRODUCED parameters with those names was
``laplace_by_definition``'s integration-by-parts branch, deleted with the macro
on 2026-08-15 (SCENARIO-MACRO case 7).  So a spelling that had worked the day
before answered "Could not interpret", while the very similar
``E5 の初期条件を … とする`` kept working: the same intent succeeded or failed
depending on how it was written.

**The repair changed direction on a user report.**  Measuring showed the bare
form has no target, and the first fix refused it with a message naming the
spellings that do work.  The researcher then said they see the bare form
written — so it is ROUTED instead, to the latest expression carrying free
parameters.  That is not an invention: it is the rule this same interceptor
already used for the bare ``C=0 とする``, and answering the two bare forms
differently would have been a second answer to one question.

A refusal remains for the case that cannot be routed — conditions with nothing
to apply them to — and it names the spellings that work.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
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


def _general_solution(eng, sup):
    """`q = k_1 sin(tw) + k_2 cos(tw)`, carrying two free parameters."""
    _run(eng, sup, "q を時間依存関数として定義する")
    e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
    return _eid(_run(eng, sup, f"{e1} を一般に解いて"))


#: Every spelling means the same thing: fix k₁,k₂ from q(0)=q0, q̇(0)=v0.
SPELLINGS = [
    r"q(0)=q0, \dot{q}(0)=v0 とする",          # bare — the orphaned one
    "{g} の初期条件を q(0)=q0, q'(0)=v0 とする",
    "{g} を q(0)=q0, q'(0)=v0 で特殊化して",
]


class TestEverySpellingReachesTheSameOperation:
    @pytest.mark.parametrize("spelling", SPELLINGS, ids=[s[:28] for s in SPELLINGS])
    def test_same_operation_and_same_solution(self, spelling):
        nb.start_engine(title="ic", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            g = _general_solution(eng, sup)
            # `.format` would choke on the `{q}` of `\dot{q}`
            r = _run(eng, sup, spelling.replace("{g}", g))
            assert r["status"] == "completed", r
            assert r["operation"] == "specialize_by_conditions"
            got = eng.expressions[_eid(r)].canonical.replace(" ", "")
            # k₁ = v0/w (sin term), k₂ = q0 (cos term) — no free parameters left
            assert "v0" in got and "q0" in got
            assert eng.expressions[_eid(r)].free_parameters == []
        finally:
            nb.stop_engine()

    def test_the_bare_form_picks_the_same_target_as_the_bare_value_form(self):
        r"""`C=0 とする` and `q(0)=0 とする` are both bare, so both take the
        latest expression carrying free parameters.  Answering them differently
        is what this repair removed."""
        nb.start_engine(title="ic2", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            g = _general_solution(eng, sup)
            r = _run(eng, sup, r"q(0)=q0, \dot{q}(0)=v0 とする")
            assert r["status"] == "completed", r
            # the target really was the general solution, not something else
            assert eng.expressions[_eid(r)].canonical.startswith("q =")
            assert g in eng.expressions
        finally:
            nb.stop_engine()


class TestWhatCannotBeRoutedIsRefusedWithAReason:
    def test_conditions_with_nothing_to_apply_them_to(self):
        """No expression carries free parameters, so there is no target to
        guess.  The message has to name the spellings that work — a refusal
        that explains nothing is what this ticket was filed about."""
        nb.start_engine(title="ic3", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup, r"q(0)=0, \dot{q}(0)=0 とする")
            assert r["status"] == "failed", r
            err = str(r["error"])
            assert "初期条件" in err
            assert "q(0)=0" in err        # it quotes what was written back
        finally:
            nb.stop_engine()


class TestTheOrphanedMachineryIsGone:
    def test_no_ic_name_building_remains(self):
        """`q_ic0` had no producer left; keeping the builder alive meant a
        branch that could never match."""
        import ui.interpret_free_params as fp
        src = open(fp.__file__, encoding="utf-8").read()
        assert "_ic{" not in src
        assert not hasattr(fp, "_latest_ic_expr")

    def test_the_plain_free_variable_form_still_works(self):
        """The other half of the same interceptor, untouched."""
        nb.start_engine(title="ic4", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "x^2", "math")
            r = _run(eng, sup, "E1 を x で不定積分して")
            assert eng.expressions[_eid(r)].free_parameters == ["C"]
            r2 = _run(eng, sup, "C=0 とする")
            assert r2["operation"] == "substitute_free_parameters", r2
        finally:
            nb.stop_engine()
