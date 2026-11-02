r"""The 代入 interceptor's `=` split is both a CAS pattern and a published label.

ONE-QUESTION-MANY-ANSWERS ratchet, fourth lowering (`ui/interpret_rules.py`
5→0).  All five sites sit in `_try_rule_dainyu` (「En を Em に代入して」) and all
five become `substitute`'s `lhs` / `rhs`.  What makes this file different from
the three before it: the same `lhs` is ALSO the timeline label, so a mis-split is
written into the notebook and from there into the exported .tex.

Measured 2026-08-13, the two shapes broke differently:

    \sum_{i=1}^{3} c_i = u a + v b
        → substitute(lhs='\sum_{i', rhs='1}^3*c_i = u*a + v*b')
        the guard refuses — LOUD, but the reason names a fragment the
        researcher never wrote

    y \le x^2 + a
        → substitute(lhs='y <', rhs='x^2 + a')  and the step COMPLETED
        the value was right (the builder lowers `y <` back to `y`), while the
        published label read `E2 を[E2] y < → x^2 + a` — wrong in what is
        WRITTEN DOWN rather than in what is computed, the same shape as
        `x + x = 2 x=y`

Which relations count is unchanged: the old guard `"=" in canon` fired on `<=`
and `>=` too, and an inequality does reach here, so only the scan became
bracket-aware.

**One of the five sites is covered only for no-regression.** The alias-key
branch (tier 1) takes its RHS from a fifth split, and no input was found that
makes it differ: to reach it the key must carry no backslash, and to break it the
canonical needs a bracketed `=` BEFORE the equation's — but a left side holding a
bounded sum makes every alias key LaTeX, so the branch is dropped and tier 3
answers instead.  The control below does exercise it (`p = u a + v b` takes tier
1); nothing here proves the split there matters.
"""
import asyncio

import pytest

import notebook.engine as engine_module
import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline


def _run(engine, supervisor, value, ttype="text"):
    token = (InputToken(type="math", value=value, latex=value, canonical="")
             if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[token]), engine, supervisor))


@pytest.fixture
def session():
    nb.start_engine(title="test_dainyu_split", mode="standard", language="ja",
                    domains=[])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=[], primary_backend="maxima")
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


@pytest.fixture
def substitute_args(monkeypatch):
    """Every ``substitute`` the interceptor asks the engine for.

    The contract between the interceptor and the engine IS the thing that broke,
    so it is what these tests read.
    """
    seen: list[dict] = []
    original = engine_module.NotebookEngine.run_operation

    def spy(self, operation, args, *a, **kw):
        if operation == "substitute":
            seen.append(dict(args))
        return original(self, operation, args, *a, **kw)

    monkeypatch.setattr(engine_module.NotebookEngine, "run_operation", spy)
    return seen


def test_a_bounded_sum_on_the_left_is_asked_for_whole(session, substitute_args):
    engine, supervisor = session
    _run(engine, supervisor, r"\sum_{i=1}^{3} c_i = u a + v b", "math")
    _run(engine, supervisor, "H = c_1 + y", "math")
    result = _run(engine, supervisor, "E1 を E2 に代入して")

    assert substitute_args == [{"expr": "E2", "lhs": r"\sum_{i=1}^3*c_i",
                                "rhs": "u*a + v*b"}]
    # The refusal is correct (that sum is not in H) — what matters is that the
    # reason names what the researcher wrote.  Asserted positively: `\sum_{i` is
    # a PREFIX of the right answer, so "the fragment is absent" would pass on
    # the correct string too.
    assert result["status"] == "failed"
    assert r"\sum_{i=1}^3*c_i" in result["error"], result["error"]


def test_an_inequality_does_not_leak_a_half_operator_into_the_notebook(
        session, substitute_args):
    engine, supervisor = session
    _run(engine, supervisor, r"y \le x^2 + a", "math")
    _run(engine, supervisor, "H = p + y", "math")
    result = _run(engine, supervisor, "E1 を E2 に代入して")

    assert substitute_args == [{"expr": "E2", "lhs": "y", "rhs": "x^2 + a"}]
    assert result["status"] == "completed"
    assert result["canonical"] == "x^2+p+a"
    label = result["operation_label"]
    assert "y <" not in label, label
    # Compared without spaces: since 2026-08-23 this label goes through the
    # display door (`_display_arg`), the same one `_format_substitute_latex`
    # uses, so that a substitution shows the researcher's spelling rather than
    # the internal name — the timeline used to read `rho → lambda/mu` where
    # `\rho` and `\lambda/\mu` were written.  The door re-renders through the
    # CAS, which normalises `x^2 + a` to `x^2+a`.  What this test guards is that
    # the INEQUALITY half does not leak into the label; the spacing is not the
    # point, and asserting it pinned an accident of the raw pass-through.
    _squash = lambda t: t.replace(" ", "")
    assert _squash("y → x^2 + a") in _squash(label), label


def test_the_everyday_substitution_is_untouched(session, substitute_args):
    """The control: this is what the route is for."""
    engine, supervisor = session
    _run(engine, supervisor, "p = u a + v b", "math")
    _run(engine, supervisor, "H = p + y", "math")
    result = _run(engine, supervisor, "E1 を E2 に代入して")

    assert substitute_args == [{"expr": "E2", "lhs": "p", "rhs": "u*a + v*b"}]
    assert result["status"] == "completed"
    assert result["canonical"] == "y+b*v+a*u"
