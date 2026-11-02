r"""The two interceptors whose WINNING behaviour was measured nowhere.

Measured 2026-08-23 by recording ``ctx.route`` through every ``run_interceptors``
call, over the whole test suite AND over the instruction corpus:

    registered interceptors            76
    win at least once in the corpus    47
    win at least once in the tests     67
    win in EITHER                      69
    NEVER win anywhere                  7

Five of the seven — ``numeric_fork_nl`` and the four ``plan_*`` — have dedicated
test files that call their HANDLER directly, so they are measured; they simply
never travel through the registry in a test.  ``set_latex_name`` and
``pending_followup`` had neither: no corpus line, no handler test, no route.

**On writing the inputs.**  Everywhere else this project insists the corpus
supply the spellings, because inventing them is how three regressions walked past
5813 tests.  There is no corpus line here to use — that is the finding.  The
inputs below are therefore taken from the system's own committed data: the
``latex_name_kw`` pattern in ``i18n/nl_patterns.json`` and the three conditions
the handler itself states (a keyword, a REGISTERED quantity name, a LaTeX
string).  That is the narrowest honest source available, and it is written down
here so a reader can see the difference.

Bilingual, per ``feedback_bilingual_unit_tests``.
"""
import asyncio

import pytest

import ui.interpret_registry as reg
import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline


def _run(engine, supervisor, value):
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[InputToken(type="text", value=value)]),
        engine, supervisor))


@pytest.fixture(params=["ja", "en"])
def session(request):
    nb.start_engine(title="unreached", mode="standard", language=request.param,
                    domains=[])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=[], primary_backend="maxima")
    try:
        yield engine, supervisor, request.param
    finally:
        nb.stop_engine()


@pytest.fixture
def routes(monkeypatch):
    """Every interceptor that WINS during the test — the same instrumentation
    the coverage was measured with, so the test asserts the thing that was
    missing (a win), not merely that the request succeeded."""
    seen: list = []
    original = reg.run_interceptors

    async def _record(ctx, interceptors):
        out = await original(ctx, interceptors)
        if getattr(ctx, "route", None):
            seen.append(ctx.route)
        return out

    for mod in ("ui.interpret_rules", "ui.interpret_early",
                "ui.interpret_equation_intercepts", "ui.interpret_pipeline"):
        m = __import__(mod, fromlist=["run_interceptors"])
        if hasattr(m, "run_interceptors"):
            monkeypatch.setattr(m, "run_interceptors", _record)
    monkeypatch.setattr(reg, "run_interceptors", _record)
    return seen


class TestSetLatexName:
    r"""`「x の LaTeX名を \phi にして」` — 2g, the rule-based short circuit."""

    def test_the_interceptor_wins_and_sets_the_name(self, session, routes):
        engine, supervisor, lang = session
        engine.run_operation("define_scalar_parameter",
                             {"name": "x", "_all_names": ["x"]})
        text = (r"x の LaTeX名を \phi にして" if lang == "ja"
                else r"set the latex name of x to \phi")
        res = _run(engine, supervisor, text)
        assert "set_latex_name" in routes, (routes, res)
        assert engine.symbol_context.get_quantity("x").latex_name == r"\phi"

    def test_it_declines_without_a_registered_quantity(self, session, routes):
        """Condition (2) from the handler: the name must be REGISTERED.  Without
        this the rule would claim any sentence carrying the keyword and a
        command — the NL-GRAMMAR ② shape."""
        engine, supervisor, lang = session
        text = (r"zzz の LaTeX名を \phi にして" if lang == "ja"
                else r"set the latex name of zzz to \phi")
        _run(engine, supervisor, text)
        assert "set_latex_name" not in routes, routes


class TestPendingFollowup:
    """A pending operation missing ONE argument, answered by the next input."""

    def test_a_bare_variable_completes_a_pending_differentiate(self, session,
                                                               routes):
        engine, supervisor, lang = session
        expr = engine.register_expression("x^2 + a*x")
        engine.pending_op = {"operation": "differentiate",
                             "args": {"expr": expr.display_id},
                             "missing": ["var"]}
        res = _run(engine, supervisor, "x")
        assert "pending_followup" in routes, (routes, res)
        assert engine.pending_op is None, "the pending op must be cleared"

    def test_it_declines_when_nothing_is_pending(self, session, routes):
        """LAST in its registry precisely so it never pre-empts another rule;
        with no pending op it must not claim the same input."""
        engine, supervisor, _ = session
        engine.register_expression("x^2 + a*x")
        assert engine.pending_op is None
        _run(engine, supervisor, "x")
        assert "pending_followup" not in routes, routes


class TestTheDeclarationBypassIsNotNarrowedByLanguage:
    r"""`should_bypass_auto_register_formula` must answer for BOTH languages.

    It is a GUARD, not a reading: it asks "could this sentence be a
    declaration?" so that the subject reaches the declaration path instead of
    being auto-registered as a formula.  The two mistakes are not symmetric — a
    false positive merely skips auto-registration, while a false NEGATIVE
    registers a declaration as a formula and reports SUCCESS, which is the
    NL-GRAMMAR ② defect (`Define k, m, \mu, \hbar as scalars`, fourteen
    scenarios).

    Pinned 2026-08-23 after migrating the pattern to nl_patterns.json narrowed
    it to one language and broke exactly this: a handler with no engine bound
    answers `DEFAULT_LANG` ("en"), so a Japanese declaration stopped being
    recognised.  **The instruction corpus did not catch it** — scenarios run
    each language in its own session, so the crossing never occurs there; the
    unit test did.
    """

    @pytest.fixture(scope="class")
    def jhandler(self):
        # Built the way test_llm_supervisor builds it — and deliberately with NO
        # engine bound, because that is the case the narrowing broke: with no
        # engine the handler answers `DEFAULT_LANG`.
        from tests.test_llm_supervisor import _make_julia_handler
        return _make_julia_handler()

    @pytest.mark.parametrize("formula,op_text", [
        ("a'(k)", "を光子の生成演算子とする"),
        ("b", "の消滅演算子をbとする"),
        ("c", "define c as an electron annihilation operator"),
        ("f", "f as a fermion operator"),
    ])
    def test_both_languages_are_recognised(self, jhandler, formula, op_text):
        assert jhandler.should_bypass_auto_register_formula(formula, op_text), op_text

    def test_an_ordinary_instruction_is_not_claimed(self, jhandler):
        """The guard is permissive, not indiscriminate."""
        assert not jhandler.should_bypass_auto_register_formula(
            "x^2 + 1", "を微分して")
