r"""A CAS question must not swallow the session (open_items SIGN-DIALOG-ASSUME).

Maxima stops and ASKS when a fact it needs is missing — "Is omega_D positive,
negative or zero?" from ``integrate``.  Measured 2026-08-01: it prints the
question and blocks on stdin, so the rest of the command block we already
wrote is eaten as invalid answers, nothing prints the end marker, the read
times out after 30 s, and every later command goes into a process still
waiting for an answer.  One unanswered question killed the whole session, with
no diagnosis anywhere — the S-c handoff measurement (docs/34 §6.2) lost four
commands that way.

We do NOT answer for the researcher: choosing pos/neg/zero is a modeling
decision (CLAUDE.md — no hardcoded modeling choices) and a wrong guess returns
a plausible wrong integral.  The adapter detects the question, interrupts the
computation (SIGINT — the only exit that answers nothing), resynchronises, and
reports what was asked.
"""
import time

import pytest

from notebook.runtime.i18n import t
from notebook.runtime.maxima_adapter import MaximaAdapter, QUESTION_MARKER

_ASKING_INTEGRAL = "integrate(1/sqrt(xi^2 + Delta^2), xi, 0, omega_D)"
_KEY = "error.cas_asked_question"


@pytest.fixture
def ad():
    a = MaximaAdapter(timeout=30.0)
    a.start()
    yield a
    a.close()


class TestQuestionIsReportedNotHung:
    def test_question_fails_fast_with_the_question_text(self, ad):
        t0 = time.monotonic()
        r = ad.execute(_ASKING_INTEGRAL)
        elapsed = time.monotonic() - t0
        assert r.ok is False
        assert QUESTION_MARKER in (r.error or "")
        assert "omega_D" in r.error
        # The point of detecting it: no 30-second wait for a process that is
        # never going to answer.
        assert elapsed < 10.0, f"took {elapsed:.1f}s — did detection fire?"

    def test_session_survives_the_question(self, ad):
        """The four commands after the question used to be lost too."""
        assert ad.execute(_ASKING_INTEGRAL).ok is False
        assert ad.execute("2+3").value == "5"
        assert ad.execute("diff(x^3, x)").value == "3*x^2"

    def test_the_same_computation_succeeds_once_told(self, ad):
        """Interrupting must not poison the computation itself — supplying the
        missing fact is all it should take."""
        assert ad.execute(_ASKING_INTEGRAL).ok is False
        ad.execute("assume(omega_D > 0)")
        ad.execute("assume(Delta > 0)")
        r = ad.execute(_ASKING_INTEGRAL)
        assert r.ok is True
        assert r.value.replace(" ", "") == "asinh(omega_D/Delta)"

    def test_ordinary_results_are_not_mistaken_for_questions(self, ad):
        """Only the PREAMBLE is scanned, so a result that reads like a question
        is still a result."""
        r = ad.execute('"Is this a question?"')
        assert r.ok is True
        assert "Is this a question?" in r.value


class TestQuestionMessageIsActionable:
    def _handler(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        return MaximaBackendHandler(builder=None)

    def test_handler_classifies_and_extracts(self):
        h = self._handler()
        err = f"{QUESTION_MARKER} Is omega_D positive, negative or zero?"
        assert h.classify_cas_error(err) == _KEY
        assert h.cas_error_args(err) == {
            "question": "Is omega_D positive, negative or zero?"}

    def test_unrelated_errors_are_left_alone(self):
        h = self._handler()
        assert h.classify_cas_error("some other Maxima error") is None
        assert h.cas_error_args("some other Maxima error") == {}

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_message_quotes_the_question_in_both_languages(self, lang):
        msg = t(_KEY, lang, question="Is omega_D positive, negative or zero?")
        assert msg != _KEY
        assert "omega_D" in msg          # names the symbol the CAS asked about
        assert "{question}" not in msg   # substituted, not literal


class TestEngineSurfacesTheMessage:
    def test_run_operation_raises_the_friendly_message(self):
        """The engine translates via classify_cas_error + cas_error_args, so
        the researcher never sees the raw marker."""
        import ui.notebook_bridge as nb
        nb.start_engine(title="q1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        try:
            with pytest.raises(ValueError) as ei:
                eng.run_operation("integrate_definite", {
                    "expr": "1/sqrt(xi^2 + Delta^2)", "var": "xi",
                    "lower": "0", "upper": "omega_D"})
            msg = str(ei.value)
            assert QUESTION_MARKER not in msg
            # The message quotes whatever was asked.  WHICH symbol Maxima asks
            # about first is its own business (measured: `Delta` here, `omega_D`
            # when the same integral is sent to the adapter directly), so the
            # assertion is that a question about one of the expression's
            # symbols reached the researcher — not which one.
            assert "Is " in msg and "?" in msg
            assert ("Delta" in msg) or ("omega_D" in msg) or ("xi" in msg)
            # and the session is still alive afterwards
            assert eng._adapter.execute("2+3").value == "5"
        finally:
            nb.stop_engine()
