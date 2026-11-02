"""Recognising Maxima failures that are really the researcher's next action.

Kept beside the handler rather than inside it (the same reason
``maxima_subst_guard`` is its own module): the handler is already the largest
file in the runtime, and error signatures are a list that only grows.  It is
still Maxima-specific code — the Backend Encapsulation Rule is about which
backend owns the knowledge, not which file holds it.
"""
from notebook.runtime.maxima_adapter import QUESTION_MARKER


def classify(error: str) -> "str | None":
    """Map a raw Maxima failure to an i18n key, or None to pass it through.

    ``QUESTION_MARKER`` is written by the adapter when Maxima stopped to ask a
    question (usually about a sign) instead of computing.  The adapter
    interrupts rather than answering — picking pos/neg/zero would be a
    modeling decision made on the researcher's behalf — so what reaches here
    is "the CAS needs a fact you have not given it", which is actionable.
    """
    if error and QUESTION_MARKER in error:
        return "error.cas_asked_question"
    return None


def message_args(error: str) -> dict:
    """The question text, so the message can quote what Maxima actually asked."""
    if error and QUESTION_MARKER in error:
        return {"question": error.split(QUESTION_MARKER, 1)[1].strip()}
    return {}
