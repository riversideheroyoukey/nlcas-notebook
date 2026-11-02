r"""`assume(...)` is a MAXIMA function, and the shared layer was writing it.

open_items BP-5.  `NotebookEngine.add_assumption` built the string
``f"assume({...})"`` itself and handed it to whatever adapter the session had.
In a Julia session the researcher's "X > 0 と仮定する" therefore came back as::

    Sym assume is not callable. Use @syms assume(var1, var2,...) to create it

— raw CAS text, naming a function the researcher never wrote (Core Principle
#1), for a limitation nobody had told them about.  This is the Backend
Encapsulation Rule's own example shape: a backend-specific command assembled in
`engine.py`.

The command now comes from the handler.  Maxima names `assume`/`forget`; a
backend that names neither refuses with a reason IN THE RESEARCHER'S LANGUAGE,
and says what to do instead.
"""
import pytest

from notebook.runtime.backend_handler import UnsupportedAssumption
from notebook.runtime.i18n import t
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler


class TestMaximaIsUnchanged:
    """The point of the move is that the Maxima session cannot tell."""

    @pytest.mark.parametrize("canonical,expect", [
        ("x > 0", "assume(x > 0)"),
        ("n >= 1", "assume(n >= 1)"),
        # `a # b` is the one condition Maxima spells differently in an
        # assumption; that rewrite already lived in `assumption_canonical` and
        # is reached through the new door unchanged.
        ("a # b", "assume(notequal(a, b))"),
    ])
    def test_assume_command(self, canonical, expect):
        assert MaximaBackendHandler(None).assumption_command(canonical) == expect

    @pytest.mark.parametrize("canonical,expect", [
        ("x > 0", "forget(x > 0)"),
        ("a # b", "forget(notequal(a, b))"),
    ])
    def test_forget_command(self, canonical, expect):
        assert MaximaBackendHandler(None).forget_command(canonical) == expect

    def test_forget_now_uses_the_same_spelling_as_assume(self):
        """`remove_assumption` used to send the RAW canonical, so an assumption
        asserted as `notequal(a,b)` was withdrawn as `a # b`.  Measured
        2026-08-17 both spellings work in Maxima's `forget`, so this is
        consistency rather than a repair — but the two must not drift, and only
        one of them is what `assume` was given."""
        h = MaximaBackendHandler(None)
        assert h.forget_command("a # b") == h.assumption_command("a # b").replace(
            "assume(", "forget(")


class TestABackendWithoutAssumptionsSaysSo:
    def test_julia_refuses_instead_of_letting_the_cas_answer(self):
        with pytest.raises(UnsupportedAssumption) as exc:
            JuliaBackendHandler(None).assumption_command("x > 0")
        assert exc.value.i18n_key == "error.julia_assumption_unsupported"

    def test_nothing_to_withdraw_is_not_a_refusal(self):
        """An assumption that was never sent has nothing to take back — the
        notebook still drops it, which is the half that was always ours."""
        assert JuliaBackendHandler(None).forget_command("x > 0") is None

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_reason_is_translated_and_names_a_way_forward(self, lang):
        msg = t("error.julia_assumption_unsupported", lang, condition="x > 0")
        assert msg != "error.julia_assumption_unsupported"
        assert "x > 0" in msg
        # It points at the handoff, the way the sibling Julia refusal does —
        # a refusal with no way forward is a dead end for the researcher.
        assert "Maxima" in msg

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_generic_reason_does_not_name_one_backend_s_library(self, lang):
        """The base class raises the GENERIC key.  Naming Symbolics.jl there
        would put one backend's knowledge in the shared layer — the very thing
        this change removes from `engine.py`."""
        msg = t("error.assumption_unsupported", lang, condition="x > 0")
        assert msg != "error.assumption_unsupported"
        assert "Symbolics" not in msg


class TestThroughTheEngine:
    def test_a_maxima_session_records_the_rendered_command(self):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="asm") as e:
            e.start()
            step, _ = e.add_assumption("x > 0")
            assert step.rendered_command == "assume(x > 0)"
            removed = e.remove_assumption_by_canonical("x > 0")
            assert removed.rendered_command == "forget(x > 0)"

    def test_a_refusal_reaches_the_caller_as_the_translated_reason(self):
        """Not as `UnsupportedAssumption`, and not as the i18n KEY: the engine
        knows the session language, the handler does not."""
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="asm") as e:
            e.start()
            e._handler.__class__._ASSUME_FN = ""      # pretend: no mechanism
            try:
                with pytest.raises(ValueError) as exc:
                    e.add_assumption("x > 0")
                assert "error.assumption_unsupported" not in str(exc.value)
                assert "x > 0" in str(exc.value)
            finally:
                e._handler.__class__._ASSUME_FN = "assume"
