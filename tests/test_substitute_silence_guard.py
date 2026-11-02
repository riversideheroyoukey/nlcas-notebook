"""substitute must never report success without substituting (docs/32 Phase 1a).

Maxima's ``subst()`` on an ATOMIC lhs is purely syntactic: an lhs that appears
in no spelling in the source cannot match, and the CAS reports that as success
with the expression unchanged.  That is the shape of open_items
AUDIT-SUBST-INDEXED — ``subst(V(i)=5, a*V_i + b)`` returned ``success`` and
``b+V_i*a``, so the researcher carried a value forward that was never
substituted.  The guard refuses instead.

The two deliberate exemptions are pinned here too, because both look like
"the lhs is not in the text" and both would be false alarms:
  * ratsubst's ALGEBRAIC match (H-10) — a compound lhs need not appear verbatim;
  * an H-1 P2 CAS-bound symbol — Maxima expands ``H`` before subst() runs.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.i18n import t


def _substitute(engine, expr_id, lhs, rhs):
    return engine.run_operation(
        "substitute", {"expr": expr_id, "lhs": lhs, "rhs": rhs})


class TestRefusesWhenNothingCanMatch:
    # NOTE: the AUDIT-SUBST-INDEXED case itself (`V_i` into `a*V_i + b`) is no
    # longer refused — docs/32 Phase 3 made expr_lookup carry backend code, so
    # the source and the lhs now speak the same language and the substitution
    # SUCCEEDS.  See tests/test_subst_indexed_root_fix.py.  This guard remains
    # for what it was always about: an lhs that genuinely is not there.

    def test_symbol_absent_from_expression_is_refused(self):
        with NotebookEngine(title="guard2") as e:
            ex = e.register_expression("a*x + b")
            with pytest.raises(ValueError):
                _substitute(e, ex.display_id, "z", "7")

    def test_message_is_i18n_keyed_in_both_languages(self):
        """The refusal reaches the researcher, so it must not be a hardcoded
        literal (i18n rule)."""
        ja = t("engine.substitute_lhs_not_found", "ja", lhs="V_i")
        en = t("engine.substitute_lhs_not_found", "en", lhs="V_i")
        assert ja != en
        assert "V_i" in ja and "V_i" in en
        assert "engine.substitute_lhs_not_found" not in ja  # key resolved


class TestStillSubstitutesWhenItCan:

    def test_declared_subscripted_name(self):
        """omega_0 declared as a NAME lowers to omega_0, so it matches."""
        with NotebookEngine(title="guard3") as e:
            e.register_quantity("omega_0", kind="scalar")
            ex = e.register_expression("omega_0^2*q + omega_0*p")
            st, val = _substitute(e, ex.display_id, "omega_0", "3")
            assert st.status.value == "success"
            assert val == "9*q+3*p"

    def test_plain_symbol(self):
        with NotebookEngine(title="guard4") as e:
            ex = e.register_expression("a*x + b")
            st, val = _substitute(e, ex.display_id, "x", "7")
            assert st.status.value == "success"
            assert val == "b+7*a"

    def test_cas_bound_symbol_is_not_refused(self):
        """H-1 P2 sent ``H : J*x*y`` to Maxima, so ``x`` IS reachable from the
        source ``H + K`` even though the text does not contain it."""
        with NotebookEngine(title="guard5") as e:
            e.register_expression("H = J*x*y")
            src = e.register_expression("H + K")
            st, val = _substitute(e, src.display_id, "x", "2")
            assert st.status.value == "success"
            assert val == "2*J*y+K"

    def test_compound_lhs_keeps_the_algebraic_path(self):
        """H-10: a compound lhs goes through ratsubst, whose match is algebraic;
        the syntactic guard must not intercept it."""
        with NotebookEngine(title="guard6") as e:
            ex = e.register_expression("a + b + c")
            st, val = _substitute(e, ex.display_id, "a + b", "0")
            assert st.status.value == "success"
            assert val == "c"
