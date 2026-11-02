"""AUDIT-SUBST-INDEXED, root fix: expr_lookup carries backend code (docs/32 Phase 3).

The defect was an asymmetry inside ONE command.  ``_build_substitute`` lowered
its ``lhs``/``rhs`` through the IR (``V_i`` -> ``V(i)``) but took the expression
BODY verbatim from ``expr_lookup`` (the stored canonical, still ``a*V_i + b``),
so Maxima was asked to replace something that was not there — and reported
success, unchanged.  Now the whole lookup is lowered at one boundary, so a
source body and an operation argument cannot disagree.

Every test here checks the VALUE.  A status-only assertion would have passed
against the bug.
"""
import pytest

from notebook.engine import NotebookEngine


def _substitute(engine, expr_id, lhs, rhs):
    return engine.run_operation(
        "substitute", {"expr": expr_id, "lhs": lhs, "rhs": rhs})


class TestTheReportedCase:

    def test_indexed_symbol_substitution_applies(self):
        with NotebookEngine(title="root1") as e:
            ex = e.register_expression("a*V_i + b")
            st, val = _substitute(e, ex.display_id, "V_i", "5")
            assert st.status.value == "success"
            assert val.replace(" ", "") == "b+5*a"

    def test_command_no_longer_mixes_the_two_spellings(self):
        """The smoking gun was `subst(V(i)=5, a*V_i + b)` — a lowered lhs
        against a raw body."""
        with NotebookEngine(title="root2") as e:
            ex = e.register_expression("a*V_i + b")
            st, _ = _substitute(e, ex.display_id, "V_i", "5")
            cmd = st.rendered_command
            assert "V(i)=5" in cmd.replace(" ", "")
            assert "V_i" not in cmd, f"raw surface form still reaches Maxima: {cmd}"


class TestOriginIndependence:
    """The bug made behaviour depend on WHERE the expression came from: a body
    that had been through the CAS once had a lowered canonical and 'happened to
    work', so users saw "it worked a moment ago"."""

    @pytest.mark.parametrize("lhs,rhs,expected", [
        ("V_i", "5", "b+5*a"),
        ("a", "2", "2*V(i)+b"),
    ])
    def test_direct_and_via_cas_agree(self, lhs, rhs, expected):
        with NotebookEngine(title="root3") as e:
            direct = e.register_expression("a*V_i + b")
            st, _ = e.run_operation("expand", {"expr": direct.display_id})
            via_cas = e.expressions[st.output_refs[0]]

            _, v1 = _substitute(e, direct.display_id, lhs, rhs)
            _, v2 = _substitute(e, via_cas.display_id, lhs, rhs)
            assert v1 == v2
            assert v1.replace(" ", "") == expected


class TestDeclarationNoLongerDecidesCorrectness:
    """omega_0 is the common case, not an edge one: it appears in four of the
    repo's saved sessions.  Undeclared it lowered to omega(0) and every
    substitution missed; declared it worked.  Correctness must not hinge on a UI
    step whose necessity the researcher is never told about."""

    @pytest.mark.parametrize("declare", [False, True])
    def test_numeric_subscript_substitutes_either_way(self, declare):
        with NotebookEngine(title=f"root4-{declare}") as e:
            if declare:
                e.register_quantity("omega_0", kind="scalar")
            ex = e.register_expression("omega_0^2*q + omega_0*p")
            st, val = _substitute(e, ex.display_id, "omega_0", "3")
            assert st.status.value == "success"
            assert val.replace(" ", "") == "9*q+3*p"


class TestLoweringBoundary:

    def test_user_form_is_kept_alongside_the_code(self):
        """Builders that need the surface text can still reach it; the raw form
        is preserved under __user__<eid> rather than thrown away."""
        from notebook.runtime.expr_lowering import lower_expr_lookup
        with NotebookEngine(title="root5") as e:
            out = lower_expr_lookup(
                e._handler, {"expr": "E1"}, {"E1": "a*V_i + b"})
            assert out["E1"] == "a * V(i) + b"
            assert out["__user__E1"] == "a*V_i + b"

    def test_unreferenced_expressions_are_left_alone(self):
        from notebook.runtime.expr_lowering import lower_expr_lookup
        with NotebookEngine(title="root6") as e:
            out = lower_expr_lookup(
                e._handler, {"expr": "E1"}, {"E1": "V_i", "E2": "W_j"})
            assert out["E1"] == "V(i)"
            assert out["E2"] == "W_j", "only what the operation refers to is lowered"
            assert "__user__E2" not in out

    def test_untouched_text_adds_no_user_copy(self):
        from notebook.runtime.expr_lowering import lower_expr_lookup
        with NotebookEngine(title="root7") as e:
            out = lower_expr_lookup(e._handler, {"expr": "E1"}, {"E1": "abs(x)"})
            assert out == {"E1": "abs(x)"}

    def test_refused_lowering_fails_the_operation(self):
        """R4: a lowering failure must not put the raw string on the wire."""
        from notebook.runtime.expr_lowering import LoweringRefused, lower_expr_lookup

        class _Boom:
            def is_opaque_handle(self, text): return False
            def parse_ir(self, text): raise RuntimeError("boom")
            def generate_backend_code(self, ir): return "x"

        with pytest.raises(LoweringRefused) as exc:
            lower_expr_lookup(_Boom(), {"expr": "E1"}, {"E1": "whatever"})
        assert exc.value.expr_id == "E1"
        assert "boom" in exc.value.reason

    @pytest.mark.parametrize("key", ["__input__E1", "__sympy__E1"])
    def test_reserved_parallel_lookups_are_never_lowered(self, key):
        """__input__ is the user-facing form Julia rebuilds IR from; lowering it
        would defeat its purpose."""
        from notebook.runtime.expr_lowering import lower_expr_lookup
        with NotebookEngine(title="root8") as e:
            out = lower_expr_lookup(e._handler, {"expr": "E1"},
                                    {"E1": "V_i", key: "a*V_i + b"})
            assert out[key] == "a*V_i + b"
