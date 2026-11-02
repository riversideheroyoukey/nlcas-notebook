"""The CAS's `V(i)` must reach the researcher as `V_i` (docs/32 Phase 4).

An undeclared subscripted symbol lowers to a CAS function call — `V_i` → `V(i)`,
`omega_0` → `omega(0)` — which is the documented policy.  But results come back
in that spelling, so the timeline was showing `\\omega\\left(0\\right)` for what
the researcher typed as `omega_0`: CAS syntax in the notebook, against Core
Principle #1.

Phase 3 made this worse before better.  Substituting onto `V_i` used to do
nothing at all (AUDIT-SUBST-INDEXED), so the leak never appeared on that path;
once the substitution actually worked, its result displayed `V\\left(i\\right)`.
Phase 4 is the counterpart, not a polish step.

`SymbolContext.indexed_bases` records which bases were written subscripted, read
from the IR of what was REGISTERED — a result canonical `a*V(i)+b` parses as an
ordinary function call and no longer says `V` was ever an index base.
"""
import pytest

from notebook.engine import NotebookEngine


def _display_after(engine, expr, operation="expand", **args):
    ex = engine.register_expression(expr)
    step, value = engine.run_operation(operation, dict(args, expr=ex.display_id))
    return value, engine.expressions[step.output_refs[0]].latex


class TestTheLeakIsClosed:

    def test_numeric_subscript(self):
        """omega_0 is the case from the plan's §2.2 — four saved sessions use it."""
        with NotebookEngine(title="p4a") as e:
            value, latex = _display_after(e, "omega_0^2*q + omega_0*p")
            assert "omega(0)" in value.replace(" ", "")     # CAS side unchanged
            assert latex == r"\omega_{0}^2 q+\omega_{0} p"
            assert r"\left(0\right)" not in latex

    def test_letter_subscripts(self):
        with NotebookEngine(title="p4b") as e:
            value, latex = _display_after(
                e, "a*V_i + b*W_j", "substitute", lhs="a", rhs="2")
            assert latex.endswith("b W_{j}+2 V_{i}")

    def test_two_indices(self):
        with NotebookEngine(title="p4c") as e:
            _, latex = _display_after(e, "A_{i,j} + b")
            assert latex == "A_{i,j}+b"


class TestWhatMustNotBeTouched:

    def test_genuine_function_calls(self):
        """f(x) is a function, not an index base — it must keep its parentheses."""
        with NotebookEngine(title="p4d") as e:
            _, latex = _display_after(e, "f(x) + g(y)")
            assert latex == r"g\left(y\right)+f\left(x\right)"

    def test_arithmetic_arguments(self):
        """`sin(x+1)` is not `sin` indexed by `x+1`."""
        with NotebookEngine(title="p4e") as e:
            e.register_expression("V_i")
            _, latex = _display_after(e, "V_i + sin(x+1)")
            assert r"\sin \left(x+1\right)" in latex
            assert "V_{i}" in latex

    def test_declared_symbols_are_unaffected(self):
        with NotebookEngine(title="p4f") as e:
            _, latex = _display_after(e, "x + y")
            assert latex == "y+x"

    def test_a_called_name_of_the_same_family_blocks_restoration(self):
        """`a_i` and `a_i(k)` are DIFFERENT objects by design (CLAUDE.md).
        Restoring `a(i)` → `a_i` next to a real `a_i(k)` would show two
        quantities under one notation, so restoration is skipped and the CAS
        spelling stays visible — honest beats pretty."""
        with NotebookEngine(title="p4g") as e:
            value, latex = _display_after(e, "a_i + a_i(k)")
            assert value.replace(" ", "") == "a_i(k)+a(i)"
            assert r"a\left(i\right)" in latex


class TestRoundTrip:
    """R3: what is displayed must convert back to the same expression.

    Checked as a CAS DIFFERENCE, not string equality — `tex1` reorders terms, so
    a string comparison reports a false gap (and the lesson from AUDIT-POW-DENOM
    is that only the CAS can answer "same expression?").
    """

    @pytest.mark.parametrize("canonical", [
        "a*V(i)+b*W(j)",
        "omega(0)^2*q",
        "A(i,j)+b",
        "V(i)/W(j)^2",
    ])
    def test_display_then_reinput_is_the_same_expression(self, canonical):
        with NotebookEngine(title="p4rt") as e:
            for src in ("a*V_i + b*W_j", "omega_0*q", "A_{i,j}"):
                e.register_expression(src)
            h = e._handler
            raw = e._adapter.execute(f"tex1({canonical})").value
            display = h.postprocess_result_latex(canonical, raw, e.symbol_context)
            relowered = h.to_backend_code(h.latex_to_canonical(display)).code
            diff = e._adapter.execute(f"ratsimp(({canonical}) - ({relowered}))")
            assert diff.value.strip() == "0", (
                f"{display!r} does not come back as {canonical!r}")


class TestRecordingAndPersistence:

    def test_bases_are_recorded_with_their_arity(self):
        with NotebookEngine(title="p4h") as e:
            for s in ("a*V_i + b", "omega_0^2*q", "A_{i,j} + c", "x + y", "a_i(k)"):
                e.register_expression(s)
            assert e.symbol_context.indexed_bases == {"V": 1, "omega": 1, "A": 2}

    def test_survives_save_and_load(self, tmp_path):
        path = tmp_path / "p4.json"
        with NotebookEngine(title="p4i") as e:
            e.register_expression("a*V_i + b")
            e.save(path)
        loaded = NotebookEngine.load(path)
        loaded.start()
        try:
            assert loaded.symbol_context.indexed_bases == {"V": 1}
            _, latex = _display_after(loaded, "V_i + c")
            assert "V_{i}" in latex
        finally:
            loaded.close()

    def test_a_file_without_the_key_loads_unchanged(self, tmp_path):
        """Additive key: an older session file must load and behave as before."""
        import json
        path = tmp_path / "old.json"
        with NotebookEngine(title="p4j") as e:
            e.register_expression("a*V_i + b")
            e.save(path)
        data = json.loads(path.read_text())
        data.pop("indexed_bases")
        path.write_text(json.dumps(data))
        loaded = NotebookEngine.load(path)
        try:
            assert loaded.symbol_context.indexed_bases == {}
        finally:
            pass


class TestCollisionWithACalledName:
    """`V_i` and `V(x)` lower to the same CAS spelling, so a session using both
    is genuinely ambiguous.  The display must not pick one and pretend."""

    def test_restoration_stops_in_either_order(self):
        for first, second in (("a*V_i + b", "V(x) + c"), ("V(x) + c", "a*V_i + b")):
            with NotebookEngine(title=f"clash-{first[:3]}") as e:
                e.register_expression(first)
                _, latex = _display_after(e, second)
                assert r"\left(" in latex, (
                    f"{first!r} then {second!r}: restored despite the collision")

    def test_both_usages_are_recorded(self):
        with NotebookEngine(title="clash-rec") as e:
            e.register_expression("a*V_i + b")
            e.register_expression("V(x) + c")
            assert e.symbol_context.indexed_bases == {"V": 1}
            assert e.symbol_context.called_bases == ["V"]

    def test_builtin_functions_are_not_recorded_as_called_bases(self):
        """`sin(x)` must not make `sin` a conflicting base."""
        with NotebookEngine(title="clash-fn") as e:
            e.register_expression("sin(x) + cos(y)")
            assert e.symbol_context.called_bases == []

    def test_declared_names_are_not_recorded(self):
        with NotebookEngine(title="clash-decl") as e:
            e.register_quantity("f", kind="function", natural_vars=["x"])
            e.register_expression("f(x) + c")
            assert e.symbol_context.called_bases == []


class TestIndexAliasOptIn:
    """docs/07 §8.2: an opt-in lowering that removes the collision outright.

    NOT the default.  Measured 2026-07-30: as a default it broke 47 tests in 9
    files, and — decisively — a new `subst(V__idx(i)=5, …)` does not match the
    `V(i)` already stored in saved sessions, silently doing nothing.  That is
    AUDIT-SUBST-INDEXED all over again, in a session-compatibility disguise.
    """

    def _alias_engine(self, title):
        from notebook.models.quantity import Quantity
        e = NotebookEngine(title=title)
        e.start()
        e.symbol_context.register(
            "V", Quantity(name="V", kind="scalar",
                          index_lowering_style="index_alias"))
        return e

    def test_indexed_and_called_become_different_objects(self):
        e = self._alias_engine("alias1")
        try:
            assert e._handler.to_backend_code("V_i").code == "V__idx(i)"
            assert e._handler.to_backend_code("V(x)").code == "V(x)"
        finally:
            e.close()

    def test_summation_still_expands(self):
        """The reason the alias exists rather than declaring `V_i` a NAME:
        a name makes `sum(V_i, i, 1, 3)` collapse to `3*V_i`, silently."""
        e = self._alias_engine("alias2")
        try:
            ex = e.register_expression("sum(V_i, i, 1, 3)")
            _, value = e.run_operation("expand", {"expr": ex.display_id})
            assert value.replace(" ", "") == "V__idx(3)+V__idx(2)+V__idx(1)"
        finally:
            e.close()

    def test_declaring_the_subscript_as_a_name_breaks_the_sum(self):
        """Pinned as the counter-example, so the alias is not 'simplified' away
        into a name declaration later."""
        with NotebookEngine(title="alias3") as e:
            e.register_quantity("V_i", kind="scalar")
            ex = e.register_expression("sum(V_i, i, 1, 3)")
            _, value = e.run_operation("expand", {"expr": ex.display_id})
            assert value.replace(" ", "") == "3*V_i"     # NOT V(1)+V(2)+V(3)

    def test_default_is_still_the_plain_function_call(self):
        with NotebookEngine(title="alias4") as e:
            assert e._handler.to_backend_code("V_i").code == "V(i)"

