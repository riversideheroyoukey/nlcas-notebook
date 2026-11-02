r"""FREE-PARAM and Laplace split expressions they then send to the CAS.

ONE-QUESTION-MANY-ANSWERS ratchet, ninth lowering: `engine_free_params_mixin.py`
2→0, `engine_laplace_mixin.py` 2→0, `engine_persistence_mixin.py` 2→0.
Counting where the six values went put all six into a CAS command, a REGISTERED
expression, or a load-time repair decision.

Measured 2026-08-13 on `\sum_{i=1}^{3} c_i = u*x + v*y`:

    substitute_free_parameters   subst([u=(2)], \sum_{i)
                                 subst([u=(2)], 1}^{3} c_i = u*x + v*y)
                                 → registered `_res_ = _res_`
    laplace_by_definition        integrate((1}^{3} c_i = u*x + v*y)*exp(-s t), …)

The `laplace_by_definition` row is kept as the RECORD of what was measured; its
test is gone because the operation is (2026-08-15, SCENARIO-MACRO case 7 — it
was a macro).  An assertion about the command shape of an operation that no
longer exists guards nothing; the split it shared with the rows above is still
pinned by them.

`_res_` is the ADAPTER'S OWN SCRATCH NAME: both calls failed, Maxima echoed the
scratch variable, and the expression was registered from it — the signature the
solve builder was repaired for in 第3期, alive in a second place.

The two `engine_persistence_mixin` sites are load-time repair migrations gated on
`_WRONG_FUNC_RE` as well; no input was found that changes their outcome, so they
were routed through the one definition without a measured fix behind them.
"""
import pytest

from notebook.engine import NotebookEngine

SUM_EQUATION = r"\sum_{i=1}^{3} c_i = u*x + v*y"
LEFT_SIDE = r"\sum_{i=1}^{3} c_i"
MANGLED = "1}^{3} c_i"


def head_lost(command: str) -> bool:
    r"""The remainder WITHOUT its `\sum_{i=` head.

    `MANGLED in command` would fire on the correct answer, since it is a
    substring of the correct left side.  Fourth time in this ratchet's tests.
    """
    return MANGLED in command.replace(LEFT_SIDE, "")


@pytest.fixture
def engine_with_sum():
    with NotebookEngine(title="mixin_split") as engine:
        engine.start()
        expr = engine.register_expression(SUM_EQUATION)
        expr.free_parameters = ["u"]
        engine.sent = []
        original = engine._adapter.execute

        def spy(command, *a, **kw):
            engine.sent.append(command)
            return original(command, *a, **kw)

        engine._adapter.execute = spy
        yield engine, expr


class TestTheCommandsCarryTheWholeSide:
    def test_free_parameter_substitution(self, engine_with_sum):
        engine, expr = engine_with_sum
        engine.substitute_free_parameters(expr.display_id, [("u", "2")])
        substs = [c for c in engine.sent if c.startswith("subst(")]
        assert substs, engine.sent
        assert not any(head_lost(c) for c in substs), substs
        # Both halves arrive WHOLE — which is what this ratchet is about — and
        # each arrives as CAS code, which is BUILDER-OPERAND-NOT-LOWERED's half
        # (closed 2026-08-15; the shapes are pinned in TestTheOperandsAreLowered
        # below).  Asserted on the head being present rather than on an exact
        # spelling, so a change in lowering does not read as a split regression.
        assert any("u * x + v * y" in c or "u*x + v*y" in c for c in substs), substs
        assert any("sum(c(i)" in c for c in substs), substs

    def test_specialize_by_conditions(self, engine_with_sum):
        """Added after a mutation of this site changed nothing in the tests.

        The first version of this class covered `substitute_free_parameters` and
        the Laplace body but not `specialize_by_conditions`, so reverting that
        one site passed — the third time in this ratchet that a mutation exposed
        an unmeasured site rather than a weak assertion.
        """
        engine, expr = engine_with_sum
        try:
            engine.specialize_by_conditions(
                expr.display_id, [{"order": 0, "point": "0", "value": "1"}],
                depvar="c", indepvar="x")
        except Exception:
            pass                      # solving is not what is asserted here
        solves = [c for c in engine.sent if "solve(" in c]
        assert solves, engine.sent
        assert not any(head_lost(c) for c in solves), solves
        assert "subst(x=(0), u*x + v*y)=(1)" in solves[0], solves[0]


class TestTheOperandsAreLowered:
    r"""Closed 2026-08-15 (open_items BUILDER-OPERAND-NOT-LOWERED).

    The left side used to reach Maxima as the researcher's own LaTeX, which it
    cannot read, so that half came back as the ADAPTER'S SCRATCH NAME and the
    registered expression read `_res_ = v*y+2*x`.  Both halves are lowered now;
    the substitution PAIRS stay raw, because they are names and values the
    caller already validated — lowering one side and not the other is what
    CLAUDE.md records as the way this goes wrong.

    The aggregate is also held as a NOUN (`'sum(...)`): without that, Maxima
    evaluates it while merely carrying it, and a sum the substitution does not
    touch comes back as its three terms.
    """

    def test_the_left_side_reaches_maxima_as_cas_code(self, engine_with_sum):
        engine, expr = engine_with_sum
        engine.substitute_free_parameters(expr.display_id, [("u", "2")])
        substs = [c for c in engine.sent if "subst([u=(2)]" in c]
        assert any("'sum(c(i), i, 1, 3)" in c for c in substs), substs
        assert not any("\\sum" in c for c in substs), substs

    def test_the_result_is_not_the_adapters_scratch_name(self, engine_with_sum):
        engine, expr = engine_with_sum
        out = engine.substitute_free_parameters(expr.display_id, [("u", "2")])
        assert "_res_" not in out.canonical, out.canonical
        assert "v*y+2*x" in out.canonical.replace(" ", ""), out.canonical
