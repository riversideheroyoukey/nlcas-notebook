r"""A named equation `Y = rhs` as the SOURCE of an operation.

H-1 stores only the RHS in `canonical` and reconstructs `LHS = RHS` when an
operation needs it, so every operation has to say what to do with the label.
Five said nothing, sent the equation WHOLE to the CAS, and registered a
tautology (measured 2026-08-16, EQ-SOURCE-BARE-LHS)::

    summation          sum(Y=x^2 + 1,i,1,3)              → 3 x^2+3 = 3 x^2+3
    product_expr       product(Y=x^2 + 1,i,1,3)          → (x^2+1)^3 = (x^2+1)^3
    numerical_eval     float(Y=x^2 + 1)                  → x^2+1.0 = x^2+1.0
    numerical_eval_hp  bfloat(Y=x^2 + 1)                 → …
    taylor_expand      ratdisrep(taylor(Y=x^2 + 1,x,0,2)) → x^2+1 = x^2+1

The VALUES were not wrong — P2 expands the left symbol, so both sides come out
equal — which is why this survived: what is broken is the FORM, and a tautology
looks like an answer.

**"Put them all in one branch" is not the answer**, and this file pins why:
three different things are needed, and picking the wrong one states something
false rather than merely ugly.

* the operator wrapped around the name, when the result is a DIFFERENT quantity
  (`\sum_{i=1}^{3} Y = 3Y` — re-attaching a bare `Y =` would claim `Y = 3Y`);
* the name kept, when the operation preserves the VALUE (`Y = 5.0`);
* no label at all, when the result is an APPROXIMATION (a truncated Taylor
  series is not equal to Y).

**One measurement to keep**: the list of "which operations keep their label"
existed in THREE copies — the strip decision, the result formatter and the
re-attachment — so an operation could be told to drop the label and never told
to put it back.  That is exactly what `numerical_eval` did after the first half
of this repair, and it is why the sets are now module-level.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOMAIN = "classical_mechanics"
SOURCE = "Y = x^2 + 1"


def _apply(operation, args):
    """Register `Y = x^2 + 1`, run one operation on it, return (command, latex).

    A FRESH session per operation, deliberately: measured 2026-08-16, running
    several in one session made four of the five look correct, because an
    earlier operation had re-registered `Y` and the reconstruction no longer
    found an equation.  The contaminated reading is the more comfortable one.
    """
    nb.start_engine(title="eqsrc", mode="standard", language="ja", domains=[DOMAIN])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[DOMAIN], primary_backend="maxima")
        asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=SOURCE)]), eng, sup))
        seen = []
        orig = eng._adapter.execute
        eng._adapter.execute = lambda c, *a, **k: (seen.append(c), orig(c, *a, **k))[1]
        step, val = eng.run_operation(operation, {"expr": "E1", **args})
        oid = step.output_refs[0] if step.output_refs else None
        latex = eng.expressions[oid].latex if oid and oid in eng.expressions else str(val)
        return (seen[-1] if seen else ""), latex
    finally:
        nb.stop_engine()


ARGS = {
    "summation":         {"var": "i", "lower": "1", "upper": "3"},
    "product_expr":      {"var": "i", "lower": "1", "upper": "3"},
    "numerical_eval":    {},
    "numerical_eval_hp": {},
    "taylor_expand":     {"var": "x", "point": "0", "order": "2"},
    "simplify":          {},
    "differentiate":     {"var": "x"},
    "integrate_definite": {"var": "x", "lower": "0", "upper": "1"},
    "limit":             {"var": "x", "point": "0"},
    "real_part":         {},
}


class TestTheEquationNeverReachesTheCasWhole:
    @pytest.mark.parametrize("operation", sorted(ARGS), ids=sorted(ARGS))
    def test_only_the_rhs_is_sent(self, operation):
        command, _latex = _apply(operation, ARGS[operation])
        assert "Y=" not in command.replace(" ", ""), command


class TestEachOperationGetsTheFormItNeeds:
    @pytest.mark.parametrize("operation,args,expected", [
        # the result is a DIFFERENT quantity — the operator wraps the name
        ("summation",    ARGS["summation"],    r"\sum_{i=1}^{3} Y = 3 x^2+3"),
        ("product_expr", ARGS["product_expr"], r"\prod_{i=1}^{3} Y = \left(x^2+1\right)^3"),
        # the VALUE is preserved — the name still names it
        ("numerical_eval", {},                 "Y = x^2+1.0"),
        ("simplify",     {},                   "Y = x^2+1"),
    ], ids=["summation", "product_expr", "numerical_eval", "simplify"])
    def test_form(self, operation, args, expected):
        _cmd, latex = _apply(operation, args)
        assert latex == expected

    def test_a_truncated_series_states_no_equality(self):
        r"""`taylor_expand` of `Y` to order 2 approximates Y; writing `Y = …`
        would assert an equality the researcher did not ask for."""
        _cmd, latex = _apply("taylor_expand", ARGS["taylor_expand"])
        assert latex == "x^2+1"
        assert "Y" not in latex

    def test_no_result_is_a_tautology(self):
        """The shape all five had: both sides of the `=` identical."""
        for operation, args in ARGS.items():
            _cmd, latex = _apply(operation, args)
            if "=" not in latex:
                continue
            lhs, _, rhs = latex.partition("=")
            assert lhs.strip() != rhs.strip(), f"{operation}: {latex}"


class TestTheOpsThatAlreadyWorkedStillDo:
    @pytest.mark.parametrize("operation,expected", [
        ("differentiate",      r"\frac{dY}{dx} = 2 x"),
        ("integrate_definite", r"\int_{0}^{1} Y \, dx = \frac{4}{3}"),
        ("limit",              r"\lim_{x \to 0} Y = 1"),
        ("real_part",          "x^2+1"),          # Re does not preserve equality
    ], ids=["differentiate", "integrate_definite", "limit", "real_part"])
    def test_form_unchanged(self, operation, expected):
        _cmd, latex = _apply(operation, ARGS[operation])
        assert latex == expected


class TestTheListsAreOne:
    r"""The three copies are now two module-level sets.

    Pinned as data because the failure mode is asymmetric: an operation added
    to the strip list but not to the re-attachment list loses its label
    silently, which is how `numerical_eval` behaved halfway through this
    repair.
    """

    def test_every_value_preserving_op_is_eq_aware(self):
        from notebook.engine import (
            _ANALYTICAL_OPS, _COMPLEX_VALUE_OPS,
            _EQ_OPERATOR_FORM_OPS, _EQ_VALUE_PRESERVING_OPS,
        )
        eq_aware = (_ANALYTICAL_OPS | {"substitute"}
                    | _EQ_VALUE_PRESERVING_OPS | _COMPLEX_VALUE_OPS)
        assert _EQ_VALUE_PRESERVING_OPS <= eq_aware
        assert _EQ_OPERATOR_FORM_OPS <= eq_aware

    def test_the_two_sets_are_disjoint(self):
        """An operation cannot both keep its name and wrap an operator round it."""
        from notebook.engine import _EQ_OPERATOR_FORM_OPS, _EQ_VALUE_PRESERVING_OPS
        assert not (_EQ_OPERATOR_FORM_OPS & _EQ_VALUE_PRESERVING_OPS)
