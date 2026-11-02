"""SUBST-DIGIT-SUB: a registered digit-subscripted scalar (a_1, b_2, ...) must
resolve to that atom, not an IndexedSymbol/function-call (a(1)).

Found via the S-s practice-loop scenario (2-variable LP vertex): solving a
constraint system and substituting the result back into the objective
corrupted every digit-subscripted coefficient into a bogus function call,
because `build_sub_name` in semantic_resolver.py only handled a Symbol
subscript (e.g. k_B), not the Number node a bare digit tokenizes to.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import MathType, Quantity


def _run(eng, sup, value):
    tok = InputToken(type="text", value=value)
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestDigitSubscriptResolverUnit:
    def test_registered_digit_subscript_resolves_to_atom_not_indexed(self):
        ctx = SymbolContext()
        ctx.register("b_2", Quantity(name="b_2", kind="scalar",
                                     math_type=MathType(algebra="scalar")))
        parser = SharedParser()
        resolver = MaximaSemanticResolver()
        surf = parser.parse("a_1 * b_2")
        # a_1 is NOT registered here — should fall back to IndexedSymbol a(1);
        # b_2 IS registered — must resolve to the atom "b_2", not b(2).
        code = MaximaCodeGenerator(ctx).generate(resolver.resolve(surf, ctx))
        assert "b_2" in code and "b(2)" not in code
        assert "a(1)" in code   # unregistered digit-subscript still indexes


class TestDigitSubscriptEndToEnd:
    """Full engine round-trip: solve_system result substituted into another
    expression must not corrupt digit-subscripted coefficients."""

    def test_solve_then_substitute_preserves_digit_subscripts(self):
        nb.start_engine(title="digit-subscript e2e", mode="standard",
                        language="ja", domains=["operations_research"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["operations_research"],
                                primary_backend=eng.session.primary_backend)
            for cmd in [
                "a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 を実数とする",
                "z = c_1 x + c_2 y",
                "a_1 x + b_1 y - d_1 = 0",
                "a_2 x + b_2 y - d_2 = 0",
                "E2 と E3 の連立方程式を x と y について解いて",   # → E4, E5
                "E1 に E4 を代入して",                              # → E6
                "E6 に E5 を代入して",                              # → E7
            ]:
                r = _run(eng, sup, cmd)
                assert r["status"] in ("registered", "confirmed", "completed"), (cmd, r)
            # The numbers moved on 2026-08-06 (INTERNAL-INTERMEDIATE-EID-GAP).
            # This test used to read `E1 に E4 を代入して` then `E7 に E6 を代入して`
            # — note the SKIPPED E5: solve_system allocated a slot, registered its
            # first row under a fresh number and copied it back, so the two
            # solutions came out as E4 and E6.  The gap is gone (the first row is
            # registered into the slot directly), so the solutions are E4 and E5
            # and everything after shifts down by one.  Having to write a hole
            # into a test to make it pass is what the ticket described.
            final = eng.expressions["E7"].canonical
            for corrupted in ("a(1)", "a(2)", "b(1)", "b(2)", "c(1)", "c(2)",
                             "d(1)", "d(2)"):
                assert corrupted not in final, final
            # z* = [c1(b2 d1 - b1 d2) + c2(a1 d2 - a2 d1)] / (a1 b2 - a2 b1)
            for atom in ("a_1", "a_2", "b_1", "b_2", "c_1", "c_2", "d_1", "d_2"):
                assert atom in final
        finally:
            nb.stop_engine()
