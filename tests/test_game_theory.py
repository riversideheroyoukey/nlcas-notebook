"""Tests for F-17 game-theory operations: pure_nash_equilibria (custom
build_command case, MaximaGameMixin) and mixed_strategy_indifference_conditions
(engine method, notebook/engine_game_mixin.py — registers 2 expressions from
one call, like stationary_conditions).  Payoff/bimatrix REPRESENTATION uses
the existing [a,b;c,d] matrix literal — no new code.
"""
import asyncio

import pytest
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.builder import MaximaCommandBuilder
from notebook.runtime.maxima_game_mixin import parse_matrix_rows

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=["game_theory"], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


@pytest.fixture(scope="module")
def b():
    return MaximaCommandBuilder()


class TestSchemaCompleteness:
    def test_pure_nash_equilibria_in_schema(self, v):
        assert "pure_nash_equilibria" in v.known_operations

    def test_pure_nash_equilibria_has_template(self, b):
        assert "pure_nash_equilibria" in b.supported_operations


class TestParseMatrixRows:
    def test_2x2(self):
        assert parse_matrix_rows("matrix([2, 0], [0, 1])") == [["2", "0"], ["0", "1"]]

    def test_not_a_matrix(self):
        assert parse_matrix_rows("2*x+1") == []


class TestPureNashEquilibria:
    def test_battle_of_the_sexes(self):
        """Classic textbook game: two pure-strategy Nash equilibria at
        (Opera,Opera)=(1,1) and (Football,Football)=(2,2)."""
        nb.start_engine(title="test_f17_nash", mode="standard", language="en",
                         domains=["game_theory"])
        eng = nb.get_engine()
        try:
            a = eng.register_expression("[2, 0; 0, 1]", latex="A")
            b = eng.register_expression("[1, 0; 0, 2]", latex="B")
            step, _ = eng.run_operation(
                "pure_nash_equilibria", {"payoff_a": a.display_id, "payoff_b": b.display_id})
            result = eng.expressions[step.output_refs[0]].canonical
            assert result.replace(" ", "") == "[[1,1],[2,2]]"
        finally:
            nb.stop_engine()

    def test_h1_named_payoff_matrices_not_corrupted(self):
        """Root-cause regression (found 2026-07-19 building the F-17
        generator replay, same bug class as FORM-LAGRANGIAN-H1-TAUTOLOGY):
        registering the payoff matrices with an H-1 symbol name ("A = [...]",
        not a bare label) used to break pure_nash_equilibria — engine.
        _build_expr_lookup's H-1 restoration handed _build_pure_nash_equilibria
        "A = matrix(...)" (LHS attached) as payoff_a, which parse_matrix_rows
        cannot parse as a matrix literal, so shape validation failed. Existing
        tests never caught this because they always registered payoffs with a
        bare (unlabelled) latex. Fixed: maxima_game_mixin._strip_h1_lhs strips
        the LHS before parsing."""
        nb.start_engine(title="test_f17_nash_h1_named", mode="standard", language="en",
                         domains=["game_theory"])
        eng = nb.get_engine()
        try:
            a = eng.register_expression("[2, 0; 0, 1]", latex="A = [2, 0; 0, 1]")
            b = eng.register_expression("[1, 0; 0, 2]", latex="B = [1, 0; 0, 2]")
            step, _ = eng.run_operation(
                "pure_nash_equilibria", {"payoff_a": a.display_id, "payoff_b": b.display_id})
            result = eng.expressions[step.output_refs[0]].canonical
            assert result.replace(" ", "") == "[[1,1],[2,2]]"
        finally:
            nb.stop_engine()

    def test_shape_mismatch_rejected(self):
        nb.start_engine(title="test_f17_shape", mode="standard", language="en",
                         domains=["game_theory"])
        eng = nb.get_engine()
        try:
            a = eng.register_expression("[1, 2; 3, 4]", latex="A")
            b = eng.register_expression("[1, 2, 3; 4, 5, 6]", latex="B")
            with pytest.raises(ValueError):
                eng.run_operation(
                    "pure_nash_equilibria", {"payoff_a": a.display_id, "payoff_b": b.display_id})
        finally:
            nb.stop_engine()


class TestMixedStrategyIndifferenceConditions:
    def test_mixed_equilibrium_by_general_operations(self):
        """The known mixed NE of Battle of the Sexes, via the route a researcher
        now takes: they write the indifference conditions themselves and solve.

        Replaces the value assertions of `mixed_strategy_indifference_conditions`
        (deleted 2026-08-14, SCENARIO-MACRO case 4) — the macro assembled these
        two equations from the matrix entries, which is game theory, not system
        behaviour.  Measured equal to the macro's own output before deleting it.
        """
        nb.start_engine(title="test_f17_mixed_general", mode="standard",
                        language="en", domains=["game_theory"])
        eng = nb.get_engine()
        try:
            # player 1 indifferent between rows, opponent mixing q on column 1
            eq_q = eng.register_expression("2*q + 0*(1-q) = 0*q + 1*(1-q)")
            # player 2 indifferent between columns, opponent mixing p on row 1
            eq_p = eng.register_expression("1*p + 0*(1-p) = 0*p + 2*(1-p)")
            sq, _ = eng.run_operation("solve_equation",
                                      {"equation": eq_q.display_id, "var": "q"})
            sp, _ = eng.run_operation("solve_equation",
                                      {"equation": eq_p.display_id, "var": "p"})
            assert eng.expressions[sq.output_refs[0]].canonical.strip() == "1/3"
            assert eng.expressions[sp.output_refs[0]].canonical.strip() == "2/3"
        finally:
            nb.stop_engine()

    def test_bp7_root_cause_operation_result_never_p2_binds(self):
        """BP-7 root-cause regression (notebook/engine.py run_operation): even
        if an OPERATION's result happens to collapse to a bare 'symbol = expr'
        equation form (indistinguishable from H-1 user input at the latex
        level), it must never trigger P2's CAS-side auto-substitute. Simulates
        the collapse directly (bypassing the mixed_strategy_indifference
        template's lhs-rhs=0 defensive workaround) via simplify(), then proves
        a later solve() referencing the same symbol is NOT corrupted."""
        nb.start_engine(title="test_f17_bp7_root", mode="standard", language="en",
                         domains=["game_theory"])
        eng = nb.get_engine()
        try:
            src = eng.register_expression("q=2*(1-q)", latex="q=2*(1-q)",
                                           skip_cas_bind=True)
            step, _ = eng.run_operation("simplify", {"expr": src.display_id})
            out = eng.expressions[step.output_refs[0]]
            assert out.latex.split("=", 1)[0].strip() == "q", (
                "test setup did not reproduce a bare-symbol-LHS operation "
                "result -- adjust the repro to still exercise the P2 guard")

            check = eng.register_expression("q=2*(1-q)", latex="q=2*(1-q)",
                                             skip_cas_bind=True)
            solved, _ = eng.run_operation(
                "solve_equation", {"equation": check.display_id, "var": "q"})
            assert eng.expressions[solved.output_refs[0]].canonical.strip() == "2/3"
        finally:
            nb.stop_engine()

    def test_pure_nash_is_reachable_from_natural_language(self):
        """It was NOT, until 2026-08-14 (SCENARIO-MACRO case 4): the positional
        field mapper did not count `payoff_a`/`payoff_b` as EXPRESSION fields,
        so neither could be filled and every phrasing failed — including typing
        the operation's own alias.  The same defect BP-6 recorded for convolve's
        `f`/`g`, never applied here; and `_EXPR_FIELDS` existed twice, with the
        BP-6 fix in only one copy.

        English uses the single-token alias because all three of this
        operation's English aliases are multi-word and cannot match a per-token
        scan (open_items EN-ALIAS-MULTIWORD).  When that is fixed this should
        become `find the pure Nash equilibria of E1 and E2`.
        """
        for lang, phrase in (("ja", "{a} と {b} の純戦略ナッシュ均衡を求めて"),
                             ("en", "pure_nash_equilibria of {a} and {b}")):
            nb.start_engine(title=f"test_f17_nl_{lang}", mode="standard",
                            language=lang, domains=["game_theory"])
            eng = nb.get_engine()
            sup = _sup()
            try:
                a = eng.register_expression("[2, 0; 0, 1]", latex="A")
                b = eng.register_expression("[1, 0; 0, 2]", latex="B")
                r = _run(eng, sup, phrase.format(a=a.display_id, b=b.display_id))
                assert r["status"] == "completed", (lang, r)
                assert r["operation"] == "pure_nash_equilibria", (lang, r)
                out = eng.expressions[r["expr_id"]].canonical.replace(" ", "")
                assert out == "[[1,1],[2,2]]", (lang, out)
            finally:
                nb.stop_engine()
