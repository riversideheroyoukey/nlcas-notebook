"""Tests for compute_lineage() and replay().

compute_lineage tests are pure (no Maxima). replay tests start a real Maxima
process to verify that saved rendered_commands reproduce the correct results.
"""
import pytest
from datetime import datetime, timezone

from notebook.models.branch import Branch
from notebook.models.expression import Expression
from notebook.models.step import Step, StepKind, StepStatus
from notebook.runtime.maxima_adapter import MaximaAdapter
from notebook.runtime.replay import compute_lineage, replay, ReplayResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc)


def make_step(step_id, branch_id, visible_action, kind=StepKind.cas_operation,
              parent_ids=None, rendered_command=None, output_refs=None,
              state_affecting=True, status=StepStatus.success):
    return Step(
        step_id=step_id,
        kind=kind,
        branch_id=branch_id,
        parent_step_ids=parent_ids or [],
        timestamp=now(),
        visible_action=visible_action,
        rendered_command=rendered_command,
        output_refs=output_refs or [],
        state_affecting=state_affecting,
        status=status,
    )


def make_branch(branch_id, head_step_id=None, forked_from=None, label=""):
    return Branch(
        branch_id=branch_id,
        head_step_id=head_step_id,
        forked_from_step_id=forked_from,
        label=label,
    )


# ---------------------------------------------------------------------------
# compute_lineage — pure tests, no Maxima
# ---------------------------------------------------------------------------

class TestComputeLineage:
    def test_empty_branch(self):
        branch = make_branch("main")
        assert compute_lineage("main", {}, {"main": branch}) == []

    def test_unknown_branch(self):
        assert compute_lineage("ghost", {}, {}) == []

    def test_single_step(self):
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration)
        branch = make_branch("main", head_step_id="S1")
        lineage = compute_lineage("main", {"S1": s1}, {"main": branch})
        assert [s.step_id for s in lineage] == ["S1"]

    def test_linear_chain_is_root_first(self):
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration)
        s2 = make_step("S2", "main", "expand E1", parent_ids=["S1"],
                       rendered_command="expand(x^2-1)", output_refs=["E2"])
        s3 = make_step("S3", "main", "diff E2", parent_ids=["S2"],
                       rendered_command="diff(x+1,x)", output_refs=["E3"])
        branch = make_branch("main", head_step_id="S3")
        steps = {"S1": s1, "S2": s2, "S3": s3}
        lineage = compute_lineage("main", steps, {"main": branch})
        assert [s.step_id for s in lineage] == ["S1", "S2", "S3"]

    def test_missing_intermediate_step_stops_walk(self):
        s3 = make_step("S3", "main", "diff", parent_ids=["S2"],
                       rendered_command="diff(x,x)", output_refs=["E3"])
        branch = make_branch("main", head_step_id="S3")
        lineage = compute_lineage("main", {"S3": s3}, {"main": branch})
        # S2 missing → walk stops; only S3 is returned
        assert [s.step_id for s in lineage] == ["S3"]

    def test_cross_branch_lineage(self):
        """Forked branch inherits ancestor steps from main."""
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration)
        s2 = make_step("S2", "main", "expand E1", parent_ids=["S1"],
                       rendered_command="expand(x^2-1)", output_refs=["E2"])
        # branch_002 forked after S2; its first step's parent is S2
        sb1 = make_step("SB1", "branch_002", "factor E2", parent_ids=["S2"],
                        rendered_command="factor(x^2-1)", output_refs=["E3"])
        main = make_branch("main", head_step_id="S2")
        fork = make_branch("branch_002", head_step_id="SB1", forked_from="S2")
        steps = {"S1": s1, "S2": s2, "SB1": sb1}
        branches = {"main": main, "branch_002": fork}
        lineage = compute_lineage("branch_002", steps, branches)
        assert [s.step_id for s in lineage] == ["S1", "S2", "SB1"]

    def test_cycle_guard(self):
        s1 = make_step("S1", "main", "x", parent_ids=["S2"])
        s2 = make_step("S2", "main", "y", parent_ids=["S1"])
        branch = make_branch("main", head_step_id="S1")
        # Must not hang
        lineage = compute_lineage("main", {"S1": s1, "S2": s2}, {"main": branch})
        assert len(lineage) == 2


# ---------------------------------------------------------------------------
# replay — integration tests with real Maxima
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def maxima():
    with MaximaAdapter() as m:
        yield m


class TestReplayLinear:
    """Replay a simple linear history: register E1, simplify → E2, diff → E3."""

    def _setup(self):
        """Build steps and expressions for the canonical E1→E2→E3 chain."""
        e1 = Expression(display_id="E1", branch_id="main",
                        latex=r"\frac{x^2-1}{x-1}", canonical="(x^2-1)/(x-1)")
        e2 = Expression(display_id="E2", branch_id="main",
                        latex="x+1", canonical="x+1",
                        created_by_step="S2")
        e3 = Expression(display_id="E3", branch_id="main",
                        latex="1", canonical="1",
                        created_by_step="S3")

        s1 = make_step("S1", "main", "E1 を登録する",
                       kind=StepKind.expression_registration,
                       output_refs=["E1"])
        s2 = make_step("S2", "main", "E1 を有理式として簡約する",
                       parent_ids=["S1"],
                       rendered_command="ratsimp((x^2-1)/(x-1))",
                       output_refs=["E2"])
        s3 = make_step("S3", "main", "E2 を x で微分する",
                       parent_ids=["S2"],
                       rendered_command="diff(x+1,x)",
                       output_refs=["E3"])

        branch = make_branch("main", head_step_id="S3")
        steps = {"S1": s1, "S2": s2, "S3": s3}
        branches = {"main": branch}
        exprs = {"E1": e1, "E2": e2, "E3": e3}
        return steps, branches, exprs

    def test_replay_rebuilds_e2_and_e3(self, maxima):
        steps, branches, exprs = self._setup()
        result = replay("main", steps, branches, exprs, maxima)
        assert result.ok
        assert "x+1" in result.expr_values["E2"]
        assert result.expr_values["E3"] == "1"

    def test_replay_counts_steps(self, maxima):
        steps, branches, exprs = self._setup()
        result = replay("main", steps, branches, exprs, maxima)
        # S1 (expression_registration) + S2 + S3 = 3 replayed, 0 skipped
        assert result.replayed_steps == 3
        assert result.skipped_steps == 0

    def test_replay_result_type(self, maxima):
        steps, branches, exprs = self._setup()
        result = replay("main", steps, branches, exprs, maxima)
        assert isinstance(result, ReplayResult)

    def test_e1_canonical_restored_from_expression_table(self, maxima):
        steps, branches, exprs = self._setup()
        result = replay("main", steps, branches, exprs, maxima)
        assert result.expr_values["E1"] == "(x^2-1)/(x-1)"


class TestReplaySkipsNonStateAffecting:
    def test_advisory_step_is_skipped(self, maxima):
        e1 = Expression(display_id="E1", branch_id="main",
                        latex="x^2", canonical="x^2")
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration, output_refs=["E1"])
        s2 = make_step("S2", "main", "LLM consideration",
                       kind=StepKind.llm_advisory,
                       parent_ids=["S1"],
                       state_affecting=False)
        s3 = make_step("S3", "main", "diff E1",
                       parent_ids=["S2"],
                       rendered_command="diff(x^2,x)",
                       output_refs=["E2"])
        branch = make_branch("main", head_step_id="S3")
        steps = {"S1": s1, "S2": s2, "S3": s3}
        result = replay("main", steps, {"main": branch}, {"E1": e1}, maxima)
        assert result.ok
        assert result.skipped_steps == 1   # S2 skipped
        assert result.replayed_steps == 2  # S1, S3
        assert "2*x" in result.expr_values["E2"]

    def test_step_without_rendered_command_is_skipped(self, maxima):
        e1 = Expression(display_id="E1", branch_id="main",
                        latex="x", canonical="x")
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration, output_refs=["E1"])
        s2 = make_step("S2", "main", "op with no command",
                       parent_ids=["S1"],
                       rendered_command=None,  # e.g. unresolved step
                       output_refs=["E2"])
        branch = make_branch("main", head_step_id="S2")
        result = replay("main", {"S1": s1, "S2": s2},
                        {"main": branch}, {"E1": e1}, maxima)
        assert result.skipped_steps == 1


class TestReplayErrorHandling:
    def test_bad_command_recorded_as_error(self, maxima):
        e1 = Expression(display_id="E1", branch_id="main",
                        latex="x", canonical="x")
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration, output_refs=["E1"])
        s2 = make_step("S2", "main", "bad op",
                       parent_ids=["S1"],
                       rendered_command='error("simulated replay failure")',
                       output_refs=["E2"])
        s3 = make_step("S3", "main", "diff E1",
                       parent_ids=["S2"],
                       rendered_command="diff(x,x)",
                       output_refs=["E3"])
        branch = make_branch("main", head_step_id="S3")
        steps = {"S1": s1, "S2": s2, "S3": s3}
        result = replay("main", steps, {"main": branch}, {"E1": e1}, maxima)
        assert not result.ok
        assert len(result.errors) == 1
        assert "S2" in result.errors[0]

    def test_replay_continues_after_error(self, maxima):
        """Errors do not halt replay — subsequent steps still run."""
        e1 = Expression(display_id="E1", branch_id="main",
                        latex="x^3", canonical="x^3")
        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration, output_refs=["E1"])
        s2 = make_step("S2", "main", "bad op",
                       parent_ids=["S1"],
                       rendered_command='error("oops")',
                       output_refs=["E2"])
        s3 = make_step("S3", "main", "diff E1",
                       parent_ids=["S2"],
                       rendered_command="diff(x^3,x)",
                       output_refs=["E3"])
        branch = make_branch("main", head_step_id="S3")
        steps = {"S1": s1, "S2": s2, "S3": s3}
        result = replay("main", steps, {"main": branch}, {"E1": e1}, maxima)
        # E3 should still be set despite S2 failing
        assert "E3" in result.expr_values
        assert "3*x^2" in result.expr_values["E3"]


class TestReplayEmptyBranch:
    def test_empty_branch_returns_empty_result(self, maxima):
        branch = make_branch("main")
        result = replay("main", {}, {"main": branch}, {}, maxima)
        assert result.ok
        assert result.expr_values == {}
        assert result.replayed_steps == 0


class TestReplayCrossBranch:
    """Replay a forked branch that inherits steps from main."""

    def test_fork_inherits_main_steps(self, maxima):
        e1 = Expression(display_id="E1", branch_id="main",
                        latex="x^2-1", canonical="x^2-1")
        e2 = Expression(display_id="E2", branch_id="main",
                        latex="(x-1)(x+1)", canonical="(x-1)*(x+1)",
                        created_by_step="S2")

        s1 = make_step("S1", "main", "register E1",
                       kind=StepKind.expression_registration, output_refs=["E1"])
        s2 = make_step("S2", "main", "factor E1",
                       parent_ids=["S1"],
                       rendered_command="factor(x^2-1)",
                       output_refs=["E2"])
        # branch_002 forks after S2 and tries expand instead
        sb1 = make_step("SB1", "branch_002", "expand E1",
                        parent_ids=["S2"],
                        rendered_command="expand(x^2-1)",
                        output_refs=["E3"])

        main = make_branch("main", head_step_id="S2")
        fork = make_branch("branch_002", head_step_id="SB1", forked_from="S2")
        steps = {"S1": s1, "S2": s2, "SB1": sb1}
        branches = {"main": main, "branch_002": fork}
        exprs = {"E1": e1, "E2": e2}

        result = replay("branch_002", steps, branches, exprs, maxima)
        assert result.ok
        # E2 from main's S2 step
        assert "(x" in result.expr_values.get("E2", "")
        # E3 from fork's SB1 step
        assert "x^2" in result.expr_values.get("E3", "")
