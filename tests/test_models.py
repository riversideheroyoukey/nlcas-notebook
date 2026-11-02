import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from notebook.models import (
    Expression, Assumption,
    Branch,
    Step, StepKind, StepStatus,
    Session, SessionMode, FallbackPolicy, LLMSupervisionConfig, LLMSupervisionMode,
)


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_minimal(self):
        e = Expression(display_id="E1", branch_id="main", latex="x+1", canonical="x + 1")
        assert e.display_id == "E1"
        assert e.expr_uid.startswith("expr_")
        assert e.created_by_step is None

    def test_uid_is_unique(self):
        a = Expression(display_id="E1", branch_id="main", latex="x", canonical="x")
        b = Expression(display_id="E2", branch_id="main", latex="x", canonical="x")
        assert a.expr_uid != b.expr_uid

    def test_explicit_uid(self):
        e = Expression(expr_uid="expr_abc123", display_id="E3", branch_id="main", latex="y", canonical="y")
        assert e.expr_uid == "expr_abc123"

    def test_with_step(self):
        e = Expression(display_id="E2", branch_id="main", latex="x^2", canonical="x**2", created_by_step="S1")
        assert e.created_by_step == "S1"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Expression(branch_id="main", latex="x", canonical="x")  # display_id missing

    def test_julia_repr_default_none(self):
        e = Expression(display_id="E1", branch_id="main", latex="x", canonical="_nb_E1")
        assert e.cas_repr is None

    def test_julia_repr_stored(self):
        e = Expression(display_id="E1", branch_id="main", latex=r"a^\dagger(1) a(1)",
                       canonical="_nb_E1", cas_repr="a†(1)a(1)")
        assert e.cas_repr == "a†(1)a(1)"

    def test_julia_repr_serialized(self):
        e = Expression(display_id="E1", branch_id="main", latex=r"a^\dagger(1) a(1)",
                       canonical="_nb_E1", cas_repr="a†(1)a(1)")
        d = e.model_dump()
        assert "cas_repr" in d
        assert d["cas_repr"] == "a†(1)a(1)"

    def test_julia_repr_round_trips(self):
        e = Expression(display_id="E1", branch_id="main", latex="x",
                       canonical="_nb_E1", cas_repr="a(1) + a'(1)")
        e2 = Expression(**e.model_dump())
        assert e2.cas_repr == e.cas_repr

    def test_ir_ast_excluded_from_serialization(self):
        from notebook.models.ir import Symbol
        e = Expression(display_id="E1", branch_id="main", latex="x", canonical="x",
                       ir_ast=Symbol(name="x"))
        assert "ir_ast" not in e.model_dump()


# ---------------------------------------------------------------------------
# Assumption
# ---------------------------------------------------------------------------

class TestAssumption:
    def test_session_scope(self):
        a = Assumption(assumption_id="A1", display="m > 0", canonical="m > 0", scope="session")
        assert a.branch_id is None

    def test_branch_scope_requires_branch_id(self):
        with pytest.raises(ValidationError):
            Assumption(assumption_id="A2", display="x > 0", canonical="x > 0", scope="branch")

    def test_branch_scope_with_branch_id(self):
        a = Assumption(
            assumption_id="A2", display="x > 0", canonical="x > 0",
            scope="branch", branch_id="branch_002"
        )
        assert a.branch_id == "branch_002"

    def test_boson_operator_assumption(self):
        a = Assumption(
            assumption_id="A3",
            display="a_k はボソン消滅演算子",
            canonical="boson_annihilation(a_k)",
            scope="session"
        )
        assert a.scope == "session"


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

class TestBranch:
    def test_main_branch(self):
        b = Branch(branch_id="main", label="main derivation")
        assert b.forked_from_step_id is None
        assert b.primary_backend == "inherit"
        assert isinstance(b.created_at, datetime)

    def test_forked_branch(self):
        b = Branch(
            branch_id="branch_002",
            label="low-energy limit",
            forked_from_step_id="S12",
            primary_backend="inherit"
        )
        assert b.forked_from_step_id == "S12"

    def test_backend_accepts_any_string(self):
        # primary_backend is an open string; validation happens at make_backend() time.
        b = Branch(branch_id="b1", primary_backend="future_backend")
        assert b.primary_backend == "future_backend"

    def test_head_step_update(self):
        b = Branch(branch_id="main", head_step_id="S20")
        assert b.head_step_id == "S20"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def _expand_step(self, **kwargs):
        defaults = dict(
            step_id="S1",
            kind=StepKind.cas_operation,
            branch_id="main",
            visible_action="E1 を展開する",
            operation="expand",
            operation_args={"expr_ref": "E1"},
            backend="maxima",
            rendered_command="expand(E1)",
            output_refs=["E2"],
            state_affecting=True,
            status=StepStatus.success,
        )
        defaults.update(kwargs)
        return Step(**defaults)

    def test_minimal_cas_step(self):
        s = self._expand_step()
        assert s.step_id == "S1"
        assert s.kind == StepKind.cas_operation
        assert s.show_command_to_user is False
        assert s.status == StepStatus.success

    def test_defaults(self):
        s = Step(step_id="S0", kind=StepKind.expression_registration,
                 branch_id="main", visible_action="E1 を登録する")
        assert s.parent_step_ids == []
        assert s.operation_args == {}
        assert s.output_refs == []
        assert s.status == StepStatus.pending
        assert s.state_affecting is True

    def test_rendered_command_hidden_by_default(self):
        s = self._expand_step(rendered_command="expand(E1)")
        assert s.show_command_to_user is False

    def test_llm_advisory_step(self):
        s = Step(
            step_id="A19",
            kind=StepKind.llm_advisory,
            branch_id="main",
            parent_step_ids=["S18"],
            visible_action="LLM考察",
            state_affecting=False,
            status=StepStatus.success,
        )
        assert s.state_affecting is False

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Step(step_id="S1", kind=StepKind.cas_operation, branch_id="main")  # visible_action missing

    def test_invalid_kind(self):
        with pytest.raises(ValidationError):
            Step(step_id="S1", kind="unknown_kind", branch_id="main", visible_action="x")

    def test_timestamp_is_utc(self):
        s = self._expand_step()
        assert s.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class TestSession:
    def _standard_session(self, **kwargs):
        defaults = dict(title="Hamiltonian derivation", mode=SessionMode.standard_symbolic,
                        primary_backend="maxima")
        defaults.update(kwargs)
        return Session(**defaults)

    def test_standard_symbolic_session(self):
        s = self._standard_session()
        assert s.mode == SessionMode.standard_symbolic
        assert s.primary_backend == "maxima"
        assert s.show_cas_syntax_to_user is False
        assert s.fallback_policy == FallbackPolicy.ask_before_cross_backend
        assert s.active_branch_id == "main"
        assert s.session_id.startswith("session_")

    def test_second_quantization_session(self):
        s = Session(title="Hubbard model", mode=SessionMode.second_quantization,
                    primary_backend="julia_full")
        assert s.primary_backend == "julia_full"

    def test_session_id_is_unique(self):
        a = self._standard_session()
        b = self._standard_session()
        assert a.session_id != b.session_id

    def test_default_language(self):
        s = self._standard_session()
        assert "ja" in s.language

    def test_llm_supervision_defaults(self):
        s = self._standard_session()
        assert s.llm_supervision.auto_execute_suggestions is False
        assert s.llm_supervision.mode == LLMSupervisionMode.on_demand

    def test_custom_llm_supervision(self):
        cfg = LLMSupervisionConfig(mode=LLMSupervisionMode.full, model_location="api")
        s = self._standard_session(llm_supervision=cfg)
        assert s.llm_supervision.mode == LLMSupervisionMode.full
        assert s.llm_supervision.model_location == "api"

    def test_backend_accepts_any_string(self):
        # primary_backend is an open string; validation happens at make_backend() time.
        s = Session(title="x", mode=SessionMode.standard_symbolic, primary_backend="maple")
        assert s.primary_backend == "maple"

    def test_missing_title(self):
        with pytest.raises(ValidationError):
            Session(mode=SessionMode.standard_symbolic, primary_backend="maxima")

    def test_created_at_is_utc(self):
        s = self._standard_session()
        assert s.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Cross-model
# ---------------------------------------------------------------------------

class TestCrossModel:
    def test_session_branch_step_expression_chain(self):
        session = Session(title="test", mode=SessionMode.standard_symbolic,
                          primary_backend="maxima")
        branch = Branch(branch_id="main", label="main derivation")
        expr = Expression(display_id="E1", branch_id="main",
                          latex=r"\frac{x^2-1}{x-1}", canonical="(x^2-1)/(x-1)")
        step = Step(
            step_id="S1",
            kind=StepKind.cas_operation,
            branch_id="main",
            visible_action="E1 を有理式として簡約する",
            operation="rational_simplify",
            operation_args={"expr_ref": expr.display_id},
            backend=session.primary_backend,
            output_refs=["E2"],
            status=StepStatus.success,
        )
        assert step.backend == "maxima"
        assert step.show_command_to_user is False
        assert "E2" in step.output_refs

    def test_fork_chain(self):
        main = Branch(branch_id="main")
        fork = Branch(branch_id="branch_002", forked_from_step_id="S12",
                      label="alternative path")
        assert fork.forked_from_step_id == main.branch_id or fork.forked_from_step_id == "S12"
        assert main.branch_id != fork.branch_id
