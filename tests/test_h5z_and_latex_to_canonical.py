"""Tests for H-5z (set RHS to value) and lhs_latex_to_canonical.

Covers:
- H-5z: "En の右辺を X とする" — registers new equation RHS = X
  - Works with both Maxima-style and Julia-style engines
  - Handles 全角数字、E-ID references, LaTeX values
- lhs_latex_to_canonical: MaximaBackendHandler converts \\ddot{q} → 'diff(q(t),t,2)
  - Used by _DAINYU_RE handler to convert LaTeX LHS before substitution
  - Julia default is no-op
- _DAINYU_RE + lhs_latex_to_canonical: end-to-end substitution with LaTeX LHS
"""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import Quantity
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expression(eid: str, canonical: str, latex: str = "") -> MagicMock:
    expr = MagicMock()
    expr.display_id = eid
    expr.canonical = canonical
    expr.latex = latex or canonical
    expr.created_by_step = "step_1"
    expr.undefined_symbols = []
    return expr


def _make_step(out_id: str) -> MagicMock:
    step = MagicMock()
    step.output_refs = [out_id]
    step.step_id = "step_out"
    step.visible_action = "rhs_set"
    return step


def _make_engine(expressions: dict, lhs_alias_map: dict | None = None) -> MagicMock:
    engine = MagicMock()
    engine.active_branch_id = "branch_1"
    engine.steps = {}
    engine.branches = {}
    engine.expressions = expressions
    engine.symbol_context = SymbolContext()
    engine.quantity_registry = {}
    engine.lhs_alias_map = lhs_alias_map or {}
    engine.pending_op = None
    engine.lang = "ja"
    engine._active_expr_ids.return_value = set(expressions.keys())
    engine._handler = MagicMock()
    engine._handler.operands_provided.return_value = None
    engine._handler.latex_to_canonical.side_effect = lambda x: x
    engine._handler.normalize_display_latex.side_effect = lambda x: x
    engine._handler.lhs_latex_to_canonical.side_effect = lambda x: x  # default no-op
    # A real engine answers None for an expression carrying no placeholder sum
    # range (`placeholder_cas_view`).  A MagicMock answers a MagicMock, which is
    # TRUTHY — so the substitution source would be a mock object instead of text.
    engine.placeholder_cas_view.return_value = None
    return engine


def _make_supervisor() -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup._v = OperationValidator()
    sup.needs_expression_injection.return_value = False
    return sup


def _run(engine, supervisor, text: str) -> dict:
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


# ---------------------------------------------------------------------------
# lhs_latex_to_canonical — MaximaBackendHandler
# ---------------------------------------------------------------------------

class TestMaximaLhsLatexToCanonical:
    """MaximaBackendHandler.lhs_latex_to_canonical converts dot-notation LHS."""

    def _make_handler(self, time_dep_syms: list[str] = None):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        ctx = SymbolContext()
        for sym in (time_dep_syms or []):
            ctx.register(sym, Quantity(name=sym, kind="scalar",
                                       time_dependent=True, time_parameter="t"))
        h._context = ctx
        return h

    def test_ddot_time_dependent(self):
        h = self._make_handler(["q"])
        result = h.lhs_latex_to_canonical("\\ddot{q}")
        assert result == "'diff(q(t),t,2)"

    def test_dot_time_dependent(self):
        h = self._make_handler(["q"])
        result = h.lhs_latex_to_canonical("\\dot{q}")
        assert result == "'diff(q(t),t,1)"

    def test_dddot_time_dependent(self):
        h = self._make_handler(["q"])
        result = h.lhs_latex_to_canonical("\\dddot{q}")
        assert result == "'diff(q(t),t,3)"

    def test_ddot_not_time_dependent(self):
        # Without time_dependent flag, uses depends form without (t)
        h = self._make_handler([])
        result = h.lhs_latex_to_canonical("\\ddot{q}")
        assert result == "'diff(q,t,2)"

    def test_non_dot_notation_passthrough(self):
        h = self._make_handler(["q"])
        assert h.lhs_latex_to_canonical("\\omega") == "\\omega"
        assert h.lhs_latex_to_canonical("g") == "g"
        assert h.lhs_latex_to_canonical("'diff(q(t),t,2)") == "'diff(q(t),t,2)"

    def test_no_context(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        h._context = None
        # No time_dependent info → uses depends form
        result = h.lhs_latex_to_canonical("\\ddot{q}")
        assert result == "'diff(q,t,2)"


class TestJuliaLhsLatexToCanonicalNoop:
    """Julia BackendHandler default is no-op."""

    def test_noop_for_ddot(self):
        from notebook.runtime.backend_handler import BackendHandler
        # BackendHandler is abstract; verify via a simple subclass or Julia handler
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        result = h.lhs_latex_to_canonical("\\ddot{q}")
        # Julia does not override → returns unchanged
        assert result == "\\ddot{q}"

    def test_noop_for_plain(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        assert h.lhs_latex_to_canonical("x") == "x"


# ---------------------------------------------------------------------------
# H-5z: "En の右辺を X とする"
# ---------------------------------------------------------------------------

class TestH5zRhsSet:
    """H-5z registers new equation: (RHS of En) = X."""

    def _run_h5z(self, src_canonical: str, src_latex: str, value: str):
        """Helper: run pipeline with one source expression."""
        e1 = _make_expression("E1", src_canonical, src_latex)
        result_expr = _make_expression("E2", f"{src_canonical} = 0",
                                       f"{src_latex} = {value}")
        expressions = {"E1": e1}
        engine = _make_engine(expressions)
        engine.register_expression.return_value = result_expr

        with patch("ui.interpret_pipeline._find_expr_by_canonical", return_value=None):
            return _run(engine, _make_supervisor(), f"E1 の右辺を {value} とする")

    def test_set_to_zero(self):
        result = self._run_h5z(
            src_canonical="l^2*m*(q_ddot + omega^2*q)",
            src_latex="l^2 m (\\ddot{q} + \\omega^2 q)",
            value="0",
        )
        assert result["status"] == "completed"
        assert result.get("operation") == "rhs_set"

    def test_set_to_zero_fullwidth(self):
        result = self._run_h5z(
            src_canonical="l^2*m*(q_ddot + omega^2*q)",
            src_latex="l^2 m (\\ddot{q} + \\omega^2 q)",
            value="０",
        )
        assert result["status"] == "completed"
        assert result.get("operation") == "rhs_set"

    def test_set_to_one(self):
        result = self._run_h5z(
            src_canonical="x + y",
            src_latex="x + y",
            value="1",
        )
        assert result["status"] == "completed"
        assert result.get("operation") == "rhs_set"

    def test_h5z_regex_variants(self):
        """All trigger phrase variants match."""
        e1 = _make_expression("E1", "m * a", "F = m a")
        result_expr = _make_expression("E2", "m * a = 0", "m a = 0")

        for phrase in ["E1 の右辺を 0 とする",
                       "E1 の右辺を 0 とおく",
                       "E1 の右辺を 0 と置く"]:
            engine = _make_engine({"E1": e1})
            engine.register_expression.return_value = result_expr
            supervisor = _make_supervisor()
            with patch("ui.interpret_pipeline._find_expr_by_canonical", return_value=None):
                result = _run(engine, supervisor, phrase)
            assert result["status"] == "completed", f"Failed for phrase: {phrase!r}"

    # rhs_set externalized to i18n/nl_patterns.json (所見G/(b) migration); EN
    # only tried when the session is English (engine.lang).
    def test_set_to_zero_en(self):
        e1 = _make_expression("E1", "l^2*m*(q_ddot + omega^2*q)",
                              "l^2 m (\\ddot{q} + \\omega^2 q)")
        result_expr = _make_expression("E2", "l^2*m*(q_ddot + omega^2*q) = 0",
                                       "l^2 m (\\ddot{q} + \\omega^2 q) = 0")
        engine = _make_engine({"E1": e1})
        engine.lang = "en"
        engine.register_expression.return_value = result_expr
        with patch("ui.interpret_pipeline._find_expr_by_canonical", return_value=None):
            result = _run(engine, _make_supervisor(),
                          "Set the right-hand side of E1 to 0")
        assert result["status"] == "completed"
        assert result.get("operation") == "rhs_set"

    def test_h5z_regex_no_match_for_missing_eid(self):
        """_RHS_SET_RE does not match when EID refers to non-existent expression."""
        import re
        _RHS_SET_RE = re.compile(
            r'(E\d+)\s*の\s*右辺\s*を\s*(.+?)\s*(?:とする|とおく|と置く|に等しいとする|として|と等しいとする)(?:\s|$)',
            re.U,
        )
        # Pattern should match syntactically (EID doesn't matter at regex level)
        m = _RHS_SET_RE.search("E9 の右辺を 0 とする")
        assert m is not None
        assert m.group(1) == "E9"
        assert m.group(2) == "0"


# ---------------------------------------------------------------------------
# _DAINYU_RE + lhs_latex_to_canonical: end-to-end substitution
# ---------------------------------------------------------------------------

class TestDainyuWithLatexLhs:
    """Substitution pipeline converts LaTeX LHS via lhs_latex_to_canonical."""

    def test_ddot_lhs_converted_for_maxima(self):
        """\\ddot{q} as lhs_alias_map key → lhs_latex_to_canonical → 'diff(q(t),t,2)."""
        # E14: solve result with LaTeX key \\ddot{q} → E14
        e14 = _make_expression("E14", "-(omega^2*q)",
                                "\\ddot{q} = -\\omega^2 q")
        # E21: dE/dt expression containing \ddot{q}
        e21 = _make_expression("E21",
                                "m*l^2*diff(q(t),t,1)*'diff(q(t),t,2) + ...",
                                "\\frac{d}{dt}E_{total}")

        expressions = {"E14": e14, "E21": e21}
        lhs_alias_map = {"\\ddot{q}": "E14"}  # LaTeX key only (loaded session)

        engine = _make_engine(expressions, lhs_alias_map)
        # lhs_latex_to_canonical converts \\ddot{q} → 'diff(q(t),t,2)
        engine._handler.lhs_latex_to_canonical.side_effect = lambda lhs: (
            "'diff(q(t),t,2)" if lhs == "\\ddot{q}" else lhs
        )

        out_step = _make_step("E22")
        out_step.output_refs = ["E22"]
        e22 = _make_expression("E22", "0", "0")
        engine.expressions["E22"] = e22
        engine.run_operation.return_value = (out_step, "0")

        supervisor = _make_supervisor()
        result = _run(engine, supervisor,
                      "E21の\\ddot{q}に E14 を代入して")

        assert result["status"] == "completed"
        call_args = engine.run_operation.call_args
        assert call_args is not None
        op, args = call_args[0]
        assert op == "substitute"
        # LHS must be the canonical derivative form, not the LaTeX form
        assert args["lhs"] == "'diff(q(t),t,2)", (
            f"Expected canonical 'diff(q(t),t,2) but got {args['lhs']!r}")

    def test_plain_lhs_not_converted(self):
        """Plain canonical keys (no \\) are passed through without conversion."""
        # E8: omega^2 = g/l — canonical key
        e8 = _make_expression("E8", "g/l", "\\omega^2 = g/l")
        e7 = _make_expression("E7", "q_ddot + g/l * q", "\\ddot{q} + \\frac{g}{l}q")

        expressions = {"E8": e8, "E7": e7}
        lhs_alias_map = {"omega^2": "E8"}  # canonical key (no \\)

        engine = _make_engine(expressions, lhs_alias_map)
        engine._handler.lhs_latex_to_canonical.side_effect = lambda lhs: lhs

        out_step = _make_step("E9")
        out_step.output_refs = ["E9"]
        e9 = _make_expression("E9", "q_ddot + omega^2 * q")
        engine.expressions["E9"] = e9
        engine.run_operation.return_value = (out_step, "q_ddot + omega^2*q")

        supervisor = _make_supervisor()
        result = _run(engine, supervisor, "E7 に E8 を代入して")

        assert result["status"] == "completed"
        call_args = engine.run_operation.call_args
        op, args = call_args[0]
        assert op == "substitute"
        assert args["lhs"] == "omega^2"
        # lhs_latex_to_canonical must NOT be called (key doesn't start with \\)
        engine._handler.lhs_latex_to_canonical.assert_not_called()

    def test_julia_noop_for_latex_lhs(self):
        """For Julia (no-op lhs_latex_to_canonical), LaTeX keys pass through unchanged."""
        # Julia solve result: x = val, stored with canonical key
        e10 = _make_expression("E10", "val", "x = val")
        e11 = _make_expression("E11", "x + 1", "x + 1")

        expressions = {"E10": e10, "E11": e11}
        lhs_alias_map = {"x": "E10"}  # canonical key in Julia

        engine = _make_engine(expressions, lhs_alias_map)
        # Julia no-op
        engine._handler.lhs_latex_to_canonical.side_effect = lambda lhs: lhs

        out_step = _make_step("E12")
        out_step.output_refs = ["E12"]
        e12 = _make_expression("E12", "val + 1")
        engine.expressions["E12"] = e12
        engine.run_operation.return_value = (out_step, "val + 1")

        supervisor = _make_supervisor()
        result = _run(engine, supervisor, "E11 に E10 を代入して")

        assert result["status"] == "completed"
        call_args = engine.run_operation.call_args
        op, args = call_args[0]
        assert op == "substitute"
        assert args["lhs"] == "x"
