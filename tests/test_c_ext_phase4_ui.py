"""Tests for C-Ext Phase 4: backend-agnostic UI integration.

Verifies that re_evaluate_expr works identically on Maxima and Julia,
and that auto-detection of substitutions from lhs_alias_map works correctly.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


# ---------------------------------------------------------------------------
# Auto-detect substitutions from lhs_alias_map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kw", [
    (SessionMode.standard_symbolic, {"coordinate_system": "cartesian"}),
    pytest.param(SessionMode.second_quantization, {},
                 marks=pytest.mark.skipif(
                     __import__("shutil").which("julia") is None,
                     reason="julia not available")),
])
def test_re_evaluate_auto_detect_both_backends(mode, kw):
    """Both backends produce a result when substitutions are auto-detected."""
    with NotebookEngine(title="test", mode=mode, **kw) as e:
        e.start()
        e.register_expression("H_0 + H_1", lhs_label="H")
        e.register_expression("p^2", lhs_label="H_0")
        e.register_expression("x^2", lhs_label="H_1")

        H_expr_id = e.lhs_alias_map["H"]
        result = e.re_evaluate_expr(H_expr_id)

        assert result is not None
        assert "p" in result.canonical or "x" in result.canonical


@pytest.mark.parametrize("mode,kw", [
    (SessionMode.standard_symbolic, {"coordinate_system": "cartesian"}),
    pytest.param(SessionMode.second_quantization, {},
                 marks=pytest.mark.skipif(
                     __import__("shutil").which("julia") is None,
                     reason="julia not available")),
])
def test_re_evaluate_explicit_substitutions_both_backends(mode, kw):
    """Explicit substitutions work on both backends."""
    with NotebookEngine(title="test", mode=mode, **kw) as e:
        e.start()
        expr = e.register_expression("a + b")
        result = e.re_evaluate_expr(expr.display_id,
                                    substitutions={"a": "1", "b": "2"})
        assert result is not None
        assert "3" in result.canonical


def test_re_evaluate_none_substitutions_returns_result():
    """Passing substitutions=None triggers auto-detect (same as empty dict)."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        e.register_expression("A + B", lhs_label="S")
        e.register_expression("1", lhs_label="A")
        e.register_expression("2", lhs_label="B")

        S_id = e.lhs_alias_map["S"]
        result = e.re_evaluate_expr(S_id, substitutions=None)
        assert result is not None
        assert "3" in result.canonical


# ---------------------------------------------------------------------------
# _auto_detect_substitutions: word-boundary matching
# ---------------------------------------------------------------------------

def test_auto_detect_no_false_positive_prefix():
    """'H' in lhs_alias_map should NOT match inside 'H_0' in canonical."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        # Register short name "H" → value "42"
        e.register_expression("42", lhs_label="H")
        # Register expression that contains "H_0" (not just "H")
        expr = e.register_expression("H_0 + H_1")

        subs = e._auto_detect_substitutions(expr)
        # "H" should NOT match because expr contains "H_0" and "H_1", not bare "H"
        assert "H" not in subs


def test_auto_detect_exact_symbol():
    """Exact symbol 'H' in canonical IS detected."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        e.register_expression("p^2", lhs_label="H")
        expr = e.register_expression("H + V")

        subs = e._auto_detect_substitutions(expr)
        assert "H" in subs
        assert "p" in subs["H"]


def test_auto_detect_multiple_symbols():
    """Multiple lhs_alias_map entries all detected."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        e.register_expression("p^2", lhs_label="H_0")
        e.register_expression("x^2", lhs_label="H_1")
        expr = e.register_expression("H_0 + H_1")

        subs = e._auto_detect_substitutions(expr)
        assert "H_0" in subs
        assert "H_1" in subs


# ---------------------------------------------------------------------------
# Schema routing: re_evaluate_expr is now a proper operation
# ---------------------------------------------------------------------------

def test_re_evaluate_schema_registered():
    """re_evaluate_expr appears in the schema and is validated."""
    from notebook.runtime.operation_validator import OperationValidator
    v = OperationValidator()
    result = v.validate("re_evaluate_expr", {"expr": "E1"}, primary_backend="maxima")
    # ValidationResult.error is None when valid
    assert result.error is None, f"Validation failed: {result.error}"


def test_re_evaluate_schema_missing_expr():
    """re_evaluate_expr without expr fails validation."""
    from notebook.runtime.operation_validator import OperationValidator
    v = OperationValidator()
    result = v.validate("re_evaluate_expr", {}, primary_backend="maxima")
    assert result.error is not None


# ---------------------------------------------------------------------------
# extract_order / truncate_series schema routing (already registered)
# ---------------------------------------------------------------------------

def test_extract_order_via_run_operation_maxima():
    """extract_order routes through run_operation on Maxima."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        expr = e.register_expression("1 + 2*x + 3*x^2")
        step, v = e.run_operation("extract_order",
                                  {"expr": expr.display_id, "var": "x", "order": 2})
        assert "3" in str(v)


def test_truncate_series_via_run_operation_maxima():
    """truncate_series routes through run_operation on Maxima."""
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        expr = e.register_expression("1 + x + x^2 + x^3")
        step, v = e.run_operation("truncate_series",
                                  {"expr": expr.display_id, "var": "x", "order": 2})
        assert "x^3" not in str(v)
        assert "x^2" in str(v)
