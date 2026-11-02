"""BP-1: notation / display metadata operations must work on Julia sessions too.

`set_latex_name` and `set_notation` are CAS-free metadata operations handled
entirely in the UI layer (interpret_pipeline / notebook_bridge) via
engine.set_latex_name / engine.set_notation, which only touch SymbolContext.
They were previously restricted to `supported_backends: ["maxima"]`, which made
the OperationValidator reject them in Julia sessions (a Julia user could not
rename a display label).  BP-1 adds `julia_full` to both.

See docs/open_items.md BP-1.
"""
import pytest

from notebook.runtime.operation_validator import OperationValidator
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


@pytest.fixture
def v():
    return OperationValidator()


class TestValidatorAcceptsBothBackends:
    def test_set_latex_name_julia_and_maxima(self, v):
        for backend in ("maxima", "julia_full"):
            r = v.validate("set_latex_name",
                           {"name": "H", "latex": r"\hat{H}"}, backend, set())
            assert r.valid, f"{backend}: {r.error}"

    def test_set_notation_julia_and_maxima(self, v):
        for backend in ("maxima", "julia_full"):
            r = v.validate("set_notation",
                           {"name": "q", "format": "dot"}, backend, set())
            assert r.valid, f"{backend}: {r.error}"

    def test_schema_lists_julia_full(self, v):
        for op in ("set_latex_name", "set_notation"):
            spec = v.get_spec(op)
            assert "julia_full" in spec["supported_backends"], op
            assert "maxima" in spec["supported_backends"], op


class TestJuliaEngineMetadata:
    """End-to-end through a live Julia session: the metadata ops must succeed
    and mutate the SymbolContext without invoking the CAS."""

    def test_set_latex_name_renames_in_julia_session(self):
        with NotebookEngine(title="bp1", mode=SessionMode.second_quantization) as e:
            e.register_quantity("H")
            qty = e.set_latex_name("H", r"\hat{H}")
            assert qty.latex_name == r"\hat{H}"
            assert e.symbol_context.get_quantity("H").latex_name == r"\hat{H}"

    def test_set_notation_switches_in_julia_session(self):
        with NotebookEngine(title="bp1", mode=SessionMode.second_quantization) as e:
            e.declare_time_dependent("q")
            qty = e.set_notation("q", "dot")
            assert qty.notation == "dot"
            # Julia canonicals are variable names, so re-render is a no-op (guarded
            # by supports_equation_source) rather than an error.
            assert e.rerender_notation_for_quantity("q") == []
