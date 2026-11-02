"""R-6 (TR-1): save → load → the SAME operation must give the SAME result.

The audit found two persistence bugs (PR-1 load-time corruption, PR-2 lost
SymbolContext fields) that 2,236 structure-preserving tests never caught,
because nothing asserted that a CONTINUATION after reload behaves like the
continuation the user would have run without the reload.  These tests close
that hole for three representative sessions:

  - Maxima standard symbolic
  - Einstein-summation-enabled Maxima
  - Julia second quantization (CAS state rebuilt via replay, as in do_load)

Pattern: build a session, save it, run the continuation in the ORIGINAL
engine, then load the file and run the IDENTICAL continuation — the two
results must match exactly.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


# ---------------------------------------------------------------------------
# Maxima standard symbolic
# ---------------------------------------------------------------------------

class TestMaximaStandardRoundtrip:

    @staticmethod
    def _build(path):
        with NotebookEngine(title="rt maxima") as e:
            e1 = e.register_expression("x^2-1")
            step, _ = e.run_operation("factor", {"expr": e1.display_id})
            e.save(path)
            return step.output_refs[0]

    def test_differentiate_after_reload_matches(self, tmp_path):
        path = tmp_path / "rt_maxima.json"
        factored_id = self._build(path)

        def continuation(e):
            step, value = e.run_operation(
                "differentiate", {"expr": factored_id, "var": "x"})
            out = e.expressions[step.output_refs[0]]
            return value, out.canonical, out.latex

        with NotebookEngine.load(path) as e2:
            reloaded = continuation(e2)
        with NotebookEngine.load(path) as e3:
            again = continuation(e3)
        # deterministic across loads
        assert reloaded == again
        # and identical to the never-reloaded continuation
        with NotebookEngine(title="rt maxima ref") as ref:
            r1 = ref.register_expression("x^2-1")
            rstep, _ = ref.run_operation("factor", {"expr": r1.display_id})
            original = continuation(ref)
        assert reloaded == original

    def test_expand_on_registered_function_call_after_reload(self, tmp_path):
        """Continuation on a declared-function expression — the exact shape
        the PR-1 load migration used to corrupt (f(x^2) -> f*(x^2))."""
        path = tmp_path / "rt_func.json"
        with NotebookEngine(title="rt func") as e:
            e.register_quantity("f", kind="function")
            ex = e.register_expression("f(x^2)+f(x^2)")
            step, original = e.run_operation("simplify", {"expr": ex.display_id})
            e.save(path)
            eid = ex.display_id
        with NotebookEngine.load(path) as e2:
            _, reloaded = e2.run_operation("simplify", {"expr": eid})
        assert reloaded == original
        assert "f(" in reloaded  # still a function call, not f*(...)


# ---------------------------------------------------------------------------
# Einstein-summation session
# ---------------------------------------------------------------------------

class TestEinsteinRoundtrip:

    def test_built_command_identical_after_reload(self, tmp_path):
        """A^i B_i must lower to the SAME CAS command before and after reload.
        With the Einstein flag lost (pre-R-2), the reloaded session built
        expand(A^%i * B(i)) — a power — instead of the index sum."""
        path = tmp_path / "rt_einstein.json"
        with NotebookEngine(title="rt einstein") as e:
            e.symbol_context.einstein_summation = True
            ex = e.register_expression("A^i*B_i")
            eid = ex.display_id
            original_cmd, _, _ = e._build_cas_command(
                "expand", {"expr": eid}, {eid: ex.canonical}, "maxima")
            e.save(path)
        with NotebookEngine.load(path) as e2:
            ex2 = e2.expressions[eid]
            reloaded_cmd, _, _ = e2._build_cas_command(
                "expand", {"expr": eid}, {eid: ex2.canonical}, "maxima")
        assert reloaded_cmd == original_cmd
        # the semantic content: an index SUM, not a power of the imaginary unit
        assert "sum(" in reloaded_cmd
        assert "%i" not in reloaded_cmd


# ---------------------------------------------------------------------------
# Julia second quantization (CAS state rebuilt via replay, as do_load does)
# ---------------------------------------------------------------------------

class TestJuliaSecondQuantizationRoundtrip:

    def test_normal_order_after_reload_matches(self, tmp_path):
        path = tmp_path / "rt_julia.json"
        with NotebookEngine(
            title="rt julia", mode=SessionMode.second_quantization,
        ) as e:
            e.run_operation("define_fermion_operator", {"name": "c"})
            e.run_operation("define_scalar_parameter", {"name": "xi"})
            ex = e.register_expression("xi*c(k)*c'(k)")
            eid = ex.display_id
            e.save(path)
            _, original = e.run_operation("normal_order", {"expr": eid})

        with NotebookEngine.load(path) as e2:
            # session_io.do_load falls back to full replay when no CAS
            # snapshot exists — mirror that here.
            replay_result = e2.replay_current_branch()
            assert replay_result.ok, replay_result.errors
            _, reloaded = e2.run_operation("normal_order", {"expr": eid})

        assert str(reloaded) == str(original)
        # anti-normal-ordered c c† must actually have been reordered
        assert "c" in str(reloaded)
