"""Integration tests for MaximaAdapter.

These tests start a real Maxima process. A single process is shared within
each test class (module-scoped fixture) to keep the suite fast and to verify
that state persists across consecutive calls.
"""
import gc
import os
import shutil
import tempfile
from pathlib import Path
import pytest
from notebook.runtime.maxima_adapter import (
    MaximaAdapter, MaximaResult, _parse_maxima_list, _maxima_argv,
    terminate_all_live, _LIVE_PROCS,
)


@pytest.fixture(scope="module")
def maxima():
    with MaximaAdapter() as m:
        yield m


# ---------------------------------------------------------------------------
# Windows maxima.bat launch workaround (_maxima_argv) — pure function, no
# live process needed. macOS's official installer ships a directly
# executable `maxima`, so real coverage of the Windows branches only exists
# via os.name/shutil.which patching here; open_items.md G-3 tracks the real
# Windows-hardware confirmation.
# ---------------------------------------------------------------------------

class TestMaximaArgvWindowsWorkaround:
    _BASE = ["maxima", "--very-quiet", "--disable-readline"]

    def test_non_windows_unchanged(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        assert _maxima_argv() == self._BASE

    def test_windows_exe_unchanged(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(shutil, "which", lambda cmd: r"C:\maxima\bin\maxima.exe")
        assert _maxima_argv() == self._BASE

    def test_windows_not_on_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        assert _maxima_argv() == self._BASE

    def test_windows_bat_wrapped_via_cmd(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(
            shutil, "which", lambda cmd: r"C:\maxima-5.49.0\bin\maxima.bat"
        )
        monkeypatch.setenv("ComSpec", r"C:\Windows\System32\cmd.exe")
        assert _maxima_argv() == [
            r"C:\Windows\System32\cmd.exe", "/c", r"C:\maxima-5.49.0\bin\maxima.bat",
            "--very-quiet", "--disable-readline",
        ]

    def test_windows_cmd_case_insensitive_extension(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(
            shutil, "which", lambda cmd: r"C:\maxima\bin\maxima.BAT"
        )
        monkeypatch.setenv("ComSpec", r"C:\Windows\System32\cmd.exe")
        assert _maxima_argv()[:2] == [r"C:\Windows\System32\cmd.exe", "/c"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_running_before_start(self):
        m = MaximaAdapter()
        assert not m.is_running

    def test_running_after_start(self):
        with MaximaAdapter() as m:
            assert m.is_running

    def test_not_running_after_close(self):
        m = MaximaAdapter()
        m.start()
        assert m.is_running
        m.close()
        assert not m.is_running

    def test_execute_without_start_raises(self):
        m = MaximaAdapter()
        with pytest.raises(RuntimeError, match="not running"):
            m.execute("1+1")

    def test_context_manager(self):
        with MaximaAdapter() as m:
            r = m.execute("1+1")
        assert r.ok
        assert r.value == "2"


# ---------------------------------------------------------------------------
# Process-leak safety net (2026-07-09): weakref finalizer + _LIVE_PROCS
# registry + terminate_all_live(), matching JuliaAdapter's design.  Maxima
# had none of these until now, even though Maxima-mode sessions vastly
# outnumber Julia ones — so any Maxima-side leak (a test instance pinning
# self.engine, or a caller overwriting a live engine reference without
# closing it first) had no automatic recovery path at all.
# ---------------------------------------------------------------------------

class TestProcessLeakSafetyNet:
    def test_start_registers_close_unregisters(self):
        before = len(_LIVE_PROCS)
        m = MaximaAdapter()
        m.start()
        try:
            assert len(_LIVE_PROCS) == before + 1
            assert m._proc in _LIVE_PROCS
        finally:
            m.close()
        assert m._proc is None
        # After close() the registry is back to its prior size (the entry was
        # discarded, not merely marked dead).
        assert len(_LIVE_PROCS) == before

    def test_dropped_without_close_is_finalized_by_gc(self):
        """An adapter dropped without close() must still have its Maxima
        process terminated by the weakref finalizer at GC time (the fixture-
        without-teardown scenario)."""
        m = MaximaAdapter()
        m.start()
        proc = m._proc
        assert proc.poll() is None       # running
        del m
        gc.collect()
        # The finalizer ran on GC and terminated the process.
        assert proc.poll() is not None   # exited

    def test_terminate_all_live_reaps_leaked_process(self):
        """terminate_all_live() (called by conftest at module boundaries) must
        kill a still-running tracked process even when nothing GC'd it."""
        m = MaximaAdapter()
        m.start()
        proc = m._proc
        # Detach the finalizer so ONLY terminate_all_live can reap it — this
        # isolates the registry path from the GC-finalizer path above.
        if m._finalizer is not None:
            m._finalizer.detach()
        try:
            assert proc in _LIVE_PROCS
            assert proc.poll() is None
            reaped = terminate_all_live()
            assert reaped >= 1
            assert proc.poll() is not None       # exited
            assert proc not in _LIVE_PROCS        # forgotten
        finally:
            if proc.poll() is None:
                proc.kill()
            _LIVE_PROCS.discard(proc)


# ---------------------------------------------------------------------------
# Basic algebraic operations
# ---------------------------------------------------------------------------

class TestAlgebraicOps:
    def test_expand(self, maxima):
        r = maxima.execute("expand((x+1)^2)")
        assert r.ok
        assert "x^2" in r.value
        assert "2*x" in r.value

    def test_factor(self, maxima):
        r = maxima.execute("factor(x^2-1)")
        assert r.ok
        assert "(x-1)" in r.value
        assert "(x+1)" in r.value

    def test_rational_simplify(self, maxima):
        r = maxima.execute("ratsimp((x^2-1)/(x-1))")
        assert r.ok
        assert "x+1" in r.value

    def test_substitute(self, maxima):
        r = maxima.execute("subst(2, x, x^2+1)")
        assert r.ok
        assert "5" in r.value

    def test_simplify(self, maxima):
        r = maxima.execute("ratsimp(x/x)")
        assert r.ok
        assert r.value.strip() == "1"


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------

class TestCalculus:
    def test_differentiate(self, maxima):
        r = maxima.execute("diff(x^3, x)")
        assert r.ok
        assert "3*x^2" in r.value

    def test_integrate_indefinite(self, maxima):
        r = maxima.execute("integrate(x^2, x)")
        assert r.ok
        assert "x^3" in r.value

    def test_limit(self, maxima):
        r = maxima.execute("limit(sin(x)/x, x, 0)")
        assert r.ok
        assert r.value.strip() == "1"

    def test_taylor(self, maxima):
        r = maxima.execute("taylor(exp(x), x, 0, 3)")
        assert r.ok
        assert "x" in r.value


# ---------------------------------------------------------------------------
# Equation solving
# ---------------------------------------------------------------------------

class TestSolving:
    def test_solve_linear(self, maxima):
        r = maxima.execute("solve(2*x+4=0, x)")
        assert r.ok
        assert "-2" in r.value

    def test_solve_quadratic(self, maxima):
        r = maxima.execute("solve(x^2-4=0, x)")
        assert r.ok
        assert "2" in r.value


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_to_latex(self, maxima):
        r = maxima.execute("tex1(x^2/2)")
        assert r.ok
        assert "\\frac" in r.value or "x^2" in r.value


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_variable_assignment_persists(self, maxima):
        maxima.execute("_test_val_: 99$")
        r = maxima.execute("_test_val_")
        assert r.ok
        assert "99" in r.value

    def test_sequential_commands_build_on_each_other(self, maxima):
        maxima.execute("_a_: 3$")
        maxima.execute("_b_: 4$")
        r = maxima.execute("_a_^2 + _b_^2")
        assert r.ok
        assert "25" in r.value


# ---------------------------------------------------------------------------
# LaTeX output via tex1
# ---------------------------------------------------------------------------

class TestLatexOutput:
    def test_result_has_latex_field(self, maxima):
        r = maxima.execute("x^2-1")
        assert hasattr(r, "latex")

    def test_simple_expr_latex_nonempty(self, maxima):
        r = maxima.execute("x^2-1")
        assert r.ok
        assert r.latex != ""

    def test_fraction_latex_uses_over_or_frac(self, maxima):
        r = maxima.execute("integrate(x^2, x)")
        assert r.ok
        assert "x^3" in r.latex
        assert "3" in r.latex

    def test_factor_latex_has_parens(self, maxima):
        r = maxima.execute("factor(x^2-1)")
        assert r.ok
        assert "x-1" in r.latex
        assert "x+1" in r.latex

    def test_diff_latex_has_exponent(self, maxima):
        r = maxima.execute("diff(x^3, x)")
        assert r.ok
        assert "x^2" in r.latex

    def test_canonical_and_latex_are_distinct_for_fraction(self, maxima):
        r = maxima.execute("integrate(x^2, x)")
        assert r.ok
        # canonical: x^3/3   latex: {{x^3}\over{3}} or similar
        assert r.value != r.latex or "/" in r.value

    def test_error_result_latex_empty(self, maxima):
        r = maxima.execute('error("forced")')
        assert not r.ok
        assert r.latex == ""

    def test_latex_field_on_algebraic_simplify(self, maxima):
        r = maxima.execute("ratsimp((x^2-1)/(x-1))")
        assert r.ok
        assert r.latex == "x+1"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_explicit_error_returns_not_ok(self, maxima):
        r = maxima.execute('error("deliberate test error")')
        assert not r.ok
        assert r.error is not None

    def test_result_is_dataclass(self, maxima):
        r = maxima.execute("1+1")
        assert isinstance(r, MaximaResult)
        assert r.error is None

    def test_adapter_still_usable_after_error(self, maxima):
        maxima.execute('error("should not break the adapter")')
        r = maxima.execute("2+2")
        assert r.ok
        assert r.value.strip() == "4"


# ---------------------------------------------------------------------------
# _parse_maxima_list
# ---------------------------------------------------------------------------

class TestParseMaximaList:
    def test_empty(self):
        assert _parse_maxima_list("[]") == []
        assert _parse_maxima_list("") == []

    def test_simple(self):
        assert _parse_maxima_list("[x,y,z]") == ["x", "y", "z"]

    def test_function_args(self):
        assert _parse_maxima_list("[f(a),g(a,b)]") == ["f(a)", "g(a,b)"]

    def test_inequalities(self):
        assert _parse_maxima_list("[y > 0,n > 0,10 > n]") == ["y > 0", "n > 0", "10 > n"]

    def test_no_brackets(self):
        assert _parse_maxima_list("x,y") == ["x", "y"]


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------

class TestSnapshot:
    """Each test uses its own fresh MaximaAdapter so state does not leak."""

    def _fresh(self):
        m = MaximaAdapter(timeout=30)
        m.start()
        return m

    def test_save_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m = self._fresh()
        try:
            m.execute("x: 7")
            assert m.save_snapshot(path)
            assert path.exists()
            assert path.stat().st_size > 0
        finally:
            m.close()
            path.unlink(missing_ok=True)

    def test_variable_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("myvar: 123")
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            r = m2.execute("myvar")
            assert r.ok
            assert "123" in r.value
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_function_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("myfunc(a) := a^2 + 3")
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            r = m2.execute("myfunc(4)")
            assert r.ok
            assert "19" in r.value
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_assume_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("assume(omega > 0)")
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            # integrate(1/omega, omega) should give log(omega) only if omega > 0 is known
            r = m2.execute("facts()")
            assert r.ok
            assert "omega" in r.value
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_depends_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("depends([q], [t])")
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            # diff of q w.r.t. t should give 'diff(q,t,1) (not 0)
            r = m2.execute("diff(q, t)")
            assert r.ok
            assert r.value != "0"
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_missing_snapshot_returns_false(self):
        m = self._fresh()
        try:
            assert not m.load_snapshot("/nonexistent/path/file.mac")
        finally:
            m.close()

    def test_snapshot_excludes_internal_vars(self):
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m = self._fresh()
        try:
            m.execute("userx: 5")   # triggers _res_, _errcatch_res_, etc.
            m.save_snapshot(path)
            content = path.read_text()
            assert "_res_" not in content
            assert "_errcatch_res_" not in content
            assert "userx" in content
        finally:
            m.close()
            path.unlink(missing_ok=True)

    def test_assume_facts_roundtrip(self):
        """assume → facts() shows assumption → snapshot → reload → facts() restored."""
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("assume(alpha > 0)")
            facts_before = m1.execute("facts()").value
            assert "alpha" in facts_before
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            facts_after = m2.execute("facts()").value
            assert "alpha" in facts_after
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_forget_not_restored(self):
        """assume → forget → snapshot → reload: deleted assumption must NOT reappear."""
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("assume(beta > 0)")
            m1.execute("forget(beta > 0)")
            facts = m1.execute("facts()").value
            assert "beta" not in facts      # confirm it's gone before saving
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            facts_after = m2.execute("facts()").value
            assert "beta" not in facts_after
        finally:
            m2.close()
            path.unlink(missing_ok=True)

    def test_depends_in_dependencies(self):
        """depends(q,t) → dependencies contains q(t) → snapshot → reload → diff(q,t) != 0."""
        with tempfile.NamedTemporaryFile(suffix='.mac', delete=False) as tf:
            path = Path(tf.name)
        m1 = self._fresh()
        try:
            m1.execute("depends([q], [t])")
            deps = m1.execute("dependencies").value
            assert "q" in deps
            m1.save_snapshot(path)
        finally:
            m1.close()

        m2 = self._fresh()
        try:
            assert m2.load_snapshot(path)
            deps_after = m2.execute("dependencies").value
            assert "q" in deps_after
            r = m2.execute("diff(q, t)")
            assert r.ok
            assert r.value.strip() != "0"
        finally:
            m2.close()
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# get_active_dependencies / remove_dependency
# ---------------------------------------------------------------------------

class TestDependencyManagement:
    """Each test uses its own fresh MaximaAdapter to avoid state leakage."""

    def _fresh(self):
        m = MaximaAdapter(timeout=30)
        m.start()
        return m

    def test_no_deps_returns_empty(self):
        m = self._fresh()
        try:
            assert m.get_active_dependencies() == []
        finally:
            m.close()

    def test_single_dep_detected(self):
        m = self._fresh()
        try:
            m.execute("depends([y], [x])")
            deps = m.get_active_dependencies()
            assert ("y", "x") in deps
        finally:
            m.close()

    def test_multiple_deps_detected(self):
        m = self._fresh()
        try:
            m.execute("depends([q], [t])")
            m.execute("depends([p], [t])")
            deps = m.get_active_dependencies()
            assert ("q", "t") in deps
            assert ("p", "t") in deps
        finally:
            m.close()

    def test_remove_dependency_clears_dep(self):
        m = self._fresh()
        try:
            m.execute("depends([y], [x])")
            assert ("y", "x") in m.get_active_dependencies()
            ok = m.remove_dependency("y")
            assert ok
            assert ("y", "x") not in m.get_active_dependencies()
        finally:
            m.close()

    def test_remove_dep_stops_diff_pollution(self):
        """After remove_dependency, diff treats var as independent."""
        m = self._fresh()
        try:
            m.execute("depends([y], [x])")
            r_before = m.execute("diff((x+y)^2, x)")
            assert "diff(y" in r_before.value   # polluted: y treated as f(x)
            m.remove_dependency("y")
            r_after = m.execute("diff((x+y)^2, x)")
            assert "diff(y" not in r_after.value  # clean: y independent
            assert "2" in r_after.value
        finally:
            m.close()
