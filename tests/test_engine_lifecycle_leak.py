"""Process-leak guards on the module-level engine lifecycle (2026-07-09).

The full test suite was observed spiking CPU and building tens of GB of swap.
Root-cause investigation found two asymmetries with no automatic recovery:

1. ui/notebook_bridge.start_engine() overwrote the module-level _engine
   reference unconditionally.  Called twice without an intervening
   stop_engine(), the first engine's CAS subprocess was orphaned with no
   caller left to close it.  Added a defensive close-before-reopen guard.

2. MaximaAdapter (unlike JuliaAdapter) had no weakref-finalizer / tracked
   registry safety net at all, even though Maxima-mode sessions dominate the
   suite.  (Covered by test_maxima_adapter.py::TestProcessLeakSafetyNet.)

These tests pin down #1 by observing the shared MaximaAdapter _LIVE_PROCS
registry: a double start_engine() must not grow the live-process count.
"""
import ui.notebook_bridge as nb
from notebook.runtime.maxima_adapter import _LIVE_PROCS


class TestStartEngineNoLeak:
    def test_double_start_without_stop_does_not_leak_process(self):
        """start_engine() twice with no stop_engine() in between must close
        the first engine's Maxima process rather than orphan it."""
        # Make sure we begin from a clean module state.
        nb.stop_engine()
        before = len(_LIVE_PROCS)
        try:
            nb.start_engine(title="leak_first", mode="standard")
            first_proc = nb.get_engine()._adapter._proc
            assert len(_LIVE_PROCS) == before + 1

            # Re-start WITHOUT stopping — the bug scenario.
            nb.start_engine(title="leak_second", mode="standard")
            second_proc = nb.get_engine()._adapter._proc

            # The first process was terminated by the guard, not leaked:
            assert first_proc is not second_proc
            assert first_proc.poll() is not None        # exited
            # …and the registry did not grow beyond a single live engine.
            assert len(_LIVE_PROCS) == before + 1
        finally:
            nb.stop_engine()
        assert len(_LIVE_PROCS) == before

    def test_normal_start_stop_is_balanced(self):
        """The ordinary start→stop path leaves the registry exactly as it
        found it (guard must not perturb the common case)."""
        nb.stop_engine()
        before = len(_LIVE_PROCS)
        nb.start_engine(title="balanced", mode="standard")
        assert len(_LIVE_PROCS) == before + 1
        nb.stop_engine()
        assert len(_LIVE_PROCS) == before
