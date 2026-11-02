"""start() must be idempotent on a LIVE adapter (T-2, 2026-08-11).

Measured on the full suite: `with NotebookEngine() as e:` starts the CAS via
__enter__, and 157 call sites then call `e.start()` again in the body.  Because
``start()`` spawned unconditionally, each of those sites

  1. paid the CAS startup cost twice (Julia ~0.92 s each), and
  2. ORPHANED the first subprocess — ``self._proc`` was overwritten, and
     ``close()`` only discards the current ``_proc``, so the first one stayed in
     ``_LIVE_PROCS`` forever and stayed resident (~1.2 GB for Julia).

test_engine.py made 48 julia.start calls from 24 adapters and
test_perturbation_series.py 36 from 18 — exactly double in both.

The sibling guard at the BRIDGE layer (start_engine, close-before-reopen) is
pinned by tests/test_engine_lifecycle_leak.py; this file pins the ADAPTER layer,
which is what actually owns the subprocess.

A restart AFTER close() must still work — close() sets ``_proc`` to None — so
that is pinned too, in both directions.
"""
import pytest

from notebook.runtime.maxima_adapter import MaximaAdapter
from notebook.runtime.maxima_adapter import _LIVE_PROCS as MAXIMA_LIVE


class TestMaximaStartIdempotent:
    def test_double_start_reuses_the_same_process(self):
        a = MaximaAdapter()
        try:
            a.start()
            first = a._proc
            live_after_first = len(MAXIMA_LIVE)

            a.start()                       # the 157-call-site pattern

            assert a._proc is first, "second start() spawned a new process"
            assert first.poll() is None, "the live process was killed"
            assert len(MAXIMA_LIVE) == live_after_first, "registry grew"
        finally:
            a.close()

    def test_adapter_still_works_after_the_redundant_start(self):
        """The guard must return a USABLE adapter, not merely a quiet one."""
        a = MaximaAdapter()
        try:
            a.start()
            a.start()
            r = a.execute("2+2;")
            assert r.ok and "4" in r.value
        finally:
            a.close()

    def test_start_after_close_starts_a_fresh_process(self):
        """close() → start() is a real restart, not swallowed by the guard."""
        a = MaximaAdapter()
        a.start()
        first = a._proc
        a.close()
        assert first.poll() is not None, "close() left the process alive"
        try:
            a.start()
            assert a._proc is not first, "restart after close() was swallowed"
            assert a._proc.poll() is None
        finally:
            a.close()


class TestJuliaStartIdempotent:
    """Same three properties for Julia — the adapter whose process costs
    ~1.2 GB, so the orphaning half of the bug is far more expensive here."""

    @pytest.fixture
    def adapter(self):
        from notebook.runtime.julia_adapter import JuliaAdapter
        a = JuliaAdapter()
        yield a
        a.close()

    def test_double_start_reuses_the_same_process(self, adapter):
        from notebook.runtime.julia_adapter import _LIVE_PROCS as JULIA_LIVE
        adapter.start()
        first = adapter._proc
        live_after_first = len(JULIA_LIVE)

        adapter.start()

        assert adapter._proc is first, "second start() spawned a new Julia"
        assert first.poll() is None
        assert len(JULIA_LIVE) == live_after_first

    def test_adapter_still_works_after_the_redundant_start(self, adapter):
        adapter.start()
        adapter.start()
        r = adapter.execute("1+1")
        assert r.ok and "2" in r.value
