"""
tests/conftest.py
-----------------
Pytest configuration shared across all test modules.

Problem: tests/browser/test_e2e.py uses Playwright which internally starts an
asyncio event loop via pytest-playwright.  When the full test suite runs, that
loop persists across module boundaries and causes subsequent tests that call
asyncio.run() (test_debug_repl.py, test_supervisor_pipeline.py,
test_persistence.py) to fail with:
  RuntimeError: asyncio.run() cannot be called from a running event loop

Fix: Apply nest_asyncio after the browser tests complete so that asyncio.run()
can be called inside an already-running event loop.
"""
import asyncio
import inspect
import re

import pytest

# T-1 (docs/open_items.md): the full suite takes ~14 minutes because it is
# dominated by CAS subprocess round-trips (thousands of julia/maxima
# execute() calls), not process-startup or memory overhead (profiled
# 2026-07-09 — see open_items.md's T-1 entry for the measurements that ruled
# out swap/leak causes). Marking every heavy integration test `slow` lets
# daily iteration run `pytest -m "not slow"` (seconds, not minutes); the full
# suite (no -m filter) remains the pre-merge gate.
#
# Auto-detected from source rather than manually decorated: with ~2800 tests
# across 130+ files, hand-annotating each one is exactly the kind of
# maintenance burden this fix is supposed to remove, and it would silently
# stop covering new tests. Detecting "this test starts a live Maxima/Julia
# session" from its own source text is a reasonable proxy for "this is the
# thing that's slow" — every CAS-touching test in this suite does so via one
# of a handful of call shapes (confirmed by grep across tests/*.py), so a
# regex over inspect.getsource(item.function) covers the actual integration
# tests without touching product code or requiring per-test opt-in.
#
# The test's OWN source isn't always enough: many files (test_maxima_adapter.py,
# test_engine.py, ...) hand the live session to tests via a pytest fixture
# (e.g. `def test_expand(self, maxima): ...`), so the CAS-starting call lives
# in the fixture's source, not the test's. Fall back to scanning every fixture
# the test (transitively) depends on — item.fixturenames already includes
# indirect dependencies, and name2fixturedefs resolves each name to its
# defining function.
_CAS_SESSION_RE = re.compile(
    r"start_engine\(|MaximaAdapter\(|JuliaAdapter\(|NotebookEngine\(|nb\.get_engine\(\)"
)


def _source_starts_cas_session(func) -> bool:
    try:
        return bool(_CAS_SESSION_RE.search(inspect.getsource(func)))
    except (OSError, TypeError):
        return False


# ---------------------------------------------------------------------------
# EXAMPLES-DRIFT opt-in gate (docs/open_items.md)
#
# tests/test_examples_up_to_date.py regenerates every worked example and compares
# it with what is committed.  That is the check that was MISSING when the
# examples silently fell ~5 root-cause fixes behind the code (measured
# 2026-08-06: regenerating on a clean tree changed 90 files, and the published
# .tex still showed `\dot{theta}` and literal `*`).
#
# It runs the whole scenario corpus through a live CAS, so it takes minutes —
# too slow for routine iteration, exactly like the Playwright E2E tests.  Same
# treatment, same precedent (tests/browser/conftest.py): skipped by default,
# enabled with `--run-scenarios` or RUN_SCENARIOS=1.  Run it before merging a
# change to the parser, the converter, the display bridge or the export.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-scenarios", action="store_true", default=False,
        help="regenerate the worked examples and check they match what is "
             "committed (slow: runs every scenario through a live CAS)",
    )


def _scenarios_enabled(config) -> bool:
    import os
    return bool(config.getoption("--run-scenarios") or os.environ.get("RUN_SCENARIOS"))


def pytest_collection_modifyitems(config, items):
    _skip_scenarios = pytest.mark.skip(
        reason="examples drift check: run with --run-scenarios or RUN_SCENARIOS=1")
    _enabled = _scenarios_enabled(config)
    # Both files replay the whole scenario corpus through a live CAS.  The gate
    # is keyed on the file, not on the `scenarios` marker, so a slow test cannot
    # be left ungated by forgetting the marker — measured 2026-08-23, when
    # test_instruction_corpus's replay class carried the marker and ran anyway
    # because only test_examples_up_to_date was named here.
    for item in items:
        if ("test_examples_up_to_date" in str(item.fspath)
                or ("test_instruction_corpus" in str(item.fspath)
                    and item.get_closest_marker("scenarios"))):
            item.add_marker(pytest.mark.scenarios)
            if not _enabled:
                item.add_marker(_skip_scenarios)
            continue
    for item in items:
        func = getattr(item, "function", None)
        if func is not None and _source_starts_cas_session(func):
            item.add_marker(pytest.mark.slow)
            continue
        info = getattr(item, "_fixtureinfo", None)
        fixturenames = getattr(item, "fixturenames", None)
        if info is None or not fixturenames:
            continue
        for name in fixturenames:
            defs = info.name2fixturedefs.get(name)
            if not defs:
                continue
            if _source_starts_cas_session(defs[-1].func):
                item.add_marker(pytest.mark.slow)
                break


def pytest_configure(config):
    """Install nest_asyncio if available so asyncio.run() works in all tests."""
    config.addinivalue_line(
        "markers",
        "scenarios: worked-example drift check (opt-in via --run-scenarios)")
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass


_LAST_TEST_MODULE = [None]


def pytest_runtest_setup(item):
    """Reap every live Julia and Maxima subprocess when the test MODULE changes.

    Many fixtures hold a second_quantization (Julia) or standard (Maxima) engine
    for the whole class/module. Under the full suite their teardown is DEFERRED
    (the persistent asyncio loop / nest_asyncio keeps the engines strongly
    reachable, so neither GC nor `with`-block close() fires promptly) and the
    live count climbs monotonically across modules — observed up to ~59 Julia
    processes (~27 GB + swap) before this hook existed.

    A teardown fixture / per-test gc.collect() can't fix this (the engines are
    strongly reachable, not cyclic garbage). This hook runs deterministically
    before each test: on a MODULE change the previous module's tests are all done
    and the new module hasn't built its fixtures yet, so terminating every tracked
    subprocess is safe and bounds the cross-module accumulation. GC-independent
    (the registries hold the subprocesses strongly).

    Reaping at the finer CLASS boundary would lower the peak further but is
    UNSAFE: a module-scoped fixture is shared across the module's classes, so
    killing at a class boundary kills an in-use engine (it broke 57 tests when
    tried for Julia). Module granularity keeps the peak bounded without that
    risk. Applies automatically to any future Julia or Maxima test; no
    session-scoped fixture of either kind exists.

    Maxima parity (2026-07-09): Maxima-mode sessions vastly outnumber Julia
    ones in this suite, but MaximaAdapter had no equivalent tracked-registry
    safety net until now (only Julia did) — any Maxima-side leak (e.g. a test
    instance pinning `self.engine`, or a caller overwriting a live engine
    reference without closing it first) had no automatic recovery at all.
    MaximaAdapter now carries the same weakref-finalizer + _LIVE_PROCS registry
    as JuliaAdapter, so the same module-boundary reap now covers both.
    """
    mod = getattr(getattr(item, "module", None), "__name__", None)
    if mod != _LAST_TEST_MODULE[0]:
        _LAST_TEST_MODULE[0] = mod
        try:
            from notebook.runtime.julia_adapter import terminate_all_live as _term_julia
            _term_julia()
        except Exception:
            pass
        try:
            from notebook.runtime.maxima_adapter import terminate_all_live as _term_maxima
            _term_maxima()
        except Exception:
            pass


def pytest_runtest_teardown(item, nextitem):
    """Drop engine references a test stored on its instance, then GC.

    Many test classes do ``self.engine = e`` in a function-scoped fixture. pytest
    keeps the test instance reachable, so that attribute pins the engine: its
    Julia subprocess is not reaped until the module ends (the live count climbs to
    dozens within a single module). A standalone engine, by contrast, dies the
    instant it goes out of scope — so simply clearing the instance attributes and
    forcing a collection lets the engine (and its Julia process) be reclaimed
    right after each test.

    A *class*-scoped engine is also held by the fixture cache, so clearing the
    instance attribute does not kill it (safe). This runs as a hook, not a
    fixture, so it is not subject to the deferred fixture finalization."""
    inst = getattr(item, "instance", None)
    cleared = False
    if inst is not None:
        for attr in ("engine", "sup", "eng", "e"):
            if attr in vars(inst):
                try:
                    setattr(inst, attr, None)
                    cleared = True
                except Exception:
                    pass
    if cleared:
        import gc
        gc.collect()
