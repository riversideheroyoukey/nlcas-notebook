"""Where does the 16 minutes actually go?

T-1 (tests/conftest.py, 2026-07-09) says the suite is dominated by CAS
subprocess ROUND-TRIPS rather than process startup.  This session has been
quoting that without measuring it.  This plugin counts and times every
adapter call so the claim becomes a number.

2026-08-11 extension (T-2, handoffs/test_time_julia_handoff.md §5-0/1/3).
The bare totals cannot answer the question that decides whether process reuse
is worth building, so three measurements were added:

  §5-1  JIT warm-up vs steady state.  Per-process, the FIRST calls are
        expensive and later ones are nearly free.  Timing each adapter
        INSTANCE separately turns "reuse recovers 50-70%" from an
        extrapolation into a measurement: the recoverable amount is the
        excess over each instance's own steady-state cost.
  §5-0  Memory.  Reuse means HOLDING warmed JIT code, so the ceiling on a
        shared-process pool is memory, not time.  A sampler thread records
        live process count and RSS.
  §5-3  Per-file start counts, and whether a file's sessions are CONCURRENT
        or SEQUENTIAL — a pool needs one slot per concurrent session, so
        counting starts alone does not size it.

The plugin only wraps and observes; it changes no behaviour.

Usage:  pytest -p cas_profile ...   (with this directory on PYTHONPATH)
        CAS_PROFILE_JSON=out.json   also dumps the raw per-call data
"""
import os
import statistics
import subprocess
import threading
import time

_stats = {}

# --- §5-1: per adapter INSTANCE, the duration of each execute() ------------
# keyed by (label, id(instance)) -> list[float].  id() is reused after an
# instance is freed, so a monotonic serial is attached to each instance on
# first sight instead (see _instance_key).
_calls = {}
_serial = {"n": 0}

# --- §5-3: julia.start / maxima.start attributed to the running test file --
_starts_by_file = {}
_current_file = ["(collection)"]

# --- §5-0: sampler ---------------------------------------------------------
_samples = []            # (t, n_julia, n_maxima, rss_julia_mb, rss_maxima_mb)
_sampler_stop = threading.Event()


def _instance_key(label, obj):
    key = getattr(obj, "_cas_profile_serial", None)
    if key is None:
        _serial["n"] += 1
        key = _serial["n"]
        try:
            obj._cas_profile_serial = key
        except Exception:                              # noqa: BLE001
            pass
    return (label, key)


def _wrap(cls, name, label, per_instance=False, count_by_file=False):
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_cas_profiled", False):
        return

    def timed(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(self, *a, **kw)
        finally:
            dt = time.perf_counter() - t0
            s = _stats.setdefault(label, [0, 0.0])
            s[0] += 1
            s[1] += dt
            if per_instance:
                _calls.setdefault(_instance_key(label, self), []).append(dt)
            if count_by_file:
                f = _starts_by_file.setdefault(_current_file[0], {})
                f[label] = f.get(label, 0) + 1

    timed._cas_profiled = True
    setattr(cls, name, timed)


# ---------------------------------------------------------------------------
# §5-0 sampler: live subprocess count and RSS.
#
# psutil is not installed in .venv and this must not add a dependency, so the
# pids come from the adapters' own _LIVE_PROCS registries (already maintained
# for the leak safety net) and RSS from one `ps` call per sample.
# ---------------------------------------------------------------------------
def _live_pids():
    out = {}
    for mod, tag in (("julia_adapter", "julia"), ("maxima_adapter", "maxima")):
        pids = []
        try:
            m = __import__(f"notebook.runtime.{mod}", fromlist=["_LIVE_PROCS"])
            for p in list(m._LIVE_PROCS):
                if p is not None and p.poll() is None:
                    pids.append(p.pid)
        except Exception:                              # noqa: BLE001
            pass
        out[tag] = pids
    return out


def _rss_mb(pids):
    """Total RSS of the given pids, in MB.  0 when none are alive."""
    if not pids:
        return 0.0
    try:
        r = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, timeout=10,
        )
        return sum(int(x) for x in r.stdout.split() if x.isdigit()) / 1024.0
    except Exception:                                  # noqa: BLE001
        return 0.0


def _sample_loop(interval=2.0):
    t0 = time.perf_counter()
    while not _sampler_stop.wait(interval):
        live = _live_pids()
        _samples.append((
            time.perf_counter() - t0,
            len(live["julia"]), len(live["maxima"]),
            _rss_mb(live["julia"]), _rss_mb(live["maxima"]),
        ))


def pytest_configure(config):
    from notebook.runtime.maxima_adapter import MaximaAdapter
    _wrap(MaximaAdapter, "execute", "maxima.execute", per_instance=True)
    _wrap(MaximaAdapter, "execute_script", "maxima.execute_script")
    _wrap(MaximaAdapter, "start", "maxima.start", count_by_file=True)
    _wrap(MaximaAdapter, "close", "maxima.close")
    try:
        from notebook.runtime.julia_adapter import JuliaAdapter
        _wrap(JuliaAdapter, "execute", "julia.execute", per_instance=True)
        _wrap(JuliaAdapter, "start", "julia.start",
              per_instance=True, count_by_file=True)
        _wrap(JuliaAdapter, "close", "julia.close")
    except Exception:                                  # noqa: BLE001
        pass
    config._cas_t0 = time.perf_counter()
    t = threading.Thread(target=_sample_loop, daemon=True)
    t.start()
    config._cas_sampler = t


def pytest_runtest_setup(item):
    _current_file[0] = str(getattr(item, "fspath", "?"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _steady(durations):
    """Per-call cost once this process is warm.

    The first call in a process pays JIT compilation; later ones do not.
    Taking the median of everything after the third call (falling back to the
    minimum when there are too few) gives the marginal cost of one more call
    on an ALREADY warm process — which is exactly what survives if the
    process is reused.
    """
    tail = durations[3:] or durations[1:]
    if not tail:
        return durations[0]
    return statistics.median(tail)


def _warmup_report(label):
    """Split this label's total into steady-state work and per-process warm-up.

    warm-up = what a reused process would NOT have to pay again.
    """
    inst = [d for (lab, _), d in _calls.items() if lab == label]
    if not inst:
        return None
    total = sum(sum(d) for d in inst)
    steady_total = sum(len(d) * _steady(d) for d in inst)
    firsts = [d[0] for d in inst]
    return {
        "processes": len(inst),
        "calls": sum(len(d) for d in inst),
        "total": total,
        "steady": steady_total,
        "warmup": total - steady_total,
        "first_call_median": statistics.median(firsts),
        "first_call_total": sum(firsts),
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    _sampler_stop.set()
    wall = time.perf_counter() - getattr(config, "_cas_t0", time.perf_counter())
    total = sum(v[1] for v in _stats.values())
    w = terminalreporter.write_line
    w("")
    w("=== CAS time profile ===")
    w(f"{'call':24} {'count':>8} {'seconds':>10} {'% of wall':>10}")
    for label, (n, t) in sorted(_stats.items(), key=lambda kv: -kv[1][1]):
        w(f"{label:24} {n:8d} {t:10.1f} {100 * t / wall:9.1f}%")
    w(f"{'TOTAL in adapters':24} {'':8} {total:10.1f} {100 * total / wall:9.1f}%")
    w(f"{'wall clock':24} {'':8} {wall:10.1f}")
    w(f"{'everything else':24} {'':8} {wall - total:10.1f} "
      f"{100 * (wall - total) / wall:9.1f}%")

    # --- §5-1 -------------------------------------------------------------
    w("")
    w("=== §5-1  execute(): JIT warm-up vs steady state ===")
    w(f"{'label':18} {'procs':>6} {'calls':>7} {'total':>8} {'steady':>8} "
      f"{'warm-up':>8} {'1st(med)':>9}")
    recoverable = 0.0
    for label in ("julia.execute", "maxima.execute"):
        r = _warmup_report(label)
        if r is None:
            continue
        w(f"{label:18} {r['processes']:6d} {r['calls']:7d} {r['total']:8.1f} "
          f"{r['steady']:8.1f} {r['warmup']:8.1f} {r['first_call_median']:9.3f}")
        if label.startswith("julia"):
            recoverable += r["warmup"]
    js = _stats.get("julia.start", [0, 0.0])
    w("")
    w(f"  julia.start total ..................... {js[1]:8.1f}s ({js[0]} starts)")
    w(f"  julia.execute warm-up (re-paid JIT) ... {recoverable:8.1f}s")
    w(f"  --> upper bound recoverable by reuse .. {js[1] + recoverable:8.1f}s "
      f"= {100 * (js[1] + recoverable) / wall:.1f}% of wall")
    w("  (upper bound: assumes ONE warm process would serve every test.")
    w("   A pool of N re-pays warm-up N times; see §5-0 for what N can be.)")

    # --- started but never used -------------------------------------------
    starters = {k for (lab, k) in _calls if lab == "julia.start"}
    users = {k for (lab, k) in _calls if lab == "julia.execute"}
    if starters:
        idle = starters - users
        idle_t = sum(sum(_calls[("julia.start", k)]) for k in idle)
        restarts = sum(len(_calls[("julia.start", k)]) - 1 for k in starters)
        w("")
        w("=== julia adapters that started but never executed ===")
        w(f"  adapter instances that called start() ... {len(starters)}")
        w(f"  ...of which never executed anything ..... {len(idle)}  "
          f"({idle_t:.1f}s of start time)")
        w(f"  extra start() calls on a reused instance  {restarts}")

    # --- §5-0 -------------------------------------------------------------
    if _samples:
        w("")
        w("=== §5-0  live CAS processes and RSS (sampled 2s) ===")
        pj = max(s[1] for s in _samples)
        pm = max(s[2] for s in _samples)
        rj = max(s[3] for s in _samples)
        rm = max(s[4] for s in _samples)
        rt = max(s[3] + s[4] for s in _samples)
        w(f"  peak live julia processes ............ {pj}")
        w(f"  peak live maxima processes ........... {pm}")
        w(f"  peak julia RSS ....................... {rj:8.0f} MB")
        w(f"  peak maxima RSS ...................... {rm:8.0f} MB")
        w(f"  peak combined CAS RSS ................ {rt:8.0f} MB")
        nz = [s[3] / s[1] for s in _samples if s[1]]
        if nz:
            w(f"  per-julia-process RSS (median) ....... "
              f"{statistics.median(nz):8.0f} MB")
        w(f"  samples .............................. {len(_samples)}")

    # --- §5-3 -------------------------------------------------------------
    jf = sorted(
        ((f, d.get("julia.start", 0)) for f, d in _starts_by_file.items()),
        key=lambda kv: -kv[1])
    jf = [x for x in jf if x[1]]
    if jf:
        w("")
        w("=== §5-3  julia.start by test file ===")
        w(f"  files starting julia: {len(jf)}   "
          f"starts: {sum(n for _, n in jf)}")
        for f, n in jf[:20]:
            w(f"  {n:5d}  {os.path.basename(f)}")
        if len(jf) > 20:
            w(f"  ... {len(jf) - 20} more files")

    path = os.environ.get("CAS_PROFILE_JSON")
    if path:
        import json
        with open(path, "w") as fh:
            json.dump({
                "wall": wall,
                "stats": _stats,
                "calls": {f"{lab}#{k}": d for (lab, k), d in _calls.items()},
                "starts_by_file": _starts_by_file,
                "samples": _samples,
            }, fh)
        w(f"\nraw data -> {path}")
