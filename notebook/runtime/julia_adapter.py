from __future__ import annotations
import subprocess
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

RESULT_BEGIN   = "<<<RESULT_BEGIN>>>"
RESULT_END     = "<<<RESULT_END>>>"
ERROR_SENTINEL = "<<<JULIA_ERROR>>>"
TEX_SEP        = "<<<TEX_SEP>>>"

# Driver script in the same directory; started as a plain Julia script so
# package loading happens at process start, not via interactive REPL stdin.
_DRIVER_PATH = Path(__file__).parent / "_julia_driver.jl"

# Repo root (notebook/runtime/ -> notebook/ -> repo). Holds Project.toml /
# Manifest.toml, the pinned Julia environment (QuantumAlgebra, Symbolics, SymPy,
# Latexify, SymbolicNumericIntegration, JSON3). Launching with --project makes
# package resolution reproducible instead of relying on the user's global env.
_PROJECT_DIR = _DRIVER_PATH.parent.parent.parent

STARTUP_TIMEOUT = 120.0   # first-run precompilation can take ~60–120 s
DEFAULT_TIMEOUT =  30.0


@dataclass
class JuliaResult:
    ok: bool
    value: str
    latex: str = ""
    error: str | None = None


def _terminate_orphan_proc(proc: subprocess.Popen | None) -> None:
    """Safety-net finalizer: terminate a Julia subprocess whose adapter was
    garbage-collected without close() being called (e.g. a test fixture that
    yields an engine but has no teardown).  Must NOT capture the adapter, or the
    weakref would never fire.  Kept module-level for exactly that reason."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# Registry of started Julia subprocesses so a test harness can reap every live
# one at a known boundary, independent of GC.  We track the Popen objects with a
# STRONG reference on purpose: a WeakSet of adapters empties as soon as the
# adapters are garbage-collected, yet their subprocesses can outlive that (the
# persistent asyncio loop / cyclic-GC deferral keeps processes alive after the
# adapter is gone), so a weak registry cannot reap them.  Holding the Popen does
# not retain the adapter, so engine GC is unaffected.
_LIVE_PROCS: "set[subprocess.Popen]" = set()


def terminate_all_live() -> int:
    """Terminate every tracked Julia subprocess that is still running and forget
    the dead ones.  Returns the number terminated.  GC-independent; used by the
    test conftest at module boundaries to bound the concurrent-process peak."""
    n = 0
    for proc in list(_LIVE_PROCS):
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            n += 1
        _LIVE_PROCS.discard(proc)
    return n


class JuliaAdapter:
    """Persistent Julia process adapter.

    Keeps a single Julia subprocess alive across multiple execute() calls so
    that operator definitions and variable bindings are preserved between
    commands.  Use as a context manager or call start() / close() explicitly.

    Julia is started with a driver script (_julia_driver.jl) that loads
    QuantumAlgebra and Symbolics once, then enters a command loop that reads
    one expression per stdin line and writes results with markers.

    First-run startup may take 60–120 s while Julia precompiles packages;
    subsequent starts use the precompile cache and typically finish in <5 s.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Idempotent: a second start() on a LIVE adapter must not spawn a
        # second process.  Without this guard `self._proc` is overwritten and
        # the first Julia is orphaned — it stays in _LIVE_PROCS forever (close()
        # only discards the CURRENT _proc) and stays resident (~1.2 GB) until
        # the adapter is garbage-collected.
        #
        # This is not hypothetical: `with NotebookEngine() as e:` already calls
        # start() via __enter__, and 157 call sites then call e.start() again in
        # the body.  Measured 2026-08-11 — test_engine.py made 48 julia.start
        # calls from 24 adapters, test_perturbation_series.py 36 from 18:
        # exactly double, half of it wasted time and orphaned memory.
        #
        # The same guard was added at the BRIDGE layer in 2026-07-09
        # (ui.notebook_bridge.start_engine, tests/test_engine_lifecycle_leak.py);
        # the adapter layer — which actually owns the subprocess — never got one.
        # A restart after close() is unaffected: close() sets _proc to None.
        if self._proc is not None and self._proc.poll() is None:
            return

        cmd = ["julia", "--startup-file=no"]

        # Pin the environment to the repo-local Project.toml/Manifest.toml when
        # present, so package versions are reproducible rather than whatever the
        # user's global ~/.julia environment happens to hold.
        if (_PROJECT_DIR / "Project.toml").exists():
            cmd.append(f"--project={_PROJECT_DIR}")

        # Check for custom sysimage built by scripts/build_julia_sysimage.jl
        runtime_dir = _DRIVER_PATH.parent
        for ext in [".dylib", ".so", ".dll"]:
            p = runtime_dir / f"sysimage{ext}"
            if p.exists():
                cmd.append(f"--sysimage={p}")
                break
                
        cmd.append(str(_DRIVER_PATH))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._read_until_end(timeout=STARTUP_TIMEOUT)
        # Safety net: if this adapter is dropped without close() (a fixture that
        # yields an engine but never tears it down), GC of the adapter still
        # terminates the Julia process instead of leaking it.
        self._finalizer = weakref.finalize(
            self, _terminate_orphan_proc, self._proc
        )
        _LIVE_PROCS.add(self._proc)

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()   # we own the shutdown from here
            self._finalizer = None
        _LIVE_PROCS.discard(self._proc)
        if self._proc is not None:
            try:
                self._send("exit()\n")
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    def __enter__(self) -> JuliaAdapter:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, command: str) -> JuliaResult:
        """Execute one Julia expression and return the result.

        The command is sent as a single stdin line (internal newlines are
        replaced with ';').  The driver wraps it in try/catch so errors are
        captured and returned as JuliaResult(ok=False, ...).
        """
        if not self.is_running:
            raise RuntimeError("JuliaAdapter is not running; call start() first.")

        safe_cmd = command.strip().replace("\n", "; ")
        self._send(safe_cmd + "\n")
        raw = self._read_until_end()

        if TEX_SEP in raw:
            main_part, tex_part = raw.split(TEX_SEP, 1)
        else:
            main_part, tex_part = raw, ""

        if ERROR_SENTINEL in main_part:
            msg = main_part.replace(ERROR_SENTINEL, "").strip()
            return JuliaResult(ok=False, value="", error=msg or "Julia evaluation error")

        return JuliaResult(ok=True, value=main_part.strip(), latex=tex_part.strip())

    # ------------------------------------------------------------------
    # Snapshot save / load
    # ------------------------------------------------------------------

    def save_snapshot(self, path: str | Path, rendered_cmds: list[str] | None = None) -> bool:
        """Write a Julia script that reconstructs the full CAS session state.

        When *rendered_cmds* is provided (by the engine from its step history),
        the script contains every executed CAS command in order — operator
        declarations (@boson_ops / @fermion_ops) followed by all _nb_Ek
        variable assignments.  Loading this script fully restores Julia state
        without requiring a separate replay pass.

        Falls back to querying Julia for operator definitions only when
        *rendered_cmds* is None (legacy / testing path).

        Returns True on success.
        """
        if rendered_cmds is not None:
            Path(path).write_text("\n".join(rendered_cmds) + "\n", encoding="utf-8")
            return True
        # Legacy fallback: query Julia for operator names only.
        if not self.is_running:
            return False
        try:
            r = self.execute(
                'isdefined(Main, :_nb_defined_ops_) ? '
                'join(string.(_nb_defined_ops_), "\\n") : ""'
            )
            lines: list[str] = []
            if r.ok and r.value and r.value not in ("nothing", '""', ""):
                for name in r.value.strip('"').splitlines():
                    name = name.strip()
                    if name:
                        lines.append(f"@boson_ops {name}")
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
        except Exception:
            return False

    def load_snapshot(self, path: str | Path) -> bool:
        """Restore operator definitions from a snapshot file.

        Returns True only if all commands succeed.
        """
        if not self.is_running:
            return False
        p = Path(path)
        if not p.exists():
            return False
        try:
            for raw_line in p.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                r = self.execute(line)
                if not r.ok:
                    return False
            return True
        except Exception:
            return False

    def get_active_dependencies(self) -> list[tuple[str, str]]:
        """Stub — QuantumAlgebra does not use Maxima-style depends() declarations."""
        return []

    def remove_dependency(self, var: str) -> bool:
        """Stub — no-op for Julia backend."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(text)
        self._proc.stdin.flush()

    def _read_until_end(self, timeout: float | None = None) -> str:
        lines: list[str] = []
        in_result = False
        finished = threading.Event()
        t_out = timeout if timeout is not None else self.timeout

        def _reader() -> None:
            nonlocal in_result
            assert self._proc is not None and self._proc.stdout is not None
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    finished.set()
                    return
                line = line.rstrip("\n")
                if RESULT_BEGIN in line:
                    in_result = True
                    continue
                if RESULT_END in line:
                    finished.set()
                    return
                if in_result:
                    lines.append(line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        if not finished.wait(timeout=t_out):
            raise TimeoutError(f"Julia did not respond within {t_out}s")

        return "\n".join(lines).strip()
