from __future__ import annotations
import os
import re
import shutil
import signal
import subprocess
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path

RESULT_BEGIN   = "<<<RESULT_BEGIN>>>"
RESULT_END     = "<<<RESULT_END>>>"
ERROR_SENTINEL = "<<<MAXIMA_ERROR>>>"
TEX_SEP        = "<<<TEX_SEP>>>"
SYNC_SENTINEL  = "<<<NB_SYNC>>>"
DEFAULT_TIMEOUT = 30.0

# A SYNTAX error is not an `errcatch` error: Maxima's parser fails before
# errcatch ever evaluates, and it then discards input only up to the next
# statement terminator.  So the guard variables kept the values the PREVIOUS
# command left behind — `_errcatch_res_` was still non-empty — and the block
# below faithfully reported success and printed `_res_`, an unbound name, as
# the answer.  Measured 2026-08-20: `simplify` on `x' + x` (a primed variable,
# which this notation layer lowers to Maxima verbatim) registered `_res_` as a
# successful result, and so did `1 +`, `(1+2` and `@@@`.  Every notation this
# system cannot yet lower correctly therefore came back as a plausible-looking
# answer instead of an error.
#
# Clearing the flag in a statement of its OWN is what makes the difference: a
# syntax error in the next statement now leaves it empty, which the existing
# `emptyp` branch already reads as failure.  This is NOT the rejected guard
# described below — a declaration still binds nothing and still succeeds.
_GUARD_RESET = "_errcatch_res_: []$\n"

# NOTE — do not add a guard here that rejects a result equal to `_res_`.
# It looks like a non-answer wearing the shape of an answer, and 2026-08-03 one
# was added on exactly that reasoning; measured, it broke five passing tests.
# `_res_` is the ORDINARY outcome for any command that is not a single
# expression: `errcatch(_res_:(declare(Delta, scalar); true))` never binds the
# name (inside parentheses Maxima's separator is `,`, not `;`), errcatch still
# returns non-empty, and the unbound name is printed.  Every declaration in the
# app takes that shape.  A `_res_` reaching the timeline as a VALUE is the real
# defect, and it is stopped where a value is produced from an operand that has
# none — `NotebookEngine._reject_display_only_operands`.

# Maxima asks the caller a question when a fact it needs is not in its
# assumption database — "Is omega_D positive, negative or zero?" from
# `integrate`, and the rest of the ask* family.  Measured 2026-08-01: it
# prints the question to stdout and then BLOCKS on stdin, so the remaining
# lines of the command block we already wrote are eaten as (invalid) answers
# and it re-asks once per line.  Nothing ever prints RESULT_END, the read
# times out after 30s, and every later command is written into a process
# still waiting for an answer — the session is dead with no diagnosis.
#
# We do NOT answer: choosing pos/neg/zero on the researcher's behalf is a
# modeling decision (CLAUDE.md — no hardcoded modeling choices), and a wrong
# guess yields a plausible wrong integral.  Instead the question is detected,
# the computation is interrupted, and the failure names what was asked.
_QUESTION_RE = re.compile(r"^\s*Is\b.*\?\s*$")
QUESTION_MARKER = "<<<CAS_QUESTION>>>"

# Commands whose results are side-effect lists (e.g. [x > 0]) that do not
# benefit from tex1() and can trigger assumption-system loops or unexpected
# stdout output inside tex1() when they contain inequalities.
_NO_TEX_CMD_RE = re.compile(r'^(?:assume|forget|depends|remove)\s*\(')


def _terminate_orphan_proc(proc: subprocess.Popen | None) -> None:
    """Safety-net finalizer: terminate a Maxima subprocess whose adapter was
    garbage-collected without close() being called (e.g. a test fixture that
    yields an engine but has no teardown).  Must NOT capture the adapter, or the
    weakref would never fire.  Kept module-level for exactly that reason.
    Mirrors julia_adapter.py's identical helper."""
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


# Registry of started Maxima subprocesses so a test harness can reap every
# live one at a known boundary, independent of GC — the same rationale and
# strong-reference design as julia_adapter.py's _LIVE_PROCS (Maxima had no
# equivalent registry, unlike Julia, even though Maxima-mode sessions vastly
# outnumber Julia ones in the test suite; any leak here — e.g. an engine
# pinned by a test instance, or a caller overwriting a live engine reference
# without closing it first — had no automatic recovery path at all).
_LIVE_PROCS: "set[subprocess.Popen]" = set()


def _maxima_argv() -> list[str]:
    """Resolve the Maxima command line to launch.

    Windows-only quirk: the official Maxima Windows installer ships
    maxima.bat (no maxima.exe on PATH), but Win32 CreateProcess - which
    Popen(..., shell=False) uses - only resolves an extensionless PATH
    lookup to .exe/.com, never .bat/.cmd, so a direct ["maxima", ...] Popen
    raises FileNotFoundError there even though `Get-Command maxima` finds it
    fine. Batch files must be launched through cmd.exe /c. macOS/Linux ship
    `maxima` as a directly executable script/binary, so os.name != "nt"
    there and this function returns argv unchanged - this branch cannot
    affect those platforms.
    """
    argv = ["maxima", "--very-quiet", "--disable-readline"]
    if os.name == "nt":
        resolved = shutil.which(argv[0])
        if resolved and resolved.lower().endswith((".bat", ".cmd")):
            return [os.environ.get("ComSpec", "cmd.exe"), "/c", resolved, *argv[1:]]
    return argv


def terminate_all_live() -> int:
    """Terminate every tracked Maxima subprocess that is still running and
    forget the dead ones.  Returns the number terminated.  GC-independent;
    used by the test conftest at module boundaries to bound the
    concurrent-process peak (mirrors julia_adapter.terminate_all_live)."""
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


@dataclass
class MaximaResult:
    ok: bool
    value: str          # canonical Maxima text
    error: str | None = None
    latex: str = ""     # LaTeX via tex1(); empty string if unavailable


class MaximaAdapter:
    """Persistent Maxima process adapter.

    Keeps a single Maxima subprocess alive across multiple execute() calls so
    that variable assignments and assumptions are preserved between commands.
    Use as a context manager or call start() / close() explicitly.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None
        # Set by _read_until_end when Maxima asked a question instead of
        # answering; consumed (and cleared) by execute(), which interrupts
        # the computation and reports it.  See _QUESTION_RE.
        self._pending_question: str | None = None
        # 素の状態にある名前（start() が埋める）。スナップショットはここからの
        # 差分だけを保存する。
        self._baseline_functions: set[str] = set()
        self._baseline_values: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Idempotent — see JuliaAdapter.start() for the measurement and the
        # reasoning.  A second start() on a live adapter would overwrite
        # _proc and orphan the first Maxima process; close() only discards the
        # CURRENT one, so the orphan stays in _LIVE_PROCS forever.  A restart
        # after close() still works: close() sets _proc to None.
        if self._proc is not None and self._proc.poll() is None:
            return

        self._proc = subprocess.Popen(
            _maxima_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge so errors appear in the stream
            text=True,
            bufsize=1,
        )
        # Suppress 2D layout and confirm the process is ready.
        #
        # **`load(nchrpl)` —— `mattrace` はこのビルドでは素で定義されていない。**
        # 定義されていない関数を Maxima は **noun のまま返し、`ok=True` を返す**
        # ので、トレースは一度も計算されないまま `completed` として公開されて
        # いた（実測 2026-09-06: `mattrace(matrix([1,2],[3,4]))` がそのまま返る。
        # ロード後は `5`、記号行列なら `d+a`)。`\it arcsin` と同じ形である。
        #
        # **テンプレートではなくここに置く。** `\text{tr}(M)` と書かれた式は
        # `mattrace` を**任意の式の中に**持ち込むので、`simplify` でも `substitute`
        # でも定義されている必要がある。トレース操作のテンプレートに閉じ込めると、
        # 研究者が記法で書いた側が直らない。所要 14ms（実測）。
        self._send(
            f'display2d:false$\n'
            f'load(nchrpl)$\n'
            f'print("{RESULT_BEGIN}")$\n'
            f'print("{RESULT_END}")$\n'
        )
        self._read_until_end()
        # **スナップショットは「素の状態からの差分」である。**  初期化で
        # `load(nchrpl)` を読むと `functions` に `mattrace` / `ncharpoly` が入り、
        # `save_snapshot` がその**定義本体**を書き出して、行ごとに実行する
        # `load_snapshot` が復元できなくなる（実測 2026-09-06、スナップショット
        # 系 8件が落ちた）。名前が `_` で始まるかどうかで足りていたのは、
        # 素の Maxima に利用者の関数が1つも無かったからにすぎない。
        # ここで基準を取っておき、差分だけを保存する。
        self._baseline_functions = self._current_names("functions")
        self._baseline_values = self._current_names("values")
        # Safety net: if this adapter is dropped without close() (a fixture that
        # yields an engine but never tears it down), GC of the adapter still
        # terminates the Maxima process instead of leaking it (mirrors
        # JuliaAdapter.start()).
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
                self._proc.stdin.write("quit();\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    def __enter__(self) -> MaximaAdapter:
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

    def execute_script(self, command: str) -> MaximaResult:
        """Execute a multi-LINE command, one statement per line.

        ``execute`` wraps a single statement in errcatch, so a command that
        Maxima needs as several statements (``define_coordinate_system`` emits
        one ``nb_*`` function per line) has to be fed line by line.  The run
        succeeds only if every line does; the first failure stops it and is
        reported.  Lives here because "one statement per call" is this adapter's
        wire protocol, and because ``MaximaResult`` is defined here.
        """
        ok, err = True, ""
        for line in (l.strip() for l in command.strip().splitlines()):
            if not line:
                continue
            r = self.execute(line)
            if not r.ok:
                ok, err = False, r.error or ""
                break
        return MaximaResult(ok=ok, value="true" if ok else "",
                            error=None if ok else err)

    def execute(self, command: str) -> MaximaResult:
        """Execute one Maxima command and return the result.

        Wraps the command in errcatch so that Maxima errors are captured
        without hanging: the markers are always printed regardless of whether
        the command succeeds or fails.
        """
        if not self.is_running:
            raise RuntimeError("MaximaAdapter is not running; call start() first.")

        # Strip trailing Maxima terminators ($, ;) so the template wrapping
        # does not produce a syntax error when callers include them.
        command = command.rstrip().rstrip("$;").rstrip()

        if ':=' in command:
            # Function-definition commands: avoid _res_:(f(x):=...) which has
            # operator-precedence issues.  Just run errcatch(command) directly;
            # return "defined" on success (the :=  side-effect is all we need).
            self._send(
                f'{_GUARD_RESET}'
                f'_errcatch_res_: errcatch({command})$\n'
                f'print("{RESULT_BEGIN}")$\n'
                f'if not emptyp(_errcatch_res_) then print("defined") else print("{ERROR_SENTINEL}")$\n'
                f'print("{TEX_SEP}")$\n'
                f'print("")$\n'
                f'print("{RESULT_END}")$\n'
            )
        elif _NO_TEX_CMD_RE.match(command.strip()):
            # Side-effect commands (assume, forget, depends, remove):
            # their results are lists like [x > 0] that do not need LaTeX
            # rendering and can cause tex1() to trigger assumption-system
            # re-evaluation, leading to stdout noise or infinite loops.
            self._send(
                f'{_GUARD_RESET}'
                f'_errcatch_res_: errcatch(_res_:({command}))$\n'
                f'print("{RESULT_BEGIN}")$\n'
                f'if not emptyp(_errcatch_res_) then print(_res_) else print("{ERROR_SENTINEL}")$\n'
                f'print("{TEX_SEP}")$\n'
                f'print("")$\n'
                f'print("{RESULT_END}")$\n'
            )
        else:
            # _errcatch_res_ is [] on error, [result] on success.
            # _res_ holds the actual expression result (may or may not be a list).
            # _tex_errcatch_ / _tex_res_ capture the tex1() representation.
            self._send(
                f'{_GUARD_RESET}'
                f'_errcatch_res_: errcatch(_res_:({command}))$\n'
                f'if not emptyp(_errcatch_res_) '
                f'then _tex_errcatch_: errcatch(_tex_res_: tex1(_res_)) '
                f'else _tex_errcatch_: []$\n'
                f'print("{RESULT_BEGIN}")$\n'
                f'if not emptyp(_errcatch_res_) then print(_res_) else print("{ERROR_SENTINEL}")$\n'
                f'print("{TEX_SEP}")$\n'
                f'if not emptyp(_tex_errcatch_) then print(_tex_res_) else print("")$\n'
                f'print("{RESULT_END}")$\n'
            )

        raw = self._read_until_end()

        if self._pending_question is not None:
            question = self._pending_question
            self._pending_question = None
            self._interrupt_and_resync()
            return MaximaResult(ok=False, value="",
                                error=f"{QUESTION_MARKER} {question}")

        if TEX_SEP in raw:
            main_part, tex_part = raw.split(TEX_SEP, 1)
        else:
            main_part, tex_part = raw, ""

        if ERROR_SENTINEL in main_part:
            msg = main_part.replace(ERROR_SENTINEL, "").strip()
            return MaximaResult(ok=False, value="", error=msg or "Maxima evaluation error")

        return MaximaResult(ok=True, value=main_part.strip(), latex=tex_part.strip())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(text)
        self._proc.stdin.flush()

    def _interrupt_and_resync(self) -> None:
        """Abandon a computation stuck on a question and leave the session usable.

        SIGINT is the only exit that does not answer for the researcher:
        measured 2026-08-01, Maxima drops the ask loop ("Interactive interrupt
        … Automatically continuing."), stays alive, and accepts commands again.
        The leftovers of that report are then drained by round-tripping a
        sentinel, so the NEXT execute() does not read this command's debris as
        its own result — which is precisely how one unanswered question used to
        take the whole session with it.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.send_signal(signal.SIGINT)
            self._send(f'print("{SYNC_SENTINEL}")$\n')
        except (ProcessLookupError, BrokenPipeError, OSError):
            return

        # Drain until our own echo comes back.  Read in a daemon thread and
        # wait with a deadline for the same reason _read_until_end does: a
        # bare readline on a process that has stopped talking never returns,
        # and a recovery path that can hang is not a recovery path.
        drained = threading.Event()

        def _drain() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            while True:
                line = self._proc.stdout.readline()
                if not line or SYNC_SENTINEL in line:
                    drained.set()
                    return

        threading.Thread(target=_drain, daemon=True).start()
        drained.wait(timeout=min(self.timeout, 10.0))

    # ------------------------------------------------------------------
    # Snapshot save / load
    # ------------------------------------------------------------------

    def _current_names(self, query: str) -> "set[str]":
        """`functions` / `values` の名前（関数は引数を落とした素の名前）。"""
        try:
            raw = self.execute(query).value
        except Exception:
            return set()
        return {item.split("(")[0].strip() for item in _parse_maxima_list(raw)}

    def save_snapshot(self, path: str | Path, rendered_cmds: list[str] | None = None) -> bool:
        """Write a Maxima command script that reconstructs the current CAS state.

        *rendered_cmds* is accepted for interface compatibility with the Julia
        adapter but is ignored here; Maxima derives its snapshot by querying
        the running process directly.

        Captures: user function definitions, user variable assignments,
        assume() facts, and depends() declarations.  Internal adapter
        variables (names starting with '_') are excluded.

        Returns True on success.  Does not raise; returns False on error.
        """
        if not self.is_running:
            return False
        try:
            lines: list[str] = []

            # 1. Function definitions
            funcs_raw = self.execute("functions").value
            for sig in _parse_maxima_list(funcs_raw):
                basename = sig.split("(")[0].strip()
                if basename.startswith("_") or basename in self._baseline_functions:
                    continue
                fdef = self.execute(f"fundef({basename})")
                if fdef.ok and fdef.value:
                    lines.append(fdef.value.rstrip(";$") + "$")

            # 2. User variable assignments
            vals_raw = self.execute("values").value
            for var in _parse_maxima_list(vals_raw):
                if var.startswith("_"):
                    continue
                val = self.execute(var)
                if val.ok and val.value:
                    lines.append(f"{var}: {val.value}$")

            # 3. Assume facts
            facts_raw = self.execute("facts()").value
            for fact in _parse_maxima_list(facts_raw):
                if fact:
                    lines.append(f"assume({fact})$")

            # 4. Depends declarations
            deps_raw = self.execute("dependencies").value
            for dep in _parse_maxima_list(deps_raw):
                m = re.match(r'^(\w+)\((.+)\)$', dep.strip())
                if m:
                    name, params = m.group(1), m.group(2)
                    lines.append(f"depends([{name}], [{params}])$")

            Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
            return True
        except Exception:
            return False

    def get_active_dependencies(self) -> list[tuple[str, str]]:
        """Return all active (dependent_var, independent_var) pairs from CAS state.

        Returns a CAS-agnostic list so callers need not know Maxima syntax.
        """
        if not self.is_running:
            return []
        r = self.execute("dependencies")
        if not r.ok:
            return []
        items = _parse_maxima_list(r.value)
        result: list[tuple[str, str]] = []
        for item in items:
            m = re.match(r'^([A-Za-z_]\w*)\((.+)\)$', item.strip())
            if m:
                var = m.group(1)
                for dep in _parse_maxima_list(m.group(2)):
                    dep = dep.strip()
                    if dep:
                        result.append((var, dep))
        return result

    def remove_dependency(self, var: str) -> bool:
        """Remove all dependency declarations for var from CAS state."""
        r = self.execute(f"remove({var}, dependency)")
        return r.ok

    def load_snapshot(self, path: str | Path) -> bool:
        """Restore CAS state from a snapshot file produced by save_snapshot().

        Executes each line individually so errors are caught per-command.
        Returns True only if all commands succeed.
        """
        if not self.is_running:
            return False
        p = Path(path)
        if not p.exists():
            return False
        try:
            for raw_line in p.read_text(encoding='utf-8').splitlines():
                line = raw_line.strip().rstrip('$;').strip()
                if not line or line.startswith(';'):
                    continue
                r = self.execute(line)
                if not r.ok:
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_until_end(self) -> str:
        """Read lines from stdout until RESULT_END marker.

        Runs readline in a daemon thread so the main thread can enforce a
        timeout without leaving a blocked read permanently occupying the
        process handle. Calls to this method are always sequential (one per
        execute()), so concurrent reads never occur.
        """
        lines: list[str] = []
        in_result = False
        finished = threading.Event()

        def _reader() -> None:
            nonlocal in_result
            assert self._proc is not None and self._proc.stdout is not None
            while True:
                line = self._proc.stdout.readline()
                if not line:        # EOF — process died
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
                elif _QUESTION_RE.match(line):
                    # Maxima is asking, not computing — it will now block on
                    # stdin forever.  Stop reading immediately (waiting out the
                    # timeout changes nothing) and let execute() interrupt and
                    # report.  Only the PREAMBLE is scanned: everything the
                    # command itself prints arrives after RESULT_BEGIN, so a
                    # result that merely looks like a question is never
                    # mistaken for one.
                    self._pending_question = line.strip()
                    finished.set()
                    return

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        if not finished.wait(timeout=self.timeout):
            raise TimeoutError(f"Maxima did not respond within {self.timeout}s")

        return "\n".join(lines).strip()


def _parse_maxima_list(s: str) -> list[str]:
    """Split a Maxima list string like '[a,b(c,d),e>0]' into ['a','b(c,d)','e>0'].

    Respects parenthesis nesting so commas inside function arguments are not
    treated as item separators.  Returns [] for empty or missing input.
    """
    s = s.strip()
    if not s or s in ('[]', ''):
        return []
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    items: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in '([':
            depth += 1
            buf.append(ch)
        elif ch in ')]':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            item = ''.join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)
    item = ''.join(buf).strip()
    if item:
        items.append(item)
    return items
