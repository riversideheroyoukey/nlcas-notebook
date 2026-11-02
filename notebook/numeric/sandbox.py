"""Sandboxed execution of generated numeric code (N-1, policy B).

Empirically verified on this machine (2026-07-04, see open_items N-1):
macOS ``sandbox-exec`` with a default-deny profile runs numpy/matplotlib
code while

  - denying ALL network access (DNS fails; inherited by child processes),
  - allowing writes ONLY under the per-execution work directory,
  - denying reads of secret locations (deny rules proven effective),
  - a host-side timeout kills runaway code.

Linux has TWO fallback mechanisms, tried in order (open_items G-2):

  1. ``bubblewrap`` (bwrap): namespace-based, reproduces all 3 guarantees
     exactly like macOS. Verified working on bare-metal Linux (2026-07-20,
     Ubuntu 22.04). Does NOT work inside the Docker image built from this
     repo's own Dockerfile — bwrap's internal loopback-configuration step
     fails there (``Failed RTM_NEWADDR: No child processes``, root cause
     undetermined after ruling out bwrap version, SIGCHLD handling,
     namespace-flag composition, and Docker's seccomp profile).
  2. Dedicated low-privilege user + Landlock (network-only) + file
     permissions: used when bwrap's probe fails (i.e. inside this repo's
     Docker image). Landlock (kernel 5.13+, network hooks need ABI v4 /
     kernel 6.7+) denies outbound TCP bind/connect for the CURRENT process
     alone, needs NO extra capability (unprivileged by design — unlike
     bwrap, it never touches namespaces).

     HONEST CHARACTERIZATION (2026-07-21): this fallback is NOT a real
     sandbox in the macOS/bwrap sense — it is weaker isolation, deliberately
     accepted as "better than nothing" rather than engineered to be airtight:
       - Network: TCP bind/connect only. UDP, raw sockets, AF_UNIX are
         untouched (Landlock's network hooks don't cover them).
       - Write: confined to the task's own workdir via chown+chmod(0700) —
         this part IS solid (verified: EACCES/EROFS on any other path).
       - Read: `sessions/` itself is locked to o+x-only (not listable), but
         `_lock_down_task_dir` only chmods the ANCESTORS of the CURRENTLY
         RUNNING task — sibling task/session directories keep whatever
         permissions they already had. If the exact path is known (or
         guessed), it IS readable: `task_id` is a small sequential counter
         (`N001`, `N002`, …) and `execution_id` is a plain timestamp
         (`notebook/models/numeric.py` `_new_execution_id`), NOT a random
         UUID, so enumeration is cheap. Concurrent numeric tasks also share
         this one system UID, so even a "locked-down" sibling dir (owned by
         the same `numeric` user) is readable by definition, not just by
         path-guessing.
     This is accepted, not fixed, because the two real code paths reaching
     this sandbox are (a) deterministic templates, whose function whitelist
     (``notebook/numeric/expr_to_numpy.py`` ``_NUMPY_FUNCS``) cannot express
     file-path or network I/O at all — the code literally cannot be told to
     read another path even if one were guessed — and (b) LLM-generated
     code, which cannot run before a human approves it
     (``approve_numeric_code``). The isolation gaps above are a backstop for
     reviewer oversight, not the sole defense, so cross-task/cross-session
     leakage within one researcher's own sessions was judged an acceptable
     residual — unlike exfiltration to the outside network (still denied)
     or writes outside the task's own output area (still denied), which are
     solid guarantees.

Availability is checked EMPIRICALLY for each mechanism (an actual probe
invocation), the same philosophy as the macOS side's own "empirically
verified" note above — nested unprivileged namespaces and Landlock's kernel
ABI level both vary by host in ways worth checking directly rather than
assumed from `sys.platform` alone. No sandbox implementation exists yet for
native (non-WSL2) Windows.

Generated code is treated as ADVERSARIAL input (it usually is not — it is
our own deterministic template — but the profile does not rely on that).
Scope: small PoC computations only; heavy computing is explicitly out of
N-1's responsibility (open_items スコープ境界).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# pwd/resource are POSIX-only stdlib modules (absent from the Windows build
# of the interpreter entirely - importing them unconditionally crashes the
# module on import there). Only referenced from the Linux
# uid_landlock/bwrap code paths below, which are never reached on Windows.
if sys.platform != "win32":
    import pwd
    import resource

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Interpreter for numeric runs: a dedicated venv so the app venv stays lean
# and package_info is recorded per execution.  Override via NB_NUMERIC_PYTHON.
# venv layout differs by platform (Scripts\ + .exe on Windows, bin/ elsewhere).
# Currently unreachable on win32 in practice — sandbox_available() short-
# circuits to False there before this path is ever checked, since no Windows
# sandbox mechanism exists yet (see open_items G-3) — but kept correct so it
# doesn't silently mislead if that gating logic ever changes.
DEFAULT_NUMERIC_PYTHON = (
    _PROJECT_ROOT / ".venv-numeric" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else _PROJECT_ROOT / ".venv-numeric" / "bin" / "python"
)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Locations the sandboxed process must never read even though imports need
# broad read access.  Deny rules come AFTER the allow, so they win (macOS
# SBPL: last-match-wins; Linux bwrap: later binds shadow earlier ones at the
# same/nested path — same idea, different mechanism).
_DENY_READS = (
    "~/.ssh", "~/.aws", "~/.gnupg", "~/.config/gh",
    str(_PROJECT_ROOT / "sessions"),
)


# ---------------------------------------------------------------------------
# macOS: sandbox-exec (SBPL profile)
# ---------------------------------------------------------------------------

_PROFILE_TEMPLATE = """(version 1)
(deny default)
(allow process-exec*)
(allow process-fork)
(allow sysctl-read)
(allow file-read*)
{deny_reads}
(allow file-read*  (subpath "{write_root}"))
(allow file-write* (subpath "{write_root}"))
(deny network*)
"""


def build_profile(write_root: str) -> str:
    # SBPL is last-match-wins.  The task's own work dir lives UNDER sessions/
    # (session-anchored numeric output), so the sessions/ deny-read below would
    # otherwise also block the sandbox from reading its OWN code.py.  Re-allow
    # reads of just this work dir AFTER the deny — the secret-protection deny
    # still covers the rest of sessions/ (other sessions' research data), only
    # the current task's exec dir becomes readable again.
    deny = "\n".join(
        f'(deny file-read* (subpath "{os.path.expanduser(p)}"))'
        for p in _DENY_READS
    )
    return _PROFILE_TEMPLATE.format(deny_reads=deny, write_root=write_root)


# ---------------------------------------------------------------------------
# Linux: bubblewrap (bwrap)
# ---------------------------------------------------------------------------

_BWRAP_PATH: "str | None | bool" = False  # False = not yet resolved
_LINUX_PROBE_RESULT: "bool | None" = None


def _resolve_bwrap() -> "str | None":
    global _BWRAP_PATH
    if _BWRAP_PATH is False:
        _BWRAP_PATH = shutil.which("bwrap")
    return _BWRAP_PATH


def _bwrap_deny_targets() -> list[str]:
    """Existing paths from _DENY_READS, filtered to what's actually present —
    bwrap's --tmpfs needs the destination to already exist in the bound tree,
    unlike SBPL's deny rule which is a harmless no-op for absent paths."""
    return [p for p in (os.path.expanduser(d) for d in _DENY_READS)
            if os.path.exists(p)]


def _build_bwrap_argv(python_path: str, code_path: "str | Path",
                       workdir: Path) -> list[str]:
    """bwrap invocation reproducing the same 3 guarantees as build_profile()
    (macOS): network deny, write confined to *workdir*, secret dirs / other
    sessions unreadable.  Argv order matters — each bind/tmpfs shadows an
    earlier one at the same or nested path, so: broad read (--ro-bind / /),
    then shadow secrets+sessions/ with empty tmpfs (deny), then re-bind just
    this task's own workdir read-write (re-allow) — mirrors the SBPL
    template's allow → deny → re-allow structure exactly."""
    bwrap = _resolve_bwrap()
    if not bwrap:
        raise RuntimeError("bwrap not found on PATH")
    argv = [
        bwrap,
        "--die-with-parent", "--new-session",
        "--unshare-net", "--unshare-pid",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]
    for target in _bwrap_deny_targets():
        argv += ["--tmpfs", target]
    argv += ["--bind", str(workdir), str(workdir)]
    argv += ["--chdir", str(workdir)]
    argv += ["--", python_path, str(code_path)]
    return argv


def _linux_bwrap_probe() -> bool:
    """Actually try to create the namespaces bwrap needs, rather than
    assuming it works from `sys.platform` alone — unprivileged user
    namespaces are commonly restricted (e.g. Ubuntu 24.04's
    kernel.apparmor_restrict_unprivileged_userns), and whether that
    restriction applies inside a given Docker host/image varies. Cached for
    the process lifetime (the environment doesn't change mid-process)."""
    global _LINUX_PROBE_RESULT
    if _LINUX_PROBE_RESULT is not None:
        return _LINUX_PROBE_RESULT
    bwrap = _resolve_bwrap()
    if not bwrap:
        _LINUX_PROBE_RESULT = False
        return False
    try:
        r = subprocess.run(
            [bwrap, "--unshare-net", "--unshare-pid",
             "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
             "--tmpfs", "/tmp", "--", "true"],
            capture_output=True, timeout=10,
        )
        _LINUX_PROBE_RESULT = (r.returncode == 0)
    except Exception:
        _LINUX_PROBE_RESULT = False
    return _LINUX_PROBE_RESULT


# ---------------------------------------------------------------------------
# Linux fallback: dedicated low-privilege user + Landlock (network only)
# ---------------------------------------------------------------------------

_NUMERIC_SYSTEM_USER = "numeric"

# x86_64 syscall numbers (landlock_add_rule unused: zero rules = full TCP deny)
_NR_LANDLOCK_CREATE_RULESET = 444
_NR_LANDLOCK_RESTRICT_SELF = 446
_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
_LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64),
                ("handled_access_net", ctypes.c_uint64)]


def _landlock_restrict_network_self() -> None:
    """Deny outbound TCP bind/connect for the CALLING process, irrevocably
    (inherited across exec). handled_access_fs=0: this rule set touches
    network only — file confinement here is handled separately via a
    dedicated UID + chmod (_lock_down_task_dir), not Landlock. Raises
    OSError if the running kernel lacks Landlock or ABI v4 network hooks."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS) failed")
    attr = _LandlockRulesetAttr(
        handled_access_fs=0,
        handled_access_net=(_LANDLOCK_ACCESS_NET_BIND_TCP
                             | _LANDLOCK_ACCESS_NET_CONNECT_TCP),
    )
    fd = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, ctypes.byref(attr),
                       ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        if libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(fd)


def _numeric_user_ids() -> "tuple[int, int] | None":
    try:
        pw = pwd.getpwnam(_NUMERIC_SYSTEM_USER)
        return pw.pw_uid, pw.pw_gid
    except KeyError:
        return None


_LINUX_UID_LANDLOCK_PROBE_RESULT: "bool | None" = None


def _linux_uid_landlock_probe() -> bool:
    """Empirically confirm the fallback works: the dedicated user exists and
    landlock_restrict_self() actually succeeds after dropping to it. Run in a
    throwaway forked child — landlock_restrict_self is irrevocable, so it
    can't be tested in the calling (long-lived) process itself. Cached for
    the process lifetime."""
    global _LINUX_UID_LANDLOCK_PROBE_RESULT
    if _LINUX_UID_LANDLOCK_PROBE_RESULT is not None:
        return _LINUX_UID_LANDLOCK_PROBE_RESULT
    ids = _numeric_user_ids()
    if ids is None:
        _LINUX_UID_LANDLOCK_PROBE_RESULT = False
        return False
    uid, gid = ids
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            os.setgid(gid)
            os.setuid(uid)
            _landlock_restrict_network_self()
            os.write(w_fd, b"1")
        except Exception:
            os.write(w_fd, b"0")
        os._exit(0)
    os.close(w_fd)
    reply = os.read(r_fd, 1)
    os.close(r_fd)
    os.waitpid(pid, 0)
    _LINUX_UID_LANDLOCK_PROBE_RESULT = (reply == b"1")
    return _LINUX_UID_LANDLOCK_PROBE_RESULT


def _lock_down_task_dir(workdir: Path, uid: int, gid: int) -> None:
    """Ancestors under sessions/ (sessions/ itself included, since it also
    holds saved session JSON — real research data, not just numeric scratch)
    get o+x (traversable by exact path) but no o+r (not listable). The leaf
    workdir and its pre-created .mpl/.tmp subdirs are chowned to the numeric
    user and made private (0700) so it can actually write there.

    NOT a real read boundary against sibling task/session dirs — this only
    locks down the ancestors of the CURRENTLY running task. A sibling's
    directory keeps whatever permissions IT got when it ran (same recipe,
    same numeric UID), and task_id/execution_id are a small counter + a
    timestamp (not random), so an exact sibling path is cheap to construct.
    Accepted as a "better than nothing" default posture, not a security
    boundary — see the module docstring."""
    sessions_root = _PROJECT_ROOT / "sessions"
    rel = workdir.relative_to(sessions_root)
    os.chmod(sessions_root, 0o711)
    current = sessions_root
    for part in rel.parts[:-1]:
        current = current / part
        os.chmod(current, 0o711)
    for target in (workdir, workdir / ".mpl", workdir / ".tmp"):
        os.chown(target, uid, gid)
        os.chmod(target, 0o700)


# Cached resolution of which Linux mechanism is active: "bwrap",
# "uid_landlock", or "unavailable". Resolved once (probes are real subprocess
# calls / forks) and reused for the process lifetime.
_LINUX_MODE: "str | None" = None


def _resolve_linux_mode() -> str:
    global _LINUX_MODE
    if _LINUX_MODE is not None:
        return _LINUX_MODE
    if _linux_bwrap_probe():
        _LINUX_MODE = "bwrap"
    elif _linux_uid_landlock_probe():
        _LINUX_MODE = "uid_landlock"
    else:
        _LINUX_MODE = "unavailable"
    return _LINUX_MODE


# ---------------------------------------------------------------------------
# Windows: dedicated low-privilege account + Task Scheduler + per-account
# firewall rule (G-4). Provisioned ONCE by setup_windows.ps1 (needs admin);
# everything here only PROBES and USES what was already set up, mirroring
# the empirical-availability philosophy above.
#
# Why Task Scheduler instead of LogonUser+CreateProcessAsUser (the original
# design): CreateProcessAsUser needs SE_ASSIGN_PRIMARYTOKEN_NAME, which is
# NOT granted to the Administrators group by default (confirmed via
# `whoami /priv` in a fully elevated session on real Windows hardware) -
# regardless of whether the app itself runs elevated, that call fails with
# ERROR_PRIVILEGE_NOT_HELD unless an operator manually grants this specific
# right via secedit. schtasks' /ru+/rp delegates the actual account switch
# to the Task Scheduler service (which already has the necessary
# privileges), so the calling process's own privilege level doesn't matter.
#
# Why kill-by-account instead of kill-by-PID or schtasks /end on timeout:
# this app's Windows venvs (both the main .venv and .venv-numeric) point at
# the SAME base Python install; venv's Scripts\python.exe is a small
# redirector stub that re-execs the base interpreter as a CHILD process at a
# different image path, so the real process tree is 3 levels deep (wrapper
# powershell.exe -> venv stub -> base python.exe). Neither
# `taskkill /PID <pid> /T /F` (on the wrapper's own PID) nor
# `schtasks /end /tn <task>` reliably reaches the grandchild - both were
# tried and left it orphaned in testing. Killing by USERNAME catches every
# descendant regardless of tree depth, since ALL sandboxed code runs
# exclusively as the dedicated account (this does mean a timeout on one
# task would kill any OTHER numeric task concurrently in flight - accepted,
# since tasks already share this one account with no mutual isolation, per
# the HONEST CHARACTERIZATION above).
# ---------------------------------------------------------------------------

_WINDOWS_NUMERIC_USER = "numeric"
_WINDOWS_PW_FILE = _PROJECT_ROOT / ".numeric_account_pw.dat"
_WINDOWS_FIREWALL_RULE = "ResearchNotebook-NumericBlock"

_WINDOWS_PROBE_RESULT: "bool | None" = None


def _windows_decrypt_password(path: Path) -> str:
    """Decrypt the DPAPI-protected password setup_windows.ps1 generated for
    the dedicated account (ConvertFrom-SecureString, CurrentUser scope).

    Empirically verified (2026-07-24) to round-trip correctly for a process
    launched interactively or via Task Scheduler - but NOT from a bare
    non-interactive `powershell -Command`/`-File` invocation over SSH, where
    even the ENCRYPT side failed with "Access is denied" (the user profile
    isn't fully loaded in that specific invocation context). This app is
    always started interactively (start_windows.ps1) or via Task Scheduler,
    never bare SSH, so this is a real-hardware-verified assumption, not a
    hopeful one.
    """
    script = (
        f"$sec = Get-Content -Path '{path}' | ConvertTo-SecureString; "
        "$ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($sec); "
        "try { [System.Runtime.InteropServices.Marshal]::PtrToStringUni($ptr) } "
        "finally { [System.Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($ptr) }"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"could not decrypt numeric account password: {r.stderr}")
    return r.stdout.rstrip("\r\n")


def _windows_numeric_probe() -> bool:
    """Empirically confirm the Windows mechanism is provisioned: the
    dedicated account exists and the per-account outbound firewall block
    rule exists (both set up once by setup_windows.ps1). Cached for the
    process lifetime."""
    global _WINDOWS_PROBE_RESULT
    if _WINDOWS_PROBE_RESULT is not None:
        return _WINDOWS_PROBE_RESULT
    if not _WINDOWS_PW_FILE.exists():
        _WINDOWS_PROBE_RESULT = False
        return False
    script = (
        f"$u = Get-LocalUser -Name '{_WINDOWS_NUMERIC_USER}' -ErrorAction SilentlyContinue; "
        f"$fw = Get-NetFirewallRule -DisplayName '{_WINDOWS_FIREWALL_RULE}' -ErrorAction SilentlyContinue; "
        "if ($u -and $fw) { 'OK' } else { 'MISSING' }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        _WINDOWS_PROBE_RESULT = (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        _WINDOWS_PROBE_RESULT = False
    return _WINDOWS_PROBE_RESULT


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    created_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class SandboxRunner:
    """Runs one Python file inside the sandbox, confined to its work dir."""

    def __init__(self, python_path: "str | None" = None, timeout_s: float = 60.0):
        self.python_path = str(
            python_path
            or os.environ.get("NB_NUMERIC_PYTHON")
            or DEFAULT_NUMERIC_PYTHON
        )
        self.timeout_s = timeout_s

    @staticmethod
    def sandbox_available() -> bool:
        if sys.platform == "darwin":
            return os.path.exists(_SANDBOX_EXEC)
        if sys.platform.startswith("linux"):
            return _resolve_linux_mode() != "unavailable"
        if sys.platform == "win32":
            return _windows_numeric_probe()
        return False

    def available(self) -> bool:
        return self.sandbox_available() and os.path.exists(self.python_path)

    def package_info(self) -> dict[str, str]:
        """Versions of the numeric packages in the runner's interpreter."""
        code = ("import json\n"
                "out = {}\n"
                "for m in ('numpy', 'matplotlib', 'scipy'):\n"
                "    try:\n"
                "        out[m] = __import__(m).__version__\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(out))\n")
        try:
            r = subprocess.run([self.python_path, "-c", code],
                               capture_output=True, text=True, timeout=30)
            import json
            return json.loads(r.stdout.strip() or "{}")
        except Exception:
            return {}

    def _build_argv(self, code_path: "str | Path", workdir: Path) -> list[str]:
        """Command-prefix mechanisms (macOS sandbox-exec, Linux bwrap). The
        uid_landlock fallback doesn't wrap argv at all — it restricts the
        forked child itself via preexec_fn — so it's handled separately in
        run_file(), not here."""
        if sys.platform == "darwin":
            profile_path = workdir / ".sandbox.sb"
            profile_path.write_text(build_profile(str(workdir)))
            return [_SANDBOX_EXEC, "-f", str(profile_path),
                    self.python_path, str(code_path)]
        if sys.platform.startswith("linux"):
            return _build_bwrap_argv(self.python_path, code_path, workdir)
        raise RuntimeError(f"no sandbox implementation for platform {sys.platform!r}")

    def run_file(self, code_path: "str | Path", workdir: "str | Path") -> SandboxResult:
        """Execute *code_path* with writes confined to *workdir*."""
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / ".mpl").mkdir(exist_ok=True)
        (workdir / ".tmp").mkdir(exist_ok=True)

        if sys.platform == "win32":
            return self._run_file_windows(code_path, workdir)
        return self._run_file_posix(code_path, workdir)

    def _run_file_posix(self, code_path: "str | Path", workdir: Path) -> SandboxResult:
        """macOS sandbox-exec / Linux bwrap / Linux uid_landlock. preexec_fn
        (resource.setrlimit, os.setuid/setgid) is POSIX-only - this method is
        never called on Windows (see run_file()), so it's safe to use here
        unconditionally."""
        linux_mode = _resolve_linux_mode() if sys.platform.startswith("linux") else None
        drop_ids = None
        if linux_mode == "uid_landlock":
            drop_ids = _numeric_user_ids()
            _lock_down_task_dir(workdir, *drop_ids)
            argv = [self.python_path, str(code_path)]
        else:
            argv = self._build_argv(code_path, workdir)

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "TMPDIR": str(workdir / ".tmp"),
            "MPLCONFIGDIR": str(workdir / ".mpl"),
            "MPLBACKEND": "Agg",
            "NB_NUMERIC_OUT": str(workdir),
        }

        def _limits() -> None:
            cpu = int(self.timeout_s) + 5
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            if drop_ids is not None:
                uid, gid = drop_ids
                os.setgid(gid)
                os.setuid(uid)
                _landlock_restrict_network_self()

        before = {p.name for p in workdir.iterdir()}
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir), env=env, capture_output=True, text=True,
                timeout=self.timeout_s, preexec_fn=_limits,
            )
            duration = time.monotonic() - start
            result = SandboxResult(proc.returncode, proc.stdout, proc.stderr,
                                   duration)
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            result = SandboxResult(
                -9,
                (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                duration, timed_out=True)

        result.created_files = sorted(
            p.name for p in workdir.iterdir()
            if p.name not in before and not p.name.startswith(".")
        )
        return result

    def _run_file_windows(self, code_path: "str | Path", workdir: Path) -> SandboxResult:
        """Windows execution: a one-shot Task Scheduler task running as the
        dedicated low-privilege account. Output is captured via redirected
        files rather than live pipes - Task Scheduler owns the launched
        process, so there is no stdout/stderr handle to inherit the way
        subprocess.run's PIPE gives on POSIX. See the module-level Windows
        section above for why this shape (schtasks, kill-by-account) was
        chosen over the originally planned LogonUser+CreateProcessAsUser."""
        import uuid

        account = _WINDOWS_NUMERIC_USER
        caller = os.environ.get("USERNAME", "")

        wrapper_path = workdir / "_wrapper.ps1"
        pid_file = workdir / "_pid.txt"
        stdout_file = workdir / "_stdout.txt"
        stderr_file = workdir / "_stderr.txt"
        exitcode_file = workdir / "_exitcode.txt"
        done_marker = workdir / "_done.marker"

        mpl_dir = workdir / ".mpl"
        tmp_dir = workdir / ".tmp"
        wrapper_path.write_text(
            f'$PID | Out-File -FilePath "{pid_file}" -Encoding ascii\n'
            f'$env:MPLBACKEND = "Agg"\n'
            f'$env:MPLCONFIGDIR = "{mpl_dir}"\n'
            f'$env:TEMP = "{tmp_dir}"\n'
            f'$env:TMP = "{tmp_dir}"\n'
            f'$env:NB_NUMERIC_OUT = "{workdir}"\n'
            f'& "{self.python_path}" "{code_path}" > "{stdout_file}" 2> "{stderr_file}"\n'
            f'$LASTEXITCODE | Out-File -FilePath "{exitcode_file}" -Encoding ascii\n'
            f'New-Item -ItemType File -Path "{done_marker}" -Force | Out-Null\n'
        )

        before = {p.name for p in workdir.iterdir()}

        # Lock down: the dedicated account needs full control (it must
        # write the results), the CALLING account gets read+execute only
        # (must read results back afterward but must not be able to tamper
        # with them while the sandboxed process is still running), SYSTEM
        # needs access for Task Scheduler's own operation. Mirrors
        # _lock_down_task_dir's chown+chmod on Linux.
        subprocess.run(["icacls", str(workdir), "/inheritance:r"], capture_output=True)
        subprocess.run(["icacls", str(workdir), "/grant", f"{account}:(OI)(CI)F"],
                        capture_output=True)
        subprocess.run(["icacls", str(workdir), "/grant", "SYSTEM:(OI)(CI)F"],
                        capture_output=True)
        if caller:
            subprocess.run(["icacls", str(workdir), "/grant", f"{caller}:(OI)(CI)(RX)"],
                            capture_output=True)

        password = _windows_decrypt_password(_WINDOWS_PW_FILE)
        task_name = f"rn_numeric_{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        timed_out = False
        try:
            cr = subprocess.run(
                ["schtasks", "/create", "/tn", task_name,
                 "/tr", f'powershell -NoProfile -ExecutionPolicy Bypass -File "{wrapper_path}"',
                 "/sc", "once", "/st", "00:00",
                 "/ru", account, "/rp", password, "/f"],
                capture_output=True, text=True, timeout=15,
            )
            if cr.returncode != 0:
                return SandboxResult(-1, "", f"schtasks /create failed: {cr.stderr}",
                                     time.monotonic() - start)
            subprocess.run(["schtasks", "/run", "/tn", task_name],
                            capture_output=True, text=True, timeout=15)

            while not done_marker.exists():
                if time.monotonic() - start > self.timeout_s:
                    timed_out = True
                    grace_deadline = time.monotonic() + 20.0
                    while (time.monotonic() < grace_deadline
                           and not done_marker.exists()):
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/FI", f"USERNAME eq {account}"],
                            capture_output=True, text=True)
                        time.sleep(1.0)
                    break
                time.sleep(0.3)

            duration = time.monotonic() - start

            def _read(p: Path) -> str:
                return p.read_text(errors="replace") if p.exists() else ""

            if timed_out:
                result = SandboxResult(-9, _read(stdout_file), _read(stderr_file),
                                       duration, timed_out=True)
            else:
                try:
                    returncode = int(_read(exitcode_file).strip() or "-1")
                except ValueError:
                    returncode = -1
                result = SandboxResult(returncode, _read(stdout_file), _read(stderr_file),
                                       duration)
        finally:
            subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"],
                            capture_output=True)

        internal_names = {wrapper_path.name, pid_file.name, stdout_file.name,
                          stderr_file.name, exitcode_file.name, done_marker.name}
        result.created_files = sorted(
            p.name for p in workdir.iterdir()
            if p.name not in before
            and p.name not in internal_names
            and not p.name.startswith(".")
        )
        return result
