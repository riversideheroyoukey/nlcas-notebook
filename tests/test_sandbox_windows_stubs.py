"""Windows-specific safety-net tests for notebook/numeric/sandbox.py (G-4).

These don't require a Windows sandbox implementation to exist yet - they
pin the two prerequisite fixes: the module must import cleanly on Windows
(pwd/resource are POSIX-only and must not be imported unconditionally), and
run_file() must not reach the POSIX-only preexec_fn path there.
"""
import sys

from notebook.numeric import sandbox


class TestWindowsImportSafety:
    def test_module_has_no_unconditional_posix_only_names(self):
        """pwd/resource must only be bound when not on win32 - reproduces
        the crash by construction rather than needing an actual Windows
        interpreter: on THIS platform they're bound (posix), so the real
        regression test is that the import above (module-level, at the top
        of this file) didn't raise on Windows in the first place (verified
        on real Windows hardware, docs/open_items.md G-4)."""
        if sys.platform == "win32":
            assert not hasattr(sandbox, "pwd")
            assert not hasattr(sandbox, "resource")
        else:
            assert hasattr(sandbox, "pwd")
            assert hasattr(sandbox, "resource")


class TestRunFileDispatch:
    def test_windows_dispatches_to_windows_method_not_posix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox.sys, "platform", "win32")
        runner = sandbox.SandboxRunner(python_path="python")
        called = {}

        def _fake_windows(code_path, workdir):
            called["windows"] = True
            return "RESULT"

        def _fake_posix(code_path, workdir):
            called["posix"] = True
            return "RESULT"

        monkeypatch.setattr(runner, "_run_file_windows", _fake_windows)
        monkeypatch.setattr(runner, "_run_file_posix", _fake_posix)
        code = tmp_path / "code.py"
        code.write_text("print(1)")
        result = runner.run_file(code, tmp_path / "work")
        assert called == {"windows": True}
        assert result == "RESULT"

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestWindowsRunFile:
    """_run_file_windows (G-4), exercised via mocked subprocess.run - none
    of these actually invoke schtasks/icacls/taskkill (macOS/Linux CI can't),
    they pin the CONTRACT verified by hand against the real Windows VM
    (docs/open_items.md G-4): wrapper script shape, icacls grants, schtasks
    argv, and the kill-by-account timeout sweep."""

    def _runner(self, tmp_path, timeout_s=60.0):
        runner = sandbox.SandboxRunner(python_path=r"C:\venv\Scripts\python.exe",
                                       timeout_s=timeout_s)
        workdir = tmp_path / "work"
        (workdir / ".mpl").mkdir(parents=True)
        (workdir / ".tmp").mkdir()
        code_path = workdir / "code.py"
        code_path.write_text("print(1)")
        return runner, workdir, code_path

    def test_wrapper_script_uses_configured_python_and_code_path(self, tmp_path, monkeypatch):
        runner, workdir, code_path = self._runner(tmp_path)
        monkeypatch.setattr(sandbox, "_windows_decrypt_password", lambda p: "pw")
        calls = []

        def _fake_run(argv, **kw):
            calls.append(argv)
            if argv[0] == "schtasks" and argv[1] == "/create":
                # Simulate the task having already produced its output by
                # the time /run is issued next.
                (workdir / "_stdout.txt").write_text("hello\n")
                (workdir / "_stderr.txt").write_text("")
                (workdir / "_exitcode.txt").write_text("0")
                (workdir / "_done.marker").touch()
            return _FakeCompleted(0)

        monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)
        result = runner._run_file_windows(code_path, workdir)

        assert result.ok
        assert result.stdout == "hello\n"
        wrapper_text = (workdir / "_wrapper.ps1").read_text()
        assert r'"C:\venv\Scripts\python.exe"' in wrapper_text
        assert str(code_path) in wrapper_text
        assert 'MPLBACKEND = "Agg"' in wrapper_text

    def test_icacls_grants_account_system_and_caller(self, tmp_path, monkeypatch):
        runner, workdir, code_path = self._runner(tmp_path)
        monkeypatch.setattr(sandbox, "_windows_decrypt_password", lambda p: "pw")
        monkeypatch.setenv("USERNAME", "kawabe")
        calls = []

        def _fake_run(argv, **kw):
            calls.append(argv)
            if argv[0] == "schtasks" and argv[1] == "/create":
                (workdir / "_exitcode.txt").write_text("0")
                (workdir / "_done.marker").touch()
            return _FakeCompleted(0)

        monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)
        runner._run_file_windows(code_path, workdir)

        icacls_grants = [c for c in calls if c[0] == "icacls" and "/grant" in c]
        granted = {c[-1] for c in icacls_grants}
        assert any(g.startswith("numeric:") for g in granted)
        assert any(g.startswith("SYSTEM:") for g in granted)
        assert any(g.startswith("kawabe:") for g in granted)
        # caller gets read+execute only, not full control - must not be able
        # to tamper with results while the sandboxed process is still running
        caller_grant = next(g for g in granted if g.startswith("kawabe:"))
        assert "(RX)" in caller_grant and "F" not in caller_grant.split(":", 1)[1].replace("(RX)", "")

    def test_schtasks_create_failure_returns_error_result_not_hang(self, tmp_path, monkeypatch):
        runner, workdir, code_path = self._runner(tmp_path)
        monkeypatch.setattr(sandbox, "_windows_decrypt_password", lambda p: "pw")

        def _fake_run(argv, **kw):
            if argv[0] == "schtasks" and argv[1] == "/create":
                return _FakeCompleted(1, stderr="ERROR: access denied")
            return _FakeCompleted(0)

        monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)
        result = runner._run_file_windows(code_path, workdir)
        assert not result.ok
        assert "access denied" in result.stderr

    def test_timeout_sweeps_taskkill_by_username_not_pid(self, tmp_path, monkeypatch):
        """done.marker never appears - must fall into the grace-window sweep
        and kill by USERNAME (not PID/schtasks-end, both proven on real
        hardware to miss the venv-redirector-stub's grandchild process)."""
        runner, workdir, code_path = self._runner(tmp_path, timeout_s=1.0)
        monkeypatch.setattr(sandbox, "_windows_decrypt_password", lambda p: "pw")
        monkeypatch.setattr(sandbox.time, "sleep", lambda s: None)

        clock = {"t": 0.0}

        def _fake_monotonic():
            clock["t"] += 5.0
            return clock["t"]

        monkeypatch.setattr(sandbox.time, "monotonic", _fake_monotonic)

        taskkill_calls = []

        def _fake_run(argv, **kw):
            if argv[0] == "taskkill":
                taskkill_calls.append(argv)
            return _FakeCompleted(0)

        monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)
        result = runner._run_file_windows(code_path, workdir)

        assert result.timed_out
        assert taskkill_calls, "expected at least one taskkill sweep"
        for argv in taskkill_calls:
            assert argv[:3] == ["taskkill", "/F", "/T"]
            assert argv[-1] == "USERNAME eq numeric"
