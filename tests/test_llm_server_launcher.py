"""Unit tests for ui/llm_server_launcher.py's config-driven start/stop.

start() with a nonexistent command reproduces the exact FileNotFoundError
that Ollama-not-installed triggers, without needing Ollama (or any real
LLM server) installed anywhere - this is how the Windows-only crash
(docs/open_items.md G-3) was verified fixed without touching the real
Windows VM's installed toolset.
"""
import os
import signal
import socket
import subprocess

import pytest

from ui import llm_server_launcher as launcher


class TestStart:
    def test_returns_none_when_auto_start_disabled(self):
        assert launcher.start({"auto_start": False, "command": ["ollama", "serve"]}) is None

    def test_returns_none_when_no_config(self):
        assert launcher.start(None) is None

    def test_returns_none_when_no_command(self):
        assert launcher.start({"auto_start": True}) is None

    def test_returns_none_when_port_already_open(self, monkeypatch):
        monkeypatch.setattr(launcher, "_port_open", lambda port, **kw: True)
        assert launcher.start(
            {"auto_start": True, "command": ["ollama", "serve"], "port": 11434}
        ) is None

    def test_missing_binary_does_not_raise(self, monkeypatch):
        """The bug: Popen(FileNotFoundError) used to propagate uncaught and
        crash the FastAPI lifespan. Reproduces on every OS - no real LLM
        server binary needed."""
        monkeypatch.setattr(launcher, "_port_open", lambda port, **kw: False)
        result = launcher.start({
            "auto_start": True,
            "command": ["definitely-not-a-real-binary-xyz"],
            "port": 0,
            "startup_wait_seconds": 0,
        })
        assert result is None

    def test_started_process_env_includes_config_overrides(self, monkeypatch):
        captured = {}

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            class _FakeProc:
                pid = 12345
            return _FakeProc()

        monkeypatch.setattr(launcher, "_port_open", lambda port, **kw: False)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        launcher.start({
            "auto_start": True,
            "command": ["ollama", "serve"],
            "port": 0,
            "startup_wait_seconds": 0,
            "env": {"OLLAMA_HOST": "127.0.0.1:11434"},
        })
        assert captured["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"


class TestStop:
    def test_none_is_a_noop(self):
        launcher.stop(None)  # must not raise

    def test_posix_uses_killpg(self, monkeypatch):
        if os.name == "nt":
            pytest.skip("POSIX-only code path")
        calls = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

        class _FakeProc:
            pid = 999
        launcher.stop(_FakeProc())
        assert calls == [(999, signal.SIGTERM)]

    def test_windows_uses_terminate_not_killpg(self, monkeypatch):
        """os.killpg/os.getpgid don't exist at all in the Windows build of
        the os module (confirmed on real Windows hardware) - stop() must
        not reference them when os.name == 'nt'."""
        monkeypatch.setattr(os, "name", "nt")
        terminated = []

        class _FakeProc:
            pid = 999
            def terminate(self):
                terminated.append(self.pid)

        launcher.stop(_FakeProc())
        assert terminated == [999]
