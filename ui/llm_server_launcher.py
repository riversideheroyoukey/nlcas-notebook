"""Config-driven local LLM inference-server lifecycle.

The app can auto-start a local OpenAI-compatible inference server (Ollama by
default; vLLM / LocalAI / llama.cpp server / etc. by editing
``config/default_settings.json`` → ``local_llm_server``).  The model *client*
side is already provider-agnostic (``models/*/endpoint.json`` ``base_url`` +
``endpoint_type``), so switching servers is a config change, not a code change.

Config (``config/default_settings.json`` → ``local_llm_server``):
  ``auto_start``           bool   start the server only if its port isn't up
  ``command``              list   argv to launch, e.g. ``["ollama", "serve"]``
  ``port``                 int    port to probe for readiness / already-running
  ``env``                  dict   extra environment variables for the process
  ``startup_wait_seconds`` int    grace period after launch before serving

Design notes:
- stdout/stderr are sent to DEVNULL so verbose backend logs (e.g. llama.cpp
  model-load output) never flood the app console.
- ``env`` is applied with ``setdefault`` so an explicit user environment wins.
- If the port is already open the server is left untouched (we never kill or
  restart a server we didn't start).
- The process is launched in its own session so shutdown can target only the
  process *we* started.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def start(config: dict | None) -> "subprocess.Popen | None":
    """Start the configured local LLM server if needed.

    Returns the ``Popen`` handle for the process we started (pass it to
    :func:`stop`), or ``None`` when auto_start is disabled, no command is
    configured, or the server is already running.
    """
    if not config or not config.get("auto_start", False):
        return None
    command = config.get("command")
    if not command:
        return None
    port = int(config.get("port", 0) or 0)
    if port and _port_open(port):
        return None  # already running — leave it (and its logging) alone

    env = os.environ.copy()
    for key, value in (config.get("env") or {}).items():
        env.setdefault(str(key), str(value))

    try:
        proc = subprocess.Popen(
            list(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,   # own process group → clean, scoped shutdown
                                       # (POSIX; a documented no-op on Windows, see stop())
            env=env,
        )
    except OSError as e:
        # command[0] not found (or not executable) - this must not crash the
        # app's startup lifespan. LLM-assisted features are unavailable until
        # the tool is installed/on PATH; everything else keeps working.
        print(f"[llm_server_launcher] Could not start local LLM server "
              f"({command[0]}): {e}. LLM-assisted features will be unavailable "
              f"until it is installed and on PATH.")
        return None
    time.sleep(int(config.get("startup_wait_seconds", 3)))
    return proc


def stop(proc: "subprocess.Popen | None") -> None:
    """Terminate a server we started (only ours, via its process group)."""
    if proc is None:
        return
    if os.name == "nt":
        # No process-group signal equivalent to POSIX killpg here, and
        # start_new_session above is a no-op on Windows (CPython's Windows
        # _execute_child accepts and ignores it) - just terminate the
        # process itself. os.killpg/os.getpgid don't exist in the Windows
        # build of the os module at all.
        try:
            proc.terminate()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
