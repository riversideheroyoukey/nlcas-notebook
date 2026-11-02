# Research Notebook — application image (Linux/x86_64 & arm64)
#
# SCOPE: application stack only — Python (FastAPI/uvicorn) + Maxima + Julia.
#
#   * The LLM (Ollama) is NOT in this image. Run Ollama HOST-NATIVE and let the
#     container reach it (docker-compose uses network_mode: host on Linux, so the
#     existing localhost:11434 model endpoints work unchanged). See docs/setup_guide.md.
#   * The NUMERIC LAYER (N-1) runs inside this container via a dedicated
#     low-privilege user + Landlock (network-only) rather than bubblewrap —
#     bwrap's own internal loopback-configuration step fails in this specific
#     container environment (`Failed RTM_NEWADDR: No child processes`, root
#     cause undetermined after ruling out bwrap version, SIGCHLD handling,
#     namespace-flag composition, and Docker's seccomp profile; verified on
#     real Ubuntu 22.04 + Docker 29.6.2, 2026-07-20 — see open_items.md G-2).
#     The fallback needs NO extra capability (`cap_add`) at all: dropping
#     from root to the `numeric` user below only ever narrows privilege, and
#     Landlock is unprivileged by kernel design (network hooks need ABI v4 /
#     kernel 6.7+; probed empirically at runtime, never assumed).
#     HONEST NOTE: this is NOT a real sandbox the way macOS sandbox-exec /
#     bare-metal bwrap are — it's weaker isolation, accepted as "better than
#     nothing" rather than airtight. Network deny is TCP-only (not UDP/raw
#     sockets). Write confinement to the task's own dir is solid. Read
#     isolation between numeric tasks/sessions is NOT solid — they share one
#     system UID and task/execution IDs are a sequential counter + a
#     timestamp, not random, so sibling paths are cheap to construct (see
#     notebook/numeric/sandbox.py module docstring for the full reasoning on
#     why this residual was accepted rather than fixed).
#     `SandboxRunner.available()` probes both this and bwrap at runtime and
#     picks whichever actually works; if neither does, the app blocks numeric
#     execution with a clear error rather than running it unsandboxed.
#
# Build:   docker build -t research-notebook .
# Run:     see docker-compose.yml  (docker compose up)

FROM python:3.14-slim-trixie

ARG JULIA_VERSION=1.12.6
ARG DEBIAN_FRONTEND=noninteractive

# --- system tools: Maxima + build/runtime libs ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        maxima bubblewrap \
        curl ca-certificates git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
# bubblewrap is kept installed and is tried FIRST at runtime (works on
# bare-metal Linux); the numeric user below is the fallback used when bwrap's
# probe fails, as it currently does inside this image (see note above). No
# `cap_add` is needed for either the bwrap probe attempt or the fallback.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin numeric

# --- Julia (official binary, version-pinned) -----------------------------
# Arch-aware download so the same Dockerfile builds on x86_64 and arm64 hosts.
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64)  jarch="x64";     tarch="x86_64"  ;; \
        aarch64) jarch="aarch64"; tarch="aarch64" ;; \
        *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    minor="${JULIA_VERSION%.*}"; \
    url="https://julialang-s3.julialang.org/bin/linux/${jarch}/${minor}/julia-${JULIA_VERSION}-linux-${tarch}.tar.gz"; \
    curl -fsSL "$url" -o /tmp/julia.tar.gz; \
    mkdir -p /opt/julia; \
    tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1; \
    rm /tmp/julia.tar.gz; \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia; \
    julia --version

WORKDIR /app

# --- Python application deps (system interpreter; no .venv needed here) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Numeric-layer interpreter (dedicated venv; kept small) ---------------
COPY requirements-numeric.txt .
RUN python -m venv /app/.venv-numeric \
    && /app/.venv-numeric/bin/pip install --no-cache-dir -r requirements-numeric.txt

# --- Julia environment: instantiate the pinned Project/Manifest ----------
# Heaviest layer (resolves + precompiles Symbolics/QuantumAlgebra/SymPy). Copy
# only the manifests first so this layer caches across app-source changes.
#
# PYTHON="" forces PyCall.jl to bootstrap its OWN private Conda-managed Python
# (via Conda.jl) instead of the base image's /usr/local/bin/python3. Without
# this, PyCall's build-time heuristic picks up whatever python3 is on PATH —
# which on this base image lacks `sympy`, so SymPy.jl's __init__() fails with
# ModuleNotFoundError (empirically hit on Ubuntu 22.04, 2026-07-20). This also
# matches the already-working macOS dev setup, where PyCall independently
# resolved to its own private Conda distribution (`conda = true` in its
# deps.jl) rather than the app's .venv/.venv-numeric — those are unrelated to
# PyCall's Python resolution on any platform.
ENV PYTHON=""
COPY Project.toml Manifest.toml ./
RUN julia --project=/app -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# --- Application source ---------------------------------------------------
COPY . .

ENV PYTHONPATH=/app \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Ollama is expected on the host (network_mode: host). The launcher's port
# check skips auto-start when :11434 is already up (ui/llm_server_launcher.py).
CMD ["python", "-m", "uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
