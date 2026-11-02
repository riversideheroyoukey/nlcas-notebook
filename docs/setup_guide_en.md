# Setup & Distribution Guide

How to install Research Notebook. Two supported platform tracks:

| Platform | Recommended path | LLM (Ollama) | Numeric layer (N-1) |
|---|---|---|---|
| **macOS** (primary) | `./setup.sh` (native) | host-native (Metal) | works (sandbox-exec) |
| **Linux native** (no Docker) | manual install | host-native | works (bubblewrap, verified on real hardware) |
| **Windows native** (recommended, verified) | `setup_windows.ps1` | host-native | works (dedicated account + Task Scheduler, verified on real hardware) |
| **Linux / Windows (Docker)** (Windows path: needs-driven, unverified) | Docker (`docker compose`) | **host-native** (separate) | works (light isolation, see below) |

> **Design assumption:** Ollama serves 20–32B-class models, so on every platform it
> runs **host-native with GPU access** (Metal on macOS, CUDA on Linux) — it is never
> containerized. This sidesteps the usual "GPU passthrough in Docker" headache
> entirely.

Each platform's section is self-contained — read only the one that applies to you
(§2 macOS, §3 Linux, §4 Windows). Shared material lives in §1 and §5 onward.

---

## 1. Prerequisites

| Tool | Version (tested) | Purpose |
|---|---|---|
| Python | 3.14.4 | application + numeric layer |
| Maxima | 5.49.0 | symbolic backend (standard sessions) |
| Julia | 1.12.6 | second-quantization sessions (QuantumAlgebra/Symbolics/SymPy) |
| Ollama | 0.20.7 | LLM Supervisor / OCR / numeric codegen |

Python dependencies are **pinned exactly from a working venv** in `requirements.txt`
(app `.venv`) and `requirements-numeric.txt` (numeric-layer `.venv-numeric`). Julia
dependencies are pinned in `Project.toml` / `Manifest.toml` (QuantumAlgebra 1.6.0,
Symbolics 7.19.0, Latexify 0.16.10, SymPy 2.3.3, SymbolicNumericIntegration 1.10.0,
JSON3 1.14.3).

---

## 2. macOS (recommended — one shot)

### 2.1 Install

```bash
./setup.sh
```

What it does:

1. Installs `maxima` / `julia` / `ollama` via Homebrew (only what's missing)
2. Creates `.venv` (app) and `.venv-numeric` (numeric layer), pip-installs both
3. Installs Playwright's Chromium (for UI tests; failure here doesn't affect the app)
4. Runs `Pkg.instantiate()` for the Julia environment from `Project.toml`
5. (optional) Builds the Julia sysimage (~10 min, big startup speedup)
6. (optional) Pulls the Ollama models

Options:

```bash
./setup.sh --with-sysimage   # build the sysimage without prompting
./setup.sh --no-models       # skip the Ollama model pull
./setup.sh --ci              # non-interactive (all optional steps default to no)
```

### 2.2 Doing it by hand (what setup.sh does under the hood)

```bash
brew install maxima julia ollama
python3.14 -m venv .venv         && ./.venv/bin/pip install -r requirements.txt
python3.14 -m venv .venv-numeric && ./.venv-numeric/bin/pip install -r requirements-numeric.txt
julia --project=. -e 'using Pkg; Pkg.instantiate()'
# optional: ./scripts/build_sysimage.sh
```

### 2.3 Ollama models

```bash
brew install ollama
ollama serve &                       # uses Metal GPU on Apple Silicon
ollama pull qwen2.5-coder:32b gpt-oss:20b qwen3-vl:latest
```

Model-to-role mapping is in §5 (`setup.sh` walks through this for you).

### 2.4 Launch

```bash
./scripts/start.sh
# → http://localhost:8000/notebook.html
```

The app starts even without Ollama running (only the LLM-assisted features are
disabled).

---

## 3. Linux (Docker)

Only the application stack (Python + Maxima + Julia) is containerized;
**Ollama runs on the host.**

> **The numeric layer (N-1) works inside a Docker container, but only with
> light isolation — NOT a real sandbox the way macOS/bare-metal Linux are.**
> Its sandbox (`notebook/numeric/sandbox.py`) tries `bubblewrap` first, but
> inside a container its internal loopback-configuration step fails with
> `Failed RTM_NEWADDR: No child processes` (root cause undetermined after
> ruling out bwrap version, SIGCHLD inheritance, namespace-flag composition,
> and Docker's seccomp profile — see `docs/open_items.md` §G-2). So Docker
> falls back to a **dedicated low-privilege user + Landlock (network-only)**
> instead — no extra capability (`cap_add`) needed either way. What that
> fallback actually guarantees: (1) network denial is **TCP-only** (UDP/raw
> sockets aren't touched by Landlock); (2) writes outside the task's own
> work dir are solidly denied; (3) read isolation from other sessions/tasks
> is **weak** — `sessions/` itself is unlistable, but task/execution IDs are
> a sequential counter and a timestamp (not random), and sibling tasks share
> this one system UID, so an exact sibling path is cheap to guess and would
> be readable. This was accepted as "better than nothing" rather than fixed,
> because the two code paths that ever reach this sandbox are (a)
> deterministic templates, whose function whitelist can't express file-path
> or network I/O at all, and (b) LLM-generated code, which can't run before
> a human approves it — OS-level isolation here is a backstop for reviewer
> oversight, not the primary defense. See `docs/open_items.md` §G-2 for the
> full investigation and reasoning.
>
> Also: Apple Vision OCR is unavailable (`ui/apfel_client.py` is macOS-only and not
> wired in by default anyway; Docker relies on Ollama's VLM `qwen3-vl` for
> handwriting OCR instead). The sysimage is built inside the image (it's
> architecture-specific, so the host's copy is never reused).

### 3.1 Installing and connecting Ollama (Linux, with GPU)

```bash
# 1) Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2) For an NVIDIA GPU: make sure the driver + CUDA are installed
nvidia-smi                           # confirm the GPU is visible

# 3) Keep it running (the installer enables a systemd service; to run manually:)
ollama serve &

# 4) Pull the models (role mapping in §5)
ollama pull qwen2.5-coder:32b
ollama pull gpt-oss:20b
ollama pull qwen3-vl:latest

# 5) Sanity check (also reachable from the container via localhost)
curl http://localhost:11434/api/tags
```

Ollama listens on `127.0.0.1:11434` by default. `docker-compose.yml` uses
`network_mode: host`, so the container's own `localhost:11434` reaches the host's
Ollama directly (no changes needed to `models/*/endpoint.json`). To use a remote
Ollama host or a bridge network instead, have Ollama listen on
`OLLAMA_HOST=0.0.0.0:11434` and point `base_url` at that host.

### 3.2 Launch

```bash
docker compose up --build
# → http://localhost:8000/notebook.html
```

---

## 4. Windows (native recommended)

> **Native Windows (no Docker) is the recommended path.** Both the app and
> the numeric sandbox have been verified end to end on real hardware (§4.1).
> Docker Desktop (§4.2) requires WSL2 anyway, so it's actually more setup
> work than native, and its numeric sandbox fallback is weaker (TCP-only).
> Use §4.1 unless you have a specific reason to want Docker
> (`docs/open_items.md` §G-3).

### 4.1 Native Windows (recommended, verified on real hardware)

The core app (the CAS + LLM notebook) is plain Python (FastAPI/uvicorn), so
there's no reason it can't run natively on Windows. The numeric layer's
sandbox is implemented too (dedicated low-privilege account + Task Scheduler +
per-account firewall rule, `docs/open_items.md` §G-4), but needs a **separate,
admin-elevated, one-time setup step** (`scripts\setup_windows_numeric_sandbox.ps1`,
below) — until that's been run, the existing fail-safe blocks numeric
execution with a clear error; the rest of the app is unaffected.

**Same philosophy as `setup.sh`:** Python, Julia, Maxima, and Ollama are
installed via their own official installers — this script only verifies
they're present and does the app-level setup (venv, pip, Julia instantiate).

```powershell
# Official installers (this script does NOT install these — only checks for them):
#   Python 3.14  https://www.python.org/downloads/
#   Julia        https://julialang.org/install/ (juliaup)
#   Maxima       https://maxima.sourceforge.io/download.html
#   Ollama       https://ollama.com/download/windows (optional — the app runs without it)

.\setup_windows.ps1
.\scripts\start_windows.ps1
# → http://localhost:8000/notebook.html
```

**Known gotchas:**

- **PowerShell execution policy.** A default `Restricted`/`RemoteSigned` policy
  can block `.ps1` scripts from running at all. If you hit that, either run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (no admin needed) or
  launch with `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1`.
- **Maxima's official installer doesn't always add itself to PATH.** If the
  `maxima` command isn't found after installing, add the install directory's
  `bin\` folder (e.g. `C:\maxima-5.49.0\bin`) to PATH manually.
- **Same PyCall/SymPy.jl Python-resolution bug as Docker, pre-empted.** The
  bug found in the Docker image (PyCall picks up the system Python, which
  lacks `sympy`, so `Pkg.precompile()` fails —
  `docs/developer_manuals/29_deployment_and_sandbox_design.md` §2.2) can
  happen here too if the official-installer Python is on PATH.
  `setup_windows.ps1` sets `$env:PYTHON = ""` before the Julia build to force
  PyCall's own private Conda bootstrap (matching the working macOS path).
- **`.venv-numeric`'s path layout is Windows-aware** (`Scripts\python.exe`,
  `notebook/numeric/sandbox.py`).
- **Numeric execution needs one more setup step.** `setup_windows.ps1` (this
  script) deliberately stays non-elevated, so it doesn't provision the
  numeric sandbox. Run this once, in an Administrator PowerShell:
  ```powershell
  .\scripts\setup_windows_numeric_sandbox.ps1
  ```
  It creates a dedicated low-privilege local account (`numeric`), stores its
  password DPAPI-encrypted, grants the "Log on as a batch job" right Task
  Scheduler's `/ru` needs, and adds a firewall rule blocking all outbound
  traffic for that account. Until this has run, code generation and review
  work, but **execution** is blocked with a clear error.

Full verification trail: `docs/developer_manuals/29_deployment_and_sandbox_design.md` §5, §7.

---

### 4.2 Windows (Docker Desktop — only if you specifically want Docker; UNVERIFIED on real hardware)

> **The numeric layer (N-1) works inside a Docker container here too, with
> the same light isolation as §3.** Docker Desktop's WSL2 backend is a real
> Linux kernel, so `notebook/numeric/sandbox.py` tries the same `bubblewrap`
> path as native Linux first, but it's likely to hit the same container
> failure (§3, untested under WSL2 specifically) and fall back to the
> dedicated low-privilege user + Landlock (network-only) mechanism — no
> extra capability needed either way. What that guarantees and what it
> doesn't (TCP-only network denial, solid write confinement, weak read
> isolation between sessions/tasks) is identical to §3's explanation.
> **This path itself (including actually building `docker-compose.windows.yml`)
> has never been run on real hardware** (`docs/open_items.md` §G-3 — priority
> downgraded to needs-driven). Use §4.1 (native Windows) unless you have a
> specific reason to want Docker.
>
> Also: Apple Vision OCR is unavailable (Docker relies on Ollama's VLM `qwen3-vl`
> for handwriting OCR instead). The sysimage is built inside the image.

#### 4.2.1 Installing and connecting Ollama (Windows)

- **Recommended:** set up WSL2 (Ubuntu), install Ollama there following the Linux
  steps in §3.1. The app container reaches it automatically at
  `host.docker.internal:11434` via Docker Desktop's WSL2 backend. NVIDIA GPUs are
  usable via WSL2 + CUDA.
- Ollama installed natively on Windows (`https://ollama.com/download/windows`)
  works the same way — reachable at `host.docker.internal:11434` with no extra
  configuration.

Model-to-role mapping is in §5.

#### 4.2.2 Launch

A ready-made Windows compose file is included:

```powershell
docker compose -f docker-compose.windows.yml up --build
# → http://localhost:8000/notebook.html
```

Two differences from `docker-compose.yml` (Linux):

1. **No `network_mode: host`.** Docker Desktop's host networking is an opt-in beta
   feature (not on by default), so this uses ordinary bridge networking with an
   explicit port mapping (`ports: ["8000:8000"]`) instead.
2. **Ollama is reached via `host.docker.internal`.** Docker Desktop resolves this
   hostname to the host automatically — no `extra_hosts` needed. But
   `models/*/endpoint.json` is checked in with `localhost` (correct for the native
   macOS/Linux setup), so this compose file bind-mounts `host.docker.internal`-
   rewritten copies over just the 4 files that matter for the default roles
   (Supervisor/Coder = `qwen2.5-coder:32b`, its alternate `gpt-oss:20b`, and the
   `qwen3-vl` OCR engine) from `models.windows/`. Because it's a file-level mount,
   sibling files (`system_prompt.txt`, etc.) still come from the image unchanged.
   If you re-target a role to a different local Ollama profile, add the matching
   override under `models.windows/` and mount it the same way.

### 4.3 Aside: using Docker on macOS (discouraged path)

On macOS, prefer §2's native `./setup.sh` path. If you must use Docker there
anyway, the same technique as `docker-compose.windows.yml` (bridge networking +
port mapping + `host.docker.internal` rewrite) applies.

---


## 5. Models used, by role (all platforms)

Based on the actual defaults in `config/default_settings.json` (full detail in
`docs/developer_manuals/16_llm_model_setup.md`). The app talks to an
OpenAI-compatible endpoint (`http://localhost:11434` by default). Switching
providers (Ollama → vLLM / LM Studio / etc.) is a config-only change: edit
`local_llm_server` in `config/default_settings.json` and the matching `base_url`
in `models/*/endpoint.json`.

| Role | Model | Config location |
|---|---|---|
| Supervisor/Coder (recommended, default) | `qwen2.5-coder:32b` | `models/ollama_qwen25_coder_32b/` |
| Supervisor (alternate) | `gpt-oss:20b` | `models/ollama_gpt_oss_20b/` |
| Handwriting OCR / VLM (default) | `qwen3-vl:latest` | `models/engines/ollama_qwen3_vl/` |

The adviser role's default target is the project owner's own remote LM Studio
machine (`models/lmstudio_remote_gpt-oss-120b/`) — unreachable from a single-machine
setup. To use it locally instead, point `adviser_target` in
`config/default_settings.json` at one of the local Ollama profiles above.

### No GPU / want smaller models

Swap the models referenced in `models/*/endpoint.json` and each role's config for
smaller ones (e.g. `llama3.2`) — each role can be changed independently. CPU-only
inference isn't practical at the 20–32B scale the defaults use, so downsizing the
models is recommended in that case.

---

## 6. Tests (for developers)

```bash
source .venv/bin/activate
pytest                      # spins up Maxima/Julia, seconds-to-minutes scale
playwright install chromium # one-time, only needed to run the UI test suite
```

---

## See also

- `README_jp.md` / `README_en.md` — feature overview
- `docs/developer_manuals/16_llm_model_setup.md` — LLM/model configuration details
- `docs/developer_manuals/22_numeric_layer_design.md` — numeric layer (N-1) design and execution model
- `docs/open_items.md` §G — open setup/distribution items (cross-platform numeric
  sandbox, etc.)
