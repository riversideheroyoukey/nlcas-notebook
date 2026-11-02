#!/usr/bin/env bash
#
# setup.sh — one-shot installer for Research Notebook (macOS-first)
#
# Brings up the full stack for a fresh machine:
#   1. system tools   : Maxima, Julia, Ollama   (via Homebrew)
#   2. Python venvs    : .venv (app) + .venv-numeric (numeric layer)
#   3. Julia env       : Project.toml / Manifest.toml -> Pkg.instantiate()
#   4. (optional)      : Julia sysimage build (~10 min, big startup speedup)
#   5. (optional)      : pull the Ollama models used by the LLM roles
#
# Ollama runs HOST-NATIVE on every platform (Metal on macOS, CUDA on Linux) —
# the 20–32B role models need a real accelerator, so it is never containerised.
# Docker users: see docs/setup_guide.md (app-only image, external Ollama).
#
# Usage:
#   ./setup.sh                 # interactive; prompts for sysimage + models
#   ./setup.sh --with-sysimage # build the Julia sysimage without asking
#   ./setup.sh --no-models     # skip the Ollama model pull
#   ./setup.sh --ci            # non-interactive, assume "no" for optional steps

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PY=python3.14
BUILD_SYSIMAGE=ask
PULL_MODELS=ask

for arg in "$@"; do
  case "$arg" in
    --with-sysimage) BUILD_SYSIMAGE=yes ;;
    --no-sysimage)   BUILD_SYSIMAGE=no ;;
    --no-models)     PULL_MODELS=no ;;
    --ci)            BUILD_SYSIMAGE=no; PULL_MODELS=no ;;
    -h|--help)       grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }

ask_yes_no() {  # $1 = prompt, returns 0 for yes
  local reply
  read -r -p "$1 [y/N] " reply || reply=""
  [[ "$reply" == "y" || "$reply" == "Y" ]]
}

# ---------------------------------------------------------------------------
# 0. Platform check
# ---------------------------------------------------------------------------
OS="$(uname -s)"
if [[ "$OS" != "Darwin" ]]; then
  warn "This script targets macOS (Homebrew). Detected: $OS."
  warn "On Linux/Windows use Docker instead — see docs/setup_guide.md."
  ask_yes_no "Continue anyway?" || exit 1
fi

# ---------------------------------------------------------------------------
# 1. System tools via Homebrew
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  warn "Homebrew not found. Install it from https://brew.sh then re-run."
  exit 1
fi

install_brew() {  # $1 = formula, $2 = command to probe
  if command -v "$2" >/dev/null 2>&1; then
    ok "$1 already present ($("$2" --version 2>/dev/null | head -1))"
  else
    info "Installing $1 ..."
    brew install "$1"
  fi
}

info "Checking system tools (Maxima, Julia, Ollama)..."
install_brew maxima maxima
install_brew julia  julia
install_brew ollama ollama

if ! command -v "$PY" >/dev/null 2>&1; then
  warn "$PY not found. Installing python@3.14 via Homebrew..."
  brew install python@3.14
  PY="$(brew --prefix)/bin/python3.14"
fi
ok "Python: $("$PY" --version)"

# ---------------------------------------------------------------------------
# 2. Python virtual environments
# ---------------------------------------------------------------------------
info "Creating application venv (.venv)..."
[[ -d .venv ]] || "$PY" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip >/dev/null
./.venv/bin/python -m pip install -r requirements.txt
ok "Application venv ready."

info "Creating numeric-layer venv (.venv-numeric)..."
[[ -d .venv-numeric ]] || "$PY" -m venv .venv-numeric
./.venv-numeric/bin/python -m pip install --upgrade pip >/dev/null
./.venv-numeric/bin/python -m pip install -r requirements-numeric.txt
ok "Numeric venv ready."

info "Installing Playwright's Chromium (for UI tests; skip-safe)..."
./.venv/bin/python -m playwright install chromium || \
  warn "Playwright browser install failed — UI tests will not run, app is unaffected."

# ---------------------------------------------------------------------------
# 3. Julia environment (pinned Project.toml / Manifest.toml)
# ---------------------------------------------------------------------------
info "Instantiating the pinned Julia environment (this can take a few minutes)..."
julia --project="$PROJECT_ROOT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
ok "Julia packages installed (QuantumAlgebra, Symbolics, SymPy, Latexify, SymbolicNumericIntegration, JSON3)."

# ---------------------------------------------------------------------------
# 4. Optional: Julia sysimage (large startup-latency win)
# ---------------------------------------------------------------------------
if [[ "$BUILD_SYSIMAGE" == "ask" ]]; then
  if ask_yes_no "Build the Julia sysimage now? (~10 min, faster second-quantization startup)"; then
    BUILD_SYSIMAGE=yes
  else
    BUILD_SYSIMAGE=no
  fi
fi
if [[ "$BUILD_SYSIMAGE" == "yes" ]]; then
  info "Building Julia sysimage..."
  ./scripts/build_sysimage.sh
  ok "Sysimage built (notebook/runtime/sysimage.*)."
else
  info "Skipping sysimage — the app runs without it (slower Julia startup only)."
fi

# ---------------------------------------------------------------------------
# 5. Optional: Ollama models
# ---------------------------------------------------------------------------
if [[ "$PULL_MODELS" == "ask" ]]; then
  echo
  info "The LLM roles use these Ollama models (edit models/ to change):"
  echo "    - qwen2.5-coder:32b    (supervisor + numeric codegen, default)"
  echo "    - gpt-oss:20b          (supervisor, alternate)"
  echo "    - qwen3-vl:latest      (handwriting OCR / VLM, default)"
  warn "Combined download is large (tens of GB). You can pull them later."
  if ask_yes_no "Pull these models now?"; then
    PULL_MODELS=yes
  else
    PULL_MODELS=no
  fi
fi
if [[ "$PULL_MODELS" == "yes" ]]; then
  ollama serve >/dev/null 2>&1 &
  sleep 3
  for m in gpt-oss:20b qwen2.5-coder:32b qwen3-vl:latest; do
    info "Pulling $m ..."; ollama pull "$m" || warn "pull failed: $m"
  done
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
ok "Setup complete."
echo "Start the notebook with:   ./scripts/start.sh"
echo "Then open:                 http://localhost:8000/notebook.html"
