#Requires -Version 5.1
<#
.SYNOPSIS
    Research Notebook — native Windows installer (no WSL2, no Docker).

.DESCRIPTION
    Mirrors setup.sh's philosophy: system tools (Python/Julia/Maxima/Ollama)
    are installed via their OWN official installers, NOT by this script — we
    only VERIFY they're on PATH and point to the official download page if
    missing. Everything after that (venvs, pip installs, Julia environment)
    is handled here, the same as setup.sh does for macOS.

    UNVERIFIED ON REAL HARDWARE (2026-07-21): designed and implemented by
    reasoning from the macOS/Linux setup scripts and this project's own
    cross-platform gotchas (see docs/open_items.md G-3, docs/developer_manuals/
    29_deployment_and_sandbox_design.md). No Windows machine was available to
    test this end-to-end. Report issues rather than assuming it "just works".

    The numeric layer (N-1) sandbox is implemented for Windows (open_items
    G-4: dedicated low-privilege account + Task Scheduler + per-account
    firewall rule) but needs a SEPARATE, admin-elevated, one-time setup step
    this script deliberately does NOT run (everything else here is
    intentionally non-elevated) - see
    scripts\setup_windows_numeric_sandbox.ps1. Until that's been run, the
    app detects this at runtime and blocks numeric execution with a clear
    error; the rest of the notebook (CAS + LLM) is unaffected.

.PARAMETER NoModels
    Skip the Ollama model pull prompt.

.PARAMETER CI
    Non-interactive: assume "no" for every optional step.
#>
[CmdletBinding()]
param(
    [switch]$NoModels,
    [switch]$CI
)

# NOT set to "Stop": Windows PowerShell 5.1 converts ANY stderr line from a
# native .exe into a terminating error under -Stop, independent of whatever
# redirect target that stream was given (2>$null does not save you). Flow
# control here relies entirely on explicit $LASTEXITCODE checks
# (Assert-LastSuccess below), exactly like bash scripts that don't use `set -e`.
$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Info  { param($Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Warn2 { param($Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Ok    { param($Msg) Write-Host "[ok] $Msg" -ForegroundColor Green }

function Confirm-Step {
    param([string]$Prompt)
    if ($CI) { return $false }
    # Read-Host throws PSInvalidOperationException in a non-interactive host
    # (e.g. an agent driving this script over SSH with no real console).
    # $ErrorActionPreference = "Continue" means that didn't crash the script
    # before, but it printed a scary-looking error trace for what is really
    # just "no console to prompt on, default to N" - catch it explicitly and
    # say so plainly instead.
    try {
        $reply = Read-Host "$Prompt [y/N]"
    } catch {
        Write-Warn2 "Non-interactive session (no console to prompt on) - defaulting to N for: $Prompt"
        return $false
    }
    return ($reply -eq "y" -or $reply -eq "Y")
}

# PowerShell does NOT abort on a failing native .exe the way bash's `set -e`
# does — $ErrorActionPreference only governs cmdlets, not external programs'
# exit codes. Call this immediately after every native command this script
# depends on for correctness, so a real failure actually stops the script
# instead of printing a false "success" message and continuing. (Kept as a
# plain post-check, not a wrapper that re-invokes the command via splatting —
# splatting args that look like flags, e.g. `-3.14`, into a custom function's
# ValueFromRemainingArguments is untested territory we don't need to risk.)
function Assert-LastSuccess {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $Description failed (exit code $LASTEXITCODE)." -ForegroundColor Red
        exit 1
    }
}

# Windows ships stub executables on PATH under App Execution Aliases
# (e.g. python.exe / python3.exe redirecting to the Microsoft Store) that
# exist even when the real tool is NOT installed. Get-Command alone can't
# tell the stub apart from the real thing, so this runs the version command,
# merges stderr into stdout (2>&1) to avoid the terminating-error conversion
# entirely for this call, and requires the output to actually look like a
# version string before trusting it.
function Test-RealVersion {
    param([string]$Exe, [string[]]$ArgList, [string]$Pattern)
    try {
        $out = & $Exe @ArgList 2>&1 | Out-String
    } catch {
        return $null
    }
    if ($out -match $Pattern) { return $out.Trim() }
    return $null
}

# ---------------------------------------------------------------------------
# 1. Verify system tools (installed via their OWN official installers)
# ---------------------------------------------------------------------------
Write-Info "Checking system tools (Python, Julia, Maxima, Ollama)..."

$missing = @()

$py = Get-Command py -ErrorAction SilentlyContinue
$pythonOk = $false
if ($py) {
    $ver = Test-RealVersion -Exe "py" -ArgList @("-3.14", "--version") -Pattern "Python \d+\.\d+"
    if ($ver) { $pythonOk = $true; Write-Ok "Python: $ver (via py launcher)" }
}
if (-not $pythonOk) {
    # Only trust a bare 'python' on PATH if it actually prints a version
    # string - the Microsoft Store App Execution Alias stub also satisfies
    # Get-Command python but does not.
    $ver = Test-RealVersion -Exe "python" -ArgList @("--version") -Pattern "Python \d+\.\d+"
    if ($ver) {
        Write-Warn2 "Found 'python' ($ver) but not confirmed as 3.14 via the py launcher. Continuing - pip/venv steps below will surface a clearer error if the version is wrong."
        $pythonOk = $true
    }
}
if (-not $pythonOk) {
    $missing += "Python 3.14 - https://www.python.org/downloads/ (check 'Add python.exe to PATH' and 'Install py launcher' during install)"
}

if (-not (Get-Command julia -ErrorAction SilentlyContinue)) {
    $missing += "Julia - https://julialang.org/install/ (official installer is juliaup; adds 'julia' to PATH)"
} else {
    Write-Ok "Julia: $((julia --version 2>&1 | Out-String).Trim())"
}

if (-not (Get-Command maxima -ErrorAction SilentlyContinue)) {
    $missing += "Maxima - https://maxima.sourceforge.io/download.html (the Windows installer does NOT always add itself to PATH - after installing, add the install dir's bin\ folder, e.g. C:\maxima-5.49.0\bin, to your PATH manually)"
} else {
    Write-Ok "Maxima found on PATH"
}

$ollamaFound = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
if (-not $ollamaFound) {
    $missing += "Ollama (optional - the app runs without it, only LLM features are disabled) - https://ollama.com/download/windows"
} else {
    Write-Ok "Ollama: $((ollama --version 2>&1 | Out-String).Trim())"
}

if ($missing.Count -gt 0) {
    Write-Warn2 "Missing tools:"
    foreach ($m in $missing) { Write-Host "    - $m" }
    Write-Warn2 "Install the missing tools above, open a NEW terminal (so PATH updates take effect), and re-run this script."
    $requiredMissing = @($missing | Where-Object { $_ -notmatch "^Ollama" })
    if ($requiredMissing.Count -gt 0) {
        exit 1
    }
    Write-Warn2 "Continuing without Ollama - LLM-assisted features will be disabled until it's installed."
}

# ---------------------------------------------------------------------------
# 2. Python virtual environments
# ---------------------------------------------------------------------------
Write-Info "Creating application venv (.venv)..."
if (-not (Test-Path ".venv")) {
    if ($py) {
        & py -3.14 -m venv .venv
    } else {
        & python -m venv .venv
    }
    Assert-LastSuccess "creating .venv"
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
Assert-LastSuccess "upgrading pip in .venv"
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Assert-LastSuccess "installing requirements.txt"
Write-Ok "Application venv ready."

Write-Info "Creating numeric-layer venv (.venv-numeric)..."
if (-not (Test-Path ".venv-numeric")) {
    if ($py) {
        & py -3.14 -m venv .venv-numeric
    } else {
        & python -m venv .venv-numeric
    }
    Assert-LastSuccess "creating .venv-numeric"
}
& .\.venv-numeric\Scripts\python.exe -m pip install --upgrade pip
Assert-LastSuccess "upgrading pip in .venv-numeric"
& .\.venv-numeric\Scripts\python.exe -m pip install -r requirements-numeric.txt
Assert-LastSuccess "installing requirements-numeric.txt"
Write-Ok "Numeric venv ready."
Write-Warn2 "Note: numeric-layer EXECUTION needs a separate one-time admin-elevated setup step: scripts\setup_windows_numeric_sandbox.ps1 (run as Administrator, once). Code generation and review work without it; execution is blocked with a clear error until it's been run."

Write-Info "Installing Playwright's Chromium (for UI tests; skip-safe)..."
& .\.venv\Scripts\python.exe -m playwright install chromium
if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "Playwright browser install failed - UI tests will not run, app is unaffected."
}

# ---------------------------------------------------------------------------
# 3. Julia environment (pinned Project.toml / Manifest.toml)
# ---------------------------------------------------------------------------
Write-Info "Instantiating the pinned Julia environment (this can take a few minutes)..."
# PYTHON="" forces PyCall.jl to bootstrap its OWN private Conda-managed Python
# instead of whatever `python` is on PATH — without this, PyCall's build-time
# heuristic picks up the just-installed system Python, which lacks `sympy`,
# and SymPy.jl's __init__() fails with ModuleNotFoundError. This is the exact
# bug hit (and fixed the same way) in the Docker image — see
# docs/developer_manuals/29_deployment_and_sandbox_design.md §2.2.
$env:PYTHON = ""
$juliaProject = "--project=$ProjectRoot"
& julia $juliaProject -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
Assert-LastSuccess "Julia Pkg.instantiate/precompile"
Write-Ok "Julia packages installed (QuantumAlgebra, Symbolics, SymPy, Latexify, SymbolicNumericIntegration, JSON3)."

# ---------------------------------------------------------------------------
# 4. Optional: Julia sysimage
# ---------------------------------------------------------------------------
$buildSysimage = $false
if (-not $CI) {
    $buildSysimage = Confirm-Step "Build the Julia sysimage now? (~10 min, faster second-quantization startup)"
}
if ($buildSysimage) {
    Write-Info "Building Julia sysimage..."
    $env:PYTHON = ""
    & julia $juliaProject "$ProjectRoot\scripts\build_julia_sysimage.jl"
    Assert-LastSuccess "Julia sysimage build"
    Write-Ok "Sysimage built (notebook\runtime\sysimage.dll)."
} else {
    Write-Info "Skipping sysimage - the app runs without it (slower Julia startup only)."
}

# ---------------------------------------------------------------------------
# 5. Optional: Ollama models
# ---------------------------------------------------------------------------
if ($ollamaFound -and (-not $NoModels)) {
    Write-Host ""
    Write-Info "The LLM roles use these Ollama models (edit models/ to change):"
    Write-Host "    - qwen2.5-coder:32b    (supervisor + coder, default)"
    Write-Host "    - gpt-oss:20b          (supervisor, alternate)"
    Write-Host "    - qwen3-vl:latest      (handwriting OCR / VLM, default)"
    Write-Warn2 "Combined download is large (tens of GB). You can pull them later."
    $pullModels = Confirm-Step "Pull these models now?"
    if ($pullModels) {
        foreach ($m in @("qwen2.5-coder:32b", "gpt-oss:20b", "qwen3-vl:latest")) {
            Write-Info "Pulling $m ..."
            & ollama pull $m
            if ($LASTEXITCODE -ne 0) { Write-Warn2 "pull failed: $m" }
        }
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Ok "Setup complete."
Write-Host "Start the notebook with:   .\scripts\start_windows.ps1"
Write-Host "Then open:                 http://localhost:8000/notebook.html"
