# Start script for Research Notebook (native Windows)
# Mirrors scripts/start.sh. Verified on real hardware (2026-07-21〜23) — see
# docs/developer_manuals/29_deployment_and_sandbox_design.md §5.1.

$ErrorActionPreference = "Stop"

# Directory of this script, and its parent (project root) — mirrors start.sh's
# $(dirname "${BASH_SOURCE[0]}")/.. pattern using PowerShell's $PSScriptRoot.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Starting Research Notebook..."

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning ".venv not found. Running with global python."
}

# Export PYTHONPATH so the notebook module is discoverable if needed
$env:PYTHONPATH = $ProjectRoot

Write-Host "Starting UI server..."
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
