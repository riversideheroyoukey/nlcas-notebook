#Requires -Version 5.1
<#
.SYNOPSIS
    One-time, ADMIN-REQUIRED provisioning for the native Windows numeric
    sandbox (open_items G-4).

.DESCRIPTION
    Run this ONCE, elevated (right-click PowerShell -> Run as Administrator),
    SEPARATELY from setup_windows.ps1 - that script deliberately stays
    non-elevated (venv/pip/Julia never need admin rights), so this
    admin-only step is kept out of it rather than forcing every user to
    elevate just to run the app without the numeric layer.

    Sets up three things, mirroring the Docker fallback's "dedicated
    low-privilege account + network-only isolation" design
    (docs/developer_manuals/29_deployment_and_sandbox_design.md §7.4):

      1. A dedicated low-privilege local account ("numeric") that sandboxed
         numeric-layer code runs as. Its password is randomly generated and
         stored DPAPI-encrypted (decryptable only by the CURRENT Windows
         account on THIS machine) - never written in plaintext.
      2. "Log on as a batch job" (SeBatchLogonRight) for that account - not
         granted by default to a freshly created user, but required for
         Task Scheduler's /ru to launch anything as it. Discovered missing
         by testing on real hardware (schtasks /run fails silently without
         it).
      3. A Windows Firewall rule blocking ALL outbound network traffic for
         that account specifically (scoped by account, not executable path
         - .venv and .venv-numeric both redirect to the SAME shared base
         Python interpreter on Windows, so a program-path rule can't tell
         them apart; see notebook/numeric/sandbox.py's module docstring).

    Safe to re-run: account creation is skipped if it already exists, but
    the password is ROTATED every run - don't re-run this while a numeric
    task might be mid-execution.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Write-Info  { param($Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Warn2 { param($Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Ok    { param($Msg) Write-Host "[ok] $Msg" -ForegroundColor Green }

$currentPrincipal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: run this script as Administrator (right-click PowerShell -> Run as Administrator)." -ForegroundColor Red
    exit 1
}

$AccountName = "numeric"
$PwFile = Join-Path $ProjectRoot ".numeric_account_pw.dat"
$FirewallRuleName = "ResearchNotebook-NumericBlock"

# ---------------------------------------------------------------------------
# 1. Dedicated low-privilege account + randomly generated password
# ---------------------------------------------------------------------------
Write-Info "Creating/updating the dedicated '$AccountName' account..."

# Character set deliberately avoids quote/shell-special characters (", ', `,
# $, %, ^, &, <, >, |, space) so the plaintext never needs escaping when
# passed to schtasks /rp on a command line.
$chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#-_'
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 32
$rng.GetBytes($bytes)
$plainPw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
$securePw = ConvertTo-SecureString -String $plainPw -AsPlainText -Force

if (Get-LocalUser -Name $AccountName -ErrorAction SilentlyContinue) {
    Set-LocalUser -Name $AccountName -Password $securePw
    Write-Ok "Account '$AccountName' already existed; password rotated."
} else {
    New-LocalUser -Name $AccountName -Password $securePw -PasswordNeverExpires -UserMayNotChangePassword `
        -Description "Research Notebook: dedicated low-privilege account for sandboxed numeric-layer code execution (open_items G-4). Not for interactive use." `
        | Out-Null
    Write-Ok "Account '$AccountName' created."
}

# DPAPI CurrentUser scope: only THIS Windows account, on THIS machine, can
# decrypt this again later (notebook/numeric/sandbox.py's
# _windows_decrypt_password). Verified round-tripping correctly from an
# interactively-launched or Task-Scheduler-launched process; NOT from a bare
# non-interactive SSH `-Command`/`-File` invocation (profile not loaded
# there) - not a concern since the app is never started that way.
ConvertTo-SecureString -String $plainPw -AsPlainText -Force | ConvertFrom-SecureString |
    Set-Content -Path $PwFile
Write-Ok "Password stored (DPAPI-encrypted) at $PwFile"

# ---------------------------------------------------------------------------
# 2. "Log on as a batch job" (SeBatchLogonRight)
# ---------------------------------------------------------------------------
Write-Info "Granting 'Log on as a batch job' right to '$AccountName'..."
$sid = (Get-LocalUser -Name $AccountName).SID.Value
$infPath = Join-Path $env:TEMP "rn_grant_batch_logon.inf"
$dbPath = Join-Path $env:TEMP "rn_secedit.sdb"
@"
[Unicode]
Unicode=yes
[Version]
signature="`$CHICAGO`$"
Revision=1
[Privilege Rights]
SeBatchLogonRight = *$sid
"@ | Set-Content -Path $infPath -Encoding Unicode
secedit /configure /db $dbPath /cfg $infPath /areas USER_RIGHTS | Out-Null
Remove-Item $infPath, $dbPath -ErrorAction SilentlyContinue
Write-Ok "Batch logon right granted."

# ---------------------------------------------------------------------------
# 3. Firewall: block ALL outbound network traffic for this account
# ---------------------------------------------------------------------------
Write-Info "Adding outbound-block firewall rule for '$AccountName'..."
Remove-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
$sddl = "D:(A;;CC;;;$sid)"
New-NetFirewallRule -DisplayName $FirewallRuleName -Direction Outbound `
    -Action Block -Profile Any -LocalUser $sddl -Enabled True | Out-Null
Write-Ok "Firewall rule '$FirewallRuleName' added (blocks ALL outbound protocols for '$AccountName', not just TCP)."

Write-Host ""
Write-Ok "Numeric sandbox provisioning complete."
Write-Host 'Verify with:  .\.venv\Scripts\python.exe -c "from notebook.numeric.sandbox import SandboxRunner; print(SandboxRunner().available())"'
