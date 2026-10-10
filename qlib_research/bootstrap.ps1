# qlib_research bootstrap - one command setup on a fresh machine
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File qlib_research\bootstrap.ps1 `
#       -RawCsv "D:\BaiduNetdiskDownload\@MNQ#C_1min_20261003.csv"
#
# What it does:
#   1. creates qlib_research\.venv (Python 3.11)
#   2. installs pinned deps (requirements.txt)
#   3. sets MLFLOW_ALLOW_FILE_STORE=true for this session + user env
#   4. rebuilds the qlib bins under qlib_research\data\mnq from the raw 1min CSV
#   5. prints the exact command(s) to continue the research
#
# Re-runnable: skips steps already done.

[CmdletBinding()]
param(
    [string]$RawCsv   = "D:\BaiduNetdiskDownload\@MNQ#C_1min_20261003.csv",
    [string]$RepoRoot = "",
    [switch]$SkipVenv,
    [switch]$SkipData,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# --- resolve paths -----------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path      # ...\qlib_research
if ([string]::IsNullOrWhiteSpace($RepoRoot)) { $RepoRoot = Split-Path -Parent $ScriptDir }
$Research  = Join-Path $RepoRoot "qlib_research"
$VenvDir   = Join-Path $Research ".venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"
$DataDir   = Join-Path $Research "data\mnq"
$ReqFile   = Join-Path $Research "requirements.txt"
$Ingest    = Join-Path $Research "tools\ingest_mnq_1min.py"

Write-Host "== qlib_research bootstrap ==" -ForegroundColor Cyan
Write-Host "RepoRoot : $RepoRoot"
Write-Host "Research : $Research"
Write-Host "DataDir  : $DataDir"
Write-Host "RawCsv   : $RawCsv"

# --- 1. venv -----------------------------------------------------------------
if ($SkipVenv) {
    Write-Host "[1/4] venv  : skipped"
} elseif (Test-Path $VenvPy) {
    Write-Host "[1/4] venv  : exists -> $VenvDir"
} else {
    Write-Host "[1/4] venv  : creating (Python 3.11) ..."
    $py = $null
    if (Get-Command py -ErrorAction SilentlyContinue) { $py = @("py","-3.11") }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = @("python") }
    else { throw "No Python launcher found. Install Python 3.11 first." }
    & $py[0] $py[1..($py.Count-1)] -m venv $VenvDir
}

# --- 2. deps -----------------------------------------------------------------
Write-Host "[2/4] deps  : installing ..."
$env:PIP_NO_INPUT = "1"
if (Test-Path $ReqFile) {
    & $VenvPy -m pip install --upgrade pip | Out-Null
    & $VenvPy -m pip install -r $ReqFile
} else {
    & $VenvPy -m pip install "pyqlib==0.9.7"
}

# --- 3. env ------------------------------------------------------------------
Write-Host "[3/4] env   : MLFLOW_ALLOW_FILE_STORE=true"
$env:MLFLOW_ALLOW_FILE_STORE = "true"
[Environment]::SetEnvironmentVariable("MLFLOW_ALLOW_FILE_STORE", "true", "User")

# --- 4. data -----------------------------------------------------------------
$dataOk = (Test-Path (Join-Path $DataDir "instruments")) -and (Test-Path (Join-Path $DataDir "features"))
if ($SkipData) {
    Write-Host "[4/4] data  : skipped"
} elseif ($dataOk -and -not $Force) {
    Write-Host "[4/4] data  : exists -> $DataDir (use -Force to rebuild)"
} else {
    if (-not (Test-Path $RawCsv)) {
        Write-Warning "Raw CSV not found: $RawCsv"
        Write-Warning "Download '@MNQ#C_1min_20261003.csv' and re-run with -RawCsv <path>, or skip with -SkipData."
    } else {
        Write-Host "[4/4] data  : rebuilding qlib bins from $RawCsv ..."
        & $VenvPy $Ingest
    }
}

Write-Host ""
Write-Host "Done. Next steps (examples):" -ForegroundColor Green
Write-Host "  # IS 15min sealed report / robustness:"
Write-Host "  $VenvPy $Research\tf_mnq\scripts\run_m2_report.py"
Write-Host "  # Stage 0/A/B/C + R1-R4:"
Write-Host "  $VenvPy $Research\tf_mnq\scripts\run_m3_stages.py"
Write-Host ""
Write-Host "Context/handoff doc: $Research\HANDOFF.md"
