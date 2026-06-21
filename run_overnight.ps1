# run_overnight.ps1 - runs diagnostic, backtest, and charts sequentially with logging.
# Launch in a regular PowerShell window:
#   cd <repo-root>
#   .\run_overnight.ps1
#
# Skip flags (set to $true to skip a stage):
param(
    [bool]$SkipDiagnostic = $false,
    [bool]$SkipBacktest = $false,
    [bool]$SkipCharts = $false
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"   # make Python flush stdout/stderr immediately
Set-Location $PSScriptRoot

$summary = "_run_summary.log"
$start = Get-Date
"=== Overnight run started: $start ===" | Out-File $summary

# Stage 0: ensure the price dataset exists (fetch free data if missing)
if (-not (Test-Path "df_2010.csv")) {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 0] df_2010.csv not found - fetching via download_data.py" | Tee-Object -FilePath $summary -Append
    python -u download_data.py *> _run_download.log
    if ($LASTEXITCODE -ne 0) {
        "[Stage 0] FAILED to fetch data (exit $LASTEXITCODE). See _run_download.log" | Tee-Object -FilePath $summary -Append
        exit 1
    }
}

# Stage 1: Diagnostic framework (produces outputs/)
if ($SkipDiagnostic) {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 1/3] SKIPPED (SkipDiagnostic=true)" | Tee-Object -FilePath $summary -Append
    $diagDone = Get-Date
} else {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 1/3] Diagnostic starting at $(Get-Date)" | Tee-Object -FilePath $summary -Append
    # python -u = unbuffered stdout/stderr so the log is readable in real time
    python -u momentum_ml_diagnostics.py *> _run_diagnostic.log
    $diagDone = Get-Date
    if ($LASTEXITCODE -eq 0) {
        "[Stage 1/3] DONE at $diagDone (took $($diagDone - $start))" | Tee-Object -FilePath $summary -Append
    } else {
        "[Stage 1/3] FAILED at $diagDone (exit $LASTEXITCODE). See _run_diagnostic.log" | Tee-Object -FilePath $summary -Append
    }
}

# Stage 2: Backtest (produces portfolio_returns.csv, portfolio_positions.csv, strategy_performance.png)
if ($SkipBacktest) {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 2/3] SKIPPED (SkipBacktest=true)" | Tee-Object -FilePath $summary -Append
    $backDone = Get-Date
} else {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 2/3] Backtest starting at $(Get-Date)" | Tee-Object -FilePath $summary -Append
    python -u momentum_ml_framework.py *> _run_backtest.log
    $backDone = Get-Date
    if ($LASTEXITCODE -eq 0) {
        "[Stage 2/3] DONE at $backDone (took $($backDone - $diagDone))" | Tee-Object -FilePath $summary -Append
    } else {
        "[Stage 2/3] FAILED at $backDone (exit $LASTEXITCODE). See _run_backtest.log" | Tee-Object -FilePath $summary -Append
    }
}

# Stage 3: Charts (consumes portfolio_returns.csv from stage 2)
if ($SkipCharts) {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 3/3] SKIPPED (SkipCharts=true)" | Tee-Object -FilePath $summary -Append
    $chartsDone = Get-Date
} else {
    "" | Tee-Object -FilePath $summary -Append
    "[Stage 3/3] Charts starting at $(Get-Date)" | Tee-Object -FilePath $summary -Append
    python -u make_charts.py *> _run_charts.log
    $chartsDone = Get-Date
    if ($LASTEXITCODE -eq 0) {
        "[Stage 3/3] DONE at $chartsDone (took $($chartsDone - $backDone))" | Tee-Object -FilePath $summary -Append
    } else {
        "[Stage 3/3] FAILED at $chartsDone (exit $LASTEXITCODE). See _run_charts.log" | Tee-Object -FilePath $summary -Append
    }
}

"" | Tee-Object -FilePath $summary -Append
"=== All stages complete at $(Get-Date). Total runtime: $($chartsDone - $start) ===" | Tee-Object -FilePath $summary -Append
"" | Tee-Object -FilePath $summary -Append
"Log files in this folder:" | Tee-Object -FilePath $summary -Append
"  _run_summary.log    (this file)" | Tee-Object -FilePath $summary -Append
"  _run_diagnostic.log (stage 1 stdout/stderr)" | Tee-Object -FilePath $summary -Append
"  _run_backtest.log   (stage 2 stdout/stderr)" | Tee-Object -FilePath $summary -Append
"  _run_charts.log     (stage 3 stdout/stderr)" | Tee-Object -FilePath $summary -Append
