param(
    [Parameter(Mandatory = $true)][datetime]$NotBefore,
    [Parameter(Mandatory = $true)][int]$QueuePid,
    [Parameter(Mandatory = $true)][int]$RobustnessRunnerPid,
    [int]$MonitorPid = 0,
    [int]$FinalizerPid = 0,
    [string]$RobustnessRoot = 'C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721',
    [string]$LogFile = 'C:\code\Fiting\outputs\experiment_queue\scheduled_pause.log',
    [string]$ManifestFile = 'C:\code\Fiting\outputs\experiment_queue\scheduled_pause_manifest.json'
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

function Get-ResultState {
    $files = @(Get-ChildItem -LiteralPath $RobustnessRoot -Recurse -Filter 'results.json' -File -ErrorAction SilentlyContinue)
    $rows = 0
    $latest = [datetime]::MinValue
    $valid = $true
    foreach ($file in $files) {
        try {
            $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            $rows += @($content).Count
            if ($file.LastWriteTime -gt $latest) { $latest = $file.LastWriteTime }
        } catch {
            # A writer can briefly expose an incomplete view while replacing a
            # JSON file.  Such a snapshot must never be used as a stop boundary.
            $valid = $false
        }
    }
    return @{ Rows = $rows; Latest = $latest; Valid = $valid }
}

function Stop-IfRunning {
    param([int]$Id, [string]$Label)
    if ($Id -le 0) { return }
    $process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
        Write-Log "Stopped $Label PID $Id."
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
Write-Log "Scheduled safe pause; not before $($NotBefore.ToString('yyyy-MM-dd HH:mm:ss'))."

while ((Get-Date) -lt $NotBefore) {
    if ($null -eq (Get-Process -Id $QueuePid -ErrorAction SilentlyContinue)) {
        Write-Log "Queue PID $QueuePid exited before the pause deadline; no action needed."
        exit 0
    }
    Start-Sleep -Seconds 10
}

do {
    $baseline = Get-ResultState
    if (-not $baseline.Valid) { Start-Sleep -Seconds 1 }
} while (-not $baseline.Valid)
Write-Log "Pause window opened at $($baseline.Rows) completed robustness rows; waiting for the next atomic result write."

$boundaryReached = $false
$waitDeadline = (Get-Date).AddMinutes(8)
while ((Get-Date) -lt $waitDeadline) {
    $state = Get-ResultState
    if ($state.Valid -and $state.Rows -gt $baseline.Rows) {
        Write-Log "Observed completed-row boundary: $($baseline.Rows) -> $($state.Rows)."
        $boundaryReached = $true
        break
    }
    if ($null -eq (Get-Process -Id $RobustnessRunnerPid -ErrorAction SilentlyContinue)) {
        Write-Log "Robustness runner exited while waiting for a boundary."
        $boundaryReached = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $boundaryReached) {
    Write-Log 'No result boundary appeared within 8 minutes; applying the hard safety cutoff. Completed rows remain resumable.'
}

# Stop the completion worker first so an intentionally paused, incomplete queue is
# not mistaken for a finished experiment. Stop the orchestration parents before
# their active child to prevent a new comparison process from being launched.
Stop-IfRunning -Id $FinalizerPid -Label 'finalizer'
Stop-IfRunning -Id $MonitorPid -Label 'boundary monitor'
Stop-IfRunning -Id $RobustnessRunnerPid -Label 'robustness runner'

$comparisonProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:\.exe)?$' -and
    $_.CommandLine -match 'run_optimizer_comparison\.py' -and
    $_.CommandLine -match [regex]::Escape($RobustnessRoot)
})
foreach ($process in $comparisonProcesses) {
    Stop-IfRunning -Id ([int]$process.ProcessId) -Label 'active comparison child'
}

Stop-IfRunning -Id $QueuePid -Label 'experiment queue'
$finalState = Get-ResultState
if ($finalState.Valid) {
    Write-Log "Safe pause complete with $($finalState.Rows) robustness rows. Resume is supported by the existing queue runner."
} else {
    Write-Log 'Safe pause complete; final row count will be verified on resume because a JSON writer was still closing.'
}

$manifest = [ordered]@{
    status = 'PAUSED_AT_COMPLETE_RESULT_BOUNDARY'
    paused_at = (Get-Date).ToString('o')
    completed_corrupted_rows = if ($finalState.Valid) { $finalState.Rows } else { $null }
    result_snapshot_valid = [bool]$finalState.Valid
    stopped_processes = [ordered]@{
        queue_pid = $QueuePid
        robustness_runner_pid = $RobustnessRunnerPid
        monitor_pid = $MonitorPid
        finalizer_pid = $FinalizerPid
    }
    robustness_root = $RobustnessRoot
    resume_command = 'powershell -ExecutionPolicy Bypass -File C:\code\Fiting\fitting\tools\resume_research_experiment_queue.ps1'
}
$manifestJson = $manifest | ConvertTo-Json -Depth 4
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ManifestFile, $manifestJson, $utf8NoBom)
Write-Log "Pause manifest written to $ManifestFile."
