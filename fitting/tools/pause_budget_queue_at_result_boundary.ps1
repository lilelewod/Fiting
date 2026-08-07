param(
    [Parameter(Mandatory = $true)][int]$QueuePid,
    [Parameter(Mandatory = $true)][int]$BudgetRunnerPid,
    [Parameter(Mandatory = $true)][int]$ComparisonRunnerPid,
    [Parameter(Mandatory = $true)][int]$FinalizerPid,
    [string]$ResultsFile = 'C:\code\Fiting\outputs\pmf_cylinder_budget_sensitivity\preregistered_20260721\fe_499920\results.json',
    [string]$LogFile = 'C:\code\Fiting\outputs\experiment_queue\budget_boundary_pause.log',
    [string]$ManifestFile = 'C:\code\Fiting\outputs\experiment_queue\budget_boundary_pause_manifest.json'
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

function Get-ResultCount {
    while ($true) {
        try {
            $content = Get-Content -LiteralPath $ResultsFile -Raw -Encoding UTF8 | ConvertFrom-Json
            return @($content).Count
        }
        catch {
            # The producer replaces the JSON file atomically enough for normal
            # readers, but retry any transient view rather than treating it as
            # a completed-result boundary.
            Start-Sleep -Milliseconds 500
        }
    }
}

function Stop-IfRunning {
    param([int]$Id, [string]$Label)
    if ($Id -le 0) { return }
    if ($null -ne (Get-Process -Id $Id -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
        Write-Log "Stopped $Label PID $Id."
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
$baseline = Get-ResultCount
Write-Log "Waiting for the next complete budget row after baseline $baseline."

while ($true) {
    if ($null -eq (Get-Process -Id $QueuePid -ErrorAction SilentlyContinue)) {
        Write-Log "Queue PID $QueuePid exited before a new row was observed; no forced stop applied."
        exit 0
    }
    $count = Get-ResultCount
    if ($count -gt $baseline) {
        Write-Log "Observed complete budget-row boundary: $baseline -> $count."
        break
    }
    Start-Sleep -Seconds 2
}

# Stop orchestration parents first so they cannot launch another formal run
# after the boundary. Then stop any worker descendants left in the snapshot.
$snapshot = @(Get-CimInstance Win32_Process)
$rootIds = @($QueuePid, $BudgetRunnerPid, $ComparisonRunnerPid)
$descendants = [System.Collections.Generic.HashSet[int]]::new()
$frontier = @($rootIds)
while ($frontier.Count -gt 0) {
    $next = @()
    foreach ($parent in $frontier) {
        foreach ($child in @($snapshot | Where-Object { $_.ParentProcessId -eq $parent })) {
            if ($descendants.Add([int]$child.ProcessId)) {
                $next += [int]$child.ProcessId
            }
        }
    }
    $frontier = $next
}

Stop-IfRunning -Id $FinalizerPid -Label 'finalizer'
Stop-IfRunning -Id $QueuePid -Label 'experiment queue'
Stop-IfRunning -Id $BudgetRunnerPid -Label 'budget orchestrator'
Stop-IfRunning -Id $ComparisonRunnerPid -Label 'comparison runner'
foreach ($id in @($descendants | Sort-Object -Descending)) {
    Stop-IfRunning -Id $id -Label 'descendant worker'
}

$finalCount = Get-ResultCount
$manifest = [ordered]@{
    status = 'PAUSED_AT_COMPLETE_BUDGET_RESULT_BOUNDARY'
    paused_at = (Get-Date).ToString('o')
    baseline_results = $baseline
    completed_results = $finalCount
    results_file = $ResultsFile
    stopped_processes = [ordered]@{
        queue_pid = $QueuePid
        budget_runner_pid = $BudgetRunnerPid
        comparison_runner_pid = $ComparisonRunnerPid
        finalizer_pid = $FinalizerPid
    }
    resume_command = 'powershell -ExecutionPolicy Bypass -File C:\code\Fiting\fitting\tools\resume_research_experiment_queue.ps1'
}
$json = $manifest | ConvertTo-Json -Depth 4
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ManifestFile, $json, $utf8NoBom)
Write-Log "Pause complete at $finalCount results; manifest written to $ManifestFile."
