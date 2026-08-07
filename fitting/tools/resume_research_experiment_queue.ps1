param(
    [switch]$DryRun,
    [int]$MonitorPollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\code\Fiting\fitting'
$OutputRoot = 'C:\code\Fiting\outputs\experiment_queue'
$QueueScript = Join-Path $ProjectRoot 'tools\continue_research_experiment_queue.ps1'
$MonitorScript = Join-Path $ProjectRoot 'tools\monitor_v3_robustness_boundaries.ps1'
$FinalizerScript = Join-Path $ProjectRoot 'tools\finalize_research_queue.ps1'
$PowerShell = (Get-Command powershell.exe).Source
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$commands = [ordered]@{
    queue = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $QueueScript, '-WaitForDensitySupportPid', '0')
    monitor = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $MonitorScript, '-RunnerPid', '<queue-pid>', '-PollSeconds', "$MonitorPollSeconds")
    finalizer = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $FinalizerScript, '-WaitForQueuePid', '<queue-pid>')
}

if ($DryRun) {
    [ordered]@{
        status = 'DRY_RUN'
        executable = $PowerShell
        commands = $commands
    } | ConvertTo-Json -Depth 5
    exit 0
}

$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'continue_research_experiment_queue\.ps1|run_v3_stratified_superquadric_robustness\.py|run_pmf_cylinder_budget_sensitivity\.py'
})
if ($active.Count -gt 0) {
    $description = ($active | ForEach-Object { "PID $($_.ProcessId): $($_.CommandLine)" }) -join [Environment]::NewLine
    throw "A research queue or formal runner is already active; refusing a duplicate launch.$([Environment]::NewLine)$description"
}

$queueStdout = Join-Path $OutputRoot "queue_resume_$Stamp.stdout.log"
$queueStderr = Join-Path $OutputRoot "queue_resume_$Stamp.stderr.log"
$monitorStdout = Join-Path $OutputRoot "monitor_resume_$Stamp.stdout.log"
$monitorStderr = Join-Path $OutputRoot "monitor_resume_$Stamp.stderr.log"
$finalizerStdout = Join-Path $OutputRoot "finalize_resume_$Stamp.stdout.log"
$finalizerStderr = Join-Path $OutputRoot "finalize_resume_$Stamp.stderr.log"
$started = @()

try {
    $queue = Start-Process -FilePath $PowerShell -ArgumentList $commands.queue `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $queueStdout -RedirectStandardError $queueStderr
    $started += $queue
    Start-Sleep -Seconds 2
    if ($queue.HasExited) {
        throw "Queue exited during startup with code $($queue.ExitCode); inspect $queueStderr"
    }

    $monitorArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $MonitorScript,
        '-RunnerPid', "$($queue.Id)", '-PollSeconds', "$MonitorPollSeconds")
    $monitor = Start-Process -FilePath $PowerShell -ArgumentList $monitorArgs `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $monitorStdout -RedirectStandardError $monitorStderr
    $started += $monitor

    $finalizerArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $FinalizerScript,
        '-WaitForQueuePid', "$($queue.Id)")
    $finalizer = Start-Process -FilePath $PowerShell -ArgumentList $finalizerArgs `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $finalizerStdout -RedirectStandardError $finalizerStderr
    $started += $finalizer

    $manifest = [ordered]@{
        status = 'RUNNING'
        started_at = (Get-Date).ToString('o')
        queue_pid = $queue.Id
        monitor_pid = $monitor.Id
        finalizer_pid = $finalizer.Id
        logs = [ordered]@{
            queue_stdout = $queueStdout
            queue_stderr = $queueStderr
            monitor_stdout = $monitorStdout
            monitor_stderr = $monitorStderr
            finalizer_stdout = $finalizerStdout
            finalizer_stderr = $finalizerStderr
        }
    }
    $manifestPath = Join-Path $OutputRoot "resume_manifest_$Stamp.json"
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    $manifest | Add-Member -NotePropertyName manifest -NotePropertyValue $manifestPath
    $manifest | ConvertTo-Json -Depth 4
}
catch {
    foreach ($process in ($started | Sort-Object Id -Descending)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
