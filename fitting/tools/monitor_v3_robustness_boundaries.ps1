param(
    [int]$RunnerPid = 72992,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\code\Fiting\fitting'
$Python = 'D:\Anaconda\envs\ML\python.exe'
$RobustnessRoot = 'C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721'
$LiveRoot = 'C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_live_audit'
$Boundaries = @(45, 90, 135, 180)
$Audited = @{}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LiveRoot | Out-Null

function Get-CorruptedResultCount {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $count = 0
        try {
            Get-ChildItem -LiteralPath $RobustnessRoot -Recurse -Filter results.json -ErrorAction SilentlyContinue |
                ForEach-Object {
                    # ConvertFrom-Json keeps a top-level JSON array as one
                    # pipeline object in Windows PowerShell.  Use the array's
                    # own Count property.  Retry the entire snapshot if a file
                    # is observed while its writer is replacing it.
                    $rows = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                    $count += $rows.Count
                }
            return $count
        }
        catch {
            if ($attempt -eq 5) { throw }
            Start-Sleep -Seconds 1
        }
    }
}

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

foreach ($boundary in $Boundaries) {
    $auditFile = Join-Path $LiveRoot "strict_external_audit_after_$boundary.json"
    $summaryFile = Join-Path $LiveRoot "summary_after_$boundary\summary.json"
    if ((Test-Path -LiteralPath $auditFile) -and (Test-Path -LiteralPath $summaryFile)) {
        try {
            $audit = Get-Content -LiteralPath $auditFile -Raw | ConvertFrom-Json
            $summary = Get-Content -LiteralPath $summaryFile -Raw | ConvertFrom-Json
            if ($audit.status -eq 'PASS' -and $summary.status -eq 'PASS') {
                $Audited[$boundary] = $true
                Write-Output "Reusing prior PASS boundary audit at $boundary/180."
            }
        }
        catch {
            Write-Output "Prior boundary artifacts at $boundary/180 are unreadable; they will be regenerated."
        }
    }
}

Write-Output "Monitoring robustness runner PID $RunnerPid at $PollSeconds-second intervals."
while ($true) {
    $count = Get-CorruptedResultCount
    foreach ($boundary in $Boundaries) {
        if ($count -ge $boundary -and -not $Audited.ContainsKey($boundary)) {
            Write-Output "Boundary $boundary/180 reached; running independent audit and partial summary."
            $auditFile = Join-Path $LiveRoot "strict_external_audit_after_$boundary.json"
            $summaryRoot = Join-Path $LiveRoot "summary_after_$boundary"
            Invoke-CheckedPython 'tools\audit_v3_superquadric_robustness.py' `
                '--robustness-root' $RobustnessRoot '--output' $auditFile '--allow-incomplete'
            Invoke-CheckedPython 'tools\summarize_v3_superquadric_robustness.py' `
                '--robustness-root' $RobustnessRoot '--output-root' $summaryRoot '--allow-incomplete'
            $Audited[$boundary] = $true
            Write-Output "Boundary $boundary/180 audit PASS."
        }
    }
    if (-not (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue)) {
        Write-Output "Robustness runner PID $RunnerPid has exited; monitor stopping at $count/180."
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
