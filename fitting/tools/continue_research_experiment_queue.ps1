param(
    [int]$WaitForDensitySupportPid = 68748
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\code\Fiting\fitting'
$Python = 'D:\Anaconda\envs\ML\python.exe'
$OutputRoot = 'C:\code\Fiting\outputs'
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Write-Output "Waiting for density-support runner PID $WaitForDensitySupportPid"
Wait-Process -Id $WaitForDensitySupportPid -ErrorAction SilentlyContinue

$DensityRoot = Join-Path $OutputRoot 'pmf_cylinder_density_support\formal_adaptive_20260721'
Invoke-CheckedPython 'tools\summarize_pmf_cylinder_density_support_ablation.py' $DensityRoot
foreach ($condition in @('clean', 'outlier_50', 'outlier_80')) {
    foreach ($variant in @('full_input', 'adaptive_density')) {
        Invoke-CheckedPython 'tools\audit_pmf_cylinder_experiment.py' (Join-Path $DensityRoot "$condition\$variant")
    }
}
Write-Output 'Density-support ablation complete and audited.'

$RobustnessRoot = Join-Path $OutputRoot 'optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721'
Invoke-CheckedPython 'tools\audit_compute_backend.py' '--output' `
    (Join-Path $OutputRoot 'environment\compute_backend_audit.json')
Invoke-CheckedPython 'tools\audit_randomized_superquadric_benchmark.py' '--data-root' `
    'C:\code\superquadic_data\v3_randomized' '--output' `
    (Join-Path $OutputRoot 'benchmark_audits\v3_randomized_audit.json')
Invoke-CheckedPython 'tools\audit_v3_outlier_support.py' '--output' `
    (Join-Path $OutputRoot 'benchmark_audits\v3_outlier20_support_audit.json')
Write-Output 'Superquadric sampling and label-free support selection audited.'
Invoke-CheckedPython 'tools\run_v3_stratified_superquadric_robustness.py' '--output-root' $RobustnessRoot
Invoke-CheckedPython 'tools\summarize_v3_superquadric_robustness.py' '--robustness-root' $RobustnessRoot '--output-root' (Join-Path $RobustnessRoot 'summary')
Invoke-CheckedPython 'tools\audit_v3_superquadric_robustness.py' '--robustness-root' $RobustnessRoot '--output' (Join-Path $RobustnessRoot 'summary\strict_external_audit.json')
Write-Output 'Superquadric robustness matrix complete and audited.'

$BudgetRoot = Join-Path $OutputRoot 'pmf_cylinder_budget_sensitivity\preregistered_20260721'
Invoke-CheckedPython 'tools\audit_cuda_nearest_neighbor_equivalence.py' '--output' `
    (Join-Path $OutputRoot 'environment\cuda_nn_equivalence_audit.json')
Invoke-CheckedPython 'tools\audit_pmf_budget_backend_benchmark.py' '--output' `
    (Join-Path $OutputRoot 'environment\pmf_budget_backend_benchmark_audit.json')
Write-Output 'CPU/CUDA equivalence and paired end-to-end backend-selection gates PASS.'
Invoke-CheckedPython 'tools\run_pmf_cylinder_budget_sensitivity.py' '--output-root' $BudgetRoot
foreach ($budget in @(50000, 199920, 499920)) {
    Invoke-CheckedPython 'tools\audit_pmf_cylinder_experiment.py' (Join-Path $BudgetRoot "fe_$budget")
}
Invoke-CheckedPython 'tools\summarize_pmf_cylinder_budget_sensitivity.py' $BudgetRoot
Write-Output 'Budget-sensitivity experiment complete and audited.'

Invoke-CheckedPython 'tools\write_paper_result_macros.py'
Invoke-CheckedPython 'paper\ieee_superquadric\generate_figures.py'
Write-Output 'Audited paper macros and reproducible figures refreshed.'
