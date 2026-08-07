param(
    [int]$WaitForQueuePid = 56376
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\code\Fiting\fitting'
$Python = 'D:\Anaconda\envs\ML\python.exe'
$RobustnessRoot = 'C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721'
$BudgetRoot = 'C:\code\Fiting\outputs\pmf_cylinder_budget_sensitivity\preregistered_20260721'
$PaperRoot = Join-Path $ProjectRoot 'paper\ieee_superquadric'
$PaperBuildRoot = Join-Path $PaperRoot 'tmp\pdfs\final_automated\build'
$PaperRenderRoot = Join-Path $PaperRoot 'tmp\pdfs\final_automated\rendered'
$PaperOutputRoot = Join-Path $PaperRoot 'output\pdf'
$FinalPdf = Join-Path $PaperOutputRoot 'robust_parametric_surface_fitting.pdf'
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Write-Output "Waiting for experiment queue PID $WaitForQueuePid"
Wait-Process -Id $WaitForQueuePid -ErrorAction SilentlyContinue

Invoke-CheckedPython 'tools\audit_compute_backend.py' '--output' `
    'C:\code\Fiting\outputs\environment\compute_backend_audit.json'
Invoke-CheckedPython 'tools\audit_randomized_superquadric_benchmark.py' '--data-root' `
    'C:\code\superquadic_data\v3_randomized' '--output' `
    'C:\code\Fiting\outputs\benchmark_audits\v3_randomized_audit.json'
Invoke-CheckedPython 'tools\audit_v3_outlier_support.py' '--output' `
    'C:\code\Fiting\outputs\benchmark_audits\v3_outlier20_support_audit.json'
Invoke-CheckedPython 'tools\audit_v3_superquadric_robustness.py' '--robustness-root' $RobustnessRoot '--output' (Join-Path $RobustnessRoot 'summary\strict_external_audit.json')
Invoke-CheckedPython 'tools\audit_cuda_nearest_neighbor_equivalence.py' '--output' `
    'C:\code\Fiting\outputs\environment\cuda_nn_equivalence_audit.json'
Invoke-CheckedPython 'tools\audit_pmf_budget_backend_benchmark.py' '--output' `
    'C:\code\Fiting\outputs\environment\pmf_budget_backend_benchmark_audit.json'
Invoke-CheckedPython 'tools\summarize_pmf_cylinder_budget_sensitivity.py' $BudgetRoot
Invoke-CheckedPython 'tools\write_paper_result_macros.py'
Invoke-CheckedPython 'paper\ieee_superquadric\generate_figures.py'
Invoke-CheckedPython '-m' 'pytest' 'tests' '-q'
Write-Output 'Final strict metric audit, paper macros, and figures complete.'

$env:Path = 'C:\texlive\2026\bin\windows;' + $env:Path
New-Item -ItemType Directory -Force -Path $PaperBuildRoot, $PaperRenderRoot, $PaperOutputRoot | Out-Null
Push-Location $PaperRoot
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error "-outdir=$PaperBuildRoot" main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX build failed ($LASTEXITCODE)"
    }
    $LogFile = Join-Path $PaperBuildRoot 'main.log'
    $BadWarnings = @(Select-String -LiteralPath $LogFile -Pattern @(
        'Overfull',
        'undefined references',
        'Citation .* undefined',
        'Reference .* undefined'
    ))
    if ($BadWarnings.Count -gt 0) {
        throw "Final LaTeX log contains publication-blocking warnings: $($BadWarnings.Line -join ' | ')"
    }
    Copy-Item -LiteralPath (Join-Path $PaperBuildRoot 'main.pdf') -Destination $FinalPdf -Force
    Get-ChildItem -LiteralPath $PaperRenderRoot -Filter 'page-*.png' -ErrorAction SilentlyContinue |
        Remove-Item -Force
    & pdftoppm -png -r 150 $FinalPdf (Join-Path $PaperRenderRoot 'page')
    if ($LASTEXITCODE -ne 0) {
        throw "PDF rendering failed ($LASTEXITCODE)"
    }
    $Info = @(& pdfinfo $FinalPdf)
    if ($LASTEXITCODE -ne 0) {
        throw "pdfinfo failed ($LASTEXITCODE)"
    }
    $PageLine = $Info | Where-Object { $_ -match '^Pages:\s+(\d+)' } | Select-Object -First 1
    if (-not $PageLine -or $PageLine -notmatch '^Pages:\s+(\d+)') {
        throw 'Could not determine final PDF page count'
    }
    $PageCount = [int]$Matches[1]
    $RenderedCount = @(Get-ChildItem -LiteralPath $PaperRenderRoot -Filter 'page-*.png').Count
    if ($RenderedCount -ne $PageCount) {
        throw "Rendered-page mismatch: $RenderedCount PNGs for $PageCount PDF pages"
    }
    Write-Output "Final manuscript compiled to $FinalPdf and rendered as $RenderedCount pages for visual QA."
    Write-Output "After manually inspecting every rendered page, record the PDF-bound QA with:"
    Write-Output "& '$Python' tools\record_pdf_visual_qa.py --confirmed-pages 1-$PageCount"
}
finally {
    Pop-Location
}

Invoke-CheckedPython 'tools\audit_research_completion.py' '--output' `
    'C:\code\Fiting\outputs\research_completion_audit.json'
Write-Output 'Automated experiment-to-manuscript completion audit PASS; manual all-page QA remains required.'
