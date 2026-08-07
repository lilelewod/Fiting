# Codex Server Handoff

Updated: 2026-08-06 (Asia/Shanghai)

## Objective

Continue the reproducible experiments and IEEE-style paper for robust
parametric model fitting.  The current paper studies optimizer choice in the
PMF framework, a guided initialization strategy, and a posed-superquadric
extension.  Do not rewrite the scientific claim as a new optimizer or a new
fitting framework: PSO/CS/CCO and the base PMF framework originate elsewhere.

## Important repository state

- The Windows working tree contains many required modified and untracked files.
- Git commit `f1462e7` alone does **not** contain the current research state.
- Before running on the server, verify that the transferred tree contains at
  least `core/optimizer/pso_fitter.py`, the ModelNet40 scripts listed below,
  `paper/ieee_superquadric/`, and the modified estimator/model files.
- Preserve existing results and use resume mode.  Never delete or overwrite
  completed experiment cells.

## Main paper

- Source: `paper/ieee_superquadric/main.tex`
- Stable PDF: `paper/ieee_superquadric/output/pdf/robust_parametric_surface_fitting.pdf`
- The character example was replaced with audited case r7-t1: it has the
  lowest Guided-PSO case-median Chamfer among the ten character cases; the
  displayed repeat is the median-Chamfer run (2.919) across three seeds.

## Completed core evidence

- Synthetic posed-superquadric extension: 30 cases, five conditions, Guided
  PSO repeated three times per case; strict audit passed.
- EMS baseline: 30 deterministic cases per condition.
- PMF partial-cylinder optimizer and budget experiments are already present.
- Character optimizer-by-guidance factorial: 10 cases, four methods, three
  paired repeats (120 cells), audit passed.
- ModelNet40 dataset preparation audit passed for 10 objects from 7 categories.

## ModelNet40 experiment state

Dataset on the Windows workstation:

`C:/code/datasets/modelnet40/real10_robustness`

Result root on the Windows workstation:

`C:/code/Fiting/outputs/optimizer_comparison/modelnet40_real10_guided_pso_3seeds_20260803`

Current state:

- 40 object-condition cells exist.
- 82 of the target 120 stochastic runs are complete.
- 21 cells have all three seeds.
- 38 runs remain to complete the full 10-object x 4-condition x 3-seed matrix.
- Completed seeds are `20260803`, `20260804`, and `20260805` where present.
- Conditions are `clean`, `noise`, `outlier_20`, and `partial_view`.
- One-seed pilot conclusion: clean/noise/outlier are usable, whereas 40%
  spatial missing is the main failure mode.  Do not treat the 0.05 Chamfer
  screening threshold as exact parameter recovery on non-superquadric objects.

Relevant files:

- `tools/prepare_modelnet40_real_object_robustness.py`
- `tools/audit_modelnet40_real_object_robustness.py`
- `tools/run_modelnet40_real_object_robustness.py`
- `paper/ieee_superquadric/protocols/modelnet40_real_object_10case.json`

## Cross-platform data-root support

The frozen protocol records the original Windows paths for provenance, but the
runner now accepts a portable dataset-root override:

`--data-root /path/to/modelnet40/real10_robustness`

The runner derives each case directory from that root and the protocol's
`case_categories` mapping.  Keep all protocol parameters, seeds, FE budget,
thresholds, and guided-support fractions unchanged.  Run a dry-run on Linux
before starting a long experiment.

## Server verification order

1. Run `nvidia-smi`, `lscpu`, `free -h`, and `df -h`.
2. Verify Python 3.11 and all imports used by the experiment.
3. Run the ModelNet40 data audit against the server data root.
4. Run one clean `bottle_0338` smoke test and compare its independent-reference
   Chamfer with approximately 0.02509 for seed 20260803.
5. Copy the existing result root to the server.
6. Resume the full three-seed matrix; only 38 missing runs should execute.
7. Summarize per-case medians, IQRs, success counts, paired degradation from
   clean, and reproducible failure cases.
8. Generate paper figures only after the result audit passes.

## Compute warning

The formal PSO search is CPU NumPy.  A visible CUDA GPU does not automatically
accelerate it.  The optional CUDA nearest-neighbor path must pass the existing
numerical-equivalence audit and a timing comparison before replacing the CPU
sklearn KDTree path.  Parallelize disjoint experiment cells only after ensuring
separate outputs/manifests cannot overwrite each other.

## First prompt for server Codex

Use this prompt after opening Codex in the repository root:

> Read CODEX_HANDOFF.md completely. Inspect the repository and server hardware.
> Do not start a long experiment yet. First verify that the transferred code,
> ModelNet40 dataset, and existing result root are complete; verify the
> cross-platform --data-root path with the data audit and one dry-run; then
> report the exact command that will resume only the 38 missing runs.
