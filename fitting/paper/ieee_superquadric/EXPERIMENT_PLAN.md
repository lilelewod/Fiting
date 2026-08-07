# Experiment and paper evidence plan

Last updated: 2026-07-21 (Asia/Shanghai)

This file is the decision log for the manuscript. A result enters the paper
only after its protocol, raw records, independent metrics, and paired summary
have been verified. Incomplete-condition statistics are diagnostic only.

## E1. PMF-style partial cylinder: PSO versus original CS

- Status: clean and 50% conditions complete (20 paired seeds each); the raw
  80% batch was stopped after one complete diagnostic pair because both methods
  already failed strongly at 50%. The saved total is 82 records.
- Data: a documented reconstruction of the D5/D6 point counts, not the PMF
  authors' original point clouds. Conditions are clean, 50% uniform-box gross
  outliers, and 80% uniform-box gross outliers. Every corrupted cloud contains
  exactly the same 2,048 clean inliers.
- Design: 20 paired base seeds, population 80, exactly 50,000 objective
  evaluations, fixed seven-dimensional bounds, identical PMF-style MM
  estimator, and independent evaluation against the clean reference.
- Success: clean-reference Chamfer no greater than twice the analytic sampling
  floor **and** F-score at least 0.9. The threshold is fixed across conditions.
- Primary report: median [IQR], success fraction, paired wins, exact two-sided
  Wilcoxon signed-rank test, and median wall time.
- Audit: `tools/audit_pmf_cylinder_experiment.py`; the retained design passes
  with `--allow-incomplete` and exactly recomputes all external metrics. The
  intentionally omitted 19 raw-input pairs at 80% are documented, not missing
  because of selective run failure.
- Interpretation rule: clean superiority is an optimizer-efficiency result.
  Robustness requires retained success under contamination; it cannot be
  inferred from clean performance.

## E2. PMF-cylinder budget sensitivity

- Status: preregistered, not started until E1 completes and passes strict audit.
- Protocol: `protocols/pmf_cylinder_budget_sensitivity.json`.
- Conditions: clean and 50% outliers; five new paired seeds, never used in E1.
- Budgets: 50,000, 199,920, and 499,920 FEs. These are exactly compatible with
  both the 80-evaluation PSO generation and the 160-evaluation CS iteration.
- Decision: if CS closes the clean gap at larger budgets, report PSO primarily
  as more evaluation-efficient. If both methods remain poor with outliers,
  diagnose the estimator/preprocessing rather than relabeling optimizer failure
  as robustness.

## E3. Canonical superquadrics and robustness

- Status: completed evidence available for five canonical clean shapes and
  their 20% outlier variants (five seeds per shape).
- Guided PSO: 25/25 successes on clean and 25/25 on 20% outliers using the
  label-free 8-NN density support (75% retained for the contaminated inputs).
- Required reporting: per-shape Chamfer, success, independent area-uniform
  evaluation, and a clear statement that the support fraction is a method
  setting rather than oracle inlier labels.
- A post-selection audit exactly reproduces the production 8-NN rule on all
  nine formal outlier cases.  Each 3,750-point support contains 3,750 true
  inliers and zero generated outliers (precision 1.0, recall 0.9375).  Labels
  are reconstructed only after label-free selection.  This also confirms that
  the uniform-volume corruption is unusually density-separable and strengthens,
  rather than relaxes, the restriction against general outlier-robustness claims.
- The formal nine-case robustness matrix is running sequentially: controlled
  Gaussian noise, missing-at-random, spatially coherent occlusion, and gross
  outliers. The v3 benchmark now has a
  separately named `occlusion_cap_80.ply`; all 30 cases pass hash,
  cardinality, clean-resolution, and exact regeneration audits.  The audit now
  reconstructs traits, strata, condition seeds, reference, clean, Gaussian
  noise, 20% gross outliers, 80% random missingness, and coherent caps from the
  base seed with zero float32 storage error.  Every gross outlier also satisfies
  its recorded 5%-of-diagonal exclusion distance. Use the same underlying truth
  and reference within each paired condition.
- The 80% random-missing and 80% coherent-cap conditions are observability
  stress tests, not continuations of the 50% volume-outlier experiment. They
  remain required even though unfiltered PMF-cylinder fitting already breaks
  down at 50% gross outliers; random sparsity and structured occlusion answer
  different questions and must be reported separately.

## E4. Randomized stratified superquadrics and EMS reference

- Status: completed for nine independent clean cases spanning smooth/mixed/boxy
  exponents and balanced/anisotropic/extreme aspect ratios.
- Guided PSO: 45 runs, 30/45 successes, run median Chamfer 0.03997.
- EMS: 9/9 successes, median Chamfer 0.02745; EMS wins 8/9 cases and is
  significantly better on per-case paired values (exact Wilcoxon p=0.0078125).
- EMS coherent-cap occlusion reference is complete for the same nine cases:
  6/9 successes, median Chamfer 0.03396, median runtime 0.359 s. Unlike random
  missingness, this exposes failures on some mixed/smooth partial caps.
- Interpretation: this is evidence that the general derivative-free extension
  can fit superquadrics, not evidence of state-of-the-art specialized recovery.
  The specialized EMS baseline is both more accurate and much faster here.
- Failure diagnosis: evaluate axis roles only after allowing proper axis
  permutations. Current hard cases have accurate centers and frames after
  permutation but wrong axis-role/exponent combinations.
- The complete 1%-noise boundary is now independently audited: 45/45 Guided-
  PSO runs and all 45 EMS condition--case records were regenerated from fitted
  parameters, with zero recomputation error in Chamfer, D2M, M2D, and F-score.
  Guided PSO has run-level median Chamfer 0.05000 [0.03202, 0.06514] and 22/45
  successes.  Relative to each case's clean median, the median absolute increase
  is 0.01628 and the median relative increase is 51.35%; the exact paired
  Wilcoxon test over nine cases gives p=0.0390625.  EMS wins all nine case-level
  comparisons (p=0.00390625).  Shape-stratified Guided-PSO success is 14/15
  smooth, 8/15 mixed, and 0/15 boxy, so the main mechanism discussion should
  emphasize shape dependence rather than a single aggregate robustness claim.
- A completed paired case-level check explains the apparently beneficial noise
  result for case 007. All five clean runs select dominant proper permutation
  210, preserve the designated z role in 0/5 runs, and have median shape MAE
  0.4725; all five 1%-noise runs select permutation 102, preserve z in 5/5,
  and reduce shape MAE to 0.0945. Their external Chamfer medians are 0.05104
  and 0.03382. This is recorded as noise perturbing the PCA/hypothesis basin,
  not as evidence that adding noise intrinsically improves recovery. The formal
  summarizer now produces the same symmetry-aware diagnostics for every case
  and condition before any broader mechanism claim is made.
- Chamfer 0.05 remains the preregistered superquadric success threshold. A
  secondary descriptive sensitivity check now reports success counts at 0.04,
  0.05, and 0.06 without changing any primary label or inferential test. The
  manuscript must lead with continuous Chamfer and use this grid only to show
  how much thresholded counts depend on the chosen cutoff.

## E5. Ablations

### Surface-area weighting

- Status: completed screening on box, ellipsoid, and cylinder, five paired seeds.
- Independent audit passes: all 30 external metrics reproduce exactly, every
  run used 5,008 FEs and analytic area-uniform evaluation, and the recorded
  model flag matches its assigned variant.
- Result: mixed. It improves box and ellipsoid medians but not cylinder median;
  therefore it may be described as area-consistent quadrature, not as a
  universally superior recovery strategy.

### Guided initialization and robust support

- Status: guided initialization is implemented. The PMF-cylinder adaptive
  support ablation is complete and strictly audited with five new paired seeds
  per condition; all six arms pass dataset, FE-budget, success-label, and
  independent external-metric checks with zero recomputation error.
- The clean guided-initialization ablation is complete and independently
  audited on three canonical shapes (all external metrics reproduce exactly):
  box improves from 1/5 to 5/5 successes and median Chamfer 0.08184 to 0.02791;
  cylinder improves from 1/5 to 5/5 and 0.07345 to 0.02946; ellipsoid remains
  5/5 with essentially unchanged median Chamfer (0.02006 versus 0.02010).
- A fixed 20% pilot failed at 50% contamination because it discarded too much
  of the cylinder (Chamfer 0.52088). A deterministic two-cluster split of log
  16-NN distances then succeeded in non-formal pilots at both 50% and 80%
  contamination (Chamfer 0.14821 and 0.14886, respectively). Formal seeds do
  not reuse either development-pilot seed.
- Formal 50% results are complete and audited: full input succeeds in 1/5 runs
  with median Chamfer 3.54490, whereas adaptive support succeeds in 5/5 with
  median Chamfer 0.14828. The median paired improvement is 3.39664; with only
  five pairs the exact Wilcoxon p-value is 0.125, so report the large effect and
  success counts without claiming conventional statistical significance.
- The 80% condition is a breakdown-boundary test, not a request for more
  evidence that unfiltered input fails. Five full-input confirmation runs are
  sufficient; the scientifically relevant arm is the five-seed adaptive-support
  arm. Any positive result must be scoped to the preregistered uniform-volume
  outlier process, whose surface/volume density separation is precisely what
  the label-free kNN rule exploits, rather than advertised as arbitrary 80%
  contamination tolerance.
- Formal 80% results confirm the boundary result: full input succeeds in 0/5
  runs with median external Chamfer 4.45124, whereas adaptive support succeeds
  in 5/5 with median 0.14886 and wins every pair. The exact two-sided Wilcoxon
  p-value is 0.0625, the smallest attainable value for five nonzero paired
  differences; report the complete success separation and effect size without
  describing it as significant at the 0.05 level. On clean input, filtering
  retains 5/5 success but incurs a small median accuracy penalty (0.17179 versus
  0.14812), so the method is a robustness--efficiency tradeoff rather than a
  universal preprocessing improvement.
- Under heavy contamination, the record-level `chamfer` field is measured
  against the complete corrupted input and can remain large even for a correct
  recovered surface (the first formal 80% adaptive run has input Chamfer 2.7170
  but independent clean-reference Chamfer 0.14886 and is a success). All paper
  recovery claims and success decisions must use the independently recomputed
  `gt_chamfer`, never this contaminated-input diagnostic.
- Required factors: random versus guided initialization, and full input versus
  density support, with the objective and FE budget held fixed.
- Report interaction effects separately; do not attribute support filtering
  gains to PSO itself.

## Paper claim gate

The current defensible story is:

1. Optimizer choice materially changes evaluation efficiency in the fixed clean
   PMF-style cylinder task; the 20-pair comparison is complete and the
   preregistered budget-sensitivity study remains pending.
2. A general PMF-style search can be extended to posed superquadrics and can be
   robust on controlled canonical contamination with explicit geometric
   initialization/support handling.
3. Generality has a cost: EMS remains superior on the randomized clean
   superquadric benchmark.
4. Area-correct sampling and evaluation are methodological necessities, but
   area weighting alone does not guarantee easier stochastic recovery.

Do not claim universal PSO superiority, reproduction of the PMF authors'
private D5/D6 data, universal outlier robustness, or state-of-the-art
superquadric recovery.

## Paper assembly and verification

- `main.tex` has been restructured around optimizer choice, support-set design,
  and the superquadric extension; the obsolete CCO-centered contribution claim
  has been removed. The current seven-page IEEE draft compiles without overfull
  boxes or undefined references, and all seven rendered pages have been visually
  inspected. Conditional FE-budget and five-condition robustness subsections,
  tables, and figure calls are staged but remain invisible until the strict
  summaries generate their result macros; no incomplete statistic enters the
  visible draft.
- `audit_v3_superquadric_robustness.py` independently regenerates 20,000
  area-uniform reference and fitted points for every PSO and EMS result, checks
  the raw PSO FE counter, and recomputes Chamfer/F-score and success labels. Its
  live partial audit passes with zero metric error; the finalizer will rerun it
  strictly after all 225 PSO runs and 45 EMS fits are available.
- `monitor_v3_robustness_boundaries.ps1` performs the same independent audit
  plus an incomplete-safe paired summary after each 45-run corrupted-condition
  boundary. This is an early-failure gate only; the final strict 225/45 audit
  remains the publication gate.
- The robustness summary now preserves the preregistered shape-exponent and
  aspect-ratio strata. The paper generator will create a descriptive 3-by-5
  success-rate heatmap for each axis only after all 45 runs in every condition
  exist. Each stratum contains three cases, so these panels diagnose failure
  structure and are not presented as additional inferential tests.
- `summarize_pmf_cylinder_budget_sensitivity.py` will require all three audited
  FE roots, preserve the five preregistered seed pairs, compare PSO and CS at
  each budget, and perform within-optimizer paired endpoint tests. The finalizer
  then refreshes LaTeX macros and the budget-sensitivity figure automatically.
- All formal signed-rank tests now use an explicit exact conditional sign
  permutation over Wilcoxon average ranks. Exact-zero differences are removed
  and counted, tied absolute differences retain average ranks, and an all-zero
  comparison reports the degenerate exact value $p=1$. The implementation
  exactly reproduces every existing formal p-value and avoids SciPy's silent
  asymptotic fallback when a future FE-budget pair is unchanged.
- The finalizer now compiles the audited manuscript to
  `output/pdf/robust_parametric_surface_fitting.pdf`, rejects unresolved or
  overfull LaTeX output, and renders exactly one PNG per PDF page. The separate
  `audit_research_completion.py` gate then checks every formal experiment,
  summary, macro, figure, and rendered-page count in one JSON report. It records
  author fields and human visual inspection as pending rather than pretending
  those judgments can be established by an automated file-existence check.
- The completion gate also consumes the freshly rerun 30-case v3 dataset
  audit: all condition cardinalities and stored SHA-256 hashes agree, every
  clean-derived resolution is reproducible, all stochastic conditions regenerate
  at float32 storage precision, and all coherent 80% caps have zero point-set
  error and a strictly positive projection margin. The cylinder audit separately
  retains its angular/height KS checks and radial-error check.
- `audit_compute_backend.py` distinguishes requested configuration from the
  executed kernels. PyTorch sees the RTX 5060 and initializes `cuda:0`, but the
  formal PSO/CS population updates are NumPy, FAISS is unavailable, and both
  mean-measure scoring and external geometry use CPU KDTree queries. Runtime
  values must therefore be described as CPU-path measurements, not GPU results.
- The optional `torch_cuda` nearest-neighbor path passes numerical equivalence
  and a real multiprocessing smoke test.  A paired pre-start 8,080-FE benchmark
  nevertheless selected `sklearn`: CPU KDTree was 1.31--1.61 times faster in
  all clean/outlier-50 by PSO/CS cells while external fitted metrics were
  identical.  The budget protocol was locked to this backend while its formal
  output count was still zero; the decision is a timing gate, not result-based
  algorithm selection.
