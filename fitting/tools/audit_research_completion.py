"""Audit the complete experiment-to-manuscript evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT.parent / "outputs"
PAPER = PROJECT_ROOT / "paper/ieee_superquadric"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def all_zero(mapping):
    return all(float(value) == 0.0 for value in mapping.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "research_completion_audit.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    errors = []
    missing = []
    pending_manuscript = []
    checks = {}

    def require_json(path, label):
        if not path.exists():
            missing.append(f"{label}: {path}")
            return None
        try:
            return load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"unreadable {label}: {exc}")
            return None

    def require_pass(report, label):
        if report is None:
            return False
        if report.get("status") != "PASS" or report.get("errors"):
            errors.append(f"{label} is not a clean PASS")
            return False
        return True

    main_root = OUTPUTS / "pmf_cylinder_comparison/pso_cs_formal20_20260722"
    report = require_json(main_root / "audit.json", "PMF main audit")
    if require_pass(report, "PMF main audit"):
        paired = report["paired_summary"]
        if paired["clean"]["paired_runs"] != 20 or paired["outlier_50"]["paired_runs"] != 20:
            errors.append("PMF main clean/outlier-50 cells are not 20 paired runs")
        if report["completed_results"] != 82 or paired["outlier_80"]["paired_runs"] != 1:
            errors.append("PMF intentionally truncated 80% diagnostic is not the documented 82-record design")
        if not all_zero(report["external_metric_recompute_max_abs_error"]):
            errors.append("PMF main external metrics do not reproduce exactly")
        uniformity = report["clean_sampling_uniformity"]
        if (
            uniformity["angle_ks_p"] < 1e-3
            or uniformity["height_ks_p"] < 1e-3
            or abs(float(uniformity["radial_max_abs_error"])) > 1e-12
        ):
            errors.append("PMF cylinder sampling audit failed")
        checks["pmf_main"] = {"records": report["completed_results"], "status": "PASS"}

    benchmark_audit = require_json(
        OUTPUTS / "benchmark_audits/v3_randomized_audit.json",
        "v3 randomized benchmark audit",
    )
    if require_pass(benchmark_audit, "v3 randomized benchmark audit"):
        sampling_audit = benchmark_audit.get("sampling_audit", {})
        if benchmark_audit["case_count"] != 30 or not all(
            sampling_audit.get(key) is True
            for key in (
                "trait_strata_and_seed_regeneration",
                "reference_clean_noise_outlier_and_random_missing_regeneration",
            )
        ):
            errors.append("v3 benchmark audit does not prove all 30 cases and sampling mechanisms")
        expected_counts = {
            "reference_uniform.ply": 20000,
            "clean.ply": 5000,
            "noise_1pct_diag.ply": 5000,
            "outlier_20.ply": 5000,
            "missing_80.ply": 1000,
            "occlusion_cap_80.ply": 1000,
        }
        for case in benchmark_audit["cases"]:
            if case["counts"] != expected_counts:
                errors.append(f'v3 benchmark cardinality mismatch: {case["case"]}')
            if any(
                float(value) != 0.0
                for value in case.get("deterministic_regeneration_max_abs_error", {}).values()
            ):
                errors.append(f'v3 deterministic condition does not regenerate exactly: {case["case"]}')
            if not case.get("deterministic_regeneration_max_abs_error"):
                errors.append(f'v3 deterministic condition regeneration is missing: {case["case"]}')
            if float(case.get("gross_outlier_minimum_distance", -1.0)) + 1e-12 < float(
                case.get("gross_outlier_required_minimum_distance", 0.0)
            ):
                errors.append(f'v3 gross-outlier exclusion distance failed: {case["case"]}')
            if float(case["occlusion_regeneration_max_distance"]) != 0.0:
                errors.append(f'v3 coherent occlusion does not regenerate exactly: {case["case"]}')
            if float(case["occlusion_projection_margin"]) < -1e-12:
                errors.append(f'v3 coherent occlusion is not a projection cap: {case["case"]}')
        checks["v3_dataset"] = {"cases": 30, "status": "PASS"}

    support_selection = require_json(
        OUTPUTS / "benchmark_audits/v3_outlier20_support_audit.json",
        "v3 outlier-support audit",
    )
    if require_pass(support_selection, "v3 outlier-support audit"):
        if (
            support_selection.get("cases_audited") != 9
            or support_selection.get("support_rule", {}).get("label_free_at_fit_time") is not True
            or float(support_selection.get("minimum_inlier_precision", -1.0)) != 1.0
            or float(support_selection.get("minimum_inlier_recall", -1.0)) != 0.9375
            or int(support_selection.get("maximum_retained_outliers", -1)) != 0
        ):
            errors.append("v3 outlier-support audit does not match the preregistered nine-case design")
        if any(
            float(case["cloud_regeneration_max_abs_error"]) != 0.0
            or float(case["production_selection_max_abs_error"]) != 0.0
            for case in support_selection.get("cases", [])
        ):
            errors.append("v3 outlier-support audit does not reproduce production selection exactly")
        checks["v3_outlier_support"] = {"cases": 9, "status": "PASS"}

    backend = require_json(
        OUTPUTS / "environment/compute_backend_audit.json",
        "formal compute-backend audit",
    )
    if require_pass(backend, "formal compute-backend audit"):
        execution = backend.get("formal_execution_path", {})
        if (
            execution.get("pso_search_backend") != "CPU NumPy"
            or execution.get("pso_tensor_compute_detected") is not False
            or execution.get("faiss_gpu_path_detected") is not False
            or not str(execution.get("mean_measure_nearest_neighbor_backend", "")).startswith("CPU ")
        ):
            errors.append("formal compute backend is not the audited CPU execution path")
        checks["compute_backend"] = {
            "pso": execution.get("pso_search_backend"),
            "cuda_visible": bool(backend.get("torch_cuda_available")),
            "status": "PASS",
        }

    cuda_equivalence = require_json(
        OUTPUTS / "environment/cuda_nn_equivalence_audit.json",
        "CUDA nearest-neighbor equivalence audit",
    )
    if require_pass(cuda_equivalence, "CUDA nearest-neighbor equivalence audit"):
        if any(
            float(cell["maximum_mean_distance_abs_error"])
            > float(cuda_equivalence["mean_distance_tolerance"])
            for cell in cuda_equivalence.get("conditions", {}).values()
        ):
            errors.append("CUDA nearest-neighbor mean-distance equivalence failed")
        checks["cuda_nearest_neighbor"] = {
            "conditions": len(cuda_equivalence.get("conditions", {})),
            "status": "PASS",
        }

    backend_benchmark = require_json(
        OUTPUTS / "environment/pmf_budget_backend_benchmark_audit.json",
        "PMF budget backend benchmark audit",
    )
    if require_pass(backend_benchmark, "PMF budget backend benchmark audit"):
        if backend_benchmark.get("selected_backend") != "sklearn":
            errors.append("PMF budget backend timing gate did not select sklearn")
        cells = backend_benchmark.get("cells", {})
        if set(cells) != {"clean", "outlier_50"} or any(
            set(condition_cells) != {"pso", "cs"}
            for condition_cells in cells.values()
        ):
            errors.append("PMF budget backend benchmark matrix is incomplete")
        else:
            for condition, condition_cells in cells.items():
                for algorithm, cell in condition_cells.items():
                    if float(cell.get("cpu_speedup_over_cuda", 0.0)) <= 1.0:
                        errors.append(
                            f"PMF backend selection not supported by timing: {condition}/{algorithm}"
                        )
                    metric_errors = cell.get("metric_abs_errors", {})
                    if (
                        float(metric_errors.get("best_score", float("inf")))
                        > float(backend_benchmark["internal_score_tolerance"])
                        or float(metric_errors.get("gt_chamfer", float("inf")))
                        > float(backend_benchmark["external_metric_tolerance"])
                        or float(metric_errors.get("gt_fscore", float("inf")))
                        > float(backend_benchmark["external_metric_tolerance"])
                    ):
                        errors.append(
                            f"PMF backend equivalence mismatch: {condition}/{algorithm}"
                        )
        checks["pmf_budget_backend"] = {
            "cells": sum(len(value) for value in cells.values()),
            "selected": backend_benchmark.get("selected_backend"),
            "status": "PASS",
        }

    support_root = OUTPUTS / "pmf_cylinder_density_support/formal_adaptive_20260721"
    support = require_json(support_root / "ablation_summary.json", "support summary")
    if require_pass(support, "support summary"):
        for condition in ("clean", "outlier_50", "outlier_80"):
            cell = support["conditions"][condition]
            if len(cell["paired_seeds"]) != 5:
                errors.append(f"support/{condition} does not contain five paired seeds")
            for variant in ("full_input", "adaptive_density"):
                if cell["variants"][variant]["completed_runs"] != 5:
                    errors.append(f"support/{condition}/{variant} is incomplete")
                audit = require_json(
                    support_root / condition / variant / "audit.json",
                    f"support audit {condition}/{variant}",
                )
                if require_pass(audit, f"support audit {condition}/{variant}"):
                    if audit["completed_results"] != 5 or not all_zero(
                        audit["external_metric_recompute_max_abs_error"]
                    ):
                        errors.append(f"support audit mismatch: {condition}/{variant}")
        checks["support_ablation"] = {"cells": 6, "status": "PASS"}

    for label, relative in (
        (
            "area_weighting",
            "area_weight_ablation/formal_v2_pso_clean_48x48_5008fe_5seeds/audit.json",
        ),
        (
            "guided_initialization",
            "optimizer_comparison/guided_initialization_ablation_summary/audit.json",
        ),
    ):
        audit = require_json(OUTPUTS / relative, f"{label} audit")
        if require_pass(audit, f"{label} audit"):
            if set(audit["shapes"]) != {"box", "cylinder", "ellipsoid"}:
                errors.append(f"{label} does not contain all three preregistered shapes")
            for shape, cell in audit["shapes"].items():
                pair_count = cell.get("paired_runs", len(cell.get("paired_seeds", [])))
                if pair_count != 5 or float(cell["external_metric_recompute_max_abs_error"]) != 0.0:
                    errors.append(f"{label}/{shape} audit mismatch")
            checks[label] = {"shapes": 3, "status": "PASS"}

    clean_summary = require_json(
        OUTPUTS
        / "optimizer_comparison/v3_stratified9_clean_guided_pso_1seed_20260716/summary_5seeds/summary.json",
        "clean stratified superquadric summary",
    )
    if clean_summary is not None:
        if (
            clean_summary["protocol"]["independent_cases"] != 9
            or clean_summary["pso_all_runs"]["chamfer"]["count"] != 45
            or clean_summary["ems_cases"]["chamfer"]["count"] != 9
        ):
            errors.append("clean stratified superquadric evidence is incomplete")
        checks["superquadric_clean"] = {"pso_runs": 45, "ems_cases": 9, "status": "PASS"}

    robust_root = OUTPUTS / "optimizer_comparison/v3_randomized30_guided_pso_3seeds_20260727"
    robust = require_json(
        robust_root / "summary_30cases_3seeds/summary.json",
        "formal 30-case robustness summary",
    )
    robust_audit = require_json(
        robust_root / "summary_30cases_3seeds/strict_external_audit.json",
        "formal 30-case robustness metric audit",
    )
    if require_pass(robust, "formal robustness summary"):
        for condition in (
            "clean",
            "noise_1pct_diag",
            "outlier_20",
            "missing_80",
            "occlusion_cap_80",
        ):
            cell = robust["conditions"][condition]
            if cell["guided_pso_runs"]["runs"] != 90 or cell["ems_cases"]["cases"] != 30:
                errors.append(f"formal robustness cell is incomplete: {condition}")
            for method_key, success_key in (
                ("guided_pso_runs", "successes"),
                ("ems_cases", "successes"),
            ):
                sensitivity = cell[method_key].get("success_threshold_sensitivity", {})
                if set(sensitivity) != {"0.040", "0.050", "0.060"}:
                    errors.append(f"threshold-sensitivity grid missing: {condition}/{method_key}")
                elif sensitivity["0.050"] != cell[method_key][success_key]:
                    errors.append(f"primary threshold count mismatch: {condition}/{method_key}")
            for axis in ("shape", "aspect"):
                groups = cell.get("strata", {}).get(axis, {})
                group_runs = [
                    group["guided_pso_runs"]["runs"] for group in groups.values()
                ]
                if (
                    len(groups) != 3
                    or any(runs <= 0 for runs in group_runs)
                    or sum(group_runs) != 90
                ):
                    errors.append(f"formal robustness stratum mismatch: {condition}/{axis}")
        checks["superquadric_robustness_summary"] = {"conditions": 5, "status": "PASS"}
    if require_pass(robust_audit, "formal robustness metric audit"):
        if (
            robust_audit["pso_results_audited"] != 450
            or robust_audit["ems_results_audited"] != 150
            or not all_zero(robust_audit["external_metric_recompute_max_abs_error"])
        ):
            errors.append("formal robustness strict audit counts or recomputation are wrong")
        checks["superquadric_strict_metrics"] = {"pso": 450, "ems": 150, "status": "PASS"}

    occlusion_bounds = require_json(
        OUTPUTS / "environment/occlusion_search_feasibility_audit.json",
        "occlusion search-feasibility audit",
    )
    missing_bounds = require_json(
        OUTPUTS / "environment/missing_search_feasibility_control.json",
        "random-missing search-feasibility control",
    )
    if require_pass(occlusion_bounds, "occlusion search-feasibility audit"):
        expected = {
            "cases": 9,
            "center_feasible_cases": 5,
            "center_infeasible_cases": 4,
            "scale_feasible_cases": 9,
            "scale_infeasible_cases": 0,
        }
        if any(occlusion_bounds.get(key) != value for key, value in expected.items()):
            errors.append("occlusion search-feasibility counts differ from the manuscript diagnosis")
        elif robust is not None and robust["conditions"]["occlusion_cap_80"][
            "guided_pso_runs"
        ]["successes"] != 0:
            errors.append("occlusion feasibility diagnosis no longer matches the recovery matrix")
        else:
            checks["occlusion_search_feasibility"] = {**expected, "status": "PASS"}
    if require_pass(missing_bounds, "random-missing search-feasibility control"):
        if (
            missing_bounds.get("cases") != 9
            or missing_bounds.get("center_feasible_cases") != 9
            or missing_bounds.get("scale_feasible_cases") != 9
        ):
            errors.append("random-missing feasibility control is not 9/9 for center and scale")
        else:
            checks["random_missing_search_feasibility"] = {
                "center_feasible_cases": 9,
                "scale_feasible_cases": 9,
                "status": "PASS",
            }

    budget_root = OUTPUTS / "pmf_cylinder_budget_sensitivity/preregistered_20260721"
    budget_registration = require_json(
        PAPER / "protocols/pmf_cylinder_budget_sensitivity.json",
        "preregistered budget protocol",
    )
    budget = require_json(budget_root / "summary.json", "budget summary")
    budget_errors_before = len(errors)
    budget_missing_before = len(missing)
    if require_pass(budget, "budget summary"):
        if budget.get("nearest_neighbor_backend") != "sklearn":
            errors.append("budget study did not use the preregistered sklearn backend")
        for evaluation_budget in (50000, 199920, 499920):
            key = str(evaluation_budget)
            if key not in budget["budgets"]:
                errors.append(f"missing budget summary cell: {evaluation_budget}")
                continue
            for condition in ("clean", "outlier_50"):
                if budget["budgets"][key]["conditions"][condition]["paired_runs"] != 5:
                    errors.append(f"budget paired cell incomplete: {evaluation_budget}/{condition}")
            audit = require_json(
                budget_root / f"fe_{evaluation_budget}/audit.json",
                f"budget audit {evaluation_budget}",
            )
            run_protocol = require_json(
                budget_root / f"fe_{evaluation_budget}/protocol.json",
                f"budget run protocol {evaluation_budget}",
            )
            if budget_registration is not None and run_protocol is not None:
                expected_protocol_fields = {
                    "conditions": list(budget_registration["dataset"]["conditions"]),
                    "algorithms": list(budget_registration["algorithms"]),
                    "base_seeds": [
                        int(value) for value in budget_registration["paired_base_seeds"]
                    ],
                    "population_size": int(budget_registration["population_size"]),
                    "num_envs": int(budget_registration["num_envs"]),
                    "max_evaluations": evaluation_budget,
                    "nearest_neighbor_backend": budget_registration[
                        "nearest_neighbor_backend"
                    ],
                    "paired_internal_seeds": True,
                    "model_evaluation_grid": list(
                        budget_registration["evaluation"]["model_grid"]
                    ),
                }
                mismatched = [
                    field
                    for field, expected in expected_protocol_fields.items()
                    if run_protocol.get(field) != expected
                ]
                if mismatched:
                    errors.append(
                        f"budget protocol differs from preregistration at "
                        f"{evaluation_budget}: {', '.join(mismatched)}"
                    )
            if require_pass(audit, f"budget audit {evaluation_budget}"):
                if audit["completed_results"] != 20 or not all_zero(
                    audit["external_metric_recompute_max_abs_error"]
                ):
                    errors.append(f"budget audit mismatch: {evaluation_budget}")
        budget_complete = (
            len(errors) == budget_errors_before and len(missing) == budget_missing_before
        )
        checks["pmf_budget_sensitivity"] = {
            "budgets": 3,
            "status": "PASS" if budget_complete else "INCOMPLETE",
        }

    macro_file = PAPER / "results_auto.tex"
    if macro_file.exists():
        macros = macro_file.read_text(encoding="utf-8")
        manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
        result_macro_pattern = re.compile(
            r"\\(?:PMF|Support|Area|Init|SQ|EMS|Budget)[A-Za-z]+"
        )
        required_macros = set(result_macro_pattern.findall(manuscript))
        provided_macros = set(
            re.findall(
                r"\\providecommand\{(\\(?:PMF|Support|Area|Init|SQ|EMS|Budget)[A-Za-z]+)\}",
                macros,
            )
        )
        absent = sorted(required_macros - provided_macros)
        if absent:
            missing.append("formal LaTeX macros: " + ", ".join(absent))
        else:
            checks["latex_result_macros"] = {
                "referenced": len(required_macros),
                "provided": len(provided_macros),
                "status": "PASS",
            }
    else:
        missing.append(f"LaTeX macro file: {macro_file}")

    for figure in (
        "superquadric_robustness.pdf",
        "superquadric_strata.pdf",
        "pmf_budget_sensitivity.pdf",
    ):
        path = PAPER / "figures" / figure
        if not path.exists() or path.stat().st_size == 0:
            missing.append(f"formal figure: {path}")

    final_pdf = PAPER / "output/pdf/robust_parametric_surface_fitting.pdf"
    build_log = PAPER / "tmp/pdfs/final_automated/build/main.log"
    render_root = PAPER / "tmp/pdfs/final_automated/rendered"
    if final_pdf.exists() and build_log.exists():
        page_count = len(PdfReader(str(final_pdf)).pages)
        rendered = len(list(render_root.glob("page-*.png")))
        bad_log = re.findall(
            r"Overfull|undefined references|Citation .* undefined|Reference .* undefined",
            build_log.read_text(encoding="utf-8", errors="replace"),
        )
        pdf_gate_pass = page_count >= 1 and rendered == page_count and not bad_log
        if not pdf_gate_pass:
            errors.append("final PDF build/render gate failed")
        checks["final_pdf"] = {
            "pages": page_count,
            "rendered_pages": rendered,
            "blocking_log_matches": len(bad_log),
            "status": "PASS" if pdf_gate_pass else "FAIL",
        }
        qa_file = PAPER / "output/pdf/visual_qa.json"
        if not qa_file.exists():
            pending_manuscript.append("manual all-page visual QA has not been recorded")
        else:
            qa = require_json(qa_file, "visual QA marker")
            digest = hashlib.sha256(final_pdf.read_bytes()).hexdigest()
            if qa is None or qa.get("status") != "PASS" or qa.get("pdf_sha256") != digest:
                errors.append("visual QA marker does not match the final PDF")
    else:
        missing.append("final compiled PDF and build log")

    reproduction_guide = PAPER / "README_zh.md"
    if not reproduction_guide.exists():
        missing.append(f"Chinese reproduction guide: {reproduction_guide}")
    else:
        guide = reproduction_guide.read_text(encoding="utf-8")
        required_guide_tokens = (
            "run_v3_stratified_superquadric_robustness.py",
            "audit_v3_superquadric_robustness.py",
            "run_pmf_cylinder_budget_sensitivity.py",
            "audit_partial_progress.json",
            "INCOMPLETE",
            "runtime_diagnosis_zh.md",
            "record_pdf_visual_qa.py",
            "resume_research_experiment_queue.ps1",
            "scheduled_pause_manifest.json",
            "PAUSED_AT_COMPLETE_RESULT_BOUNDARY",
            "audit_research_completion.py",
            "随机缺失不能写成遮挡",
        )
        absent = [token for token in required_guide_tokens if token not in guide]
        if absent:
            errors.append("reproduction guide misses required protocol commands or claim limits")
        if "\ufffd" in guide or "CCO" in guide:
            errors.append("reproduction guide contains mojibake or obsolete CCO framing")
        if not absent and "\ufffd" not in guide and "CCO" not in guide:
            checks["reproduction_guide"] = {"language": "zh-CN", "status": "PASS"}

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "Author Name" in main_tex or "University Name" in main_tex:
        pending_manuscript.append("author and affiliation placeholders remain")

    automated_status = "PASS" if not errors and not missing else "INCOMPLETE" if not errors else "FAIL"
    completion_status = (
        "PASS"
        if automated_status == "PASS" and not pending_manuscript
        else "PENDING_MANUSCRIPT_QA"
        if automated_status == "PASS"
        else automated_status
    )
    result = {
        "automated_status": automated_status,
        "completion_status": completion_status,
        "checks": checks,
        "missing": sorted(set(missing)),
        "pending_manuscript": sorted(set(pending_manuscript)),
        "errors": sorted(set(errors)),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors or (missing and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
