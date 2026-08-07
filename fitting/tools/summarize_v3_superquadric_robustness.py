"""Summarize Guided-PSO robustness and the deterministic EMS reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.exact_statistics import exact_wilcoxon_signed_rank
from tools.summarize_stratified_superquadric_pso import symmetry_aware_errors


DIAGNOSTIC_KEYS = (
    "center_error_normalized",
    "axis_frame_error_deg_any_permutation",
    "scale_relative_mae_at_best_frame",
    "shape_mae",
)


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "maximum": float(np.max(values)),
    }


def exact_wilcoxon(differences):
    return exact_wilcoxon_signed_rank(differences)


def summarize_strata(case_strata, axis, labels, condition_pso, condition_ems, case_medians):
    """Return descriptive, case-aware summaries for one preregistered stratum axis."""
    result = {}
    for label in labels:
        cases = sorted(case for case, strata in case_strata.items() if strata[axis] == label)
        pso_rows = [row for row in condition_pso if row["case"] in cases]
        ems_rows = [row for row in condition_ems if row["case"] in cases]
        medians = [case_medians[case] for case in cases if case in case_medians]
        result[label] = {
            "cases": cases,
            "guided_pso_runs": {
                "chamfer": describe([row["gt_chamfer"] for row in pso_rows]) if pso_rows else None,
                "successes": int(sum(row["success"] for row in pso_rows)),
                "runs": len(pso_rows),
            },
            "guided_pso_case_medians": describe(medians) if medians else None,
            "ems_cases": {
                "chamfer": describe([row["gt_chamfer"] for row in ems_rows]) if ems_rows else None,
                "successes": int(sum(row["success"] for row in ems_rows)),
                "cases": len(ems_rows),
            },
        }
    return result


def summarize_diagnostics(rows):
    if not rows:
        return None
    permutations = Counter(row["best_axis_permutation"] for row in rows)
    return {
        **{key: describe([row[key] for row in rows]) for key in DIAGNOSTIC_KEYS},
        "z_role_preserved": int(sum(row["z_role_preserved"] for row in rows)),
        "runs": len(rows),
        "dominant_axis_permutation": permutations.most_common(1)[0][0],
        "axis_permutation_counts": dict(sorted(permutations.items())),
    }


def threshold_sensitivity(rows, primary_threshold):
    thresholds = (0.8 * primary_threshold, primary_threshold, 1.2 * primary_threshold)
    return {
        f"{value:.3f}": int(sum(row["gt_chamfer"] <= value for row in rows))
        for value in thresholds
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/ieee_superquadric/protocols/v3_stratified_superquadric_robustness.json"),
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_clean_guided_pso_1seed_20260716"),
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721"),
    )
    parser.add_argument(
        "--ems-root",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\ems_baseline\v3_randomized_fixedprior01"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    seeds = {int(seed) for seed in protocol["guided_pso"]["paired_base_seeds"]}
    expected_fe = int(protocol["guided_pso"]["max_evaluations"])
    threshold = float(protocol["independent_evaluation"]["chamfer_success_threshold"])
    conditions = list(protocol["conditions"])
    errors = []
    pso_rows = []
    ems_rows = []
    data_root = Path(protocol["data_root"])
    case_strata = {}
    ground_truth = {}
    reuse = protocol.get("reuse_completed", {})
    reused_cases = set(reuse.get("cases", []))
    reused_clean_root = Path(reuse["clean_root"]) if reuse.get("clean_root") else None
    reused_robustness_root = (
        Path(reuse["robustness_root"]) if reuse.get("robustness_root") else None
    )
    for case in protocol["cases"]:
        metadata_file = data_root / case / "metadata.json"
        if not metadata_file.exists():
            errors.append(f"missing case metadata: {case}")
            continue
        strata = json.loads(metadata_file.read_text(encoding="utf-8")).get("strata", {})
        if strata.get("shape") not in {"smooth", "mixed", "boxy"}:
            errors.append(f"invalid shape stratum: {case}")
        if strata.get("aspect") not in {"balanced", "anisotropic", "extreme"}:
            errors.append(f"invalid aspect stratum: {case}")
        case_strata[case] = strata
        trait_document = json.loads((data_root / case / "trait.json").read_text(encoding="utf-8"))
        ground_truth[case] = trait_document.get("trait", trait_document)

    for condition in conditions:
        for case in protocol["cases"]:
            if case in reused_cases and condition == "clean":
                pso_file = reused_clean_root.resolve() / case / "results.json"
            elif case in reused_cases:
                pso_file = (
                    reused_robustness_root.resolve()
                    / condition
                    / case
                    / "results.json"
                )
            elif condition == "clean":
                pso_file = args.clean_root.resolve() / case / "results.json"
                extension_file = (
                    args.clean_root.resolve() / "clean" / case / "results.json"
                )
                if not pso_file.exists() and extension_file.exists():
                    pso_file = extension_file
            else:
                pso_file = args.robustness_root.resolve() / condition / case / "results.json"
            rows = json.loads(pso_file.read_text(encoding="utf-8")) if pso_file.exists() else []
            all_seeds = [int(row["seed"]) for row in rows]
            if len(set(all_seeds)) != len(all_seeds):
                errors.append(f"duplicate Guided-PSO seeds: {condition}/{case}")
            if case not in reused_cases and set(all_seeds) - seeds:
                errors.append(f"unexpected Guided-PSO seeds: {condition}/{case}")
            # Reused nine-case matrices may contain five repeats.  The frozen
            # extension protocol explicitly reuses only its three paired seeds.
            rows = [row for row in rows if int(row["seed"]) in seeds]
            actual_seeds = {int(row["seed"]) for row in rows}
            if actual_seeds != seeds and not args.allow_incomplete:
                errors.append(
                    f"incomplete Guided-PSO cell: {condition}/{case} "
                    f"({len(rows)}/{len(seeds)})"
                )
            for row in rows:
                diagnostics = symmetry_aware_errors(ground_truth[case], row["trait"])
                if row["algorithm"] != "pso" or int(row["evaluations"]) != expected_fe:
                    errors.append(f"optimizer/FE mismatch: {condition}/{case}/{row['seed']}")
                if row.get("evaluation_reference_mode") != "analytic-area-uniform":
                    errors.append(f"nonuniform external evaluation: {condition}/{case}/{row['seed']}")
                if int(row.get("success", -1)) != int(float(row["gt_chamfer"]) <= threshold):
                    errors.append(f"success mismatch: {condition}/{case}/{row['seed']}")
                expected_support = float(
                    protocol["guided_pso"]["initialization_support_fraction"][condition]
                )
                if not np.isclose(float(row["pso_guided_support_fraction"]), expected_support):
                    errors.append(f"initialization support mismatch: {condition}/{case}/{row['seed']}")
                pso_rows.append(
                    {
                        "condition": condition,
                        "case": case,
                        "seed": int(row["seed"]),
                        "gt_chamfer": float(row["gt_chamfer"]),
                        "gt_fscore": float(row["gt_fscore"]),
                        "success": int(row["success"]),
                        "wall_time_s": float(row["wall_time_s"]),
                        **{key: diagnostics[key] for key in DIAGNOSTIC_KEYS},
                        "best_axis_permutation": diagnostics["best_axis_permutation"],
                        "z_role_preserved": diagnostics["z_role_preserved"],
                    }
                )

            ems_condition = protocol["conditions"][condition]["file"].replace(".ply", "")
            ems_dir = args.ems_root.resolve() / case / ems_condition
            evaluation_file = ems_dir / "evaluation.json"
            fit_file = ems_dir / "result.json"
            if not evaluation_file.exists() or not fit_file.exists():
                if not args.allow_incomplete:
                    errors.append(f"missing EMS result: {condition}/{case}")
                continue
            evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
            fit = json.loads(fit_file.read_text(encoding="utf-8"))
            diagnostics = symmetry_aware_errors(ground_truth[case], fit["trait"])
            if evaluation.get("reference_mode") != "analytic-area-uniform":
                errors.append(f"nonuniform EMS evaluation: {condition}/{case}")
            if int(evaluation["evaluation_points"]) != int(
                protocol["independent_evaluation"]["reference_points"]
            ):
                errors.append(f"EMS evaluation-point mismatch: {condition}/{case}")
            ems_rows.append(
                {
                    "condition": condition,
                    "case": case,
                    "gt_chamfer": float(evaluation["gt_chamfer"]),
                    "gt_fscore": float(evaluation["gt_fscore"]),
                    "success": int(float(evaluation["gt_chamfer"]) <= threshold),
                    "wall_time_s": float(fit["wall_time_s"]),
                    **{key: diagnostics[key] for key in DIAGNOSTIC_KEYS},
                    "best_axis_permutation": diagnostics["best_axis_permutation"],
                    "z_role_preserved": diagnostics["z_role_preserved"],
                }
            )

    clean_case_medians = {}
    for case in protocol["cases"]:
        values = [row["gt_chamfer"] for row in pso_rows if row["condition"] == "clean" and row["case"] == case]
        if values:
            clean_case_medians[case] = float(np.median(values))

    report = {"status": "PASS", "conditions": {}, "errors": sorted(set(errors))}
    for condition in conditions:
        condition_pso = [row for row in pso_rows if row["condition"] == condition]
        condition_ems = [row for row in ems_rows if row["condition"] == condition]
        case_medians = {}
        for case in protocol["cases"]:
            values = [row["gt_chamfer"] for row in condition_pso if row["case"] == case]
            if values:
                case_medians[case] = float(np.median(values))
        paired_cases = sorted(set(case_medians) & {row["case"] for row in condition_ems})
        ems_by_case = {row["case"]: row for row in condition_ems}
        pso_minus_ems = [case_medians[case] - ems_by_case[case]["gt_chamfer"] for case in paired_cases]
        case_diagnostics = {}
        for case in protocol["cases"]:
            rows = [row for row in condition_pso if row["case"] == case]
            if rows:
                case_diagnostics[case] = {
                    "chamfer": describe([row["gt_chamfer"] for row in rows]),
                    "fscore": describe([row["gt_fscore"] for row in rows]),
                    "successes": int(sum(row["success"] for row in rows)),
                    "runs": len(rows),
                    "success_threshold_sensitivity": threshold_sensitivity(
                        rows, threshold
                    ),
                    **summarize_diagnostics(rows),
                }
        condition_report = {
            "guided_pso_runs": {
                "chamfer": describe([row["gt_chamfer"] for row in condition_pso]) if condition_pso else None,
                "fscore": describe([row["gt_fscore"] for row in condition_pso]) if condition_pso else None,
                "successes": int(sum(row["success"] for row in condition_pso)),
                "runs": len(condition_pso),
                "runtime_s": describe([row["wall_time_s"] for row in condition_pso]) if condition_pso else None,
                "success_threshold_sensitivity": threshold_sensitivity(
                    condition_pso, threshold
                ),
            },
            "guided_pso_case_medians": describe(list(case_medians.values())) if case_medians else None,
            "guided_pso_diagnostics": summarize_diagnostics(condition_pso),
            "guided_pso_case_diagnostics": case_diagnostics,
            "ems_cases": {
                "chamfer": describe([row["gt_chamfer"] for row in condition_ems]) if condition_ems else None,
                "successes": int(sum(row["success"] for row in condition_ems)),
                "cases": len(condition_ems),
                "runtime_s": describe([row["wall_time_s"] for row in condition_ems]) if condition_ems else None,
                "success_threshold_sensitivity": threshold_sensitivity(
                    condition_ems, threshold
                ),
                "diagnostics": summarize_diagnostics(condition_ems),
            },
            "pso_median_vs_ems": {
                "paired_cases": paired_cases,
                "pso_wins": int(sum(case_medians[c] < ems_by_case[c]["gt_chamfer"] for c in paired_cases)),
                "ems_wins": int(sum(ems_by_case[c]["gt_chamfer"] < case_medians[c] for c in paired_cases)),
                "median_pso_minus_ems": float(np.median(pso_minus_ems)) if pso_minus_ems else None,
                "wilcoxon": exact_wilcoxon(pso_minus_ems),
            },
            "strata": {
                "shape": summarize_strata(
                    case_strata,
                    "shape",
                    ("smooth", "mixed", "boxy"),
                    condition_pso,
                    condition_ems,
                    case_medians,
                ),
                "aspect": summarize_strata(
                    case_strata,
                    "aspect",
                    ("balanced", "anisotropic", "extreme"),
                    condition_pso,
                    condition_ems,
                    case_medians,
                ),
            },
        }
        if condition != "clean":
            robustness_cases = sorted(set(case_medians) & set(clean_case_medians))
            degradation = [case_medians[c] - clean_case_medians[c] for c in robustness_cases]
            relative = [
                (case_medians[c] - clean_case_medians[c]) / clean_case_medians[c]
                for c in robustness_cases
            ]
            condition_report["paired_to_clean"] = {
                "cases": robustness_cases,
                "chamfer_difference": describe(degradation) if degradation else None,
                "relative_degradation": describe(relative) if relative else None,
                "wilcoxon": exact_wilcoxon(degradation),
            }
        report["conditions"][condition] = condition_report

    if errors:
        report["status"] = "FAIL"
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_root.resolve() / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if pso_rows:
        with (args.output_root.resolve() / "pso_rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pso_rows[0]))
            writer.writeheader()
            writer.writerows(pso_rows)
    if ems_rows:
        with (args.output_root.resolve() / "ems_rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ems_rows[0]))
            writer.writeheader()
            writer.writerows(ems_rows)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
