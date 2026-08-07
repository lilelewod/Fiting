"""Audit the paired superquadric surface-area weighting ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.superquadric_evaluation import geometric_metrics, load_trait, sample_trait, trait_from_mapping
from tools.exact_statistics import exact_wilcoxon_signed_rank


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    errors = []
    report = {"status": "PASS", "shapes": {}}
    for shape_root in sorted(path for path in root.iterdir() if path.is_dir()):
        protocol_file = shape_root / "protocol.json"
        results_file = shape_root / "results.json"
        if not protocol_file.exists() or not results_file.exists():
            continue
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        rows = json.loads(results_file.read_text(encoding="utf-8"))
        expected_seeds = {int(protocol["base_seed"]) + i for i in range(int(protocol["runs"]))}
        by_variant = {
            variant: {int(row["seed"]): row for row in rows if row["variant"] == variant}
            for variant in ("uniform", "area_weighted")
        }
        if any(set(group) != expected_seeds for group in by_variant.values()):
            errors.append(f"{shape_root.name}: incomplete or unexpected paired seeds")
        truth = load_trait(protocol["ground_truth_trait"])
        reference = sample_trait(
            truth,
            int(protocol["evaluation_points"]),
            int(protocol["evaluation_seed"]),
            int(protocol["evaluation_grid"]),
        )
        max_metric_error = 0.0
        for variant, group in by_variant.items():
            expected_weight_flag = variant == "area_weighted"
            for seed, row in group.items():
                if int(row["evaluations"]) != int(protocol["max_evaluations"]):
                    errors.append(f"{shape_root.name}/{variant}/{seed}: wrong FE count")
                if bool(row["area_weights"]) != expected_weight_flag:
                    errors.append(f"{shape_root.name}/{variant}/{seed}: wrong variant flag")
                if row.get("evaluation_reference_mode") != "analytic-area-uniform":
                    errors.append(f"{shape_root.name}/{variant}/{seed}: nonuniform evaluation")
                model = sample_trait(
                    trait_from_mapping(row["trait"]),
                    int(protocol["evaluation_points"]),
                    int(row["evaluation_model_seed"]),
                    int(protocol["evaluation_grid"]),
                )
                metrics = geometric_metrics(reference, model, float(protocol["gt_threshold"]))
                for metric in ("gt_chamfer", "gt_fscore"):
                    difference = abs(float(row[metric]) - float(metrics[metric]))
                    max_metric_error = max(max_metric_error, difference)
                    if difference > 1e-10:
                        errors.append(f"{shape_root.name}/{variant}/{seed}: {metric} mismatch")
                if int(row["success"]) != int(metrics["gt_chamfer"] <= protocol["success_chamfer"]):
                    errors.append(f"{shape_root.name}/{variant}/{seed}: success mismatch")
                record = json.loads(Path(row["record_file"]).read_text(encoding="utf-8"))
                if bool(record["cfg"]["model"]["use_area_weights"]) != expected_weight_flag:
                    errors.append(f"{shape_root.name}/{variant}/{seed}: recorded config mismatch")
        common = sorted(set(by_variant["uniform"]) & set(by_variant["area_weighted"]))
        uniform = np.asarray([by_variant["uniform"][seed]["gt_chamfer"] for seed in common])
        weighted = np.asarray([by_variant["area_weighted"][seed]["gt_chamfer"] for seed in common])
        difference = uniform - weighted
        test = exact_wilcoxon_signed_rank(difference)
        report["shapes"][shape_root.name] = {
            "paired_runs": len(common),
            "uniform_chamfer": describe(uniform),
            "area_weighted_chamfer": describe(weighted),
            "uniform_successes": int(sum(by_variant["uniform"][seed]["success"] for seed in common)),
            "area_weighted_successes": int(sum(by_variant["area_weighted"][seed]["success"] for seed in common)),
            "area_weighted_wins": int(np.sum(weighted < uniform)),
            "median_uniform_minus_area_weighted": float(np.median(difference)),
            "wilcoxon_statistic": test["statistic"],
            "wilcoxon_exact_two_sided_p": test["exact_two_sided_p"],
            "wilcoxon_nonzero_pairs": test["nonzero_pairs"],
            "wilcoxon_zero_pairs": test["zero_pairs"],
            "external_metric_recompute_max_abs_error": max_metric_error,
        }
    report["errors"] = sorted(set(errors))
    if errors:
        report["status"] = "FAIL"
    (root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
