"""Audit the paired random-versus-guided PSO initialization evidence."""

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


FOLDERS = {
    "box": {
        "random": "v2_box_clean_pso_5seeds_20260716_20260720",
        "guided": "v2_box_clean_guided_pso_5seeds_20260716_20260720",
    },
    "cylinder": {
        "random": "v2_cylinder_clean_pso_5seeds_20260716_20260720",
        "guided": "v2_cylinder_clean_guided_pso_5seeds_20260716_20260720",
    },
    "ellipsoid": {
        "random": "v2_ellipsoid_clean_5seeds_20260716_20260720",
        "guided": "v2_ellipsoid_clean_guided_pso_5seeds_20260716_20260720",
    },
}


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs/optimizer_comparison",
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\code\superquadic_data\v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_seeds = set(range(20260716, 20260721))
    errors = []
    report = {"status": "PASS", "shapes": {}}

    for shape, variants in FOLDERS.items():
        rows_by_variant = {}
        truth = load_trait(args.data_root.resolve() / shape / "trait.json")
        reference = sample_trait(truth, 20000, 20260716, 256)
        max_metric_error = 0.0
        for variant, folder in variants.items():
            rows = json.loads(
                (args.result_root.resolve() / folder / "results.json").read_text(encoding="utf-8")
            )
            rows = [row for row in rows if row["algorithm"] == "pso"]
            by_seed = {int(row["seed"]): row for row in rows}
            if set(by_seed) != expected_seeds or len(by_seed) != len(rows):
                errors.append(f"{shape}/{variant}: incomplete or duplicate seeds")
            for seed, row in by_seed.items():
                if int(row["evaluations"]) != 10000:
                    errors.append(f"{shape}/{variant}/{seed}: wrong FE count")
                if row.get("evaluation_reference_mode") != "analytic-area-uniform":
                    errors.append(f"{shape}/{variant}/{seed}: nonuniform evaluation")
                record = json.loads(Path(row["record_file"]).read_text(encoding="utf-8"))
                recorded_guided = bool(
                    record["cfg"]["fitter"].get("pso_guided_initialization", False)
                )
                if recorded_guided != (variant == "guided"):
                    errors.append(f"{shape}/{variant}/{seed}: recorded initialization mismatch")
                model = sample_trait(trait_from_mapping(row["trait"]), 20000, 20260717, 256)
                metrics = geometric_metrics(reference, model, 0.05)
                for metric in ("gt_chamfer", "gt_fscore"):
                    difference = abs(float(row[metric]) - float(metrics[metric]))
                    max_metric_error = max(max_metric_error, difference)
                    if difference > 1e-10:
                        errors.append(f"{shape}/{variant}/{seed}: {metric} mismatch")
                if int(row["success"]) != int(metrics["gt_chamfer"] <= 0.05):
                    errors.append(f"{shape}/{variant}/{seed}: success mismatch")
            rows_by_variant[variant] = by_seed

        seeds = sorted(set(rows_by_variant["random"]) & set(rows_by_variant["guided"]))
        random_cd = np.asarray([rows_by_variant["random"][seed]["gt_chamfer"] for seed in seeds])
        guided_cd = np.asarray([rows_by_variant["guided"][seed]["gt_chamfer"] for seed in seeds])
        difference = random_cd - guided_cd
        test = exact_wilcoxon_signed_rank(difference)
        report["shapes"][shape] = {
            "paired_seeds": seeds,
            "random_chamfer": describe(random_cd),
            "guided_chamfer": describe(guided_cd),
            "random_successes": int(sum(rows_by_variant["random"][seed]["success"] for seed in seeds)),
            "guided_successes": int(sum(rows_by_variant["guided"][seed]["success"] for seed in seeds)),
            "guided_wins": int(np.sum(guided_cd < random_cd)),
            "median_random_minus_guided": float(np.median(difference)),
            "wilcoxon_statistic": test["statistic"],
            "wilcoxon_exact_two_sided_p": test["exact_two_sided_p"],
            "wilcoxon_nonzero_pairs": test["nonzero_pairs"],
            "wilcoxon_zero_pairs": test["zero_pairs"],
            "external_metric_recompute_max_abs_error": max_metric_error,
        }

    report["errors"] = sorted(set(errors))
    if errors:
        report["status"] = "FAIL"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
