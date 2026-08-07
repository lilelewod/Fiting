"""Validate and summarize the formal PMF-cylinder density-support ablation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.exact_statistics import exact_wilcoxon_signed_rank


def read_json_retry(path: Path, attempts: int = 10):
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1)


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/ieee_superquadric/protocols/pmf_cylinder_density_support_ablation.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    protocol = read_json_retry(args.protocol.resolve())
    seeds = {int(seed) for seed in protocol["paired_base_seeds"]}
    variants = [variant["name"] for variant in protocol["variants"]]
    errors = []
    output = {"status": "PASS", "conditions": {}}

    for condition in protocol["dataset"]["conditions"]:
        rows_by_variant = {}
        for variant in variants:
            result_file = root / condition / variant / "results.json"
            rows = read_json_retry(result_file) if result_file.exists() else []
            by_seed = {int(row["seed"]): row for row in rows}
            if len(by_seed) != len(rows):
                errors.append(f"duplicate seeds in {condition}/{variant}")
            if set(by_seed) - seeds:
                errors.append(f"unexpected seeds in {condition}/{variant}")
            if set(by_seed) != seeds and not args.allow_incomplete:
                errors.append(f"incomplete {condition}/{variant}: {len(by_seed)}/{len(seeds)}")
            for row in rows:
                if int(row["evaluations"]) != int(protocol["max_evaluations"]):
                    errors.append(f"wrong FE count in {condition}/{variant}/{row['seed']}")
                if row["condition"] != condition or row["algorithm"] != protocol["optimizer"]:
                    errors.append(f"wrong condition/optimizer in {condition}/{variant}/{row['seed']}")
            rows_by_variant[variant] = by_seed

        common = sorted(set.intersection(*(set(rows_by_variant[v]) for v in variants)))
        condition_summary = {"paired_seeds": common, "variants": {}}
        for variant in variants:
            rows = [rows_by_variant[variant][seed] for seed in common]
            condition_summary["variants"][variant] = {
                "completed_runs": len(rows_by_variant[variant]),
                "paired_chamfer": describe([row["gt_chamfer"] for row in rows]) if rows else None,
                "paired_fscore": describe([row["gt_fscore"] for row in rows]) if rows else None,
                "paired_successes": int(sum(row["success"] for row in rows)),
                "paired_wall_time_s": describe([row["wall_time_s"] for row in rows]) if rows else None,
            }
        if common:
            full = np.asarray([rows_by_variant["full_input"][seed]["gt_chamfer"] for seed in common])
            adaptive = np.asarray([rows_by_variant["adaptive_density"][seed]["gt_chamfer"] for seed in common])
            difference = full - adaptive
            paired = {
                "adaptive_wins": int(np.sum(adaptive < full)),
                "full_input_wins": int(np.sum(full < adaptive)),
                "median_full_minus_adaptive": float(np.median(difference)),
            }
            test = exact_wilcoxon_signed_rank(difference)
            if test is not None:
                paired["wilcoxon_statistic"] = test["statistic"]
                paired["wilcoxon_exact_two_sided_p"] = test["exact_two_sided_p"]
                paired["wilcoxon_nonzero_pairs"] = test["nonzero_pairs"]
                paired["wilcoxon_zero_pairs"] = test["zero_pairs"]
            condition_summary["paired_comparison"] = paired
        output["conditions"][condition] = condition_summary

    if errors:
        output["status"] = "FAIL"
    output["errors"] = sorted(set(errors))
    (root / "ablation_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
