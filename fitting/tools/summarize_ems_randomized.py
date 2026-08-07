"""Summarize EMS results on the randomized superquadric benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ("clean", "noise_1pct_diag", "outlier_20", "missing_80")


def summary(values, success, runtime):
    values = np.asarray(values, dtype=np.float64)
    runtime = np.asarray(runtime, dtype=np.float64)
    return {
        "count": int(len(values)),
        "chamfer_median": float(np.median(values)),
        "chamfer_q1": float(np.percentile(values, 25)),
        "chamfer_q3": float(np.percentile(values, 75)),
        "chamfer_mean": float(np.mean(values)),
        "success_count": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "runtime_median_s": float(np.median(runtime)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--success-chamfer", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for case_record in manifest["case_records"]:
        case = case_record["case"]
        metadata = json.loads((args.data_root / case / "metadata.json").read_text(encoding="utf-8"))
        for condition in CONDITIONS:
            fit = json.loads(
                (args.result_root / case / condition / "result.json").read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                (args.result_root / case / condition / "evaluation.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "case": case,
                    "condition": condition,
                    "shape_stratum": metadata["strata"]["shape"],
                    "aspect_stratum": metadata["strata"]["aspect"],
                    "gt_chamfer": evaluation["gt_chamfer"],
                    "normalized_chamfer": evaluation["normalized_chamfer"],
                    "gt_fscore": evaluation["gt_fscore"],
                    "success": int(evaluation["gt_chamfer"] <= args.success_chamfer),
                    "wall_time_s": fit["wall_time_s"],
                    "mean_inlier_probability": fit["posterior"]["mean_inlier_probability"],
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[("condition", row["condition"])].append(row)
        grouped[("shape_stratum", row["shape_stratum"])].append(row)
        grouped[("aspect_stratum", row["aspect_stratum"])].append(row)
    report = {
        "protocol": {
            "algorithm": "EMS",
            "fixed_outlier_prior": 0.1,
            "success_chamfer": args.success_chamfer,
            "independent_cases": len(manifest["case_records"]),
            "conditions": list(CONDITIONS),
        },
        "overall": summary(
            [row["gt_chamfer"] for row in rows],
            [row["success"] for row in rows],
            [row["wall_time_s"] for row in rows],
        ),
        "groups": {},
    }
    for (group_type, group_name), group_rows in grouped.items():
        report["groups"].setdefault(group_type, {})[group_name] = summary(
            [row["gt_chamfer"] for row in group_rows],
            [row["success"] for row in group_rows],
            [row["wall_time_s"] for row in group_rows],
        )
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
