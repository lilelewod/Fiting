"""Run and aggregate the paired area-weight screening suite."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parents[1] / "superquadic_data" / "v2"
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "outputs"
    / "area_weight_ablation"
    / "formal_v2_pso_clean_48x48_5008fe_5seeds"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", nargs="+", default=["box", "ellipsoid", "cylinder"])
    parser.add_argument("--algorithm", default="pso", choices=["cco", "cs", "pso", "de"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260714)
    parser.add_argument("--max-evaluations", type=int, default=5008)
    parser.add_argument("--sample-eta", type=int, default=48)
    parser.add_argument("--sample-omega", type=int, default=48)
    parser.add_argument("--force", action="store_true", help="Rerun completed shapes.")
    return parser.parse_args()


def is_complete(result_file, expected_rows):
    if not result_file.exists():
        return False
    try:
        with open(result_file, encoding="utf-8") as stream:
            return len(json.load(stream)) == expected_rows
    except (OSError, ValueError):
        return False


def aggregate(output_root, shapes):
    rows = []
    for shape in shapes:
        result_file = output_root / shape / "results.json"
        if not result_file.exists():
            continue
        with open(result_file, encoding="utf-8") as stream:
            shape_rows = json.load(stream)
        for row in shape_rows:
            row["shape"] = shape
            row.pop("trait", None)
            rows.append(row)

    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(output_root / "all_results.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    paired = []
    for shape in shapes:
        shape_rows = [row for row in rows if row["shape"] == shape]
        for repeat in sorted({row["repeat"] for row in shape_rows}):
            pair = {row["variant"]: row for row in shape_rows if row["repeat"] == repeat}
            if set(pair) != {"uniform", "area_weighted"}:
                continue
            paired.append({
                "shape": shape,
                "repeat": repeat,
                "seed": pair["uniform"]["seed"],
                "chamfer_improvement": pair["uniform"]["gt_chamfer"] - pair["area_weighted"]["gt_chamfer"],
                "fscore_improvement": pair["area_weighted"]["gt_fscore"] - pair["uniform"]["gt_fscore"],
            })

    improvements = np.asarray([x["chamfer_improvement"] for x in paired], dtype=float)
    nonzero = improvements[improvements != 0.0]
    test = wilcoxon(nonzero, alternative="two-sided", method="auto") if nonzero.size else None
    summary = {
        "pairs": len(paired),
        "area_weighted_chamfer_win_rate": sum(x["chamfer_improvement"] > 0 for x in paired) / len(paired),
        "area_weighted_fscore_win_rate": sum(x["fscore_improvement"] > 0 for x in paired) / len(paired),
        "mean_chamfer_improvement": sum(x["chamfer_improvement"] for x in paired) / len(paired),
        "median_chamfer_improvement": float(np.median(improvements)),
        "mean_fscore_improvement": sum(x["fscore_improvement"] for x in paired) / len(paired),
        "wilcoxon_statistic": float(test.statistic) if test else None,
        "wilcoxon_pvalue_two_sided": float(test.pvalue) if test else None,
        "paired_results": paired,
    }
    with open(output_root / "suite_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_rows = args.runs * 2

    for shape in args.shapes:
        shape_output = output_root / shape
        if not args.force and is_complete(shape_output / "results.json", expected_rows):
            print(f"Skipping completed shape: {shape}")
            continue
        shape_root = data_root / shape
        metadata_file = shape_root / "metadata.json"
        trait_file = shape_root / "trait.json"
        if not metadata_file.exists() or not trait_file.exists():
            raise FileNotFoundError(f"v2 metadata/trait missing for shape: {shape_root}")
        with open(metadata_file, encoding="utf-8") as stream:
            metadata = json.load(stream)
        estimator = metadata["fixed_estimator_protocol"]
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "run_area_weight_ablation.py"),
            "--data-file", str(shape_root / "clean.ply"),
            "--ground-truth", str(shape_root / "reference_uniform.ply"),
            "--ground-truth-trait", str(trait_file),
            "--algorithm", args.algorithm,
            "--runs", str(args.runs),
            "--base-seed", str(args.base_seed),
            "--population-size", "16",
            "--num-envs", "1",
            "--max-evaluations", str(args.max_evaluations),
            "--sample-eta", str(args.sample_eta),
            "--sample-omega", str(args.sample_omega),
            "--data-resolution", str(estimator["data_resolution"]),
            "--model-resolution", str(estimator["model_resolution"]),
            "--gt-threshold", "0.05",
            "--success-chamfer", "0.05",
            "--evaluation-points", "20000",
            "--evaluation-grid", "256",
            "--evaluation-seed", "20260716",
            "--output-root", str(shape_output),
            "--quiet",
            "--resume",
        ]
        print(f"Running shape: {shape}")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    aggregate(output_root, args.shapes)
    print(f"Saved screening suite to: {output_root}")


if __name__ == "__main__":
    main()
