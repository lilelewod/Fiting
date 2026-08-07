"""Summarize PMF-cylinder runs against the same clean reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree

from tools.data_tool import read_point_cloud


def clean_reference_metrics(reference, model, threshold):
    ref_to_model = KDTree(model).query(reference, k=1)[0].ravel()
    model_to_ref = KDTree(reference).query(model, k=1)[0].ravel()
    precision = float(np.mean(model_to_ref < threshold))
    recall = float(np.mean(ref_to_model < threshold))
    return {
        "gt_chamfer": float(ref_to_model.mean() + model_to_ref.mean()),
        "gt_ref_to_model": float(ref_to_model.mean()),
        "gt_model_to_ref": float(model_to_ref.mean()),
        "gt_fscore": float(2.0 * precision * recall / (precision + recall + 1e-8)),
    }


def latest_formal_record(root, evaluations):
    candidates = []
    for path in root.glob("**/record.json"):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if (int(record["cfg"]["fitter"]["max_episode"]) == evaluations
                and int(record.get("num_evaluations", -1)) == evaluations):
            candidates.append((path.stat().st_mtime, path, record))
    if not candidates:
        raise FileNotFoundError(f"No completed {evaluations}-FE record under {root}")
    return max(candidates, key=lambda item: item[0])[1:]


def last_trait(record_file):
    evolution = record_file.parent / "evolution_of_round_0_instance_0.json"
    with evolution.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    return entries[-1]["trait"]


def circular_error(a, b):
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../outputs/pmf_cylinder_comparison/single_seed_20260715")
    parser.add_argument("--threshold-factor", type=float, default=5.0)
    parser.add_argument("--evaluations", type=int, default=50000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    outputs = project_root.parent / "outputs"
    reference = read_point_cloud(str(project_root / "datasets/pmf_cylinder/clean.ply"))
    nearest = KDTree(reference).query(reference, k=2)[0][:, 1]
    threshold = args.threshold_factor * float(np.median(nearest[nearest > 0.0]))
    metadata = json.loads((project_root / "datasets/pmf_cylinder/metadata.json").read_text(encoding="utf-8"))
    truth = metadata["ground_truth"]

    rows = []
    for condition in ("clean", "outlier_50", "outlier_80"):
        for algorithm in ("cco", "cs"):
            root = outputs / algorithm / "3d/pmf_cylinder/pmf_cylinder" / condition / "run_1"
            record_file, record = latest_formal_record(root, args.evaluations)
            fitted = read_point_cloud(str(record_file.parent / "best_cloud_of_instance_0.ply"))
            trait = last_trait(record_file)
            row = {
                "condition": condition,
                "algorithm": algorithm,
                "evaluations": int(record["num_evaluations"]),
                "seed_signature": ",".join(str(x) for x in record["cfg"]["seeds"]),
                "best_score": float(record["best_score"]),
                "input_chamfer": float(record["chamfer"]),
                "input_fscore": float(record["f5"]),
                "clean_threshold": threshold,
                "center_error": float(np.linalg.norm([
                    trait["x0"] - truth["x0"],
                    trait["y0"] - truth["y0"],
                    trait["z0"] - truth["z0"],
                ])),
                "radius_error": abs(float(trait["radius"] - truth["radius"])),
                "height_error": abs(float(trait["height"] - truth["height"])),
                "start_angle_error": circular_error(float(trait["start_angle"]), truth["start_angle"]),
                "span_error": abs(float(trait["angular_span"] - truth["angular_span"])),
                "record_file": str(record_file),
            }
            row.update(clean_reference_metrics(reference, fitted, threshold))
            rows.append(row)

    for algorithm in ("cco", "cs"):
        clean = next(row for row in rows if row["algorithm"] == algorithm and row["condition"] == "clean")
        for row in rows:
            if row["algorithm"] != algorithm:
                continue
            row["chamfer_relative_degradation"] = (
                (row["gt_chamfer"] - clean["gt_chamfer"])
                / (clean["gt_chamfer"] + np.finfo(float).eps)
            )
            row["fscore_retention"] = row["gt_fscore"] / (clean["gt_fscore"] + np.finfo(float).eps)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
