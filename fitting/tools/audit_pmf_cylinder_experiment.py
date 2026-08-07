"""Audit and summarize a paired PMF-cylinder PSO--CS experiment.

The audit is deliberately independent of the optimizer's recorded objective:
it reconstructs every fitted cylinder and recomputes Chamfer/F-score against
the clean reference.  It also checks exact FE budgets, paired seeds, dataset
cardinality/provenance, and the empirical uniformity of clean surface samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kstest
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.surface.pmf_cylinder_rule import PMFCylinderTrait, sample_partial_cylinder
from tools.data_tool import read_point_cloud
from tools.exact_statistics import exact_wilcoxon_signed_rank


def read_json_retry(path: Path, attempts: int = 10):
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1)


def trait_from_mapping(mapping):
    trait = PMFCylinderTrait()
    for key in ("x0", "y0", "z0", "radius", "height", "start_angle", "angular_span"):
        setattr(trait, key, float(mapping[key]))
    trait.end_angle = trait.start_angle + trait.angular_span
    return trait


def external_metrics(reference, model, threshold):
    ref_to_model = KDTree(model).query(reference, k=1)[0].ravel()
    model_to_ref = KDTree(reference).query(model, k=1)[0].ravel()
    precision = float(np.mean(model_to_ref < threshold))
    recall = float(np.mean(ref_to_model < threshold))
    return {
        "gt_chamfer": float(ref_to_model.mean() + model_to_ref.mean()),
        "gt_fscore": float(2.0 * precision * recall / (precision + recall + 1e-8)),
    }


def descriptive(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def audit_status(errors, missing):
    """Return PASS only for a complete, error-free result matrix."""
    if errors:
        return "FAIL"
    if missing:
        return "INCOMPLETE"
    return "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/pmf_cylinder"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional audit destination; defaults to <experiment_root>/audit.json.",
    )
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    data_root = args.data_root.resolve()
    protocol = read_json_retry(root / "protocol.json")
    rows = read_json_retry(root / "results.json")
    metadata = read_json_retry(data_root / "metadata.json")
    truth = metadata["ground_truth"]
    clean = read_point_cloud(str(data_root / "clean.ply"))

    errors = []
    notes = []
    expected_keys = {
        (condition, algorithm, int(seed))
        for condition in protocol["conditions"]
        for algorithm in protocol["algorithms"]
        for seed in protocol["base_seeds"]
    }
    actual_keys = [(r["condition"], r["algorithm"], int(r["seed"])) for r in rows]
    if len(actual_keys) != len(set(actual_keys)):
        errors.append("duplicate condition/algorithm/seed result keys")
    unexpected = set(actual_keys) - expected_keys
    missing = expected_keys - set(actual_keys)
    if unexpected:
        errors.append(f"{len(unexpected)} unexpected result keys")
    if missing and not args.allow_incomplete:
        errors.append(f"{len(missing)} expected result keys are missing")
    elif missing:
        notes.append(f"partial run: {len(missing)} of {len(expected_keys)} results remain")

    threshold = float(protocol["metric_threshold"])
    success_cd = float(protocol["success_chamfer"])
    success_f = float(protocol["success_fscore"])
    recompute_max_error = {"gt_chamfer": 0.0, "gt_fscore": 0.0}
    expected_backend = protocol.get("nearest_neighbor_backend")
    for row in rows:
        if int(row["evaluations"]) != int(protocol["max_evaluations"]):
            errors.append(f"wrong FE count for {row['condition']}/{row['algorithm']}/{row['seed']}")
        if int(row["population_size"]) != int(protocol["population_size"]):
            errors.append(f"wrong population for {row['condition']}/{row['algorithm']}/{row['seed']}")
        if expected_backend is not None and row.get("nearest_neighbor_backend") != expected_backend:
            errors.append(
                f"wrong nearest-neighbor backend for {row['condition']}/{row['algorithm']}/{row['seed']}"
            )
        if not np.isclose(float(row["metric_threshold"]), threshold):
            errors.append("metric threshold changed between runs")
        model = sample_partial_cylinder(trait_from_mapping(row["trait"]), 64, 32)
        metrics = external_metrics(clean, model, threshold)
        for name in recompute_max_error:
            delta = abs(float(row[name]) - metrics[name])
            recompute_max_error[name] = max(recompute_max_error[name], delta)
            if delta > 1e-10:
                errors.append(f"external {name} mismatch for {row['condition']}/{row['algorithm']}/{row['seed']}")
        expected_success = int(metrics["gt_chamfer"] <= success_cd and metrics["gt_fscore"] >= success_f)
        if int(row["success"]) != expected_success:
            errors.append(f"success label mismatch for {row['condition']}/{row['algorithm']}/{row['seed']}")

    by_pair = {}
    for row in rows:
        by_pair.setdefault((row["condition"], int(row["seed"])), {})[row["algorithm"]] = row
    for (condition, seed), pair in by_pair.items():
        if set(pair) == set(protocol["algorithms"]):
            seed_vectors = {tuple(r["shared_seeds"]) for r in pair.values()}
            if len(seed_vectors) != 1:
                errors.append(f"unpaired internal seeds for {condition}/{seed}")

    # The clean points must be retained exactly in both contaminated clouds.
    dataset_checks = {}
    for filename, counts in metadata["datasets"].items():
        cloud = read_point_cloud(str(data_root / filename))
        expected_count = int(counts["inliers"] + counts["outliers"])
        nearest_clean = KDTree(cloud).query(clean, k=1)[0].ravel()
        check = {
            "points": int(len(cloud)),
            "expected_points": expected_count,
            "max_clean_subset_distance": float(nearest_clean.max()),
        }
        dataset_checks[filename] = check
        if len(cloud) != expected_count:
            errors.append(f"wrong point count in {filename}")
        if float(nearest_clean.max()) > 1e-7:
            errors.append(f"clean inliers are not preserved in {filename}")

    # Uniform angle and height imply uniform physical area on a cylinder.
    centered = clean[:, :2] - np.asarray([truth["x0"], truth["y0"]])
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    angle_u = ((angle - truth["start_angle"]) % (2.0 * np.pi)) / truth["angular_span"]
    height_u = (clean[:, 2] - truth["z0"]) / truth["height"]
    uniformity = {
        "angle_ks_p": float(kstest(angle_u, "uniform").pvalue),
        "height_ks_p": float(kstest(height_u, "uniform").pvalue),
        "radial_max_abs_error": float(np.max(np.abs(np.linalg.norm(centered, axis=1) - truth["radius"]))),
    }
    if not np.all((angle_u >= -1e-7) & (angle_u <= 1.0 + 1e-7)):
        errors.append("clean angles fall outside the declared partial-cylinder span")
    if not np.all((height_u >= -1e-7) & (height_u <= 1.0 + 1e-7)):
        errors.append("clean heights fall outside the declared cylinder")
    if uniformity["angle_ks_p"] < 1e-3 or uniformity["height_ks_p"] < 1e-3:
        errors.append("clean samples fail the preregistered empirical uniformity check")

    summaries = {}
    for condition in protocol["conditions"]:
        paired = [pair for (cond, _), pair in by_pair.items() if cond == condition and set(pair) == {"pso", "cs"}]
        if not paired:
            continue
        pso = np.asarray([pair["pso"]["gt_chamfer"] for pair in paired], dtype=float)
        cs = np.asarray([pair["cs"]["gt_chamfer"] for pair in paired], dtype=float)
        differences = cs - pso
        stats = {
            "paired_runs": len(paired),
            "pso_chamfer": descriptive(pso),
            "cs_chamfer": descriptive(cs),
            "pso_successes": int(sum(pair["pso"]["success"] for pair in paired)),
            "cs_successes": int(sum(pair["cs"]["success"] for pair in paired)),
            "pso_wins": int(np.sum(pso < cs)),
            "cs_wins": int(np.sum(cs < pso)),
            "median_cs_minus_pso": float(np.median(differences)),
        }
        test = exact_wilcoxon_signed_rank(differences)
        if test is not None:
            stats["wilcoxon_statistic"] = test["statistic"]
            stats["wilcoxon_exact_two_sided_p"] = test["exact_two_sided_p"]
            stats["wilcoxon_nonzero_pairs"] = test["nonzero_pairs"]
            stats["wilcoxon_zero_pairs"] = test["zero_pairs"]
        summaries[condition] = stats

    audit = {
        "status": audit_status(errors, missing),
        "completed_results": len(rows),
        "expected_results": len(expected_keys),
        "errors": sorted(set(errors)),
        "notes": notes,
        "external_metric_recompute_max_abs_error": recompute_max_error,
        "dataset_checks": dataset_checks,
        "clean_sampling_uniformity": uniformity,
        "paired_summary": summaries,
    }
    output = args.output.resolve() if args.output is not None else root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
