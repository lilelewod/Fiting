"""Run a paired PSO--CS comparison on the PMF-style partial cylinder.

All algorithms receive the same population size, exact objective-evaluation
budget, and per-repeat seed vector.  Fits from corrupted inputs are evaluated
against the same clean reference point cloud.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrypoints.fit_point_cloud import prepare_3d_cfg, run_experiment
from models.surface.pmf_cylinder_rule import PMFCylinderTrait, sample_partial_cylinder
from tools.data_tool import read_point_cloud
from tools.tool import json_default


CONDITION_FILES = {
    "clean": "clean.ply",
    "outlier_50": "outlier_50.ply",
    "outlier_80": "outlier_80.ply",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fit_pmf_cylinder.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/pmf_cylinder"))
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITION_FILES), default=list(CONDITION_FILES)
    )
    parser.add_argument("--algorithms", nargs="+", choices=("pso", "cs"), default=("pso", "cs"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260722)
    parser.add_argument("--seed-list", type=int, nargs="+", default=None)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=50000)
    parser.add_argument("--success-floor-factor", type=float, default=2.0)
    parser.add_argument("--success-fscore", type=float, default=0.9)
    parser.add_argument("--threshold-neighbor-factor", type=float, default=5.0)
    parser.add_argument("--density-support-fraction", type=float, default=1.0)
    parser.add_argument("--density-support-neighbors", type=int, default=8)
    parser.add_argument("--density-support-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument(
        "--nearest-neighbor-backend",
        choices=("legacy", "sklearn", "faiss", "torch_cuda"),
        default="legacy",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def trait_from_mapping(mapping):
    trait = PMFCylinderTrait()
    for key in (
        "x0", "y0", "z0", "radius", "height", "start_angle", "angular_span", "end_angle"
    ):
        if key in mapping:
            setattr(trait, key, float(mapping[key]))
    trait.end_angle = trait.start_angle + trait.angular_span
    return trait


def clean_reference_metrics(reference, model, threshold):
    ref_to_model = KDTree(model).query(reference, k=1)[0].ravel()
    model_to_ref = KDTree(reference).query(model, k=1)[0].ravel()
    precision = float(np.mean(model_to_ref < threshold))
    recall = float(np.mean(ref_to_model < threshold))
    return {
        "gt_chamfer": float(ref_to_model.mean() + model_to_ref.mean()),
        "gt_ref_to_model": float(ref_to_model.mean()),
        "gt_model_to_ref": float(model_to_ref.mean()),
        "gt_precision": precision,
        "gt_recall": recall,
        "gt_fscore": float(2.0 * precision * recall / (precision + recall + 1e-8)),
    }


def circular_error(a, b):
    return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))


def write_results(output_root, rows):
    with (output_root / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, default=json_default, indent=2)
    scalar_keys = [
        key for key, value in rows[0].items() if key != "trait" and not isinstance(value, (dict, list))
    ]
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "mean": float(np.mean(values)),
    }


def write_summary(output_root, rows, conditions, algorithms):
    summary = []
    for condition in conditions:
        for algorithm in algorithms:
            group = [
                row for row in rows
                if row["condition"] == condition and row["algorithm"] == algorithm
            ]
            if not group:
                continue
            summary.append(
                {
                    "condition": condition,
                    "algorithm": algorithm,
                    "runs": len(group),
                    "gt_chamfer": descriptive([row["gt_chamfer"] for row in group]),
                    "gt_fscore": descriptive([row["gt_fscore"] for row in group]),
                    "wall_time_s": descriptive([row["wall_time_s"] for row in group]),
                    "success_count": int(sum(row["success"] for row in group)),
                    "success_rate": float(np.mean([row["success"] for row in group])),
                }
            )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.runs <= 0 or args.population_size < 4 or args.num_envs <= 0:
        raise ValueError("runs and num-envs must be positive; population-size must be at least 4")
    if args.population_size % args.num_envs:
        raise ValueError("population-size must be divisible by num-envs")
    if args.max_evaluations < args.population_size or (
        args.max_evaluations - args.population_size
    ) % (2 * args.population_size):
        raise ValueError(
            "Exact PSO--CS fairness requires budget = population + k * 2 * population"
        )
    if args.seed_list is not None and len(set(args.seed_list)) != len(args.seed_list):
        raise ValueError("--seed-list must not contain duplicates")
    if not 0.0 < args.density_support_fraction <= 1.0:
        raise ValueError("--density-support-fraction must lie in (0, 1]")
    if args.density_support_neighbors < 2:
        raise ValueError("--density-support-neighbors must be at least 2")

    data_root = args.data_root.resolve()
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    truth = metadata["ground_truth"]
    truth_trait = trait_from_mapping(truth)
    clean_reference = read_point_cloud(str(data_root / "clean.ply"))
    nearest = KDTree(clean_reference).query(clean_reference, k=2)[0][:, 1]
    metric_threshold = args.threshold_neighbor_factor * float(
        np.median(nearest[nearest > 0.0])
    )
    truth_model = sample_partial_cylinder(truth_trait, sample_angle=64, sample_height=32)
    sampling_floor = clean_reference_metrics(
        clean_reference, truth_model, metric_threshold
    )["gt_chamfer"]
    success_chamfer = args.success_floor_factor * sampling_floor

    seeds = (
        args.seed_list
        if args.seed_list is not None
        else [args.base_seed + index for index in range(args.runs)]
    )
    protocol = {
        "dataset_provenance": metadata["provenance"],
        "conditions": list(args.conditions),
        "algorithms": list(args.algorithms),
        "base_seeds": seeds,
        "population_size": args.population_size,
        "num_envs": args.num_envs,
        "max_evaluations": args.max_evaluations,
        "paired_internal_seeds": True,
        "density_support_fraction": args.density_support_fraction,
        "density_support_neighbors": args.density_support_neighbors,
        "density_support_mode": args.density_support_mode,
        "nearest_neighbor_backend": args.nearest_neighbor_backend,
        "clean_reference": str((data_root / "clean.ply").resolve()),
        "model_evaluation_grid": [64, 32],
        "metric_threshold": metric_threshold,
        "truth_sampling_floor_chamfer": sampling_floor,
        "success_chamfer": success_chamfer,
        "success_fscore": args.success_fscore,
        "success_definition": "gt_chamfer <= threshold AND gt_fscore >= threshold",
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_file = output_root / "protocol.json"
    if protocol_file.exists():
        existing_protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if existing_protocol != protocol:
            raise ValueError("resume protocol does not match the existing output directory")
        if not args.resume:
            raise FileExistsError(f"existing experiment requires --resume: {output_root}")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    results_file = output_root / "results.json"
    rows = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else []
    completed = {
        (row["condition"], row["algorithm"], int(row["seed"])) for row in rows
    }
    with args.config.open(encoding="utf-8") as stream:
        template = yaml.safe_load(stream)
    template["fitter"]["num_envs"] = args.num_envs
    template["fitter"]["episodes_per_env"] = args.population_size // args.num_envs
    template["fitter"]["max_episode"] = args.max_evaluations
    template["estimator"]["density_support_fraction"] = args.density_support_fraction
    template["estimator"]["density_support_neighbors"] = args.density_support_neighbors
    template["estimator"]["density_support_mode"] = args.density_support_mode
    template["estimator"]["nearest_neighbor_backend"] = args.nearest_neighbor_backend
    template.setdefault("record", {})["visualization"] = None

    print(json.dumps(protocol, indent=2))
    total = len(args.conditions) * len(args.algorithms) * len(seeds)
    for condition in args.conditions:
        for repeat, base_seed in enumerate(seeds, start=1):
            seed_sequence = np.random.SeedSequence(base_seed)
            shared_seeds = [
                int(value) for value in seed_sequence.generate_state(args.num_envs + 1)
            ]
            for algorithm in args.algorithms:
                key = (condition, algorithm, base_seed)
                if key in completed:
                    print(f"{condition} [{repeat}/{len(seeds)}] {algorithm.upper()} complete; skipping")
                    continue
                cfg = deepcopy(template)
                cfg["data_file"] = str((data_root / CONDITION_FILES[condition]).resolve())
                cfg.setdefault("model", {})["type"] = "pmf_cylinder"
                cfg["fitter"]["algo_name"] = algorithm
                cfg["seeds"] = shared_seeds.copy()
                cfg = prepare_3d_cfg(cfg)
                cfg["record"]["root_dir"] = (
                    output_root / condition / algorithm / f"repeat_{repeat:02d}"
                ).as_posix() + "/"
                cfg["record"]["timestamp"] = datetime.now().strftime("%Y-%m%d/%H%M-%S-%f")
                cfg["experiment"] = {
                    "comparison": "paired-PSO-CS-PMF-cylinder",
                    "condition": condition,
                    "repeat": repeat,
                    "base_seed": base_seed,
                    "shared_seeds": shared_seeds,
                    "population_size": args.population_size,
                    "max_evaluations": args.max_evaluations,
                    "nearest_neighbor_backend": args.nearest_neighbor_backend,
                }
                print(
                    f"\n{condition} [{repeat}/{len(seeds)}] {algorithm.upper()} "
                    f"| completed={len(rows)}/{total} | seeds={shared_seeds}"
                )
                started = time.perf_counter()
                record = run_experiment(cfg)
                wall_time = time.perf_counter() - started
                if int(record.num_evaluations) != args.max_evaluations:
                    raise RuntimeError(
                        f"{algorithm} used {record.num_evaluations}, expected {args.max_evaluations} evaluations"
                    )
                token = record.best_token_set[0]
                fitted_trait = token.trait
                evaluation_cloud = sample_partial_cylinder(
                    fitted_trait, sample_angle=64, sample_height=32
                )
                metrics = clean_reference_metrics(
                    clean_reference, evaluation_cloud, metric_threshold
                )
                center_error = float(
                    np.linalg.norm(
                        [
                            fitted_trait.x0 - truth["x0"],
                            fitted_trait.y0 - truth["y0"],
                            fitted_trait.z0 - truth["z0"],
                        ]
                    )
                )
                row = {
                    "condition": condition,
                    "algorithm": algorithm,
                    "repeat": repeat,
                    "seed": base_seed,
                    "shared_seeds": shared_seeds,
                    "evaluations": int(record.num_evaluations),
                    "population_size": args.population_size,
                    "nearest_neighbor_backend": args.nearest_neighbor_backend,
                    "wall_time_s": wall_time,
                    "best_score": float(record.best_score),
                    "input_chamfer": float(record.chamfer),
                    "input_fscore": float(record.f5),
                    "trait": dict(fitted_trait),
                    "center_error": center_error,
                    "radius_error": float(abs(fitted_trait.radius - truth["radius"])),
                    "height_error": float(abs(fitted_trait.height - truth["height"])),
                    "start_angle_error": circular_error(
                        fitted_trait.start_angle, truth["start_angle"]
                    ),
                    "span_error": float(abs(fitted_trait.angular_span - truth["angular_span"])),
                    "metric_threshold": metric_threshold,
                    "success_chamfer_threshold": success_chamfer,
                    "success_fscore_threshold": args.success_fscore,
                    **metrics,
                    "record_file": str(Path(record.out_json_file_name).resolve()),
                }
                row["success"] = int(
                    row["gt_chamfer"] <= success_chamfer
                    and row["gt_fscore"] >= args.success_fscore
                )
                rows.append(row)
                completed.add(key)
                write_results(output_root, rows)
                write_summary(output_root, rows, args.conditions, args.algorithms)

    print(f"\nCompleted {len(rows)}/{total} runs: {output_root}")


if __name__ == "__main__":
    main()
