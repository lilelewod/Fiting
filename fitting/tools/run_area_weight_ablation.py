"""Paired area-weight ablation for superquadric fitting with a fixed optimizer."""

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
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrypoints.fit_point_cloud import prepare_3d_cfg, run_experiment
from tools.data_tool import read_point_cloud
from tools.superquadric_evaluation import geometric_metrics, load_trait, sample_trait
from tools.tool import json_default


VARIANTS = (
    ("uniform", False),
    ("area_weighted", True),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fit_superquadric.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--ground-truth-trait", default=None)
    parser.add_argument("--algorithm", default="cco", choices=["cco", "cs", "pso", "de"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260714)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=10000)
    parser.add_argument("--data-resolution", type=float, default=None)
    parser.add_argument("--model-resolution", type=float, default=None)
    parser.add_argument("--sample-eta", type=int, default=None)
    parser.add_argument("--sample-omega", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gt-threshold", type=float, default=0.05)
    parser.add_argument("--evaluation-points", type=int, default=20000)
    parser.add_argument("--evaluation-grid", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=20260716)
    parser.add_argument("--success-chamfer", type=float, default=0.05)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_outputs(output_root, rows):
    with open(output_root / "results.json", "w", encoding="utf-8") as stream:
        json.dump(rows, stream, default=json_default, indent=2)
    scalar_keys = [
        key for key, value in rows[0].items()
        if key != "trait" and not isinstance(value, (list, dict))
    ]
    with open(output_root / "results.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summaries = []
    for variant, _ in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary = {"variant": variant, "runs": len(selected)}
        for metric in ("wall_time_s", "gt_chamfer", "gt_d2m", "gt_m2d", "gt_fscore", "success"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        summaries.append(summary)

    paired = []
    for repeat in sorted({row["repeat"] for row in rows}):
        pair = {row["variant"]: row for row in rows if row["repeat"] == repeat}
        if set(pair) != {"uniform", "area_weighted"}:
            continue
        paired.append({
            "repeat": repeat,
            "seed": pair["uniform"]["seed"],
            # Positive improvements always favor area weighting.
            "chamfer_improvement": pair["uniform"]["gt_chamfer"] - pair["area_weighted"]["gt_chamfer"],
            "fscore_improvement": pair["area_weighted"]["gt_fscore"] - pair["uniform"]["gt_fscore"],
        })
    chamfer_differences = np.asarray(
        [item["chamfer_improvement"] for item in paired], dtype=float
    )
    nonzero = chamfer_differences[chamfer_differences != 0.0]
    if nonzero.size:
        test = wilcoxon(nonzero, alternative="two-sided", method="auto")
        paired_statistics = {
            "pairs": len(paired),
            "area_weighted_wins": int(np.sum(chamfer_differences > 0.0)),
            "uniform_wins": int(np.sum(chamfer_differences < 0.0)),
            "ties": int(np.sum(chamfer_differences == 0.0)),
            "median_chamfer_improvement": float(np.median(chamfer_differences)),
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_pvalue_two_sided": float(test.pvalue),
        }
    else:
        paired_statistics = {
            "pairs": len(paired),
            "area_weighted_wins": 0,
            "uniform_wins": 0,
            "ties": len(paired),
            "median_chamfer_improvement": 0.0,
            "wilcoxon_statistic": None,
            "wilcoxon_pvalue_two_sided": None,
        }
    return summaries, paired, paired_statistics


def main():
    args = parse_args()
    if args.runs <= 0 or args.num_envs <= 0 or args.population_size < 4:
        raise ValueError("runs and num-envs must be positive; population-size must be at least 4")
    if args.evaluation_points <= 0 or args.evaluation_grid < 8:
        raise ValueError("evaluation-points must be positive and evaluation-grid must be at least 8")
    if args.success_chamfer <= 0.0:
        raise ValueError("success-chamfer must be positive")
    if args.population_size % args.num_envs:
        raise ValueError("population-size must be divisible by num-envs")
    if args.max_evaluations < args.population_size or (
        args.max_evaluations - args.population_size
    ) % (2 * args.population_size):
        raise ValueError(
            "Exact fairness requires max-evaluations = population-size + "
            "k * (2 * population-size). With population 16, use 10000."
        )

    with open(args.config, encoding="utf-8") as stream:
        base_cfg = yaml.safe_load(stream)
    base_cfg["data_file"] = str(Path(args.data_file).resolve())
    base_cfg.setdefault("model", {})["type"] = "superquadric"
    if args.sample_eta is not None:
        base_cfg["model"]["sample_eta"] = args.sample_eta
    if args.sample_omega is not None:
        base_cfg["model"]["sample_omega"] = args.sample_omega
    base_cfg["fitter"]["algo_name"] = args.algorithm
    base_cfg["fitter"]["num_envs"] = args.num_envs
    base_cfg["fitter"]["episodes_per_env"] = args.population_size // args.num_envs
    base_cfg["fitter"]["max_episode"] = args.max_evaluations
    if args.data_resolution is not None:
        if args.data_resolution <= 0.0:
            raise ValueError("data-resolution must be positive")
        base_cfg["estimator"]["data_resolution"] = args.data_resolution
        base_cfg["estimator"]["model_resolution"] = (
            args.model_resolution if args.model_resolution is not None
            else 0.45 * args.data_resolution
        )
    elif args.model_resolution is not None:
        raise ValueError("--model-resolution requires --data-resolution")
    base_cfg.setdefault("record", {})["visualization"] = None
    if args.quiet:
        base_cfg["record"]["verbose"] = False

    experiment_name = datetime.now().strftime("area-weight-%Y%m%d-%H%M%S")
    output_root = Path(
        args.output_root
        or PROJECT_ROOT.parent / "outputs" / "area_weight_ablation" / experiment_name
    )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "config": str(Path(args.config).resolve()),
        "data_file": str(Path(args.data_file).resolve()),
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "ground_truth_trait": str(Path(args.ground_truth_trait).resolve()) if args.ground_truth_trait else None,
        "algorithm": args.algorithm,
        "runs": args.runs,
        "base_seed": args.base_seed,
        "population_size": args.population_size,
        "num_envs": args.num_envs,
        "max_evaluations": args.max_evaluations,
        "data_resolution": args.data_resolution,
        "model_resolution": args.model_resolution,
        "sample_eta": args.sample_eta,
        "sample_omega": args.sample_omega,
        "gt_threshold": args.gt_threshold,
        "success_chamfer": args.success_chamfer,
        "evaluation_points": args.evaluation_points,
        "evaluation_grid": args.evaluation_grid,
        "evaluation_seed": args.evaluation_seed,
    }
    protocol_file = output_root / "protocol.json"
    results_file = output_root / "results.json"
    if protocol_file.exists():
        with open(protocol_file, encoding="utf-8") as stream:
            previous_protocol = json.load(stream)
        if previous_protocol != protocol:
            raise ValueError("resume protocol does not match the existing output directory")
    else:
        with open(protocol_file, "w", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2)
    if results_file.exists() and not args.resume:
        raise FileExistsError(f"results already exist: {results_file}; pass --resume to continue")
    if args.ground_truth_trait:
        gt_cloud = sample_trait(
            load_trait(args.ground_truth_trait),
            count=args.evaluation_points,
            seed=args.evaluation_seed,
            grid_resolution=args.evaluation_grid,
        )
        evaluation_reference_mode = "analytic-area-uniform"
    else:
        gt_cloud = read_point_cloud(args.ground_truth)
        evaluation_reference_mode = "provided-point-cloud-density-dependent"
        print(
            "WARNING: --ground-truth-trait was not supplied; the reference-side "
            "metric remains dependent on the provided point-cloud density."
        )

    if results_file.exists():
        with open(results_file, encoding="utf-8") as stream:
            rows = json.load(stream)
    else:
        rows = []
    completed = {(int(row["repeat"]), row["variant"]) for row in rows}
    for repeat in range(args.runs):
        seed_sequence = np.random.SeedSequence(args.base_seed + repeat)
        shared_seeds = [int(x) for x in seed_sequence.generate_state(args.num_envs + 1)]
        for variant, enabled in VARIANTS:
            if (repeat + 1, variant) in completed:
                print(f"Skipping completed pair member: repeat={repeat + 1}, variant={variant}")
                continue
            cfg = deepcopy(base_cfg)
            cfg["model"]["use_area_weights"] = enabled
            cfg["seeds"] = shared_seeds.copy()
            cfg = prepare_3d_cfg(cfg)
            cfg["record"]["root_dir"] = (output_root / variant / f"repeat_{repeat + 1:02d}").as_posix() + "/"
            cfg["record"]["timestamp"] = datetime.now().strftime("%Y-%m%d/%H%M-%S-%f")
            cfg["experiment"] = {
                "comparison": "uniform-vs-area-weighted",
                "variant": variant,
                "repeat": repeat + 1,
                "base_seed": args.base_seed + repeat,
                "shared_seeds": shared_seeds,
                "algorithm": args.algorithm,
                "population_size": args.population_size,
                "max_evaluations": args.max_evaluations,
                "data_resolution": cfg["estimator"]["data_resolution"],
                "model_resolution": cfg["estimator"]["model_resolution"],
                "sample_eta": int(cfg["model"].get("sample_eta", 96)),
                "sample_omega": int(cfg["model"].get("sample_omega", 96)),
                "evaluation_points": args.evaluation_points,
                "evaluation_grid": args.evaluation_grid,
                "evaluation_reference_seed": args.evaluation_seed,
                "evaluation_model_seed": args.evaluation_seed + 1,
                "evaluation_reference_mode": evaluation_reference_mode,
            }

            print(f"\n[{repeat + 1}/{args.runs}] {variant} | seeds={shared_seeds}")
            started = time.perf_counter()
            record = run_experiment(cfg)
            row = {
                "repeat": repeat + 1,
                "variant": variant,
                "area_weights": enabled,
                "algorithm": args.algorithm,
                "seed": args.base_seed + repeat,
                "evaluations": int(getattr(record, "num_evaluations", args.max_evaluations)),
                "wall_time_s": time.perf_counter() - started,
                "objective_score": float(record.best_score),
                "trait": record.best_token_set[0].trait if record.best_token_set[0] is not None else None,
                "record_file": str(Path(record.out_json_file_name).resolve()),
            }
            best_token = record.best_token_set[0] if record.best_token_set else None
            if best_token is None or best_token.trait is None:
                raise RuntimeError("fitted superquadric trait is unavailable for area-uniform evaluation")
            evaluation_cloud = sample_trait(
                best_token.trait,
                count=args.evaluation_points,
                seed=args.evaluation_seed + 1,
                grid_resolution=args.evaluation_grid,
            )
            row.update(geometric_metrics(gt_cloud, evaluation_cloud, args.gt_threshold))
            row["success"] = int(row["gt_chamfer"] <= args.success_chamfer)
            row.update({
                "evaluation_points": args.evaluation_points,
                "evaluation_grid": args.evaluation_grid,
                "evaluation_reference_seed": args.evaluation_seed,
                "evaluation_model_seed": args.evaluation_seed + 1,
                "evaluation_reference_mode": evaluation_reference_mode,
            })
            rows.append(row)
            completed.add((repeat + 1, variant))
            write_outputs(output_root, rows)

    summaries, paired, paired_statistics = summarize(rows)
    with open(output_root / "summary.json", "w", encoding="utf-8") as stream:
        json.dump({
            "variants": summaries,
            "paired_improvements": paired,
            "paired_statistics": paired_statistics,
        }, stream, indent=2)
    print(f"\nSaved area-weight ablation to: {output_root}")


if __name__ == "__main__":
    main()
