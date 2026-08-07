"""Strictly recompute every external metric in the v3 robustness study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.superquadric_evaluation import (
    geometric_metrics,
    load_trait,
    sample_trait,
    trait_from_mapping,
)


METRIC_KEYS = ("gt_chamfer", "gt_d2m", "gt_m2d", "gt_fscore")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_metric_error(max_error, stored, recomputed, context, errors, tolerance):
    for key in METRIC_KEYS:
        difference = abs(float(stored[key]) - float(recomputed[key]))
        max_error[key] = max(max_error[key], difference)
        if difference > tolerance:
            errors.append(f"external metric mismatch ({key}, {difference:.3g}): {context}")


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
        default=Path(
            r"C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_clean_guided_pso_1seed_20260716"
        ),
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path(
            r"C:\code\Fiting\outputs\optimizer_comparison\v3_stratified9_robustness_guided_pso_5seeds_20260721"
        ),
    )
    parser.add_argument(
        "--ems-root",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\ems_baseline\v3_randomized_fixedprior01"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    protocol = load_json(args.protocol.resolve())
    data_root = Path(protocol["data_root"])
    cases = protocol["cases"]
    conditions = protocol["conditions"]
    seeds = {int(value) for value in protocol["guided_pso"]["paired_base_seeds"]}
    expected_fe = int(protocol["guided_pso"]["max_evaluations"])
    evaluation = protocol["independent_evaluation"]
    point_count = int(evaluation["reference_points"])
    model_count = int(evaluation["model_points"])
    grid = int(evaluation["grid_resolution"])
    reference_seed = int(evaluation["reference_seed"])
    model_seed = int(evaluation["model_seed"])
    threshold = float(evaluation["fscore_distance_threshold"])
    success_threshold = float(evaluation["chamfer_success_threshold"])
    errors = []
    max_error = {key: 0.0 for key in METRIC_KEYS}
    audited_pso = 0
    audited_ems = 0
    reuse = protocol.get("reuse_completed", {})
    reused_cases = set(reuse.get("cases", []))
    reused_clean_root = Path(reuse["clean_root"]) if reuse.get("clean_root") else None
    reused_robustness_root = (
        Path(reuse["robustness_root"]) if reuse.get("robustness_root") else None
    )

    for case in cases:
        case_root = data_root / case
        reference = sample_trait(
            load_trait(case_root / "trait.json"),
            count=point_count,
            seed=reference_seed,
            grid_resolution=grid,
        )
        for condition in conditions:
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
            rows = load_json(pso_file) if pso_file.exists() else []
            all_seeds = [int(row["seed"]) for row in rows]
            if len(set(all_seeds)) != len(all_seeds):
                errors.append(f"duplicate PSO seeds: {condition}/{case}")
            if case not in reused_cases and set(all_seeds) - seeds:
                errors.append(f"unexpected PSO seeds: {condition}/{case}")
            rows = [row for row in rows if int(row["seed"]) in seeds]
            actual_seeds = {int(row["seed"]) for row in rows}
            if actual_seeds != seeds and not args.allow_incomplete:
                errors.append(
                    f"incomplete PSO cell: {condition}/{case} "
                    f"({len(rows)}/{len(seeds)})"
                )
            for row in rows:
                context = f"PSO/{condition}/{case}/{row['seed']}"
                if row.get("evaluation_reference_mode") != "analytic-area-uniform":
                    errors.append(f"wrong reference mode: {context}")
                metadata = (
                    int(row.get("evaluation_points", -1)),
                    int(row.get("evaluation_grid", -1)),
                    int(row.get("evaluation_reference_seed", -1)),
                    int(row.get("evaluation_model_seed", -1)),
                )
                if metadata != (model_count, grid, reference_seed, model_seed):
                    errors.append(f"evaluation metadata mismatch: {context}")
                if int(row.get("evaluations", -1)) != expected_fe:
                    errors.append(f"FE mismatch in results row: {context}")
                record_path = Path(row.get("record_file", ""))
                if not record_path.exists():
                    errors.append(f"missing raw record: {context}")
                else:
                    record = load_json(record_path)
                    if int(record.get("episode", -1)) != expected_fe:
                        errors.append(f"FE mismatch in raw record: {context}")
                model = sample_trait(
                    trait_from_mapping(row["trait"]),
                    count=model_count,
                    seed=model_seed,
                    grid_resolution=grid,
                )
                recomputed = geometric_metrics(reference, model, threshold)
                update_metric_error(
                    max_error, row, recomputed, context, errors, args.tolerance
                )
                expected_success = int(recomputed["gt_chamfer"] <= success_threshold)
                if int(row.get("success", -1)) != expected_success:
                    errors.append(f"success mismatch: {context}")
                audited_pso += 1

            ems_dir = args.ems_root.resolve() / case / condition
            result_file = ems_dir / "result.json"
            eval_file = ems_dir / "evaluation.json"
            if not result_file.exists() or not eval_file.exists():
                if not args.allow_incomplete:
                    errors.append(f"missing EMS result/evaluation: {condition}/{case}")
                continue
            result = load_json(result_file)
            stored = load_json(eval_file)
            context = f"EMS/{condition}/{case}"
            if result.get("implementation_commit") != protocol["ems"]["implementation_commit"]:
                errors.append(f"EMS commit mismatch: {context}")
            settings = result.get("settings", {})
            expected_settings = {
                "outlier_ratio": protocol["ems"]["outlier_prior"],
                "max_iteration_em": protocol["ems"]["max_iteration_em"],
                "max_optimization_iterations": protocol["ems"]["max_optimization_iterations"],
                "max_switches": protocol["ems"]["max_switches"],
                "adaptive_upper_bound": protocol["ems"]["adaptive_upper_bound"],
            }
            for key, value in expected_settings.items():
                if settings.get(key) != value:
                    errors.append(f"EMS setting mismatch ({key}): {context}")
            ems_metadata = (
                int(stored.get("evaluation_points", -1)),
                int(stored.get("evaluation_grid", -1)),
                int(stored.get("reference_seed", -1)),
                int(stored.get("model_seed", -1)),
                stored.get("reference_mode"),
            )
            if ems_metadata != (
                model_count,
                grid,
                reference_seed,
                model_seed,
                "analytic-area-uniform",
            ):
                errors.append(f"EMS evaluation metadata mismatch: {context}")
            model = sample_trait(
                trait_from_mapping(result["trait"]),
                count=model_count,
                seed=model_seed,
                grid_resolution=grid,
            )
            recomputed = geometric_metrics(reference, model, threshold)
            update_metric_error(
                max_error, stored, recomputed, context, errors, args.tolerance
            )
            audited_ems += 1

    expected_pso = len(cases) * len(conditions) * len(seeds)
    expected_ems = len(cases) * len(conditions)
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "pso_results_audited": audited_pso,
        "pso_results_expected": expected_pso,
        "ems_results_audited": audited_ems,
        "ems_results_expected": expected_ems,
        "external_metric_recompute_max_abs_error": max_error,
        "tolerance": args.tolerance,
        "errors": sorted(set(errors)),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
