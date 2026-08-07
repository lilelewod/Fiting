"""Run or resume the frozen ModelNet40 ten-object robustness experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "paper/ieee_superquadric/protocols/modelnet40_real_object_10case.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "outputs/optimizer_comparison/modelnet40_real10_guided_pso_3seeds_20260803"
)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"results file is not a list: {path}")
    return rows


def audit_rows(path: Path, seeds: list[int], expected_fe: int) -> tuple[bool, str]:
    rows = read_rows(path)
    by_seed = {int(row["seed"]): row for row in rows}
    if len(by_seed) != len(rows):
        return False, "duplicate seeds"
    missing = sorted(set(seeds) - set(by_seed))
    if missing:
        return False, f"missing seeds {missing}"
    errors = []
    for seed in seeds:
        row = by_seed[seed]
        if row.get("algorithm") != "pso":
            errors.append(f"seed {seed}: algorithm")
        if int(row.get("evaluations", -1)) != expected_fe:
            errors.append(f"seed {seed}: FE")
        if not bool(row.get("pso_guided_initialization")):
            errors.append(f"seed {seed}: guidance")
        if int(row.get("evaluation_points", -1)) != 20_000:
            errors.append(f"seed {seed}: evaluation points")
        if row.get("evaluation_reference_mode") != "provided-point-cloud-density-dependent":
            errors.append(f"seed {seed}: reference mode")
    return not errors, "; ".join(errors) if errors else "complete"


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Portable dataset-root override. Case directories are resolved as "
            "DATA_ROOT/CATEGORY/CASE instead of using paths stored in the protocol."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--seed-list", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_root = args.data_root.resolve() if args.data_root else None
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pso = protocol["guided_pso"]
    evaluation = protocol["independent_evaluation"]
    seeds = args.seed_list or [int(seed) for seed in pso["paired_base_seeds"]]
    conditions = args.conditions or list(protocol["conditions"])
    cases = args.cases or list(protocol["cases"])
    unknown_conditions = sorted(set(conditions) - set(protocol["conditions"]))
    unknown_cases = sorted(set(cases) - set(protocol["cases"]))
    if unknown_conditions or unknown_cases:
        raise ValueError(
            f"unknown conditions={unknown_conditions}, unknown cases={unknown_cases}"
        )
    expected_fe = int(pso["max_evaluations"])
    jobs = [
        {"condition": condition, "case": case, "status": "pending"}
        for condition in conditions for case in cases
    ]
    manifest_path = output_root / "job_manifest.json"
    manifest = {
        "protocol": str(protocol_path),
        "data_root_override": str(data_root) if data_root else None,
        "output_root": str(output_root),
        "conditions": conditions,
        "cases": cases,
        "seeds": seeds,
        "expected_cells": len(jobs),
        "expected_runs": len(jobs) * len(seeds),
        "jobs": jobs,
    }
    save_manifest(manifest_path, manifest)

    for job in jobs:
        condition = job["condition"]
        case = job["case"]
        if data_root is not None:
            category = protocol["case_categories"][case]
            case_root = data_root / category / case
        else:
            case_root = Path(protocol["case_directories"][case])
        if not case_root.is_dir():
            raise FileNotFoundError(
                f"case directory does not exist for {case}: {case_root}"
            )
        metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
        resolution = metadata["fixed_estimator_protocol"]
        condition_cfg = protocol["conditions"][condition]
        run_root = output_root / condition / case
        result_file = run_root / "results.json"
        complete, detail = audit_rows(result_file, seeds, expected_fe)
        if complete:
            job.update(status="complete", result_file=str(result_file), completed_runs=len(seeds))
            print(f"[skip] {condition}/{case}: {len(seeds)}/{len(seeds)}")
            save_manifest(manifest_path, manifest)
            continue

        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools/run_optimizer_comparison.py"),
            "--data-file", str(case_root / condition_cfg["file"]),
            "--ground-truth", str(case_root / evaluation["reference_file"]),
            "--algorithms", "pso",
            "--seed-list", *[str(seed) for seed in seeds],
            "--runs", str(len(seeds)),
            "--population-size", str(pso["population_size"]),
            "--max-evaluations", str(expected_fe),
            "--data-resolution", str(resolution["data_resolution"]),
            "--model-resolution", str(resolution["model_resolution"]),
            "--success-chamfer", str(evaluation["chamfer_screening_threshold"]),
            "--gt-threshold", str(evaluation["fscore_distance_threshold"]),
            "--evaluation-points", str(evaluation["reference_points"]),
            "--evaluation-grid", "256",
            "--evaluation-seed", "20260803",
            "--pso-guided-initialization",
            "--pso-guided-fraction", str(pso["guided_fraction"]),
            "--pso-guided-jitter", str(pso["guided_jitter"]),
            "--pso-guided-extent-quantile", str(pso["extent_quantile"]),
            "--pso-guided-support-fraction", str(condition_cfg["guided_support_fraction"]),
            "--pso-guided-support-neighbors", str(pso["support_neighbors"]),
            "--output-root", str(run_root),
        ]
        if result_file.exists():
            command.append("--resume")
        job.update(status="running", result_file=str(result_file), prior_state=detail)
        save_manifest(manifest_path, manifest)
        print(f"[run] {condition}/{case}: {detail}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            job["status"] = "dry-run"
        else:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            complete, detail = audit_rows(result_file, seeds, expected_fe)
            if not complete:
                raise RuntimeError(f"cell did not complete {condition}/{case}: {detail}")
            job.update(status="complete", completed_runs=len(seeds), audit=detail)
        save_manifest(manifest_path, manifest)

    manifest["status"] = "DRY_RUN" if args.dry_run else "COMPLETE"
    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
