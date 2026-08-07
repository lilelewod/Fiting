"""Run the frozen 30-case Guided-PSO extension with audited result reuse."""

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
    / "paper/ieee_superquadric/protocols/v3_superquadric_30case_extension.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "outputs/optimizer_comparison/v3_randomized30_guided_pso_3seeds_20260727"
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
        if row.get("evaluation_reference_mode") != "analytic-area-uniform":
            errors.append(f"seed {seed}: reference mode")
    return not errors, "; ".join(errors) if errors else "complete"


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_root = Path(protocol["data_root"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "job_manifest.json"
    pso = protocol["guided_pso"]
    evaluation = protocol["independent_evaluation"]
    seeds = [int(seed) for seed in pso["paired_base_seeds"]]
    expected_fe = int(pso["max_evaluations"])
    reused = set(protocol["reuse_completed"]["cases"])
    clean_root = Path(protocol["reuse_completed"]["clean_root"])
    robustness_root = Path(protocol["reuse_completed"]["robustness_root"])

    jobs = []
    for condition in protocol["conditions"]:
        for case in protocol["cases"]:
            jobs.append({"condition": condition, "case": case, "status": "pending"})
    manifest = {
        "protocol": str(protocol_path),
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root),
        "expected_cells": len(jobs),
        "expected_new_runs": (len(protocol["cases"]) - len(reused)) * len(protocol["conditions"]) * len(seeds),
        "jobs": jobs,
    }
    save_manifest(manifest_path, manifest)

    for job in jobs:
        condition = job["condition"]
        case = job["case"]
        case_root = data_root / case
        if not case_root.is_dir():
            raise FileNotFoundError(case_root)
        if case in reused:
            result_file = (
                clean_root / case / "results.json"
                if condition == "clean"
                else robustness_root / condition / case / "results.json"
            )
            complete, detail = audit_rows(result_file, seeds, expected_fe)
            if not complete:
                raise RuntimeError(f"invalid reusable cell {condition}/{case}: {detail}")
            job.update(status="reused", result_file=str(result_file), completed_runs=len(seeds))
            print(f"[reuse] {condition}/{case}: {len(seeds)}/{len(seeds)}")
            save_manifest(manifest_path, manifest)
            continue

        metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
        resolution = metadata["fixed_estimator_protocol"]
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
            "--data-file", str(case_root / protocol["conditions"][condition]["file"]),
            "--ground-truth", str(case_root / "reference_uniform.ply"),
            "--ground-truth-trait", str(case_root / "trait.json"),
            "--algorithms", "pso",
            "--seed-list", *[str(seed) for seed in seeds],
            "--runs", str(len(seeds)),
            "--population-size", str(pso["population_size"]),
            "--max-evaluations", str(expected_fe),
            "--data-resolution", str(resolution["data_resolution"]),
            "--model-resolution", str(resolution["model_resolution"]),
            "--success-chamfer", str(evaluation["chamfer_success_threshold"]),
            "--gt-threshold", str(evaluation["fscore_distance_threshold"]),
            "--evaluation-points", str(evaluation["reference_points"]),
            "--evaluation-grid", str(evaluation["grid_resolution"]),
            "--evaluation-seed", str(evaluation["reference_seed"]),
            "--pso-guided-initialization",
            "--pso-guided-fraction", str(pso["guided_fraction"]),
            "--pso-guided-jitter", str(pso["guided_jitter"]),
            "--pso-guided-extent-quantile", str(pso["extent_quantile"]),
            "--pso-guided-support-fraction", str(pso["initialization_support_fraction"][condition]),
            "--pso-guided-support-neighbors", str(pso["support_neighbors"]),
            "--output-root", str(run_root),
        ]
        if result_file.exists():
            command.append("--resume")
        job.update(status="running", result_file=str(result_file), prior_state=detail)
        save_manifest(manifest_path, manifest)
        print(f"[run] {condition}/{case}: {detail}")
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
