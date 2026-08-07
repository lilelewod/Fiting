"""Run the preregistered Guided-PSO robustness matrix on SQ v3 cases 000--008."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "paper/ieee_superquadric/protocols/v3_stratified_superquadric_robustness.json"


def count_results(path: Path) -> int:
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs/optimizer_comparison/v3_stratified9_robustness_guided_pso_5seeds_20260721",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_root = Path(protocol["data_root"])
    pso = protocol["guided_pso"]
    evaluation = protocol["independent_evaluation"]
    conditions = [name for name in protocol["conditions"] if name != "clean"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    job_manifest = {
        "protocol": str(protocol_path),
        "data_root": str(data_root.resolve()),
        "jobs": [],
    }

    for condition in conditions:
        filename = protocol["conditions"][condition]["file"]
        for case in protocol["cases"]:
            case_root = data_root / case
            metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
            resolution = metadata["fixed_estimator_protocol"]
            run_root = output_root / condition / case
            done = count_results(run_root / "results.json")
            expected = len(pso["paired_base_seeds"])
            job_manifest["jobs"].append(
                {"condition": condition, "case": case, "completed": done, "expected": expected}
            )
            if done == expected:
                print(f"[skip] {condition}/{case}: complete")
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/run_optimizer_comparison.py"),
                "--data-file", str(case_root / filename),
                "--ground-truth", str(case_root / "reference_uniform.ply"),
                "--ground-truth-trait", str(case_root / "trait.json"),
                "--algorithms", "pso",
                "--seed-list", *[str(seed) for seed in pso["paired_base_seeds"]],
                "--runs", str(expected),
                "--population-size", str(pso["population_size"]),
                "--max-evaluations", str(pso["max_evaluations"]),
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
            if (run_root / "results.json").exists():
                command.append("--resume")
            print(f"[run] {condition}/{case}: {done}/{expected}")
            if args.dry_run:
                print(subprocess.list2cmdline(command))
            else:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    (output_root / "job_manifest.json").write_text(
        json.dumps(job_manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
