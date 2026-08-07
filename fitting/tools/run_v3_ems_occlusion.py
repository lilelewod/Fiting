"""Fit and independently evaluate EMS on the preregistered SQ occlusion cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "paper/ieee_superquadric/protocols/v3_stratified_superquadric_robustness.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs/ems_baseline/v3_randomized_fixedprior01",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evaluation-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing the project's evaluation dependencies.",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    data_root = Path(protocol["data_root"])
    ems = protocol["ems"]
    evaluation = protocol["independent_evaluation"]
    filename = protocol["conditions"]["occlusion_cap_80"]["file"]

    for case in protocol["cases"]:
        case_root = data_root / case
        output = args.output_root.resolve() / case / "occlusion_cap_80"
        fit_file = output / "result.json"
        evaluation_file = output / "evaluation.json"
        if not fit_file.exists():
            fit_command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/run_ems_fit.py"),
                "--data-file", str(case_root / filename),
                "--output", str(fit_file),
                "--outlier-ratio", str(ems["outlier_prior"]),
                "--max-iteration-em", str(ems["max_iteration_em"]),
                "--max-optimization-iterations", str(ems["max_optimization_iterations"]),
                "--max-switches", str(ems["max_switches"]),
            ]
            print(subprocess.list2cmdline(fit_command))
            if not args.dry_run:
                subprocess.run(fit_command, cwd=PROJECT_ROOT, check=True)
        if not evaluation_file.exists():
            evaluate_command = [
                str(args.evaluation_python.resolve()),
                str(PROJECT_ROOT / "tools/evaluate_external_superquadric.py"),
                "--fit-result", str(fit_file),
                "--ground-truth-trait", str(case_root / "trait.json"),
                "--output", str(evaluation_file),
                "--points", str(evaluation["reference_points"]),
                "--grid", str(evaluation["grid_resolution"]),
                "--reference-seed", str(evaluation["reference_seed"]),
                "--model-seed", str(evaluation["model_seed"]),
                "--threshold", str(evaluation["fscore_distance_threshold"]),
            ]
            print(subprocess.list2cmdline(evaluate_command))
            if not args.dry_run:
                subprocess.run(evaluate_command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
