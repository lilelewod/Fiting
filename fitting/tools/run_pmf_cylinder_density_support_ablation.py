"""Run the preregistered full-input versus adaptive-density PMF ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "paper/ieee_superquadric/protocols/pmf_cylinder_density_support_ablation.json"


def result_count(root: Path) -> int:
    path = root / "results.json"
    return len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs/pmf_cylinder_density_support/formal_adaptive_20260721",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["paired_base_seeds"]]
    expected = len(seeds)
    for condition in protocol["dataset"]["conditions"]:
        for variant in protocol["variants"]:
            root = args.output_root.resolve() / condition / variant["name"]
            done = result_count(root)
            if done == expected:
                print(f"[skip] {condition}/{variant['name']} complete")
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/run_pmf_cylinder_optimizer_comparison.py"),
                "--conditions", condition,
                "--algorithms", protocol["optimizer"],
                "--seed-list", *[str(seed) for seed in seeds],
                "--runs", str(expected),
                "--population-size", str(protocol["population_size"]),
                "--max-evaluations", str(protocol["max_evaluations"]),
                "--density-support-mode", variant["density_support_mode"],
                "--density-support-fraction", str(variant["density_support_fraction"]),
                "--density-support-neighbors", str(variant["density_support_neighbors"]),
                "--output-root", str(root),
            ]
            if root.exists():
                command.append("--resume")
            print(f"[run] {condition}/{variant['name']}: {done}/{expected}")
            print(subprocess.list2cmdline(command))
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
