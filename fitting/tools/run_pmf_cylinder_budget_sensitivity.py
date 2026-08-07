"""Execute the preregistered PMF-cylinder PSO--CS budget study sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "paper/ieee_superquadric/protocols/pmf_cylinder_budget_sensitivity.json"


def completed_count(root: Path) -> int:
    result_file = root / "results.json"
    if not result_file.exists():
        return 0
    return len(json.loads(result_file.read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs/pmf_cylinder_budget_sensitivity/preregistered_20260721",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    seeds = [int(value) for value in protocol["paired_base_seeds"]]
    conditions = list(protocol["dataset"]["conditions"])
    algorithms = list(protocol["algorithms"])
    population = int(protocol["population_size"])
    num_envs = int(protocol["num_envs"])
    nearest_backend = str(protocol["nearest_neighbor_backend"])
    expected = len(seeds) * len(conditions) * len(algorithms)

    equivalence_file = PROJECT_ROOT.parent / "outputs/environment/cuda_nn_equivalence_audit.json"
    if nearest_backend == "torch_cuda":
        if not equivalence_file.exists():
            raise FileNotFoundError(
                "CUDA nearest-neighbor equivalence audit must run before the budget study"
            )
        equivalence = json.loads(equivalence_file.read_text(encoding="utf-8"))
        if equivalence.get("status") != "PASS" or equivalence.get("errors"):
            raise RuntimeError("CUDA nearest-neighbor equivalence audit did not pass")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Preregistered seeds must be unique")
    if set(algorithms) != {"pso", "cs"}:
        raise ValueError("This runner is restricted to the preregistered PSO--CS comparison")

    for budget_value in protocol["evaluation_budgets"]:
        budget = int(budget_value)
        if budget < population or (budget - population) % (2 * population):
            raise ValueError(f"Budget {budget} is not exactly compatible with both optimizers")
        run_root = args.output_root.resolve() / f"fe_{budget}"
        done = completed_count(run_root)
        if done == expected:
            print(f"[skip] FE={budget}: already complete ({done}/{expected})")
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools/run_pmf_cylinder_optimizer_comparison.py"),
            "--conditions", *conditions,
            "--algorithms", *algorithms,
            "--seed-list", *[str(seed) for seed in seeds],
            "--runs", str(len(seeds)),
            "--population-size", str(population),
            "--num-envs", str(num_envs),
            "--max-evaluations", str(budget),
            "--nearest-neighbor-backend", nearest_backend,
            "--output-root", str(run_root),
        ]
        if run_root.exists():
            command.append("--resume")
        print(f"[run] FE={budget}: existing {done}/{expected}")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
