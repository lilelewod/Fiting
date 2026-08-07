"""Summarize the preregistered PMF-cylinder PSO--CS FE-budget study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.exact_statistics import exact_wilcoxon_signed_rank


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def exact_test(differences):
    return exact_wilcoxon_signed_rank(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(r"C:\code\Fiting\outputs\pmf_cylinder_budget_sensitivity\preregistered_20260721"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/ieee_superquadric/protocols/pmf_cylinder_budget_sensitivity.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    protocol = load(args.protocol.resolve())
    budgets = [int(value) for value in protocol["evaluation_budgets"]]
    conditions = list(protocol["dataset"]["conditions"])
    algorithms = list(protocol["algorithms"])
    expected_backend = str(protocol["nearest_neighbor_backend"])
    seeds = {int(value) for value in protocol["paired_base_seeds"]}
    expected_results = len(conditions) * len(algorithms) * len(seeds)
    errors = []
    rows_by_budget = {}
    summary_budgets = {}
    csv_rows = []

    for budget in budgets:
        budget_root = root / f"fe_{budget}"
        audit_file = budget_root / "audit.json"
        results_file = budget_root / "results.json"
        if not audit_file.exists() or not results_file.exists():
            if not args.allow_incomplete:
                errors.append(f"missing audited budget: {budget}")
            continue
        audit = load(audit_file)
        rows = load(results_file)
        if audit.get("status") != "PASS":
            errors.append(f"failed source audit: {budget}")
        if int(audit.get("completed_results", -1)) != expected_results:
            errors.append(
                f"incomplete source audit: {budget} "
                f"({audit.get('completed_results')}/{expected_results})"
            )
        if audit.get("errors"):
            errors.append(f"source audit contains errors: {budget}")
        keyed = {}
        for row in rows:
            key = (row["condition"], row["algorithm"], int(row["seed"]))
            if key in keyed:
                errors.append(f"duplicate row: {budget}/{key}")
            keyed[key] = row
            if int(row.get("evaluations", -1)) != budget:
                errors.append(f"FE mismatch: {budget}/{key}")
            if row.get("nearest_neighbor_backend") != expected_backend:
                errors.append(f"nearest-neighbor backend mismatch: {budget}/{key}")
        expected_keys = {
            (condition, algorithm, seed)
            for condition in conditions
            for algorithm in algorithms
            for seed in seeds
        }
        if set(keyed) != expected_keys:
            errors.append(f"paired-key mismatch: {budget}")
        rows_by_budget[budget] = keyed
        budget_conditions = {}
        for condition in conditions:
            cell = audit.get("paired_summary", {}).get(condition)
            if cell is None or int(cell.get("paired_runs", -1)) != len(seeds):
                errors.append(f"missing paired audit cell: {budget}/{condition}")
                continue
            budget_conditions[condition] = cell
            csv_rows.append(
                {
                    "budget": budget,
                    "condition": condition,
                    "pso_median_chamfer": cell["pso_chamfer"]["median"],
                    "pso_q1_chamfer": cell["pso_chamfer"]["q1"],
                    "pso_q3_chamfer": cell["pso_chamfer"]["q3"],
                    "pso_successes": cell["pso_successes"],
                    "cs_median_chamfer": cell["cs_chamfer"]["median"],
                    "cs_q1_chamfer": cell["cs_chamfer"]["q1"],
                    "cs_q3_chamfer": cell["cs_chamfer"]["q3"],
                    "cs_successes": cell["cs_successes"],
                    "pso_wins": cell["pso_wins"],
                    "cs_wins": cell["cs_wins"],
                    "pso_vs_cs_exact_p": cell.get("wilcoxon_exact_two_sided_p"),
                }
            )
        summary_budgets[str(budget)] = {"conditions": budget_conditions}

    within_algorithm = {}
    if set(rows_by_budget) == set(budgets):
        comparisons = list(zip(budgets[:-1], budgets[1:])) + [(budgets[0], budgets[-1])]
        for condition in conditions:
            within_algorithm[condition] = {}
            for algorithm in algorithms:
                algorithm_cells = {}
                for low, high in comparisons:
                    differences = np.asarray(
                        [
                            float(rows_by_budget[high][(condition, algorithm, seed)]["gt_chamfer"])
                            - float(rows_by_budget[low][(condition, algorithm, seed)]["gt_chamfer"])
                            for seed in sorted(seeds)
                        ],
                        dtype=float,
                    )
                    algorithm_cells[f"{low}_to_{high}"] = {
                        "paired_high_minus_low_chamfer": describe(differences),
                        "higher_budget_wins": int(np.sum(differences < 0.0)),
                        "lower_budget_wins": int(np.sum(differences > 0.0)),
                        "wilcoxon": exact_test(differences),
                    }
                within_algorithm[condition][algorithm] = algorithm_cells

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "protocol": str(args.protocol.resolve()),
        "nearest_neighbor_backend": expected_backend,
        "budgets": summary_budgets,
        "within_algorithm": within_algorithm,
        "errors": sorted(set(errors)),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if csv_rows:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
