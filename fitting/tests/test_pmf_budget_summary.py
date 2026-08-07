import json
import sys

import numpy as np

from tools.summarize_pmf_cylinder_budget_sensitivity import main


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _paired_cell():
    description = {"count": 2, "median": 0.5, "q1": 0.4, "q3": 0.6}
    return {
        "paired_runs": 2,
        "pso_chamfer": description,
        "cs_chamfer": description,
        "pso_successes": 1,
        "cs_successes": 1,
        "pso_wins": 1,
        "cs_wins": 1,
        "wilcoxon_exact_two_sided_p": 1.0,
    }


def test_budget_endpoints_are_paired_by_seed_not_row_order(tmp_path, monkeypatch):
    budgets = (50000, 199920, 499920)
    seeds = (11, 22)
    conditions = ("clean", "outlier_50")
    algorithms = ("pso", "cs")
    protocol = {
        "evaluation_budgets": list(budgets),
        "dataset": {"conditions": list(conditions)},
        "algorithms": list(algorithms),
        "nearest_neighbor_backend": "sklearn",
        "paired_base_seeds": list(seeds),
    }
    protocol_path = tmp_path / "protocol.json"
    root = tmp_path / "runs"
    _write_json(protocol_path, protocol)

    endpoint_values = {
        50000: {11: 1.0, 22: 10.0},
        199920: {11: 2.0, 22: 8.0},
        499920: {11: 3.0, 22: 7.0},
    }
    for budget in budgets:
        rows = []
        # Deliberately reverse row order at the high endpoint. Correct pairing
        # must still produce differences (+2, -3), not differences by position.
        row_seeds = tuple(reversed(seeds)) if budget == 499920 else seeds
        for condition in conditions:
            for algorithm in algorithms:
                for seed in row_seeds:
                    value = endpoint_values[budget][seed]
                    if condition == "outlier_50":
                        value += 100.0
                    if algorithm == "cs":
                        value += 10.0
                    rows.append(
                        {
                            "condition": condition,
                            "algorithm": algorithm,
                            "seed": seed,
                            "evaluations": budget,
                            "nearest_neighbor_backend": "sklearn",
                            "gt_chamfer": value,
                        }
                    )
        audit = {
            "status": "PASS",
            "completed_results": len(rows),
            "errors": [],
            "paired_summary": {condition: _paired_cell() for condition in conditions},
        }
        _write_json(root / f"fe_{budget}" / "results.json", rows)
        _write_json(root / f"fe_{budget}" / "audit.json", audit)

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize", str(root), "--protocol", str(protocol_path)],
    )
    main()

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    endpoint = summary["within_algorithm"]["clean"]["pso"]["50000_to_499920"]
    assert summary["status"] == "PASS"
    assert endpoint["higher_budget_wins"] == 1
    assert endpoint["lower_budget_wins"] == 1
    assert np.isclose(endpoint["paired_high_minus_low_chamfer"]["median"], -0.5)
