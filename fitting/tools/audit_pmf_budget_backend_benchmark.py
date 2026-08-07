"""Audit the paired CPU/CUDA end-to-end gate for the PMF budget study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT.parent / "outputs"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON array: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-results",
        type=Path,
        default=OUTPUTS / "tmp_budget_backend_cpu_8080_20260721/results.json",
    )
    parser.add_argument(
        "--cuda-clean-results",
        type=Path,
        default=OUTPUTS / "tmp_cuda_budget_benchmark_8080_20260721/results.json",
    )
    parser.add_argument(
        "--cuda-outlier-results",
        type=Path,
        default=OUTPUTS / "tmp_budget_backend_cuda_out50_8080_20260721/results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "environment/pmf_budget_backend_benchmark_audit.json",
    )
    args = parser.parse_args()

    errors: list[str] = []
    rows = (
        load_rows(args.cpu_results.resolve())
        + load_rows(args.cuda_clean_results.resolve())
        + load_rows(args.cuda_outlier_results.resolve())
    )
    keyed: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("condition")),
            str(row.get("algorithm")),
            str(row.get("nearest_neighbor_backend")),
        )
        if key in keyed:
            errors.append(f"duplicate benchmark row: {key}")
        keyed[key] = row
        if int(row.get("evaluations", -1)) != 8080:
            errors.append(f"FE mismatch: {key}")
        if int(row.get("population_size", -1)) != 80:
            errors.append(f"population mismatch: {key}")
        if int(row.get("seed", -1)) != 20260801:
            errors.append(f"seed mismatch: {key}")

    expected = {
        (condition, algorithm, backend)
        for condition in ("clean", "outlier_50")
        for algorithm in ("pso", "cs")
        for backend in ("sklearn", "torch_cuda")
    }
    if set(keyed) != expected:
        errors.append(
            f"benchmark matrix mismatch: missing={sorted(expected - set(keyed))}, "
            f"extra={sorted(set(keyed) - expected)}"
        )

    cells: dict[str, dict] = {}
    internal_score_tolerance = 1e-6
    external_metric_tolerance = 1e-12
    for condition in ("clean", "outlier_50"):
        cells[condition] = {}
        for algorithm in ("pso", "cs"):
            cpu = keyed.get((condition, algorithm, "sklearn"))
            cuda = keyed.get((condition, algorithm, "torch_cuda"))
            if cpu is None or cuda is None:
                continue
            metric_errors = {
                name: abs(float(cpu[name]) - float(cuda[name]))
                for name in ("best_score", "gt_chamfer", "gt_fscore")
            }
            if metric_errors["best_score"] > internal_score_tolerance:
                errors.append(f"CPU/CUDA objective mismatch: {condition}/{algorithm}")
            if max(metric_errors[name] for name in ("gt_chamfer", "gt_fscore")) > external_metric_tolerance:
                errors.append(f"CPU/CUDA external-metric mismatch: {condition}/{algorithm}")
            if cpu.get("shared_seeds") != cuda.get("shared_seeds"):
                errors.append(f"internal seed mismatch: {condition}/{algorithm}")
            cpu_seconds = float(cpu["wall_time_s"])
            cuda_seconds = float(cuda["wall_time_s"])
            if not cpu_seconds < cuda_seconds:
                errors.append(f"CPU was not faster: {condition}/{algorithm}")
            cells[condition][algorithm] = {
                "cpu_seconds": cpu_seconds,
                "cuda_seconds": cuda_seconds,
                "cpu_speedup_over_cuda": cuda_seconds / cpu_seconds,
                "metric_abs_errors": metric_errors,
            }

    report = {
        "status": "PASS" if not errors else "FAIL",
        "purpose": (
            "Paired pre-start backend selection for the PMF PSO-CS budget study; "
            "this is a timing gate, not a recovery-result comparison."
        ),
        "formal_budget_results_at_gate": 0,
        "protocol": {
            "seed": 20260801,
            "population_size": 80,
            "evaluations": 8080,
            "conditions": ["clean", "outlier_50"],
            "algorithms": ["pso", "cs"],
            "backends": ["sklearn", "torch_cuda"],
        },
        "internal_score_tolerance": internal_score_tolerance,
        "external_metric_tolerance": external_metric_tolerance,
        "selected_backend": "sklearn",
        "cells": cells,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
