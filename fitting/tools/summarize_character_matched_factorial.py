"""Audit, summarize, and plot the matched-character factorial experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from exact_statistics import exact_wilcoxon_signed_rank


METHODS = ("pso", "guided_pso", "cs", "guided_cs")
METHOD_LABELS = {
    "pso": "Vanilla PSO",
    "guided_pso": "Guided PSO",
    "cs": "Vanilla CS",
    "guided_cs": "Guided CS",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def quantiles(values) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--figure-prefix", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.experiment_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = args.output_root or args.experiment_root / "summary"
    figure_prefix = args.figure_prefix or output_root / "character_dimension_effect"

    cases = [tuple(case) for case in manifest["cases"]]
    repeats = [int(value) for value in manifest["repeats"]]
    expected_keys = {
        (repeat, run_id, test_id, method)
        for repeat in repeats
        for run_id, test_id in cases
        for method in METHODS
    }
    cells = manifest["cells"]
    actual_keys = [
        (int(cell["repeat"]), int(cell["run_id"]), int(cell["test_id"]), cell["method"])
        for cell in cells
    ]
    counts = Counter(actual_keys)
    errors = []
    if manifest.get("status") != "COMPLETE":
        errors.append(f"manifest status is {manifest.get('status')}")
    if set(actual_keys) != expected_keys:
        errors.append("cell key set does not match the preregistered factorial")
    duplicate_keys = [key for key, count in counts.items() if count != 1]
    if duplicate_keys:
        errors.append(f"non-unique cell keys: {duplicate_keys}")

    by_key = {}
    seed_groups = defaultdict(set)
    dimensions = defaultdict(set)
    for cell in cells:
        cell_key = (
            int(cell["repeat"]),
            int(cell["run_id"]),
            int(cell["test_id"]),
            str(cell["method"]),
        )
        if cell.get("status") != "COMPLETE" or cell.get("return_code") != 0:
            errors.append(f"incomplete cell {cell_key}")
            continue
        record_path = Path(cell["record"])
        if not record_path.is_file():
            errors.append(f"record missing for {cell_key}: {record_path}")
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if int(record.get("num_evaluations", -1)) != int(manifest["max_evaluations"]):
            errors.append(f"FE mismatch for {cell_key}")
        if int(record.get("action_dim", -1)) != int(cell["action_dim"]):
            errors.append(f"action dimension mismatch for {cell_key}")
        if not np.isclose(float(record["best_score"]), float(cell["score"]), rtol=0, atol=1e-12):
            errors.append(f"score mismatch for {cell_key}")
        if not np.isclose(float(record["chamfer"]), float(cell["chamfer"]), rtol=0, atol=1e-12):
            errors.append(f"Chamfer mismatch for {cell_key}")
        guided = cell["method"].startswith("guided_")
        initialization = record.get("guided_initialization")
        if guided:
            if not initialization:
                errors.append(f"guided metadata missing for {cell_key}")
            elif (
                initialization.get("mode") != "template_zero_action_with_gaussian_neighborhood"
                or initialization.get("guided_count") != 8
                or initialization.get("random_count") != 8
                or not np.isclose(initialization.get("guided_sigma"), 0.15)
            ):
                errors.append(f"guided metadata invalid for {cell_key}")
        elif initialization:
            errors.append(f"unexpected guided metadata for {cell_key}")
        seed_groups[cell_key[:3]].add(int(cell["seed"]))
        dimensions[(cell_key[1], cell_key[2])].add(int(cell["action_dim"]))
        by_key[cell_key] = cell

    for group, seeds in seed_groups.items():
        if len(seeds) != 1:
            errors.append(f"methods are not seed-paired for {group}: {sorted(seeds)}")
    for case, values in dimensions.items():
        if len(values) != 1:
            errors.append(f"action dimension inconsistent for {case}: {sorted(values)}")

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest_path),
        "expected_cells": len(expected_keys),
        "observed_cells": len(cells),
        "complete_cells": sum(cell.get("status") == "COMPLETE" for cell in cells),
        "failed_cells": sum(cell.get("status") == "FAILED" for cell in cells),
        "required_evaluations": int(manifest["max_evaluations"]),
        "paired_seed_groups": len(seed_groups),
        "cases": len(cases),
        "repeats": len(repeats),
        "methods": list(METHODS),
        "errors": errors,
    }
    write_json(output_root / "audit.json", audit)
    if errors:
        raise RuntimeError("factorial audit failed; see audit.json")

    rows = []
    gains = {"pso": {"score": [], "chamfer": []}, "cs": {"score": [], "chamfer": []}}
    dimensions_ordered = []
    for run_id, test_id in cases:
        case = (run_id, test_id)
        dimension = next(iter(dimensions[case]))
        dimensions_ordered.append(dimension)
        row = {"case": f"r{run_id}-t{test_id}", "run_id": run_id, "test_id": test_id, "action_dim": dimension}
        method_values = {}
        for method in METHODS:
            scores = [float(by_key[(repeat, run_id, test_id, method)]["score"]) for repeat in repeats]
            chamfers = [float(by_key[(repeat, run_id, test_id, method)]["chamfer"]) for repeat in repeats]
            method_values[method] = {"scores": scores, "chamfers": chamfers}
            row[f"{method}_score_median"] = float(np.median(scores))
            row[f"{method}_chamfer_median"] = float(np.median(chamfers))
        for optimizer in ("pso", "cs"):
            guided = f"guided_{optimizer}"
            score_differences = np.asarray(method_values[guided]["scores"]) - np.asarray(method_values[optimizer]["scores"])
            chamfer_differences = np.asarray(method_values[optimizer]["chamfers"]) - np.asarray(method_values[guided]["chamfers"])
            sq1, smedian, sq3 = quantiles(score_differences)
            cq1, cmedian, cq3 = quantiles(chamfer_differences)
            row[f"{optimizer}_score_gain_q1"] = sq1
            row[f"{optimizer}_score_gain_median"] = smedian
            row[f"{optimizer}_score_gain_q3"] = sq3
            row[f"{optimizer}_chamfer_reduction_q1"] = cq1
            row[f"{optimizer}_chamfer_reduction_median"] = cmedian
            row[f"{optimizer}_chamfer_reduction_q3"] = cq3
            gains[optimizer]["score"].append(smedian)
            gains[optimizer]["chamfer"].append(cmedian)
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "case_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methods_summary = {}
    for method in METHODS:
        case_score_medians = [row[f"{method}_score_median"] for row in rows]
        case_chamfer_medians = [row[f"{method}_chamfer_median"] for row in rows]
        methods_summary[method] = {
            "label": METHOD_LABELS[method],
            "score_case_median": float(np.median(case_score_medians)),
            "score_case_iqr": [float(x) for x in np.quantile(case_score_medians, [0.25, 0.75])],
            "chamfer_case_median": float(np.median(case_chamfer_medians)),
            "chamfer_case_iqr": [float(x) for x in np.quantile(case_chamfer_medians, [0.25, 0.75])],
        }

    comparisons = {}
    for optimizer in ("pso", "cs"):
        score_gain = np.asarray(gains[optimizer]["score"])
        chamfer_gain = np.asarray(gains[optimizer]["chamfer"])
        score_test = exact_wilcoxon_signed_rank(score_gain)
        chamfer_test = exact_wilcoxon_signed_rank(chamfer_gain)
        score_rho = spearmanr(dimensions_ordered, score_gain).statistic
        chamfer_rho = spearmanr(dimensions_ordered, chamfer_gain).statistic
        comparisons[optimizer] = {
            "score_gain_definition": f"guided_{optimizer} minus {optimizer}",
            "score_gain_case_median": float(np.median(score_gain)),
            "score_gain_positive_cases": int(np.sum(score_gain > 0)),
            "score_gain_exact_wilcoxon": score_test,
            "score_gain_dimension_spearman_rho_descriptive": float(score_rho),
            "chamfer_reduction_definition": f"{optimizer} minus guided_{optimizer}",
            "chamfer_reduction_case_median": float(np.median(chamfer_gain)),
            "chamfer_reduction_positive_cases": int(np.sum(chamfer_gain > 0)),
            "chamfer_reduction_exact_wilcoxon": chamfer_test,
            "chamfer_reduction_dimension_spearman_rho_descriptive": float(chamfer_rho),
        }

    summary = {
        "status": "PASS",
        "design": {
            "cases": len(cases),
            "repeats_per_cell": len(repeats),
            "methods": list(METHODS),
            "cells": len(cells),
            "max_evaluations": int(manifest["max_evaluations"]),
            "population": int(manifest["population"]),
            "statistical_unit": "case median across three paired optimization repeats",
        },
        "action_dimensions": sorted(set(dimensions_ordered)),
        "methods": methods_summary,
        "guided_vs_vanilla": comparisons,
        "rows": rows,
    }
    write_json(output_root / "summary.json", summary)

    configure_plot()
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    style = {
        "pso": {"label": "PSO: Guided - Vanilla", "color": "#2f6fb0", "marker": "o", "offset": -0.35},
        "cs": {"label": "CS: Guided - Vanilla", "color": "#d95f02", "marker": "s", "offset": 0.35},
    }
    panels = (
        ("score", "(a) PMF similarity gain", "Guided - Vanilla similarity"),
        ("chamfer", "(b) Chamfer reduction", "Vanilla - Guided Chamfer"),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels):
        axis.axhline(0.0, color="#68717a", linewidth=0.8, linestyle="--", zorder=0)
        for optimizer in ("pso", "cs"):
            values = np.asarray(gains[optimizer][metric])
            q1 = np.asarray([row[f"{optimizer}_{'score_gain' if metric == 'score' else 'chamfer_reduction'}_q1"] for row in rows])
            q3 = np.asarray([row[f"{optimizer}_{'score_gain' if metric == 'score' else 'chamfer_reduction'}_q3"] for row in rows])
            x = np.asarray(dimensions_ordered, dtype=float) + style[optimizer]["offset"]
            yerr = np.vstack([values - q1, q3 - values])
            rho = comparisons[optimizer][f"{metric if metric == 'score' else 'chamfer_reduction'}_gain_dimension_spearman_rho_descriptive"] if metric == "score" else comparisons[optimizer]["chamfer_reduction_dimension_spearman_rho_descriptive"]
            axis.errorbar(
                x,
                values,
                yerr=yerr,
                fmt=style[optimizer]["marker"],
                color=style[optimizer]["color"],
                markerfacecolor="white",
                markeredgewidth=1.0,
                markersize=4.5,
                elinewidth=0.9,
                capsize=2.2,
                linestyle="none",
                label=f"{style[optimizer]['label']} (rho={rho:.2f})",
            )
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Action dimension")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#d8dde3", linewidth=0.6)
        axis.legend(frameon=False, loc="upper left" if metric == "score" else "lower left")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(figure_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
