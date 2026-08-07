"""Evaluate uniform and area-weighted superquadric surface integration.

The dense, area-weighted directed distance is treated as a numerical reference.
Low-resolution uniform and area-weighted estimates use exactly the same points;
only their integration weights differ.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.surface.superquadric_rule import SuperquadricRule


SHAPES = {
    "sphere": (1.0, 1.0, 1.0, 1.0, 1.0),
    "ellipsoid": (1.4, 0.9, 0.6, 1.0, 1.0),
    "cylinder_like": (1.0, 1.0, 1.4, 1.0, 0.25),
    "box_like": (1.2, 0.9, 0.7, 0.25, 0.25),
    "flat": (1.2, 0.9, 0.25, 0.8, 1.0),
}

PERTURBATIONS = {
    "translation": {"translation": np.array([0.08, -0.04, 0.05])},
    "anisotropic_scale": {"scale_factor": np.array([1.08, 0.94, 1.05])},
    "combined": {
        "translation": np.array([-0.05, 0.06, 0.04]),
        "scale_factor": np.array([0.95, 1.07, 1.04]),
        "shape_factor": np.array([1.12, 0.90]),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[16, 24, 32, 48, 64, 96])
    parser.add_argument("--reference-eta", type=int, default=384)
    parser.add_argument("--reference-omega", type=int, default=192)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT.parent / "outputs" / "area_quadrature"),
    )
    return parser.parse_args()


def surface(parameters, n_eta, n_omega):
    return SuperquadricRule._spherical_product(
        *parameters,
        n_eta=n_eta,
        n_omega=n_omega,
        return_weights=True,
    )


def perturbed_parameters(parameters, perturbation):
    values = np.asarray(parameters, dtype=float).copy()
    values[:3] *= perturbation.get("scale_factor", 1.0)
    values[3:] *= perturbation.get("shape_factor", 1.0)
    values[3:] = np.clip(values[3:], 0.10, 2.50)
    return tuple(values)


def apply_translation(points, perturbation):
    return points + perturbation.get("translation", 0.0)


def directed_distances(points, target_tree):
    return target_tree.query(points, k=1, workers=-1)[0]


def main():
    args = parse_args()
    if min(args.resolutions) < 8:
        raise ValueError("all eta resolutions must be at least 8")
    if args.reference_eta < max(args.resolutions):
        raise ValueError("reference resolution must exceed low resolutions")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for shape_name, true_parameters in SHAPES.items():
        target_points, _, _ = surface(
            true_parameters, args.reference_eta, args.reference_omega
        )
        target_tree = cKDTree(target_points)

        for perturbation_name, perturbation in PERTURBATIONS.items():
            candidate_parameters = perturbed_parameters(true_parameters, perturbation)
            reference_points, reference_weights, _ = surface(
                candidate_parameters, args.reference_eta, args.reference_omega
            )
            reference_points = apply_translation(reference_points, perturbation)
            reference_distances = directed_distances(reference_points, target_tree)
            reference_mean = float(
                np.dot(reference_distances, reference_weights)
                / np.sum(reference_weights)
            )

            for n_eta in args.resolutions:
                n_omega = max(5, n_eta // 2)
                points, weights, _ = surface(candidate_parameters, n_eta, n_omega)
                points = apply_translation(points, perturbation)
                distances = directed_distances(points, target_tree)
                uniform = float(np.mean(distances))
                weighted = float(np.dot(distances, weights) / np.sum(weights))
                denominator = max(abs(reference_mean), np.finfo(float).eps)
                uniform_error = abs(uniform - reference_mean) / denominator
                weighted_error = abs(weighted - reference_mean) / denominator
                rows.append({
                    "shape": shape_name,
                    "perturbation": perturbation_name,
                    "n_eta": n_eta,
                    "n_omega": n_omega,
                    "num_points": n_eta * n_omega,
                    "reference_mean": reference_mean,
                    "uniform_mean": uniform,
                    "area_weighted_mean": weighted,
                    "uniform_relative_error": uniform_error,
                    "area_weighted_relative_error": weighted_error,
                    "weighted_wins": int(weighted_error < uniform_error),
                })

    fieldnames = list(rows[0])
    with open(output_root / "quadrature_results.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for n_eta in args.resolutions:
        selected = [row for row in rows if row["n_eta"] == n_eta]
        summaries.append({
            "n_eta": n_eta,
            "cases": len(selected),
            "uniform_mape": float(np.mean([row["uniform_relative_error"] for row in selected])),
            "area_weighted_mape": float(np.mean([row["area_weighted_relative_error"] for row in selected])),
            "weighted_win_rate": float(np.mean([row["weighted_wins"] for row in selected])),
        })

    by_shape = []
    for shape_name in SHAPES:
        selected = [row for row in rows if row["shape"] == shape_name]
        by_shape.append({
            "shape": shape_name,
            "cases": len(selected),
            "uniform_mape": float(np.mean([row["uniform_relative_error"] for row in selected])),
            "area_weighted_mape": float(np.mean([row["area_weighted_relative_error"] for row in selected])),
            "weighted_win_rate": float(np.mean([row["weighted_wins"] for row in selected])),
        })

    summary = {
        "reference_resolution": [args.reference_eta, args.reference_omega],
        "by_resolution": summaries,
        "by_shape": by_shape,
    }
    with open(output_root / "quadrature_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved quadrature ablation to: {output_root}")


if __name__ == "__main__":
    main()
