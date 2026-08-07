"""Audit label-free density-support quality on preregistered v3 outlier cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import read_point_cloud
from tools.prepare_superquadric_benchmark import gross_outliers
from tools.superquadric_evaluation import sample_trait, trait_from_mapping
from tools.superquadric_initialization import density_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/ieee_superquadric/protocols/v3_stratified_superquadric_robustness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\benchmark_audits\v3_outlier20_support_audit.json"),
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    data_root = Path(protocol["data_root"])
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    cases = list(protocol["cases"])
    support_fraction = float(protocol["guided_pso"]["initialization_support_fraction"]["outlier_20"])
    neighbors = int(protocol["guided_pso"]["support_neighbors"])
    grid = int(manifest["grid_resolution"])
    reference_points = int(manifest["reference_points"])
    errors = []
    rows = []

    for case in cases:
        case_root = data_root / case
        metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
        trait = trait_from_mapping(metadata["trait"])
        seeds = metadata["seeds"]
        total = int(metadata["conditions"]["outlier_20.ply"]["points"])
        outlier_fraction = float(
            metadata["conditions"]["outlier_20.ply"]["outlier_fraction_of_total"]
        )
        outlier_count = int(round(total * outlier_fraction))
        inlier_count = total - outlier_count

        reference = sample_trait(
            trait,
            reference_points,
            int(seeds["reference"]),
            grid,
        )
        rng = np.random.default_rng(int(seeds["outlier_20"]))
        inliers = sample_trait(trait, inlier_count, int(seeds["outlier_20"]) + 1, grid)
        outliers = gross_outliers(reference, outlier_count, rng)
        regenerated = np.vstack((inliers, outliers))
        is_inlier = np.concatenate(
            (np.ones(inlier_count, dtype=bool), np.zeros(outlier_count, dtype=bool))
        )
        permutation = rng.permutation(total)
        regenerated = regenerated[permutation]
        is_inlier = is_inlier[permutation]
        stored = read_point_cloud(str(case_root / "outlier_20.ply"))
        regeneration_error = float(
            np.max(np.abs(regenerated.astype(np.float32).astype(np.float64) - stored), initial=0.0)
        )
        if regeneration_error != 0.0:
            errors.append(f"{case}: contaminated cloud or hidden audit labels cannot be regenerated")

        k = min(max(2, neighbors), total - 1)
        distances = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(stored).kneighbors(
            stored, return_distance=True
        )[0][:, -1]
        keep_count = max(4, int(np.floor(total * support_fraction)))
        selected = np.argpartition(distances, keep_count - 1)[:keep_count]
        selected_sorted = np.sort(selected)
        production_support = density_support(stored, support_fraction, neighbors)
        selection_error = float(
            np.max(np.abs(production_support - stored[selected_sorted]), initial=0.0)
        )
        if selection_error != 0.0:
            errors.append(f"{case}: audited support does not match production density_support")

        retained_inliers = int(is_inlier[selected].sum())
        retained_outliers = int(keep_count - retained_inliers)
        rows.append(
            {
                "case": case,
                "input_points": total,
                "true_inliers": inlier_count,
                "true_outliers": outlier_count,
                "retained_points": keep_count,
                "retained_inliers": retained_inliers,
                "retained_outliers": retained_outliers,
                "inlier_precision": float(retained_inliers / keep_count),
                "inlier_recall": float(retained_inliers / inlier_count),
                "cloud_regeneration_max_abs_error": regeneration_error,
                "production_selection_max_abs_error": selection_error,
            }
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "protocol": str(args.protocol.resolve()),
        "support_rule": {
            "label_free_at_fit_time": True,
            "support_fraction": support_fraction,
            "neighbors": neighbors,
            "selection": "smallest k-nearest-neighbor radii",
        },
        "cases_audited": len(rows),
        "minimum_inlier_precision": float(min(row["inlier_precision"] for row in rows)),
        "minimum_inlier_recall": float(min(row["inlier_recall"] for row in rows)),
        "maximum_retained_outliers": int(max(row["retained_outliers"] for row in rows)),
        "scope_warning": (
            "Quality is descriptive for the registered uniform-volume gross-outlier generator; "
            "hidden labels are used only after label-free selection and do not imply general outlier robustness."
        ),
        "cases": rows,
        "errors": errors,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
