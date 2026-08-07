"""Prepare a frozen multi-category ModelNet40 robustness benchmark.

The case selection rule is fixed before fitting: within each requested category,
take the lexicographically first successful models from the existing benchmark
manifest.  Corruptions are deterministic and the independent 20k surface sample
is retained for final full-object evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from sklearn.neighbors import KDTree

from prepare_superquadric_benchmark import gross_outliers, write_ply


DEFAULT_SOURCE = Path(r"C:\code\datasets\modelnet40\superquadric_benchmark")
DEFAULT_OUTPUT = Path(r"C:\code\datasets\modelnet40\real10_robustness")
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "paper/ieee_superquadric/protocols/modelnet40_real_object_10case.json"
)
CATEGORY_COUNTS = {
    "bottle": 2,
    "bowl": 2,
    "cone": 1,
    "flower_pot": 1,
    "glass_box": 1,
    "vase": 2,
    "xbox": 1,
}


def read_ply(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    return np.asarray(points, dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_seed(base_seed: int, case_index: int) -> dict[str, int]:
    children = np.random.SeedSequence([base_seed, case_index]).spawn(4)
    names = ("noise", "outliers", "missing_view", "shuffle")
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--noise-diagonal-fraction", type=float, default=0.005)
    parser.add_argument("--outlier-fraction", type=float, default=0.20)
    parser.add_argument("--partial-retained-fraction", type=float, default=0.60)
    args = parser.parse_args()

    if not 0.0 < args.noise_diagonal_fraction < 0.1:
        raise ValueError("noise fraction must lie in (0, 0.1)")
    if not 0.0 < args.outlier_fraction < 0.5:
        raise ValueError("outlier fraction must lie in (0, 0.5)")
    if not 0.1 <= args.partial_retained_fraction < 1.0:
        raise ValueError("partial retained fraction must lie in [0.1, 1)")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    protocol_path = args.protocol.resolve()
    records = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    selected = []
    for category, count in CATEGORY_COUNTS.items():
        candidates = sorted(
            (
                row for row in records
                if row.get("status") == "ok" and row.get("category") == category
            ),
            key=lambda row: row["model"],
        )
        if len(candidates) < count:
            raise RuntimeError(f"not enough successful {category} cases")
        selected.extend(candidates[:count])
    if len(selected) != 10:
        raise AssertionError("the frozen protocol must contain exactly 10 cases")

    output_root.mkdir(parents=True, exist_ok=True)
    case_records = []
    for case_index, source in enumerate(selected):
        category = source["category"]
        model = source["model"]
        case_root = output_root / category / model
        case_root.mkdir(parents=True, exist_ok=True)
        clean = read_ply(Path(source["fit_file"]))
        reference = read_ply(Path(source["reference_file"]))
        if clean.shape != (5000, 3) or reference.shape != (20000, 3):
            raise RuntimeError(f"unexpected source sizes for {model}")
        if not np.isfinite(clean).all() or not np.isfinite(reference).all():
            raise RuntimeError(f"non-finite source points for {model}")

        seeds = case_seed(args.base_seed, case_index)
        bbox_diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
        noise_sigma = args.noise_diagonal_fraction * bbox_diagonal
        noisy = clean + np.random.default_rng(seeds["noise"]).normal(
            0.0, noise_sigma, clean.shape
        )

        outlier_count = int(round(len(clean) * args.outlier_fraction))
        outlier_rng = np.random.default_rng(seeds["outliers"])
        inlier_indices = outlier_rng.choice(
            len(clean), size=len(clean) - outlier_count, replace=False
        )
        outliers = gross_outliers(reference, outlier_count, outlier_rng)
        contaminated = np.vstack((clean[inlier_indices], outliers))
        contaminated = contaminated[
            np.random.default_rng(seeds["shuffle"]).permutation(len(contaminated))
        ]

        view_rng = np.random.default_rng(seeds["missing_view"])
        view_direction = view_rng.normal(size=3)
        view_direction /= np.linalg.norm(view_direction)
        retained_count = int(round(len(clean) * args.partial_retained_fraction))
        projection = clean @ view_direction
        retained = clean[np.argsort(projection, kind="stable")[-retained_count:]]

        nearest = KDTree(clean).query(clean, k=2)[0][:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        data_resolution = float(np.median(nearest))
        clouds = {
            "reference.ply": reference,
            "clean.ply": clean,
            "noise_0p5pct_diag.ply": noisy,
            "outlier_20.ply": contaminated,
            "partial_view_40missing.ply": retained,
        }
        hashes = {}
        for file_name, points in clouds.items():
            path = case_root / file_name
            write_ply(path, points)
            hashes[file_name] = sha256(path)

        metadata = {
            "schema_version": 1,
            "category": category,
            "model": model,
            "selection_index": case_index,
            "source_fit_file": str(Path(source["fit_file"]).resolve()),
            "source_reference_file": str(Path(source["reference_file"]).resolve()),
            "source_sampling": "triangle-area-proportional mesh-surface sampling",
            "reference_role": "independent full-mesh evaluation-only point cloud",
            "seeds": seeds,
            "reference_bbox_diagonal": bbox_diagonal,
            "fixed_estimator_protocol": {
                "source": "median positive nearest-neighbor distance of clean.ply",
                "data_resolution": data_resolution,
                "model_resolution": 0.45 * data_resolution,
                "apply_unchanged_to_all_conditions": True,
            },
            "conditions": {
                "clean": {"file": "clean.ply", "points": len(clean)},
                "noise": {
                    "file": "noise_0p5pct_diag.ply",
                    "points": len(noisy),
                    "sigma": noise_sigma,
                    "sigma_fraction_reference_bbox_diagonal": args.noise_diagonal_fraction,
                },
                "outlier_20": {
                    "file": "outlier_20.ply",
                    "points": len(contaminated),
                    "inlier_points": len(clean) - outlier_count,
                    "outlier_points": outlier_count,
                    "outlier_fraction": args.outlier_fraction,
                    "minimum_outlier_distance_fraction_bbox_diagonal": 0.05,
                },
                "partial_view": {
                    "file": "partial_view_40missing.ply",
                    "points": len(retained),
                    "retained_fraction": args.partial_retained_fraction,
                    "spatial_missing_fraction": 1.0 - args.partial_retained_fraction,
                    "selection": "largest projections along fixed view direction",
                    "view_direction": view_direction.tolist(),
                },
            },
            "sha256": hashes,
        }
        (case_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        case_records.append(
            {
                "case": model,
                "category": category,
                "directory": str(case_root),
                "metadata": str(case_root / "metadata.json"),
            }
        )
        print(
            f"{model}: category={category}, clean=5000, partial={len(retained)}, "
            f"resolution={data_resolution:.8f}"
        )

    manifest = {
        "schema_version": 1,
        "benchmark": "ModelNet40 ten-object multi-category robustness extension",
        "data_origin": "ModelNet40 CAD meshes; these are not raw physical scanner acquisitions",
        "generator": str(Path(__file__).resolve()),
        "base_seed": args.base_seed,
        "selection_rule": (
            "For each preregistered category, select the lexicographically first N "
            "successful entries in the pre-existing area-sampled benchmark manifest."
        ),
        "category_counts": CATEGORY_COUNTS,
        "cases": case_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    protocol = {
        "schema_version": 1,
        "protocol_name": "modelnet40_real_object_10case_guided_pso",
        "frozen_before_fitting": True,
        "data_root": str(output_root),
        "dataset_manifest": str(output_root / "manifest.json"),
        "selection_rule": manifest["selection_rule"],
        "cases": [row["case"] for row in case_records],
        "case_categories": {row["case"]: row["category"] for row in case_records},
        "case_directories": {row["case"]: row["directory"] for row in case_records},
        "conditions": {
            "clean": {"file": "clean.ply", "guided_support_fraction": 1.0},
            "noise": {"file": "noise_0p5pct_diag.ply", "guided_support_fraction": 1.0},
            "outlier_20": {"file": "outlier_20.ply", "guided_support_fraction": 0.75},
            "partial_view": {"file": "partial_view_40missing.ply", "guided_support_fraction": 1.0},
        },
        "guided_pso": {
            "population_size": 16,
            "max_evaluations": 10000,
            "paired_base_seeds": [20260803, 20260804, 20260805],
            "guided_fraction": 0.75,
            "guided_jitter": 0.04,
            "extent_quantile": 0.005,
            "support_neighbors": 8,
        },
        "pilot": {
            "seed": 20260803,
            "scope": "all 10 cases and all four conditions",
        },
        "independent_evaluation": {
            "reference_file": "reference.ply",
            "reference_points": 20000,
            "reference_mode": "provided independent area-sampled full mesh",
            "fscore_distance_threshold": 0.01,
            "chamfer_screening_threshold": 0.05,
            "interpretation": "shape-approximation quality, not exact parameter recovery",
        },
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved dataset: {output_root}")
    print(f"Saved frozen protocol: {protocol_path}")


if __name__ == "__main__":
    main()
