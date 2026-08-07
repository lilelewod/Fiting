"""Generate a documented, area-uniform synthetic superquadric benchmark."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.superquadric_evaluation import sample_trait, trait_from_mapping


SHAPES = {
    "ellipsoid": {
        "center": [0.0, 0.0, 0.0],
        "scale": [1.0, 0.8, 0.6],
        "shape": [1.0, 1.0],
        "rotation": [0.20, -0.25, 0.35],
    },
    "box": {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.8, 0.6, 0.5],
        "shape": [0.2, 0.2],
        "rotation": [-0.30, 0.20, 0.45],
    },
    "cylinder": {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.55, 0.55, 1.4],
        "shape": [1.0, 0.2],
        "rotation": [0.35, -0.25, 0.15],
    },
    "elongated": {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.35, 0.30, 1.8],
        "shape": [1.0, 1.0],
        "rotation": [0.15, 0.40, -0.20],
    },
    "flat": {
        "center": [0.0, 0.0, 0.0],
        "scale": [1.2, 1.0, 0.18],
        "shape": [1.0, 0.35],
        "rotation": [-0.25, 0.15, 0.30],
    },
}


def write_ply(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    vertices = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"], vertices["y"], vertices["z"] = points.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def gross_outliers(reference, count, rng, minimum_distance_fraction=0.05):
    reference = np.asarray(reference, dtype=np.float64)
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    extent = hi - lo
    diagonal = float(np.linalg.norm(extent))
    lower = lo - 0.5 * extent
    upper = hi + 0.5 * extent
    minimum_distance = minimum_distance_fraction * diagonal
    tree = KDTree(reference)
    accepted = []
    remaining = int(count)
    while remaining:
        candidates = rng.uniform(lower, upper, size=(max(remaining * 3, 1024), 3))
        distances = tree.query(candidates, k=1)[0].ravel()
        selected = candidates[distances >= minimum_distance]
        take = min(remaining, len(selected))
        if take:
            accepted.append(selected[:take])
            remaining -= take
    return np.vstack(accepted).astype(np.float32)


def trait_payload(mapping):
    payload = {key: list(value) for key, value in mapping.items()}
    payload["rotation_matrix"] = Rotation.from_euler("xyz", payload["rotation"]).as_matrix().tolist()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="datasets/superquadric_v2")
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    parser.add_argument("--observation-points", type=int, default=5000)
    parser.add_argument("--reference-points", type=int, default=20000)
    parser.add_argument("--grid-resolution", type=int, default=256)
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument("--noise-diagonal-fraction", type=float, default=0.01)
    args = parser.parse_args()

    if args.observation_points < 100 or args.reference_points < args.observation_points:
        raise ValueError("reference-points must be at least observation-points, and observation-points >= 100")
    if args.grid_resolution < 32 or args.noise_diagonal_fraction <= 0.0:
        raise ValueError("grid-resolution must be >= 32 and noise-diagonal-fraction must be positive")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "generator": str(Path(__file__).resolve()),
        "base_seed": args.base_seed,
        "surface_sampling": "triangle-area-proportional with uniform barycentric coordinates",
        "grid_resolution": args.grid_resolution,
        "observation_points": args.observation_points,
        "reference_points": args.reference_points,
        "shapes": [],
    }

    for shape_index, shape_name in enumerate(args.shapes):
        shape_root = output_root / shape_name
        shape_root.mkdir(parents=True, exist_ok=True)
        payload = trait_payload(SHAPES[shape_name])
        trait = trait_from_mapping(payload)
        seed_sequence = np.random.SeedSequence([args.base_seed, shape_index])
        seeds = [int(value) for value in seed_sequence.generate_state(8)]

        reference = sample_trait(trait, args.reference_points, seeds[0], args.grid_resolution)
        clean = sample_trait(trait, args.observation_points, seeds[1], args.grid_resolution)
        clean_nearest = KDTree(clean).query(clean, k=2)[0][:, 1]
        clean_data_resolution = float(np.median(clean_nearest))
        clean_model_resolution = 0.45 * clean_data_resolution
        noise_base = sample_trait(trait, args.observation_points, seeds[2], args.grid_resolution)
        diagonal = float(np.linalg.norm(reference.max(axis=0) - reference.min(axis=0)))
        noise_sigma = args.noise_diagonal_fraction * diagonal
        noise_rng = np.random.default_rng(seeds[3])
        noisy = noise_base + noise_rng.normal(0.0, noise_sigma, noise_base.shape).astype(np.float32)

        condition_clouds = {
            "reference_uniform.ply": reference,
            "clean.ply": clean,
            "noise_1pct_diag.ply": noisy,
        }
        for fraction, seed, name in (
            (0.20, seeds[4], "outlier_20.ply"),
            (0.50, seeds[5], "outlier_50.ply"),
        ):
            rng = np.random.default_rng(seed)
            outlier_count = int(round(args.observation_points * fraction))
            inlier_count = args.observation_points - outlier_count
            inliers = sample_trait(trait, inlier_count, seed + 1, args.grid_resolution)
            outliers = gross_outliers(reference, outlier_count, rng)
            contaminated = np.vstack((inliers, outliers))
            condition_clouds[name] = contaminated[rng.permutation(len(contaminated))]

        missing_count = int(round(0.20 * args.observation_points))
        condition_clouds["missing_80.ply"] = sample_trait(
            trait, missing_count, seeds[6], args.grid_resolution
        )
        partial_candidates = sample_trait(
            trait, args.observation_points * 2, seeds[7], args.grid_resolution
        )
        view_direction = np.asarray([1.0, -0.4, 0.7], dtype=np.float64)
        view_direction /= np.linalg.norm(view_direction)
        projection = partial_candidates @ view_direction
        condition_clouds["partial_view_50.ply"] = partial_candidates[
            np.argsort(projection)[-args.observation_points // 2:]
        ]

        for file_name, points in condition_clouds.items():
            write_ply(shape_root / file_name, points)

        metadata = {
            "schema_version": 2,
            "shape": shape_name,
            "trait": payload,
            "surface_sampling": manifest["surface_sampling"],
            "grid_resolution": args.grid_resolution,
            "seeds": {
                "reference": seeds[0],
                "clean": seeds[1],
                "noise_surface": seeds[2],
                "noise_values": seeds[3],
                "outlier_20": seeds[4],
                "outlier_50": seeds[5],
                "missing_80": seeds[6],
                "partial_view_50": seeds[7],
            },
            "conditions": {
                "reference_uniform.ply": {"points": args.reference_points, "role": "evaluation-only reference"},
                "clean.ply": {"points": args.observation_points, "independent_from_reference": True},
                "noise_1pct_diag.ply": {"points": args.observation_points, "sigma": noise_sigma, "sigma_fraction_bbox_diagonal": args.noise_diagonal_fraction},
                "outlier_20.ply": {"points": args.observation_points, "outlier_fraction_of_total": 0.20, "gross_outlier_minimum_distance_fraction_bbox_diagonal": 0.05},
                "outlier_50.ply": {"points": args.observation_points, "outlier_fraction_of_total": 0.50, "gross_outlier_minimum_distance_fraction_bbox_diagonal": 0.05},
                "missing_80.ply": {"points": missing_count, "random_missing_fraction": 0.80},
                "partial_view_50.ply": {"points": args.observation_points // 2, "half_space_retained_fraction": 0.50, "view_direction": view_direction.tolist()},
            },
            "fixed_estimator_protocol": {
                "source": "median nearest-neighbor distance of clean.ply",
                "data_resolution": clean_data_resolution,
                "model_resolution": clean_model_resolution,
                "apply_unchanged_to_all_conditions": True,
            },
        }
        with open(shape_root / "trait.json", "w", encoding="utf-8") as stream:
            json.dump({"trait": payload}, stream, indent=2)
        with open(shape_root / "metadata.json", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
        manifest["shapes"].append({"name": shape_name, "directory": str(shape_root)})
        print(f"{shape_name}: {len(condition_clouds)} files")

    with open(output_root / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Saved benchmark to: {output_root}")


if __name__ == "__main__":
    main()
