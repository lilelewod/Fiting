"""Generate a stratified randomized convex-superquadric benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_superquadric_benchmark import gross_outliers, write_ply
from tools.superquadric_evaluation import sample_trait, trait_from_mapping


SHAPE_STRATA = ("smooth", "mixed", "boxy")
ASPECT_STRATA = ("balanced", "anisotropic", "extreme")


def _log_uniform(rng: np.random.Generator, lower: float, upper: float, size: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lower), np.log(upper), size=size))


def randomized_trait(case_index: int, base_seed: int) -> tuple[dict, dict, dict]:
    """Return a deterministic trait, stratum labels and per-condition seeds."""
    sequence = np.random.SeedSequence([int(base_seed), int(case_index)])
    children = sequence.spawn(7)
    rng = np.random.default_rng(children[0])
    shape_stratum = SHAPE_STRATA[case_index % len(SHAPE_STRATA)]
    aspect_stratum = ASPECT_STRATA[(case_index // len(SHAPE_STRATA)) % len(ASPECT_STRATA)]

    # EMS uses [epsilon1 meridional, epsilon2 azimuthal].  The existing
    # project trait convention stores these in the opposite array order.
    if shape_stratum == "smooth":
        ems_shape = rng.uniform(0.75, 1.0, size=2)
    elif shape_stratum == "mixed":
        ems_shape = np.array([rng.uniform(0.15, 0.45), rng.uniform(0.75, 1.0)])
        if rng.random() < 0.5:
            ems_shape = ems_shape[::-1]
    else:
        ems_shape = rng.uniform(0.15, 0.45, size=2)

    minimum_ratio = {"balanced": 0.70, "anisotropic": 0.35, "extreme": 0.15}[aspect_stratum]
    scale = _log_uniform(rng, minimum_ratio, 1.0, size=3)
    scale *= 1.2 / np.max(scale)
    scale = scale[rng.permutation(3)]
    center = rng.uniform(-0.25, 0.25, size=3)
    rotation = Rotation.random(random_state=rng)
    euler_xyz = rotation.as_euler("xyz")

    payload = {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "shape": ems_shape[::-1].tolist(),
        "rotation": euler_xyz.tolist(),
        "rotation_matrix": rotation.as_matrix().tolist(),
    }
    strata = {
        "shape": shape_stratum,
        "aspect": aspect_stratum,
        "ems_shape_epsilon1_meridional_epsilon2_azimuthal": ems_shape.tolist(),
        "project_shape_azimuthal_meridional": payload["shape"],
    }
    names = ("reference", "clean", "noise_surface", "noise_values", "outlier_20", "missing_80")
    seeds = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children[1:])
    }
    return payload, strata, seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/superquadric_v3_randomized"))
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--observation-points", type=int, default=5000)
    parser.add_argument("--reference-points", type=int, default=20_000)
    parser.add_argument("--grid-resolution", type=int, default=256)
    parser.add_argument("--base-seed", type=int, default=20_260_721)
    parser.add_argument("--noise-diagonal-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases < 9:
        raise ValueError("at least 9 cases are required to cover all strata")
    if args.observation_points < 100 or args.reference_points < args.observation_points:
        raise ValueError("invalid observation/reference point counts")
    if args.grid_resolution < 32 or args.noise_diagonal_fraction <= 0.0:
        raise ValueError("invalid grid resolution or noise fraction")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "benchmark": "stratified randomized convex superquadrics",
        "generator": str(Path(__file__).resolve()),
        "base_seed": args.base_seed,
        "cases": args.cases,
        "observation_points": args.observation_points,
        "reference_points": args.reference_points,
        "grid_resolution": args.grid_resolution,
        "surface_sampling": "triangle-area-proportional with uniform barycentric coordinates",
        "shape_strata": list(SHAPE_STRATA),
        "aspect_strata": list(ASPECT_STRATA),
        "case_records": [],
    }

    for index in range(args.cases):
        case_name = f"case_{index:03d}"
        case_root = output_root / case_name
        case_root.mkdir(parents=True, exist_ok=True)
        payload, strata, seeds = randomized_trait(index, args.base_seed)
        trait = trait_from_mapping(payload)
        reference = sample_trait(trait, args.reference_points, seeds["reference"], args.grid_resolution)
        clean = sample_trait(trait, args.observation_points, seeds["clean"], args.grid_resolution)
        nearest = KDTree(clean).query(clean, k=2)[0][:, 1]
        data_resolution = float(np.median(nearest[nearest > 0.0]))

        noise_base = sample_trait(
            trait, args.observation_points, seeds["noise_surface"], args.grid_resolution
        )
        bbox_diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
        noise_sigma = args.noise_diagonal_fraction * bbox_diagonal
        noise_rng = np.random.default_rng(seeds["noise_values"])
        noisy = noise_base + noise_rng.normal(0.0, noise_sigma, noise_base.shape)

        outlier_rng = np.random.default_rng(seeds["outlier_20"])
        outlier_count = int(round(0.20 * args.observation_points))
        inliers = sample_trait(
            trait,
            args.observation_points - outlier_count,
            seeds["outlier_20"] + 1,
            args.grid_resolution,
        )
        outliers = gross_outliers(reference, outlier_count, outlier_rng)
        contaminated = np.vstack((inliers, outliers))
        contaminated = contaminated[outlier_rng.permutation(len(contaminated))]
        missing = sample_trait(
            trait,
            int(round(0.20 * args.observation_points)),
            seeds["missing_80"],
            args.grid_resolution,
        )

        clouds = {
            "reference_uniform.ply": reference,
            "clean.ply": clean,
            "noise_1pct_diag.ply": noisy,
            "outlier_20.ply": contaminated,
            "missing_80.ply": missing,
        }
        hashes = {}
        for name, points in clouds.items():
            path = case_root / name
            write_ply(path, points)
            hashes[name] = sha256(path)

        (case_root / "trait.json").write_text(
            json.dumps({"trait": payload}, indent=2), encoding="utf-8"
        )
        metadata = {
            "schema_version": 3,
            "case": case_name,
            "trait": payload,
            "strata": strata,
            "seeds": seeds,
            "reference_bbox_diagonal": bbox_diagonal,
            "fixed_estimator_protocol": {
                "source": "median nearest-neighbor distance of clean.ply",
                "data_resolution": data_resolution,
                "model_resolution": 0.45 * data_resolution,
                "apply_unchanged_to_all_conditions": True,
            },
            "conditions": {
                "clean.ply": {"points": args.observation_points, "independent_surface_sample": True},
                "noise_1pct_diag.ply": {
                    "points": args.observation_points,
                    "sigma": noise_sigma,
                    "sigma_fraction_bbox_diagonal": args.noise_diagonal_fraction,
                    "independent_surface_sample": True,
                },
                "outlier_20.ply": {
                    "points": args.observation_points,
                    "outlier_fraction_of_total": 0.20,
                    "gross_outlier_minimum_distance_fraction_bbox_diagonal": 0.05,
                    "independent_surface_sample": True,
                },
                "missing_80.ply": {
                    "points": len(missing),
                    "random_missing_fraction": 0.80,
                    "independent_surface_sample": True,
                },
            },
            "sha256": hashes,
        }
        (case_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest["case_records"].append(
            {"case": case_name, "strata": strata, "directory": str(case_root)}
        )
        print(f"{case_name}: shape={strata['shape']}, aspect={strata['aspect']}")

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved randomized benchmark to: {output_root}")


if __name__ == "__main__":
    main()
