"""Audit files, hashes, resolutions, and the coherent occlusion construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import read_point_cloud
from tools.prepare_randomized_superquadric_benchmark import randomized_trait
from tools.prepare_superquadric_benchmark import gross_outliers
from tools.superquadric_evaluation import sample_trait, trait_from_mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_error(regenerated, stored):
    """Maximum coordinate error after reproducing the PLY float32 write path."""
    regenerated = np.asarray(regenerated, dtype=np.float32).astype(np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    if regenerated.shape != stored.shape:
        return float("inf")
    return float(np.max(np.abs(regenerated - stored), initial=0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\code\superquadic_data\v3_randomized"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    cases = []
    expected_files = {
        "reference_uniform.ply": int(manifest["reference_points"]),
        "clean.ply": int(manifest["observation_points"]),
        "noise_1pct_diag.ply": int(manifest["observation_points"]),
        "outlier_20.ply": int(manifest["observation_points"]),
        "missing_80.ply": int(round(0.2 * manifest["observation_points"])),
        "occlusion_cap_80.ply": int(round(0.2 * manifest["observation_points"])),
    }
    for index, record in enumerate(manifest["case_records"]):
        case_root = root / record["case"]
        metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
        regenerated_trait, regenerated_strata, regenerated_seeds = randomized_trait(
            index, int(manifest["base_seed"])
        )
        if regenerated_trait != metadata["trait"]:
            errors.append(f"{record['case']}: trait cannot be regenerated from base seed")
        if regenerated_strata != metadata["strata"] or regenerated_strata != record["strata"]:
            errors.append(f"{record['case']}: stratum labels cannot be regenerated")
        recorded_core_seeds = {
            key: int(metadata["seeds"][key]) for key in regenerated_seeds
        }
        if regenerated_seeds != recorded_core_seeds or len(set(regenerated_seeds.values())) != len(
            regenerated_seeds
        ):
            errors.append(f"{record['case']}: core condition seeds are wrong or duplicated")
        counts = {}
        stored_clouds = {}
        for filename, expected_count in expected_files.items():
            path = case_root / filename
            if not path.exists():
                errors.append(f"{record['case']}: missing {filename}")
                continue
            points = read_point_cloud(str(path))
            stored_clouds[filename] = points
            counts[filename] = int(len(points))
            if len(points) != expected_count:
                errors.append(f"{record['case']}/{filename}: {len(points)} != {expected_count}")
            if metadata.get("sha256", {}).get(filename) != sha256(path):
                errors.append(f"{record['case']}/{filename}: SHA-256 mismatch")

        trait = trait_from_mapping(metadata["trait"])
        grid = int(manifest["grid_resolution"])
        observation_count = int(manifest["observation_points"])
        reference = sample_trait(
            trait, int(manifest["reference_points"]), regenerated_seeds["reference"], grid
        )
        clean = sample_trait(trait, observation_count, regenerated_seeds["clean"], grid)
        noise_base = sample_trait(
            trait, observation_count, regenerated_seeds["noise_surface"], grid
        )
        bbox_diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
        noise_meta = metadata["conditions"]["noise_1pct_diag.ply"]
        noise_sigma = float(noise_meta["sigma_fraction_bbox_diagonal"]) * bbox_diagonal
        noise_rng = np.random.default_rng(regenerated_seeds["noise_values"])
        noisy = noise_base + noise_rng.normal(0.0, noise_sigma, noise_base.shape)

        outlier_count = int(round(0.20 * observation_count))
        outlier_rng = np.random.default_rng(regenerated_seeds["outlier_20"])
        inliers = sample_trait(
            trait, observation_count - outlier_count, regenerated_seeds["outlier_20"] + 1, grid
        )
        outliers = gross_outliers(reference, outlier_count, outlier_rng)
        contaminated = np.vstack((inliers, outliers))
        contaminated = contaminated[outlier_rng.permutation(len(contaminated))]
        missing = sample_trait(
            trait, int(round(0.20 * observation_count)), regenerated_seeds["missing_80"], grid
        )
        regenerated_clouds = {
            "reference_uniform.ply": reference,
            "clean.ply": clean,
            "noise_1pct_diag.ply": noisy,
            "outlier_20.ply": contaminated,
            "missing_80.ply": missing,
        }
        regeneration_errors = {
            name: storage_error(points, stored_clouds[name])
            for name, points in regenerated_clouds.items()
            if name in stored_clouds
        }
        for name, difference in regeneration_errors.items():
            if difference != 0.0:
                errors.append(f"{record['case']}/{name}: deterministic regeneration error {difference}")
        if not np.isclose(
            float(metadata["reference_bbox_diagonal"]), bbox_diagonal, rtol=0.0, atol=1e-12
        ) or not np.isclose(float(noise_meta["sigma"]), noise_sigma, rtol=0.0, atol=1e-12):
            errors.append(f"{record['case']}: reference diagonal or noise sigma mismatch")
        minimum_outlier_distance = float(KDTree(reference).query(outliers, k=1)[0].min())
        required_outlier_distance = float(
            metadata["conditions"]["outlier_20.ply"][
                "gross_outlier_minimum_distance_fraction_bbox_diagonal"
            ]
        ) * bbox_diagonal
        if minimum_outlier_distance + 1e-12 < required_outlier_distance:
            errors.append(f"{record['case']}: gross-outlier exclusion distance is violated")

        clean = stored_clouds["clean.ply"]
        nearest = KDTree(clean).query(clean, k=2)[0][:, 1]
        recomputed_resolution = float(np.median(nearest[nearest > 0.0]))
        recorded_resolution = float(metadata["fixed_estimator_protocol"]["data_resolution"])
        if not np.isclose(recomputed_resolution, recorded_resolution, rtol=0.0, atol=1e-12):
            errors.append(f"{record['case']}: clean resolution mismatch")

        occlusion_meta = metadata["conditions"]["occlusion_cap_80.ply"]
        surface_seed = int(metadata["seeds"]["occlusion_cap_80_surface"])
        candidates = sample_trait(
            trait_from_mapping(metadata["trait"]),
            count=int(occlusion_meta["candidate_surface_points"]),
            seed=surface_seed,
            grid_resolution=int(manifest["grid_resolution"]),
        )
        direction = np.asarray(occlusion_meta["direction"], dtype=float)
        projection = (candidates - np.asarray(metadata["trait"]["center"])) @ direction
        retain = expected_files["occlusion_cap_80.ply"]
        selected = np.argpartition(projection, -retain)[-retain:]
        regenerated = candidates[selected]
        stored = read_point_cloud(str(case_root / "occlusion_cap_80.ply"))
        reconstruction_error = float(KDTree(regenerated).query(stored, k=1)[0].max())
        if reconstruction_error > 1e-7:
            errors.append(f"{record['case']}: coherent occlusion cannot be regenerated")
        omitted = np.setdiff1d(np.arange(len(candidates)), selected, assume_unique=False)
        projection_margin = float(projection[selected].min() - projection[omitted].max())
        if projection_margin < -1e-12:
            errors.append(f"{record['case']}: occlusion selection is not a coherent projection cap")
        cases.append(
            {
                "case": record["case"],
                "counts": counts,
                "deterministic_regeneration_max_abs_error": regeneration_errors,
                "gross_outlier_minimum_distance": minimum_outlier_distance,
                "gross_outlier_required_minimum_distance": required_outlier_distance,
                "clean_resolution": recorded_resolution,
                "occlusion_regeneration_max_distance": reconstruction_error,
                "occlusion_projection_margin": projection_margin,
            }
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "case_count": len(cases),
        "sampling_audit": {
            "trait_strata_and_seed_regeneration": True,
            "reference_clean_noise_outlier_and_random_missing_regeneration": True,
            "ply_storage_precision": "float32",
        },
        "errors": errors,
        "cases": cases,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
