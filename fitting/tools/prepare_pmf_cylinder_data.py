"""Generate documented PMF-style cylinder robustness data.

These point clouds are reconstructed from the counts and corruption categories
reported for D5/D6 in Zhang et al. (Pattern Recognition, 2019).  They are not
the authors' original data files.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def sample_inliers(rng, count, center, radius, height, start_angle, angular_span):
    angles = rng.uniform(start_angle, start_angle + angular_span, count)
    z = rng.uniform(center[2], center[2] + height, count)
    return np.column_stack(
        (center[0] + radius * np.cos(angles),
         center[1] + radius * np.sin(angles), z)
    )


def sample_gross_outliers(rng, count, center, radius, height):
    """Uniform box contamination, rejecting points close to the full cylinder."""
    accepted = []
    while sum(x.shape[0] for x in accepted) < count:
        candidates = rng.uniform(
            [center[0] - 3.0 * radius, center[1] - 3.0 * radius, center[2] - 0.5 * height],
            [center[0] + 3.0 * radius, center[1] + 3.0 * radius, center[2] + 1.5 * height],
            size=(max(count, 1024), 3),
        )
        radial = np.linalg.norm(candidates[:, :2] - center[:2], axis=1)
        near_lateral = (
            (np.abs(radial - radius) < 0.35 * radius)
            & (candidates[:, 2] >= center[2])
            & (candidates[:, 2] <= center[2] + height)
        )
        accepted.append(candidates[~near_lateral])
    return np.vstack(accepted)[:count]


def write_ply(path, points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/pmf_cylinder")
    parser.add_argument("--seed", type=int, default=2019)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    center = np.asarray([0.0, 0.0, -3.0])
    truth = {
        "x0": 0.0, "y0": 0.0, "z0": -3.0,
        "radius": 2.0, "height": 6.0,
        "start_angle": -2.4, "angular_span": 4.8,
        "end_angle": 2.4,
    }
    inliers = sample_inliers(rng, 2048, center, truth["radius"], truth["height"],
                             truth["start_angle"], truth["angular_span"])
    low_outliers = sample_gross_outliers(rng, 2048, center, truth["radius"], truth["height"])
    high_outliers = sample_gross_outliers(rng, 8192, center, truth["radius"], truth["height"])

    datasets = {
        "clean": inliers,
        "outlier_50": np.vstack((inliers, low_outliers)),
        "outlier_80": np.vstack((inliers, high_outliers)),
    }
    for name, points in datasets.items():
        order = rng.permutation(points.shape[0])
        write_ply(output_dir / f"{name}.ply", points[order])

    metadata = {
        "provenance": (
            "PMF-style reconstruction based on the point counts and corruption categories "
            "reported for D5/D6; not the authors' original point clouds."
        ),
        "seed": args.seed,
        "ground_truth": truth,
        "datasets": {
            "clean.ply": {"inliers": 2048, "outliers": 0},
            "outlier_50.ply": {"inliers": 2048, "outliers": 2048},
            "outlier_80.ply": {"inliers": 2048, "outliers": 8192},
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
