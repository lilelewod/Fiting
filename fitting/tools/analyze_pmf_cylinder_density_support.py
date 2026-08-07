"""Measure how well label-free k-NN density support retains cylinder inliers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import read_point_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/pmf_cylinder"))
    parser.add_argument("--neighbors", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.50, 0.75])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    clean = read_point_cloud(str(data_root / "clean.ply"))
    clean_tree = KDTree(clean)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    truth = metadata["ground_truth"]
    records = []
    for condition in ("outlier_50", "outlier_80"):
        points = read_point_cloud(str(data_root / f"{condition}.ply"))
        inlier = clean_tree.query(points, k=1)[0].ravel() <= 1e-7
        for neighbors in args.neighbors:
            k = min(max(2, int(neighbors)), len(points) - 1)
            distance = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points).kneighbors(
                points, return_distance=True
            )[0][:, -1]
            order = np.argsort(distance, kind="stable")
            for fraction in args.fractions:
                keep = max(4, int(np.floor(len(points) * fraction)))
                selected = order[:keep]
                tp = int(inlier[selected].sum())
                selected_inliers = points[selected][inlier[selected]]
                angle = np.arctan2(
                    selected_inliers[:, 1] - truth["y0"],
                    selected_inliers[:, 0] - truth["x0"],
                )
                angle_u = ((angle - truth["start_angle"]) % (2.0 * np.pi)) / truth["angular_span"]
                height_u = (selected_inliers[:, 2] - truth["z0"]) / truth["height"]
                angle_bins = np.clip((angle_u * 12).astype(int), 0, 11)
                height_bins = np.clip((height_u * 6).astype(int), 0, 5)
                records.append(
                    {
                        "method": "fixed_fraction",
                        "condition": condition,
                        "neighbors": k,
                        "support_fraction": float(fraction),
                        "selected_points": keep,
                        "inlier_precision": float(tp / keep),
                        "inlier_recall": float(tp / int(inlier.sum())),
                        "selected_inliers": tp,
                        "available_inliers": int(inlier.sum()),
                        "occupied_angle_bins_12": int(np.unique(angle_bins).size),
                        "occupied_height_bins_6": int(np.unique(height_bins).size),
                    }
                )
            labels = KMeans(n_clusters=2, random_state=0, n_init=20).fit_predict(
                np.log(np.maximum(distance, np.finfo(float).eps)).reshape(-1, 1)
            )
            centers = [float(np.median(distance[labels == label])) for label in (0, 1)]
            dense_label = int(np.argmin(centers))
            selected = np.flatnonzero(labels == dense_label)
            tp = int(inlier[selected].sum())
            selected_inliers = points[selected][inlier[selected]]
            angle = np.arctan2(
                selected_inliers[:, 1] - truth["y0"],
                selected_inliers[:, 0] - truth["x0"],
            )
            angle_u = ((angle - truth["start_angle"]) % (2.0 * np.pi)) / truth["angular_span"]
            height_u = (selected_inliers[:, 2] - truth["z0"]) / truth["height"]
            records.append(
                {
                    "method": "adaptive_log_kdistance_kmeans",
                    "condition": condition,
                    "neighbors": k,
                    "support_fraction": float(len(selected) / len(points)),
                    "selected_points": int(len(selected)),
                    "inlier_precision": float(tp / len(selected)),
                    "inlier_recall": float(tp / int(inlier.sum())),
                    "selected_inliers": tp,
                    "available_inliers": int(inlier.sum()),
                    "occupied_angle_bins_12": int(np.unique(np.clip((angle_u * 12).astype(int), 0, 11)).size),
                    "occupied_height_bins_6": int(np.unique(np.clip((height_u * 6).astype(int), 0, 5)).size),
                    "dense_median_kdistance": centers[dense_label],
                    "sparse_median_kdistance": centers[1 - dense_label],
                }
            )
    result = {"selection_is_label_free": True, "labels_used_only_for_post_hoc_analysis": True, "records": records}
    text = json.dumps(result, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
