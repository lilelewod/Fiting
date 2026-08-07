"""PMF-style M1 partial-similarity experiment.

The experiment reconstructs the model/data definitions and point counts from
Section 6.1 of Zhang et al., Pattern Recognition 85 (2019).  It evaluates M1
against D1, D2, and D4; these are reconstructed data, not the authors' files.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree


POINTS_PER_QUADRANT = 3072
EPSILON = 1e-8
LAMBDA = 2.0
H = 5.0


def quadrant_points(theta, quadrant, seed, count=POINTS_PER_QUADRANT):
    """Uniformly sample one quadrant of the square ring."""
    if theta >= 2.0 or count <= 0:
        return np.empty((0, 2)), np.empty(0)
    rng = np.random.default_rng(seed)
    # In absolute coordinates the quadrant is [0,2]^2 minus [0,theta]^2.
    # Split it into two disjoint rectangles and allocate samples by area.
    vertical_fraction = 2.0 / (2.0 + theta)
    n_vertical = int(round(count * vertical_fraction))
    n_vertical = min(max(n_vertical, 1), count - 1)
    n_horizontal = count - n_vertical

    uv = rng.random((count, 2))
    vertical = np.column_stack((
        theta + (2.0 - theta) * uv[:n_vertical, 0],
        2.0 * uv[:n_vertical, 1],
    ))
    horizontal = np.column_stack((
        theta * uv[n_vertical:, 0],
        theta + (2.0 - theta) * uv[n_vertical:, 1],
    ))
    points = np.vstack((vertical, horizontal))
    sx = -1.0 if quadrant in (0, 2) else 1.0
    sy = -1.0 if quadrant in (0, 1) else 1.0
    points[:, 0] *= sx
    points[:, 1] *= sy
    area = 4.0 - theta * theta
    return points, np.full(count, area / count)


def model_points(theta, seed_base):
    points, areas, labels = [], [], []
    for quadrant in range(4):
        p, a = quadrant_points(theta, quadrant, seed_base + quadrant)
        points.append(p)
        areas.append(a)
        labels.append(np.full(p.shape[0], quadrant, dtype=int))
    return np.vstack(points), np.concatenate(areas), np.concatenate(labels)


def make_data():
    quadrants = [quadrant_points(1.0, q, 2019 + q)[0] for q in range(4)]
    return {
        "D1": np.vstack(quadrants),
        "D2": np.vstack(quadrants[:3]),
        "D4": quadrants[:1][0],
    }


def voxel_set(points, resolution):
    indices = np.floor((points + 2.0) / resolution).astype(np.int32)
    return set(map(tuple, indices))


def evaluate(theta, model, cell_areas, labels, data, data_tree, data_voxels,
             inlier_threshold=0.04, voxel_resolution=0.04):
    if model.shape[0] == 0:
        return {
            "wmm": 0.0, "mm": 0.0, "negative_edm": -np.inf,
            "negative_hausdorff": -np.inf, "inlier_number": 0.0,
            "negative_voxel_difference": -np.inf,
        }
    model_to_data = data_tree.query(model, k=1)[0].ravel()
    model_tree = KDTree(model)
    data_to_model = model_tree.query(data, k=1)[0].ravel()

    weights = np.exp(-H * model_to_data)
    weighted_measure = float(np.sum(weights * cell_areas))
    weighted_error = float(np.sum(weights * model_to_data) / np.sum(weights))
    full_measure = float(np.sum(cell_areas))
    full_error = float(np.mean(model_to_data))

    model_voxels = voxel_set(model, voxel_resolution)
    voxel_difference = len(model_voxels.symmetric_difference(data_voxels))
    return {
        "wmm": weighted_measure / (EPSILON + weighted_error ** LAMBDA),
        "mm": full_measure / (EPSILON + full_error ** LAMBDA),
        "negative_edm": -float(np.mean(data_to_model)),
        "negative_hausdorff": -float(max(model_to_data.max(), data_to_model.max())),
        "inlier_number": float(np.sum(data_to_model <= inlier_threshold)),
        "negative_voxel_difference": -float(voxel_difference),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-steps", type=int, default=401)
    parser.add_argument("--output-dir", default="../outputs/pmf_m1_partial_similarity")
    args = parser.parse_args()
    if args.theta_steps < 3:
        raise ValueError("theta-steps must be at least 3")

    data_sets = make_data()
    expected = {"D1": 12288, "D2": 9216, "D4": 3072}
    actual = {name: int(points.shape[0]) for name, points in data_sets.items()}
    if actual != expected:
        raise AssertionError(f"Unexpected data counts: {actual}")

    thetas = np.linspace(0.0, 2.0, args.theta_steps)
    curves = []
    for data_name, data in data_sets.items():
        data_tree = KDTree(data)
        data_voxels = voxel_set(data, 0.04)
        for theta in thetas:
            model, areas, labels = model_points(float(theta), seed_base=42)
            scores = evaluate(float(theta), model, areas, labels, data, data_tree, data_voxels)
            curves.append({"data": data_name, "theta": float(theta), **scores})

    metrics = [
        "wmm", "mm", "negative_edm", "negative_hausdorff",
        "inlier_number", "negative_voxel_difference",
    ]
    summary = []
    for data_name in data_sets:
        selected = [row for row in curves if row["data"] == data_name]
        for metric in metrics:
            best = max(selected, key=lambda row: row[metric])
            theta_hat = float(best["theta"])
            summary.append({
                "data": data_name,
                "retained_fraction": {"D1": 1.0, "D2": 0.75, "D4": 0.25}[data_name],
                "metric": metric,
                "theta_hat": theta_hat,
                "absolute_theta_error": abs(theta_hat - 1.0),
                "best_value": float(best[metric]),
                "success_at_0.02": int(abs(theta_hat - 1.0) <= 0.0200001),
            })

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, rows in (("curves.csv", curves), ("summary.csv", summary)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "provenance": "PMF-style reconstruction; not the authors' original data files.",
        "settings": {
            "epsilon": EPSILON, "lambda": LAMBDA, "h": H,
            "theta_steps": args.theta_steps, "theta_step": float(thetas[1] - thetas[0]),
            "data_counts": actual,
        },
        "summary": summary,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
