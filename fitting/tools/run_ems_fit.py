"""Run the external EMS implementation and save a provenance-rich fit record.

This adapter intentionally does not reimplement or alter the EMS objective.  It
only loads a PLY point cloud, calls ``EMS_recovery``, and serializes the fitted
superquadric in the trait convention used by this project.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

from EMS.EMS_recovery import EMS_recovery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.external_parameter_conventions import ems_shape_to_project


def load_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"invalid XYZ point cloud: {path}")
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outlier-ratio", type=float, default=0.1)
    parser.add_argument("--max-iteration-em", type=int, default=20)
    parser.add_argument("--max-optimization-iterations", type=int, default=3)
    parser.add_argument("--max-switches", type=int, default=2)
    parser.add_argument("--adaptive-upper-bound", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_xyz(args.data_file)
    start = time.perf_counter()
    fitted, inlier_posterior = EMS_recovery(
        points,
        OutlierRatio=args.outlier_ratio,
        MaxIterationEM=args.max_iteration_em,
        MaxOptiIterations=args.max_optimization_iterations,
        MaxiSwitch=args.max_switches,
        AdaptiveUpperBound=args.adaptive_upper_bound,
    )
    elapsed = time.perf_counter() - start

    result = {
        "algorithm": "EMS",
        "implementation": "bmlklwx/EMS-superquadric_fitting",
        "implementation_commit": "5a213d46e8b013b5a153f5aa914b0925ee517af7",
        "data_file": str(args.data_file.resolve()),
        "point_count": int(len(points)),
        "wall_time_s": float(elapsed),
        "python": platform.python_version(),
        "settings": {
            "outlier_ratio": args.outlier_ratio,
            "max_iteration_em": args.max_iteration_em,
            "max_optimization_iterations": args.max_optimization_iterations,
            "max_switches": args.max_switches,
            "adaptive_upper_bound": args.adaptive_upper_bound,
            "rescale": True,
        },
        "posterior": {
            "mean_inlier_probability": float(np.mean(inlier_posterior)),
            "median_inlier_probability": float(np.median(inlier_posterior)),
        },
        "ems_raw": {
            "shape_epsilon1_meridional_epsilon2_azimuthal": fitted.shape.tolist(),
            "scale": fitted.scale.tolist(),
            "euler_zyx": fitted.euler.tolist(),
            "rotation_matrix": fitted.RotM.tolist(),
            "translation": fitted.translation.tolist(),
        },
        "trait": {
            "center": fitted.translation.tolist(),
            "scale": fitted.scale.tolist(),
            # This project stores [azimuthal, meridional], whereas EMS stores
            # the standard [epsilon1 meridional, epsilon2 azimuthal] order.
            "shape": ems_shape_to_project(fitted.shape).tolist(),
            "shape_conversion": "project=[EMS epsilon2 azimuthal, EMS epsilon1 meridional]",
            "rotation": fitted.euler.tolist(),
            "rotation_convention": "ZYX Euler; rotation_matrix is authoritative",
            "rotation_matrix": fitted.RotM.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
