"""Audit whether occluded-superquadric ground truth lies in the search bounds.

The production SuperquadricRule derives center and scale bounds from the input
cloud.  Under spatially coherent occlusion, the visible cap can exclude the
true object center even though the scale remains admissible.  This audit
reconstructs those exact bounds without fitting or changing the formal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_vertices(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64, copy=False
    )


def audit_case(case_dir: Path, condition: str) -> dict:
    points = load_vertices(case_dir / f"{condition}.ply")
    trait = json.loads((case_dir / "trait.json").read_text(encoding="utf-8"))["trait"]
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))

    center = np.asarray(trait["center"], dtype=np.float64)
    scale = np.asarray(trait["scale"], dtype=np.float64)
    observed_min = points.min(axis=0)
    observed_max = points.max(axis=0)
    observed_extent = observed_max - observed_min
    observed_diagonal = float(np.linalg.norm(observed_extent))

    # Keep these formulas exactly aligned with SuperquadricRule._init_bounds.
    center_lower = observed_min - 0.2 * observed_extent
    center_upper = observed_max + 0.2 * observed_extent
    scale_lower = 0.02 * observed_diagonal
    scale_upper = 1.20 * observed_diagonal

    center_violation = np.maximum(center_lower - center, 0.0) + np.maximum(
        center - center_upper, 0.0
    )
    center_feasible = bool(np.all(center_violation == 0.0))
    scale_feasible = bool(np.all((scale >= scale_lower) & (scale <= scale_upper)))
    reference_diagonal = float(metadata["reference_bbox_diagonal"])

    return {
        "case": case_dir.name,
        "condition": condition,
        "observed_points": int(points.shape[0]),
        "center_feasible": center_feasible,
        "center_violation_xyz": center_violation.tolist(),
        "center_violation_norm": float(np.linalg.norm(center_violation)),
        "center_violation_norm_over_reference_diagonal": float(
            np.linalg.norm(center_violation) / reference_diagonal
        ),
        "scale_feasible": scale_feasible,
        "observed_diagonal": observed_diagonal,
        "center_lower": center_lower.tolist(),
        "center_upper": center_upper.tolist(),
        "scale_lower": float(scale_lower),
        "scale_upper": float(scale_upper),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\code\superquadic_data\v3_randomized"),
    )
    parser.add_argument("--condition", default="occlusion_cap_80")
    parser.add_argument("--cases", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        audit_case(args.dataset_root / f"case_{index:03d}", args.condition)
        for index in range(args.cases)
    ]
    center_feasible = sum(row["center_feasible"] for row in rows)
    scale_feasible = sum(row["scale_feasible"] for row in rows)
    report = {
        "status": "PASS",
        "purpose": (
            "Reconstruct the production input-derived SuperquadricRule search bounds; "
            "PASS means the audit executed and does not imply ground-truth feasibility."
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "condition": args.condition,
        "cases": len(rows),
        "center_feasible_cases": int(center_feasible),
        "center_infeasible_cases": int(len(rows) - center_feasible),
        "scale_feasible_cases": int(scale_feasible),
        "scale_infeasible_cases": int(len(rows) - scale_feasible),
        "all_ground_truth_parameters_feasible_cases": int(
            sum(row["center_feasible"] and row["scale_feasible"] for row in rows)
        ),
        "rows": rows,
        "errors": [],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
