"""Evaluate an externally fitted superquadric with the common protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.superquadric_evaluation import geometric_metrics, load_trait, sample_trait, trait_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--ground-truth-trait", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=20_000)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--reference-seed", type=int, default=20_260_716)
    parser.add_argument("--model-seed", type=int, default=20_260_717)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_record = json.loads(args.fit_result.read_text(encoding="utf-8"))
    fitted_trait = trait_from_mapping(fit_record)
    ground_truth_trait = load_trait(args.ground_truth_trait)
    reference = sample_trait(
        ground_truth_trait,
        count=args.points,
        seed=args.reference_seed,
        grid_resolution=args.grid,
    )
    model = sample_trait(
        fitted_trait,
        count=args.points,
        seed=args.model_seed,
        grid_resolution=args.grid,
    )
    metrics = geometric_metrics(reference, model, threshold=args.threshold)
    bbox_diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
    result = {
        **metrics,
        "normalized_chamfer": metrics["gt_chamfer"] / bbox_diagonal,
        "reference_bbox_diagonal": bbox_diagonal,
        "metric_threshold": args.threshold,
        "evaluation_points": args.points,
        "evaluation_grid": args.grid,
        "reference_seed": args.reference_seed,
        "model_seed": args.model_seed,
        "reference_mode": "analytic-area-uniform",
        "fit_result": str(args.fit_result.resolve()),
        "ground_truth_trait": str(args.ground_truth_trait.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
