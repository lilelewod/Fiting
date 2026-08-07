"""Audit CUDA-selected nearest neighbors against the CPU PMF objective path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.pmf_cylinder_rule import PMFCylinderTrait, sample_partial_cylinder
from tools.data_tool import read_point_cloud

def candidate_traits(truth, count, seed):
    rng = np.random.default_rng(seed)
    traits = []
    for _ in range(count):
        trait = PMFCylinderTrait()
        trait.x0 = float(truth["x0"] + rng.normal(0.0, 0.35))
        trait.y0 = float(truth["y0"] + rng.normal(0.0, 0.35))
        trait.z0 = float(truth["z0"] + rng.normal(0.0, 0.35))
        trait.radius = float(truth["radius"] * rng.uniform(0.75, 1.25))
        trait.height = float(truth["height"] * rng.uniform(0.75, 1.25))
        trait.start_angle = float(truth["start_angle"] + rng.normal(0.0, 0.3))
        trait.angular_span = float(
            np.clip(truth["angular_span"] * rng.uniform(0.75, 1.20), 0.3490658504, 2 * np.pi)
        )
        trait.end_angle = trait.start_angle + trait.angular_span
        traits.append(trait)
    return traits


def cuda_errors(data, model):
    estimator = MeanMeasureEstimator.__new__(MeanMeasureEstimator)
    estimator.data = np.asarray(data, dtype=np.float64)
    estimator.cfg = {"device": torch.device("cuda:0")}
    estimator._torch_device = None
    estimator._torch_data = None
    estimator._torch_cached_data_to_model_errors = None
    errors, _ = estimator._torch_cuda_bidirectional_nearest(model)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/pmf_cylinder"))
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--mean-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\environment\cuda_nn_equivalence_audit.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA equivalence audit requires a visible CUDA device")

    data_root = args.data_root.resolve()
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    traits = candidate_traits(metadata["ground_truth"], args.candidates, args.seed)
    errors = []
    condition_reports = {}
    for condition in ("clean", "outlier_50"):
        data = read_point_cloud(str(data_root / f"{condition}.ply"))
        tree = KDTree(data)
        mean_errors = []
        point_errors = []
        cpu_times = []
        cuda_times = []
        for trait in traits:
            model = sample_partial_cylinder(trait, sample_angle=64, sample_height=32)
            started = time.perf_counter()
            cpu = tree.query(model, k=1)[0].reshape(-1)
            cpu_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            gpu = cuda_errors(data, model)
            torch.cuda.synchronize()
            cuda_times.append(time.perf_counter() - started)
            mean_errors.append(abs(float(gpu.mean()) - float(cpu.mean())))
            point_errors.append(float(np.max(np.abs(gpu - cpu))))
        report = {
            "candidates": len(traits),
            "data_points": int(len(data)),
            "model_points": 64 * 32,
            "maximum_mean_distance_abs_error": float(max(mean_errors)),
            "maximum_point_distance_abs_error": float(max(point_errors)),
            "cpu_query_median_s": float(np.median(cpu_times)),
            "cuda_query_median_s": float(np.median(cuda_times)),
            "median_kernel_speedup": float(np.median(cpu_times) / np.median(cuda_times)),
        }
        if report["maximum_mean_distance_abs_error"] > args.mean_tolerance:
            errors.append(
                f"{condition}: mean nearest-neighbor error exceeds {args.mean_tolerance}"
            )
        condition_reports[condition] = report

    result = {
        "status": "PASS" if not errors else "FAIL",
        "purpose": (
            "Numerical-equivalence gate for the optional torch_cuda nearest-neighbor "
            "backend. Backend selection additionally requires a paired end-to-end timing "
            "gate; independent publication metrics remain CPU-exact."
        ),
        "cuda_device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "mean_distance_tolerance": args.mean_tolerance,
        "conditions": condition_reports,
        "errors": errors,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
