"""Strict audit for the frozen ModelNet40 ten-object robustness dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from sklearn.neighbors import KDTree


DEFAULT_ROOT = Path(r"C:\code\datasets\modelnet40\real10_robustness")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1].parent
    / "outputs/benchmark_audits/modelnet40_real10_data_audit.json"
)


def read_ply(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    rows = []
    if len(manifest.get("cases", [])) != 10:
        errors.append("manifest does not contain exactly 10 cases")
    if len(set(row["category"] for row in manifest.get("cases", []))) < 5:
        errors.append("fewer than five categories")

    for case in manifest.get("cases", []):
        case_root = Path(case["directory"])
        metadata = json.loads((case_root / "metadata.json").read_text(encoding="utf-8"))
        clouds = {
            key: read_ply(case_root / item["file"])
            for key, item in metadata["conditions"].items()
        }
        reference = read_ply(case_root / "reference.ply")
        clean = clouds["clean"]
        prefix = metadata["model"]
        expected_counts = {"clean": 5000, "noise": 5000, "outlier_20": 5000, "partial_view": 3000}
        for condition, count in expected_counts.items():
            if clouds[condition].shape != (count, 3):
                errors.append(f"{prefix}/{condition}: wrong shape")
            if not np.isfinite(clouds[condition]).all():
                errors.append(f"{prefix}/{condition}: non-finite points")
        if reference.shape != (20000, 3) or not np.isfinite(reference).all():
            errors.append(f"{prefix}/reference: invalid")

        diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
        noise_rms_axis = float(np.sqrt(np.mean((clouds["noise"] - clean) ** 2)))
        target_sigma = 0.005 * diagonal
        if not np.isclose(noise_rms_axis, target_sigma, rtol=0.05):
            errors.append(f"{prefix}/noise: RMS does not match fixed sigma")

        partial_cfg = metadata["conditions"]["partial_view"]
        direction = np.asarray(partial_cfg["view_direction"], dtype=float)
        threshold = np.partition(clean @ direction, -3000)[-3000]
        if float(np.min(clouds["partial_view"] @ direction)) < threshold - 1e-6:
            errors.append(f"{prefix}/partial_view: not the frozen projection subset")

        outlier_distances = KDTree(reference).query(clouds["outlier_20"], k=1)[0].ravel()
        gross_count = int(np.count_nonzero(outlier_distances >= 0.05 * diagonal - 1e-7))
        if gross_count < 1000:
            errors.append(f"{prefix}/outlier_20: fewer than 1000 certified gross outliers")

        nearest = KDTree(clean).query(clean, k=2)[0][:, 1]
        resolution = float(np.median(nearest[nearest > 0.0]))
        recorded = float(metadata["fixed_estimator_protocol"]["data_resolution"])
        if not np.isclose(resolution, recorded, rtol=1e-6, atol=1e-9):
            errors.append(f"{prefix}: clean resolution mismatch")
        for file_name, digest in metadata["sha256"].items():
            if sha256(case_root / file_name) != digest:
                errors.append(f"{prefix}/{file_name}: hash mismatch")

        rows.append(
            {
                "case": prefix,
                "category": metadata["category"],
                "reference_points": len(reference),
                "noise_sigma_fraction_diagonal_observed": noise_rms_axis / diagonal,
                "certified_gross_outliers": gross_count,
                "partial_points": len(clouds["partial_view"]),
                "data_resolution": recorded,
            }
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "cases": len(rows),
        "categories": sorted(set(row["category"] for row in rows)),
        "checks": rows,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(rows), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
