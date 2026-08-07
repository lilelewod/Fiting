"""Add a deterministic spatially coherent 80% occlusion split to SQ v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_superquadric_benchmark import write_ply
from tools.superquadric_evaluation import sample_trait, trait_from_mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\code\superquadic_data\v3_randomized"))
    parser.add_argument("--candidate-points", type=int, default=5000)
    parser.add_argument("--retained-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.candidate_points < 100 or not 0.0 < args.retained_fraction < 1.0:
        raise ValueError("invalid candidate count or retained fraction")

    root = args.data_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_seed = int(manifest["base_seed"])
    retained_count = int(round(args.candidate_points * args.retained_fraction))
    for index, record in enumerate(manifest["case_records"]):
        case_root = root / record["case"]
        metadata_path = case_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trait = trait_from_mapping(metadata["trait"])
        sequence = np.random.SeedSequence([base_seed, index, 0x0CC1])
        surface_seed, direction_seed = [
            int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(2)
        ]
        candidates = sample_trait(
            trait,
            count=args.candidate_points,
            seed=surface_seed,
            grid_resolution=int(manifest["grid_resolution"]),
        )
        rng = np.random.default_rng(direction_seed)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        projection = (candidates - np.asarray(metadata["trait"]["center"])) @ direction
        selected = np.argpartition(projection, -retained_count)[-retained_count:]
        selected = selected[np.argsort(projection[selected])]
        occluded = candidates[selected]
        path = case_root / "occlusion_cap_80.ply"
        write_ply(path, occluded)

        metadata.setdefault("seeds", {})["occlusion_cap_80_surface"] = surface_seed
        metadata["seeds"]["occlusion_cap_80_direction"] = direction_seed
        metadata.setdefault("conditions", {})["occlusion_cap_80.ply"] = {
            "points": int(len(occluded)),
            "spatially_coherent_missing_fraction": float(1.0 - args.retained_fraction),
            "construction": "retain the top projection quantile along one deterministic random direction",
            "direction": direction.tolist(),
            "candidate_surface_points": int(args.candidate_points),
            "independent_surface_sample": True,
        }
        metadata.setdefault("sha256", {})[path.name] = sha256(path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{record['case']}: {len(occluded)} coherent-cap points")

    manifest.setdefault("augmentations", {})["occlusion_cap_80"] = {
        "generator": str(Path(__file__).resolve()),
        "filename": "occlusion_cap_80.ply",
        "candidate_points": int(args.candidate_points),
        "retained_fraction": float(args.retained_fraction),
        "spatially_coherent": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
