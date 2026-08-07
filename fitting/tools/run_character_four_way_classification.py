"""Run a paired four-way character-classification smoke experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT.parent / "outputs"


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-evaluations", type=int, default=1000)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for class_id in args.classes:
        token = PROJECT / f"datasets/character/test/run{args.run_id}_test{class_id}_1.mat"
        image = PROJECT / f"datasets/character/test/saltpepper_noise/0.6/{class_id - 1}/noisy_{args.run_id}.png"
        if not token.is_file() or not image.is_file():
            raise FileNotFoundError(f"incomplete class {class_id}: token={token.is_file()} image={image.is_file()}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest = {
        "status": "RUNNING",
        "started_at": datetime.now().isoformat(),
        "run_id": args.run_id,
        "classes": args.classes,
        "max_evaluations": args.max_evaluations,
        "population": args.population,
        "cells": [],
        "predictions": {},
    }
    save(manifest_path, manifest)

    for observation in args.classes:
        for candidate in args.classes:
            paired_seed = args.seed + 100 * observation + candidate
            for algorithm in ("pso", "cs"):
                name = f"obs{observation}_cand{candidate}_{algorithm}"
                stdout_path = args.output_root / f"{name}.stdout.log"
                stderr_path = args.output_root / f"{name}.stderr.log"
                command = [
                    sys.executable, "entrypoints/fit_character.py",
                    "--algo", algorithm,
                    "--estimator", "mm",
                    "--nearest-neighbor-backend", "sklearn",
                    "--run-id", str(args.run_id),
                    "--test-id", str(observation),
                    "--template-test-id", str(candidate),
                    "--num-envs", "1",
                    "--episodes-per-env", str(args.population),
                    "--max-episode", str(args.max_evaluations),
                    "--runs", "1",
                    "--seed", str(paired_seed),
                    "--visualization", "none",
                ]
                cell = {
                    "observation": observation,
                    "candidate": candidate,
                    "algorithm": algorithm,
                    "seed": paired_seed,
                    "status": "RUNNING",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "started_at": datetime.now().isoformat(),
                }
                manifest["cells"].append(cell)
                save(manifest_path, manifest)
                start = time.perf_counter()
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
                cell["elapsed_seconds"] = time.perf_counter() - start
                cell["return_code"] = result.returncode
                cell["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
                cell["finished_at"] = datetime.now().isoformat()
                if result.returncode == 0:
                    log = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    timestamp = re.search(r"current timestamp is ([^\r\n]+)", log).group(1).strip()
                    date, clock = timestamp.split("/")
                    record_path = (
                        OUTPUTS / algorithm / "character_classification" / "saltpepper_noise" / "0.6"
                        / f"run_{args.run_id}" / f"observation_{observation}" / f"candidate_{candidate}"
                        / date / clock / "record.json"
                    )
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    cell["record"] = str(record_path)
                    cell["score"] = float(record["best_score"])
                    cell["chamfer"] = float(record["chamfer"])
                save(manifest_path, manifest)
                if result.returncode != 0:
                    manifest["status"] = "FAILED"
                    save(manifest_path, manifest)
                    return result.returncode

    for algorithm in ("pso", "cs"):
        correct = 0
        predictions = []
        for observation in args.classes:
            candidates = [
                cell for cell in manifest["cells"]
                if cell["algorithm"] == algorithm and cell["observation"] == observation
            ]
            winner = max(candidates, key=lambda cell: cell["score"])
            correct += int(winner["candidate"] == observation)
            ordered = sorted(candidates, key=lambda cell: cell["score"], reverse=True)
            predictions.append({
                "observation": observation,
                "prediction": winner["candidate"],
                "correct": winner["candidate"] == observation,
                "margin": ordered[0]["score"] - ordered[1]["score"],
                "scores": {str(cell["candidate"]): cell["score"] for cell in candidates},
            })
        manifest["predictions"][algorithm] = {
            "correct": correct,
            "total": len(args.classes),
            "accuracy": correct / len(args.classes),
            "items": predictions,
        }
    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = datetime.now().isoformat()
    save(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
