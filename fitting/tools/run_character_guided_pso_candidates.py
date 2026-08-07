"""Run template-guided PSO for selected candidate classes of one observation."""

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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--observation", type=int, default=None)
    parser.add_argument("--observations", type=int, nargs="+", default=None)
    parser.add_argument("--candidates", type=int, nargs="+", required=True)
    parser.add_argument("--max-evaluations", type=int, default=10000)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--guided-fraction", type=float, default=0.5)
    parser.add_argument("--guided-sigma", type=float, default=0.15)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest = {"status": "RUNNING", "started_at": datetime.now().isoformat(), "cells": []}
    save(manifest_path, manifest)
    observations = args.observations or ([args.observation] if args.observation is not None else [])
    if not observations:
        raise ValueError("provide --observation or --observations")
    for observation in observations:
      for candidate in args.candidates:
        paired_seed = args.seed + 100 * observation + candidate
        stdout_path = args.output_root / f"observation_{observation}_candidate_{candidate}.stdout.log"
        stderr_path = args.output_root / f"observation_{observation}_candidate_{candidate}.stderr.log"
        command = [
            sys.executable, "entrypoints/fit_character.py",
            "--algo", "pso", "--estimator", "mm",
            "--nearest-neighbor-backend", "sklearn",
            "--run-id", str(args.run_id), "--test-id", str(observation),
            "--template-test-id", str(candidate), "--num-envs", "1",
            "--episodes-per-env", str(args.population),
            "--max-episode", str(args.max_evaluations), "--runs", "1",
            "--seed", str(paired_seed), "--visualization", "none",
            "--pso-template-guided-initialization",
            "--pso-template-guided-fraction", str(args.guided_fraction),
            "--pso-template-guided-sigma", str(args.guided_sigma),
        ]
        cell = {"observation": observation, "candidate": candidate, "status": "RUNNING", "seed": paired_seed}
        manifest["cells"].append(cell)
        save(manifest_path, manifest)
        start = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
        cell.update({
            "status": "COMPLETE" if result.returncode == 0 else "FAILED",
            "return_code": result.returncode,
            "elapsed_seconds": time.perf_counter() - start,
        })
        if result.returncode == 0:
            log = stdout_path.read_text(encoding="utf-8", errors="ignore")
            timestamp = re.search(r"current timestamp is ([^\r\n]+)", log).group(1).strip()
            date, clock = timestamp.split("/")
            record_path = (
                OUTPUTS / "pso" / "character_classification" / "saltpepper_noise" / "0.6"
                / f"run_{args.run_id}" / f"observation_{observation}" / f"candidate_{candidate}"
                / date / clock / "record.json"
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            cell.update({"score": record["best_score"], "chamfer": record["chamfer"], "record": str(record_path)})
        save(manifest_path, manifest)
        if result.returncode != 0:
            manifest["status"] = "FAILED"
            save(manifest_path, manifest)
            return result.returncode
    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = datetime.now().isoformat()
    save(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
