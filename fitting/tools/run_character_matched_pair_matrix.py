"""Run one paired PSO/CS fit for every converted character token/image pair."""

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
DATA = PROJECT / "datasets/character/test"
TOKEN_RE = re.compile(r"run(\d+)_test(\d+)_1\.mat$")


def discover_cases(noise_type: str, noise_level: str):
    cases = []
    for token in sorted(DATA.glob("run*_test*_1.mat")):
        match = TOKEN_RE.match(token.name)
        if not match:
            continue
        run_id, test_id = map(int, match.groups())
        image = DATA / noise_type / noise_level / str(test_id - 1) / f"noisy_{run_id}.png"
        if image.is_file():
            cases.append((run_id, test_id))
    return cases


def write_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-type", default="saltpepper_noise")
    parser.add_argument("--noise-level", default="0.6")
    parser.add_argument("--max-evaluations", type=int, default=10000)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--exclude", action="append", default=[], help="Case as RUN:TEST")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    excluded = {tuple(map(int, item.split(":"))) for item in args.exclude}
    cases = [case for case in discover_cases(args.noise_type, args.noise_level) if case not in excluded]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest = {
        "status": "RUNNING",
        "started_at": datetime.now().isoformat(),
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "max_evaluations": args.max_evaluations,
        "population": args.population,
        "excluded": sorted([list(case) for case in excluded]),
        "cases": [list(case) for case in cases],
        "cells": [],
    }
    write_manifest(manifest_path, manifest)

    for case_index, (run_id, test_id) in enumerate(cases):
        paired_seed = args.seed + 100 * case_index
        for algorithm in ("pso", "cs"):
            cell_name = f"run{run_id}_test{test_id}_{algorithm}"
            stdout_path = args.output_root / f"{cell_name}.stdout.log"
            stderr_path = args.output_root / f"{cell_name}.stderr.log"
            command = [
                sys.executable,
                "entrypoints/fit_character.py",
                "--algo", algorithm,
                "--estimator", "mm",
                "--nearest-neighbor-backend", "sklearn",
                "--run-id", str(run_id),
                "--test-id", str(test_id),
                "--num-envs", "1",
                "--episodes-per-env", str(args.population),
                "--max-episode", str(args.max_evaluations),
                "--runs", "1",
                "--seed", str(paired_seed),
                "--visualization", "none",
            ]
            cell = {
                "run_id": run_id,
                "test_id": test_id,
                "algorithm": algorithm,
                "seed": paired_seed,
                "status": "RUNNING",
                "started_at": datetime.now().isoformat(),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            manifest["cells"].append(cell)
            write_manifest(manifest_path, manifest)
            start = time.perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
            cell["elapsed_seconds"] = time.perf_counter() - start
            cell["return_code"] = result.returncode
            cell["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
            cell["finished_at"] = datetime.now().isoformat()
            write_manifest(manifest_path, manifest)
            if result.returncode != 0:
                manifest["status"] = "FAILED"
                write_manifest(manifest_path, manifest)
                return result.returncode

    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = datetime.now().isoformat()
    write_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
