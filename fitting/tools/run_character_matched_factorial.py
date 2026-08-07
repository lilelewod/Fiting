"""Run a resumable optimizer-by-initialization factorial on matched characters."""

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
DATA = PROJECT / "datasets" / "character" / "test"
TOKEN_RE = re.compile(r"run(\d+)_test(\d+)_1\.mat$")


def discover_cases(noise_type: str, noise_level: str) -> list[tuple[int, int]]:
    cases = []
    for token in sorted(DATA.glob("run*_test*_1.mat")):
        match = TOKEN_RE.match(token.name)
        if match is None:
            continue
        run_id, test_id = map(int, match.groups())
        image = DATA / noise_type / noise_level / str(test_id - 1) / f"noisy_{run_id}.png"
        if image.is_file():
            cases.append((run_id, test_id))
    return cases


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def key(cell: dict) -> tuple[int, int, int, str]:
    return cell["repeat"], cell["run_id"], cell["test_id"], cell["method"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-type", default="saltpepper_noise")
    parser.add_argument("--noise-level", default="0.6")
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pso", "guided_pso", "cs", "guided_cs"],
        default=["pso", "guided_pso", "cs", "guided_cs"],
    )
    parser.add_argument("--max-evaluations", type=int, default=10000)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260727)
    parser.add_argument("--guided-fraction", type=float, default=0.5)
    parser.add_argument("--guided-sigma", type=float, default=0.15)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cases = discover_cases(args.noise_type, args.noise_level)
    if not cases:
        raise RuntimeError("no complete matched character cases found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    protocol = {
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "repeats": args.repeats,
        "methods": args.methods,
        "max_evaluations": args.max_evaluations,
        "population": args.population,
        "base_seed": args.base_seed,
        "guided_fraction": args.guided_fraction,
        "guided_sigma": args.guided_sigma,
        "cases": [list(case) for case in cases],
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, value in protocol.items():
            if manifest.get(name) != value:
                raise ValueError(f"resume configuration mismatch for {name}")
        manifest["status"] = "RUNNING"
        manifest["resumed_at"] = datetime.now().isoformat()
    else:
        manifest = {
            "status": "RUNNING",
            "started_at": datetime.now().isoformat(),
            **protocol,
            "seed_formula": "base_seed + 10000*(repeat-1) + 100*case_index",
            "cells": [],
        }
    save(manifest_path, manifest)

    complete = {
        key(cell)
        for cell in manifest["cells"]
        if cell.get("status") == "COMPLETE"
    }
    total = len(args.repeats) * len(cases) * len(args.methods)
    for repeat in args.repeats:
        for case_index, (run_id, test_id) in enumerate(cases):
            paired_seed = args.base_seed + 10000 * (repeat - 1) + 100 * case_index
            for method in args.methods:
                cell_key = (repeat, run_id, test_id, method)
                if cell_key in complete:
                    continue
                algorithm = "pso" if method == "guided_pso" else "cs" if method == "guided_cs" else method
                name = f"rep{repeat}_run{run_id}_test{test_id}_{method}"
                stdout_path = args.output_root / f"{name}.stdout.log"
                stderr_path = args.output_root / f"{name}.stderr.log"
                command = [
                    sys.executable,
                    "entrypoints/fit_character.py",
                    "--algo",
                    algorithm,
                    "--estimator",
                    "mm",
                    "--nearest-neighbor-backend",
                    "sklearn",
                    "--run-id",
                    str(run_id),
                    "--test-id",
                    str(test_id),
                    "--num-envs",
                    "1",
                    "--episodes-per-env",
                    str(args.population),
                    "--max-episode",
                    str(args.max_evaluations),
                    "--runs",
                    "1",
                    "--seed",
                    str(paired_seed),
                    "--visualization",
                    "none",
                ]
                if method in {"guided_pso", "guided_cs"}:
                    command.extend(
                        [
                            "--template-guided-initialization",
                            "--template-guided-fraction",
                            str(args.guided_fraction),
                            "--template-guided-sigma",
                            str(args.guided_sigma),
                        ]
                    )
                cell = {
                    "repeat": repeat,
                    "case_index": case_index,
                    "run_id": run_id,
                    "test_id": test_id,
                    "method": method,
                    "seed": paired_seed,
                    "status": "RUNNING",
                    "started_at": datetime.now().isoformat(),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
                manifest["cells"] = [old for old in manifest["cells"] if key(old) != cell_key]
                manifest["cells"].append(cell)
                save(manifest_path, manifest)
                start = time.perf_counter()
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr:
                    result = subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
                cell.update(
                    {
                        "elapsed_seconds": time.perf_counter() - start,
                        "return_code": result.returncode,
                        "status": "COMPLETE" if result.returncode == 0 else "FAILED",
                        "finished_at": datetime.now().isoformat(),
                    }
                )
                if result.returncode == 0:
                    log = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    match = re.search(r"current timestamp is ([^\r\n]+)", log)
                    if match is None:
                        raise RuntimeError(f"timestamp missing from {stdout_path}")
                    date, clock = match.group(1).strip().split("/")
                    record_path = (
                        OUTPUTS
                        / algorithm
                        / "character"
                        / args.noise_type
                        / args.noise_level
                        / str(test_id - 1)
                        / f"noisy_{run_id}"
                        / date
                        / clock
                        / "record.json"
                    )
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    cell.update(
                        {
                            "record": str(record_path),
                            "action_dim": int(record["action_dim"]),
                            "score": float(record["best_score"]),
                            "chamfer": float(record["chamfer"]),
                        }
                    )
                    complete.add(cell_key)
                manifest["progress"] = {"complete": len(complete), "total": total}
                save(manifest_path, manifest)
                if result.returncode != 0:
                    manifest["status"] = "FAILED"
                    save(manifest_path, manifest)
                    return result.returncode

    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = datetime.now().isoformat()
    manifest["progress"] = {"complete": len(complete), "total": total}
    save(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
