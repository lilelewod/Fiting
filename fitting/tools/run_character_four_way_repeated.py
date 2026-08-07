"""Run resumable repeated four-way character fitting for three search variants."""

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


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def cell_key(cell: dict) -> tuple[int, int, int, str]:
    return (
        int(cell["repeat"]),
        int(cell["observation"]),
        int(cell["candidate"]),
        str(cell["method"]),
    )


def update_predictions(manifest: dict, classes: list[int], repeats: list[int]) -> None:
    completed = {
        cell_key(cell): cell
        for cell in manifest["cells"]
        if cell.get("status") == "COMPLETE"
    }
    predictions = {}
    for repeat in repeats:
        repeat_result = {}
        for method in manifest["methods"]:
            items = []
            for observation in classes:
                candidates = [
                    completed.get((repeat, observation, candidate, method))
                    for candidate in classes
                ]
                if any(cell is None for cell in candidates):
                    continue
                scores = {int(cell["candidate"]): float(cell["score"]) for cell in candidates}
                prediction = max(scores, key=scores.get)
                competitor = max(score for candidate, score in scores.items() if candidate != observation)
                ordered = sorted(scores.values(), reverse=True)
                items.append(
                    {
                        "observation": observation,
                        "prediction": prediction,
                        "correct": prediction == observation,
                        "true_class_margin": scores[observation] - competitor,
                        "winner_margin": ordered[0] - ordered[1],
                        "scores": {str(candidate): score for candidate, score in scores.items()},
                    }
                )
            if len(items) == len(classes):
                correct = sum(item["correct"] for item in items)
                repeat_result[method] = {
                    "correct": correct,
                    "total": len(items),
                    "accuracy": correct / len(items),
                    "items": items,
                }
        predictions[str(repeat)] = repeat_result
    manifest["predictions"] = predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pso", "guided_pso", "cs", "guided_cs"],
        default=["pso", "guided_pso", "cs"],
    )
    parser.add_argument("--max-evaluations", type=int, default=10000)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260730)
    parser.add_argument("--guided-fraction", type=float, default=0.5)
    parser.add_argument("--guided-sigma", type=float, default=0.15)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for class_id in args.classes:
        token = PROJECT / f"datasets/character/test/run{args.run_id}_test{class_id}_1.mat"
        image = PROJECT / (
            f"datasets/character/test/saltpepper_noise/0.6/{class_id - 1}/"
            f"noisy_{args.run_id}.png"
        )
        if not token.is_file() or not image.is_file():
            raise FileNotFoundError(
                f"incomplete class {class_id}: token={token.is_file()} image={image.is_file()}"
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "run_id": args.run_id,
            "classes": args.classes,
            "repeats": args.repeats,
            "methods": args.methods,
            "max_evaluations": args.max_evaluations,
            "population": args.population,
            "base_seed": args.base_seed,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"resume configuration mismatch for {key}")
        manifest["status"] = "RUNNING"
        manifest["resumed_at"] = datetime.now().isoformat()
    else:
        manifest = {
            "status": "RUNNING",
            "started_at": datetime.now().isoformat(),
            "run_id": args.run_id,
            "classes": args.classes,
            "repeats": args.repeats,
            "methods": args.methods,
            "max_evaluations": args.max_evaluations,
            "population": args.population,
            "base_seed": args.base_seed,
            "guided_fraction": args.guided_fraction,
            "guided_sigma": args.guided_sigma,
            "seed_formula": "base_seed + 10000*(repeat-1) + 100*observation + candidate",
            "cells": [],
            "predictions": {},
        }
    save(manifest_path, manifest)

    completed = {
        cell_key(cell)
        for cell in manifest["cells"]
        if cell.get("status") == "COMPLETE"
    }
    total = len(args.repeats) * len(args.classes) ** 2 * len(args.methods)

    for repeat in args.repeats:
        for observation in args.classes:
            for candidate in args.classes:
                paired_seed = (
                    args.base_seed + 10000 * (repeat - 1) + 100 * observation + candidate
                )
                for method in args.methods:
                    key = (repeat, observation, candidate, method)
                    if key in completed:
                        continue
                    if method == "guided_pso":
                        algorithm = "pso"
                    elif method == "guided_cs":
                        algorithm = "cs"
                    else:
                        algorithm = method
                    name = f"rep{repeat}_obs{observation}_cand{candidate}_{method}"
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
                        str(args.run_id),
                        "--test-id",
                        str(observation),
                        "--template-test-id",
                        str(candidate),
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
                        "observation": observation,
                        "candidate": candidate,
                        "method": method,
                        "seed": paired_seed,
                        "status": "RUNNING",
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "started_at": datetime.now().isoformat(),
                    }
                    manifest["cells"] = [
                        old for old in manifest["cells"] if cell_key(old) != key
                    ]
                    manifest["cells"].append(cell)
                    save(manifest_path, manifest)
                    start = time.perf_counter()
                    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                        "w", encoding="utf-8"
                    ) as stderr:
                        result = subprocess.run(
                            command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False
                        )
                    cell["elapsed_seconds"] = time.perf_counter() - start
                    cell["return_code"] = result.returncode
                    cell["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
                    cell["finished_at"] = datetime.now().isoformat()
                    if result.returncode == 0:
                        log = stdout_path.read_text(encoding="utf-8", errors="ignore")
                        match = re.search(r"current timestamp is ([^\r\n]+)", log)
                        if match is None:
                            raise RuntimeError(f"timestamp missing from {stdout_path}")
                        date, clock = match.group(1).strip().split("/")
                        record_path = (
                            OUTPUTS
                            / algorithm
                            / "character_classification"
                            / "saltpepper_noise"
                            / "0.6"
                            / f"run_{args.run_id}"
                            / f"observation_{observation}"
                            / f"candidate_{candidate}"
                            / date
                            / clock
                            / "record.json"
                        )
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        cell.update(
                            {
                                "record": str(record_path),
                                "score": float(record["best_score"]),
                                "chamfer": float(record["chamfer"]),
                            }
                        )
                        completed.add(key)
                    update_predictions(manifest, args.classes, args.repeats)
                    manifest["progress"] = {
                        "complete": len(completed),
                        "total": total,
                    }
                    save(manifest_path, manifest)
                    if result.returncode != 0:
                        manifest["status"] = "FAILED"
                        save(manifest_path, manifest)
                        return result.returncode

    update_predictions(manifest, args.classes, args.repeats)
    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = datetime.now().isoformat()
    manifest["progress"] = {"complete": len(completed), "total": total}
    save(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
