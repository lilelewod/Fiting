"""Combine compatible optimizer-comparison result roots without rerunning fits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row if key != "trait"))
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    rows = []
    for input_root in args.inputs:
        with open(Path(input_root) / "results.json", encoding="utf-8") as stream:
            rows.extend(json.load(stream))
    keys = [(int(row["seed"]), row["algorithm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed/algorithm pairs across inputs")
    required_protocol = (
        "evaluations", "gt_metric_threshold", "evaluation_points", "evaluation_grid",
        "evaluation_reference_seed", "evaluation_model_seed", "evaluation_reference_mode",
    )
    for name in required_protocol:
        values = {str(row.get(name)) for row in rows}
        if len(values) != 1:
            raise ValueError(f"incompatible {name}: {sorted(values)}")

    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    seed_order = sorted({int(row["seed"]) for row in rows})
    repeat_map = {seed: index + 1 for index, seed in enumerate(seed_order)}
    rows.sort(key=lambda row: (int(row["seed"]), algorithms.index(row["algorithm"])))
    for row in rows:
        row["repeat"] = repeat_map[int(row["seed"])]

    summaries = []
    metric_names = (
        "best_score", "wall_time_s", "input_chamfer", "input_fscore",
        "gt_chamfer", "gt_fscore", "success",
    )
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        summary = {"algorithm": algorithm, "runs": len(selected)}
        for name in metric_names:
            values = np.asarray([row[name] for row in selected], dtype=float)
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{name}_median"] = float(np.median(values))
            summary[f"{name}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        summaries.append(summary)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "results.json", "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    with open(output_root / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    write_csv(output_root / "results.csv", rows)
    write_csv(output_root / "summary.csv", summaries)
    print(f"Combined {len(seed_order)} seeds and {len(rows)} rows into {output_root}")


if __name__ == "__main__":
    main()
