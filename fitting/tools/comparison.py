"""Plot aligned best-so-far convergence curves from fitting record files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_records(patterns, required_evaluations=None):
    records = []
    seen = set()
    for pattern in patterns:
        for filename in glob.glob(pattern, recursive=True):
            path = str(Path(filename).resolve())
            if path in seen:
                continue
            seen.add(path)
            with open(path, "r", encoding="utf-8") as stream:
                record = json.load(stream)
            evaluations = int(record.get("num_evaluations", 0))
            if required_evaluations is not None and evaluations != required_evaluations:
                continue
            episodes = np.asarray(record["evolving_episodes"], dtype=int)
            scores = np.asarray(record["evolving_scores"], dtype=float)
            if episodes.size == 0 or episodes.size != scores.size:
                continue
            order = np.argsort(episodes)
            records.append({
                "path": path,
                "episodes": episodes[order],
                "scores": np.maximum.accumulate(scores[order]),
                "evaluations": evaluations,
            })
    if not records:
        raise FileNotFoundError(f"no usable records matched: {patterns}")
    return records


def align(records, max_evaluations):
    grid = np.unique(np.concatenate([
        np.array([0, max_evaluations], dtype=int),
        *[record["episodes"][record["episodes"] <= max_evaluations] for record in records],
    ]))
    aligned = np.zeros((len(records), grid.size), dtype=float)
    for row, record in enumerate(records):
        episodes = np.insert(record["episodes"], 0, 0)
        scores = np.insert(record["scores"], 0, 0.0)
        indices = np.searchsorted(episodes, grid, side="right") - 1
        aligned[row] = scores[np.maximum(indices, 0)]
    return grid, aligned


def parse_series(items):
    series = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--series must be LABEL=GLOB")
        label, pattern = item.split("=", 1)
        series.setdefault(label, []).append(pattern)
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True, help="LABEL=GLOB; repeat as needed")
    parser.add_argument("--max-evaluations", type=int, required=True)
    parser.add_argument("--required-evaluations", type=int, default=None)
    parser.add_argument("--statistic", choices=["mean", "median"], default="median")
    parser.add_argument("--band", choices=["std", "iqr", "none"], default="iqr")
    parser.add_argument("--xlabel", default="Function evaluations")
    parser.add_argument("--ylabel", default="Best fitting similarity")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    styles = [
        ("#2b6cb0", "-"), ("#d95f02", "--"), ("#2f855a", "-."),
        ("#805ad5", ":"), ("#718096", "-"),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.4,
        "axes.titlesize": 9.2, "axes.labelsize": 8.6,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    metadata = {"max_evaluations": args.max_evaluations, "series": {}}

    for index, (label, patterns) in enumerate(parse_series(args.series).items()):
        records = load_records(patterns, args.required_evaluations)
        grid, values = align(records, args.max_evaluations)
        if args.statistic == "median":
            center = np.median(values, axis=0)
        else:
            center = np.mean(values, axis=0)
        color, linestyle = styles[index % len(styles)]
        ax.plot(grid, center, color=color, linestyle=linestyle, linewidth=1.6,
                label=f"{label} (n={len(records)})")
        if args.band == "iqr":
            lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
            ax.fill_between(grid, lower, upper, color=color, alpha=0.16, linewidth=0)
        elif args.band == "std":
            spread = np.std(values, axis=0)
            ax.fill_between(grid, center - spread, center + spread,
                            color=color, alpha=0.16, linewidth=0)
        metadata["series"][label] = {
            "records": len(records),
            "patterns": patterns,
            "final_center": float(center[-1]),
            "record_files": [record["path"] for record in records],
        }

    ax.set_xlim(0, args.max_evaluations)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title, loc="left", fontweight="semibold")
    ax.grid(True, color="#d7dce2", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.show:
        plt.show()
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
