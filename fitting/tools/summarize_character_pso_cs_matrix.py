"""Summarize paired character PSO/CS records and create a paper-ready figure."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


def record_from_cell(cell):
    log = Path(cell["stdout"]).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"current timestamp is ([^\r\n]+)", log)
    if not match:
        raise RuntimeError(f"timestamp missing from {cell['stdout']}")
    date, clock = match.group(1).strip().split("/")
    return (
        OUTPUTS / cell["algorithm"] / "character" / "saltpepper_noise" / "0.6"
        / str(cell["test_id"] - 1) / f"noisy_{cell['run_id']}" / date / clock / "record.json"
    )


def load_row(path, run_id, test_id, algorithm, elapsed_seconds=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "run_id": int(run_id),
        "test_id": int(test_id),
        "case": f"r{run_id}-t{test_id}",
        "algorithm": algorithm.upper(),
        "score": float(record["best_score"]),
        "chamfer": float(record["chamfer"]),
        "d2m": float(record["d2m"]),
        "m2d": float(record["m2d"]),
        "evaluations": int(record["num_evaluations"]),
        "elapsed_seconds": float(elapsed_seconds if elapsed_seconds is not None else record["evolving_times"][-1]),
        "record": str(Path(path).resolve()),
    }


def paired_summary(rows, metric, higher_is_better):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["algorithm"]] = row
    pso = np.array([by_case[case]["PSO"][metric] for case in by_case], dtype=float)
    cs = np.array([by_case[case]["CS"][metric] for case in by_case], dtype=float)
    delta = pso - cs if higher_is_better else cs - pso
    test = wilcoxon(delta, alternative="two-sided", method="exact")
    return {
        "pso_median": float(np.median(pso)),
        "cs_median": float(np.median(cs)),
        "pso_mean": float(np.mean(pso)),
        "cs_mean": float(np.mean(cs)),
        "pso_wins": int(np.sum(delta > 0)),
        "cs_wins": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p": float(test.pvalue),
    }


def plot(rows, summary, prefix):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["algorithm"]] = row
    cases = list(by_case)
    x = np.arange(len(cases))
    colors = {"PSO": "#2b6cb0", "CS": "#d95f02"}
    markers = {"PSO": "o", "CS": "s"}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9.0, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.2,
        "legend.fontsize": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), gridspec_kw={"wspace": 0.28})
    panels = [
        ("score", "(a) PMF fitting similarity", "Best PMF similarity (higher is better)"),
        ("chamfer", "(b) Geometric discrepancy", "Chamfer distance (lower is better)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for index, case in enumerate(cases):
            values = [by_case[case][algorithm][metric] for algorithm in ("PSO", "CS")]
            ax.plot([index - 0.09, index + 0.09], values, color="#b8bec6", linewidth=0.8, zorder=1)
        for offset, algorithm in ((-0.09, "PSO"), (0.09, "CS")):
            values = [by_case[case][algorithm][metric] for case in cases]
            ax.scatter(x + offset, values, s=24, color=colors[algorithm], marker=markers[algorithm],
                       edgecolor="white", linewidth=0.45, label=algorithm, zorder=2)
        stats = summary[metric]
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, cases, rotation=35, ha="right")
        ax.grid(axis="y", color="#d7dce2", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02, 0.97,
            f"wins P/C = {stats['pso_wins']}/{stats['cs_wins']};  exact p = {stats['wilcoxon_p']:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
        )
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.25)
    fig.savefig(prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extra-pso", type=Path, required=True)
    parser.add_argument("--extra-cs", type=Path, required=True)
    parser.add_argument("--extra-case", default="1:2")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("matrix is not complete")
    rows = []
    for cell in manifest["cells"]:
        if cell["status"] != "COMPLETE":
            raise RuntimeError(f"incomplete cell: {cell}")
        rows.append(load_row(record_from_cell(cell), cell["run_id"], cell["test_id"],
                             cell["algorithm"], cell["elapsed_seconds"]))
    run_id, test_id = map(int, args.extra_case.split(":"))
    rows.append(load_row(args.extra_pso, run_id, test_id, "pso"))
    rows.append(load_row(args.extra_cs, run_id, test_id, "cs"))
    rows.sort(key=lambda row: (row["run_id"], row["test_id"], row["algorithm"]))

    summary = {
        "status": "COMPLETE",
        "cases": len(rows) // 2,
        "records": len(rows),
        "score": paired_summary(rows, "score", True),
        "chamfer": paired_summary(rows, "chamfer", False),
        "runtime": {
            algorithm: {
                "median_seconds": float(np.median([r["elapsed_seconds"] for r in rows if r["algorithm"] == algorithm])),
                "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in rows if r["algorithm"] == algorithm])),
            }
            for algorithm in ("PSO", "CS")
        },
        "rows": rows,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, summary, args.output_prefix)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
