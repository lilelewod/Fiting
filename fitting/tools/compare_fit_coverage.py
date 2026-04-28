#!/usr/bin/env python3
"""Compare point-cloud fitting coverage for Fiting and NURBSFit outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from textwrap import wrap

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


DEFAULT_DATA = "/home/m25lll/code/nurbsfit/data/input/00873042_lessp/00873042_lessp.ply"
NURBSFIT_CURRENT = "/home/m25lll/code/nurbsfit/data/output/uv_trimmed_surface_color/mask_80.0_theta_1.0"
NURBSFIT_BEFORE = "/home/m25lll/code/nurbsfit/data/output/00873042_lessp_before_paper_params/uv_trimmed_surface_color/mask_80.0_theta_1.0"
FITING_RUN_10 = "/home/m25lll/code/Fiting/fitting/outputs/cco/3d/nurbs_surface/nurbs/00873042_lessp/run_10/2026-0428/1513-28"


def _load_vertices(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    if vertices.size:
        return vertices

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if points.size:
        return points

    raise ValueError(f"no vertices/points found in {path}")


def _expand_input(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_dir():
        files = []
        for suffix in ("*.ply", "*.off"):
            files.extend(path.glob(suffix))
        return sorted(files)

    if any(char in pattern for char in "*?[]"):
        return sorted(Path().glob(pattern))

    return [path]


def collect_vertices(inputs: list[str], include_combined: bool = False) -> tuple[np.ndarray, list[Path]]:
    files: list[Path] = []
    for item in inputs:
        files.extend(_expand_input(item))

    files = [path for path in files if path.suffix.lower() in {".ply", ".off"}]
    if not include_combined:
        files = [path for path in files if "combined" not in path.stem.lower()]
    if not files:
        raise FileNotFoundError(f"no .ply/.off files found from: {inputs}")

    vertices = [_load_vertices(path) for path in files]
    return np.vstack(vertices), files


def compute_metrics(data_points: np.ndarray, model_points: np.ndarray, threshold: float) -> dict[str, float]:
    data_tree = KDTree(data_points)
    model_tree = KDTree(model_points)

    data_to_model = model_tree.query(data_points, k=1, return_distance=True)[0].reshape(-1)
    model_to_data = data_tree.query(model_points, k=1, return_distance=True)[0].reshape(-1)

    return {
        "data_points": int(data_points.shape[0]),
        "model_points": int(model_points.shape[0]),
        "threshold": threshold,
        "data_covered": float(np.mean(data_to_model <= threshold)),
        "data_uncovered": float(np.mean(data_to_model > threshold)),
        "mean_data_to_model": float(np.mean(data_to_model)),
        "max_data_to_model": float(np.max(data_to_model)),
        "model_over_ratio": float(np.mean(model_to_data > threshold)),
        "mean_model_to_data": float(np.mean(model_to_data)),
        "max_model_to_data": float(np.max(model_to_data)),
    }


def print_metrics(name: str, files: list[Path], metrics: dict[str, float]) -> None:
    print(f"\n[{name}]")
    print(f"files: {len(files)}")
    for path in files:
        print(f"  {path}")
    print(f"data_points: {metrics['data_points']}")
    print(f"model_points: {metrics['model_points']}")
    print(f"threshold: {metrics['threshold']:.6f}")
    print(f"data_covered: {metrics['data_covered']:.6f} ({metrics['data_covered'] * 100:.2f}%)")
    print(f"data_uncovered: {metrics['data_uncovered']:.6f} ({metrics['data_uncovered'] * 100:.2f}%)")
    print(f"mean_data_to_model: {metrics['mean_data_to_model']:.6f}")
    print(f"max_data_to_model: {metrics['max_data_to_model']:.6f}")
    print(f"model_over_ratio: {metrics['model_over_ratio']:.6f} ({metrics['model_over_ratio'] * 100:.2f}%)")
    print(f"mean_model_to_data: {metrics['mean_model_to_data']:.6f}")
    print(f"max_model_to_data: {metrics['max_model_to_data']:.6f}")


def parse_case(value: str) -> tuple[str, list[str]]:
    if "=" in value:
        name, raw_paths = value.split("=", 1)
        paths = [part for part in raw_paths.split(",") if part]
        return name, paths

    path = Path(value)
    return path.name or value, [value]


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "name",
        "files",
        "data_points",
        "model_points",
        "threshold",
        "data_covered",
        "data_uncovered",
        "mean_data_to_model",
        "max_data_to_model",
        "model_over_ratio",
        "mean_model_to_data",
        "max_model_to_data",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pdf(path: Path, rows: list[dict[str, float | str]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = [str(row["name"]) for row in rows]
    wrapped_names = ["\n".join(wrap(name, 18)) for name in names]
    covered = [float(row["data_covered"]) * 100.0 for row in rows]
    over = [float(row["model_over_ratio"]) * 100.0 for row in rows]
    mean_data_to_model = [float(row["mean_data_to_model"]) for row in rows]

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        fig.suptitle("Fitting Coverage Comparison", fontsize=16, fontweight="bold")

        ax_table = fig.add_subplot(grid[0, :])
        ax_table.axis("off")
        table_rows = []
        for row in rows:
            table_rows.append(
                [
                    str(row["name"]),
                    str(row["files"]),
                    f"{float(row['data_covered']) * 100:.2f}%",
                    f"{float(row['data_uncovered']) * 100:.2f}%",
                    f"{float(row['model_over_ratio']) * 100:.2f}%",
                    f"{float(row['mean_data_to_model']):.6f}",
                    f"{float(row['mean_model_to_data']):.6f}",
                ]
            )
        table = ax_table.table(
            cellText=table_rows,
            colLabels=[
                "case",
                "files",
                "covered",
                "uncovered",
                "over",
                "mean data->model",
                "mean model->data",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)

        ax_cov = fig.add_subplot(grid[1, 0])
        ax_cov.bar(wrapped_names, covered, color="#4c78a8")
        ax_cov.set_title("Data Coverage")
        ax_cov.set_ylabel("covered points (%)")
        ax_cov.set_ylim(0, 105)
        ax_cov.tick_params(axis="x", labelrotation=0, labelsize=8)
        ax_cov.grid(axis="y", alpha=0.25)
        for index, value in enumerate(covered):
            ax_cov.text(index, value + 1, f"{value:.2f}%", ha="center", fontsize=8)

        ax_err = fig.add_subplot(grid[1, 1])
        x = np.arange(len(rows))
        width = 0.36
        ax_err.bar(x - width / 2, over, width, label="over (%)", color="#e45756")
        ax_err_2 = ax_err.twinx()
        ax_err_2.bar(
            x + width / 2,
            mean_data_to_model,
            width,
            label="mean data->model",
            color="#72b7b2",
        )
        ax_err.set_title("Overrun And Mean Distance")
        ax_err.set_ylabel("overrun vertices (%)")
        ax_err_2.set_ylabel("mean distance")
        ax_err.set_xticks(x)
        ax_err.set_xticklabels(wrapped_names, fontsize=8)
        ax_err.grid(axis="y", alpha=0.25)

        handles_1, labels_1 = ax_err.get_legend_handles_labels()
        handles_2, labels_2 = ax_err_2.get_legend_handles_labels()
        ax_err.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fitted model coverage against an original point cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Common paths:
  original point cloud:
    {DEFAULT_DATA}

  current NURBSFit uv-trimmed result:
    {NURBSFIT_CURRENT}

  NURBSFit before_paper_params uv-trimmed result:
    {NURBSFIT_BEFORE}

  Fiting run_10 result directory:
    {FITING_RUN_10}

Examples:
  python tools/compare_fit_coverage.py --quick

  python tools/compare_fit_coverage.py \\
    --data {DEFAULT_DATA} \\
    --case nurbsfit={NURBSFIT_CURRENT} \\
    --case fiting_run10={FITING_RUN_10}/final_merged_mesh_uv_trimmed.ply
""",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="Original point cloud path.")
    parser.add_argument("--threshold", type=float, default=0.025, help="Coverage/overrun distance threshold.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="A comparison target. Use name=path or name=path1,path2. Path can be file, directory, or glob.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--pdf", type=Path, default=None, help="Optional PDF report output path.")
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Include files whose names contain 'combined'. By default they are skipped to avoid double counting.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compare the saved NURBSFit current, NURBSFit before_paper_params, and Fiting run_10 outputs.",
    )
    args = parser.parse_args()

    cases = list(args.case)
    if args.quick:
        cases.extend(
            [
                f"nurbsfit_current={NURBSFIT_CURRENT}",
                f"nurbsfit_before_paper={NURBSFIT_BEFORE}",
                f"fiting_run10_uv={FITING_RUN_10}/final_merged_mesh_uv_trimmed.ply",
                f"fiting_run10_trimmed={FITING_RUN_10}/final_merged_mesh_trimmed.ply",
            ]
        )

    if not cases:
        parser.error("provide at least one --case, or use --quick")

    data_points = _load_vertices(Path(args.data))
    rows: list[dict[str, float | str]] = []
    for case in cases:
        name, inputs = parse_case(case)
        model_points, files = collect_vertices(inputs, include_combined=args.include_combined)
        metrics = compute_metrics(data_points, model_points, args.threshold)
        print_metrics(name, files, metrics)
        rows.append(
            {
                "name": name,
                "files": len(files),
                **metrics,
            }
        )

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"\nCSV saved to: {args.csv}")

    if args.pdf is not None:
        write_pdf(args.pdf, rows)
        print(f"PDF saved to: {args.pdf}")


if __name__ == "__main__":
    main()
