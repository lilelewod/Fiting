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
FITING_RUN_10 = "/home/m25lll/code/Fiting/fitting/outputs/cco/3d/nurbs_surface/nurbs/00873042_lessp/run_10/2026-0428/1513-28"

# Edit these paths directly if you prefer running the script without CLI args.
CONFIG_DATA = DEFAULT_DATA
CONFIG_CASE_NAME = "result"
CONFIG_CASE_PATHS = [
    f"{FITING_RUN_10}/final_merged_mesh_uv_trimmed.ply",
]
CONFIG_THRESHOLD = 0.025
CONFIG_INCLUDE_COMBINED = False
CONFIG_CSV = None
CONFIG_PDF = Path("/home/m25lll/code/Fiting/fitting/tools/compare_fit_coverage_report.pdf")

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


def run_cases(
    data_path: str,
    threshold: float,
    case_specs: list[str],
    include_combined: bool,
    csv_path: Path | None,
    pdf_path: Path | None,
) -> None:
    data_points = _load_vertices(Path(data_path))
    rows: list[dict[str, float | str]] = []
    for case in case_specs:
        name, inputs = parse_case(case)
        model_points, files = collect_vertices(inputs, include_combined=include_combined)
        metrics = compute_metrics(data_points, model_points, threshold)
        print_metrics(name, files, metrics)
        rows.append(
            {
                "name": name,
                "files": len(files),
                **metrics,
            }
        )

    if csv_path is not None:
        write_csv(csv_path, rows)
        print(f"\nCSV saved to: {csv_path}")

    if pdf_path is not None:
        write_pdf(pdf_path, rows)
        print(f"PDF saved to: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fitted model coverage against an original point cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Edit the config at the top of this file and run:
  python tools/compare_fit_coverage.py

Optional CLI example:
  python tools/compare_fit_coverage.py \\
    --data {DEFAULT_DATA} \\
    --case result={FITING_RUN_10}/final_merged_mesh_uv_trimmed.ply
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
    args = parser.parse_args()

    if len(__import__("sys").argv) == 1:
        case_paths = ",".join(CONFIG_CASE_PATHS)
        run_cases(
            data_path=CONFIG_DATA,
            threshold=CONFIG_THRESHOLD,
            case_specs=[f"{CONFIG_CASE_NAME}={case_paths}"],
            include_combined=CONFIG_INCLUDE_COMBINED,
            csv_path=Path(CONFIG_CSV) if CONFIG_CSV else None,
            pdf_path=Path(CONFIG_PDF) if CONFIG_PDF else None,
        )
        return

    cases = list(args.case)
    if not cases:
        parser.error("provide at least one --case, or edit the config at the top of this file and run without args")
    run_cases(
        data_path=args.data,
        threshold=args.threshold,
        case_specs=cases,
        include_combined=args.include_combined,
        csv_path=args.csv,
        pdf_path=args.pdf,
    )


if __name__ == "__main__":
    main()
