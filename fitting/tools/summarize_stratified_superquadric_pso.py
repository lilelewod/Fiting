"""Summarize repeated Guided-PSO fits and compare case medians with EMS.

The statistical unit for the paired PSO--EMS comparison is an independently
generated shape case.  PSO repeats are first reduced to one median per case;
the deterministic EMS value is then paired with that median.  This avoids
treating optimizer repeats on the same point cloud as independent cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import permutations, product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.exact_statistics import exact_wilcoxon_signed_rank


HARD_CASES = ("case_002", "case_004", "case_007")


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def rotation_angle_degrees(matrix):
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def xy_symmetries():
    """Return proper signed permutations that keep the z axis in its class."""
    matrices = []
    for xy_order in ((0, 1), (1, 0)):
        order = (*xy_order, 2)
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for new_axis, old_axis in enumerate(order):
                matrix[old_axis, new_axis] = signs[new_axis]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


XY_SYMMETRIES = xy_symmetries()


def signed_permutation_frames():
    frames = []
    for order in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for new_axis, old_axis in enumerate(order):
                matrix[old_axis, new_axis] = signs[new_axis]
            if np.linalg.det(matrix) > 0.5:
                frames.append((order, matrix))
    return frames


SIGNED_PERMUTATION_FRAMES = signed_permutation_frames()


def symmetry_aware_errors(ground_truth, fitted):
    gt_center = np.asarray(ground_truth["center"], dtype=np.float64)
    fit_center = np.asarray(fitted["center"], dtype=np.float64)
    gt_scale = np.asarray(ground_truth["scale"], dtype=np.float64)
    fit_scale = np.asarray(fitted["scale"], dtype=np.float64)
    gt_shape = np.asarray(ground_truth["shape"], dtype=np.float64)
    fit_shape = np.asarray(fitted["shape"], dtype=np.float64)
    gt_rotation = np.asarray(
        ground_truth.get("rotation_matrix", ground_truth.get("rot_matrix")),
        dtype=np.float64,
    )
    fit_rotation = np.asarray(
        fitted.get("rotation_matrix", fitted.get("rot_matrix")), dtype=np.float64
    )

    candidates = []
    for symmetry in XY_SYMMETRIES:
        equivalent_rotation = gt_rotation @ symmetry
        angle = rotation_angle_degrees(equivalent_rotation.T @ fit_rotation)
        equivalent_scale = gt_scale @ np.abs(symmetry)
        scale_relative_mae = np.mean(
            np.abs(fit_scale - equivalent_scale) / np.maximum(equivalent_scale, 1e-12)
        )
        candidates.append((angle, scale_relative_mae))
    rotation_error, scale_error = min(candidates, key=lambda item: item[0])
    frame_candidates = []
    for order, symmetry in SIGNED_PERMUTATION_FRAMES:
        equivalent_rotation = gt_rotation @ symmetry
        angle = rotation_angle_degrees(equivalent_rotation.T @ fit_rotation)
        equivalent_scale = gt_scale @ np.abs(symmetry)
        scale_relative_mae = np.mean(
            np.abs(fit_scale - equivalent_scale) / np.maximum(equivalent_scale, 1e-12)
        )
        frame_candidates.append((angle, scale_relative_mae, order))
    frame_error, frame_scale_error, best_order = min(
        frame_candidates, key=lambda item: item[0]
    )
    diagonal = 2.0 * float(np.linalg.norm(gt_scale))
    return {
        "center_error": float(np.linalg.norm(fit_center - gt_center)),
        "center_error_normalized": float(
            np.linalg.norm(fit_center - gt_center) / diagonal
        ),
        "rotation_error_deg_xy_symmetry": rotation_error,
        "scale_relative_mae_xy_symmetry": scale_error,
        "axis_frame_error_deg_any_permutation": frame_error,
        "scale_relative_mae_at_best_frame": frame_scale_error,
        "best_axis_permutation": "".join(str(axis) for axis in best_order),
        "z_role_preserved": int(best_order[2] == 2),
        "shape_mae": float(np.mean(np.abs(fit_shape - gt_shape))),
        "shape_azimuthal_error": float(abs(fit_shape[0] - gt_shape[0])),
        "shape_meridional_error": float(abs(fit_shape[1] - gt_shape[1])),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_ems_clean(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {
            row["case"]: row
            for row in csv.DictReader(stream)
            if row["condition"] == "clean"
        }


def latex_table(case_rows):
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Case & Stratum & PSO median & IQR & Success & EMS \\",
        r"\midrule",
    ]
    for row in case_rows:
        stratum = f"{row['shape_stratum']}/{row['aspect_stratum']}"
        iqr = f"[{row['pso_chamfer_q1']:.6f}, {row['pso_chamfer_q3']:.6f}]"
        success = f"{row['pso_success_count']}/{row['pso_runs']}"
        lines.append(
            f"{row['case']} & {stratum} & {row['pso_chamfer_median']:.6f} & "
            f"{iqr} & {success} & {row['ems_chamfer']:.6f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pso-root", type=Path, required=True)
    parser.add_argument("--ems-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=9)
    parser.add_argument("--success-chamfer", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    strata = {
        record["case"]: record["strata"] for record in manifest["case_records"]
    }
    ems = load_ems_clean(args.ems_rows)

    run_rows = []
    case_rows = []
    for case_index in range(args.cases):
        case = f"case_{case_index:03d}"
        ground_truth_document = json.loads(
            (args.data_root / case / "trait.json").read_text(encoding="utf-8")
        )
        ground_truth = ground_truth_document.get("trait", ground_truth_document)
        fits = json.loads((args.pso_root / case / "results.json").read_text(encoding="utf-8"))
        for fit in fits:
            errors = symmetry_aware_errors(ground_truth, fit["trait"])
            gt_scale = np.asarray(ground_truth["scale"], dtype=np.float64)
            fit_scale = np.asarray(fit["trait"]["scale"], dtype=np.float64)
            gt_shape = np.asarray(ground_truth["shape"], dtype=np.float64)
            fit_shape = np.asarray(fit["trait"]["shape"], dtype=np.float64)
            run_rows.append(
                {
                    "case": case,
                    "shape_stratum": strata[case]["shape"],
                    "aspect_stratum": strata[case]["aspect"],
                    "repeat": fit["repeat"],
                    "seed": fit["seed"],
                    "gt_chamfer": fit["gt_chamfer"],
                    "gt_d2m": fit["gt_d2m"],
                    "gt_m2d": fit["gt_m2d"],
                    "gt_fscore": fit["gt_fscore"],
                    "success": int(fit["gt_chamfer"] <= args.success_chamfer),
                    "wall_time_s": fit["wall_time_s"],
                    "gt_scale_x": gt_scale[0],
                    "gt_scale_y": gt_scale[1],
                    "gt_scale_z": gt_scale[2],
                    "fit_scale_x": fit_scale[0],
                    "fit_scale_y": fit_scale[1],
                    "fit_scale_z": fit_scale[2],
                    "gt_shape_azimuthal": gt_shape[0],
                    "gt_shape_meridional": gt_shape[1],
                    "fit_shape_azimuthal": fit_shape[0],
                    "fit_shape_meridional": fit_shape[1],
                    **errors,
                }
            )
        rows = [row for row in run_rows if row["case"] == case]
        chamfer = [row["gt_chamfer"] for row in rows]
        runtime = [row["wall_time_s"] for row in rows]
        errors = {
            name: descriptive([row[name] for row in rows])["median"]
            for name in (
                "center_error_normalized",
                "rotation_error_deg_xy_symmetry",
                "scale_relative_mae_xy_symmetry",
                "shape_mae",
                "axis_frame_error_deg_any_permutation",
                "scale_relative_mae_at_best_frame",
            )
        }
        axis_permutations = Counter(row["best_axis_permutation"] for row in rows)
        ems_row = ems[case]
        pso_median = float(np.median(chamfer))
        case_rows.append(
            {
                "case": case,
                "shape_stratum": strata[case]["shape"],
                "aspect_stratum": strata[case]["aspect"],
                "pso_runs": len(rows),
                "pso_success_count": sum(row["success"] for row in rows),
                "pso_success_rate": float(np.mean([row["success"] for row in rows])),
                "pso_chamfer_median": pso_median,
                "pso_chamfer_q1": float(np.percentile(chamfer, 25)),
                "pso_chamfer_q3": float(np.percentile(chamfer, 75)),
                "pso_runtime_median_s": float(np.median(runtime)),
                "dominant_axis_permutation": axis_permutations.most_common(1)[0][0],
                "z_role_preserved_rate": float(np.mean([row["z_role_preserved"] for row in rows])),
                "gt_shape_azimuthal": rows[0]["gt_shape_azimuthal"],
                "gt_shape_meridional": rows[0]["gt_shape_meridional"],
                "median_fit_shape_azimuthal": float(
                    np.median([row["fit_shape_azimuthal"] for row in rows])
                ),
                "median_fit_shape_meridional": float(
                    np.median([row["fit_shape_meridional"] for row in rows])
                ),
                "ems_chamfer": float(ems_row["gt_chamfer"]),
                "ems_success": int(ems_row["success"]),
                "ems_runtime_s": float(ems_row["wall_time_s"]),
                "pso_minus_ems": pso_median - float(ems_row["gt_chamfer"]),
                "pso_wins": int(pso_median < float(ems_row["gt_chamfer"])),
                **{f"median_{key}": value for key, value in errors.items()},
            }
        )

    pso_case_values = np.asarray(
        [row["pso_chamfer_median"] for row in case_rows], dtype=np.float64
    )
    ems_case_values = np.asarray(
        [row["ems_chamfer"] for row in case_rows], dtype=np.float64
    )
    differences = pso_case_values - ems_case_values
    paired = exact_wilcoxon_signed_rank(differences)
    grouped_case_medians = {"shape_stratum": {}, "aspect_stratum": {}}
    for group_field in grouped_case_medians:
        for group_name in sorted({row[group_field] for row in case_rows}):
            group = [row for row in case_rows if row[group_field] == group_name]
            grouped_case_medians[group_field][group_name] = {
                "independent_cases": len(group),
                "pso_chamfer_medians": descriptive(
                    [row["pso_chamfer_median"] for row in group]
                ),
                "ems_chamfer": descriptive([row["ems_chamfer"] for row in group]),
                "pso_run_success_rate": float(
                    sum(row["pso_success_count"] for row in group)
                    / sum(row["pso_runs"] for row in group)
                ),
                "pso_wins": int(sum(row["pso_wins"] for row in group)),
            }
    report = {
        "protocol": {
            "pso_runs_per_case": 5,
            "independent_cases": args.cases,
            "success_chamfer": args.success_chamfer,
            "paired_statistical_unit": "per-case PSO median versus deterministic EMS",
            "rotation_error_note": (
                "minimum frame error over proper x/y signed-permutation symmetries"
            ),
        },
        "pso_all_runs": {
            "chamfer": descriptive([row["gt_chamfer"] for row in run_rows]),
            "runtime_s": descriptive([row["wall_time_s"] for row in run_rows]),
            "success_count": int(sum(row["success"] for row in run_rows)),
            "success_rate": float(np.mean([row["success"] for row in run_rows])),
        },
        "pso_case_medians": descriptive(pso_case_values),
        "ems_cases": {
            "chamfer": descriptive(ems_case_values),
            "success_count": int(sum(row["ems_success"] for row in case_rows)),
            "runtime_median_s": float(np.median([row["ems_runtime_s"] for row in case_rows])),
        },
        "paired_comparison": {
            "pso_wins": int(sum(row["pso_wins"] for row in case_rows)),
            "ems_wins": int(sum(not row["pso_wins"] for row in case_rows)),
            "median_pso_minus_ems": float(np.median(differences)),
            "wilcoxon_statistic": paired["statistic"],
            "wilcoxon_exact_two_sided_p": paired["exact_two_sided_p"],
            "wilcoxon_nonzero_pairs": paired["nonzero_pairs"],
            "wilcoxon_zero_pairs": paired["zero_pairs"],
        },
        "case_median_groups": grouped_case_medians,
        "hard_case_diagnostics": {
            row["case"]: {
                key: value
                for key, value in row.items()
                if key.startswith("median_")
                or key
                in {
                    "pso_success_count",
                    "pso_chamfer_median",
                    "pso_chamfer_q1",
                    "pso_chamfer_q3",
                    "ems_chamfer",
                }
            }
            for row in case_rows
            if row["case"] in HARD_CASES
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "pso_runs.csv", run_rows)
    write_csv(args.output_root / "per_case.csv", case_rows)
    write_csv(
        args.output_root / "hard_case_runs.csv",
        [row for row in run_rows if row["case"] in HARD_CASES],
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_root / "per_case_table.tex").write_text(
        latex_table(case_rows), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
