"""Generate reproducible figures for the current IEEE manuscript."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plyfile import PlyData


PAPER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_ROOT.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.superquadric_evaluation import sample_trait, trait_from_mapping


FIGURE_ROOT = PAPER_ROOT / "figures"
DATA_ROOT = Path(r"C:\code\superquadic_data\v3_randomized")
SQ_RESULT_ROOT = (
    WORKSPACE_ROOT
    / "outputs/optimizer_comparison/v3_stratified9_clean_guided_pso_1seed_20260716"
)
PMF_RESULT = (
    WORKSPACE_ROOT
    / "outputs/pmf_cylinder_comparison/pso_cs_formal20_20260722/results.json"
)
AREA_ROOT = (
    WORKSPACE_ROOT
    / "outputs/area_weight_ablation/formal_v2_pso_clean_48x48_5008fe_5seeds"
)
GUIDED_AUDIT = (
    WORKSPACE_ROOT
    / "outputs/optimizer_comparison/guided_initialization_ablation_summary/audit.json"
)
SUPPORT_SUMMARY = (
    WORKSPACE_ROOT
    / "outputs/pmf_cylinder_density_support/formal_adaptive_20260721/ablation_summary.json"
)
ROBUSTNESS_SUMMARY = (
    WORKSPACE_ROOT
    / "outputs/optimizer_comparison/v3_randomized30_guided_pso_3seeds_20260727/summary_30cases_3seeds/summary.json"
)
BUDGET_SUMMARY = (
    WORKSPACE_ROOT
    / "outputs/pmf_cylinder_budget_sensitivity/preregistered_20260721/summary.json"
)
PMF_M1_CURVES = WORKSPACE_ROOT / "outputs/pmf_m1_partial_similarity/curves.csv"


COLORS = {
    "blue": "#3B6FB6",
    "orange": "#D97925",
    "green": "#3D8D7A",
    "red": "#C64B4B",
    "gray": "#6B7280",
    "light": "#EEF2F5",
    "dark": "#263238",
}


def configure():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save(fig, name):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(
            FIGURE_ROOT / f"{name}{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=300,
        )
    plt.close(fig)


def read_json_retry(path, attempts=5):
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.2)


def read_ply(path):
    vertex = PlyData.read(str(path))["vertex"]
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)


def equal_3d(ax, points, radius=None):
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    if radius is None:
        radius = 0.55 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-56)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.37, 0.14, 0.32, "Observed\npoint cloud", "clean / corrupted", COLORS["light"]),
        (0.205, 0.37, 0.15, 0.32, "Robust\nsupport", "density filter + PCA", "#E7F2EE"),
        (0.405, 0.37, 0.14, 0.32, "Hypothesis\nbank", "axis roles + anchors", "#FFF1E3"),
        (0.595, 0.37, 0.14, 0.32, "PSO\nsearch", "paired FE budget", "#E8EFF9"),
        (0.785, 0.37, 0.19, 0.32, "Procedural\nmodel", "pose + scale + shape", "#F8E8E8"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9, edgecolor=COLORS["dark"], facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.215, title, ha="center", va="center",
                weight="bold", fontsize=7.8, linespacing=0.95)
        ax.text(x + w / 2, y + 0.075, subtitle, ha="center", va="center", fontsize=7.0, color="#4B5563")
    for left, right in zip(boxes[:-1], boxes[1:]):
        start = (left[0] + left[2] + 0.006, 0.53)
        end = (right[0] - 0.006, 0.53)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=COLORS["dark"]))
    ax.add_patch(
        FancyArrowPatch((0.88, 0.35), (0.67, 0.35), connectionstyle="arc3,rad=-0.35",
                        arrowstyle="-|>", mutation_scale=10, lw=1.1, color=COLORS["red"])
    )
    ax.text(0.77, 0.105, "area-weighted model-to-data score", ha="center", color=COLORS["red"], fontsize=7.5)
    ax.plot([0.88, 0.88], [0.70, 0.82], color=COLORS["gray"], lw=0.9)
    ax.add_patch(FancyArrowPatch((0.88, 0.82), (0.97, 0.82), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=COLORS["gray"]))
    ax.text(0.86, 0.82, "independent area-uniform evaluation", ha="right", va="center", fontsize=7.3, color=COLORS["gray"])
    ax.text(0.5, 0.98, "Unified derivative-free parametric fitting pipeline", ha="center", va="top", weight="bold", fontsize=9)
    save(fig, "pipeline")


def stratified_shapes_figure():
    fig = plt.figure(figsize=(7.16, 5.0))
    shape_names = ("Smooth", "Mixed", "Boxy")
    aspect_names = ("Balanced", "Anisotropic", "Extreme")
    rng = np.random.default_rng(17)
    for row in range(3):
        for col in range(3):
            case = row * 3 + col
            points = read_ply(DATA_ROOT / f"case_{case:03d}" / "reference_uniform.ply")
            ids = rng.choice(len(points), 1700, replace=False)
            shown = points[ids]
            ax = fig.add_subplot(3, 3, case + 1, projection="3d")
            ax.scatter(*shown.T, s=0.75, color=COLORS["blue"], alpha=0.58, linewidths=0, rasterized=True)
            equal_3d(ax, points, radius=1.5)
            if row == 0:
                ax.set_title(shape_names[col], pad=0, weight="bold")
            if col == 0:
                ax.text2D(-0.08, 0.50, aspect_names[row], transform=ax.transAxes,
                          rotation=90, va="center", ha="center", weight="bold", fontsize=9)
            ax.text2D(0.50, 0.01, f"case {case:03d}", transform=ax.transAxes,
                      ha="center", fontsize=7.5, color="#374151")
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.015, top=0.96, wspace=-0.05, hspace=0.01)
    save(fig, "stratified_superquadrics")


def superquadric_results_figure():
    rows = read_json_retry(SQ_RESULT_ROOT / "summary_5seeds/summary.json")
    del rows
    import csv
    with (SQ_RESULT_ROOT / "summary_5seeds/per_case.csv").open(encoding="utf-8-sig", newline="") as stream:
        case_rows = list(csv.DictReader(stream))
    x = np.arange(9)
    pso = np.array([float(row["pso_chamfer_median"]) for row in case_rows])
    ems = np.array([float(row["ems_chamfer"]) for row in case_rows])
    success = np.array([int(row["pso_success_count"]) for row in case_rows]).reshape(3, 3)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    ax.plot(x, pso, "o-", color=COLORS["orange"], lw=1.4, ms=4, label="Guided PSO median")
    ax.plot(x, ems, "s-", color=COLORS["green"], lw=1.4, ms=3.7, label="EMS")
    ax.axhline(0.05, color=COLORS["red"], ls="--", lw=0.9, label="success threshold")
    ax.set_xticks(x, [f"{i:03d}" for i in x])
    ax.set_xlabel("Stratified randomized case")
    ax.set_ylabel("Chamfer distance")
    ax.set_ylim(0.015, 0.062)
    ax.grid(axis="y", color="#D1D5DB", lw=0.5)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.set_title("(a) Per-case recovery accuracy", loc="left")
    ax = axes[1]
    image = ax.imshow(success, vmin=0, vmax=5, cmap=mpl.colors.LinearSegmentedColormap.from_list("success", ["#F7D8D8", "#FFF1CC", "#CFE8DD"]))
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{success[r,c]}/5", ha="center", va="center", weight="bold", color=COLORS["dark"])
    ax.set_xticks(range(3), ["Smooth", "Mixed", "Boxy"])
    ax.set_yticks(range(3), ["Balanced", "Anisotropic", "Extreme"])
    ax.set_title("(b) Guided-PSO successes", loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.90, wspace=0.28)
    save(fig, "superquadric_results")


def fit_examples_figure():
    selected = (0, 2, 4, 7)
    fig = plt.figure(figsize=(7.16, 2.15))
    rng = np.random.default_rng(31)
    for index, case in enumerate(selected):
        fits = read_json_retry(SQ_RESULT_ROOT / f"case_{case:03d}/results.json")
        fits = sorted(fits, key=lambda row: row["gt_chamfer"])
        fit = fits[len(fits) // 2]
        truth = read_ply(DATA_ROOT / f"case_{case:03d}/reference_uniform.ply")
        predicted = sample_trait(trait_from_mapping(fit["trait"]), 5000, seed=991 + case, grid_resolution=256)
        center = truth.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov((truth - center).T))
        axes = axes[:, ::-1]
        truth_view = (truth - center) @ axes
        predicted_view = (predicted - center) @ axes
        ax = fig.add_subplot(1, 4, index + 1, projection="3d")
        gt_ids = rng.choice(len(truth_view), 1400, replace=False)
        pred_ids = rng.choice(len(predicted_view), 1100, replace=False)
        ax.scatter(*truth_view[gt_ids].T, s=0.5, color="#9CA3AF", alpha=0.32, linewidths=0, rasterized=True)
        ax.scatter(*predicted_view[pred_ids].T, s=0.65, color=COLORS["red"], alpha=0.47, linewidths=0, rasterized=True)
        equal_3d(ax, np.vstack((truth_view, predicted_view)))
        status = "success" if fit["success"] else "failure"
        ax.set_title(f"case {case:03d}: {status}\nCD={fit['gt_chamfer']:.4f}", pad=-4)
    fig.text(0.5, 0.01, "Gray: analytic reference; red: fitted superquadric (median run)", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.08, top=0.92, wspace=-0.12)
    save(fig, "fit_examples")


def pmf_clean_figure():
    rows = read_json_retry(PMF_RESULT)
    pso = {row["seed"]: row for row in rows if row["condition"] == "clean" and row["algorithm"] == "pso"}
    cs = {row["seed"]: row for row in rows if row["condition"] == "clean" and row["algorithm"] == "cs"}
    seeds = sorted(set(pso) & set(cs))
    if len(seeds) != 20:
        raise RuntimeError(f"Expected 20 completed clean pairs, found {len(seeds)}")
    pso_y = np.array([pso[seed]["gt_chamfer"] for seed in seeds])
    cs_y = np.array([cs[seed]["gt_chamfer"] for seed in seeds])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for left, right in zip(pso_y, cs_y):
        ax.plot([0, 1], [left, right], color="#B8BEC6", lw=0.65, alpha=0.75, zorder=1)
    rng = np.random.default_rng(5)
    ax.scatter(rng.normal(0, 0.018, len(pso_y)), pso_y, color=COLORS["blue"], s=16, zorder=3, label="paired run")
    ax.scatter(rng.normal(1, 0.018, len(cs_y)), cs_y, color=COLORS["orange"], s=16, zorder=3)
    ax.scatter([0, 1], [np.median(pso_y), np.median(cs_y)], marker="D", s=34, color=COLORS["dark"], zorder=4, label="median")
    ax.axhline(0.29613998436222805, color=COLORS["red"], ls="--", lw=0.9, label="success threshold")
    ax.set_xticks([0, 1], ["PSO\n18/20", "CS\n6/20"])
    ax.set_ylabel("Clean-reference Chamfer")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.10, 2.45)
    ax.grid(axis="y", color="#D1D5DB", lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("PMF-style partial cylinder, clean input")
    fig.tight_layout(pad=0.4)
    save(fig, "pmf_clean_paired")


def pmf_convergence_figure():
    rows = read_json_retry(PMF_RESULT)
    grid = np.linspace(0, 50000, 251)
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    for algorithm, color, line_style, label in (
        ("pso", COLORS["blue"], "-", "PSO"),
        ("cs", COLORS["orange"], "--", "CS"),
    ):
        curves = []
        selected = [
            row for row in rows
            if row["condition"] == "clean" and row["algorithm"] == algorithm
        ]
        if len(selected) != 20:
            raise RuntimeError(f"Expected 20 clean {algorithm} traces, found {len(selected)}")
        for row in selected:
            record = read_json_retry(Path(row["record_file"]))
            episodes = np.asarray(record["evolving_episodes"], dtype=float)
            scores = np.asarray(record["evolving_scores"], dtype=float)
            order = np.argsort(episodes)
            episodes, scores = episodes[order], scores[order]
            running = np.maximum.accumulate(scores)
            indices = np.searchsorted(episodes, grid, side="right") - 1
            indices = np.clip(indices, 0, len(running) - 1)
            curves.append(running[indices])
        curves = np.asarray(curves)
        median = np.median(curves, axis=0)
        q1 = np.percentile(curves, 25, axis=0)
        q3 = np.percentile(curves, 75, axis=0)
        ax.plot(grid, median, color=color, ls=line_style, lw=1.5, label=label)
        ax.fill_between(grid, q1, q3, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best PMF objective")
    ax.set_xlim(0, 50000)
    ax.grid(color="#D1D5DB", lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Clean partial-cylinder convergence")
    fig.tight_layout(pad=0.4)
    save(fig, "pmf_clean_convergence")


def area_ablation_figure():
    shapes = ("box", "ellipsoid", "cylinder")
    uniform, weighted, success_u, success_w = [], [], [], []
    for shape in shapes:
        summary = read_json_retry(AREA_ROOT / shape / "summary.json")
        by_variant = {row["variant"]: row for row in summary["variants"]}
        uniform.append(by_variant["uniform"]["gt_chamfer_median"])
        weighted.append(by_variant["area_weighted"]["gt_chamfer_median"])
        success_u.append(round(5 * by_variant["uniform"]["success_mean"]))
        success_w.append(round(5 * by_variant["area_weighted"]["success_mean"]))
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    ax.set_axisbelow(True)
    left = ax.bar(x - width / 2, uniform, width, color="#A8B2BD", hatch="//",
                  edgecolor="#68717A", linewidth=0.45, label="Uniform angular mean")
    right = ax.bar(x + width / 2, weighted, width, color=COLORS["green"], label="Area weighted")
    for bars, successes in ((left, success_u), (right, success_w)):
        for bar, count in zip(bars, successes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{count}/5", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [name.capitalize() for name in shapes])
    ax.set_ylabel("Median Chamfer")
    ax.set_ylim(0, 0.165)
    ax.grid(axis="y", color="#D1D5DB", lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc="upper center", ncol=1)
    ax.set_title("Area-weighting ablation (labels: successes)")
    fig.tight_layout(pad=0.4)
    save(fig, "area_ablation")


def guided_initialization_figure():
    audit = read_json_retry(GUIDED_AUDIT)
    shapes = ("box", "cylinder", "ellipsoid")
    random_cd = [audit["shapes"][shape]["random_chamfer"]["median"] for shape in shapes]
    guided_cd = [audit["shapes"][shape]["guided_chamfer"]["median"] for shape in shapes]
    random_success = [audit["shapes"][shape]["random_successes"] for shape in shapes]
    guided_success = [audit["shapes"][shape]["guided_successes"] for shape in shapes]
    x = np.arange(len(shapes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    ax.set_axisbelow(True)
    left = ax.bar(x - width / 2, random_cd, width, color="#A8B2BD", hatch="//",
                  edgecolor="#68717A", linewidth=0.45, label="Random PSO")
    right = ax.bar(x + width / 2, guided_cd, width, color=COLORS["blue"], label="Guided PSO")
    for bars, successes in ((left, random_success), (right, guided_success)):
        for bar, count in zip(bars, successes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{count}/5", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [name.capitalize() for name in shapes])
    ax.set_ylabel("Median Chamfer")
    ax.set_ylim(0, 0.095)
    ax.grid(axis="y", color="#D1D5DB", lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Initialization ablation (labels: successes)")
    fig.tight_layout(pad=0.4)
    save(fig, "guided_initialization_ablation")


def density_support_ablation_figure():
    summary = read_json_retry(SUPPORT_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("Density-support ablation has not passed strict summarization")
    conditions = ("clean", "outlier_50", "outlier_80")
    labels = ("Clean", "50% outliers", "80% outliers")
    full_cd, adaptive_cd, full_success, adaptive_success = [], [], [], []
    for condition in conditions:
        cell = summary["conditions"][condition]["variants"]
        full_cd.append(cell["full_input"]["paired_chamfer"]["median"])
        adaptive_cd.append(cell["adaptive_density"]["paired_chamfer"]["median"])
        full_success.append(cell["full_input"]["paired_successes"])
        adaptive_success.append(cell["adaptive_density"]["paired_successes"])
    x = np.arange(len(conditions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    ax.set_axisbelow(True)
    left = ax.bar(x - width / 2, full_cd, width, color="#A8B2BD", hatch="//",
                  edgecolor="#68717A", linewidth=0.45, label="Full input")
    right = ax.bar(x + width / 2, adaptive_cd, width, color=COLORS["green"], label="Adaptive support")
    for bars, successes in ((left, full_success), (right, adaptive_success)):
        for bar, count in zip(bars, successes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                    f"{count}/5", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median clean-reference Chamfer")
    ax.set_yscale("log")
    ax.set_ylim(0.11, 7.0)
    ax.grid(axis="y", which="both", color="#D1D5DB", lw=0.4, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Density-support ablation (labels: successes)")
    fig.tight_layout(pad=0.4)
    save(fig, "density_support_ablation")


def robustness_conditions_figure():
    case_root = DATA_ROOT / "case_004"
    filenames = (
        "clean.ply",
        "noise_1pct_diag.ply",
        "outlier_20.ply",
        "missing_80.ply",
        "occlusion_cap_80.ply",
    )
    titles = ("Clean", "1% noise", "20% outliers", "80% random missing", "80% spatial occlusion")
    reference = read_ply(case_root / "reference_uniform.ply")
    center = reference.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((reference - center).T))
    axes = axes[:, ::-1]
    reference_view = (reference - center) @ axes
    radius = 0.58 * float(np.max(np.ptp(reference_view, axis=0)))
    fig = plt.figure(figsize=(7.16, 1.75))
    rng = np.random.default_rng(43)
    for index, (filename, title) in enumerate(zip(filenames, titles)):
        points = (read_ply(case_root / filename) - center) @ axes
        count = min(1300, len(points))
        shown = points[rng.choice(len(points), count, replace=False)]
        ax = fig.add_subplot(1, 5, index + 1, projection="3d")
        ax.scatter(*shown.T, s=0.75, color=COLORS["blue"], alpha=0.58,
                   linewidths=0, rasterized=True)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-56)
        ax.set_title(title, pad=-3, fontsize=9.5, weight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.92, wspace=-0.08)
    save(fig, "robustness_conditions")


def superquadric_robustness_figure():
    if not ROBUSTNESS_SUMMARY.exists():
        return False
    summary = read_json_retry(ROBUSTNESS_SUMMARY)
    conditions = (
        "clean",
        "noise_1pct_diag",
        "outlier_20",
        "missing_80",
        "occlusion_cap_80",
    )
    labels = ("Clean", "1% noise", "20% outliers", "80% random\nmissing", "80% spatial\nocclusion")
    cells = [summary["conditions"][condition] for condition in conditions]
    if any(int(cell["guided_pso_runs"]["runs"]) <= 0 for cell in cells):
        return False
    pso_cd = [cell["guided_pso_case_medians"]["median"] for cell in cells]
    ems_cd = [cell["ems_cases"]["chamfer"]["median"] for cell in cells]
    pso_denominators = [int(cell["guided_pso_runs"]["runs"]) for cell in cells]
    ems_denominators = [int(cell["ems_cases"]["cases"]) for cell in cells]
    pso_success = [
        cell["guided_pso_runs"]["successes"] / denominator
        for cell, denominator in zip(cells, pso_denominators)
    ]
    ems_success = [
        cell["ems_cases"]["successes"] / denominator
        for cell, denominator in zip(cells, ems_denominators)
    ]
    pso_counts = [cell["guided_pso_runs"]["successes"] for cell in cells]
    ems_counts = [cell["ems_cases"]["successes"] for cell in cells]
    x = np.arange(len(conditions))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ax = axes[0]
    ax.plot(x, pso_cd, "o-", color=COLORS["orange"], lw=1.4, ms=4,
            label="Guided PSO case median")
    ax.plot(x, ems_cd, "s-", color=COLORS["green"], lw=1.4, ms=3.7,
            label="EMS")
    ax.axhline(0.05, color=COLORS["red"], ls="--", lw=0.9,
               label="success threshold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Chamfer distance")
    ax.grid(axis="y", color="#D1D5DB", lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Accuracy across corruption types", loc="left")
    ax = axes[1]
    left = ax.bar(x - width / 2, pso_success, width, color=COLORS["orange"],
                  label="Guided PSO (runs)")
    right = ax.bar(x + width / 2, ems_success, width, color=COLORS["green"],
                   label="EMS (cases)")
    for bars, counts, denominators in (
        (left, pso_counts, pso_denominators),
        (right, ems_counts, ems_denominators),
    ):
        for bar, count, denominator in zip(bars, counts, denominators):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    f"{count}/{denominator}", ha="center", va="bottom", fontsize=8.5,
                    rotation=90 if len(str(count)) > 1 else 0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", color="#D1D5DB", lw=0.5)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("(b) Thresholded recovery", loc="left")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.90, wspace=0.25)
    save(fig, "superquadric_robustness")
    return True


def superquadric_strata_figure():
    if not ROBUSTNESS_SUMMARY.exists():
        return False
    summary = read_json_retry(ROBUSTNESS_SUMMARY)
    conditions = (
        "clean",
        "noise_1pct_diag",
        "outlier_20",
        "missing_80",
        "occlusion_cap_80",
    )
    condition_labels = ("Clean", "Noise", "Outliers", "Random miss.", "Occlusion")
    cells = [summary["conditions"][condition] for condition in conditions]
    if any(int(cell["guided_pso_runs"]["runs"]) <= 0 or "strata" not in cell for cell in cells):
        return False

    axes_spec = (
        ("shape", ("smooth", "mixed", "boxy"), "(a) Shape-exponent stratum"),
        ("aspect", ("balanced", "anisotropic", "extreme"), "(b) Aspect-ratio stratum"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    image = None
    for ax, (axis_name, labels, title) in zip(axes, axes_spec):
        rates = np.asarray(
            [
                [
                    cells[column]["strata"][axis_name][label]["guided_pso_runs"]["successes"]
                    / cells[column]["strata"][axis_name][label]["guided_pso_runs"]["runs"]
                    for column in range(len(conditions))
                ]
                for label in labels
            ],
            dtype=float,
        )
        image = ax.imshow(rates, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        for row, label in enumerate(labels):
            for column in range(len(conditions)):
                record = cells[column]["strata"][axis_name][label]["guided_pso_runs"]
                color = "white" if rates[row, column] > 0.58 else COLORS["dark"]
                ax.text(
                    column,
                    row,
                    f'{record["successes"]}/{record["runs"]}',
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        ax.set_xticks(np.arange(len(conditions)), condition_labels, rotation=24, ha="right")
        ax.set_yticks(np.arange(len(labels)), [label.capitalize() for label in labels])
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
    fig.subplots_adjust(left=0.085, right=0.88, bottom=0.26, top=0.88, wspace=0.30)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.035)
    colorbar.set_label("Guided-PSO success rate")
    save(fig, "superquadric_strata")
    return True


def pmf_budget_sensitivity_figure():
    if not BUDGET_SUMMARY.exists():
        return False
    summary = read_json_retry(BUDGET_SUMMARY)
    budgets = (50000, 199920, 499920)
    if summary.get("status") != "PASS" or any(
        str(budget) not in summary.get("budgets", {}) for budget in budgets
    ):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    for ax, condition, title in zip(
        axes,
        ("clean", "outlier_50"),
        ("(a) Clean input", "(b) 50% unfiltered outliers"),
    ):
        for algorithm, color, marker, line_style, label in (
            ("pso", COLORS["blue"], "o", "-", "PSO"),
            ("cs", COLORS["orange"], "s", "--", "CS"),
        ):
            medians, q1, q3, successes = [], [], [], []
            for budget in budgets:
                cell = summary["budgets"][str(budget)]["conditions"][condition]
                metric = cell[f"{algorithm}_chamfer"]
                medians.append(metric["median"])
                q1.append(metric["q1"])
                q3.append(metric["q3"])
                successes.append(cell[f"{algorithm}_successes"])
            medians = np.asarray(medians)
            ax.plot(budgets, medians, marker=marker, ls=line_style,
                    color=color, lw=1.4, ms=4, label=label)
            ax.fill_between(budgets, q1, q3, color=color, alpha=0.14, linewidth=0)
            for x_value, y_value, count in zip(budgets, medians, successes):
                ax.annotate(
                    f"{count}/5",
                    (x_value, y_value),
                    xytext=(-8, 7) if algorithm == "pso" else (8, -13),
                    textcoords="offset points",
                    ha="right" if algorithm == "pso" else "left",
                    fontsize=8.5,
                    color=color,
                )
        ax.set_xscale("log")
        ax.set_xticks(budgets, ("50k", "200k", "500k"))
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Clean-reference Chamfer")
        ax.grid(axis="y", color="#D1D5DB", lw=0.5)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.90, wspace=0.27)
    save(fig, "pmf_budget_sensitivity")
    return True


def pmf_m1_partial_similarity_figure():
    """Show why weighted mean measure remains identifiable under missing parts."""
    if not PMF_M1_CURVES.exists():
        return False
    rows = np.genfromtxt(
        PMF_M1_CURVES,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), sharey=True)
    retained = {"D1": "100% observed", "D2": "75% observed", "D4": "25% observed"}
    for ax, data_name in zip(axes, ("D1", "D2", "D4")):
        selected = rows[rows["data"] == data_name]
        order = np.argsort(selected["theta"])
        selected = selected[order]
        theta = selected["theta"]
        for metric, color, line_style, label in (
            ("wmm", COLORS["blue"], "-", "WMM"),
            ("mm", COLORS["orange"], "--", "unweighted MM"),
        ):
            values = np.asarray(selected[metric], dtype=float)
            scale = max(float(np.nanmax(values)), np.finfo(float).eps)
            normalized = values / scale
            ax.plot(theta, normalized, color=color, ls=line_style, lw=1.45, label=label)
            peak = int(np.nanargmax(values))
            ax.plot(theta[peak], normalized[peak], "o", color=color, ms=3.2)
        ax.axvline(1.0, color=COLORS["red"], ls="--", lw=0.9, label="target")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Model parameter value")
        ax.set_title(f"{data_name}: {retained[data_name]}")
        ax.grid(color="#D1D5DB", lw=0.45, alpha=0.8)
    axes[0].set_ylabel("Normalized similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.80, wspace=0.18)
    save(fig, "pmf_m1_partial_similarity")
    return True


def main():
    configure()
    architecture_figure()
    stratified_shapes_figure()
    superquadric_results_figure()
    fit_examples_figure()
    pmf_clean_figure()
    pmf_convergence_figure()
    area_ablation_figure()
    guided_initialization_figure()
    density_support_ablation_figure()
    robustness_conditions_figure()
    superquadric_robustness_figure()
    superquadric_strata_figure()
    pmf_budget_sensitivity_figure()
    pmf_m1_partial_similarity_figure()
    print(f"Figures written to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
