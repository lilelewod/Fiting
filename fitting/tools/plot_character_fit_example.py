"""Create a paper-ready character fitting example from one audited record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def configure_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def setup_image_axis(ax, title):
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlim(-1, 52)
    ax.set_ylim(52, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c8cdd3")
        spine.set_linewidth(0.7)


def zhang_suen_thinning(binary):
    """Return a one-pixel centerline without changing the fitted geometry."""
    image = (np.asarray(binary) > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            remove = []
            for row in range(1, image.shape[0] - 1):
                for col in range(1, image.shape[1] - 1):
                    if image[row, col] == 0:
                        continue
                    p2 = image[row - 1, col]
                    p3 = image[row - 1, col + 1]
                    p4 = image[row, col + 1]
                    p5 = image[row + 1, col + 1]
                    p6 = image[row + 1, col]
                    p7 = image[row + 1, col - 1]
                    p8 = image[row, col - 1]
                    p9 = image[row - 1, col - 1]
                    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
                    count = sum(neighbors)
                    transitions = sum(
                        neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                        for index in range(8)
                    )
                    if not (2 <= count <= 6 and transitions == 1):
                        continue
                    if phase == 0:
                        keep = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        keep = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if keep:
                        remove.append((row, col))
            if remove:
                changed = True
                for row, col in remove:
                    image[row, col] = 0
    return image.astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--style", choices=["analysis", "pmf"], default="analysis")
    parser.add_argument(
        "--pmf-line-width",
        type=int,
        default=3,
        help="Displayed centerline width in pixels for PMF style (odd integer).",
    )
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    observed = np.asarray(record["data_cloud"], dtype=float)
    fitted = np.asarray(record["best_cloud"], dtype=float)
    episodes = np.asarray(record["evolving_episodes"], dtype=float)
    scores = np.asarray(record["evolving_scores"], dtype=float)
    order = np.argsort(episodes)
    episodes = episodes[order]
    scores = np.maximum.accumulate(scores[order])
    image = np.asarray(Image.open(args.image).convert("L"))

    configure_style()
    if args.style == "pmf":
        observed_canvas = 1.0 - image.astype(float) / 255.0
        height, width = image.shape
        fitted_canvas = np.zeros((height, width), dtype=float)
        # CharacterRule maps the 105-pixel renderer to the 52-unit fitting
        # coordinates as cloud = pixel / 2 + 0.25; invert that map here.
        fitted_rc = np.rint(2.0 * fitted - 0.5).astype(int)
        valid = (
            (fitted_rc[:, 0] >= 0) & (fitted_rc[:, 0] < height)
            & (fitted_rc[:, 1] >= 0) & (fitted_rc[:, 1] < width)
        )
        fitted_rc = fitted_rc[valid]
        fitted_canvas[fitted_rc[:, 0], fitted_rc[:, 1]] = 1.0
        fitted_centerline = zhang_suen_thinning(fitted_canvas)
        if args.pmf_line_width < 1 or args.pmf_line_width % 2 == 0:
            raise ValueError("--pmf-line-width must be a positive odd integer")
        radius = args.pmf_line_width // 2
        if radius:
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = xx * xx + yy * yy <= radius * radius
            fitted_display = binary_dilation(fitted_centerline > 0, structure=disk).astype(float)
        else:
            fitted_display = fitted_centerline

        fig, axes = plt.subplots(1, 3, figsize=(4.55, 1.72), gridspec_kw={"wspace": 0.16})
        for ax in axes:
            ax.set_facecolor("black")
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0].imshow(observed_canvas, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
        axes[1].imshow(fitted_display, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
        overlay = np.zeros((height, width, 3), dtype=float)
        observed_mask = observed_canvas > 0.5
        overlay[observed_mask] = (0.48, 0.48, 0.48)
        centerline_mask = fitted_display > 0
        overlay[centerline_mask] = (0.93, 0.08, 0.11)
        axes[2].imshow(overlay, interpolation="nearest")

        labels = ("(a)", "(b)", "(c)")
        for ax, label in zip(axes, labels):
            ax.text(0.5, -0.085, label, transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, family="Arial")

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.18)
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight",
                    facecolor="white")
        fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(args.output_prefix.with_suffix(".png"))
        print(args.output_prefix.with_suffix(".pdf"))
        return

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.18), gridspec_kw={"wspace": 0.28})

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    setup_image_axis(axes[0], "(a) Noisy observation")

    axes[1].scatter(fitted[:, 1], fitted[:, 0], s=2.0, c="#d95f02", alpha=0.72,
                    linewidths=0, rasterized=True)
    setup_image_axis(axes[1], "(b) PSO-fitted character")

    axes[2].scatter(observed[:, 1], observed[:, 0], s=3.0, c="#8b96a3", alpha=0.40,
                    linewidths=0, label="observed", rasterized=True)
    axes[2].scatter(fitted[:, 1], fitted[:, 0], s=1.6, c="#d95f02", alpha=0.68,
                    linewidths=0, label="fitted", rasterized=True)
    setup_image_axis(axes[2], "(c) Observation + fit")
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                   frameon=False, handletextpad=0.25, columnspacing=0.7, markerscale=2)

    axes[3].step(episodes, scores, where="post", color="#2b6cb0", linewidth=1.5)
    axes[3].scatter([episodes[-1]], [scores[-1]], s=18, color="#d95f02", zorder=3)
    axes[3].annotate(f"{scores[-1]:.2f}", (episodes[-1], scores[-1]),
                     xytext=(-4, 6), textcoords="offset points", ha="right")
    axes[3].set_title("(d) Search convergence", loc="left", fontweight="semibold")
    axes[3].set_xlabel("Function evaluations")
    axes[3].set_ylabel("Best PMF similarity")
    axes[3].set_xlim(0, max(10000, episodes[-1]))
    axes[3].grid(True, color="#d7dce2", linewidth=0.55, alpha=0.8)
    axes[3].spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.005,
        f"run1-test2, salt-and-pepper level 0.6   |   Chamfer = {record['chamfer']:.4f}"
        f"   |   FEs = {int(record['num_evaluations']):,}",
        ha="center", va="bottom", fontsize=7.8,
    )
    fig.subplots_adjust(left=0.025, right=0.992, top=0.88, bottom=0.23)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output_prefix.with_suffix(".png"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
