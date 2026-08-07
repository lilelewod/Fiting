"""Create an IEEE-style four-panel figure for one superquadric fit."""

import argparse
import base64
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from plyfile import PlyData
from sklearn.neighbors import KDTree


def read_ply(path):
    vertex = PlyData.read(str(path))['vertex']
    return np.column_stack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float64)


def common_pca(reference, *clouds):
    center = reference.mean(axis=0)
    covariance = np.cov((reference - center).T)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return [(cloud - center) @ axes for cloud in (reference,) + clouds]


def equal_axes(ax, points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-58)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--input-cloud', required=True)
    parser.add_argument('--reference-cloud', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--html-output', default=None)
    parser.add_argument('--sample-eta', type=int, default=96)
    parser.add_argument('--sample-omega', type=int, default=96)
    args = parser.parse_args()

    root = Path(args.results_root)
    with open(root / 'results.json', encoding='utf-8') as stream:
        result = json.load(stream)[0]
    algorithm = str(result.get('algorithm', 'optimizer')).upper()
    if algorithm == 'PSO' and result.get('pso_guided_initialization'):
        algorithm_label = 'Guided-PSO'
    else:
        algorithm_label = algorithm
    fitted_file = Path(result['record_file']).parent / 'best_cloud_of_instance_0.ply'
    observed = read_ply(args.input_cloud)
    reference = read_ply(args.reference_cloud)
    fitted = read_ply(fitted_file)
    reference, observed, fitted = common_pca(reference, observed, fitted)
    combined = np.vstack((reference, fitted))
    grid = fitted.reshape(args.sample_eta, args.sample_omega, 3)

    distances = KDTree(reference).query(fitted, k=1)[0].ravel()
    distance_grid = distances.reshape(args.sample_eta, args.sample_omega)
    color_max = max(float(np.percentile(distances, 95)), np.finfo(float).eps)
    norm = Normalize(0.0, color_max)
    cmap = mpl.colormaps['viridis']

    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9.5,
        'axes.titlesize': 9.5,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'axes.linewidth': 0.75,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig = plt.figure(figsize=(7.16, 5.0), constrained_layout=False)
    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]

    rng = np.random.default_rng(7)
    shown = rng.choice(observed.shape[0], min(4000, observed.shape[0]), replace=False)
    axes[0].scatter(*observed[shown].T, s=0.7, c='#2878B5', alpha=0.72, linewidths=0, rasterized=True)
    axes[0].set_title('(a) Observed ModelNet40 points', pad=-2)

    axes[1].plot_surface(
        grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
        rstride=2, cstride=2, color='#E8943A', linewidth=0, antialiased=True,
        shade=True, alpha=1.0, rasterized=True,
    )
    axes[1].set_title(f'(b) {algorithm_label}-fitted superquadric surface', pad=-2)

    overlay_ids = rng.choice(observed.shape[0], min(2500, observed.shape[0]), replace=False)
    axes[2].scatter(*observed[overlay_ids].T, s=0.6, c='#2878B5', alpha=0.35, linewidths=0, rasterized=True)
    axes[2].plot_wireframe(
        grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
        rstride=5, cstride=5, color='#D95F02', linewidth=0.35, alpha=0.85,
    )
    axes[2].set_title('(c) Observation and fitted surface', pad=-2)

    axes[3].plot_surface(
        grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
        rstride=1, cstride=1, facecolors=cmap(norm(distance_grid)),
        linewidth=0, antialiased=False, shade=False, rasterized=True,
    )
    axes[3].set_title('(d) Model-to-reference error', pad=-2)

    for ax in axes:
        equal_axes(ax, combined)

    # Keep all four 3-D panels at exactly the same size.  Passing ``ax=axes[3]``
    # to ``fig.colorbar`` shrinks only the lower-right panel and visibly breaks
    # the 2x2 alignment, so the colorbar lives in an inset instead.
    colorbar_axis = axes[3].inset_axes([0.92, 0.16, 0.026, 0.66])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis,
    )
    colorbar.set_label('Nearest-surface distance')
    colorbar.ax.tick_params(labelsize=8.5, width=0.5, length=2)

    metrics = (
        f"GT Chamfer = {result['gt_chamfer']:.5f}    "
        f"F-score@{result['gt_metric_threshold']:.2f} = {result['gt_fscore']:.4f}    "
        f"FEs = {int(result['evaluations']):,}    Time = {result['wall_time_s']:.1f} s"
    )
    fig.text(0.5, 0.018, metrics, ha='center', va='bottom', fontsize=9.5)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.98, bottom=0.075, wspace=0.01, hspace=0.02)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_file = prefix.with_suffix('.png')
    pdf_file = prefix.with_suffix('.pdf')
    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.025)
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', pad_inches=0.025)
    plt.close(fig)

    if args.html_output:
        encoded = base64.b64encode(png_file.read_bytes()).decode('ascii')
        html = (
            '<div id="sq-paper-result" style="width:100%;">\n'
            f'  <img alt="Four-panel paper figure comparing observed ModelNet40 bottle points, the {algorithm_label}-fitted superquadric, their overlay, and nearest-surface error" '
            'style="display:block;width:100%;height:auto;" '
            f'src="data:image/png;base64,{encoded}">\n'
            '</div>\n'
        )
        html_path = Path(args.html_output)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding='utf-8')
    print(png_file)
    print(pdf_file)


if __name__ == '__main__':
    main()
