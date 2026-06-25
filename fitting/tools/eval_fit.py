#!/usr/bin/env python3
"""拟合质量评估 — 同时报告 MM 分数和纯几何指标

用法:
    python tools/eval_fit.py outputs/memetic/.../run_1/2026-0624-1551-37/
    python tools/eval_fit.py outputs/memetic/.../run_1/2026-0624-1551-37/ --compare outputs/cco/.../run_1/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_cloud(path_or_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load model and data clouds from a record directory.

    Returns (model_pts, data_pts, meta_dict).
    """
    dir_path = Path(path_or_dir)
    if dir_path.is_dir():
        rec_path = dir_path / 'record.json'
        ply_path = dir_path / 'best_cloud_of_instance_0.ply'
    else:
        rec_path = dir_path
        ply_path = dir_path.parent / 'best_cloud_of_instance_0.ply'

    with open(rec_path) as f:
        rec = json.load(f)

    # Data cloud
    cfg = rec.get('cfg', {})
    data_file = cfg.get('data_file', '')
    if data_file and os.path.exists(data_file):
        import open3d as o3d
        data_pcd = o3d.io.read_point_cloud(data_file)
        data_pts = np.asarray(data_pcd.points)
    elif 'data_cloud' in rec:
        dc = rec['data_cloud']
        if isinstance(dc, list) and len(dc) > 0 and isinstance(dc[0], list):
            data_pts = np.array(dc, dtype=np.float32)
        else:
            data_pts = np.zeros((0, 3), dtype=np.float32)
    else:
        data_pts = np.zeros((0, 3), dtype=np.float32)

    # Model cloud
    if ply_path.exists():
        import open3d as o3d
        model_pcd = o3d.io.read_point_cloud(str(ply_path))
        model_pts = np.asarray(model_pcd.points)
    else:
        model_pts = np.zeros((0, 3), dtype=np.float32)

    # Meta
    algo = cfg.get('fitter', {}).get('algo_name', '?')
    model_type = cfg.get('model', {}).get('type', '?')
    lamb = cfg.get('estimator', {}).get('regularization_factor', '?')
    best_score = rec.get('best_score', float('nan'))
    data_name = os.path.basename(data_file) if data_file else '?'

    # Extract measure from token
    measure = float('nan')
    ts = rec.get('best_token_set')
    if isinstance(ts, list) and len(ts) > 0:
        t = ts[0]
        if isinstance(t, dict):
            measure = float(t.get('measure', float('nan')))
    elif isinstance(ts, dict):
        tokens = ts.get('tokens', [])
        if tokens and isinstance(tokens[0], dict):
            measure = float(tokens[0].get('measure', float('nan')))

    # Runtime
    runtime = 0.0
    evo_files = list(dir_path.glob('evolution_of_round_0_instance_0.json'))
    if evo_files:
        with open(evo_files[0]) as f:
            evo = json.load(f)
        if evo:
            runtime = evo[-1].get('elpased_time', 0)

    meta = {
        'algo': algo, 'model': model_type, 'lambda': lamb,
        'score': best_score, 'measure': measure, 'data': data_name,
        'runtime': runtime,
    }
    return model_pts, data_pts, meta


def compute_metrics(model_pts: np.ndarray, data_pts: np.ndarray) -> dict:
    """Compute geometric error metrics."""
    if len(model_pts) == 0 or len(data_pts) == 0:
        return {}

    tree_m = cKDTree(model_pts)
    d2m, _ = tree_m.query(data_pts)

    tree_d = cKDTree(data_pts)
    m2d, _ = tree_d.query(model_pts)

    # Coverage: fraction of data within thresholds
    coverage_001 = float((d2m < 0.01).mean())
    coverage_002 = float((d2m < 0.02).mean())
    coverage_005 = float((d2m < 0.05).mean())

    return {
        # Data→Model
        'd2m_mean': float(d2m.mean()),
        'd2m_median': float(np.median(d2m)),
        'd2m_p90': float(np.percentile(d2m, 90)),
        'd2m_p95': float(np.percentile(d2m, 95)),
        'd2m_max': float(d2m.max()),
        'd2m_std': float(d2m.std()),
        # Model→Data
        'm2d_mean': float(m2d.mean()),
        'm2d_median': float(np.median(m2d)),
        'm2d_p95': float(np.percentile(m2d, 95)),
        'm2d_max': float(m2d.max()),
        # Chamfer & Hausdorff
        'chamfer': float(d2m.mean() + m2d.mean()),
        'hausdorff': float(max(d2m.max(), m2d.max())),
        # Coverage
        'cov_0.01': coverage_001,
        'cov_0.02': coverage_002,
        'cov_0.05': coverage_005,
    }


def print_row(meta: dict, m: dict, name: str = ''):
    label = name or f"{meta['algo']}/{meta['model']}"
    print(f"  {label:<20s}  "
          f"MM={meta['score']:>8.4f}  "
          f"Chamfer={m.get('chamfer', 0):>7.4f}  "
          f"Hausdorff={m.get('hausdorff', 0):>7.4f}  "
          f"D→M mean={m.get('d2m_mean', 0):>7.4f}  "
          f"cov<0.01={m.get('cov_0.01', 0):>6.1%}  "
          f"|M|={meta['measure']:>8.1f}  "
          f"time={meta['runtime']:>6.0f}s")


def main():
    parser = argparse.ArgumentParser(description='拟合质量评估')
    parser.add_argument('path', type=str, nargs='+',
                        help='record.json 或输出目录路径 (支持多个)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示完整误差分布')
    parser.add_argument('--csv', action='store_true',
                        help='CSV格式输出')
    args = parser.parse_args()

    if args.csv:
        header = ['name', 'algo', 'model', 'lambda', 'score', 'chamfer', 'hausdorff',
                  'd2m_mean', 'd2m_median', 'd2m_p95', 'd2m_max',
                  'm2d_mean', 'm2d_median', 'm2d_p95', 'm2d_max',
                  'cov_0.01', 'cov_0.02', 'cov_0.05', 'measure', 'runtime', 'data']
        print(','.join(header))

    for path in args.path:
        model_pts, data_pts, meta = load_cloud(path)
        if len(model_pts) == 0:
            print(f"Error: no model points in {path}", file=sys.stderr)
            continue

        m = compute_metrics(model_pts, data_pts)

        if args.csv:
            name = os.path.basename(str(path).rstrip('/'))
            row = [name, meta['algo'], meta['model'], str(meta['lambda']),
                   f"{meta['score']:.4f}", f"{m.get('chamfer',0):.6f}",
                   f"{m.get('hausdorff',0):.6f}",
                   f"{m.get('d2m_mean',0):.6f}", f"{m.get('d2m_median',0):.6f}",
                   f"{m.get('d2m_p95',0):.6f}", f"{m.get('d2m_max',0):.6f}",
                   f"{m.get('m2d_mean',0):.6f}", f"{m.get('m2d_median',0):.6f}",
                   f"{m.get('m2d_p95',0):.6f}", f"{m.get('m2d_max',0):.6f}",
                   f"{m.get('cov_0.01',0):.4f}", f"{m.get('cov_0.02',0):.4f}",
                   f"{m.get('cov_0.05',0):.4f}",
                   f"{meta['measure']:.2f}", f"{meta['runtime']:.1f}",
                   meta['data']]
            print(','.join(row))
        else:
            print(f"\n{'='*70}")
            print(f"  {meta['algo']} / {meta['model']}  λ={meta['lambda']}  data={meta['data']}")
            print(f"{'='*70}")
            print(f"  MM Score:      {meta['score']:.4f}")
            print(f"  Measure |M|:   {meta['measure']:.1f}")
            print(f"  Wall time:     {meta['runtime']:.0f}s")
            print()
            print(f"  {'Metric':<18s} {'Value':>10s}")
            print(f"  {'-'*28}")
            print(f"  {'Chamfer (mean)':<18s} {m.get('chamfer', 0):>10.4f}")
            print(f"  {'Hausdorff (max)':<18s} {m.get('hausdorff', 0):>10.4f}")
            print(f"  {'D→M mean':<18s} {m.get('d2m_mean', 0):>10.4f}")
            print(f"  {'D→M median':<18s} {m.get('d2m_median', 0):>10.4f}")
            print(f"  {'D→M P95':<18s} {m.get('d2m_p95', 0):>10.4f}")
            print(f"  {'D→M max':<18s} {m.get('d2m_max', 0):>10.4f}")
            print(f"  {'M→D mean':<18s} {m.get('m2d_mean', 0):>10.4f}")
            print(f"  {'M→D P95':<18s} {m.get('m2d_p95', 0):>10.4f}")
            print(f"  {'Coverage <0.01':<18s} {m.get('cov_0.01', 0):>10.1%}")
            print(f"  {'Coverage <0.02':<18s} {m.get('cov_0.02', 0):>10.1%}")
            print(f"  {'Coverage <0.05':<18s} {m.get('cov_0.05', 0):>10.1%}")

            if args.verbose and len(model_pts) > 0:
                bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, float('inf')]
                tree_m = cKDTree(model_pts)
                d2m, _ = tree_m.query(data_pts)
                print(f"\n  D→M 误差直方图:")
                for i in range(len(bins)-1):
                    mask = (d2m >= bins[i]) & (d2m < bins[i+1])
                    pct = mask.sum() / len(d2m) * 100
                    bar = '█' * int(pct)
                    print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {mask.sum():5d} ({pct:5.1f}%) {bar}")


if __name__ == '__main__':
    main()
