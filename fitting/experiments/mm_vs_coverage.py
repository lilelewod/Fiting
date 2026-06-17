"""
Mean Measure vs Coverage — 对比实验
=====================================
同一数据、同一初始化，对比：
  A) Coverage 模式 (sigmoid软覆盖，需设阈值)
  B) MM 模式     (几何测度奖励，无阈值)

用法：
    cd /home/m25lll/code/Fiting/fitting
    python experiments/mm_vs_coverage.py
"""

import os, sys, time, json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import read_point_cloud
from tools.tool import current_timestamp
from core.optimizer.gd_fitter import Fitter
from core.estimator.gd_estimator import GDEstimator
from models.surface.nurbs_surface_rule import NURBSSurfaceRule
from experiments.diffcd_verify import (
    generate_split_patches, save_ply, build_config, _make_loader,
)

# ─── 配置 ──────────────────────────────────────────

DATA_CONFIG = dict(patch_size=1.0, gap_width=0.6, num_points=6000, noise_std=0.005, seed=42)
MAX_STEPS = 3000
SMOOTHNESS = 0.01
DEVICE = 'cuda:1'  # 或 'cpu'

GROUPS = [
    dict(label='A_coverage', coverage_weight=0.3, measure_weight=0.0),
    dict(label='B_measure',  coverage_weight=0.0, measure_weight=0.3),
    dict(label='C_coverage_strong', coverage_weight=0.6, measure_weight=0.0),
    dict(label='D_measure_strong',  coverage_weight=0.0, measure_weight=0.6),
]

# ─── 运行 ──────────────────────────────────────────

def run_group(cfg, label):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"[{label}] 开始")
    print(f"  coverage_weight={cfg['fitter']['gd_coverage_weight']}")
    print(f"  measure_weight={cfg['fitter']['gd_measure_weight']}")
    fitter = Fitter(cfg)
    fitter.fit()
    elapsed = time.time() - t0

    # 收集结果
    best_pts = None
    cloud_path = os.path.join(fitter.record.log_dir, 'best_cloud_of_instance_0.ply')
    if os.path.exists(cloud_path):
        best_pts = read_point_cloud(cloud_path)

    result = dict(
        label=label,
        best_score=fitter.record.best_score,
        output_dir=fitter.record.log_dir,
        elapsed=elapsed,
    )

    if best_pts is not None:
        gap_left, gap_right = -0.3, 0.3  # 从 gap_width=0.6
        in_gap = (best_pts[:, 0] > gap_left) & (best_pts[:, 0] < gap_right) & (np.abs(best_pts[:, 1]) < 0.6)
        result['gap_points'] = int(in_gap.sum())
        result['total_points'] = best_pts.shape[0]
        result['gap_ratio'] = in_gap.sum() / best_pts.shape[0]

    fitter.close()
    print(f"[{label}] 完成: score={result['best_score']:.4f}, "
          f"gap={result.get('gap_points','?')}/{result.get('total_points','?')}, "
          f"耗时={elapsed:.1f}s")
    return result


def main():
    output_root = Path(__file__).resolve().parent / 'output_mm_vs_coverage' / time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(str(output_root), exist_ok=True)

    # 生成数据
    print("生成数据...")
    points, gap_fn, gap_bounds = generate_split_patches(**DATA_CONFIG)
    input_ply = str(output_root / 'input.ply')
    save_ply(points, input_ply)
    print(f"输入: {points.shape[0]} 点, 间隙 x∈({gap_bounds[0]:.2f},{gap_bounds[1]:.2f})")

    results = []
    for g in GROUPS:
        cfg = build_config(input_ply, str(output_root / g['label']),
                           model_to_data_weight=1.0, seed=42)
        cfg['fitter']['max_episode'] = MAX_STEPS
        cfg['fitter']['gd_coverage_weight'] = g['coverage_weight']
        cfg['fitter']['gd_measure_weight'] = g['measure_weight']
        cfg['fitter']['gd_smoothness_weight'] = SMOOTHNESS
        cfg['device']['train_device'] = DEVICE

        result = run_group(cfg, g['label'])
        results.append(result)

    # ─── 报告 ──────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("Mean Measure vs Coverage — 对比结果")
    lines.append("=" * 60)
    lines.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"数据: 双补丁 {points.shape[0]}点, 间隙宽度 0.6")
    lines.append(f"步数: {MAX_STEPS}, smoothness: {SMOOTHNESS}")
    lines.append("")
    lines.append(f"{'组':<25} {'coverage_w':>10} {'measure_w':>10} {'score':>8} {'gap_pts':>8} {'gap_%':>7} {'time':>6}")
    lines.append("-" * 60)
    for r in results:
        lines.append(
            f"{r['label']:<25} "
            f"{GROUPS[[x['label'] for x in GROUPS].index(r['label'])]['coverage_weight']:>10.1f} "
            f"{GROUPS[[x['label'] for x in GROUPS].index(r['label'])]['measure_weight']:>10.1f} "
            f"{r['best_score']:>8.4f} "
            f"{r.get('gap_points','?'):>8} "
            f"{r.get('gap_ratio',0)*100:>6.1f}% "
            f"{r['elapsed']:>5.1f}s"
        )
    lines.append("-" * 60)
    lines.append("")

    # 对比 coverage vs measure 在相同权重下的表现
    a = results[0]  # A_coverage 0.3
    b = results[1]  # B_measure 0.3
    if a.get('gap_points') and b.get('gap_points'):
        ratio = b['gap_points'] / max(a['gap_points'], 1)
        lines.append(f"A (coverage 0.3)  gap内: {a['gap_points']}点, score: {a['best_score']:.4f}")
        lines.append(f"B (measure 0.3)   gap内: {b['gap_points']}点, score: {b['best_score']:.4f}")
        if b['best_score'] > a['best_score']:
            lines.append("→ MM模式score更高，且无需调阈值")
        else:
            lines.append("→ Coverage模式score更高")

    report = "\n".join(lines)
    print("\n" + report)
    with open(str(output_root / 'report.txt'), 'w') as f:
        f.write(report)
    print(f"\n报告: {output_root}/report.txt")
    print(f"输出: {output_root}/")


if __name__ == '__main__':
    main()
