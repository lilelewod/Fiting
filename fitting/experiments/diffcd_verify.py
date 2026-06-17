"""
DiffCD 双向 Chamfer 验证实验
===============================
验证 DiffCD (ECCV 2024) 的理论发现：
单向 Chamfer 距离 (data→model) 在缺失数据区域产生伪曲面，
双向 Chamfer (data→model + model→data) 抑制伪曲面。

实验：用部分球面（切掉下半部 30%）拟合 NURBS 曲面，
对比 group A (model_to_data_weight=0.0, 单向) vs group B (model_to_data_weight=1.0, 双向)。

预期结果：
- A 组在缺失区域（球面下半部）产生额外的曲面点
- B 组曲面忠实于输入数据，不填补缺失区域

用法：
    cd /home/m25lll/code/Fiting/fitting
    python experiments/diffcd_verify.py
"""

import os
import sys
import time
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

# ensure the fitting package is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import current_timestamp, set_seed, get_seeds
from core.estimator.gd_estimator import GDEstimator
from core.optimizer.gd_fitter import Fitter
from models.surface.nurbs_surface_rule import NURBSSurfaceRule


# ──────────────────────────────────────────────
# 1. 合成数据生成
# ──────────────────────────────────────────────

def generate_split_patches(patch_size=1.0, gap_width=0.5, num_points=6000,
                            noise_std=0.005, seed=42):
    """
    生成两个矩形平面补丁（z=0），中间有一条间隙。
    这是测试单向 Chamfer 盲区的最干净场景：
    - 间隙区域无数据点
    - 单向 Chamfer：无惩罚 → NURBS 可能填充间隙
    - 双向 Chamfer：model→data 惩罚间隙中的曲面点 → 抑制填充

    Returns:
        points: (N, 3) 点云
        gap_mask_fn: 函数，输入 (N, 3) 点返回 True 表示落在间隙区域
        gap_bounds: (x_left, x_right) 间隙的 x 坐标范围
    """
    rng = np.random.default_rng(seed)
    half = num_points // 2

    # Patch A: x in [-patch_size - gap_width/2, -gap_width/2]
    x_a = rng.uniform(-patch_size - gap_width / 2, -gap_width / 2, half)
    y_a = rng.uniform(-patch_size / 2, patch_size / 2, half)

    # Patch B: x in [gap_width/2, patch_size + gap_width/2]
    x_b = rng.uniform(gap_width / 2, patch_size + gap_width / 2, half)
    y_b = rng.uniform(-patch_size / 2, patch_size / 2, half)

    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])
    z = np.zeros(num_points)

    # 加噪声（z方向微量）
    if noise_std > 0:
        x += rng.normal(0, noise_std * patch_size, num_points)
        y += rng.normal(0, noise_std * patch_size, num_points)
        z += rng.normal(0, noise_std * patch_size * 0.1, num_points)

    points = np.column_stack((x, y, z))

    gap_left = -gap_width / 2
    gap_right = gap_width / 2

    def gap_mask_fn(pts):
        """pts 中 x 坐标落在间隙区域返回 True"""
        in_gap_x = (pts[:, 0] > gap_left) & (pts[:, 0] < gap_right)
        in_patch_y = np.abs(pts[:, 1]) < patch_size / 2 * 1.2
        return in_gap_x & in_patch_y

    print(f"[数据] 生成双补丁: {points.shape[0]} 个点, "
          f"间隙 x∈({gap_left:.2f}, {gap_right:.2f}), 宽度={gap_width:.2f}")

    return points, gap_mask_fn, (gap_left, gap_right)


def save_ply(points, filepath):
    """保存 numpy (N,3) 为 PLY 文件（优先用 open3d，否则手动写 ASCII PLY）"""
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        # 手动写 ASCII PLY（无任何库依赖）
        with open(filepath, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"[保存] {filepath}")


# ──────────────────────────────────────────────
# 2. 构建配置
# ──────────────────────────────────────────────

def build_config(ply_path, output_root, model_to_data_weight, seed=None):
    """
    构建一个最小可用的 GD 拟合配置。

    Args:
        ply_path: 输入 PLY 文件路径
        output_root: 输出根目录
        model_to_data_weight: Chamfer M→D 权重 (0.0 = 单向, 1.0 = 双向)
        seed: 随机种子
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 100000))

    cfg = {
        'task_type': '3d',
        'data_file': ply_path,
        'run_id': 1,

        'model': {
            'type': 'nurbs_surface',
            'num_ctrl_u': 6,
            'num_ctrl_v': 6,
            'degree_u': 3,
            'degree_v': 3,
            'sample_u': 40,       # 较小采样密度加速实验
            'sample_v': 40,
            'weight_lb': 0.8,
            'weight_ub': 1.2,
        },

        'device': {
            'train_device': 'cuda:1',
            'cuda_deterministic': False,
        },
        'seeds': [seed],

        'estimator': {
            'type': 'gd',
            'data_file': ply_path,
            'data_resolution': 0.02,
            'model_resolution': 0.008,
            'rule_class': NURBSSurfaceRule,
            'estimator_class': GDEstimator,
            'estimator_instance': None,
            'load_data_fn': _make_loader(),
            'regularization_factor': 1.2,
            'incremental_coverage': False,   # 单实例清零
            'overlap_penalty_factor': 0.0,   # 单实例不需要
            'outlier_distance_factor': 0.0,  # 合成数据无离群点
            'outlier_penalty_factor': 0.0,
            'bbox_margin_factor': 0.0,
            'bbox_penalty_factor': 0.0,
            'control_smoothness_penalty_factor': 0.5,
        },

        'fitter': {
            'algo_name': 'gd',
            'num_instances': 1,              # 单实例，简化分析
            # 训练控制
            'max_episode': 5000,             # 合成小数据足够
            'gd_lr': 0.01,
            'gd_lr_min_factor': 0.1,
            'gd_eval_interval': 250,         # 减少评估开销
            'gd_data_batch_size': 0,         # 全批量
            # Loss 权重 — 这里设置实验变量
            'gd_data_to_model_weight': 1.0,
            'gd_model_to_data_weight': model_to_data_weight,  # ★ 实验变量
            'gd_coverage_weight': 0.2,
            'gd_coverage_threshold_factor': 2.5,
            'gd_coverage_temperature_factor': 0.5,
            'gd_smoothness_weight': 0.05,
            'gd_bbox_weight': 0.0,
            'gd_weight_reg_weight': 0.01,
            'gd_overlap_weight': 0.0,        # 单实例不需要
            'gd_overlap_margin_factor': 2.0,
            'gd_exclude_covered': False,
        },

        'collector': {
            'name': 'hybrid',
            'num_episodes_per_rollout': 1,
            'num_rollouts': 1,
            'parallel': False,
            'context': 'spawn',
            'reinforcement_episode_size': 1,
            'evaluation_episode_size': 1,
        },

        'record': {
            'root_dir': output_root,
            'pulse_size': 1000,
            'visualization': None,
            'use_thread_time': False,
            'verbose': True,
            'trim_final_mesh': False,
            'uv_trim_final_mesh': False,
        },
    }
    return cfg


def _make_loader():
    """返回一个适配 GDEstimator.load_data 接口的数据加载函数"""
    from tools.data_tool import read_point_cloud

    def load_data(estimator):
        cfg = estimator.cfg['estimator']
        data_path = cfg['data_file']
        print(f'[加载] {data_path}')
        data = read_point_cloud(data_path)
        estimator.raw_data = data.copy()
        estimator.data_resolution = cfg['data_resolution']
        estimator.model_resolution = cfg['model_resolution']
        estimator.min_point = data.min(0)
        estimator.max_point = data.max(0)
        estimator.resolution = estimator.model_resolution
        return data

    return load_data


# ──────────────────────────────────────────────
# 3. 运行单次实验
# ──────────────────────────────────────────────

def run_single_experiment(cfg, label):
    """运行一次完整的 GD 拟合，返回最优 NPRE score 和输出目录"""
    print(f"\n{'='*60}")
    print(f"[实验] {label}")
    weight = cfg['fitter']['gd_model_to_data_weight']
    print(f"[实验] model_to_data_weight = {weight}")
    print(f"{'='*60}")

    # 更安全的 GDEstimator 实例化
    cfg['estimator']['estimator_instance'] = None

    t_start = time.time()
    fitter = Fitter(cfg)
    fitter.fit()
    fitter.close()
    elapsed = time.time() - t_start

    output_dir = fitter.record.log_dir
    best_score = fitter.record.best_score

    # 加载最优曲面
    best_cloud_path = os.path.join(output_dir, 'best_cloud_of_instance_0.ply')
    best_cloud = None
    if os.path.exists(best_cloud_path):
        from tools.data_tool import read_point_cloud
        best_cloud = read_point_cloud(best_cloud_path)

    print(f"[完成] {label}: best_score={best_score:.4f}, "
          f"model_points={best_cloud.shape[0] if best_cloud is not None else '?'}, "
          f"耗时={elapsed:.1f}s")

    return {
        'label': label,
        'weight': weight,
        'output_dir': output_dir,
        'best_score': best_score,
        'best_cloud': best_cloud,
        'elapsed': elapsed,
    }


# ──────────────────────────────────────────────
# 4. 分析
# ──────────────────────────────────────────────

def analyze_results(result_a, result_b, gap_mask_fn, gap_bounds, output_dir):
    """对比 A (单向) 和 B (双向) 在间隙区域的曲面填充情况"""
    cloud_a = result_a['best_cloud']
    cloud_b = result_b['best_cloud']

    if cloud_a is None or cloud_b is None:
        print("[分析] 错误：无法加载最优曲面，跳过分析")
        return

    gap_a = gap_mask_fn(cloud_a)
    gap_b = gap_mask_fn(cloud_b)
    n_gap_a = gap_a.sum()
    n_gap_b = gap_b.sum()

    lines = []
    lines.append("=" * 60)
    lines.append("DiffCD 双向 Chamfer 验证 — 实验结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"输入: 双平面补丁, 间隙 x∈({gap_bounds[0]:.2f}, {gap_bounds[1]:.2f})")
    lines.append(f"       data_resolution=0.02")
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"{'指标':<40} {'A (单向)':>8} {'B (双向)':>8}")
    lines.append("─" * 60)
    lines.append(f"{'model_to_data_weight':<40} {result_a['weight']:>8.1f} {result_b['weight']:>8.1f}")
    lines.append(f"{'NPRE best score':<40} {result_a['best_score']:>8.4f} {result_b['best_score']:>8.4f}")
    lines.append(f"{'曲面总点数':<40} {cloud_a.shape[0]:>8d} {cloud_b.shape[0]:>8d}")
    lines.append(f"{'间隙内曲面点数':<40} {n_gap_a:>8d} {n_gap_b:>8d}")
    lines.append(f"{'间隙内点比例':<40} {n_gap_a/cloud_a.shape[0]:>8.2%} {n_gap_b/cloud_b.shape[0]:>8.2%}")
    lines.append(f"{'耗时 (秒)':<40} {result_a['elapsed']:>8.1f} {result_b['elapsed']:>8.1f}")
    lines.append("─" * 60)
    lines.append("")

    ratio = n_gap_a / max(n_gap_b, 1)
    if ratio > 2.0:
        lines.append(f"✓ 结论：A 组（单向）在间隙区域产生了 {ratio:.1f}x 更多的伪曲面点，")
        lines.append("  验证了 DiffCD 的单向 Chamfer 盲区理论。")
        lines.append("  B 组（双向）的 model→data 方向有效抑制了间隙填充。")
    elif ratio > 1.3:
        lines.append(f"~ 结论：A 组（单向）在间隙区域产生了 {ratio:.1f}x 更多的伪曲面点，")
        lines.append("  趋势符合 DiffCD 预期。")
    else:
        lines.append(f"✗ 结论：两组在间隙区域无显著差异 (A/B = {ratio:.2f}x)。")
        lines.append("  可能原因：NURBS 控制网格自由度不足或数据分辨率的 gap 不够宽。")

    lines.append("")
    lines.append("文件位置:")
    lines.append(f"  A (单向): {result_a['output_dir']}")
    lines.append(f"  B (双向): {result_b['output_dir']}")
    lines.append(f"  输入点云: {os.path.join(output_dir, 'split_patches_input.ply')}")

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[保存] 报告: {report_path}")


# ──────────────────────────────────────────────
# 5. 主函数
# ──────────────────────────────────────────────

def main():
    # 设置路径
    experiment_dir = Path(__file__).resolve().parent
    output_base = experiment_dir / 'output_diffcd'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_root = output_base / timestamp

    # 生成合成数据
    print("=" * 60)
    print("DiffCD 双向 Chamfer 验证实验")
    print("=" * 60)

    points, gap_mask_fn, gap_bounds = generate_split_patches(
        patch_size=1.0, gap_width=0.6, num_points=6000, noise_std=0.005, seed=42
    )

    # 保存输入点云
    os.makedirs(str(output_root), exist_ok=True)
    input_ply = str(output_root / 'split_patches_input.ply')
    save_ply(points, input_ply)

    print(f"\n[输出] 根目录: {output_root}")

    # 构建两组配置
    seed = 42
    output_a = str(output_root / 'A_one_sided')
    output_b = str(output_root / 'B_two_sided')
    cfg_a = build_config(input_ply, output_a, model_to_data_weight=0.0, seed=seed)
    cfg_b = build_config(input_ply, output_b, model_to_data_weight=1.0, seed=seed)

    # ★ 关键：smoothness 设为 0，让 NURBS 有完全自由去填补或留空间隙
    cfg_a['fitter']['gd_smoothness_weight'] = 0.0
    cfg_b['fitter']['gd_smoothness_weight'] = 0.0
    # 减少步数——平面拟合比球面快得多
    cfg_a['fitter']['max_episode'] = 3000
    cfg_b['fitter']['max_episode'] = 3000
    # 增大控制网格以便跨越间隙
    cfg_a['model']['sample_u'] = 60
    cfg_a['model']['sample_v'] = 60
    cfg_b['model']['sample_u'] = 60
    cfg_b['model']['sample_v'] = 60
    # 放宽松 weight 边界
    cfg_a['model']['weight_lb'] = 0.5
    cfg_a['model']['weight_ub'] = 2.0
    cfg_b['model']['weight_lb'] = 0.5
    cfg_b['model']['weight_ub'] = 2.0

    # 保存配置
    with open(str(output_root / 'config_A.json'), 'w') as f:
        json.dump({k: str(v) for k, v in cfg_a.items()}, f, indent=2, default=str)
    with open(str(output_root / 'config_B.json'), 'w') as f:
        json.dump({k: str(v) for k, v in cfg_b.items()}, f, indent=2, default=str)

    print("\n开始实验...")

    # 运行实验
    result_a = run_single_experiment(cfg_a, "A — 单向 Chamfer (model→data=0)")
    result_b = run_single_experiment(cfg_b, "B — 双向 Chamfer (model→data=1)")

    # 分析
    analyze_results(result_a, result_b, gap_mask_fn, gap_bounds, str(output_root))

    print("\n实验完成！")


if __name__ == '__main__':
    main()
