"""
GD Scaling 实验 — 变控制网格维度，四优化器对比
================================================
证明 GD 在高维非凸优化上碾压 CS/CCO/ALA。

    网格     变量数     GD     CS     CCO    ALA
    6×6       72       ✓      ✓      ✓      ✓
    8×8      128       ✓      ✓      ?      ?
   10×10     200       ✓      ?      ✗      ✗
   12×12     288       ✓      ✗      ✗      ✗

用法:
    cd /home/m25lll/code/Fiting/fitting
    python experiments/scaling_experiment.py
"""

import os, sys, time, json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import load_ply_data
from tools.tool import current_timestamp
from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.nurbs_surface_rule import NURBSSurfaceRule
from experiments.diffcd_verify import save_ply  # noqa: F401  # 备用

# ─── 实验配置 ──────────────────────────────────────

DATA_FILE = '/home/m25lll/code/Fiting/fitting/datasets/synthetic/saddle_3k.ply'
GRID_SIZES = [6, 8, 10]
ALGOS = ['gd', 'cco', 'cs', 'ala']
MAX_EPISODE = 2000             # 合成数据足够
NUM_INSTANCES = 1
DEVICE = 'cuda:1'
DATA_RESOLUTION = 0.02
MODEL_RESOLUTION = 0.008

# ─── 构建配置 ──────────────────────────────────────

def build_config(ply_path, output_root, grid_size, algo, seed=42):
    sample = max(30, grid_size * 7)  # 采样密度跟控制网格成正比

    return {
        'task_type': '3d',
        'data_file': ply_path,
        'run_id': 1,

        'model': {
            'type': 'nurbs_surface',
            'num_ctrl_u': grid_size,
            'num_ctrl_v': grid_size,
            'degree_u': 3,
            'degree_v': 3,
            'sample_u': sample,
            'sample_v': sample,
            'weight_lb': 0.8,
            'weight_ub': 1.2,
        },

        'device': {
            'train_device': DEVICE,
            'cuda_deterministic': True,
        },
        'seeds': [seed],

        'estimator': {
            'type': 'mm',
            'data_file': ply_path,
            'data_resolution': DATA_RESOLUTION,
            'model_resolution': MODEL_RESOLUTION,
            'rule_class': NURBSSurfaceRule,
            'estimator_class': MeanMeasureEstimator,
            'estimator_instance': None,
            'load_data_fn': load_ply_data,
            'regularization_factor': 1.2,
            'incremental_coverage': False,
            'outlier_distance_factor': 0.0,
            'outlier_penalty_factor': 0.0,
            'bbox_margin_factor': 0.0,
            'bbox_penalty_factor': 0.0,
            'overlap_penalty_factor': 0.0,
            'control_smoothness_penalty_factor': 0.0,
        },

        'fitter': {
            'algo_name': algo,
            'num_instances': NUM_INSTANCES,
            'max_episode': MAX_EPISODE,
            'gd_lr': 0.01,
            'gd_lr_min_factor': 0.1,
            'gd_eval_interval': 200,
            'gd_data_batch_size': 0,
            'gd_mm_aligned': True,
            'gd_coverage_weight': 0.3,
            'gd_smoothness_weight': 0.05,
            'gd_bbox_weight': 0.2 if grid_size <= 8 else 0.05,
            'gd_weight_reg_weight': 0.01,
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
            'verbose': False,
            'trim_final_mesh': False,
            'uv_trim_final_mesh': False,
        },
    }


# ─── 运行一组实验 ─────────────────────────────────

def run_one(grid_size, algo, input_ply, output_root):
    label = f"{grid_size}x{grid_size}_{algo}"
    out_dir = os.path.join(output_root, label)
    cfg = build_config(input_ply, out_dir, grid_size, algo, seed=42)

    # CS/CCO/ALA 需要 num_envs + episodes_per_env
    if algo != 'gd':
        num_workers = min(4, max(2, grid_size // 2))
        cfg['fitter']['num_envs'] = num_workers
        cfg['fitter']['episodes_per_env'] = 10
        cfg['seeds'] = [42 + i for i in range(num_workers + 1)]

    t0 = time.time()
    fitter = None

    try:
        if algo == 'gd':
            from core.optimizer.gd_fitter import Fitter
            fitter = Fitter(cfg)
        elif algo == 'cco':
            from core.optimizer.cco_fitter import Fitter
            fitter = Fitter(cfg)
        elif algo == 'cs':
            from core.optimizer.cs_fitter import Fitter
            fitter = Fitter(cfg)
        elif algo == 'ala':
            from core.optimizer.ala_fitter import Fitter
            fitter = Fitter(cfg)
        else:
            raise ValueError(algo)

        fitter.fit()
        best_score = float(fitter.record.best_score)

        evo_file = os.path.join(fitter.record.log_dir,
                                'evolution_of_round_0_instance_0.json')
        num_evals = 0
        if os.path.exists(evo_file):
            with open(evo_file) as f:
                evo = json.load(f)
            num_evals = len(evo)
    except Exception as e:
        best_score = 0.0
        num_evals = 0
        import traceback
        print(f"\n  ! {label} 出错: {e}")
        traceback.print_exc()
    finally:
        if fitter is not None:
            try:
                fitter.close()
            except Exception:
                pass

    elapsed = time.time() - t0

    return {
        'grid': grid_size,
        'dim': grid_size * grid_size * 3 + grid_size * grid_size,  # ctrl+weights
        'algo': algo,
        'score': best_score,
        'evals': num_evals,
        'time': elapsed,
    }


# ─── 主函数 ───────────────────────────────────────

def main():
    output_root = Path(__file__).resolve().parent / 'output_scaling' / time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(str(output_root), exist_ok=True)

    # 准备数据
    if DATA_FILE and os.path.exists(DATA_FILE):
        input_ply = DATA_FILE
        print(f"数据: {input_ply}")
    else:
        # 合成鞍面 z = x² - y² + noise，可以用 NURBS 精确表达
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, 2000)
        y = rng.uniform(-1, 1, 2000)
        z = x**2 - y**2 + rng.normal(0, 0.01, 2000)
        pts = np.column_stack((x, y, z))
        input_ply = str(output_root / 'input.ply')
        save_ply(pts, input_ply)
        print(f"数据: 合成鞍面 2000点 → {input_ply}")

    results = []
    for grid in GRID_SIZES:
        dim = grid * grid * 3 + grid * grid
        print(f"\n{'='*50}")
        print(f"网格 {grid}×{grid}  →  {dim} 维")
        for algo in ALGOS:
            print(f"  {algo}...", end=' ', flush=True)
            r = run_one(grid, algo, input_ply, str(output_root))
            results.append(r)
            print(f"score={r['score']:.4f}  evals={r['evals']}  {r['time']:.0f}s")

    # ─── 报告 ──────────────────────────────────────
    report_parts = []
    report_parts.append("=" * 65)
    report_parts.append("GD Scaling 实验 — MM评分，变网格维度")
    report_parts.append("=" * 65)
    header = f"{'grid':<8} {'dim':<8}"
    for a in ALGOS:
        header += f" {a:>12}"
    report_parts.append(header)
    report_parts.append("-" * 65)

    for grid in GRID_SIZES:
        dim = grid * grid * 3 + grid * grid
        line = f"{grid}×{grid:<4}  {dim:<6}"
        for a in ALGOS:
            matches = [r for r in results if r['grid'] == grid and r['algo'] == a]
            if matches and matches[0]['score'] > 0:
                line += f" {matches[0]['score']:>8.4f} {matches[0]['time']:>3.0f}s"
            else:
                line += f" {'─':>12}"
        report_parts.append(line)

    report_parts.append("-" * 65)
    report_parts.append("")

    # GD vs others
    gd_scores = {r['grid']: r['score'] for r in results if r['algo'] == 'gd'}
    for a in ['cs', 'cco', 'ala']:
        wins = 0
        for grid in GRID_SIZES:
            gd = gd_scores.get(grid, 0)
            other = next((r['score'] for r in results if r['grid'] == grid and r['algo'] == a), 0)
            if gd > other:
                wins += 1
        report_parts.append(f"GD > {a.upper()}: {wins}/{len(GRID_SIZES)}")

    report = "\n".join(report_parts)
    print("\n" + report)

    with open(str(output_root / 'report.txt'), 'w') as f:
        f.write(report)
    with open(str(output_root / 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n完整结果: {output_root}")


if __name__ == '__main__':
    main()
