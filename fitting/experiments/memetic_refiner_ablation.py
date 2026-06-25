#!/usr/bin/env python3
"""NL-SHADE+Adam vs 纯NL-SHADE 对比实验 — NURBS鞍面 (16×16, 1024D)

      python experiments/memetic_refiner_ablation.py
"""

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np

from tools.tool import current_timestamp, set_project_root_as_working_directory
set_project_root_as_working_directory(__file__)

from core.estimator.mm_estimator import MeanMeasureEstimator
from core.optimizer.memetic_fitter import Fitter as MemeticFitter
from models.surface.nurbs_surface_rule import NURBSSurfaceRule
from tools.data_tool import load_ply_data as load_data


def build_config(refine_every: int, run_id: int) -> dict:
    return {
        'task_type': '3d',
        'run_id': run_id,
        'data_file': str(PROJECT_ROOT / 'datasets/synthetic/saddle_3k.ply'),
        'model': {
            'type': 'nurbs_surface',
            'num_ctrl_u': 16,
            'num_ctrl_v': 16,
            'degree_u': 3,
            'degree_v': 3,
            'sample_u': 100,
            'sample_v': 100,
            'weight_lb': 0.8,
            'weight_ub': 1.2,
        },
        'device': {'train_device': 'cuda:1', 'cuda_deterministic': True},
        'seeds': None,
        'estimator': {
            'type': 'mm',
            'data_resolution': 0.01,
            'model_resolution': 0.004,
            'regularization_factor': 0.5,
            'incremental_coverage': True,
            'outlier_distance_factor': 2.5,
            'outlier_penalty_factor': 0.0,
            'bbox_margin_factor': 1.0,
            'bbox_penalty_factor': 0.0,
            'overlap_penalty_factor': 0.0,
            'control_smoothness_penalty_factor': 0.0,
            'mm_bbox_penalty_factor': 0.0,
            'early_rejection': True,
            'use_faiss': True,
            'rule_class': NURBSSurfaceRule,
            'estimator_class': MeanMeasureEstimator,
            'estimator_instance': None,
            'load_data_fn': load_data,
            'data_file': str(PROJECT_ROOT / 'datasets/synthetic/saddle_3k.ply'),
        },
        'fitter': {
            'algo_name': 'memetic',
            'num_instances': 1,
            'max_episode': 30000,
            'gd_lr': 0.01,
            'mem_pop_size': 200,
            'mem_min_pop': 20,
            'mem_refine_every': refine_every,
            'mem_refine_steps': 50,
            'mem_refine_method': 'adam',
            'mem_refine_top_k': 1,
            'mem_adaptive_K': True,
            'mem_num_workers': 12,
            'num_envs': 8,
            'episodes_per_env': 50,
        },
        'record': {
            'root_dir': str(PROJECT_ROOT.parent / 'outputs/memetic_refiner_ablation/'),
            'pulse_size': 20,
            'verbose': True,
            'trim_final_mesh': True,
        },
    }


def run_one(name: str, refine_every: int, runs: int = 5):
    print(f"\n{'='*70}")
    print(f"  {name}: refine_every={refine_every}, {runs} runs")
    print(f"{'='*70}")

    scores = []
    for i in range(runs):
        cfg = build_config(refine_every, run_id=i + 1)
        cfg['record']['timestamp'] = current_timestamp()
        cfg['record']['root_dir'] = str(
            PROJECT_ROOT.parent / f'outputs/memetic_refiner_ablation/{name}/run_{i+1}/'
        )

        print(f"\n[{name}] Run {i+1}/{runs}...")
        fitter = MemeticFitter(cfg)
        fitter.fit()
        fitter.close()

        # Read best score from record (handle nested timestamp dirs)
        import json, os
        record_dir = cfg['record']['root_dir']
        score = float('nan')
        for dirpath, dirnames, filenames in os.walk(record_dir):
            if 'record.json' in filenames:
                rec_path = os.path.join(dirpath, 'record.json')
                with open(rec_path) as f:
                    s = json.load(f).get('best_score', 0)
                if s > 0:
                    score = max(score, s)
        scores.append(score)
        print(f"  -> Score: {score:.4f}")

    arr = np.array(scores)
    print(f"\n{name} summary ({runs} runs):")
    print(f"  Mean={arr.mean():.4f}  Std={arr.std():.4f}  Min={arr.min():.4f}  Max={arr.max():.4f}")
    return arr


def main():
    results = {}

    # NL-SHADE + Adam (refiner ON)
    results['NL-SHADE+Adam'] = run_one('nlshade_adam', refine_every=10, runs=5)

    # 纯 NL-SHADE (refiner OFF)
    results['NL-SHADE'] = run_one('nlshade_pure', refine_every=999, runs=5)

    print(f"\n{'='*70}")
    print("  对比结果")
    print(f"{'='*70}")
    for name, scores in results.items():
        print(f"  {name:<20s}: {scores.mean():.4f} ± {scores.std():.4f}  [{scores.min():.4f}, {scores.max():.4f}]")

    delta = results['NL-SHADE+Adam'].mean() - results['NL-SHADE'].mean()
    print(f"\n  Δ (Adam - Pure): {delta:+.4f}")
    if delta > 0:
        print("  → Adam refiner 有效")
    else:
        print("  → Adam refiner 无效或负面")


if __name__ == '__main__':
    main()
